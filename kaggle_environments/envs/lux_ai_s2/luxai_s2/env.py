import functools
import math
import traceback
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers

from luxai_s2.actions import (
    Action,
    DigAction,
    FactoryBuildAction,
    FactoryWaterAction,
    MoveAction,
    PickupAction,
    RechargeAction,
    SelfDestructAction,
    TransferAction,
    format_action_vec,
    format_factory_action,
    move_deltas,
    resource_to_name,
    validate_actions,
)
from luxai_s2.config import EnvConfig
from luxai_s2.factory import Factory
from luxai_s2.map.board import Board
from luxai_s2.map.position import Position
from luxai_s2.pyvisual.visualizer import Visualizer
from luxai_s2.spaces.act_space import (
    get_act_space,
    get_act_space_bid,
    get_act_space_init,
    get_act_space_placement,
)
from luxai_s2.spaces.obs_space import get_obs_space
from luxai_s2.state import (
    ObservationStateDict,
    State,
    StatsStateDict,
    create_empty_stats,
)
from luxai_s2.team import FactionTypes, Team
from luxai_s2.unit import Unit, UnitType
from luxai_s2.utils.utils import get_top_two_power_units, is_day

# some utility types
ActionsByType = Dict[str, List[Tuple[Unit, Action]]]


def env():
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = raw_env()
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class LuxAI_S2(ParallelEnv):
    metadata = {"render.modes": ["human", "html", "rgb_array"], "name": "luxai_s2_v0"}

    def __init__(self, collect_stats: bool = False, **kwargs):
        self.collect_stats = collect_stats  # note: added here instead of in configs since it would break existing bots
        default_config = EnvConfig(**kwargs)
        self.env_cfg = default_config
        self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.max_episode_length = self.env_cfg.max_episode_length

        self.state: State = State(
            seed_rng=None, seed=-1, env_cfg=self.env_cfg, env_steps=-1, board=None
        )

        self.seed_rng: np.random.RandomState = np.random.RandomState()

        self.py_visualizer: Visualizer = None

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return get_obs_space(
            config=self.env_cfg, agent_names=self.possible_agents, agent=agent
        )

    # @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        if self.env_cfg.BIDDING_SYSTEM:
            if self.env_steps == 0:
                # bid first, then place factories
                return get_act_space_bid(config=self.env_cfg, agent=agent)
            if self.env_steps <= self.state.board.factories_per_team + 1:
                return get_act_space_placement(config=self.env_cfg, agent=agent)
            return get_act_space(
                self.state.units, self.state.factories, config=self.env_cfg, agent=agent
            )
        else:
            if self.env_steps == 0:
                return get_act_space_init(config=self.env_cfg, agent=agent)
            return get_act_space(
                self.state.units, self.state.factories, config=self.env_cfg, agent=agent
            )

    def _init_render(self):
        if self.py_visualizer is None:
            self.py_visualizer = Visualizer(self.state)
            return True
        return False

    def render(self, mode="human", **kwargs):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """

        if mode == "human":
            if self._init_render():
                self.py_visualizer.init_window()

            self.py_visualizer.update_scene(self.state)
            self.py_visualizer.render()
        elif mode == "rgb_array":
            self._init_render()
            self.py_visualizer.update_scene(self.state)
            VIDEO_W = 400
            VIDEO_H = 400
            if "width" in kwargs:
                VIDEO_W = kwargs["width"]
            if "height" in kwargs:
                VIDEO_H = kwargs["height"]
            return self.py_visualizer._create_image_array(
                self.py_visualizer.surf, (VIDEO_W, VIDEO_H)
            )

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        try:
            import pygame

            pygame.display.quit()
            pygame.quit()
        except:
            print("No pygame installed, ignoring import")
        pass

    def get_state(self):
        return self.state

    def set_state(self, state: State):
        self.state = state
        self.env_steps = state.env_steps
        self.seed_rng = state.seed_rng
        self.seed = state.seed
        # TODO - throw warning if setting state from a different configuration than initialized with
        self.env_cfg = state.env_cfg
        self.max_episode_length = self.env_cfg.max_episode_length

    def reset(self, seed=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.

        Returns the observations for each agent
        """

        self.agents = self.possible_agents[:]
        self.env_steps = 0
        if seed is not None:
            self.seed_val = seed
            self.seed_rng = np.random.RandomState(seed=seed)
        else:
            self.seed_val = np.random.randint(0, 2**32 - 1, dtype=np.int64)
            self.seed_rng = np.random.RandomState(seed=self.seed_val)
        board = Board(seed=self.seed_rng.randint(0, 2**32 - 1, dtype=np.int64), env_cfg=self.env_cfg)
        self.state: State = State(
            seed_rng=self.seed_rng,
            seed=self.seed_val,
            env_cfg=self.state.env_cfg,
            env_steps=0,
            board=board,
        )
        self.max_episode_length = self.env_cfg.max_episode_length
        for agent in self.possible_agents:
            self.state.units[agent] = OrderedDict()
            self.state.factories[agent] = OrderedDict()
            if self.collect_stats:
                self.state.stats[agent] = create_empty_stats()
        obs = self.state.get_obs()
        observations = {agent: obs for agent in self.agents}
        return observations
    
    def log_error(self, *m):
        if self.env_cfg.verbose > 0:
            print(f"{self.state.real_env_steps}: {' '.join(m)}")

    def log_warning(self, *m):
        if self.env_cfg.verbose > 1:
            print(f"{self.state.real_env_steps}: {' '.join(m)}")

    def log_info(self, *m):
        if self.env_cfg.verbose > 2:
            print(f"{self.state.real_env_steps}: {' '.join(m)}")

    def _handle_bid(self, actions):
        failed_agents = {agent: False for agent in self.agents}
        highest_bid = -1
        highest_bid_agent = None
        for k, a in actions.items():
            if a is None:
                failed_agents[k] = True
                continue
            if k not in self.agents:
                raise ValueError(f"Invalid player {k}")
            try:
                if "faction" in a and "bid" in a:
                    if a["faction"] not in [e.name for e in FactionTypes]:
                        self.log_error(
                            f"{k} initialized with invalid faction name {a['faction']}"
                        )
                        failed_agents[k] = True
                        continue
                    self.state.teams[k] = Team(
                        team_id=self.agent_name_mapping[k],
                        agent=k,
                        faction=FactionTypes[a["faction"]],
                    )
                    self.state.teams[
                        k
                    ].init_water = self.env_cfg.INIT_WATER_METAL_PER_FACTORY * (
                        self.state.board.factories_per_team
                    )
                    self.state.teams[
                        k
                    ].init_metal = self.env_cfg.INIT_WATER_METAL_PER_FACTORY * (
                        self.state.board.factories_per_team
                    )
                    self.state.teams[
                        k
                    ].factories_to_place = self.state.board.factories_per_team
                    # verify bid is valid
                    valid_action = True
                    bid = math.floor(abs(a["bid"]))
                    self.state.teams[k].bid = a["bid"]
                    if bid > self.state.teams[k].init_water:
                        valid_action = False
                    if not valid_action:
                        failed_agents[k] = True
                        continue
                    if bid > highest_bid:
                        highest_bid = bid
                        highest_bid_agent = k
                    elif bid == highest_bid:
                        # if bids are the same, player 0 defaults to the winner and pays the bid.
                        highest_bid_agent = "player_0"
                else:
                    # team k loses
                    failed_agents[k] = True
            except Exception as e:
                print(traceback.format_exc())
                failed_agents[k] = True
        for agent in self.agents:
            if failed_agents[agent]:
                return failed_agents
        if highest_bid_agent is None:
            # no valid bids made, all agents failed.
            pass
        else:
            lowest_bid_agent = "player_1"
            if highest_bid_agent == "player_1":
                lowest_bid_agent = "player_0"
            self.state.teams[highest_bid_agent].init_water -= highest_bid
            self.state.teams[highest_bid_agent].init_metal -= highest_bid

            # highest bid agent either won because of a tie, which then player_0 default wins, or because they had a stronger bid
            # who ever wins will always get their choice
            if self.state.teams[highest_bid_agent].bid < 0:
                self.state.teams[highest_bid_agent].place_first = False
                self.state.teams[lowest_bid_agent].place_first = True
            else:
                self.state.teams[highest_bid_agent].place_first = True
                self.state.teams[lowest_bid_agent].place_first = False
        return failed_agents

    def _handle_factory_placement_step(self, actions):
        # factory placement rounds, which are sequential

        player_to_place_factory: str
        if self.state.teams["player_0"].place_first:
            if self.state.env_steps % 2 == 1:
                player_to_place_factory = "player_0"
            else:
                player_to_place_factory = "player_1"
        else:
            if self.state.env_steps % 2 == 1:
                player_to_place_factory = "player_1"
            else:
                player_to_place_factory = "player_0"

        failed_agents = {agent: False for agent in self.agents}
        for k, a in actions.items():
            if a is None:
                failed_agents[k] = True
                continue
            if k not in self.agents:
                raise ValueError(f"Invalid player {k}")
            try:
                if "spawn" in a and "metal" in a and "water" in a:
                    if k != player_to_place_factory:
                        self.log_warning(
                            f"{k} tried to perform an action in the early phase when it is not its turn right now."
                        )
                        continue
                    if self.state.teams[k].factories_to_place <= 0:
                        self.log_warning(
                            f"{k} cannot place additional factories. Cancelled placement of factory"
                        )
                        continue
                    if a["water"] < 0 or a["metal"] < 0:
                        self.log_warning(
                            f"{k} tried to place negative water/metal in factory. Cancelled placement of factory"
                        )
                        continue
                    if a["water"] > self.state.teams[k].init_water:
                        a["water"] = self.state.teams[k].init_water
                        self.log_warning(
                            f" Warning - {k} does not have enough water. Using {a['water']}"
                        )
                    if a["metal"] > self.state.teams[k].init_metal:
                        a["metal"] = self.state.teams[k].init_metal
                        self.log_warning(
                            f" Warning - {k} does not have enough metal. Using {a['metal']}"
                        )
                    factory = self.add_factory(self.state.teams[k], a["spawn"])
                    if factory is None:
                        continue
                    a["water"] = math.floor(a["water"])
                    a["metal"] = math.floor(a["metal"])
                    factory.cargo.water = a["water"]
                    factory.cargo.metal = a["metal"]
                    factory.power = self.env_cfg.INIT_POWER_PER_FACTORY
                    self.state.teams[k].factories_to_place -= 1
                    self.state.teams[k].init_metal -= a["metal"]
                    self.state.teams[k].init_water -= a["water"]
                else:
                    # pass, turn is skipped.
                    pass
            except Exception as e:
                print(traceback.format_exc())
                failed_agents[k] = True
        return failed_agents

    def _handle_nobidding_early_game(self, actions):
        failed_agents = {agent: False for agent in self.agents}
        for k, a in actions.items():
            if a is None:
                failed_agents[k] = True
                continue
            if k not in self.agents:
                raise ValueError(f"Invalid player {k}")
            if "spawns" in a and "faction" in a:
                self.state.teams[k] = Team(
                    team_id=self.agent_name_mapping[k],
                    agent=k,
                    faction=FactionTypes[a["faction"]],
                )
                if len(a["spawns"]) > self.state.board.factories_per_team:
                    self.log_warning(
                        f"{k} tried to spawn more factories than allocated in board.factories_per_team. Spawning only the first {self.state.board.factories_per_team} locations"
                    )
                for spawn_loc in a["spawns"][: self.state.board.factories_per_team]:
                    self.add_factory(self.state.teams[k], spawn_loc)
            else:
                # team k loses
                failed_agents[k] = True

    def _step_early_game(self, actions):
        # handle initialization
        failed_agents = {agent: False for agent in self.agents}
        if self.env_cfg.BIDDING_SYSTEM:
            if self.env_steps == 0:
                failed_agents = self._handle_bid(actions)
            else:
                failed_agents = self._handle_factory_placement_step(actions)
        else:
            self._handle_nobidding_early_game(actions)
        return failed_agents

    def _handle_transfer_actions(self, actions_by_type: ActionsByType):
        # It is important to first sub resource from all units, and then add
        # resource to targets. Only When splitted into two loops, the transfer
        # action is irrelevant to unit id.

        # sub from unit cargo
        amount_list = []
        for unit, transfer_action in actions_by_type["transfer"]:
            transfer_action: TransferAction
            transfer_amount = unit.sub_resource(
                transfer_action.resource, transfer_action.transfer_amount
            )
            amount_list.append(transfer_amount)

        # add to target cargo
        for (unit, transfer_action), transfer_amount in zip(
            actions_by_type["transfer"], amount_list
        ):
            transfer_action: TransferAction
            transfer_pos: Position = (
                unit.pos + move_deltas[transfer_action.transfer_dir]
            )
            units_there = self.state.board.get_units_at(transfer_pos)
            # if there is a factory, we prefer transferring to that entity
            factory_id = f"factory_{self.state.board.factory_occupancy_map[transfer_pos.x, transfer_pos.y]}"
            if factory_id in self.state.factories[unit.team.agent]:
                factory = self.state.factories[unit.team.agent][factory_id]
                actually_transferred = factory.add_resource(
                    transfer_action.resource, transfer_amount
                )
                if self.collect_stats:
                    self.state.stats[unit.team.agent]["transfer"][
                        resource_to_name[transfer_action.resource]
                    ] += actually_transferred
            elif units_there is not None:
                assert len(units_there) == 1, "Fatal error here, this is a bug"
                target_unit = units_there[0]
                # add resources to target. This will waste (transfer_amount - actually_transferred) resources
                actually_transferred = target_unit.add_resource(
                    transfer_action.resource, transfer_amount
                )
                if self.collect_stats:
                    self.state.stats[unit.team.agent]["transfer"][
                        resource_to_name[transfer_action.resource]
                    ] += actually_transferred
            unit.repeat_action(transfer_action)

    def _handle_pickup_actions(self, actions_by_type: ActionsByType):
        for unit, pickup_action in actions_by_type["pickup"]:
            pickup_action: PickupAction
            factory = self.state.board.get_factory_at(self.state, unit.pos)
            pickup_amount = factory.sub_resource(
                pickup_action.resource, pickup_action.pickup_amount
            )
            # may waste resources if tried to pickup more than one can hold.
            actually_pickedup = unit.add_resource(pickup_action.resource, pickup_amount)
            unit.repeat_action(pickup_action)
            if self.collect_stats:
                self.state.stats[unit.team.agent]["pickup"][
                    resource_to_name[pickup_action.resource]
                ] += actually_pickedup

    def _handle_dig_actions(self, actions_by_type: ActionsByType):
        for unit, dig_action in actions_by_type["dig"]:
            dig_action: DigAction
            if self.state.board.rubble[unit.pos.x, unit.pos.y] > 0:
                if self.collect_stats:
                    rubble_before = self.state.board.rubble[unit.pos.x, unit.pos.y]
                self.state.board.rubble[unit.pos.x, unit.pos.y] = max(
                    self.state.board.rubble[unit.pos.x, unit.pos.y]
                    - unit.unit_cfg.DIG_RUBBLE_REMOVED,
                    0,
                )
                if self.collect_stats:
                    self.state.stats[unit.team.agent]["destroyed"]["rubble"][
                        unit.unit_type.name
                    ] -= (
                        self.state.board.rubble[unit.pos.x, unit.pos.y] - rubble_before
                    )
            elif self.state.board.lichen[unit.pos.x, unit.pos.y] > 0:
                if self.collect_stats:
                    lichen_before = self.state.board.lichen[unit.pos.x, unit.pos.y]
                lichen_left = max(
                    self.state.board.lichen[unit.pos.x, unit.pos.y]
                    - unit.unit_cfg.DIG_LICHEN_REMOVED,
                    0,
                )
                self.state.board.lichen[unit.pos.x, unit.pos.y] = lichen_left
                if lichen_left == 0:  # dug out the last lichen
                    self.state.board.rubble[
                        unit.pos.x, unit.pos.y
                    ] = self.state.env_cfg.ROBOTS[unit.unit_type.name].DIG_RESOURCE_GAIN
                if self.collect_stats:
                    self.state.stats[unit.team.agent]["destroyed"]["lichen"][
                        unit.unit_type.name
                    ] -= (
                        self.state.board.lichen[unit.pos.x, unit.pos.y] - lichen_before
                    )
            elif self.state.board.ice[unit.pos.x, unit.pos.y] > 0:
                gained = unit.add_resource(0, unit.unit_cfg.DIG_RESOURCE_GAIN)
                if self.collect_stats:
                    self.state.stats[unit.team.agent]["generation"]["ice"][
                        unit.unit_type.name
                    ] += gained
            elif self.state.board.ore[unit.pos.x, unit.pos.y] > 0:
                gained = unit.add_resource(1, unit.unit_cfg.DIG_RESOURCE_GAIN)
                if self.collect_stats:
                    self.state.stats[unit.team.agent]["generation"]["ore"][
                        unit.unit_type.name
                    ] += gained
            unit.power -= self.state.env_cfg.ROBOTS[unit.unit_type.name].DIG_COST
            unit.repeat_action(dig_action)

    def _handle_self_destruct_actions(self, actions_by_type: ActionsByType):
        for unit, self_destruct_action in actions_by_type["self_destruct"]:
            unit: Unit
            self_destruct_action: SelfDestructAction
            pos_hash = self.state.board.pos_hash(unit.pos)
            del self.state.board.units_map[pos_hash]
            self.destroy_unit(unit)
            if self.collect_stats:
                self.state.stats[unit.team.agent]["destroyed"][unit.unit_type.name] += 1

    def _handle_factory_build_actions(self, actions_by_type: ActionsByType):
        for factory, factory_build_action in actions_by_type["factory_build"]:
            factory: Factory
            factory_build_action: FactoryBuildAction
            team = self.state.teams[factory.team.agent]

            # note that this happens after unit resource pickup. Thus we have to reverify these actions
            # if self.state.env_cfg.validate_action_space:

            if factory_build_action.unit_type == UnitType.HEAVY:
                if factory.cargo.metal < self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                    self.log_warning(
                        f"{factory} doesn't have enough metal to build a heavy despite having enough metal at the start of the turn. This is likely because a unit picked up some of the metal."
                    )
                    continue
                if factory.power < factory_build_action.power_cost:
                    self.log_warning(
                        f"{factory} doesn't have enough power to build a heavy despite having enough power at the start of the turn. This is likely because a unit picked up some of the power."
                    )
                    continue
                spent_metal = factory.sub_resource(
                    3, self.env_cfg.ROBOTS["HEAVY"].METAL_COST
                )
                spent_power = factory.sub_resource(4, factory_build_action.power_cost)
            else:
                if factory.cargo.metal < self.env_cfg.ROBOTS["LIGHT"].METAL_COST:
                    self.log_warning(
                        f"{factory} doesn't have enough metal to build a light despite having enough metal at the start of the turn. This is likely because a unit picked up some of the metal."
                    )
                    continue
                if factory.power < factory_build_action.power_cost:
                    self.log_warning(
                        f"{factory} doesn't have enough power to build a light despite having enough power at the start of the turn. This is likely because a unit picked up some of the power."
                    )
                    continue
                spent_metal = factory.sub_resource(
                    3, self.env_cfg.ROBOTS["LIGHT"].METAL_COST
                )
                spent_power = factory.sub_resource(4, factory_build_action.power_cost)

            self.add_unit(
                team=team,
                unit_type=factory_build_action.unit_type,
                pos=factory.pos.pos,
            )
            if self.collect_stats:
                self.state.stats[factory.team.agent]["generation"]["built"][
                    factory_build_action.unit_type.name
                ] += 1
                self.state.stats[factory.team.agent]["consumption"][
                    "metal"
                ] += spent_metal
                self.state.stats[factory.team.agent]["consumption"]["power"][
                    "FACTORY"
                ] += spent_power

    def _handle_movement_actions(self, actions_by_type: ActionsByType):
        new_units_map: Dict[str, List[Unit]] = defaultdict(list)
        heavy_entered_pos: Dict[str, List[Unit]] = defaultdict(list)
        light_entered_pos: Dict[str, List[Unit]] = defaultdict(list)

        for unit, move_action in actions_by_type["move"]:
            move_action: MoveAction
            # skip move center
            if move_action.move_dir != 0:
                old_pos_hash = self.state.board.pos_hash(unit.pos)
                target_pos = (
                    unit.pos + move_action.dist * move_deltas[move_action.move_dir]
                )
                power_required = move_action.power_cost
                unit.pos = target_pos
                new_pos_hash = self.state.board.pos_hash(unit.pos)

                # remove unit from map temporarily
                if len(self.state.board.units_map[old_pos_hash]) == 1:
                    del self.state.board.units_map[old_pos_hash]
                else:
                    self.state.board.units_map[old_pos_hash].remove(unit)

                new_units_map[new_pos_hash].append(unit)
                unit.power -= power_required

                if unit.unit_type == UnitType.HEAVY:
                    heavy_entered_pos[new_pos_hash].append(unit)
                else:
                    light_entered_pos[new_pos_hash].append(unit)

            unit.repeat_action(move_action)

        for pos_hash, units in self.state.board.units_map.items():
            # add in all the stationary units
            new_units_map[pos_hash] += units

        all_destroyed_units: Set[Unit] = set()
        new_units_map_after_collision: Dict[str, List[Unit]] = defaultdict(list)
        for pos_hash, units in new_units_map.items():
            destroyed_units: Set[Unit] = set()
            if len(units) <= 1:
                new_units_map_after_collision[pos_hash] += units
                continue
            if len(heavy_entered_pos[pos_hash]) > 1:
                # all units collide, find the top 2 units by power
                (most_power_unit, next_most_power_unit) = get_top_two_power_units(units, UnitType.HEAVY)
                if most_power_unit.power == next_most_power_unit.power:
                    # tie, all units break
                    for u in units:
                        destroyed_units.add(u)
                    self.log_info(
                        f"{len(destroyed_units)} Units: ({', '.join([u.unit_id for u in destroyed_units])}) collided at {pos_hash}"
                    )
                else:
                    most_power_unit_power_loss = math.ceil(
                        next_most_power_unit.power * self.env_cfg.POWER_LOSS_FACTOR
                    )
                    most_power_unit.power -= most_power_unit_power_loss
                    surviving_unit = most_power_unit
                    for u in units:
                        if u.unit_id != surviving_unit.unit_id:
                            destroyed_units.add(u)
                    self.log_info(
                        f"{len(destroyed_units)} Units: ({', '.join([u.unit_id for u in destroyed_units])}) collided at {pos_hash} with {surviving_unit} surviving with {surviving_unit.power} power"
                    )
                    new_units_map_after_collision[pos_hash].append(surviving_unit)
                all_destroyed_units.update(destroyed_units)
            elif len(heavy_entered_pos[pos_hash]) > 0:
                # all other units collide and break
                surviving_unit = heavy_entered_pos[pos_hash][0]
                for u in units:
                    if u.unit_id != surviving_unit.unit_id:
                        destroyed_units.add(u)
                self.log_info(
                    f"{len(destroyed_units)} Units: ({', '.join([u.unit_id for u in destroyed_units])}) collided at {pos_hash} with {surviving_unit} surviving with {surviving_unit.power} power"
                )
                new_units_map_after_collision[pos_hash].append(surviving_unit)
                all_destroyed_units.update(destroyed_units)
            else:
                # check for stationary heavy unit there
                surviving_unit = None
                heavy_stationary_unit = None
                for u in units:
                    if u.unit_type == UnitType.HEAVY:
                        if heavy_stationary_unit is not None:
                            heavy_stationary_unit = None
                            # we found >= 2 heavies stationary in a tile where no heavies are entering.
                            # should only happen when spawning units
                            self.log_info(
                                f"At {pos_hash}, >= 2 heavies crashed as they were all stationary"
                            )
                            break
                        heavy_stationary_unit = u

                if heavy_stationary_unit is not None:
                    surviving_unit = heavy_stationary_unit
                else:
                    if len(light_entered_pos[pos_hash]) > 1:
                        # all units collide, get top 2 units by power
                        (
                            most_power_unit,
                            next_most_power_unit,
                        ) = get_top_two_power_units(units, UnitType.LIGHT)
                        if most_power_unit.power == next_most_power_unit.power:
                            # tie, all units break
                            for u in units:
                                destroyed_units.add(u)
                        else:
                            most_power_unit_power_loss = math.ceil(
                                next_most_power_unit.power
                                * self.env_cfg.POWER_LOSS_FACTOR
                            )
                            most_power_unit.power -= most_power_unit_power_loss
                            surviving_unit = most_power_unit
                    elif len(light_entered_pos[pos_hash]) > 0:
                        # light crashes into stationary light unit
                        surviving_unit = light_entered_pos[pos_hash][0]
                if surviving_unit is None:
                    for u in units:
                        destroyed_units.add(u)
                    self.log_info(
                        f"{len(destroyed_units)} Units: ({', '.join([u.unit_id for u in destroyed_units])}) collided at {pos_hash}"
                    )
                    all_destroyed_units.update(destroyed_units)
                else:
                    for u in units:
                        if u.unit_id != surviving_unit.unit_id:
                            destroyed_units.add(u)
                    self.log_info(
                        f"{len(destroyed_units)} Units: ({', '.join([u.unit_id for u in destroyed_units])}) collided at {pos_hash} with {surviving_unit} surviving with {surviving_unit.power} power"
                    )
                    new_units_map_after_collision[pos_hash].append(surviving_unit)
                    all_destroyed_units.update(destroyed_units)
        self.state.board.units_map = new_units_map_after_collision

        for u in all_destroyed_units:
            self.destroy_unit(u)
            if self.collect_stats:
                self.state.stats[u.team.agent]["destroyed"][u.unit_type.name] += 1

    def _handle_recharge_actions(self, actions_by_type: ActionsByType):
        # for recharging actions, check if unit has enough power. If not, add action back to the queue
        for unit, recharge_action in actions_by_type["recharge"]:
            recharge_action: RechargeAction
            if unit.power < recharge_action.power:
                pass
            else:
                # if unit got enough power, handle the action and consider it for repeating
                unit.repeat_action(recharge_action)

    def _handle_factory_water_actions(self, actions_by_type: ActionsByType):
        for factory, factory_water_action in actions_by_type["factory_water"]:
            factory_water_action: FactoryWaterAction
            water_cost = factory.water_cost(self.env_cfg)
            if water_cost > factory.cargo.water:
                self.log_warning(
                    f"{factory} has insufficient water to grow lichen, factory has {factory.cargo.water}, but requires {water_cost} to water lichen. This cost may have changed a little during this turn due to rubble changes and new tiles being grown on"
                )
                continue
            factory.cargo.water -= water_cost
            indexable_positions = (
                [v[0] for v in factory.grow_lichen_positions],
                [v[1] for v in factory.grow_lichen_positions],
            )
            self.state.board.lichen[indexable_positions] += 2
            self.state.board.lichen_strains[indexable_positions] = factory.num_id
            if self.collect_stats:
                self.state.stats[factory.team.agent]["consumption"][
                    "water"
                ] += water_cost

    def step(
        self, actions
    ) -> Tuple[
        Dict[str, ObservationStateDict],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            raise ValueError("No actions given")
            self.agents = []
            return {}, {}, {}, {}
        
        failed_agents = {agent: False for agent in self.agents}
        # Turn 1 logic, handle
        early_game = False
        if self.env_steps == 0:
            early_game = True
        if self.env_cfg.BIDDING_SYSTEM and self.state.real_env_steps < 0:
            early_game = True
        if early_game:
            failed_agents = self._step_early_game(actions)
        else:
            # 1. Check for malformed actions

            if self.env_cfg.validate_action_space:
                # This part is not absolutely necessary if you know for sure your actions are well formatted
                for agent, unit_actions in actions.items():
                    valid_acts, err_reason = self.action_space(agent).contains(
                        unit_actions
                    )
                    if not valid_acts:
                        failed_agents[agent] = True
                        self.log_error(
                            f"{self.state.teams[agent]} Inappropriate action given. {err_reason}"
                        )

            # we should except that actions is always of type dict, if not then erroring here is fine
            for agent, unit_actions in actions.items():
                try:
                    for unit_id, action in unit_actions.items():
                        if "factory" in unit_id:
                            self.state.factories[agent][unit_id].action_queue.append(
                                format_factory_action(action)
                            )
                        elif "unit" in unit_id:
                            unit = self.state.units[agent][unit_id]
                            # if unit does not have more than ACTION_QUEUE_POWER_COST power, we skip updating the action queue and print warning
                            update_power_req = self.state.env_cfg.ROBOTS[
                                unit.unit_type.name
                            ].ACTION_QUEUE_POWER_COST
                            if self.collect_stats:
                                self.state.stats[agent][
                                    "action_queue_updates_total"
                                ] += 1
                            if unit.power < update_power_req:
                                self.log_info(
                                    f"{agent} Tried to update action queue for {unit} requiring {update_power_req} power but only had {unit.power} power"
                                )
                                continue
                            formatted_actions = []
                            if type(action) == list or (
                                type(action) == np.ndarray and (len(action.shape) == 2 or len(action) == 0)
                            ):
                                trunked_actions = action[
                                    : self.env_cfg.UNIT_ACTION_QUEUE_SIZE
                                ]
                                formatted_actions = [
                                    format_action_vec(a) for a in trunked_actions
                                ]
                            else:
                                self.log_error(
                                    f"{agent} Tried to update action queue for {unit} but did not provide an action queue, provided {action}"
                                )
                                failed_agents[agent] = True
                                continue
                            unit.power -= update_power_req
                            if self.collect_stats:
                                self.state.stats[agent][
                                    "action_queue_updates_success"
                                ] += 1
                            self.state.units[agent][
                                unit_id
                            ].action_queue = formatted_actions
                except Exception as e:
                    # catch errors when trying to format unit or factory actions
                    print(traceback.format_exc())
                    failed_agents[agent] = True

            # 2. store actions by type
            actions_by_type: ActionsByType = defaultdict(list)
            for agent in self.agents:
                if failed_agents[agent]:  # skip failed agents
                    continue
                for unit in self.state.units[agent].values():
                    unit_a: Action = unit.next_action()
                    if unit_a is None:
                        continue
                    actions_by_type[unit_a.act_type].append((unit, unit_a))
                for factory in self.state.factories[agent].values():
                    if len(factory.action_queue) > 0:
                        unit_a: Action = factory.action_queue.pop(0)
                        actions_by_type[unit_a.act_type].append((factory, unit_a))

            # 3. validate all actions against current state, throw away impossible actions
            actions_by_type = validate_actions(
                self.env_cfg, self.state, actions_by_type, verbose=self.env_cfg.verbose
            )

            if self.collect_stats:
                lichen_before = self.state.board.lichen.copy()
                lichen_strains_before = self.state.board.lichen_strains.copy()

            self._handle_dig_actions(actions_by_type)
            self._handle_self_destruct_actions(actions_by_type)
            self._handle_factory_build_actions(actions_by_type)
            self._handle_movement_actions(actions_by_type)
            self._handle_recharge_actions(actions_by_type)

            for agent in self.agents:
                for factory in self.state.factories[agent].values():
                    # update information for lichen growing and cache it
                    factory.cache_water_info(self.state.board, self.env_cfg)

            self._handle_factory_water_actions(actions_by_type)
            self._handle_transfer_actions(actions_by_type)
            self._handle_pickup_actions(actions_by_type)

            # Update lichen
            self.state.board.lichen -= 1
            self.state.board.lichen = self.state.board.lichen.clip(
                0, self.env_cfg.MAX_LICHEN_PER_TILE
            )
            self.state.board.lichen_strains[self.state.board.lichen == 0] = -1
            if self.collect_stats:
                lichen_change = self.state.board.lichen - lichen_before
                for agent in self.agents:
                    for strain in self.state.teams[agent].factory_strains:
                        start_of_step_lichen_tiles: np.ndarray = lichen_change[
                            lichen_strains_before == strain
                        ]
                        lichen_lost = start_of_step_lichen_tiles[
                            start_of_step_lichen_tiles < 0
                        ].sum()
                        lichen_gained = start_of_step_lichen_tiles.sum() - lichen_lost
                        self.state.stats[agent]["generation"]["lichen"] += lichen_gained

            # resources refining
            for agent in self.agents:
                factories_to_destroy: Set[Factory] = set()
                for factory in self.state.factories[agent].values():
                    if self.collect_stats:
                        water_before = factory.cargo.water
                        metal_before = factory.cargo.metal
                    factory.refine_step(self.env_cfg)
                    if self.collect_stats:
                        self.state.stats[agent]["generation"]["metal"] += (
                            factory.cargo.metal - metal_before
                        )
                        self.state.stats[agent]["generation"]["water"] += (
                            factory.cargo.water - water_before
                        )
                    factory.cargo.water -= self.env_cfg.FACTORY_WATER_CONSUMPTION
                    if factory.cargo.water < 0:
                        factories_to_destroy.add(factory)
                for factory in factories_to_destroy:
                    # destroy factories that ran out of water
                    self.destroy_factory(factory)
                    if self.collect_stats:
                        self.state.stats[factory.team.agent]["destroyed"][
                            "FACTORY"
                        ] += 1
            # power gain
            if is_day(self.env_cfg, self.state.real_env_steps):
                for agent in self.agents:
                    for u in self.state.units[agent].values():
                        if self.collect_stats:
                            power_before = u.power
                        u.power = u.power + self.env_cfg.ROBOTS[u.unit_type.name].CHARGE
                        u.power = min(u.power, u.unit_cfg.BATTERY_CAPACITY)
                        if self.collect_stats:
                            self.state.stats[agent]["generation"]["power"][
                                u.unit_type.name
                            ] += (u.power - power_before)
            for agent in self.agents:
                for f in self.state.factories[agent].values():
                    if self.collect_stats:
                        power_before = f.power
                    # natural nuclear energy generation
                    f.power = f.power + self.env_cfg.FACTORY_CHARGE
                    # lichen/plant power
                    f.power = (
                        f.power
                        + len(f.connected_lichen_positions)
                        * self.env_cfg.POWER_PER_CONNECTED_LICHEN_TILE
                    )
                    if self.collect_stats:
                        self.state.stats[agent]["generation"]["power"]["FACTORY"] += (
                            f.power - power_before
                        )

        # always set rubble under factories to 0.
        self.state.board.rubble[self.state.board.factory_occupancy_map != -1] = 0

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {}
        for agent in self.agents:
            if agent in self.state.teams:
                strain_ids = self.state.teams[agent].factory_strains
                factories_left = len(self.state.factories[agent])
                if factories_left == 0 and self.state.real_env_steps >= 0:
                    failed_agents[agent] = True
                    self.log_warning(f"{agent} lost all factories")
                if failed_agents[agent]:
                    rewards[agent] = -1000
                else:
                    agent_lichen_mask = np.isin(
                        self.state.board.lichen_strains, strain_ids
                    )
                    rewards[agent] = self.state.board.lichen[agent_lichen_mask].sum()
            else:
                # if this was not initialize then agent failed in step 0
                failed_agents[agent] = True
                rewards[agent] = -1000

        self.env_steps += 1
        self.state.env_steps += 1
        env_done = self.state.real_env_steps >= self.state.env_cfg.max_episode_length
        env_done = (
            env_done or failed_agents["player_0"] or failed_agents["player_1"]
        )  # env is done if any agent fails.
        dones = {agent: env_done or failed_agents[agent] for agent in self.agents}

        # generate observations
        obs = self.state.get_obs()
        observations = {}
        for k in self.agents:
            observations[k] = obs

        # log stats and other things
        infos = {agent: {} for agent in self.agents}

        if env_done:
            self.agents = []

        return observations, rewards, dones, infos

    ### Game Logic ###
    def add_unit(self, team: Team, unit_type, pos: np.ndarray):
        unit = Unit(
            team=team,
            unit_type=unit_type,
            unit_id=f"unit_{self.state.global_id}",
            env_cfg=self.env_cfg,
        )
        unit.pos.pos = pos.copy()
        self.state.global_id += 1
        self.state.units[team.agent][unit.unit_id] = unit
        self.state.board.units_map[self.state.board.pos_hash(unit.pos)].append(unit)
        return unit

    def add_factory(self, team: Team, pos: np.ndarray):
        factory = Factory(
            team=team,
            unit_id=f"factory_{self.state.global_id}",
            num_id=self.state.global_id,
        )
        factory.pos.pos = np.array([pos[0], pos[1]]).astype(int)
        factory.cargo.water = self.env_cfg.INIT_WATER_METAL_PER_FACTORY
        factory.cargo.metal = self.env_cfg.INIT_WATER_METAL_PER_FACTORY
        factory.power = self.env_cfg.INIT_POWER_PER_FACTORY
        if self.state.board.valid_spawns_mask[pos[0], pos[1]] == 0:
            # Check if any tiles under the factory are invalid spawn tile.
            # TODO - min distance between Factories? stone: I think it's a bad strategy to try and enclose a opponent factory anyway,
            # wastes a few factories and hard to maintain
            self.log_warning(
                f"{team.agent} cannot place factory at {pos[0]}, {pos[1]} as it overlaps an existing factory or is on top of a resource"
            )
            return None
        if self.state.board.factory_occupancy_map[factory.pos_slice].max() >= 0:
            self.log_warning(
                f"{team.agent} cannot overlap factory placement. Existing factory at {factory.pos} already."
            )
            return None

        self.state.teams[team.agent].factory_strains += [factory.num_id]

        self.state.factories[team.agent][factory.unit_id] = factory
        self.state.board.factory_map[self.state.board.pos_hash(factory.pos)] = factory
        self.state.board.factory_occupancy_map[factory.pos_slice] = factory.num_id
        invalid_spawn_indices = factory.min_dist_slice
        # TODO: perf - this can be faster
        # self.state.board.valid_spawns_mask[factory.pos_slice] = False
        for x, y in invalid_spawn_indices:
            if (
                x < 0
                or y < 0
                or x >= self.state.board.rubble.shape[0]
                or y >= self.state.board.rubble.shape[1]
            ):
                continue
            self.state.board.valid_spawns_mask[x, y] = False
        self.state.board.rubble[factory.pos_slice] = 0
        self.state.board.ice[factory.pos_slice] = 0
        self.state.board.ore[factory.pos_slice] = 0

        self.state.global_id += 1
        return factory

    def destroy_unit(self, unit: Unit):
        """
        # NOTE this doesn't remove unit reference from board map
        """
        self.state.board.rubble[unit.pos.x, unit.pos.y] = min(
            self.state.board.rubble[unit.pos.x, unit.pos.y]
            + unit.unit_cfg.RUBBLE_AFTER_DESTRUCTION,
            self.env_cfg.MAX_RUBBLE,
        )
        self.state.board.lichen[unit.pos.x, unit.pos.y] = 0
        self.state.board.lichen_strains[unit.pos.x, unit.pos.y] = -1
        del self.state.units[unit.team.agent][unit.unit_id]

    def destroy_factory(self, factory: Factory):
        # spray rubble on every factory tile
        self.state.board.rubble[
            factory.pos_slice
        ] += self.env_cfg.FACTORY_RUBBLE_AFTER_DESTRUCTION
        self.state.board.rubble[factory.pos_slice] = self.state.board.rubble[
            factory.pos_slice
        ].clip(0, self.env_cfg.MAX_RUBBLE)
        self.state.board.factory_occupancy_map[factory.pos_slice] = -1
        del self.state.factories[factory.team.agent][factory.unit_id]
        del self.state.board.factory_map[self.state.board.pos_hash(factory.pos)]


def raw_env() -> LuxAI_S2:
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = LuxAI_S2()
    # env = parallel_to_aec(env)
    return env


import gym

gym.register(
    id="LuxAI_S2-v0",
    entry_point="luxai_s2.env:LuxAI_S2"
)
