from collections import OrderedDict, defaultdict
import functools
import math
from typing import Dict, List, Set, Tuple, Union

import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from luxai2022.map import weather
from luxai2022.actions import (
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
    validate_actions,
    move_deltas,
)

from luxai2022.config import EnvConfig
from luxai2022.factory import Factory
from luxai2022.map.board import Board
from luxai2022.map.position import Position
from luxai2022.pyvisual.visualizer import Visualizer
from luxai2022.spaces.act_space import get_act_space, get_act_space_bid, get_act_space_init, get_act_space_placement
from luxai2022.spaces.obs_space import get_obs_space
from luxai2022.state import State
from luxai2022.team import FactionTypes, Team
from luxai2022.unit import Unit, UnitType
from luxai2022.utils.utils import is_day

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


class LuxAI2022(ParallelEnv):
    metadata = {"render.modes": ["human", "html", "rgb_array"], "name": "luxai2022_v0"}

    def __init__(self, **kwargs):
        default_config = EnvConfig(**kwargs)
        self.env_cfg = default_config
        self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.max_episode_length = self.env_cfg.max_episode_length

        self.state: State = State(seed_rng=None, seed=-1, env_cfg=self.env_cfg, env_steps=-1, board=None, weather_schedule=None)

        self.seed_rng: np.random.RandomState = None

        self.py_visualizer: Visualizer = None

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return get_obs_space(config=self.env_cfg, agent_names=self.possible_agents, agent=agent)

    # @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        if self.env_cfg.BIDDING_SYSTEM:
            if self.env_steps == 0:
                # bid first, then place factories
                return get_act_space_bid(config=self.env_cfg, agent=agent)
            if self.env_steps <= self.state.board.factories_per_team + 1:
                return get_act_space_placement(config=self.env_cfg, agent=agent)
            return get_act_space(self.state.units, self.state.factories, config=self.env_cfg, agent=agent)
        else:
            if self.env_steps == 0:
                return get_act_space_init(config=self.env_cfg, agent=agent)
            return get_act_space(self.state.units, self.state.factories, config=self.env_cfg, agent=agent)

    def _init_render(self):
        if self.py_visualizer is None:
            self.py_visualizer = Visualizer(self.state)
            return True
        return False
    def render(self, mode="human"):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        # if len(self.agents) == 2:
        #     string = ("Current state: Agent1: {} , Agent2: {}".format(MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]))
        # else:
        #     string = "Game over"

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
            return self.py_visualizer._create_image_array(self.py_visualizer.surf, (VIDEO_W, VIDEO_H))

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
        seed_rng = np.random.RandomState(seed=seed)
        self.agents = self.possible_agents[:]
        self.env_steps = 0
        self.seed = seed
        board = Board(seed=seed, env_cfg=self.env_cfg)
        weather_schedule = weather.generate_weather_schedule(seed_rng, self.state.env_cfg)
        self.state: State = State(seed_rng=seed_rng, seed=seed, env_cfg=self.state.env_cfg, env_steps=0, board=board, weather_schedule=weather_schedule)
        self.max_episode_length = self.env_cfg.max_episode_length
        for agent in self.possible_agents:
            self.state.units[agent] = OrderedDict()
            self.state.factories[agent] = OrderedDict()
        obs = self.state.get_obs()
        observations = {agent: obs for agent in self.agents}
        return observations

    def _log(self, *m):
        if self.env_cfg.verbose > 0:
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
            if "faction" in a and "bid" in a:
                self.state.teams[k] = Team(
                    team_id=self.agent_name_mapping[k], agent=k, faction=FactionTypes[a["faction"]]
                )
                self.state.teams[k].init_water = self.env_cfg.INIT_WATER_METAL_PER_FACTORY * (self.state.board.factories_per_team)
                self.state.teams[k].init_metal = self.env_cfg.INIT_WATER_METAL_PER_FACTORY * (self.state.board.factories_per_team)
                self.state.teams[k].factories_to_place = self.state.board.factories_per_team
                # verify bid is valid
                valid_action = True
                bid = a["bid"]
                if bid < 0 or bid > self.state.teams[k].init_water:
                    valid_action = False
                if not valid_action:
                    failed_agents[k] = True
                    continue
                if bid > highest_bid:
                    highest_bid = bid
                    highest_bid_agent = k
                elif bid == highest_bid:
                    # no one gets the extra factory
                    highest_bid_agent = "tie"
            else:
                # team k loses
                failed_agents[k] = True
        if highest_bid_agent == "tie":
            pass
        elif highest_bid_agent is None:
            # no valid bids made, all agents failed.
            pass
        else:
            self.state.teams[highest_bid_agent].init_water -= highest_bid
            self.state.teams[highest_bid_agent].init_metal -= highest_bid
            self.state.teams[highest_bid_agent].factories_to_place += 1
        return failed_agents
    def _handle_factory_placement_step(self, actions):
        # factory placement rounds
        failed_agents = {agent: False for agent in self.agents}
        for k, a in actions.items():
            if a is None:
                failed_agents[k] = True
                continue
            if k not in self.agents:
                raise ValueError(f"Invalid player {k}")
            if "spawn" in a and "metal" in a and "water" in a:
                if self.state.teams[k].factories_to_place <= 0:
                    self._log(f"{k} cannot place additional factories. Cancelled placement of factory")
                    continue
                if a["water"] < 0 or a["metal"] < 0:
                    self._log(f"{k} tried to place negative water/metal in factory. Cancelled placement of factory")
                    continue
                if a["water"] > self.state.teams[k].init_water:
                    self._log(f"{k} does not have enough water. Cancelled placement of factory")
                    continue
                if a["metal"] > self.state.teams[k].init_metal:
                    self._log(f"{k} does not have enough metal. Cancelled placement of factory")
                    continue

                factory = self.add_factory(self.state.teams[k], a["spawn"])
                if factory is None: continue
                factory.cargo.water = a["water"]
                factory.cargo.metal = a["metal"]
                factory.power = self.env_cfg.INIT_POWER_PER_FACTORY
                self.state.teams[k].factories_to_place -= 1
                self.state.teams[k].init_metal -= a["metal"]
                self.state.teams[k].init_water -= a["water"]
            else:
                # pass, turn is skipped.
                pass
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
                    team_id=self.agent_name_mapping[k], agent=k, faction=FactionTypes[a["faction"]]
                )
                if len(a["spawns"]) > self.state.board.factories_per_team:
                    if self.env_cfg.verbose > 0: self._log(f"{k} tried to spawn more factories than allocated in board.factories_per_team. Spawning only the first {self.state.board.factories_per_team} locations")
                for spawn_loc in a["spawns"][:self.state.board.factories_per_team]:
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
        for unit, transfer_action in actions_by_type["transfer"]:
            transfer_action: TransferAction
            transfer_amount = unit.sub_resource(transfer_action.resource, transfer_action.transfer_amount)
            transfer_pos: Position = unit.pos + move_deltas[transfer_action.transfer_dir]
            units_there = self.state.board.get_units_at(transfer_pos)

            # if there is a factory, we prefer transferring to that entity
            factory_id = f"factory_{self.state.board.factory_occupancy_map[transfer_pos.y, transfer_pos.x]}"
            if factory_id in self.state.factories[unit.team.agent]:
                factory = self.state.factories[unit.team.agent][factory_id]
                actually_transferred = factory.add_resource(transfer_action.resource, transfer_action.transfer_amount)
            elif units_there is not None:
                assert len(units_there) == 1, "Fatal error here, this is a bug"
                target_unit = units_there[0]
                # add resources to target. This will waste (transfer_amount - actually_transferred) resources
                actually_transferred = target_unit.add_resource(transfer_action.resource, transfer_amount)
    def _handle_pickup_actions(self, actions_by_type: ActionsByType):
        for unit, pickup_action in actions_by_type["pickup"]:
            pickup_action: PickupAction
            factory = self.state.board.get_factory_at(self.state, unit.pos)
            pickup_amount = factory.sub_resource(pickup_action.resource, pickup_action.pickup_amount)
            # may waste resources if tried to pickup more than one can hold.
            actually_pickedup = unit.add_resource(pickup_action.resource, pickup_amount)

    def _handle_dig_actions(self, actions_by_type: ActionsByType, weather_cfg):
        for unit, dig_action in actions_by_type["dig"]:
            dig_action: DigAction
            if self.state.board.rubble[unit.pos.y, unit.pos.x] > 0:
                self.state.board.rubble[unit.pos.y, unit.pos.x] = max(self.state.board.rubble[unit.pos.y, unit.pos.x] - unit.unit_cfg.DIG_RUBBLE_REMOVED, 0)
            elif self.state.board.lichen[unit.pos.y, unit.pos.x] > 0:
                self.state.board.lichen[unit.pos.y, unit.pos.x] = max(self.state.board.lichen[unit.pos.y, unit.pos.x] - unit.unit_cfg.DIG_LICHEN_REMOVED, 0)
            elif self.state.board.ice[unit.pos.y, unit.pos.x] > 0:
                unit.add_resource(0, unit.unit_cfg.DIG_RESOURCE_GAIN)
            elif self.state.board.ore[unit.pos.y, unit.pos.x] > 0:
                unit.add_resource(1, unit.unit_cfg.DIG_RESOURCE_GAIN)
            unit.power -= math.ceil(self.state.env_cfg.ROBOTS[unit.unit_type.name].DIG_COST * weather_cfg["power_loss_factor"])
    def _handle_self_destruct_actions(self, actions_by_type: ActionsByType):
        for unit, self_destruct_action in actions_by_type["self_destruct"]:
            unit: Unit
            self_destruct_action: SelfDestructAction
            pos_hash = self.state.board.pos_hash(unit.pos)
            del self.state.board.units_map[pos_hash]
            self.destroy_unit(unit)
    def _handle_factory_build_actions(self, actions_by_type: ActionsByType, weather_cfg):
        for factory, factory_build_action in actions_by_type["factory_build"]:
            factory: Factory
            factory_build_action: FactoryBuildAction
            team = self.state.teams[factory.team.agent]
            self.add_unit(
                team=team,
                unit_type=factory_build_action.unit_type,
                pos=factory.pos.pos,
            )
            if factory_build_action.unit_type == UnitType.HEAVY:
                factory.sub_resource(3, self.env_cfg.ROBOTS["HEAVY"].METAL_COST)
                factory.sub_resource(4, math.ceil(self.env_cfg.ROBOTS["HEAVY"].POWER_COST * weather_cfg["power_loss_factor"]))
            else:
                factory.sub_resource(3, self.env_cfg.ROBOTS["LIGHT"].METAL_COST)
                factory.sub_resource(4, math.ceil(self.env_cfg.ROBOTS["LIGHT"].POWER_COST * weather_cfg["power_loss_factor"]))
    def _handle_movement_actions(self, actions_by_type: ActionsByType, weather_cfg):
        new_units_map: Dict[str, List[Unit]] = defaultdict(list)
        heavy_entered_pos: Dict[str, List[Unit]] = defaultdict(list)
        light_entered_pos: Dict[str, List[Unit]] = defaultdict(list)

        for unit, move_action in actions_by_type["move"]:
            move_action: MoveAction
            # skip move center
            if move_action.move_dir == 0:
                continue
            old_pos_hash = self.state.board.pos_hash(unit.pos)
            target_pos = unit.pos + move_action.dist * move_deltas[move_action.move_dir]
            rubble = self.state.board.rubble[target_pos.y, target_pos.x]
            power_required = unit.unit_cfg.MOVE_COST + unit.unit_cfg.RUBBLE_MOVEMENT_COST * rubble
            power_required = math.ceil(power_required * weather_cfg["power_loss_factor"])
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

        for pos_hash, units in self.state.board.units_map.items():
            # add in all the stationary units
            new_units_map[pos_hash] += units

        destroyed_units: Set[Unit] = set()
        new_units_map_after_collision: Dict[str, List[Unit]] = defaultdict(list)
        for pos_hash, units in new_units_map.items():
            if len(units) <= 1:
                new_units_map_after_collision[pos_hash] += units
                continue
            if len(heavy_entered_pos[pos_hash]) > 1:
                # all units collide and break
                for u in units:
                    destroyed_units.add(u)
                self._log(f"{len(destroyed_units)} Units collided at {pos_hash}")
            elif len(heavy_entered_pos[pos_hash]) > 0:
                # all other units collide and break
                surviving_unit = heavy_entered_pos[pos_hash][0]
                for u in units:
                    if u.unit_id != surviving_unit.unit_id:
                        destroyed_units.add(u)
                self._log(f"{len(destroyed_units)} Units collided at {pos_hash} with {surviving_unit} surviving")
                new_units_map_after_collision[pos_hash].append(surviving_unit)
            else:
                # check for stationary heavy unit there
                surviving_unit = None
                heavy_stationary_unit = None
                for u in units:
                    if u.unit_type == UnitType.HEAVY:
                        if heavy_stationary_unit is not None:
                            heavy_stationary_unit = None
                            # we found >= 2 heavies stationary in a tile where no heavies are entering.
                            self._log(f"At {pos_hash}, >= 2 heavies crashed as they were all stationary")
                            break
                        heavy_stationary_unit = u

                if heavy_stationary_unit is not None:
                    surviving_unit = heavy_stationary_unit
                else:
                    if len(light_entered_pos[pos_hash]) > 1:
                        # all units collide
                        surviving_unit = None
                    elif len(light_entered_pos[pos_hash]) > 0:
                        # light crashes into stationary light unit
                        surviving_unit = light_entered_pos[pos_hash][0]
                if surviving_unit is None:
                    for u in units:
                        destroyed_units.add(u)
                    self._log(f"{len(destroyed_units)} Units collided at {pos_hash}")
                else:
                    for u in units:
                        if u.unit_id != surviving_unit.unit_id:
                            destroyed_units.add(u)
                    self._log(f"{len(destroyed_units)} Units collided at {pos_hash} with {surviving_unit} surviving")
                    new_units_map_after_collision[pos_hash].append(surviving_unit)
        self.state.board.units_map = new_units_map_after_collision

        for u in destroyed_units:
            self.destroy_unit(u)
    def _handle_recharge_actions(self, actions_by_type: ActionsByType):
        # for recharging actions, check if unit has enough power. If not, add action back to the queue
        for unit, recharge_action in actions_by_type["recharge"]:
            recharge_action: RechargeAction
            if unit.power < recharge_action.power:
                unit.action_queue.insert(0, recharge_action)
                # by default actions with repeat=True will be placed to the back of queue
                # remove from back of queue if we re-inserted into the front
                if recharge_action.repeat:
                    unit.action_queue = unit.action_queue[:-1]
    def _handle_factory_water_actions(self, actions_by_type: ActionsByType):
        for factory, factory_water_action in actions_by_type["factory_water"]:
            factory_water_action: FactoryWaterAction
            water_cost = factory.water_cost(self.env_cfg)
            factory.cargo.water -= water_cost # earlier validation ensures this is always possible.
            indexable_positions = ([v[1] for v in factory.grow_lichen_positions], [v[0] for v in factory.grow_lichen_positions])
            self.state.board.lichen[indexable_positions] += 2
            self.state.board.lichen_strains[indexable_positions] = factory.num_id
    def step(self, actions):
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
        if self.env_cfg.BIDDING_SYSTEM and self.env_steps <= self.state.board.factories_per_team + 1:
            early_game = True

        if early_game:
            failed_agents = self._step_early_game(actions)
        else:
            # handle weather effects
            current_weather = self.state.weather_schedule[self.state.real_env_steps]
            current_weather = self.state.env_cfg.WEATHER_ID_TO_NAME[current_weather]
            weather_cfg = weather.apply_weather(self.state, self.agents, current_weather)

            # 1. Check for malformed actions
            if self.env_cfg.validate_action_space:
                # This part is not absolutely necessary if you know for sure your actions are well formatted
                for agent, unit_actions in actions.items():
                    valid_acts, err_reason = self.action_space(agent).contains(unit_actions)
                    if not valid_acts:
                        failed_agents[agent] = True
                        raise ValueError(f"{self.state.teams[agent]} Inappropriate action given. {err_reason}")

            # we should except that actions is always of type dict, if not then erroring here is fine
            for agent, unit_actions in actions.items():
                try:
                    for unit_id, action in unit_actions.items():
                        if "factory" in unit_id:
                            self.state.factories[agent][unit_id].action_queue.append(format_factory_action(action))
                        elif "unit" in unit_id:
                            unit = self.state.units[agent][unit_id]
                            # if unit does not have more than UNIT_ACTION_QUEUE_POWER_COST[unit.unit_type.name] power, we skip updating the action queue and print warning
                            update_power_req = self.state.env_cfg.UNIT_ACTION_QUEUE_POWER_COST[unit.unit_type.name] * weather_cfg["power_loss_factor"]
                            if unit.power < update_power_req:
                                self._log(f"Tried to update action queue for {unit} requiring ({self.state.env_cfg.UNIT_ACTION_QUEUE_POWER_COST[unit.unit_type.name]} x {weather_cfg['power_loss_factor']}) = {update_power_req} power but only had {unit.power} power. Power cost factor is {weather_cfg['power_loss_factor']} ")
                                continue
                            unit.power -= update_power_req
                            formatted_actions = []
                            if type(action) == list or (type(action) == np.ndarray and len(action.shape) == 2):
                                trunked_actions = action[: self.env_cfg.UNIT_ACTION_QUEUE_SIZE]
                                formatted_actions = [format_action_vec(a) for a in trunked_actions]
                            else:
                                formatted_actions = [format_action_vec(action)]
                            self.state.units[agent][unit_id].action_queue = formatted_actions
                except ValueError as e:
                    # catch errors when trying to format unit or factory actions
                    print(e)
                    failed_agents[agent] = True
        
            # 2. store actions by type
            actions_by_type: ActionsByType = defaultdict(list)
            for agent in self.agents:
                if failed_agents[agent]: # skip failed agents
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

            for agent in self.agents:
                for factory in self.state.factories[agent].values():
                    # update information for lichen growing and cache it
                    factory.cache_water_info(self.state.board, self.env_cfg)

            # 3. validate all actions against current state, throw away impossible actions TODO
            actions_by_type = validate_actions(self.env_cfg, self.state, actions_by_type, verbose=self.env_cfg.verbose, weather_cfg=weather_cfg)

            self._handle_transfer_actions(actions_by_type)
            self._handle_pickup_actions(actions_by_type)
            self._handle_dig_actions(actions_by_type, weather_cfg)
            self._handle_self_destruct_actions(actions_by_type)
            self._handle_factory_build_actions(actions_by_type, weather_cfg)
            self._handle_movement_actions(actions_by_type, weather_cfg)
            self._handle_recharge_actions(actions_by_type)
            self._handle_factory_water_actions(actions_by_type)
            
            # Update lichen
            self.state.board.lichen -= 1
            self.state.board.lichen = self.state.board.lichen.clip(0)
            self.state.board.lichen_strains[self.state.board.lichen == 0] = -1

            # resources refining
            for agent in self.agents:
                factories_to_destroy: Set[Factory] = set()
                for factory in self.state.factories[agent].values():
                    factory.refine_step(self.env_cfg)
                    factory.cargo.water -= self.env_cfg.FACTORY_WATER_CONSUMPTION
                    if factory.cargo.water < 0:
                        factories_to_destroy.add(factory)
                for factory in factories_to_destroy:
                    # destroy factories that ran out of water
                    self.destroy_factory(factory)
            # power gain
            if is_day(self.env_cfg, self.state.real_env_steps):
                for agent in self.agents:
                    for u in self.state.units[agent].values():
                        u.power = u.power + math.ceil(self.env_cfg.ROBOTS[u.unit_type.name].CHARGE * weather_cfg["power_gain_factor"])
            for agent in self.agents:
                for f in self.state.factories[agent].values():
                    # Factories are immune to weather thanks to using nuclear reactors instead
                    f.power = f.power + math.ceil(self.env_cfg.FACTORY_CHARGE)


        # always set rubble under factories to 0.
        self.state.board.rubble[self.state.board.factory_occupancy_map != -1] = 0

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {}
        for agent in self.agents:
            strain_ids = self.state.teams[agent].factory_strains
            if failed_agents[agent]:
                rewards[agent] = -1000
            else:
                agent_lichen_mask = np.isin(self.state.board.lichen_strains, strain_ids)
                rewards[agent] = self.state.board.lichen[agent_lichen_mask].sum()

        self.env_steps += 1
        self.state.env_steps += 1
        env_done = self.state.real_env_steps >= self.state.env_cfg.max_episode_length
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
        unit = Unit(team=team, unit_type=unit_type, unit_id=f"unit_{self.state.global_id}", env_cfg=self.env_cfg)
        unit.pos.pos = pos.copy()
        self.state.global_id += 1
        self.state.units[team.agent][unit.unit_id] = unit
        self.state.board.units_map[self.state.board.pos_hash(unit.pos)].append(unit)
        return unit

    def add_factory(self, team: Team, pos: np.ndarray):
        factory = Factory(team=team, unit_id=f"factory_{self.state.global_id}", num_id=self.state.global_id)
        factory.pos.pos = list(pos)
        factory.cargo.water = self.env_cfg.INIT_WATER_METAL_PER_FACTORY
        factory.cargo.metal = self.env_cfg.INIT_WATER_METAL_PER_FACTORY
        factory.power = self.env_cfg.INIT_POWER_PER_FACTORY
        if not self.state.board.spawn_masks[team.agent][pos[0], pos[1]]:
            self._log(f"{team.agent} cannot place factory at {pos[0]}, {pos[1]} as it is on the other half of map.")
            return None
        if self.state.board.factory_occupancy_map[factory.pos_slice].sum() >= 0:
            self._log(f"{team.agent} cannot overlap factory placement. Existing factory at {factory.pos} already.")
            return None
        
        self.state.teams[team.agent].factory_strains += [factory.num_id]

        self.state.factories[team.agent][factory.unit_id] = factory
        self.state.board.factory_map[self.state.board.pos_hash(factory.pos)] = factory
        self.state.board.factory_occupancy_map[factory.pos_slice] = factory.num_id
        self.state.board.rubble[factory.pos_slice] = 0
        self.state.board.ice[factory.pos_slice] = 0
        self.state.board.ore[factory.pos_slice] = 0

        self.state.global_id += 1
        return factory

    def destroy_unit(self, unit: Unit):
        """
        # NOTE this doesn't remove unit reference from board map
        """
        self.state.board.rubble[unit.pos.y, unit.pos.x] = min(
            self.state.board.rubble[unit.pos.y, unit.pos.x] + unit.unit_cfg.RUBBLE_AFTER_DESTRUCTION,
            self.env_cfg.MAX_RUBBLE,
        )
        del self.state.units[unit.team.agent][unit.unit_id]

    def destroy_factory(self, factory: Factory):
        # spray rubble on every factory tile
        self.state.board.rubble[factory.pos_slice] += self.env_cfg.FACTORY_RUBBLE_AFTER_DESTRUCTION
        self.state.board.rubble[factory.pos_slice] = self.state.board.rubble[factory.pos_slice].clip(0, self.env_cfg.MAX_RUBBLE)
        self.state.board.factory_occupancy_map[factory.pos_slice] = -1
        del self.state.factories[factory.team.agent][factory.unit_id]
        del self.state.board.factory_map[self.state.board.pos_hash(factory.pos)]

def raw_env() -> LuxAI2022:
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = LuxAI2022()
    # env = parallel_to_aec(env)
    return env
