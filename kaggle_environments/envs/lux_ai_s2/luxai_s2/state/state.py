import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List

try:
    from typing import TypedDict    
except:
    from typing_extensions import TypedDict

import numpy as np
import numpy.typing as npt

from luxai_s2.actions import format_action_vec
from luxai_s2.config import EnvConfig
from luxai_s2.factory import Factory, FactoryStateDict
from luxai_s2.map.board import Board, BoardStateDict
from luxai_s2.map_generator.generator import GameMap
from luxai_s2.state.stats import StatsStateDict
from luxai_s2.team import Team, TeamStateDict
from luxai_s2.unit import (FactionTypes, Unit, UnitCargo, UnitStateDict,
                           UnitType)


class SparseBoardStateDict(TypedDict):
    rubble: Dict[str, int]
    lichen: Dict[str, int]
    lichen_strains: Dict[str, int]
    factories_per_team: int


class DeltaObservationStateDict(TypedDict):
    units: int
    teams: Dict[str, TeamStateDict]
    factories: Dict[str, Dict[str, FactoryStateDict]]
    board: SparseBoardStateDict
    real_env_steps: int
    global_id: int


class ObservationStateDict(TypedDict):
    units: Dict[str, Dict[str, UnitStateDict]]
    teams: Dict[str, TeamStateDict]
    factories: Dict[str, Dict[str, FactoryStateDict]]
    board: BoardStateDict
    real_env_steps: int
    global_id: int


@dataclass
class State:
    seed_rng: np.random.RandomState
    seed: int
    env_steps: int
    env_cfg: EnvConfig
    board: Board = None
    units: Dict[str, Dict[str, Unit]] = field(default_factory=dict)
    factories: Dict[str, Dict[str, Factory]] = field(default_factory=dict)
    teams: Dict[str, Team] = field(default_factory=dict)
    global_id: int = 0
    stats: Dict[str, StatsStateDict] = field(default_factory=dict)

    @property
    def real_env_steps(self):
        """
        the actual env step in the environment, which subtracts the time spent bidding and placing factories
        """
        if self.env_cfg.BIDDING_SYSTEM:
            # + 1 for bidding step, * 2 for total factories placed by two teams
            return self.env_steps - (self.board.factories_per_team * 2 + 1)
        else:
            return self.env_steps

    def real_env_steps_to_env_steps(self, real_env_steps):
        if self.env_cfg.BIDDING_SYSTEM:
            # + 1 for bidding step, * 2 for total factories placed by two teams
            return real_env_steps + (self.board.factories_per_team * 2 + 1)
        else:
            return real_env_steps

    # can be somewhat slow, can we parallelize this easily?
    @staticmethod
    def generate_unit_data(units_dict: Dict[str, Dict[str, Unit]]):
        units = dict()
        for team in units_dict:
            units[team] = dict()
            for unit in units_dict[team].values():
                state_dict = unit.state_dict()
                units[team][unit.unit_id] = state_dict
        return units

    @staticmethod
    def generate_team_data(teams_dict):
        teams = dict()
        for k, v in teams_dict.items():
            teams[k] = v.state_dict()
        return teams

    @staticmethod
    def generate_factory_data(factories_dict):
        factories = dict()
        for team in factories_dict:
            factories[team] = dict()
            for factory in factories_dict[team].values():
                state_dict = factory.state_dict()
                factories[team][factory.unit_id] = state_dict
        return factories

    def get_obs(self) -> ObservationStateDict:
        units = State.generate_unit_data(self.units)
        teams = State.generate_team_data(self.teams)
        factories = State.generate_factory_data(self.factories)
        board = self.board.state_dict()
        return dict(
            units=units,
            teams=teams,
            factories=factories,
            board=board,
            real_env_steps=self.real_env_steps,
            global_id=self.global_id,
        )

    def get_compressed_obs(self):
        # return everything on turn 0
        if self.env_steps == 0:
            return self.get_obs()
        data = self.get_obs()
        # convert lichen and lichen strains to sparse matrix format?
        del data["board"]["ore"]
        del data["board"]["ice"]
        if self.real_env_steps >= 0:
            del data["board"]["valid_spawns_mask"]
        return data

    def get_change_obs(
        self, prev_state: ObservationStateDict
    ) -> DeltaObservationStateDict:
        """
        returns sparse dicts for large matrices of where values change only by comparing against a given previous observation/state
        """
        data = self.get_compressed_obs()

        data["board"]["rubble"] = dict()
        data["board"]["lichen"] = dict()
        data["board"]["lichen_strains"] = dict()
        change_indices = np.argwhere(self.board.rubble != prev_state["board"]["rubble"])
        for ind in change_indices:
            x, y = ind[0], ind[1]
            data["board"]["rubble"][f"{x},{y}"] = self.board.rubble[x, y]
        change_indices = np.argwhere(self.board.lichen != prev_state["board"]["lichen"])
        for ind in change_indices:
            x, y = ind[0], ind[1]
            data["board"]["lichen"][f"{x},{y}"] = self.board.lichen[x, y]
        change_indices = np.argwhere(
            self.board.lichen_strains != prev_state["board"]["lichen_strains"]
        )
        for ind in change_indices:
            x, y = ind[0], ind[1]
            data["board"]["lichen_strains"][f"{x},{y}"] = self.board.lichen_strains[
                x, y
            ]
        return data

    @staticmethod
    def accumulate_board_changes(board: Board, board_change_observations: List):
        # Accumulates the delta changes to a board from change observations.
        # Should be used with `from_obs` if the obs given is missing initial observation info
        for obs in board_change_observations:
            for item in ["rubble", "lichen", "lichen_strains"]:
                for k, v in obs[item].items():
                    k = k.split(",")
                    x, y = int(k[0]), int(k[1])
                    board.__getattribute__(item)[x, y] = v

    @classmethod
    def from_obs(cls, obs: ObservationStateDict, env_cfg: EnvConfig):
        real_env_steps = obs["real_env_steps"]
        factories_per_team = obs["board"]["factories_per_team"]
        if env_cfg.BIDDING_SYSTEM:
            # + 1 for bidding step, * 2 for total factories placed by two teams
            env_steps = real_env_steps + (factories_per_team * 2 + 1)
        else:
            env_steps = real_env_steps

        teams = dict()
        for agent in obs["teams"]:
            team_data = obs["teams"][agent]
            faction = FactionTypes[team_data["faction"]]
            team = Team(team_id=team_data["team_id"], agent=agent, faction=faction)
            team.bid = team_data["bid"]
            team.init_water = team_data["water"]
            team.init_metal = team_data["metal"]
            team.factories_to_place = team_data["factories_to_place"]
            team.factory_strains = team_data["factory_strains"]
            team.place_first = team_data["place_first"]
            teams[agent] = team

        existing_map = GameMap(
            np.array(obs["board"]["rubble"]),
            np.array(obs["board"]["ice"]),
            np.array(obs["board"]["ore"]),
            None,
        )

        board = Board(seed=0, env_cfg=env_cfg, existing_map=existing_map)
        board.factories_per_team = factories_per_team

        units = dict()
        for agent in obs["units"]:
            units[agent] = dict()
            for unit_id in obs["units"][agent]:
                unit_data = obs["units"][agent][unit_id]
                cargo = UnitCargo(**unit_data["cargo"])
                unit = Unit(
                    teams[agent], UnitType[unit_data["unit_type"]], unit_id, env_cfg
                )
                unit.pos.pos = np.array(unit_data["pos"])
                unit.cargo = cargo
                unit.power = unit_data["power"]
                unit.action_queue = [
                    format_action_vec(a) for a in unit_data["action_queue"]
                ]

                units[agent][unit_id] = unit
                board.units_map[board.pos_hash(unit.pos)].append(unit)

        factory_occupancy_map = np.ones_like(obs["board"]["rubble"], dtype=int) * -1
        factories = dict()
        for agent in obs["factories"]:
            factories[agent] = dict()
            for unit_id in obs["factories"][agent]:
                f_data = obs["factories"][agent][unit_id]
                cargo = UnitCargo(**f_data["cargo"])
                factory = Factory(teams[agent], unit_id, f_data["strain_id"])
                factory.cargo = cargo
                factory.power = f_data["power"]
                factory.pos.pos = np.array(f_data["pos"])
                factories[agent][unit_id] = factory

                # code below mimics env.py's add_factory function
                factory_occupancy_map[factory.pos_slice] = factory.num_id

                board.factory_map[board.pos_hash(factory.pos)] = factory

                invalid_spawn_indices = factory.min_dist_slice
                for x, y in invalid_spawn_indices:
                    if (
                        x < 0
                        or y < 0
                        or x >= board.rubble.shape[0]
                        or y >= board.rubble.shape[1]
                    ):
                        continue
                    board.valid_spawns_mask[x, y] = False

                # bottom 3 might not be necessary
                board.rubble[factory.pos_slice] = 0
                board.ice[factory.pos_slice] = 0
                board.ore[factory.pos_slice] = 0

        # update maps inside board
        board.factory_occupancy_map = factory_occupancy_map

        # generate state from full observation
        return cls(
            seed_rng=np.random.RandomState(seed=0),  # unused
            seed=0,  # unused
            env_steps=env_steps,
            env_cfg=env_cfg,
            board=board,
            units=units,
            teams=teams,
            factories=factories,
            global_id=obs["global_id"],
            stats=dict(),  # not saved
        )
