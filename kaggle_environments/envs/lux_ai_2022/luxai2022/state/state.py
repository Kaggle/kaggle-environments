from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from luxai2022.config import EnvConfig
from luxai2022.factory import Factory
from luxai2022.map.board import Board
from luxai2022.team import Team

from luxai2022.unit import Unit
from collections import OrderedDict
import copy

@dataclass
class State:
    seed_rng: np.random.RandomState
    seed: int
    env_steps: int
    env_cfg: EnvConfig
    board: Board = None
    weather_schedule: np.ndarray = None
    units: Dict[str, Dict[str, Unit]] = field(default_factory=dict)
    factories: Dict[str, Dict[str, Factory]] = field(default_factory=dict)
    teams: Dict[str, Team] = field(default_factory=dict)
    global_id: int = 0
    stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
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

    def generate_unit_data(units_dict: Dict[str, Dict[str, Unit]]):
        units = dict()
        for team in units_dict:
            units[team] = dict()
            for unit in units_dict[team].values():
                state_dict = unit.state_dict()
                # if self.env_cfg.UNIT_ACTION_QUEUE_SIZE == 1:
                #     # if config is such that action queue is size 1, we do not include the queue as it is always empty
                #     del state_dict["action_queue"]
                units[team][unit.unit_id] = state_dict
        return units
    def generate_team_data(teams_dict):
        teams = dict()
        for k, v in teams_dict.items():
            teams[k] = v.state_dict()
        return teams
    def generate_factory_data(factories_dict):
        factories = dict()
        for team in factories_dict:
            factories[team] = dict()
            for factory in factories_dict[team].values():
                state_dict = factory.state_dict()
                factories[team][factory.unit_id] = state_dict
        return factories

    def get_obs(self):
        units = State.generate_unit_data(self.units)
        teams = State.generate_team_data(self.teams)
        factories = State.generate_factory_data(self.factories)
        board = self.board.state_dict()
        return dict(
            units=units,
            teams=teams,
            factories=factories,
            board=board,
            weather_schedule=self.weather_schedule,
            real_env_steps=self.real_env_steps
        )
    def get_compressed_obs(self):
        # return everything on turn 0
        if self.env_steps == 0:
            return self.get_obs()
        data = self.get_obs()
        # convert lichen and lichen strains to sparse matrix format?
        del data["board"]["ore"]
        del data["board"]["ice"]
        del data["weather_schedule"]
        if self.real_env_steps >= 0:
            del data["board"]["valid_spawns_mask"]
        return data
    def get_change_obs(self, prev_state):
        """
        returns sparse dicts for large matrices of where values change only
        """
        data = self.get_compressed_obs()

        data["board"]["rubble"] = dict()
        data["board"]["lichen"] = dict()
        data["board"]["lichen_strains"] = dict()
        change_indices = np.argwhere(self.board.rubble != prev_state["board"]["rubble"])
        for ind in change_indices:
            x,y = ind[0], ind[1]
            data["board"]["rubble"][f"{x},{y}"] = self.board.rubble[x, y]
        change_indices = np.argwhere(self.board.lichen != prev_state["board"]["lichen"])
        for ind in change_indices:
            x,y = ind[0], ind[1]
            data["board"]["lichen"][f"{x},{y}"] = self.board.lichen[x, y]
        change_indices = np.argwhere(self.board.lichen_strains != prev_state["board"]["lichen_strains"])
        for ind in change_indices:
            x,y = ind[0], ind[1]
            data["board"]["lichen_strains"][f"{x},{y}"] = self.board.lichen_strains[x, y]
        return data

    def from_obs(obs):
        # generate state from compressed obs
        pass
