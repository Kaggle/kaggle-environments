import math
import sys
from typing import List
import numpy as np
from dataclasses import dataclass
if __package__ == "":
    from lux.weather import get_weather_config
    from lux.cargo import UnitCargo
    from lux.config import EnvConfig
else:
    from .weather import get_weather_config
    from .cargo import UnitCargo
    from .config import EnvConfig
# a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])

@dataclass
class Unit:
    team_id: int
    unit_id: str
    unit_type: str # "LIGHT" or "HEAVY"
    pos: np.ndarray
    power: int
    cargo: UnitCargo
    env_cfg: EnvConfig
    unit_cfg: dict
    action_queue: List

    def action_queue_cost(self, game_state):
        cost = self.env_cfg.UNIT_ACTION_QUEUE_POWER_COST[self.unit_type]
        current_weather = game_state.weather_schedule[game_state.real_env_steps]
        weather_cfg = get_weather_config(current_weather, self.env_cfg)
        return cost * weather_cfg["power_loss_factor"]

    def move_cost(self, game_state, direction):
        board = game_state.board
        target_pos = self.pos + move_deltas[direction]
        if target_pos[0] < 0 or target_pos[1] < 0 or target_pos[1] >= len(board.rubble) or target_pos[0] >= len(board.rubble[0]):
            # print("Warning, tried to get move cost for going off the map", file=sys.stderr)
            return None
        factory_there = board.factory_occupancy_map[target_pos[1], target_pos[0]]
        if factory_there != self.team_id and factory_there != -1:
            # print("Warning, tried to get move cost for going onto a opposition factory", file=sys.stderr)
            return None
        rubble_at_target = board.rubble[target_pos[1]][target_pos[0]]
        
        current_weather = game_state.weather_schedule[game_state.real_env_steps]
        weather_cfg = get_weather_config(current_weather, self.env_cfg)
        return math.ceil((self.unit_cfg.MOVE_COST + self.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target) * weather_cfg["power_loss_factor"])
    def move(self, direction, repeat=True):
        if isinstance(direction, int):
            direction = direction
        else:
            pass
        return np.array([0, direction, 0, 0, 1 if repeat else 0])

    def transfer(self, transfer_direction, transfer_resource, transfer_amount, repeat=True):
        assert transfer_resource < 5 and transfer_resource >= 0
        assert transfer_direction < 5 and transfer_direction >= 0
        return np.array([1, transfer_direction, transfer_resource, transfer_amount, 1 if repeat else 0])
    
    def pickup(self, pickup_resource, pickup_amount, repeat=True):
        assert pickup_resource < 5 and pickup_resource >= 0
        return np.array([2, 0, pickup_resource, pickup_amount, 1 if repeat else 0])
    
    def dig_cost(self, game_state):
        current_weather = game_state.weather_schedule[game_state.real_env_steps]
        weather_cfg = get_weather_config(current_weather, self.env_cfg)
        return math.ceil(self.unit_cfg.DIG_COST * weather_cfg["power_loss_factor"])
    def dig(self, repeat=True):
        return np.array([3, 0, 0, 0, 1 if repeat else 0])

    def self_destruct_cost(self, game_state):
        current_weather = game_state.weather_schedule[game_state.real_env_steps]
        weather_cfg = get_weather_config(current_weather, self.env_cfg)
        return math.ceil(self.unit_cfg.SELF_DESTRUCT_COST * weather_cfg["power_loss_factor"])
    def self_destruct(self, repeat=True):
        return np.array([4, 0, 0, 0, 1 if repeat else 0])

    def recharge(self, x, repeat=True):
        return np.array([5, 0, 0, x, 1 if repeat else 0])