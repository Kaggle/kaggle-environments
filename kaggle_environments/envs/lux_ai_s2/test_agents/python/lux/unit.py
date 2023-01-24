import math
import sys
from typing import List
import numpy as np
from dataclasses import dataclass
if __package__ == "":
    from lux.cargo import UnitCargo
    from lux.config import EnvConfig
else:
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

    @property
    def agent_id(self):
        if self.team_id == 0: return "player_0"
        return "player_1"

    def action_queue_cost(self, game_state):
        cost = self.env_cfg.ROBOTS[self.unit_type].ACTION_QUEUE_POWER_COST
        return cost

    def move_cost(self, game_state, direction):
        board = game_state.board
        target_pos = self.pos + move_deltas[direction]
        if target_pos[0] < 0 or target_pos[1] < 0 or target_pos[1] >= len(board.rubble) or target_pos[0] >= len(board.rubble[0]):
            # print("Warning, tried to get move cost for going off the map", file=sys.stderr)
            return None
        factory_there = board.factory_occupancy_map[target_pos[0], target_pos[1]]
        if factory_there not in game_state.teams[self.agent_id].factory_strains and factory_there != -1:
            # print("Warning, tried to get move cost for going onto a opposition factory", file=sys.stderr)
            return None
        rubble_at_target = board.rubble[target_pos[0]][target_pos[1]]
        
        return math.floor(self.unit_cfg.MOVE_COST + self.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target)
    def move(self, direction, repeat=0, n=1):
        if isinstance(direction, int):
            direction = direction
        else:
            pass
        return np.array([0, direction, 0, 0, repeat, n])

    def transfer(self, transfer_direction, transfer_resource, transfer_amount, repeat=0, n=1):
        assert transfer_resource < 5 and transfer_resource >= 0
        assert transfer_direction < 5 and transfer_direction >= 0
        return np.array([1, transfer_direction, transfer_resource, transfer_amount, repeat, n])
    
    def pickup(self, pickup_resource, pickup_amount, repeat=0, n=1):
        assert pickup_resource < 5 and pickup_resource >= 0
        return np.array([2, 0, pickup_resource, pickup_amount, repeat, n])
    
    def dig_cost(self, game_state):
        return self.unit_cfg.DIG_COST
    def dig(self, repeat=0, n=1):
        return np.array([3, 0, 0, 0, repeat, n])

    def self_destruct_cost(self, game_state):
        return self.unit_cfg.SELF_DESTRUCT_COST
    def self_destruct(self, repeat=0, n=1):
        return np.array([4, 0, 0, 0, repeat, n])

    def recharge(self, x, repeat=0, n=1):
        return np.array([5, 0, 0, x, repeat, n])

    def __str__(self) -> str:
        out = f"[{self.team_id}] {self.unit_id} {self.unit_type} at {self.pos}"
        return out