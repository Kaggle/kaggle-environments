from __future__ import annotations
from collections import deque
from itertools import product
import time
from typing import List
import numpy as np
from luxai2022.config import EnvConfig
from luxai2022.map.board import Board
from luxai2022.map.position import Position
from luxai2022.team import Team
from luxai2022.unit import UnitCargo
from luxai2022.actions import move_deltas
from luxai2022.globals import TERM_COLORS
try:
    from termcolor import colored
except:
    pass


def compute_water_info(init: np.ndarray, MIN_LICHEN_TO_SPREAD: int, lichen: np.ndarray, lichen_strains: np.ndarray, strain_id: int, forbidden: np.ndarray):
    # TODO - improve the performance here with cached solution
    frontier = deque(init)
    seen = set(map(tuple, init))
    grow_lichen_positions = set()
    H, W = lichen.shape
    ct = 0
    while len(frontier) > 0:
        ct += 1
        if ct > 1_000_000:
            print("Error! Lichen Growth calculation took too long")
            break
        pos = frontier.popleft()
        if pos[0] < 0 or pos[1] < 0 or pos[0] >= forbidden.shape[1] or pos[1] >= forbidden.shape[0]:
            continue

        if forbidden[pos[1], pos[0]]:
            continue
        pos_lichen = lichen[pos[1], pos[0]]
        # check for surrounding tiles with lichen and no incompatible lichen strains, grow on those
        can_grow = True
        for move_delta in move_deltas[1:]:
            check_pos = pos + move_delta
            if (check_pos[0], check_pos[1]) in seen: continue
            # check surrounding tiles on the map
            if check_pos[0] < 0 or check_pos[1] < 0 or check_pos[0] >= W or check_pos[1] >= H: continue
            adj_strain = lichen_strains[check_pos[1], check_pos[0]]
            if adj_strain == -1:
                if pos_lichen >= MIN_LICHEN_TO_SPREAD:
                    seen.add(tuple(check_pos))
                    frontier.append(check_pos)
            elif adj_strain != strain_id:
                    # adjacent tile is not empty and is not a strain this factory owns.
                    can_grow = False
                    seen.add(tuple(check_pos))
            else:
                # adjacent tile has our own strain, we can grow here too
                seen.add(tuple(check_pos))
                frontier.append(check_pos)

        if can_grow:
            grow_lichen_positions.add((pos[0], pos[1]))
    return grow_lichen_positions

class Factory:
    def __init__(self, team: Team, unit_id: str, num_id: int) -> None:
        self.team_id = team.team_id
        self.team = team
        self.unit_id = unit_id
        self.pos = Position(np.zeros(2, dtype=int))
        self.power = 0
        self.cargo = UnitCargo()
        self.num_id = num_id
        self.action_queue = []
        self.grow_lichen_positions = None

    @property
    def pos_slice(self):
        return slice(self.pos.y - 1, self.pos.y + 2), slice(self.pos.x - 1, self.pos.x + 2)

    def refine_step(self, config: EnvConfig):
        max_consumed_ice = min(self.cargo.ice, config.FACTORY_PROCESSING_RATE_WATER)
        max_consumed_ore = min(self.cargo.ore, config.FACTORY_PROCESSING_RATE_METAL)
        # permit refinement of blocks of resources, no floats.
        produced_water = max_consumed_ice // config.ICE_WATER_RATIO
        produced_metal = max_consumed_ore // config.ORE_METAL_RATIO
        self.cargo.ice -= produced_water * config.ICE_WATER_RATIO
        self.cargo.ore -= produced_metal * config.ORE_METAL_RATIO

        self.cargo.water += produced_water
        self.cargo.metal += produced_metal

    def cache_water_info(self, board: Board, env_cfg: EnvConfig):
        # TODO this can easily be a fairly slow function, can we make it much faster?
        # Caches information about which tiles lichen can grow on for this factory

        # perform a BFS from the factory position and look for non rubble, non factory tiles.
        # find the current frontier from 4 starting positions x marked below
        """
             x
           _ _ _
          |     |
        x |     | x
          |_ _ _|
             x

        """
        forbidden = (board.rubble > 0) | (board.factory_occupancy_map != -1)
        init_arr = np.stack([self.pos.pos + np.array([0, -2]), self.pos.pos + np.array([2, 0]), self.pos.pos + np.array([0, 2]), self.pos.pos + np.array([-2, 0])])
        self.grow_lichen_positions = compute_water_info(init_arr, env_cfg.MIN_LICHEN_TO_SPREAD, board.lichen, board.lichen_strains, self.num_id, forbidden)
    def water_cost(self, config: EnvConfig):
        return int(np.ceil(len(self.grow_lichen_positions) / config.LICHEN_WATERING_COST_FACTOR) + 1)

    ### Add and sub resource functions copied over from unit.py code, can we consolidate them somewhere?
    def add_resource(self, resource_id, transfer_amount):
        if transfer_amount < 0: transfer_amount = 0
        if resource_id == 0:
            self.cargo.ice += transfer_amount
        elif resource_id == 1:
            self.cargo.ore += transfer_amount
        elif resource_id == 2:
            self.cargo.water += transfer_amount
        elif resource_id == 3:
            self.cargo.metal += transfer_amount
        elif resource_id == 4:
            self.power += transfer_amount
        return int(transfer_amount)
    def sub_resource(self, resource_id, amount):
        # subtract/transfer out as much as you min(have, request)
        if amount < 0: amount = 0
        if resource_id == 0:
            transfer_amount = min(self.cargo.ice, amount)
            self.cargo.ice -= transfer_amount
        elif resource_id == 1:
            transfer_amount = min(self.cargo.ore, amount)
            self.cargo.ore -= transfer_amount
        elif resource_id == 2:
            transfer_amount = min(self.cargo.water, amount)
            self.cargo.water -= transfer_amount
        elif resource_id == 3:
            transfer_amount = min(self.cargo.metal, amount)
            self.cargo.metal -= transfer_amount
        elif resource_id == 4:
            transfer_amount = min(self.power, amount)
            self.power -= transfer_amount
        return int(transfer_amount)




    def state_dict(self):
        return dict(
            pos=self.pos.pos,
            power=self.power,
            cargo=self.cargo.state_dict(),
            unit_id=self.unit_id,
            strain_id=self.num_id, # number version of unit_id
            team_id=self.team_id

        )

    def __str__(self) -> str:
        out = f"[{self.team_id}] {self.unit_id} Factory at {self.pos}"
        if TERM_COLORS:
            return colored(out, self.team.faction.value.color)
        return out
