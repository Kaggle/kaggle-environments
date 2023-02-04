from __future__ import annotations

import time
from collections import deque
from itertools import product
from typing import List
try:
    from typing import TypedDict    
except:
    from typing_extensions import TypedDict

import numpy as np
import numpy.typing as npt

from luxai_s2.actions import move_deltas
from luxai_s2.config import EnvConfig
from luxai_s2.globals import TERM_COLORS
from luxai_s2.map.board import Board
from luxai_s2.map.position import Position
from luxai_s2.team import Team
from luxai_s2.unit import UnitCargo, UnitCargoStateDict

try:
    from termcolor import colored
except:
    pass


def compute_water_info(
    init: np.ndarray,
    MIN_LICHEN_TO_SPREAD: int,
    lichen: np.ndarray,
    lichen_strains: np.ndarray,
    factory_occupancy_map: np.ndarray,
    strain_id: int,
    forbidden: np.ndarray,
):
    # TODO - improve the performance here with cached solution
    frontier = deque(init)
    seen = set(map(tuple, init))
    grow_lichen_positions = set()
    connected_lichen_positions = set()
    H, W = lichen.shape
    ct = 0
    while len(frontier) > 0:
        ct += 1
        if ct > 1_000_000:
            print("Error! Lichen Growth calculation took too long")
            break
        pos = frontier.popleft()
        if (
            pos[0] < 0
            or pos[1] < 0
            or pos[0] >= forbidden.shape[0]
            or pos[1] >= forbidden.shape[1]
        ):
            continue

        if forbidden[pos[0], pos[1]]:
            continue
        pos_lichen = lichen[pos[0], pos[1]]
        pos_strain = lichen_strains[pos[0], pos[1]]
        # check for surrounding tiles with lichen and no incompatible lichen strains, grow on those
        can_grow = True
        for move_delta in move_deltas[1:]:
            check_pos = pos + move_delta
            # check surrounding tiles on the map
            if (
                check_pos[0] < 0
                or check_pos[1] < 0
                or check_pos[0] >= H
                or check_pos[1] >= W
            ):
                continue

            # If any neighbor 1. has a different strain, or 2. is a different factory,
            # then the current pos cannot grow
            adj_strain = lichen_strains[check_pos[0], check_pos[1]]
            adj_factory = factory_occupancy_map[check_pos[0], check_pos[1]]
            if (adj_strain != -1 and adj_strain != strain_id) or (
                adj_factory != -1 and adj_factory != strain_id
            ):
                can_grow = False

            # if seen, skip
            if (check_pos[0], check_pos[1]) in seen:
                continue

            # we add it to the frontier only in two cases:
            #  1. it is an empty tile, and current pos has enough lichen to expand.
            #  2. both current tile and check_pos are of our strain.
            if (adj_strain == -1 and pos_lichen >= MIN_LICHEN_TO_SPREAD) or (
                adj_strain == strain_id and pos_strain == strain_id
            ):
                seen.add(tuple(check_pos))
                frontier.append(check_pos)

        if can_grow or (lichen_strains[pos[0], pos[1]] == strain_id):
            grow_lichen_positions.add((pos[0], pos[1]))
            if lichen_strains[pos[0], pos[1]] == strain_id:
                connected_lichen_positions.add((pos[0], pos[1]))
    return grow_lichen_positions, connected_lichen_positions


class FactoryStateDict(TypedDict):
    pos: npt.NDArray[np.int_]
    power: int
    cargo: UnitCargoStateDict
    unit_id: str
    strain_id: int
    team_id: int


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
        self.grow_lichen_positions = set()
        self.connected_lichen_positions = set()

    @property
    def pos_slice(self):
        return slice(self.pos.x - 1, self.pos.x + 2), slice(
            self.pos.y - 1, self.pos.y + 2
        )

    @property
    def min_dist_slice(self):
        deltas = np.array(
            [
                (4, 0),
                (5, 1),
                (0, -4),
                (0, 5),
                (2, 2),
                (-4, 1),
                (-1, -1),
                (-1, -2),
                (-2, -1),
                (-2, -2),
                (4, 2),
                (0, -2),
                (0, -1),
                (2, 4),
                (-1, 0),
                (-2, 0),
                (-3, 1),
                (0, 0),
                (2, -3),
                (-2, 2),
                (-1, 2),
                (3, 1),
                (5, -1),
                (-3, 3),
                (0, 2),
                (1, 3),
                (-4, -1),
                (-1, -5),
                (-4, -2),
                (-1, 4),
                (-2, 4),
                (3, 3),
                (5, 0),
                (-6, 0),
                (0, -5),
                (1, -4),
                (1, 5),
                (-4, 0),
                (-1, -3),
                (-2, -3),
                (-5, 1),
                (-3, -1),
                (0, -3),
                (-3, -2),
                (1, -1),
                (1, -2),
                (-4, 2),
                (3, -1),
                (3, -2),
                (-3, 0),
                (1, 0),
                (3, 0),
                (-3, 2),
                (1, 2),
                (0, 4),
                (2, 1),
                (3, 2),
                (4, 1),
                (-5, -1),
                (1, -5),
                (1, 4),
                (0, 6),
                (2, 3),
                (-1, -4),
                (-5, 0),
                (-3, -3),
                (1, -3),
                (2, -4),
                (-1, 1),
                (-2, 1),
                (3, -3),
                (0, 1),
                (2, -1),
                (2, -2),
                (-1, 3),
                (-2, 3),
                (4, -1),
                (4, -2),
                (0, -6),
                (1, 1),
                (0, 3),
                (2, 0),
                (6, 0),
                (-2, -4),
                (-1, 5),
            ]
        )
        return self.pos.pos + deltas

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
        # find the current frontier from 12 starting positions x marked below
        """
           x x x
           _ _ _
        x |     | x
        x |     | x
        x |_ _ _| x
           x x x

        """
        forbidden = (
            (board.rubble > 0)
            | (board.factory_occupancy_map != -1)
            | (board.ice > 0)
            | (board.ore > 0)
        )
        deltas = [
            np.array([0, -2]),
            np.array([-1, -2]),
            np.array([1, -2]),
            np.array([0, 2]),
            np.array([-1, 2]),
            np.array([1, 2]),
            np.array([2, 0]),
            np.array([2, -1]),
            np.array([2, 1]),
            np.array([-2, 0]),
            np.array([-2, -1]),
            np.array([-2, 1]),
        ]
        init_arr = np.stack(deltas) + self.pos.pos
        self.grow_lichen_positions, self.connected_lichen_positions = compute_water_info(
            init_arr,
            env_cfg.MIN_LICHEN_TO_SPREAD,
            board.lichen,
            board.lichen_strains,
            board.factory_occupancy_map,
            self.num_id,
            forbidden,
        )

    def water_cost(self, config: EnvConfig):
        return int(
            np.ceil(
                len(self.grow_lichen_positions) / config.LICHEN_WATERING_COST_FACTOR
            )
        )

    ### Add and sub resource functions copied over from unit.py code, can we consolidate them somewhere?
    def add_resource(self, resource_id, transfer_amount):
        if transfer_amount < 0:
            transfer_amount = 0
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
        if amount < 0:
            amount = 0
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

    def state_dict(self) -> FactoryStateDict:
        return dict(
            pos=self.pos.pos,
            power=self.power,
            cargo=self.cargo.state_dict(),
            unit_id=self.unit_id,
            strain_id=self.num_id,  # number version of unit_id
            team_id=self.team_id,
        )

    def __str__(self) -> str:
        out = f"[{self.team_id}] {self.unit_id} Factory at {self.pos}"
        if TERM_COLORS:
            return colored(out, self.team.faction.value.color)
        return out
