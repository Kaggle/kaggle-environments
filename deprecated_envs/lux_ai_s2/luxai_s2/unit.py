import math
from dataclasses import dataclass
from enum import Enum
from typing import List

try:
    from typing import TypedDict    
except:
    from typing_extensions import TypedDict

import numpy as np
import numpy.typing as npt

try:
    from termcolor import colored
except:
    pass
from luxai_s2.config import EnvConfig, UnitConfig
from luxai_s2.globals import TERM_COLORS
from luxai_s2.map.position import Position
from luxai_s2.team import FactionTypes, Team


class UnitType(Enum):
    LIGHT = "Light"
    HEAVY = "Heavy"


class UnitCargoStateDict(TypedDict):
    ice: int
    ore: int
    water: int
    metal: int


ActionType = npt.NDArray[np.int_]


class FactoryPlacementActionType(TypedDict):
    metal: int
    water: int
    spawn: npt.NDArray[np.int_]


class BidActionType(TypedDict):
    faction: str
    bid: int


class UnitStateDict(TypedDict):
    team_id: int
    unit_id: str
    power: int
    unit_type: str
    pos: npt.NDArray[np.int_]
    cargo: UnitCargoStateDict
    action_queue: List[ActionType]


@dataclass
class UnitCargo:
    ice: int = 0
    ore: int = 0
    water: int = 0
    metal: int = 0

    def state_dict(self) -> UnitCargoStateDict:
        return dict(ice=self.ice, ore=self.ore, water=self.water, metal=self.metal)


class Unit:
    def __init__(
        self, team: Team, unit_type: UnitType, unit_id: str, env_cfg: EnvConfig
    ) -> None:
        self.unit_type = unit_type
        self.team_id = team.team_id
        self.team = team
        self.unit_id = unit_id
        self.pos = Position(np.zeros(2, dtype=int))

        self.cargo = UnitCargo()
        # TODO - replace with a deque perhaps?
        self.action_queue: List = []
        self.unit_cfg: UnitConfig = env_cfg.ROBOTS[unit_type.name]
        self.power = env_cfg.ROBOTS[unit_type.name].INIT_POWER
        self.cargo_space = env_cfg.ROBOTS[unit_type.name].CARGO_SPACE
        self.battery_capacity = env_cfg.ROBOTS[unit_type.name].BATTERY_CAPACITY

    def __str__(self) -> str:
        out = f"[{self.team_id}] {self.unit_id} {self.unit_type} at {self.pos}"
        if TERM_COLORS:
            return colored(out, self.team.faction.value.color)
        return out

    def is_heavy(self) -> bool:
        return self.unit_type == UnitType.HEAVY

    def next_action(self):
        """
        get next action
        """
        if len(self.action_queue) == 0:
            return None
        action = self.action_queue[0]
        return action

    def repeat_action(self, action):
        action.n -= 1
        if action.n <= 0:
            # remove from front of queue
            self.action_queue.pop(0)
            # endless repeat puts action back at end of queue
            if action.repeat > 0:
                action.n = action.repeat
                self.action_queue.append(action)

    def move_power_cost(self, rubble_at_target: int):
        return math.floor(
            self.unit_cfg.MOVE_COST
            + self.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target
        )

    def state_dict(self) -> UnitStateDict:
        return dict(
            team_id=self.team_id,
            unit_id=self.unit_id,
            power=self.power,
            unit_type=self.unit_type.name,
            pos=self.pos.pos,
            cargo=self.cargo.state_dict(),
            action_queue=[a.state_dict() for a in self.action_queue],
        )

    def add_resource(self, resource_id, amount):
        if amount < 0:
            amount = 0
        if resource_id == 0:
            transfer_amount = min(self.cargo_space - self.cargo.ice, amount)
            self.cargo.ice += transfer_amount
        elif resource_id == 1:
            transfer_amount = min(self.cargo_space - self.cargo.ore, amount)
            self.cargo.ore += transfer_amount
        elif resource_id == 2:
            transfer_amount = min(self.cargo_space - self.cargo.water, amount)
            self.cargo.water += transfer_amount
        elif resource_id == 3:
            transfer_amount = min(self.cargo_space - self.cargo.metal, amount)
            self.cargo.metal += transfer_amount
        elif resource_id == 4:
            transfer_amount = min(self.battery_capacity - self.power, amount)
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


if __name__ == "__main__":
    u = Unit(team=Team(0, FactionTypes.AlphaStrike), unit_type=UnitType.HEAVY)
    print(u)
    u = Unit(team=Team(1, FactionTypes.MotherMars), unit_type=UnitType.HEAVY)
    u.pos += np.array([1, 4])
    print(u)
    # u = Unit(team=Team(0, FactionTypes.AlphaStrike), unit_type=UnitType.HEAVY)
    # print(u)
