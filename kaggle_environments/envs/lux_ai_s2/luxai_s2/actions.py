import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

import numpy as np

from luxai_s2.config import EnvConfig

if TYPE_CHECKING:
    from luxai_s2.factory import Factory
    from luxai_s2.state import State

import luxai_s2.unit as luxai_unit
from luxai_s2.map.position import Position


# (0 = move, 1 = transfer X amount of R, 2 = pickup X amount of R, 3 = dig, 4 = self destruct, 5 = recharge X)
class Action:
    def __init__(self, act_type: str) -> None:
        self.act_type = act_type
        self.n = 1  # number of times to execute the action
        self.power_cost = 0
        self.repeat = 0  # if 0, no recycling. If > 0, we recycle action and set n equal to this.

    def state_dict(self):
        raise NotImplementedError("")


class FactoryBuildAction(Action):
    def __init__(self, unit_type: luxai_unit.UnitType) -> None:
        super().__init__("factory_build")
        self.unit_type = unit_type
        self.power_cost = 0

    def state_dict(self):
        if self.unit_type == luxai_unit.UnitType.LIGHT:
            return 0
        return 1

    def __str__(self) -> str:
        return f"{self.act_type} {self.unit_type.name}"


class FactoryWaterAction(Action):
    def __init__(self) -> None:
        super().__init__("factory_water")
        self.water_cost = None
        self.power_cost = 0

    def state_dict(self):
        return 2

    def __str__(self) -> str:
        return f"{self.act_type}: (cost: {self.water_cost}"


direction_to_name = ["center", "up", "right", "down", "left"]
resource_to_name = ["ice", "ore", "water", "metal", "power"]


class MoveAction(Action):
    def __init__(
        self, move_dir: int, dist: int = 1, repeat: int = 0, n: int = 1
    ) -> None:
        super().__init__("move")
        # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
        self.move_dir = move_dir
        self.dist = dist
        self.repeat = repeat
        self.n = n
        self.power_cost = 0

    def state_dict(self):
        return np.array([0, self.move_dir, 0, 0, self.repeat, self.n])

    def __str__(self) -> str:
        return f"{self.act_type} {direction_to_name[self.move_dir]} (n: {self.n}, r: {self.repeat})"


class TransferAction(Action):
    def __init__(
        self,
        transfer_dir: int,
        resource: int,
        transfer_amount: int,
        repeat: int = 0,
        n: int = 1,
    ) -> None:
        super().__init__("transfer")
        # a[2] = R = resource type (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 power)
        self.transfer_dir = transfer_dir
        self.resource = resource
        self.transfer_amount = transfer_amount
        self.repeat = repeat
        self.n = n
        self.power_cost = 0

    def state_dict(self):
        return np.array(
            [
                1,
                self.transfer_dir,
                self.resource,
                self.transfer_amount,
                self.repeat,
                self.n
            ]
        )

    def __str__(self) -> str:
        return f"{self.act_type} {resource_to_name[self.resource]} {direction_to_name[self.transfer_dir]} (n: {self.n}, r: {self.repeat})"


class PickupAction(Action):
    def __init__(
        self, resource: int, pickup_amount: int, repeat: int = 0, n: int = 1
    ) -> None:
        super().__init__("pickup")
        # a[2] = R = resource type (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 power)
        self.resource = resource
        self.pickup_amount = pickup_amount
        self.repeat = repeat
        self.n = n
        self.power_cost = 0

    def state_dict(self):
        return np.array([2, 0, self.resource, self.pickup_amount, self.repeat, self.n])

    def __str__(self) -> str:
        return f"{self.act_type} {self.pickup_amount} {resource_to_name[self.resource]} (n: {self.n}, r: {self.repeat})"


class DigAction(Action):
    def __init__(self, repeat: int = 0, n: int = 1) -> None:
        super().__init__("dig")
        self.repeat = repeat
        self.n = n
        self.power_cost = 0

    def state_dict(self):
        return np.array([3, 0, 0, 0, self.repeat, self.n])

    def __str__(self) -> str:
        return f"{self.act_type} (n: {self.n}, r: {self.repeat})"


class SelfDestructAction(Action):
    def __init__(self, repeat: int = 0, n: int = 1) -> None:
        super().__init__("self_destruct")
        self.repeat = repeat
        self.n = n
        self.power_cost = 0

    def state_dict(self):
        return np.array([4, 0, 0, 0, self.repeat, self.n])

    def __str__(self) -> str:
        return f"{self.act_type} (n: {self.n}, r: {self.repeat})"


class RechargeAction(Action):
    def __init__(self, power: int, repeat: int = 0, n: int = 1) -> None:
        super().__init__("recharge")
        self.power = power
        self.repeat = repeat
        self.n = n
        self.power_cost = 0

    def state_dict(self):
        return np.array([5, 0, 0, self.power, self.repeat, self.n])

    def __str__(self) -> str:
        return f"{self.act_type} {self.power} (n: {self.n}, r: {self.repeat})"


def format_factory_action(a: int):
    if a == 0 or a == 1:
        unit_type = luxai_unit.UnitType.HEAVY
        if a == 0:
            unit_type = luxai_unit.UnitType.LIGHT
        return FactoryBuildAction(unit_type=unit_type)
    elif a == 2:
        return FactoryWaterAction()
    else:
        raise ValueError(f"Action {a} for factory is invalid")


def format_action_vec(a: np.ndarray):
    # (0 = move, 1 = transfer X amount of R, 2 = pickup X amount of R, 3 = dig, 4 = self destruct, 5 = recharge X)
    a = np.array(a).astype(int)
    a_type = a[0]
    if a_type == 0:
        act = MoveAction(a[1], dist=1, repeat=a[4], n=a[5])
    elif a_type == 1:
        act = TransferAction(a[1], a[2], a[3], repeat=a[4], n=a[5])
    elif a_type == 2:
        act =  PickupAction(a[2], a[3], repeat=a[4], n=a[5])
    elif a_type == 3:
        act = DigAction(repeat=a[4], n=a[5])
    elif a_type == 4:
        act = SelfDestructAction(repeat=a[4], n=a[5])
    elif a_type == 5:
        act =  RechargeAction(a[3], repeat=a[4], n=a[5])
    else:
        raise ValueError(f"Action {a} is invalid type, {a[0]} is not valid")
    return act
    


# a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])


def validate_actions(env_cfg: EnvConfig, state: "State", actions_by_type, verbose=1):
    """
    validates actions and logs warnings for any invalid actions. Invalid actions are subsequently not evaluated and
    won't count towards the `n` value of actions that execute multiple times.
    """
    actions_by_type_validated = defaultdict(list)
    valid_action = True

    def invalidate_action(msg):
        nonlocal valid_action
        valid_action = False
        if verbose > 1:
            print(f"{state.real_env_steps}: {msg}")

    for unit, transfer_action in actions_by_type["transfer"]:
        valid_action = True
        unit: luxai_unit.Unit
        transfer_action: TransferAction
        transfer_pos: Position = unit.pos + move_deltas[transfer_action.transfer_dir]
        if transfer_action.resource > 4 or transfer_action.resource < 0:
            invalidate_action(
                f"Invalid Transfer Action for unit {unit}, transferring invalid resource id {transfer_action.resource}"
            )
            continue
        if (
            transfer_pos.x < 0
            or transfer_pos.y < 0
            or transfer_pos.x >= state.board.width
            or transfer_pos.y >= state.board.height
        ):
            invalidate_action(
                f"Invalid Transfer action for unit {unit} - Tried to transfer to {transfer_pos} which is off the map"
            )
            continue
        # if transfer_action.transfer_amount < 0: do not need to check as action space permits range of [0, max_transfer_amount] anyway
        resource_id = transfer_action.resource
        amount = transfer_action.transfer_amount

        if valid_action:
            actions_by_type_validated["transfer"].append((unit, transfer_action))

    for unit, dig_action in actions_by_type["dig"]:
        valid_action = True
        dig_action: DigAction
        dig_cost = unit.unit_cfg.DIG_COST
        if dig_cost > unit.power:
            invalidate_action(
                f"Invalid Dig Action for unit {unit} - Tried to dig requiring {dig_cost} power"
            )
            continue
        # verify not digging over a factory which is not allowed
        if state.board.factory_occupancy_map[unit.pos.x, unit.pos.y] != -1:
            invalidate_action(
                f"Invalid Dig Action for unit {unit} - Tried to dig on top of a factory"
            )
            continue
        if valid_action:
            actions_by_type_validated["dig"].append((unit, dig_action))

    for unit, pickup_action in actions_by_type["pickup"]:
        valid_action = True
        pickup_action: PickupAction
        unit: luxai_unit.Unit
        factory = state.board.get_factory_at(state, unit.pos)
        if factory is None:
            invalidate_action(f"No factory to pickup from for unit {unit}")
            continue
        if valid_action:
            actions_by_type_validated["pickup"].append((unit, pickup_action))

    for unit, move_action in actions_by_type["move"]:
        valid_action = True
        move_action: MoveAction
        target_pos: Position = (
            unit.pos + move_action.dist * move_deltas[move_action.move_dir]
        )
        if (
            target_pos.x < 0
            or target_pos.y < 0
            or target_pos.x >= state.board.width
            or target_pos.y >= state.board.height
        ):
            invalidate_action(
                f"Invalid movement action for unit {unit} - Tried to move to {target_pos} which is off the map"
            )
            continue
        if state.board.factory_occupancy_map[target_pos.x, target_pos.y] != -1:
            factory_id = state.board.factory_occupancy_map[target_pos.x, target_pos.y]
            if f"factory_{factory_id}" not in state.factories[unit.team.agent]:
                # if there is a factory but not same team
                invalidate_action(
                    f"Invalid movement action for unit {unit} - Tried to move to {target_pos} which is on an opponent factory"
                )
                continue
        rubble = state.board.rubble[target_pos.x, target_pos.y]
        power_required = (
            0 if move_action.move_dir == 0 else unit.move_power_cost(rubble)
        )

        if power_required > unit.power:
            invalidate_action(
                f"Invalid movement action for unit {unit} - Tried to move to {target_pos} requiring {power_required} power"
            )
            continue
        if valid_action:
            move_action.power_cost = power_required
            actions_by_type_validated["move"].append((unit, move_action))

    for unit, self_destruct_action in actions_by_type["self_destruct"]:
        valid_action = True
        self_destruct_action: SelfDestructAction
        power_required = unit.unit_cfg.SELF_DESTRUCT_COST
        if power_required > unit.power:
            invalidate_action(
                f"Invalid self destruct action for unit {unit} - Tried to self destruct requiring {power_required} power"
            )
            continue
        if valid_action:
            self_destruct_action.power_cost = power_required
            actions_by_type_validated["self_destruct"].append(
                (unit, self_destruct_action)
            )

    for unit, recharge_action in actions_by_type["recharge"]:
        actions_by_type_validated["recharge"].append((unit, recharge_action))

    for factory, build_action in actions_by_type["factory_build"]:
        valid_action = True
        build_action: FactoryBuildAction
        factory: "Factory"

        unit_cfg = env_cfg.ROBOTS[build_action.unit_type.name]
        if factory.cargo.metal < unit_cfg.METAL_COST:
            invalidate_action(
                f"Invalid factory build action for factory {factory} - Insufficient metal, factory has {factory.cargo.metal}, but requires {unit_cfg.METAL_COST} to build {build_action.unit_type}"
            )
            continue
        power_required = unit_cfg.POWER_COST
        if factory.power < power_required:
            invalidate_action(
                f"Invalid factory build action for factory {factory} - Insufficient power, factory has {factory.power}, but requires {power_required} power"
            )
            continue
        if valid_action:
            build_action.power_cost = power_required
            actions_by_type_validated["factory_build"].append((factory, build_action))
        pass

    for factory, water_action in actions_by_type["factory_water"]:
        valid_action = True
        water_cost = factory.water_cost(env_cfg)
        if water_cost > factory.cargo.water:
            invalidate_action(
                f"Invalid factory water action for factory {factory} - Insufficient water, factory has {factory.cargo.water}, but requires {water_cost} to water lichen"
            )
            continue
        if valid_action:
            actions_by_type_validated["factory_water"].append((factory, water_action))

    return actions_by_type_validated
