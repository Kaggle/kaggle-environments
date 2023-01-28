import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List

try:
    from typing import TypedDict    
except:
    from typing_extensions import TypedDict


class RobotStatsStateDict(TypedDict):
    LIGHT: int
    HEAVY: int


def create_robot_stats() -> RobotStatsStateDict:
    return dict(LIGHT=0, HEAVY=0)


class AllStatsStateDict(TypedDict):
    LIGHT: int
    HEAVY: int
    FACTORY: int


def create_all_stats() -> AllStatsStateDict:
    return dict(LIGHT=0, HEAVY=0, FACTORY=0)


class GenerationStatsStateDict(TypedDict):
    power: AllStatsStateDict
    water: int
    metal: int
    ore: RobotStatsStateDict  # amount dug out by HEAVY or LIGHT
    ice: RobotStatsStateDict  # amount dug out by HEAVY or LIGHT
    lichen: int  # amount grown
    built: RobotStatsStateDict  # amount built


def create_generation_stats() -> GenerationStatsStateDict:
    return dict(
        power=create_all_stats(),
        water=0,
        metal=0,
        ore=create_robot_stats(),
        ice=create_robot_stats(),
        lichen=0,
        built=create_robot_stats(),
    )


class ConsumptionStatsStateDict(TypedDict):
    power: AllStatsStateDict
    water: int
    metal: int


def create_consumption_stats() -> ConsumptionStatsStateDict:
    return dict(
        power=create_all_stats(),
        water=0,
        metal=0,
        ore=create_robot_stats(),
        ice=create_robot_stats(),
    )


class TransferStatsStateDict(TypedDict):
    power: int
    water: int
    metal: int
    ice: int
    ore: int


def create_transfer_pickup_stats():
    return dict(power=0, water=0, metal=0, ice=0, ore=0)


class PickUpStatsStateDict(TypedDict):
    power: int
    water: int
    metal: int
    ice: int
    ore: int


class DestroyedStatsStateDict(TypedDict):
    FACTORY: int
    HEAVY: int
    LIGHT: int
    rubble: RobotStatsStateDict
    lichen: RobotStatsStateDict


def create_destroyed_stats():
    return dict(
        FACTORY=0,
        HEAVY=0,
        LIGHT=0,
        rubble=create_robot_stats(),
        lichen=create_robot_stats(),
    )


class StatsStateDict(TypedDict):
    consumption: ConsumptionStatsStateDict
    generation: GenerationStatsStateDict
    action_queue_updates_success: int
    action_queue_updates_total: int
    destroyed: DestroyedStatsStateDict
    transfer: TransferStatsStateDict
    pickup: PickUpStatsStateDict


## TODO add collision stats
## TODO add above as a wrapper for jax env?


def create_empty_stats() -> StatsStateDict:
    stats: StatsStateDict = dict()
    stats["action_queue_updates_total"] = 0
    stats["action_queue_updates_success"] = 0
    stats["consumption"] = create_consumption_stats()
    stats["destroyed"] = create_destroyed_stats()
    stats["generation"] = create_generation_stats()
    stats["pickup"] = create_transfer_pickup_stats()
    stats["transfer"] = create_transfer_pickup_stats()
    return stats
