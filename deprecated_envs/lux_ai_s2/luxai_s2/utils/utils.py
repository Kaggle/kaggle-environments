from typing import List

from luxai_s2.config import EnvConfig
from luxai_s2.unit import Unit, UnitType


def is_day(config: EnvConfig, env_step):
    return env_step % config.CYCLE_LENGTH < config.DAY_LENGTH


def get_top_two_power_units(units: List[Unit], unit_type: UnitType):
    most_power_unit: Unit = units[0]
    most_power = -1
    next_most_power_unit: Unit = units[1]
    next_most_power = -1
    for u in units:
        if u.unit_type == unit_type:
            if u.power > most_power:
                next_most_power_unit = most_power_unit
                most_power_unit = u
                most_power = u.power
            elif (
                u.power >= next_most_power
            ):  # >= check since we want to top 2 power units which can tie
                next_most_power_unit = u
                next_most_power = u.power

    return (most_power_unit, next_most_power_unit)


def my_turn_to_place_factory(place_first: bool, step: int):
    if place_first:
        if step % 2 == 1:
            return True
    else:
        if step % 2 == 0:
            return True
    return False
