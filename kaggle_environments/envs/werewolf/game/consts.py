from enum import Enum


class Phase(str, Enum):
    DAY = "Day"
    NIGHT = "Night"


class Team(str, Enum):
    VILLAGERS = "Villagers"
    WEREWOLVES = "Werewolves"


class RoleConst(str, Enum):
    VILLAGER = "Villager"
    WEREWOLF = "Werewolf"
    DOCTOR = "Doctor"
    SEER = "Seer"
