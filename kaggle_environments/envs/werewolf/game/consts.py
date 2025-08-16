from enum import Enum


MODERATOR_ID = "MODERATOR"


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


class ActionType(str, Enum):
    NO_OP = "NO_OP"
    NIGHT_KILL_VOTE = "NIGHT_KILL_VOTE"
    NIGHT_SAVE_TARGET = "NIGHT_SAVE_TARGET"
    NIGHT_INSPECT_TARGET = "NIGHT_INSPECT_TARGET"
    DAY_DISCUSS = "DAY_DISCUSS"
    DAY_LYNCH_VOTE = "DAY_LYNCH_VOTE"


class PerceivedThreatLevel(str, Enum):
    SAFE = "SAFE"
    UNEASY = "UNEASY"
    DANGER = "DANGER"
