from enum import Enum


MODERATOR_ID = "MODERATOR"


class Phase(str, Enum):
    DAY = "Day"
    NIGHT = "Night"


class PhaseDivider(str, Enum):
    NIGHT_START = "NIGHT START"
    NIGHT_END = "NIGHT END"
    DAY_START = "DAY START"
    DAY_END = "DAY END"
    NIGHT_VOTE_START = "NIGHT VOTE START"
    NIGHT_VOTE_END = "NIGHT VOTE END"
    DAY_CHAT_START = "DAY CHAT START"
    DAY_CHAT_END = "DAY CHAT END"
    DAY_VOTE_START = "DAY VOTE START"
    DAY_VOTE_END = "DAY VOTE END"


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
