from enum import Enum


MODERATOR_ID = "MODERATOR"


class StrEnum(Enum):
    def __str__(self):
        return str(self.value)


class Phase(str, StrEnum):
    DAY = "Day"
    NIGHT = "Night"


class PhaseDivider(str, StrEnum):
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


class Team(str, StrEnum):
    VILLAGERS = "Villagers"
    WEREWOLVES = "Werewolves"


class RoleConst(str, StrEnum):
    VILLAGER = "Villager"
    WEREWOLF = "Werewolf"
    DOCTOR = "Doctor"
    SEER = "Seer"


class ActionType(str, StrEnum):
    NO_OP = "NO_OP"
    NIGHT_KILL_VOTE = "NIGHT_KILL_VOTE"
    NIGHT_SAVE_TARGET = "NIGHT_SAVE_TARGET"
    NIGHT_INSPECT_TARGET = "NIGHT_INSPECT_TARGET"
    DAY_DISCUSS = "DAY_DISCUSS"
    DAY_LYNCH_VOTE = "DAY_LYNCH_VOTE"


class PerceivedThreatLevel(str, StrEnum):
    SAFE = "SAFE"
    UNEASY = "UNEASY"
    DANGER = "DANGER"


class EnvInfoKeys:
    MODERATOR_OBS = "MODERATOR_OBSERVATION"
    GAME_END = "GAME_END"


class ObsKeys:
    RAW_OBSERVATION = "raw_observation"


class DetailedPhase(StrEnum):
    # Night Phases
    NIGHT_START = "NIGHT_START"
    NIGHT_AWAIT_ACTIONS = "NIGHT_AWAIT_ACTIONS"
    # Day Phases
    DAY_START = "DAY_START"
    DAY_BIDDING_AWAIT = "DAY_BIDDING_AWAIT"
    DAY_CHAT_AWAIT = "DAY_CHAT_AWAIT"
    DAY_VOTING_AWAIT = "DAY_VOTING_AWAIT"
    GAME_OVER = "GAME_OVER"
