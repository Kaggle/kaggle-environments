from enum import Enum

MODERATOR_ID = "MODERATOR"


class StrEnum(str, Enum):
    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)


class Phase(StrEnum):
    DAY = "Day"
    NIGHT = "Night"
    GAME_OVER = "Game Over"


DAY, NIGHT, GAME_OVER = Phase


class PhaseDivider(StrEnum):
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


class Team(StrEnum):
    VILLAGERS = "Villagers"
    WEREWOLVES = "Werewolves"


class RoleConst(StrEnum):
    VILLAGER = "Villager"
    WEREWOLF = "Werewolf"
    DOCTOR = "Doctor"
    SEER = "Seer"


class ActionType(StrEnum):
    NO_OP = "NO_OP"
    NIGHT_KILL_VOTE = "NIGHT_KILL_VOTE"
    NIGHT_SAVE_TARGET = "NIGHT_SAVE_TARGET"
    NIGHT_INSPECT_TARGET = "NIGHT_INSPECT_TARGET"
    DAY_DISCUSS = "DAY_DISCUSS"
    DAY_LYNCH_VOTE = "DAY_LYNCH_VOTE"


class PerceivedThreatLevel(StrEnum):
    SAFE = "SAFE"
    UNEASY = "UNEASY"
    DANGER = "DANGER"


class EnvInfoKeys:
    MODERATOR_OBS = "MODERATOR_OBSERVATION"
    GAME_END = "GAME_END"


class ObsKeys:
    RAW_OBSERVATION = "raw_observation"


class DetailedPhase(StrEnum):
    def __new__(cls, value, category: Phase):
        # This creates the string object from the value
        obj = str.__new__(cls, value)
        # This sets the _value_ attribute, which is what Enum uses internally
        obj._value_ = value
        # Now, attach your custom category attribute
        obj.category = category
        return obj

    # Night Phases
    NIGHT_START = "NIGHT_START", NIGHT
    NIGHT_AWAIT_ACTIONS = "NIGHT_AWAIT_ACTIONS", NIGHT
    NIGHT_CONCLUDE = "NIGHT_CONCLUDE", NIGHT

    # Day Phases
    DAY_START = "DAY_START", DAY

    DAY_BIDDING_AWAIT = "DAY_BIDDING_AWAIT", DAY
    DAY_BIDDING_CONCLUDE = "DAY_BIDDING_CONCLUDE", DAY

    DAY_CHAT_AWAIT = "DAY_CHAT_AWAIT", DAY
    DAY_CHAT_CONCLUDE = "DAY_CHAT_CONCLUDE", DAY

    DAY_VOTING_START = "DAY_VOTING_START", DAY
    DAY_VOTING_AWAIT = "DAY_VOTING_AWAIT", DAY
    DAY_VOTING_CONCLUDE = "DAY_VOTING_CONCLUDE", DAY

    # Game Over
    GAME_OVER = "GAME_OVER", GAME_OVER


EVENT_HANDLER_FOR_ATTR_NAME = "_event_handler_for"


class EventName(str, Enum):
    GAME_START = "game_start"
    PHASE_CHANGE = "phase_change"
    PHASE_DIVIDER = "phase_divider"
    ELIMINATION = "elimination"

    VOTE_REQUEST = "vote_request"
    VOTE_ACTION = "vote_action"
    VOTE_RESULT = "vote_result"
    VOTE_ORDER = "vote_order"

    HEAL_REQUEST = "heal_request"
    HEAL_ACTION = "heal_action"
    HEAL_RESULT = "heal_result"

    INSPECT_REQUEST = "inspect_request"
    INSPECT_ACTION = "inspect_action"
    INSPECT_RESULT = "inspect_result"

    CHAT_REQUEST = "chat_request"
    DISCUSSION = "discussion"
    DISCUSSION_ORDER = "discussion_order"

    BID_REQEUST = "bid_request"
    BID_RESULT = "bid_result"
    BID_ACTION = "bid_action"
    BIDDING_INFO = "bidding_info"

    ELIMINATE_PROPOSAL_ACTION = "eliminate_proposal_action"
    NOOP_ACTION = "no_op_action"

    GAME_END = "game_end"
    MODERATOR_ANNOUNCEMENT = "moderator_announcement"
    ACTION_CONFIRMATION = "action_confirmation"
    ERROR = "error"
    NIGHT_START = "night_start"
    DAY_START = "day_start"
    NIGHT_END = "night_end"
    DAY_END = "day_end"


class RevealLevel(StrEnum):
    NO_REVEAL = "no_reveal"
    """No reveal during elimination."""

    TEAM = "team"
    """Only reveal team during elimination."""

    ROLE = "role"
    """Reveal detailed role information during elimination."""
