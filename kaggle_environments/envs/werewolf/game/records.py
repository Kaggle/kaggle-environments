import json
from abc import ABC
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Self
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_serializer, ConfigDict

from kaggle_environments.envs.werewolf.game.consts import Phase
from kaggle_environments.envs.werewolf.game.actions import Action


def get_utc_now():
    return str(datetime.now(ZoneInfo("UTC")))


class HistoryEntryType(str, Enum):
    GAME_START = "game_start"
    PHASE_CHANGE = "phase_change"
    PHASE_DIVIDER = "phase_divider"
    ACTION_RESULT = "action_result"
    ELIMINATION = "elimination"
    VOTE_ACTION = "vote_action"
    HEAL_ACTION = "heal_action"
    VOTE_RESULT = "vote_result"
    VOTE_ORDER = "vote_order"
    DISCUSSION = "discussion"
    DISCUSSION_ORDER = "discussion_order"
    BIDDING_INFO = "bidding_info"
    GAME_END = "game_end"
    MODERATOR_ANNOUNCEMENT = "moderator_announcement"
    PROMPT_FOR_ACTION = "prompt_for_action"
    ERROR = "error"
    NIGHT_START = "night_start"
    DAY_START = "day_start"
    NIGHT_END = "night_end"
    DAY_END = "day_end"


class DataEntry(BaseModel, ABC):
    """Abstract base class for all data entry types."""
    def public_view(self) -> Self:
        """
        Returns a public-facing dictionary representation of the data.
        By default, this includes all fields. Subclasses with private
        data should override this method to exclude sensitive information.
        """
        return self.model_copy(deep=True)


class ActionDataMixin(BaseModel):
    """
    A mixin for action-related DataEntry models.
    Includes the actor performing the action and their private reasoning.
    """
    actor_id: str
    reasoning: Optional[str] = Field(default=None, description="Private reasoning for moderator analysis.")
    perceived_threat_level: Optional[str] = 'SAFE'
    action: Optional[Action] = None

    def public_view(self) -> Self:
        """
        Returns a public view of the action, excluding private reasoning.
        This overrides the default behavior from the DataEntry base class.
        """
        return self.model_copy(update={'reasoning': None, 'perceived_threat_level': None, 'action': None}, deep=True)


class HistoryEntry(BaseModel):
    day: int  # Day number, 0 for initial night
    phase: Phase
    entry_type: HistoryEntryType
    description: str
    public: bool = False
    visible_to: List[str] = Field(default_factory=list)
    data: Optional[dict | DataEntry] = None
    source: str
    created_at: str = Field(default_factory=get_utc_now)

    @field_serializer('data')
    def serialize_data(self, data):
        if data is None: return None
        if isinstance(data, dict):
            return data
        if isinstance(data, BaseModel):
            return data.model_dump()
        return None


# --- Game State and Setup Data Entries ---
class GameStartDataEntry(DataEntry):
    player_ids: List[str]
    number_of_players: int
    role_counts: Dict[str, int]
    team_member_counts: Dict[str, int]
    day_discussion_protocol_name: str
    day_discussion_protocol_rule: str
    night_werewolf_discussion_protocol_name: str
    night_werewolf_discussion_protocol_rule: str
    day_voting_protocol_name: str
    day_voting_protocol_rule: str


class GameStartRoleDataEntry(DataEntry):
    player_id: str
    team: str
    role: str
    rule_of_role: str


class SetNewPhaseDataEntry(DataEntry):
    new_detailed_phase: str


class PhaseDividerDataEntry(DataEntry):
    divider_type: str


# --- Request for Action Data Entries ---
class RequestForActionDataEntry(DataEntry):
    action_json_schema: str


class RequestDoctorSaveDataEntry(RequestForActionDataEntry):
    valid_candidates: List[str]


class RequestSeerRevealDataEntry(RequestForActionDataEntry):
    valid_candidates: List[str]


class RequestWerewolfVotingDataEntry(RequestForActionDataEntry):
    valid_targets: List[str]
    alive_werewolve_player_ids: List[str]
    voting_protocol_name: str
    voting_protocol_rule: str


class RequestVillagerToSpeakDataEntry(RequestForActionDataEntry):
    pass


# --- Action and Result Data Entries ---
class SeerInspectResultDataEntry(DataEntry):
    actor_id: str
    target_id: str
    role: str
    team: str


class TargetedActionDataEntry(DataEntry, ActionDataMixin):
    target_id: str


class SeerInspectActionDataEntry(TargetedActionDataEntry):
    """This records the Seer's choice of target to inspect."""


class DoctorHealActionDataEntry(TargetedActionDataEntry):
    """This records the Doctor's choice of target to heal."""


class WerewolfNightVoteDataEntry(TargetedActionDataEntry):
    """Records a werewolf's vote, including private reasoning."""


class DayExileVoteDataEntry(TargetedActionDataEntry):
    """Records a player's vote to exile, including private reasoning."""


class DoctorSaveDataEntry(DataEntry):
    """This records that a player was successfully saved by a doctor."""
    saved_player_id: str


class VoteOrderDataEntry(DataEntry):
    vote_order_of_player_ids: List[str]


class WerewolfNightEliminationElectedDataEntry(DataEntry):
    """This record the elected elimination target by werewolves."""
    elected_target_player_id: str


class WerewolfNightEliminationDataEntry(DataEntry):
    """This record the one eventually got eliminated by werewolves without doctor safe."""
    eliminated_player_id: str
    eliminated_player_role_name: Optional[str] = None
    eliminated_player_team_name: Optional[str] = None


class DayExileElectedDataEntry(DataEntry):
    elected_player_id: str
    elected_player_role_name: Optional[str] = None
    elected_player_team_name: Optional[str] = None


class DiscussionOrderDataEntry(DataEntry):
    chat_order_of_player_ids: List[str]


class ChatDataEntry(DataEntry, ActionDataMixin):
    """Records a chat message from a player, including private reasoning."""
    # actor_id and reasoning are inherited from ActionDataMixin
    message: str
    mentioned_player_ids: List[str] = Field(default_factory=list)


class BidDataEntry(DataEntry, ActionDataMixin):
    bid_amount: int


class BidResultDataEntry(DataEntry):
    winner_player_ids: List[str]
    bid_overview: Dict[str, int]
    mentioned_players_in_previous_turn: List[str] = []


# --- Game End and Observation Models (Unchanged) ---
class GameEndResultsDataEntry(DataEntry):
    model_config = ConfigDict(use_enum_values=True)

    winner_team: str
    winner_ids: List[str]
    loser_ids: List[str]
    scores: Dict[str, int | float]
    reason: str
    last_day: int
    last_phase: str
    survivors_until_last_round_and_role: Dict[str, str]
    all_players_and_role: Dict[str, str]
    elimination_info: List[Dict]
    """list each player's elimination status, see GameState.get_elimination_info"""

    all_players: List[Dict]
    """provide the info dump for each player"""


class VisibleRawData(BaseModel):
    data_type: str
    json_str: str
    """json dump"""

    @classmethod
    def from_entry(cls, entry: dict | DataEntry):
        if not entry: return
        if isinstance(entry, dict):
            return cls(data_type=entry.__class__.__name__, json_str=json.dumps(entry))
        return cls(data_type=entry.data.__class__.__name__, json_str=entry.model_dump_json())


class WerewolfObservationModel(BaseModel):
    player_id: str
    role: str
    team: str
    is_alive: bool
    day: int
    phase: str
    all_player_ids: List[str]
    player_thumbnails: Dict[str, str] = {}
    alive_players: List[str]
    revealed_players_by_role: Dict[str, str] = {}
    new_visible_announcements: List[str]
    new_visible_raw_data: List[VisibleRawData]
    game_state_phase: str

    def get_human_readable(self) -> str:
        # This is a placeholder implementation. A real implementation would format this nicely.
        return json.dumps(self.model_dump(), indent=2)