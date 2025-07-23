import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Set, Optional, List, Dict, Any, Self

from pydantic import BaseModel, Field, field_serializer, ConfigDict

from kaggle_environments.envs.werewolf.game.consts import Phase


class HistoryEntryType(str, Enum):
    GAME_START = "game_start"
    PHASE_CHANGE = "phase_change"
    ACTION_RESULT = "action_result"
    ELIMINATION = "elimination"
    VOTE_ACTION = "vote_action"
    HEAL_ACTION = "heal_action"
    VOTE_RESULT = "vote_result"
    DISCUSSION = "discussion"
    BIDDING_INFO = "bidding_info"
    GAME_END = "game_end"
    MODERATOR_ANNOUNCEMENT = "moderator_announcement"
    PROMPT_FOR_ACTION = "prompt_for_action"
    ERROR = "error"


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

    def public_view(self) -> Self:
        """
        Returns a public view of the action, excluding private reasoning.
        This overrides the default behavior from the DataEntry base class.
        """
        return self.model_copy(update={'reasoning': None}, deep=True)


class HistoryEntry(BaseModel):
    day: int  # Day number, 0 for initial night
    phase: Phase
    entry_type: HistoryEntryType
    description: str
    public: bool = False
    visible_to: List[str] = Field(default_factory=list)
    data: Optional[dict | DataEntry]
    source: str

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


class WerewolfNightEliminationElectedDataEntry(DataEntry):
    """This record the elected elimination target by werewolves."""
    elected_target_player_id: str


class WerewolfNightEliminationDataEntry(DataEntry):
    """This record the one eventually got eliminated by werewolves without doctor safe."""
    eliminated_player_id: str
    eliminated_player_role_name: str


class DayExileElectedDataEntry(DataEntry):
    elected_player_id: str
    elected_player_role_name: str
    elected_player_team_name: str


class ChatDataEntry(DataEntry, ActionDataMixin):
    """Records a chat message from a player, including private reasoning."""
    # actor_id and reasoning are inherited from ActionDataMixin
    message: str


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