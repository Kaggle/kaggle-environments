import json
from abc import ABC
from datetime import datetime
from enum import IntEnum
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_serializer

from .base import BaseAction, BaseEvent, PlayerID
from .consts import DetailedPhase, EventName, ObsKeys, PerceivedThreatLevel, Phase, PhaseDivider, RoleConst, Team


def get_utc_now():
    return str(datetime.now(ZoneInfo("UTC")))


class DataAccessLevel(IntEnum):
    PUBLIC = 0
    PERSONAL = 1


class DataEntry(BaseModel, ABC):
    """Abstract base class for all data entry types."""

    pass


class ActionDataMixin(BaseModel):
    """
    A mixin for action-related DataEntry models.
    Includes the actor performing the action and their private reasoning.
    """

    actor_id: PlayerID
    reasoning: Optional[str] = Field(
        default=None, description="Private reasoning for moderator analysis.", access=DataAccessLevel.PERSONAL
    )
    perceived_threat_level: Optional[PerceivedThreatLevel] = Field(
        default=PerceivedThreatLevel.SAFE, access=DataAccessLevel.PERSONAL
    )
    action: Optional[BaseAction] = Field(default=None, access=DataAccessLevel.PERSONAL)


class VisibleRawData(BaseModel):
    data_type: str
    json_str: str


class PlayerEventView(BaseModel):
    day: int
    phase: Phase
    detailed_phase: DetailedPhase
    event_name: EventName
    description: str
    data: Optional[dict | DataEntry] = None
    source: str
    created_at: str

    @model_serializer
    def serialize(self) -> dict:
        if isinstance(self.data, DataEntry):
            data = self.data.model_dump()
        else:
            data = self.data
        return dict(
            day=self.day,
            phase=self.phase,
            detailed_phase=self.detailed_phase,
            event_name=self.event_name,
            description=self.description,
            data=data,
            source=self.source,
            created_at=self.created_at,
        )


class Event(BaseEvent):
    day: int  # Day number, 0 for initial night
    phase: Phase
    detailed_phase: DetailedPhase
    event_name: EventName
    description: str
    public: bool = False
    visible_to: List[str] = Field(default_factory=list)
    data: Optional[dict | DataEntry] = None
    source: str
    created_at: str = Field(default_factory=get_utc_now)
    visible_in_ui: bool = True
    """Determine if visible to game viewer in UI. Has no effect to game engine flow."""

    @field_serializer("data")
    def serialize_data(self, data):
        if data is None:
            return None
        if isinstance(data, dict):
            return data
        if isinstance(data, BaseModel):
            return data.model_dump()
        return None

    def serialize(self):
        # TODO: this is purely constructed for compatibility with html renderer. Need to refactor werewolf.js to handle
        #    a direct model_dump of Event
        data_dict = self.model_dump()
        return VisibleRawData(data_type=self.data.__class__.__name__, json_str=json.dumps(data_dict)).model_dump()

    def view_by_access(self, user_level: DataAccessLevel) -> PlayerEventView:
        if isinstance(self.data, ActionDataMixin):
            fields_to_include = set()
            fields_to_exclude = set()
            for name, info in self.data.__class__.model_fields.items():
                if info.json_schema_extra:
                    if user_level >= info.json_schema_extra.get("access", DataAccessLevel.PUBLIC):
                        fields_to_include.add(name)
                    else:
                        fields_to_exclude.add(name)
                else:
                    fields_to_include.add(name)
            data = self.data.model_dump(include=fields_to_include, exclude=fields_to_exclude)
        else:
            data = self.data
        out = PlayerEventView(
            day=self.day,
            phase=self.phase,
            detailed_phase=self.detailed_phase,
            event_name=self.event_name,
            description=self.description,
            data=data,
            source=self.source,
            created_at=self.created_at,
        )
        return out


# --- Game State and Setup Data Entries ---
class GameStartDataEntry(DataEntry):
    player_ids: List[PlayerID]
    number_of_players: int
    role_counts: Dict[RoleConst, int]
    team_member_counts: Dict[Team, int]
    day_discussion_protocol_name: str
    day_discussion_display_name: str
    day_discussion_protocol_rule: str
    night_werewolf_discussion_protocol_name: str
    night_werewolf_discussion_display_name: str
    night_werewolf_discussion_protocol_rule: str
    day_voting_protocol_name: str
    day_voting_display_name: str
    day_voting_protocol_rule: str


class GameStartRoleDataEntry(DataEntry):
    player_id: PlayerID
    team: Team
    role: RoleConst
    rule_of_role: str


class SetNewPhaseDataEntry(DataEntry):
    new_detailed_phase: DetailedPhase


class PhaseDividerDataEntry(DataEntry):
    divider_type: PhaseDivider


# --- Request for Action Data Entries ---
class RequestForActionDataEntry(DataEntry):
    action_json_schema: str


class RequestDoctorSaveDataEntry(RequestForActionDataEntry):
    valid_candidates: List[PlayerID]


class RequestSeerRevealDataEntry(RequestForActionDataEntry):
    valid_candidates: List[PlayerID]


class RequestWerewolfVotingDataEntry(RequestForActionDataEntry):
    valid_targets: List[PlayerID]
    alive_werewolve_player_ids: List[PlayerID]
    voting_protocol_name: str
    voting_protocol_rule: str


class RequestVillagerToSpeakDataEntry(RequestForActionDataEntry):
    pass


# --- Action and Result Data Entries ---
class SeerInspectResultDataEntry(DataEntry):
    actor_id: PlayerID
    target_id: PlayerID
    role: Optional[RoleConst]
    team: Optional[Team]


class TargetedActionDataEntry(ActionDataMixin, DataEntry):
    target_id: PlayerID


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

    saved_player_id: PlayerID


class VoteOrderDataEntry(DataEntry):
    vote_order_of_player_ids: List[PlayerID]


class WerewolfNightEliminationElectedDataEntry(DataEntry):
    """This record the elected elimination target by werewolves."""

    elected_target_player_id: Optional[PlayerID]


class WerewolfNightEliminationDataEntry(DataEntry):
    """This record the one eventually got eliminated by werewolves without doctor safe."""

    eliminated_player_id: Optional[PlayerID]
    eliminated_player_role_name: Optional[RoleConst] = None
    eliminated_player_team_name: Optional[Team] = None


class DayExileElectedDataEntry(DataEntry):
    elected_player_id: Optional[PlayerID]
    elected_player_role_name: Optional[RoleConst] = None
    elected_player_team_name: Optional[Team] = None


class DiscussionOrderDataEntry(DataEntry):
    chat_order_of_player_ids: List[PlayerID]


class ChatDataEntry(ActionDataMixin, DataEntry):
    """Records a chat message from a player, including private reasoning."""

    # actor_id and reasoning are inherited from ActionDataMixin
    message: str
    mentioned_player_ids: List[PlayerID] = Field(default_factory=list)


class BidDataEntry(ActionDataMixin, DataEntry):
    bid_amount: int


class BidResultDataEntry(DataEntry):
    winner_player_ids: List[PlayerID]
    bid_overview: Dict[PlayerID, int]
    mentioned_players_in_previous_turn: List[PlayerID] = []


# --- Game End and Observation Models (Unchanged) ---
class GameEndResultsDataEntry(DataEntry):
    model_config = ConfigDict(use_enum_values=True)

    winner_team: Team
    winner_ids: List[PlayerID]
    loser_ids: List[PlayerID]
    scores: Dict[str, int | float]
    reason: str
    last_day: int
    last_phase: Phase
    survivors_until_last_round_and_role: Dict[PlayerID, RoleConst]
    all_players_and_role: Dict[PlayerID, RoleConst]
    elimination_info: List[Dict]
    """list each player's elimination status, see GameState.get_elimination_info"""

    all_players: List[Dict]
    """provide the info dump for each player"""


class WerewolfObservationModel(BaseModel):
    player_id: PlayerID
    role: RoleConst
    team: Team
    is_alive: bool
    day: int
    detailed_phase: DetailedPhase
    all_player_ids: List[PlayerID]
    player_thumbnails: Dict[PlayerID, str] = {}
    alive_players: List[PlayerID]
    revealed_players: Dict[PlayerID, RoleConst | Team | None] = {}
    new_visible_announcements: List[str]
    new_player_event_views: List[PlayerEventView]
    game_state_phase: Phase

    def get_human_readable(self) -> str:
        # This is a placeholder implementation. A real implementation would format this nicely.
        return json.dumps(self.model_dump(), indent=2)


def set_raw_observation(kaggle_player_state, raw_obs: WerewolfObservationModel):
    """Persist raw observations for players in kaggle's player state

    Args:
        kaggle_player_state: Kaggle's interpreter state is a list of player state. This arg is one player state item.
        raw_obs: the raw observation for a player extracted from game engine.

    Note: using raw_obs.model_dump_json() will greatly increase rendering speed (due to kaggle environment's use
        of deepcopy for serialization) at the expense of harder to parse JSON rendering, since we are getting a json
        string instead of human-readable dump. We choose raw_obs.model_dump() for clarity.
    """
    kaggle_player_state.observation[ObsKeys.RAW_OBSERVATION] = raw_obs.model_dump()


def get_raw_observation(kaggle_observation) -> WerewolfObservationModel:
    """

    Args:
        kaggle_observation:

    Returns: a dict of WerewolfObservationModel dump
    """
    return WerewolfObservationModel(**kaggle_observation[ObsKeys.RAW_OBSERVATION])
