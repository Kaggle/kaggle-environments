from abc import ABC
from enum import Enum
from typing import Set, Optional, List, Dict

from pydantic import BaseModel, Field

from kaggle_environments.envs.werewolf.game.consts import Phase


class HistoryEntryType(str, Enum):
    GAME_START = "game_start"
    PHASE_CHANGE = "phase_change"
    ACTION_RESULT = "action_result"
    ELIMINATION = "elimination"
    VOTE_ACTION = "vote_action"
    VOTE_RESULT = "vote_result"
    DISCUSSION = "discussion"
    BIDDING_INFO = "bidding_info"
    GAME_END = "game_end"
    MODERATOR_ANNOUNCEMENT = "moderator_announcement"
    PROMPT_FOR_ACTION = "prompt_for_action"
    ERROR = "error"


class DataEntry(BaseModel, ABC):
    pass


class HistoryEntry(BaseModel):
    day: int  # Day number, 0 for initial night
    phase: Phase
    entry_type: HistoryEntryType
    description: str
    public: bool = False
    visible_to: Set[str] = Field(default_factory=set)
    data: Optional[DataEntry]
    source: str


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


class AskForActionDataEntry(DataEntry):
    action_json_schema: str


class AskDoctorSaveDataEntry(AskForActionDataEntry):
    valid_candidates: List[str]


class AskSeerRevealDataEntry(AskForActionDataEntry):
    valid_candidates: List[str]


class AskWerewolfVotingDataEntry(AskForActionDataEntry):
    valid_targets: List[str]
    alive_werewolve_player_ids: List[str]
    voting_protocol_name: str
    voting_protocol_rule: str


class AskVillagerToSpeakDataEntry(AskForActionDataEntry):
    pass


class SeerInspectResultDataEntry(DataEntry):
    actor_id: str
    target_id: str
    role: str
    team: str


class WerewolfNightVoteDataEntry(DataEntry):
    actor_id: str
    target_id: str
    reasoning: Optional[str]


class WerewolfNightEliminationElectedDataEntry(DataEntry):
    """This record the elected elimination target by werewolves."""
    elected_target_player_id: str


class WerewolfNightEliminationDataEntry(DataEntry):
    """This record the one eventually got eliminated by werewolves without doctor safe."""
    eliminated_player_id: str
    eliminated_player_role_name: str


class DayExileVoteDataEntry(DataEntry):
    actor_id: str
    target_id: str
    reasoning: Optional[str]


class DayExileElectedDataEntry(DataEntry):
    elected_player_id: str
    elected_player_role_name: str
    elected_player_team_name: str


class GameEndResultsDataEntry(DataEntry):
    winner_team: str
    scores: Dict[str, int | float]
    reason: str
    survivors_until_last_round_and_role: Dict[str, str]
    all_players_and_role: Dict[str, str]
