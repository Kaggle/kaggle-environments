
from abc import ABC
from enum import Enum
from typing import List, Dict, Optional, Any, Set, Union


from pydantic import BaseModel, PrivateAttr, Field

# Renamed to avoid clash
from .roles import Player, Phase, RoleConst, Team


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


class AskDoctorSaveDataEntry(DataEntry):
    valid_candidates: List[str]
    action_json_schema: str


class AskSeerRevealDataEntry(DataEntry):
    valid_candidates: List[str]
    action_json_schema: str


class AskWerewolfVotingDataEntry(DataEntry):
    valid_targets: List[str]
    alive_werewolve_player_ids: List[str]
    voting_protocol_name: str
    voting_protocol_rule: str
    action_json_schema: str


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


class GameState(BaseModel):
    players: List[Player]
    phase: Phase = Phase.NIGHT
    day_count: int = 0
    history: Dict[int, List[HistoryEntry]] = Field(default_factory=dict)
    wallet: dict[str, int] = Field(default_factory=dict)
    _id_to_player: Dict[str, Player] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: Any, /) -> None:
        self._id_to_player = {p.id: p for p in self.players}

    def get_player_by_id(self, pid: str):
        return self._id_to_player.get(pid)

    def alive_players(self):
        return [p for p in self.players if p.alive]

    def alive_players_by_role(self, role: RoleConst):
        return [p for p in self.alive_players() if p.role.name == role]
    
    def alive_players_by_team(self, team: Team):
        return [p for p in self.alive_players() if p.role.team == team]
    
    def alive_player_counts_per_role(self):
        counts = {role: len(self.alive_players_by_role(role)) for role in RoleConst}
        return counts
    
    def alive_player_counts_per_team(self):
        return {team: len(self.alive_players_by_team(team)) for team in Team}

    _night_eliminate_queue: List[str] = PrivateAttr(default_factory=list)

    def queue_eliminate(self, target: Player):
        self._night_eliminate_queue.append(target.id)

    def clear_eliminate_queue(self):
        self._night_eliminate_queue.clear()

    _night_doctor_save_queue: List[str] = PrivateAttr(default_factory=list)

    def queue_doctor_save(self, target: Player):
        self._night_doctor_save_queue.append(target.id)

    def add_history_entry(self, description: str, entry_type: HistoryEntryType, public: bool,
                          visible_to: Optional[List[str]] = None, data: Optional[Union[DataEntry,Dict[str, Any]]] = None):
        # Night 0 will use day_count 0, Day 1 will use day_count 1, etc.
        day_key = self.day_count
        self.history.setdefault(day_key, [])
        entry = HistoryEntry(day=day_key, phase=self.phase, entry_type=entry_type,
                             description=description, public=public,
                             visible_to=set(visible_to) if visible_to is not None else set(),
                             data=data)
        self.history[day_key].append(entry)

    def eliminate_player(self, pid: str):
        player = self.get_player_by_id(pid)
        if player:
            player.alive = False
