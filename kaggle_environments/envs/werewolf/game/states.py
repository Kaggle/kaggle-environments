from typing import List, Dict, Optional, Any, Union, Deque
from collections import defaultdict, deque
from functools import cached_property

from pydantic import BaseModel, PrivateAttr, Field, computed_field

from .records import HistoryEntryType, DataEntry, HistoryEntry
from .roles import Player, Role
from .consts import Phase, Team, RoleConst, MODERATOR_ID


class GameState(BaseModel):
    players: List[Player]
    phase: Phase = Phase.NIGHT
    day_count: int = 0
    history: Dict[int, List[HistoryEntry]] = Field(default_factory=dict)
    wallet: dict[str, int] = Field(default_factory=dict)
    _id_to_player: Dict[str, Player] = PrivateAttr(default_factory=dict)
    _history_entry_by_type: Dict[HistoryEntryType, List[HistoryEntry]] = PrivateAttr(
        default_factory=lambda: defaultdict(list))
    _history_queue: Deque[HistoryEntry] = PrivateAttr(default_factory=deque)

    @computed_field
    @cached_property
    def all_player_ids(self) -> List[str]:
        return [player.id for player in self.players]
    
    @computed_field
    @cached_property
    def all_unique_roles(self) -> List[Role]:
        role_dict = {player.role.name: player.role for player in self.players}
        return list(role_dict.values())

    def model_post_init(self, context: Any, /) -> None:
        self._id_to_player = {p.id: p for p in self.players}

    def get_player_by_id(self, pid: str):
        return self._id_to_player.get(pid)

    def get_players_by_team(self, team: Team):
        return [p for p in self.players if p.role.team == team]

    def alive_players(self):
        return [p for p in self.players if p.alive]
    
    def eliminated_players(self):
        return [p for p in self.players if not p.alive]
    
    def revealed_players(self):
        return {p.id: p.role.name for p in self.players if not p.alive}

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

    def get_history_by_type(self, entry_type: HistoryEntryType) -> List[HistoryEntry]:
        return self._history_entry_by_type[entry_type]

    def add_history_entry(self, description: str, entry_type: HistoryEntryType, public: bool,
                          visible_to: Optional[List[str]] = None, data: Optional[Union[
                DataEntry, Dict[str, Any]]] = None, source=MODERATOR_ID):
        visible_to = visible_to or []
        # Night 0 will use day_count 0, Day 1 will use day_count 1, etc.
        day_key = self.day_count
        self.history.setdefault(day_key, [])
        sys_entry = HistoryEntry(day=day_key, phase=self.phase, entry_type=entry_type,
                                 description=description, public=public,
                                 visible_to=visible_to or [],
                                 data=data,
                                 source=source)

        self.history[day_key].append(sys_entry)
        self._history_entry_by_type[entry_type].append(sys_entry)
        self._history_queue.append(sys_entry)

        public_data = data.public_view() if isinstance(data, DataEntry) else data

        public_entry = HistoryEntry(
            day=day_key, phase=self.phase, entry_type=entry_type,
            description=description, public=public,
            visible_to=visible_to or [],
            data=public_data,
            source=source
        )

        # observers message pushing below
        if public:
            for player in self.players:
                player.update(public_entry)
        else:
            for player_id in visible_to:
                player = self.get_player_by_id(player_id)
                if player:
                    player.update(public_entry)

    def eliminate_player(self, pid: str):
        player = self.get_player_by_id(pid)
        if player:
            player.eliminate(day=self.day_count, phase=self.phase)

    def consume_messages(self) -> List[HistoryEntry]:
        messages = list(self._history_queue)
        self._history_queue.clear()
        return messages

    def get_elimination_info(self):
        return [player.report_elimination() for player in self.players]
