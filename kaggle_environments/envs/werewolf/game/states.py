import logging
from collections import defaultdict, deque
from functools import cached_property
from typing import List, Dict, Optional, Any, Union, Deque, Sequence, DefaultDict

from pydantic import PrivateAttr, Field, computed_field, ConfigDict

from .base import BaseState, EventHandler, PlayerID, BaseRole
from .consts import Phase, Team, RoleConst, MODERATOR_ID, PhaseDivider, DetailedPhase, EventName
from .records import DataEntry, HistoryEntry, PhaseDividerDataEntry, DataAccessLevel, \
    PlayerHistoryEntryView
from .roles import Player

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self):
        self._subs: DefaultDict[EventName, List[EventHandler]] = defaultdict(list)
    
    def register(
            self,
            event_name: EventName,
            handler: EventHandler
    ):
        self._subs[event_name].append(handler)
    
    def dispatch(self, entry: HistoryEntry):
        for handler in self._subs[entry.event_name]:
            handler(entry)


class GameState(BaseState):
    model_config = ConfigDict(use_enum_values=True)

    players: List[Player]
    phase: Phase = Phase.NIGHT
    detailed_phase: DetailedPhase = DetailedPhase.NIGHT_START
    day_count: int = 0
    history: Dict[int, List[HistoryEntry]] = Field(default_factory=dict)
    wallet: dict[PlayerID, int] = Field(default_factory=dict)
    reveal_night_elimination_role: bool = True
    reveal_day_exile_role: bool = True
    _id_to_player: Dict[PlayerID, Player] = PrivateAttr(default_factory=dict)
    _history_entry_by_type: Dict[EventName, List[HistoryEntry]] = PrivateAttr(
        default_factory=lambda: defaultdict(list))
    _history_queue: Deque[HistoryEntry] = PrivateAttr(default_factory=deque)
    _night_elimination_player_ids: List[PlayerID] = PrivateAttr(default_factory=list)
    _day_exile_player_ids: List[PlayerID] = PrivateAttr(default_factory=list)
    _event_bus: EventBus = PrivateAttr(default_factory=EventBus)

    @computed_field
    @cached_property
    def all_player_ids(self) -> List[str]:
        return [player.id for player in self.players]
    
    @computed_field
    @cached_property
    def all_unique_roles(self) -> List[BaseRole]:
        role_dict = {player.role.name: player.role for player in self.players}
        return list(role_dict.values())

    def model_post_init(self, context: Any, /) -> None:
        self._id_to_player = {p.id: p for p in self.players}

    def get_player_by_id(self, pid: PlayerID):
        return self._id_to_player.get(pid)

    def get_players_by_role(self, role: RoleConst):
        return [p for p in self.players if p.role.name == role]

    def get_players_by_team(self, team: Team):
        return [p for p in self.players if p.role.team == team]

    def alive_players(self):
        return [p for p in self.players if p.alive]
    
    def eliminated_players(self):
        return [p for p in self.players if not p.alive]
    
    def revealed_players(self):
        revealed = {p.id: p.role.name for p in self.players if not p.alive}
        if not self.reveal_day_exile_role:
            for pid in self._day_exile_player_ids:
                revealed.pop(pid, None)
        if not self.reveal_night_elimination_role:
            for pid in self._night_elimination_player_ids:
                revealed.pop(pid, None)
        return revealed

    def is_alive(self, player_id: PlayerID):
        return self.get_player_by_id(player_id).alive

    def alive_players_by_role(self, role: RoleConst):
        return [p for p in self.alive_players() if p.role.name == role]

    def alive_players_by_team(self, team: Team):
        return [p for p in self.alive_players() if p.role.team == team]

    def alive_player_counts_per_role(self):
        counts = {role: len(self.alive_players_by_role(role)) for role in RoleConst}
        return counts

    def alive_player_counts_per_team(self):
        return {team: len(self.alive_players_by_team(team)) for team in Team}

    _night_eliminate_queue: List[PlayerID] = PrivateAttr(default_factory=list)

    def queue_eliminate(self, target: Player):
        self._night_eliminate_queue.append(target.id)

    def clear_eliminate_queue(self):
        self._night_eliminate_queue.clear()

    _night_doctor_save_queue: List[PlayerID] = PrivateAttr(default_factory=list)

    def queue_doctor_save(self, target: Player):
        self._night_doctor_save_queue.append(target.id)

    def get_event_by_name(self, event_name: EventName) -> List[HistoryEntry]:
        return self._history_entry_by_type[event_name]

    def push_event(self,
                   description: str,
                   event_name: EventName,
                   public: bool,
                   visible_to: Optional[List[PlayerID]] = None,
                   data: Optional[Union[DataEntry, Dict[str, Any]]] = None, source=MODERATOR_ID):
        visible_to = visible_to or []
        # Night 0 will use day_count 0, Day 1 will use day_count 1, etc.
        day_key = self.day_count
        self.history.setdefault(day_key, [])
        sys_entry = HistoryEntry(
            day=day_key,
            phase=self.phase,
            detailed_phase=self.detailed_phase,
            event_name=event_name,
            description=description,
            public=public,
            visible_to=visible_to or [],
            data=data,
            source=source
        )

        self.history[day_key].append(sys_entry)
        self._history_entry_by_type[event_name].append(sys_entry)
        self._history_queue.append(sys_entry)

        public_view = sys_entry.view_by_access(user_level=DataAccessLevel.PUBLIC)
        personal_view = sys_entry.view_by_access(user_level=DataAccessLevel.PERSONAL)

        # observers message pushing below
        if public:
            for player in self.players:
                if player.id == source:
                    player.update(personal_view)
                else:
                    player.update(public_view)
        else:
            for player_id in visible_to:
                player = self.get_player_by_id(player_id)
                if player:
                    if player.id == source:
                        player.update(personal_view)
                    else:
                        player.update(public_view)

        # publish events
        self._event_bus.dispatch(sys_entry)
    
    def add_phase_divider(self, divider: PhaseDivider):
        """The phase divider is used to clearly separate phase boundary. This is very useful 
            for visualizer updates, where some updates naturally takes a time slice of events as input.
        """
        self.push_event(
            description=divider.value,
            event_name=EventName.PHASE_DIVIDER,
            public=False,
            data=PhaseDividerDataEntry(divider_type=divider.value)
        )

    def eliminate_player(self, pid: PlayerID):
        if pid not in self.all_player_ids:
            logger.warning(f"Tried to eliminate {pid} who is not within valid player ids {self.all_player_ids}.")
            return
        player = self.get_player_by_id(pid)
        if self.phase == Phase.NIGHT:
            self._night_elimination_player_ids.append(pid)
        else:
            self._day_exile_player_ids.append(pid)
        if player:
            player.eliminate(day=self.day_count, phase=self.phase)

    def consume_messages(self) -> List[HistoryEntry]:
        messages = list(self._history_queue)
        self._history_queue.clear()
        return messages

    def get_elimination_info(self):
        return [player.report_elimination() for player in self.players]

    def register_event_handler(self, event_name: EventName, handler: EventHandler):
        self._event_bus.register(event_name, handler)


def get_last_action_request(
        history_entries: Sequence[PlayerHistoryEntryView],
        event_name: EventName
) -> None | PlayerHistoryEntryView:
    """Get the action request from the new player history entry view updates."""
    return next((entry for entry in history_entries if entry.event_name == event_name), None)
