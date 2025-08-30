

import json
from enum import Enum
from typing import List, Dict, Tuple, Type, Sequence

from .actions import Action, VoteAction, HealAction, InspectAction, ChatAction, BidAction
from .consts import Phase, Team, RoleConst, PhaseDivider
from .protocols import TurnByTurnBiddingDiscussion
from .protocols import DiscussionProtocol, VotingProtocol
from .records import (
    HistoryEntryType, HistoryEntry, GameStartDataEntry, GameStartRoleDataEntry, DoctorSaveDataEntry,
    RequestDoctorSaveDataEntry, RequestSeerRevealDataEntry, RequestWerewolfVotingDataEntry, SeerInspectResultDataEntry,
    WerewolfNightEliminationElectedDataEntry, WerewolfNightEliminationDataEntry, DayExileElectedDataEntry,
    GameEndResultsDataEntry, DoctorHealActionDataEntry, SeerInspectActionDataEntry, SetNewPhaseDataEntry
)

from .roles import Player
from .states import GameState


class DetailedPhase(Enum):
    # Night Phases
    NIGHT_START = "NIGHT_START"
    NIGHT_AWAIT_ACTIONS = "NIGHT_AWAIT_ACTIONS"
    # Day Phases
    DAY_START = "DAY_START"
    DAY_BIDDING_AWAIT = "DAY_BIDDING_AWAIT"
    DAY_CHAT_AWAIT = "DAY_CHAT_AWAIT"
    DAY_VOTING_AWAIT = "DAY_VOTING_AWAIT"
    GAME_OVER = "GAME_OVER"


class ActionQueue:
    """A data structure for managing player ids in action specific queues."""

    def __init__(self):
        self._action_queue: Dict[str, List[str]] = {}

    def clear(self):
        self._action_queue = {}

    def append(self, action_cls: Type[Action], player_id: str):
        action_type = action_cls.__name__
        self._action_queue.setdefault(action_type, [])
        if player_id in self._action_queue[action_type]:
            raise ValueError(f'player {player_id} is already in the action queue. '
                             'One turn only one action is allowed in the queue.')
        self._action_queue[action_type].append(player_id)

    def extend(self, action_cls: Type[Action], player_ids: Sequence[str]):
        for player_id in player_ids:
            self.append(action_cls, player_id)

    def get(self, action_cls: Type[Action]) -> List[str]:
        """return a list of player_id for the selected action."""
        return self._action_queue.get(action_cls.__name__, [])

    def get_active_player_ids(self) -> List[str]:
        all_players = set()
        for players in self._action_queue.values():
            all_players.update(players)
        return list(all_players)


class Moderator:
    """Drives the finite-state machine for the game."""

    def __init__(
        self,
        state: GameState,
        discussion: DiscussionProtocol,
        day_voting: VotingProtocol,  # Renamed for clarity
        night_voting: VotingProtocol,
        allow_doctor_self_save: bool = False,
        reveal_night_elimination_role: bool = True,
        reveal_day_exile_role: bool = True
    ):
        self.state = state
        self.discussion = discussion
        self.day_voting = day_voting
        self.night_voting = night_voting
        self.allow_doctor_self_save = allow_doctor_self_save
        self._reveal_night_elimination_role = reveal_night_elimination_role
        self._reveal_day_exile_role = reveal_day_exile_role

        self._player_history_cursors: Dict[str, Tuple[int, int]] = {
            p.id: (0, 0) for p in self.state.players
        }

        # Detailed phase management
        self.detailed_phase: DetailedPhase = DetailedPhase.NIGHT_START
        assert state.phase == Phase.NIGHT

        self._active_night_roles_queue: List[Player] = []
        self._night_save_queue: Dict[str, List[str]] = {}
        self._action_queue = ActionQueue()

        # below is the state transition function table
        # each transition function has the signature tr_func(actions: List[Action]) where the input is a list of actions
        # with the length the same as the number of agents

        self._phase_handlers = {
            DetailedPhase.NIGHT_START: self._handle_night_start,
            DetailedPhase.NIGHT_AWAIT_ACTIONS: self._handle_night_await_actions,
            DetailedPhase.DAY_START: self._handle_day_start,
            DetailedPhase.DAY_BIDDING_AWAIT: self._handle_day_bidding_await,
            DetailedPhase.DAY_CHAT_AWAIT: self._handle_day_chat_await,
            DetailedPhase.DAY_VOTING_AWAIT: self._handle_day_voting_await,
            # DetailedPhase.GAME_OVER: self._handle_game_over,
        }

        self.night_step = 0

        # add initial announcements
        data = GameStartDataEntry(
            player_ids=[p.id for p in self.state.alive_players()],
            number_of_players=len(self.state.alive_players()),
            role_counts=self.state.alive_player_counts_per_role(),
            team_member_counts=self.state.alive_player_counts_per_team(),
            day_discussion_protocol_name=self.discussion.__class__.__name__,
            day_discussion_protocol_rule=self.discussion.discussion_rule,
            night_werewolf_discussion_protocol_name=self.night_voting.__class__.__name__,
            night_werewolf_discussion_protocol_rule=self.night_voting.voting_rule,
            day_voting_protocol_name=self.day_voting.__class__.__name__,
            day_voting_protocol_rule=self.day_voting.voting_rule
        )

        role_msg = "\n".join(
            ["The following explain the function of each role."] +
            [f"  * Role name {role.name.value} - team {role.team.value} - {role.descriptions}"
             for role in self.state.all_unique_roles])
        self.doctor_special_msg = "Doctor is allowed to save themselves during night time." if allow_doctor_self_save \
            else "Doctor is NOT allowed to save themselves during night time."
        day_exile_reveal_msg = "If a player is exiled in the day, their role will be revealed." if reveal_day_exile_role \
            else "If a player is exiled in the day, their role will NOT be revealed."
        night_elimination_reveal_msg = "If a player is eliminated at night, their role will be revealed." \
            if reveal_night_elimination_role else "If a player is eliminated at night, their role will NOT be revealed."

        description = "\n - ".join([
            "Werewolf game begins.",
            f"All player ids: {data.player_ids}",
            f"Number of alive players: {data.number_of_players}.",
            f"Role counts: {data.role_counts}.",
            f"Alive team member counts: {data.team_member_counts}",
            f"Day discussion protocol ({data.day_discussion_protocol_name}): {data.day_discussion_protocol_name}",
            f"Day voting protocol ({data.day_voting_protocol_name}): {data.day_voting_protocol_rule}",
            f"Night werewolf voting protocol ({data.night_werewolf_discussion_protocol_name}): {data.night_werewolf_discussion_protocol_rule}",
            role_msg,
            self.doctor_special_msg,
            day_exile_reveal_msg,
            night_elimination_reveal_msg
        ])
        self.state.add_history_entry(
            description=description,
            entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
            public=True,
            data=data
        )
        # add role specific announcements
        for player in self.state.alive_players():
            data = GameStartRoleDataEntry(
                player_id=player.id,
                team=player.role.team,
                role=player.role.name,
                rule_of_role=player.role.descriptions
            )
            self.state.add_history_entry(
                description=f'Your player id is "{data.player_id}". Your team is "{data.team}". Your role is "{data.role}".\n'
                            f"The rule of your role: {data.rule_of_role}",
                entry_type=HistoryEntryType.GAME_START,
                public=False,
                visible_to=[player.id],
                data=data
            )

    def set_new_phase(self, new_detailed_phase, add_one_day=False):
        self.detailed_phase = new_detailed_phase
        if new_detailed_phase == DetailedPhase.NIGHT_START or new_detailed_phase == DetailedPhase.NIGHT_AWAIT_ACTIONS:
            self.state.phase = Phase.NIGHT
        else:
            self.state.phase = Phase.DAY
        if add_one_day:
            self.state.day_count += 1

        data = SetNewPhaseDataEntry(new_detailed_phase=new_detailed_phase)
        self.state.add_history_entry(
            description=f"{new_detailed_phase} begins.",
            entry_type=HistoryEntryType.PHASE_CHANGE,
            public=False,
            data=data
        )

    def get_active_player_ids(self) -> List[str]:
        return self._action_queue.get_active_player_ids()

    def get_observation(self, player_id: str) -> Tuple[List[HistoryEntry], Tuple[int, int]]:
        """
        Retrieves new history entries for a player since their last call.
        Returns the entries and the new potential cursor position.
        Does NOT update the player's history cursor itself.
        """
        read_from_day, read_from_idx_in_day = self._player_history_cursors.get(player_id, (0, 0))
        newly_visible_entries: List[HistoryEntry] = []

        # These will store the position of the next entry to be read after this call
        updated_cursor_day = read_from_day
        updated_cursor_idx_in_day = read_from_idx_in_day

        sorted_days = sorted(self.state.history.keys())

        for day_num in sorted_days:
            if day_num < read_from_day:
                continue

            day_entries = self.state.history[day_num]
            current_processing_idx = 0
            if day_num == read_from_day:
                current_processing_idx = read_from_idx_in_day

            while current_processing_idx < len(day_entries):
                entry = day_entries[current_processing_idx]
                if entry.public or player_id in entry.visible_to:
                    newly_visible_entries.append(entry)

                current_processing_idx += 1
                updated_cursor_day = day_num
                updated_cursor_idx_in_day = current_processing_idx  # Next index to read

        return newly_visible_entries, (updated_cursor_day, updated_cursor_idx_in_day)

    def update_player_cursor(self, player_id: str, new_cursor: Tuple[int, int]):
        """
        Updates the history cursor for a given player.
        """
        self._player_history_cursors[player_id] = new_cursor

    def advance(self, player_actions: Dict[str, Action]):
        # Process the incoming actions for the current phase.
        current_handler = self._phase_handlers.get(self.detailed_phase)
        if current_handler:
            current_handler(player_actions)
        else:
            raise ValueError(f"Unhandled detailed_phase: {self.detailed_phase}")

        # Loop through automatic state transitions (those that don't need agent actions)
        # This continues until the game is over or requires new agent input.
        # this logic is required since Environments in core.py requires that there are some players being ACTIVE to
        # continue. Otherwise, if all INACTIVE the game is marked done.
        while not self.get_active_player_ids() and not self.is_game_over():
            next_handler = self._phase_handlers.get(self.detailed_phase)
            if next_handler:
                next_handler({})  # Pass empty actions for automatic transitions
            else:
                raise ValueError(f"Unhandled detailed_phase during transition: {self.detailed_phase}")

        # After all transitions, check for game over.
        if self.is_game_over() and self.detailed_phase != DetailedPhase.GAME_OVER:
            # clear action queue
            self._action_queue.clear()
            self.set_new_phase(DetailedPhase.GAME_OVER)
            self._determine_and_log_winner()

    def _handle_night_start(self, player_actions: Dict[str, Action]):
        self._action_queue.clear()
        self.state.add_phase_divider(PhaseDivider.NIGHT_START)
        self.state.add_phase_divider(PhaseDivider.NIGHT_VOTE_START)
        self.state.add_history_entry(
            description=f"Night {self.state.day_count} begins.",
            entry_type=HistoryEntryType.PHASE_CHANGE,
            public=True,
        )
        # announce await action to doctor

        self.valid_doctor_save_ids = {}
        for doctor in self.state.alive_players_by_role(RoleConst.DOCTOR):
            self._action_queue.append(HealAction, doctor.id)
            self.valid_doctor_save_ids[doctor.id] = [f"{p.id}" for p in self.state.alive_players()] \
                if self.allow_doctor_self_save else [f"{p.id}" for p in self.state.alive_players() if p != doctor]
            data_entry = RequestDoctorSaveDataEntry(
                valid_candidates=self.valid_doctor_save_ids[doctor.id],
                action_json_schema=json.dumps(HealAction.schema_for_player())
            )
            self.state.add_history_entry(
                description=f"Wake up Doctor. Who would you like to save? "
                            f"The options are {data_entry.valid_candidates}.\n{self.doctor_special_msg}",
                entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                public=False,
                visible_to=[doctor.id],
                data=data_entry
            )

        # announce await action to seer
        for seer in self.state.alive_players_by_role(RoleConst.SEER):
            self._action_queue.append(InspectAction, seer.id)
            data_entry = RequestSeerRevealDataEntry(
                valid_candidates=[p.id for p in self.state.alive_players() if p != seer],
                action_json_schema=json.dumps(InspectAction.schema_for_player())
            )
            self.state.add_history_entry(
                description=f"Wake up Seer. Who would you like to see their true role? The options are {data_entry.valid_candidates}.",
                entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                public=False,
                visible_to=[seer.id],
                data=data_entry
            )

        # initialize werewolves voting
        self.state.add_phase_divider(PhaseDivider.NIGHT_VOTE_START)
        alive_werewolves = self.state.alive_players_by_role(RoleConst.WEREWOLF)
        alive_werewolf_ids = list({p.id for p in alive_werewolves})
        potential_targets = self.state.alive_players_by_team(Team.VILLAGERS)  # Target non-werewolves

        if alive_werewolves:
            data = RequestWerewolfVotingDataEntry(
                valid_targets=[f"{p.id}" for p in potential_targets],
                alive_werewolve_player_ids=[f"{p.id}" for p in alive_werewolves],
                voting_protocol_name=self.night_voting.__class__.__name__,
                voting_protocol_rule=self.night_voting.voting_rule,
                action_json_schema=json.dumps(VoteAction.schema_for_player()),
            )
            self.state.add_history_entry(
                description=f"Wake up Werewolves. Your fellow alive werewolves are: {data.alive_werewolve_player_ids}. "
                            f"Choose one target player to eliminate tonight. The voting rule ({data.voting_protocol_name}): {data.voting_protocol_rule} "
                            f"Who would you like to eliminate tonight? Options: {data.valid_targets}.",
                entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                public=False,
                visible_to=alive_werewolf_ids,
                data=data
            )

        self.night_voting.begin_voting(
            state=self.state,
            alive_voters=alive_werewolves,
            potential_targets=potential_targets
        )
        self._action_queue.extend(VoteAction, self.night_voting.get_next_voters())

        # state transition
        self.set_new_phase(DetailedPhase.NIGHT_AWAIT_ACTIONS)
        self.night_step = 1  # Mark that initial night roles (doc, seer) + first WW vote are expected

    def _handle_night_await_actions(self, player_actions: Dict[str, Action]):
        # Process actions from Doctor, Seer (if it's the first step of night actions)
        # and Werewolves (always processed if they send votes)

        if self.night_step == 1:
            for actor_id, action in player_actions.items():
                player = self.state.get_player_by_id(actor_id)
                if not player or not player.alive:
                    continue

                if player.role.name == RoleConst.DOCTOR and isinstance(action, HealAction):
                    if not self.allow_doctor_self_save:
                        if action.target_id == actor_id:
                            self.state.add_history_entry(
                                description=f'Player "{actor_id}", doctor is not allowed to self save. '
                                            f'Your target is {action.target_id}, which is your own id.',
                                entry_type=HistoryEntryType.ERROR,
                                public=False,
                                visible_to=[actor_id]
                            )
                            # skip since doctor can't self save
                            continue
                    data = DoctorHealActionDataEntry(
                        actor_id=actor_id,
                        target_id=action.target_id,
                        reasoning=action.reasoning,
                        perceived_threat_level=action.perceived_threat_level
                    )
                    self.state.add_history_entry(
                        description=f'Player "{actor_id}", you chose to heal player "{action.target_id}".',
                        entry_type=HistoryEntryType.HEAL_ACTION,
                        public=False,
                        visible_to=[actor_id],
                        data=data
                    )
                    target_id = action.target_id
                    if target_id not in self._night_save_queue:
                        self._night_save_queue[target_id] = []
                    self._night_save_queue[target_id].append(actor_id)
                elif player.role.name == RoleConst.SEER and isinstance(action, InspectAction):

                    action_data = SeerInspectActionDataEntry(
                        actor_id=actor_id,
                        target_id=action.target_id,
                        reasoning=action.reasoning,
                        perceived_threat_level=action.perceived_threat_level
                    )
                    self.state.add_history_entry(
                        description=f'Player "{actor_id}", you chose to inspect player "{action.target_id}".',
                        entry_type=HistoryEntryType.ACTION_RESULT,
                        public=False,
                        visible_to=[actor_id],
                        data=action_data
                    )
                    target_player = self.state.get_player_by_id(action.target_id)
                    if target_player:  # Ensure target exists
                        data = SeerInspectResultDataEntry(
                            actor_id=actor_id,
                            target_id=action.target_id,
                            role=target_player.role.name,
                            team=target_player.role.team.value
                        )
                        self.state.add_history_entry(
                            description=f'Player "{actor_id}", you inspected {target_player.id}. '
                                        f'Their role is a "{target_player.role.name}" in team '
                                        f'"{target_player.role.team.value}".',
                            entry_type=HistoryEntryType.ACTION_RESULT,
                            public=False,
                            visible_to=[actor_id],
                            data=data
                        )
                    else:
                        self.state.add_history_entry(
                            description=f'Player "{actor_id}", you inspected player "{action.target_id}",'
                                        f' but this player could not be found.',
                            entry_type=HistoryEntryType.ERROR,
                            public=False,
                            visible_to=[actor_id]
                        )

        # Process werewolf votes
        werewolf_voters_expected = self._action_queue.get(VoteAction)
        if werewolf_voters_expected:
            # Using collect_votes (plural) assuming it can handle timeouts by comparing
            # the actions in player_actions against the list of expected voters.
            # This is consistent with day_voting.
            self.night_voting.collect_votes(player_actions, self.state, werewolf_voters_expected)

        if self.night_step == 1:
            self.night_step += 1  # Increment so Doctor/Seer actions aren't re-processed if WW voting is sequential

        self._action_queue.clear()  # Prepare for next set of actors or end of night

        if not self.night_voting.done():
            # Werewolf voting is not complete (e.g., sequential voting)
            next_ww_voters = self.night_voting.get_next_voters()
            self._action_queue.extend(VoteAction, next_ww_voters)

            # Re-prompt only the werewolves whose turn it is now
            vote_action_queue = self._action_queue.get(VoteAction)
            alive_werewolves_still_to_vote = [p for p in self.state.alive_players_by_role(RoleConst.WEREWOLF) if
                                              p.id in vote_action_queue]
            if alive_werewolves_still_to_vote:
                for ww_voter in alive_werewolves_still_to_vote:
                    prompt = self.night_voting.get_voting_prompt(self.state, ww_voter.id)  # Use protocol's prompt
                    self.state.add_history_entry(
                        description=prompt,
                        entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                        public=False,
                        visible_to=[ww_voter.id]
                    )
            # Stay in NIGHT_AWAIT_ACTIONS
        else:
            # All werewolf votes are in, or voting is otherwise complete
            werewolf_target_id = self.night_voting.get_elected()

            data = WerewolfNightEliminationElectedDataEntry(elected_target_player_id=werewolf_target_id)
            self.state.add_history_entry(
                description=f'Werewolves elected to eliminate player "{data.elected_target_player_id}".',
                entry_type=HistoryEntryType.ACTION_RESULT,
                public=False,
                visible_to=[p.id for p in self.state.alive_players_by_team(Team.WEREWOLVES)],
                data=data
            )

            if werewolf_target_id:
                werewolf_target_player = self.state.get_player_by_id(werewolf_target_id)
                if werewolf_target_player:
                    if werewolf_target_id in self._night_save_queue:
                        saving_doctor_ids = self._night_save_queue[werewolf_target_id]
                        save_data = DoctorSaveDataEntry(saved_player_id=werewolf_target_id)

                        # Private message to doctor(s) with data for the renderer
                        self.state.add_history_entry(
                            description=f"Your heal on player \"{werewolf_target_id}\" was successful!",
                            entry_type=HistoryEntryType.ACTION_RESULT,
                            public=False,
                            data=save_data,
                            visible_to=saving_doctor_ids
                        )

                        # Generic public announcement for all other players
                        self.state.add_history_entry(
                            description=f"A player was attacked but was saved by the Doctor!",
                            entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                            public=True
                        )
                    else:
                        original_role_name = werewolf_target_player.role.name.value
                        self.state.eliminate_player(werewolf_target_id)
                        if self._reveal_night_elimination_role:
                            data = WerewolfNightEliminationDataEntry(
                                eliminated_player_id=werewolf_target_id,
                                eliminated_player_role_name=original_role_name,
                                eliminated_player_team_name=werewolf_target_player.role.team.value
                            )
                            self.state.add_history_entry(
                                description=f'Last night, player "{werewolf_target_id}" was eliminated by werewolves. '
                                            f'Their role was a "{original_role_name}".',
                                entry_type=HistoryEntryType.ELIMINATION,
                                public=True,
                                data=data
                            )
                        else:
                            data = WerewolfNightEliminationDataEntry(eliminated_player_id=werewolf_target_id)
                            self.state.add_history_entry(
                                description=f'Last night, player "{werewolf_target_id}" was eliminated by werewolves.',
                                entry_type=HistoryEntryType.ELIMINATION,
                                public=True,
                                data=data
                            )
                else:
                    self.state.add_history_entry(
                        description=f'Last night, werewolves targeted player "{werewolf_target_id}", '
                                    f'but this player could not be found. No one was eliminated by werewolves.',
                        entry_type=HistoryEntryType.ERROR,
                        public=True
                    )
            else:
                # TODO: add fail over if no consensus can be reached (all werewolf action failed) for several voting rounds.
                self.state.add_history_entry(
                    description="Last night, the werewolves did not reach a consensus (or no valid target was chosen). No one was eliminated by werewolves.",
                    entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,  # This could also be a VOTE_RESULT type
                    data={
                        "outcome": "no_elimination",
                        "reason": "no_consensus_werewolves"
                    },
                    public=True
                )

            self.night_voting.reset()
            self._night_save_queue = {}
            self.state.add_phase_divider(PhaseDivider.NIGHT_VOTE_END)
            self.state.add_phase_divider(PhaseDivider.NIGHT_END)
            if not self.is_game_over():
                self.set_new_phase(DetailedPhase.DAY_START, add_one_day=True)

    def _handle_day_start(self, player_actions: Dict[str, Action]):
        self.state.add_phase_divider(PhaseDivider.DAY_START)
        self._action_queue.clear()
        self.night_step = 0  # Reset night step counter

        self.state.add_history_entry(
            description=f"Day {self.state.day_count} begins.",
            entry_type=HistoryEntryType.DAY_START,
            public=True
        )

        self.state.add_history_entry(
            description=f"Villagers, let's decide who to exile. The discussion rule is: {self.discussion.discussion_rule}",
            entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
            public=True,
            data={'discussion_rule': self.discussion.discussion_rule}
        )

        self.state.add_phase_divider(PhaseDivider.DAY_CHAT_START)
        self.discussion.begin(self.state)

        # Check if the protocol starts with bidding
        if isinstance(self.discussion, TurnByTurnBiddingDiscussion):
            self.set_new_phase(DetailedPhase.DAY_BIDDING_AWAIT)
            # In bidding, all alive players can be active
            bidders = self.discussion.speakers_for_tick(self.state)
            self._action_queue.extend(BidAction, bidders)
            self.discussion.prompt_speakers_for_tick(self.state, bidders)
        else:
            # If no bidding, go straight to chatting
            self.set_new_phase(DetailedPhase.DAY_CHAT_AWAIT)
            initial_speakers = self.discussion.speakers_for_tick(self.state)
            self._action_queue.extend(ChatAction, initial_speakers)
            self.discussion.prompt_speakers_for_tick(self.state, initial_speakers)

    def _handle_day_bidding_await(self, player_actions: Dict[str, Action]):
        current_bidders = self._action_queue.get(BidAction)
        self._action_queue.clear()

        # The protocol processes bid actions
        self.discussion.process_actions(list(player_actions.values()), current_bidders, self.state)

        # We need to explicitly check if the bidding sub-phase is over
        # This requires a reference to the bidding protocol within TurnByTurnBiddingDiscussion
        bidding_protocol = self.discussion.bidding
        if bidding_protocol.is_finished(self.state):
            self.state.add_history_entry(
                description="Bidding has concluded. The discussion will now begin.",
                entry_type=HistoryEntryType.PHASE_CHANGE,
                public=True
            )
            # Transition to the chat phase
            self.set_new_phase(DetailedPhase.DAY_CHAT_AWAIT)

            # Get the first speakers for the chat phase (could be bid winners)
            next_speakers = self.discussion.speakers_for_tick(self.state)
            self._action_queue.extend(ChatAction, next_speakers)
            self.discussion.prompt_speakers_for_tick(self.state, next_speakers)
        else:
            # Bidding is not over (e.g., sequential auction), get next bidders
            next_bidders = self.discussion.speakers_for_tick(self.state)
            self._action_queue.extend(BidAction, next_bidders)
            self.discussion.prompt_speakers_for_tick(self.state, next_bidders)
            # Stay in DAY_BIDDING_AWAIT

    def _handle_day_chat_await(self, player_actions: Dict[str, Action]):
        speaker_ids = self._action_queue.get(ChatAction)
        self._action_queue.clear()
        self.discussion.process_actions(list(player_actions.values()), speaker_ids, self.state)

        if self.discussion.is_discussion_over(self.state):
            self.state.add_history_entry(
                description="Daytime discussion has concluded. Moving to day vote.",
                entry_type=HistoryEntryType.PHASE_CHANGE,
                public=True
            )
            self.discussion.reset()

            self.state.add_phase_divider(PhaseDivider.DAY_CHAT_END)
            self.state.add_phase_divider(PhaseDivider.DAY_VOTE_START)
            alive_players = self.state.alive_players()
            self.day_voting.begin_voting(self.state, alive_players, alive_players)
            self.set_new_phase(DetailedPhase.DAY_VOTING_AWAIT)
            self.state.add_history_entry(
                description="Voting phase begins. We will decide who to exile today."
                            f"\nDay voting Rule: {self.day_voting.voting_rule}"
                            f"\nCurrent alive players are: {[player.id for player in alive_players]}",
                entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                public=True,
                data={"voting_rule": self.day_voting.voting_rule}
            )
            next_voters_ids = self.day_voting.get_next_voters()
            self._action_queue.extend(VoteAction, next_voters_ids)

            vote_queue = self._action_queue.get(VoteAction)
            if vote_queue:
                for voter_id in vote_queue:  # Prompt only the current batch of voters
                    player = self.state.get_player_by_id(voter_id)
                    if player and player.alive:
                        prompt = self.day_voting.get_voting_prompt(self.state, voter_id)
                        self.state.add_history_entry(
                            description=prompt, entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                            public=False, visible_to=[voter_id]
                        )
        else:
            # Discussion is not over. Check if we need to go back to bidding action and phase.
            action_cls = ChatAction
            if isinstance(self.discussion, TurnByTurnBiddingDiscussion) and self.discussion._phase == "bidding":
                self.set_new_phase(DetailedPhase.DAY_BIDDING_AWAIT)
                action_cls = BidAction

            # Get the next active players (either bidders or the next speaker)
            next_actors = self.discussion.speakers_for_tick(self.state)
            self._action_queue.extend(action_cls, next_actors)
            self.discussion.prompt_speakers_for_tick(self.state, next_actors)

    def _handle_day_voting_await(self, player_actions: Dict[str, Action]):
        vote_queue = self._action_queue.get(VoteAction)
        self.day_voting.collect_votes(player_actions, self.state, vote_queue)
        self._action_queue.clear()  # Clear previous voters

        if self.day_voting.done():
            exiled_player_id = self.day_voting.get_elected()
            if exiled_player_id:
                exiled_player = self.state.get_player_by_id(exiled_player_id)
                if exiled_player:
                    original_role_name = exiled_player.role.name.value
                    self.state.eliminate_player(exiled_player_id)
                    if self._reveal_day_exile_role:
                        data = DayExileElectedDataEntry(
                            elected_player_id=exiled_player_id,
                            elected_player_role_name=original_role_name,
                            elected_player_team_name=exiled_player.role.team.value
                        )
                        self.state.add_history_entry(
                            description=f'"{exiled_player_id}" in team {data.elected_player_team_name} is exiled by vote. The player is a {original_role_name}.',
                            entry_type=HistoryEntryType.ELIMINATION,
                            public=True,
                            data=data
                        )
                    else:
                        data = DayExileElectedDataEntry(
                            elected_player_id=exiled_player_id
                        )
                        self.state.add_history_entry(
                            description=f'"{exiled_player_id}" in team {data.elected_player_team_name} is exiled by vote.',
                            entry_type=HistoryEntryType.ELIMINATION,
                            public=True,
                            data=data
                        )
            else:
                self.state.add_history_entry(
                    description="The vote resulted in no exile (e.g., a tie, no majority, or all abstained).",
                    entry_type=HistoryEntryType.VOTE_RESULT,
                    public=True,
                    data={
                        "vote_type": "day_exile",
                        "outcome": "no_exile",
                        "reason": "tie_or_no_majority"
                    }
                )

            self.day_voting.reset()
            self.state.add_phase_divider(PhaseDivider.DAY_VOTE_END)
            self.state.add_phase_divider(PhaseDivider.DAY_END)
            if not self.is_game_over():
                self.set_new_phase(DetailedPhase.NIGHT_START)
        else:
            next_voters_ids = self.day_voting.get_next_voters()
            self._action_queue.extend(VoteAction, next_voters_ids)
            if next_voters_ids:
                for voter_id in next_voters_ids:
                    player = self.state.get_player_by_id(voter_id)
                    if player and player.alive:
                        prompt = self.day_voting.get_voting_prompt(self.state, voter_id)
                        self.state.add_history_entry(
                            description=prompt, entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                            public=False, visible_to=[voter_id]
                        )
            # Stay in DetailedPhase.DAY_VOTING_AWAIT

    def _determine_and_log_winner(self):
        # Check if a GAME_END entry already exists
        game_end_history = self.state.get_history_by_type(HistoryEntryType.GAME_END)
        if game_end_history:
            return  # Winner already logged for this day count

        wolves = [p for p in self.state.alive_players() if p.role.team == Team.WEREWOLVES]
        villagers = [p for p in self.state.alive_players() if p.role.team == Team.VILLAGERS]

        if not wolves:
            winner_team = Team.VILLAGERS.value
            winner_message = "Game Over: Villagers Win!"
            reason = "Reason: All werewolves exiled."
            scores = {p.id: 1 for p in self.state.get_players_by_team(team=Team.VILLAGERS)}
            scores.update({p.id: 0 for p in self.state.get_players_by_team(team=Team.WEREWOLVES)})
            winner_ids = [p.id for p in self.state.get_players_by_team(Team.VILLAGERS)]
            loser_ids = [p.id for p in self.state.get_players_by_team(Team.WEREWOLVES)]
        else:
            winner_team = Team.WEREWOLVES.value
            winner_message = "Game Over: Werewolves Win!"
            reason = f"Reason: len(werewolves) >= len(villagers). Final counts: len(werewolves)={len(wolves)}, len(villagers)={len(villagers)})."
            scores = {p.id: 1 for p in self.state.get_players_by_team(team=Team.WEREWOLVES)}
            scores.update({p.id: 0 for p in self.state.get_players_by_team(team=Team.VILLAGERS)})
            loser_ids = [p.id for p in self.state.get_players_by_team(Team.VILLAGERS)]
            winner_ids = [p.id for p in self.state.get_players_by_team(Team.WEREWOLVES)]

        data = GameEndResultsDataEntry(
            winner_team=winner_team,
            winner_ids=winner_ids,
            loser_ids=loser_ids,
            scores=scores,
            reason=reason,
            last_day=self.state.day_count,
            last_phase=self.state.phase.value,
            survivors_until_last_round_and_role={p.id: p.role.name.value for p in self.state.alive_players()},
            all_players_and_role={p.id: p.role.name.value for p in self.state.players},
            elimination_info=self.state.get_elimination_info(),
            all_players=[p.model_dump() for p in self.state.players]
        )

        self.state.add_history_entry(
            description=f"{winner_message}\n{reason}\nScores: {scores}\n"
                        f"Survivors: {data.survivors_until_last_round_and_role}\n"
                        f"All player roles: {data.all_players_and_role}",
            entry_type=HistoryEntryType.GAME_END,
            public=True,
            data=data
        )

    def is_game_over(self) -> bool:
        if self.detailed_phase == DetailedPhase.GAME_OVER:
            return True
        wolves = self.state.alive_players_by_team(Team.WEREWOLVES)
        villagers = self.state.alive_players_by_team(Team.VILLAGERS)
        if not wolves and villagers: return True
        if wolves and len(wolves) >= len(villagers): return True
        return False