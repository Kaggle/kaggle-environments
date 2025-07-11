from enum import Enum
from typing import List, Dict, Tuple
import json

from .actions import (
    Action, VoteAction,
    HealAction, InspectAction, ChatAction, EliminateProposalAction, NoOpAction
)
from .protocols import DiscussionProtocol, VotingProtocol
from .roles import Player
from .consts import Phase, Team, RoleConst
from .states import GameState
from .records import HistoryEntryType, HistoryEntry, GameStartDataEntry, GameStartRoleDataEntry, DoctorSaveDataEntry, \
    RequestDoctorSaveDataEntry, \
    RequestSeerRevealDataEntry, RequestWerewolfVotingDataEntry, SeerInspectResultDataEntry, WerewolfNightVoteDataEntry, \
    WerewolfNightEliminationElectedDataEntry, WerewolfNightEliminationDataEntry, DayExileElectedDataEntry, \
    GameEndResultsDataEntry, DoctorHealActionDataEntry, SeerInspectActionDataEntry


class DetailedPhase(Enum):
    # Night Phases
    NIGHT_START = "NIGHT_START"
    NIGHT_AWAIT_ACTIONS = "NIGHT_AWAIT_ACTIONS"
    # Day Phases
    DAY_START = "DAY_START"
    DAY_DISCUSSION_AWAIT_CHAT = "DAY_DISCUSSION_AWAIT_CHAT"
    DAY_VOTING_AWAIT = "DAY_VOTING_AWAIT"
    GAME_OVER = "GAME_OVER"


class Moderator:
    """Drives the finite-state machine for the game."""

    def __init__(self,
                 state: GameState,
                 discussion: DiscussionProtocol,
                 day_voting: VotingProtocol,  # Renamed for clarity
                 night_voting: VotingProtocol,
                 allow_doctor_self_save: bool = False):
        self.state = state
        self.discussion = discussion
        self.day_voting = day_voting
        self.night_voting = night_voting
        self.allow_doctor_self_save = allow_doctor_self_save

        self._player_history_cursors: Dict[str, Tuple[int, int]] = {
            p.id: (0, 0) for p in self.state.players
        }

        # Detailed phase management
        self.detailed_phase: DetailedPhase = DetailedPhase.NIGHT_START
        assert state.phase == Phase.NIGHT

        self._active_night_roles_queue: List[Player] = []
        self._night_save_queue: List[str] = []
        self._action_queue: List[str] = []  # Player IDs expected to act

        # below is the state transition function table
        # each transition function has the signature tr_func(actions: List[Action]) where the input is a list of actions
        # with the length the same as the number of agents

        self._phase_handlers = {
            DetailedPhase.NIGHT_START: self._handle_night_start,
            DetailedPhase.NIGHT_AWAIT_ACTIONS: self._handle_night_await_actions,
            DetailedPhase.DAY_START: self._handle_day_start,
            DetailedPhase.DAY_DISCUSSION_AWAIT_CHAT: self._handle_day_discussion_await_chat,
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
        self.state.add_history_entry(
            description="\n".join([
                "Werewolf game begins.",
                f"All player ids: {', '.join(data.player_ids)}",
                f"Number of alive players: {data.number_of_players}.",
                f"Role counts: {data.role_counts}."
                f"Alive team member counts: {data.team_member_counts}",
                f"Day discussion protocol ({data.day_discussion_protocol_name}): {data.day_discussion_protocol_name}",
                f"Day voting protocol ({data.day_voting_protocol_name}): {data.day_voting_protocol_rule}",
                f"Night werewolf voting protocol ({data.night_werewolf_discussion_protocol_name}): {data.night_werewolf_discussion_protocol_rule}"
            ]),
            entry_type=HistoryEntryType.GAME_START,
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

    def get_active_player_ids(self) -> List[str]:
        return list(self._action_queue)  # Return a copy

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
            self.detailed_phase = DetailedPhase.GAME_OVER
            self._determine_and_log_winner()

    def _handle_night_start(self, player_actions: Dict[str, Action]):
        self._action_queue.clear()
        self.state.add_history_entry(
            description=f"Night {self.state.day_count} begins.",
            entry_type=HistoryEntryType.PHASE_CHANGE,
            public=True,
        )
        # announce await action to doctor

        self.valid_doctor_save_ids = {}
        for doctor in self.state.alive_players_by_role(RoleConst.DOCTOR):
            if doctor.id not in self._action_queue: self._action_queue.append(doctor.id)
            self.valid_doctor_save_ids[doctor.id] = [f"{p.id}" for p in self.state.alive_players()] \
                if self.allow_doctor_self_save else [f"{p.id}" for p in self.state.alive_players() if p != doctor]
            data_entry = RequestDoctorSaveDataEntry(
                valid_candidates=self.valid_doctor_save_ids[doctor.id],
                action_json_schema=json.dumps(HealAction.model_json_schema())
            )
            self.state.add_history_entry(
                description=f"Wake up Doctor. Who would you like to save? The options are {data_entry.valid_candidates}.",
                entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                public=False,
                visible_to=[doctor.id],
                data=data_entry
            )

        # announce await action to seer
        for seer in self.state.alive_players_by_role(RoleConst.SEER):
            if seer.id not in self._action_queue: self._action_queue.append(seer.id)
            data_entry = RequestSeerRevealDataEntry(
                valid_candidates=[p.id for p in self.state.alive_players() if p != seer],
                action_json_schema=json.dumps(InspectAction.model_json_schema())
            )
            self.state.add_history_entry(
                description=f"Wake up Seer. Who would you like to see their true role? The options are {data_entry.valid_candidates}.",
                entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                public=False,
                visible_to=[seer.id],
                data=data_entry
            )

        # initialize werewolves voting
        alive_werewolves = self.state.alive_players_by_role(RoleConst.WEREWOLF)
        alive_werewolf_ids = list({p.id for p in alive_werewolves})
        potential_targets = self.state.alive_players_by_team(Team.VILLAGERS)  # Target non-werewolves

        if alive_werewolves:
            data = RequestWerewolfVotingDataEntry(
                valid_targets=[f"{p.id}" for p in potential_targets],
                alive_werewolve_player_ids=[f"{p.id}" for p in alive_werewolves],
                voting_protocol_name=self.night_voting.__class__.__name__,
                voting_protocol_rule=self.night_voting.voting_rule,
                action_json_schema=json.dumps(VoteAction.model_json_schema())
            )
            self.state.add_history_entry(
                description=f"Wake up Werewolves. Your fellow alive werewolves are: {data.alive_werewolve_player_ids}. "
                            f"Choose one target player to eliminate tonight.The voting rule ({data.voting_protocol_name}): {data.voting_protocol_rule}."
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
        for ww_voter_id in self.night_voting.get_next_voters():  # Should give all WWs if simultaneous
            if ww_voter_id not in self._action_queue: self._action_queue.append(ww_voter_id)

        # state transition 
        self.detailed_phase = DetailedPhase.NIGHT_AWAIT_ACTIONS
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
                        target_id=action.target_id
                    )
                    self.state.add_history_entry(
                        description=f'Player "{actor_id}", you chose to heal player "{action.target_id}".',
                        entry_type=HistoryEntryType.HEAL_ACTION,
                        public=False,
                        visible_to=[actor_id],
                        data=data
                    )
                    self._night_save_queue.append(action.target_id)
                elif player.role.name == RoleConst.SEER and isinstance(action, InspectAction):

                    action_data = SeerInspectActionDataEntry(
                        actor_id=actor_id,
                        target_id=action.target_id
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

        # Process werewolf votes from any received actions
        for actor_id, action in player_actions.items():
            player = self.state.get_player_by_id(actor_id)
            if player and player.alive and player.role.name == RoleConst.WEREWOLF:
                self.night_voting.collect_vote(action, self.state)

        if self.night_step == 1:
            self.night_step += 1  # Increment so Doctor/Seer actions aren't re-processed if WW voting is sequential

        self._action_queue.clear()  # Prepare for next set of actors or end of night

        if not self.night_voting.done():
            # Werewolf voting is not complete (e.g., sequential voting)
            next_ww_voters = self.night_voting.get_next_voters()
            for voter_id in next_ww_voters:
                if voter_id not in self._action_queue: self._action_queue.append(voter_id)

            # Re-prompt only the werewolves whose turn it is now
            alive_werewolves_still_to_vote = [p for p in self.state.alive_players_by_role(RoleConst.WEREWOLF) if
                                              p.id in self._action_queue]
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
                        save_data = DoctorSaveDataEntry(saved_player_id=werewolf_target_id)
                        self.state.add_history_entry(
                            description=f"Last night, player \"{werewolf_target_id}\" was attacked but was saved "
                                        f"by a Doctor! No one was eliminated.",
                            entry_type=HistoryEntryType.ACTION_RESULT,
                            public=False,
                            data=save_data,
                            visible_to=[]
                        )
                        # public info cannot reveal who's attacked and who's saved
                        self.state.add_history_entry(
                            description=f"Last night, No one was eliminated.",
                            entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                            public=True
                        )
                    else:
                        original_role_name = werewolf_target_player.role.name.value
                        self.state.eliminate_player(werewolf_target_id)
                        data = WerewolfNightEliminationDataEntry(
                            eliminated_player_id=werewolf_target_id,
                            eliminated_player_role_name=original_role_name,
                        )
                        self.state.add_history_entry(
                            description=f'Last night, player "{werewolf_target_id}" was eliminated by werewolves. '
                                        f'Their role was a "{original_role_name}".',
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
            self._night_save_queue = []
            if not self.is_game_over():
                self.detailed_phase = DetailedPhase.DAY_START

    def _handle_day_start(self, player_actions: Dict[str, Action]):
        self._action_queue.clear()
        self.state.day_count += 1
        self.state.phase = Phase.DAY
        self.night_step = 0  # Reset night step counter

        self.state.add_history_entry(
            description=f"Day {self.state.day_count} begins.",
            entry_type=HistoryEntryType.PHASE_CHANGE,
            public=True
        )

        self.state.add_history_entry(
            description=f"Villagers, let's discuss who to exile today. The discussion rule is: {self.discussion.discussion_rule}",
            entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
            public=True,
            data={'discussion_rule': self.discussion.discussion_rule}
        )

        self.discussion.begin(self.state)
        # Initial speakers for discussion/bidding
        current_speakers = self.discussion.speakers_for_tick(self.state)
        self._action_queue.extend(s_id for s_id in current_speakers if s_id not in self._action_queue)
        if self._action_queue:  # Only prompt if there are speakers
            self.discussion.prompt_speakers_for_tick(self.state, self._action_queue)

        self.detailed_phase = DetailedPhase.DAY_DISCUSSION_AWAIT_CHAT
        # If _action_queue is empty here (e.g. discussion protocol immediately ends),
        # Env will call advance({}) and _handle_day_discussion_await_chat will transition.

    def _handle_day_discussion_await_chat(self, player_actions: Dict[str, Action]):
        # current_speakers_last_tick refers to who was in _action_queue for the player_actions received
        current_speakers_last_tick = list(self._action_queue)  # Who was expected to act
        self._action_queue.clear()

        # The discussion protocol processes actions.
        # player_actions are from current_speakers_last_tick
        # The protocol needs to know who was expected to speak when these actions were generated.
        self.discussion.process_actions(list(player_actions.values()), current_speakers_last_tick, self.state)

        if self.discussion.is_discussion_over(self.state):
            self.state.add_history_entry(
                description="Daytime activity (discussion/bidding) has concluded. Moving to day vote.",
                entry_type=HistoryEntryType.PHASE_CHANGE,
                public=True
            )
            self.discussion.reset()
            alive_players = self.state.alive_players()
            self.day_voting.begin_voting(self.state, alive_players, alive_players)
            self.detailed_phase = DetailedPhase.DAY_VOTING_AWAIT
            self.state.add_history_entry(
                description=f"Voting phase begins.\nRule: {self.day_voting.voting_rule}",
                entry_type=HistoryEntryType.PHASE_CHANGE,
                public=True
            )
            next_voters_ids = self.day_voting.get_next_voters()
            self._action_queue.extend(v_id for v_id in next_voters_ids if v_id not in self._action_queue)
            if self._action_queue:
                for voter_id in self._action_queue:  # Prompt only the current batch of voters
                    player = self.state.get_player_by_id(voter_id)
                    if player and player.alive:
                        prompt = self.day_voting.get_voting_prompt(self.state, voter_id)
                        self.state.add_history_entry(
                            description=prompt, entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                            public=False, visible_to=[voter_id]
                        )
        else:
            # Discussion not over, get next speakers
            next_speakers = self.discussion.speakers_for_tick(self.state)
            self._action_queue.extend(s_id for s_id in next_speakers if s_id not in self._action_queue)
            if self._action_queue:  # Prompt if there are speakers for the next tick
                self.discussion.prompt_speakers_for_tick(self.state, self._action_queue)
            # Stay in DAY_DISCUSSION_AWAIT_CHAT

    def _handle_day_voting_await(self, player_actions: Dict[str, Action]):
        # Actions in player_actions are from the voters queued in the previous step.
        for actor_id, action in player_actions.items():
            # Ensure the action is from an expected voter (though Env should filter)
            # and is a valid type for voting protocol.
            # The collect_vote method in the protocol should handle validation.
            self.day_voting.collect_vote(action, self.state)

        self._action_queue.clear()  # Clear previous voters

        if self.day_voting.done():
            exiled_player_id = self.day_voting.get_elected()
            if exiled_player_id:
                exiled_player = self.state.get_player_by_id(exiled_player_id)
                if exiled_player:
                    original_role_name = exiled_player.role.name.value
                    self.state.eliminate_player(exiled_player_id)
                    data = DayExileElectedDataEntry(
                        elected_player_id=exiled_player_id,
                        elected_player_role_name=original_role_name,
                        elected_player_team_name=exiled_player.role.team.value
                    )
                    self.state.add_history_entry(
                        description=f'"{exiled_player_id}" in team {data.elected_player_team_name} is exiled by vote. The player is a {original_role_name}.',
                        # Keep description
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
            if not self.is_game_over():
                self.detailed_phase = DetailedPhase.NIGHT_START
                self.state.phase = Phase.NIGHT
            # _action_queue is clear; NIGHT_START will populate it.
        else:
            # Voting is not done, prompt next voters
            next_voter_ids = self.day_voting.get_next_voters()
            self._action_queue.extend(v_id for v_id in next_voter_ids if v_id not in self._action_queue)
            if self._action_queue:
                for voter_id in self._action_queue:  # Prompt only the current batch
                    player = self.state.get_player_by_id(voter_id)
                    if player and player.alive:
                        prompt = self.day_voting.get_voting_prompt(self.state, voter_id)
                        self.state.add_history_entry(
                            description=prompt,
                            entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                            public=False,
                            visible_to=[voter_id]
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
        else:
            winner_team = Team.WEREWOLVES.value
            winner_message = "Game Over: Werewolves Win!"
            reason = f"Reason: len(werewolves) >= len(villagers). Final counts: len(werewolves)={len(wolves)}, len(villagers)={len(villagers)})."
            scores = {p.id: 1 for p in self.state.get_players_by_team(team=Team.WEREWOLVES)}
            scores.update({p.id: 0 for p in self.state.get_players_by_team(team=Team.VILLAGERS)})

        data = GameEndResultsDataEntry(
            winner_team=winner_team,
            scores=scores,
            reason=reason,
            survivors_until_last_round_and_role={p.id: p.role.name for p in self.state.alive_players()},
            all_players_and_role={p.id: p.role.name for p in self.state.players}
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
