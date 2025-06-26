from enum import Enum
from typing import List, Dict, Tuple

from .actions import (
    Action, VoteAction,
    HealAction, InspectAction, ChatAction
)
from .protocols import (
    DiscussionProtocol, VotingProtocol
)
from .roles import Player, Team, Phase, RoleConst
from .states import GameState, HistoryEntry, HistoryEntryType


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
                 night_voting: VotingProtocol):
        self.state = state
        self.discussion = discussion
        self.day_voting = day_voting
        self.night_voting = night_voting

        self._player_history_cursors: Dict[str, Tuple[int, int]] = {
            p.id: (0, 0) for p in self.state.players
        }

        # Detailed phase management
        self.detailed_phase: DetailedPhase = DetailedPhase.NIGHT_START
        assert state.phase == Phase.NIGHT

        self._active_night_roles_queue: List[Player] = []
        self._night_save_queue: List[str] = []
        self._action_queue: List[str] = [] # Player IDs expected to act

        # below is the state transition function table
        # each transition function has the signature tr_func(actions: List[Action]) where the input is a list of actions
        # with the length the same as the number of agents

        self._phase_handlers = {
            DetailedPhase.NIGHT_START: self._handle_night_start,
            DetailedPhase.NIGHT_AWAIT_ACTIONS: self._handle_night_await_actions,
            DetailedPhase.DAY_START: self._handle_day_start,
            DetailedPhase.DAY_DISCUSSION_AWAIT_CHAT: self._handle_day_discussion_await_chat,
            DetailedPhase.DAY_VOTING_AWAIT: self._handle_day_voting_await,
            DetailedPhase.GAME_OVER: self._handle_game_over,
        }

        self.night_step = 0

    def get_active_player_ids(self) -> List[str]:
        return list(self._action_queue) # Return a copy

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
                updated_cursor_idx_in_day = current_processing_idx # Next index to read

        return newly_visible_entries, (updated_cursor_day, updated_cursor_idx_in_day)

    def update_player_cursor(self, player_id: str, new_cursor: Tuple[int, int]):
        """
        Updates the history cursor for a given player.
        """
        self._player_history_cursors[player_id] = new_cursor

    def advance(self, player_actions: Dict[str, Action]):
        if self.is_game_over() and self.detailed_phase != DetailedPhase.GAME_OVER:
            self.detailed_phase = DetailedPhase.GAME_OVER
            self._determine_and_log_winner()
            return

        handler = self._phase_handlers.get(self.detailed_phase) # type: ignore
        if handler:
            handler(player_actions)
        else:
            # Consider logging an error or raising an exception for unhandled phase
            print(f"Warning: No handler for detailed_phase {self.detailed_phase}")

        # Check game over condition again, as handlers might change the state to game over
        if self.is_game_over() and self.detailed_phase != DetailedPhase.GAME_OVER:
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
        self.state.add_history_entry(
            description="Wake up Doctor. Who would you like to save?",
            entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
            public=False,
            visible_to={str(p.id) for p in self.state.alive_players() if p.role.name == RoleConst.DOCTOR}
        )
        for p in self.state.alive_players_by_role(RoleConst.DOCTOR):
            if p.id not in self._action_queue: self._action_queue.append(p.id)

        # announce await action to seer
        self.state.add_history_entry(
            description="Wake up Seer. Who would you like to see their true role?",
            entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
            public=False,
            visible_to={str(p.id) for p in self.state.alive_players() if p.role.name == RoleConst.SEER}
        )
        for p in self.state.alive_players_by_role(RoleConst.SEER):
            if p.id not in self._action_queue: self._action_queue.append(p.id)

        # initialize werewolves voting
        alive_werewolves = self.state.alive_players_by_role(RoleConst.WEREWOLF)
        alive_werewolf_ids = {p.id for p in alive_werewolves}
        potential_targets = self.state.alive_players_by_team(Team.VILLAGERS)  # Target non-werewolves

        if alive_werewolves:
            werewolf_team_list_str = ", ".join([f"{p.id}" for p in alive_werewolves])
            target_options_str = ", ".join([f"{p.id} ({p.role.name})" for p in potential_targets])
            self.state.add_history_entry(
                description=f"Wake up Werewolves ({werewolf_team_list_str}). Your fellow werewolves are: {werewolf_team_list_str}. "
                            f"The voting rules are: {self.night_voting.voting_rule}."
                            f"Who would you like to eliminate tonight? Options: {target_options_str}",
                entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                public=False,
                visible_to=alive_werewolf_ids
            )

        self.night_voting.begin_voting(
            state=self.state,
            alive_voters=alive_werewolves,
            potential_targets=potential_targets
        )
        for ww_voter_id in self.night_voting.get_next_voters(): # Should give all WWs if simultaneous
            if ww_voter_id not in self._action_queue: self._action_queue.append(ww_voter_id)

        # state transition 
        self.detailed_phase = DetailedPhase.NIGHT_AWAIT_ACTIONS
        self.night_step = 1 # Mark that initial night roles (doc, seer) + first WW vote are expected

    def _handle_night_await_actions(self, player_actions: Dict[str, Action]):
        # Process actions from Doctor, Seer (if it's the first step of night actions)
        # and Werewolves (always processed if they send votes)

        if self.night_step == 1:
            for actor_id, action in player_actions.items():
                player = self.state.get_player_by_id(actor_id)
                if not player or not player.alive:
                    continue

                if player.role.name == RoleConst.DOCTOR and isinstance(action, HealAction):
                    self._night_save_queue.append(action.target_id)
                    # Optional: Log Doctor's choice privately or publicly if desired
                    # self.state.add_history_entry(f"P{actor_id} (Doctor) chose to protect P{action.target_id}.", ...)
                elif player.role.name == RoleConst.SEER and isinstance(action, InspectAction):
                    target_player = self.state.get_player_by_id(action.target_id)
                    if target_player: # Ensure target exists
                        self.state.add_history_entry(
                            description=f"You inspected P{target_player.id}. They are a {target_player.role.name} ({target_player.role.team.value}).",
                            entry_type=HistoryEntryType.ACTION_RESULT,
                            public=False,
                            visible_to=[actor_id],
                            data={"target_id": target_player.id, "target_role_name": target_player.role.name, "target_team": target_player.role.team.value}
                        )
        
        # Process werewolf votes from any received actions
        for actor_id, action in player_actions.items():
            player = self.state.get_player_by_id(actor_id)
            if player and player.alive and player.role.name == RoleConst.WEREWOLF and isinstance(action, VoteAction):
                self.night_voting.collect_vote(action, self.state)

        if self.night_step == 1:
            self.night_step += 1 # Increment so Doctor/Seer actions aren't re-processed if WW voting is sequential

        self._action_queue.clear() # Prepare for next set of actors or end of night

        if not self.night_voting.done():
            # Werewolf voting is not complete (e.g., sequential voting)
            next_ww_voters = self.night_voting.get_next_voters()
            for voter_id in next_ww_voters:
                if voter_id not in self._action_queue: self._action_queue.append(voter_id)
            
            # Re-prompt only the werewolves whose turn it is now
            alive_werewolves_still_to_vote = [p for p in self.state.alive_players_by_role(RoleConst.WEREWOLF) if p.id in self._action_queue]
            if alive_werewolves_still_to_vote:
                potential_targets_str = ", ".join([f"P{p.id}({p.role.name})" for p in self.night_voting.get_valid_targets()])
                current_tally_str = str(self.night_voting.get_current_tally_info(self.state))
                for ww_voter in alive_werewolves_still_to_vote:
                    prompt = self.night_voting.get_voting_prompt(self.state, ww_voter.id) # Use protocol's prompt
                    # prompt = f"P{ww_voter.id}, it's your turn to vote for elimination. Tally: {current_tally_str}. Options: {potential_targets_str}"
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
            
            if werewolf_target_id:
                werewolf_target_player = self.state.get_player_by_id(werewolf_target_id)
                if werewolf_target_player: 
                    if werewolf_target_id in self._night_save_queue:
                        self.state.add_history_entry(
                            description=f"Last night, P{werewolf_target_id} ({werewolf_target_player.role.name}) was attacked by werewolves but saved by the Doctor!", # Keep description for human readability
                            entry_type=HistoryEntryType.ACTION_RESULT,
                            public=True,
                            data={
                                "action_type": "heal_outcome",
                                "saved_player_id": werewolf_target_id,
                                "saved_player_role_name": werewolf_target_player.role.name,
                                "saved_by_role": RoleConst.DOCTOR.value,
                                "attacked_by_team": Team.WEREWOLVES.value,
                                "outcome": "saved"
                            }
                        )
                    else: 
                        original_role_name = werewolf_target_player.role.name 
                        self.state.eliminate_player(werewolf_target_id)
                        self.state.add_history_entry(
                            description=f"Last night, P{werewolf_target_id} was eliminated by werewolves. They were a {original_role_name}.", # Keep description
                            entry_type=HistoryEntryType.ELIMINATION,
                            public=True,
                            data={
                                "eliminated_player_id": werewolf_target_id,
                                "eliminated_player_role_name": original_role_name,
                                "elimination_reason": "werewolves",
                                "saved_by_doctor": False
                            }
                        )
                else: 
                    self.state.add_history_entry(
                        description=f"Last night, werewolves targeted P{werewolf_target_id}, but this player could not be found. No one was eliminated by werewolves.",
                        entry_type=HistoryEntryType.ERROR,
                        public=True
                    )
            else: 
                self.state.add_history_entry(
                    description="Last night, the werewolves did not reach a consensus (or no valid target was chosen). No one was eliminated by werewolves.",
                    entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT, # This could also be a VOTE_RESULT type
                    data={
                        "outcome": "no_elimination",
                        "reason": "no_consensus_werewolves"
                    },
                    public=True
                )

            self._night_save_queue = []
            if not self.is_game_over():
                self.detailed_phase = DetailedPhase.DAY_START
            # _action_queue is already clear, DAY_START will populate it.

    def _handle_day_start(self, player_actions: Dict[str, Action]):
        self._action_queue.clear()
        self.state.day_count += 1
        self.state.phase = Phase.DAY
        self.night_step = 0 # Reset night step counter

        self.state.add_history_entry(
            description=f"Day {self.state.day_count} begins.",
            entry_type=HistoryEntryType.PHASE_CHANGE,
            public=True
        )

        self.discussion.begin(self.state)
        # Initial speakers for discussion/bidding
        current_speakers = self.discussion.speakers_for_tick(self.state)
        self._action_queue.extend(s_id for s_id in current_speakers if s_id not in self._action_queue)
        if self._action_queue: # Only prompt if there are speakers
            self.discussion.prompt_speakers_for_tick(self.state, self._action_queue)
        
        self.detailed_phase = DetailedPhase.DAY_DISCUSSION_AWAIT_CHAT
        # If _action_queue is empty here (e.g. discussion protocol immediately ends),
        # Env will call advance({}) and _handle_day_discussion_await_chat will transition.

    def _handle_day_discussion_await_chat(self, player_actions: Dict[str, Action]):
        # current_speakers_last_tick refers to who was in _action_queue for the player_actions received
        current_speakers_last_tick = list(self._action_queue) # Who was expected to act
        self._action_queue.clear()

        # The discussion protocol processes actions.
        # player_actions are from current_speakers_last_tick
        # The protocol needs to know who was expected to speak when these actions were generated.
        self.discussion.process_actions(list(player_actions.values()), current_speakers_last_tick, self.state)

        if self.discussion.is_discussion_over(self.state):
            self.state.add_history_entry(
                description="Daytime activity (discussion/bidding) has concluded. Moving to vote.",
                entry_type=HistoryEntryType.PHASE_CHANGE,
                public=True
            )
            alive_players = self.state.alive_players()
            self.day_voting.begin_voting(self.state, alive_players, alive_players)
            self.detailed_phase = DetailedPhase.DAY_VOTING_AWAIT
            self.state.add_history_entry(
                description=f"Voting phase begins. Rule: {self.day_voting.voting_rule}",
                entry_type=HistoryEntryType.PHASE_CHANGE,
                public=True
            )
            next_voters_ids = self.day_voting.get_next_voters()
            self._action_queue.extend(v_id for v_id in next_voters_ids if v_id not in self._action_queue)
            if self._action_queue:
                for voter_id in self._action_queue: # Prompt only the current batch of voters
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
            if self._action_queue: # Prompt if there are speakers for the next tick
                self.discussion.prompt_speakers_for_tick(self.state, self._action_queue)
            # Stay in DAY_DISCUSSION_AWAIT_CHAT

    def _determine_and_log_winner(self):
        # Check if a GAME_END entry for the current day_count already exists
        day_history = self.state.history.get(self.state.day_count, [])
        if any(entry.entry_type == HistoryEntryType.GAME_END for entry in day_history):
            return # Winner already logged for this day count

        winner_message = "Game Over: Undetermined."
        winner_data = {"winner_team": "Undetermined", "reason": "unknown"}

        wolves = [p for p in self.state.alive_players() if p.role.team == Team.WEREWOLVES]
        villagers = [p for p in self.state.alive_players() if p.role.team == Team.VILLAGERS]

        if not wolves and villagers: 
            winner_message = "Game Over: Villagers Win!"
            winner_data = {"winner_team": Team.VILLAGERS.value, "reason": "no_werewolves_left"}
        elif wolves and not villagers: 
            winner_message = "Game Over: Werewolves Win!"
            winner_data = {"winner_team": Team.WEREWOLVES.value, "reason": "no_villagers_left"}
        elif wolves and len(wolves) >= len(villagers): 
            winner_message = "Game Over: Werewolves Win!"
            winner_data = {"winner_team": Team.WEREWOLVES.value, "reason": "werewolves_majority"}
        elif not wolves and not villagers: 
            winner_message = "Game Over: Draw! No one is left."
            winner_data = {"winner_team": "Draw", "reason": "no_one_left"}
        
        self.state.add_history_entry(
            description=winner_message,
            entry_type=HistoryEntryType.GAME_END,
            public=True,
            data=winner_data
        )

    def _handle_day_voting_await(self, player_actions: Dict[str, Action]):
        # Actions in player_actions are from the voters queued in the previous step.
        for actor_id, action in player_actions.items():
            # Ensure the action is from an expected voter (though Env should filter)
            # and is a valid type for voting protocol.
            # The collect_vote method in the protocol should handle validation.
            if isinstance(action, (VoteAction, ChatAction)): # TODO: ChatAction is a bug, should be NoOpAction
                self.day_voting.collect_vote(action, self.state)

        self._action_queue.clear() # Clear previous voters

        if self.day_voting.done():
            exiled_player_id = self.day_voting.get_elected()
            if exiled_player_id:
                exiled_player = self.state.get_player_by_id(exiled_player_id)
                if exiled_player: 
                    original_role_name = exiled_player.role.name
                    self.state.eliminate_player(exiled_player_id)
                    self.state.add_history_entry(
                        description=f"P{exiled_player_id} ({original_role_name}) was exiled by vote. They were a {original_role_name}.", # Keep description
                        entry_type=HistoryEntryType.ELIMINATION,
                        public=True,
                        data={
                            "eliminated_player_id": exiled_player_id,
                            "eliminated_player_role_name": original_role_name,
                            "elimination_reason": "vote",
                            "saved_by_doctor": None # Not applicable for day eliminations
                        }
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

            if not self.is_game_over():
                self.detailed_phase = DetailedPhase.NIGHT_START
                self.state.phase = Phase.NIGHT
            # _action_queue is clear; NIGHT_START will populate it.
        else:
            # Voting is not done, prompt next voters
            next_voter_ids = self.day_voting.get_next_voters()
            self._action_queue.extend(v_id for v_id in next_voter_ids if v_id not in self._action_queue)
            if self._action_queue:
                for voter_id in self._action_queue: # Prompt only the current batch
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

    def is_game_over(self) -> bool:
        if self.detailed_phase == DetailedPhase.GAME_OVER:
            return True

        wolves = [p for p in self.state.alive_players() if p.role.team == Team.WEREWOLVES]
        villagers = [p for p in self.state.alive_players()
                     if p.role.team == Team.VILLAGERS]
        # Game ends if no werewolves are left (villagers win)
        # OR if number of werewolves is equal to or greater than villagers (werewolves win)
        # OR if no villagers are left (werewolves win, covered by parity or if only WWs remain)
        if not wolves and villagers: return True # Villagers win
        if wolves and len(wolves) >= len(villagers): return True # Werewolves win by numbers or if no villagers
        if wolves and not villagers: return True # Werewolves win (all villagers eliminated)
        
        return False # Game continues
    
    def _handle_game_over(self, actions: Dict[str, Action]):
        self._action_queue.clear()

        # resolve werewolves action
        for action, player in zip(actions, self.state.players):
            if player.alive and player.role == RoleConst.WEREWOLF and isinstance(action, VoteAction):
                self.night_voting.collect_vote(action, self.state)

        if not self.night_voting.done():
            alive_werewolves = self.state.alive_players_by_role(RoleConst.WEREWOLF)
            werewolf_team_list_str = ", ".join([f"{p.id}" for p in alive_werewolves])

            for voter in self.night_voting.get_next_voters():
                self.state.add_history_entry(
                    description=f"Wake up Werewolves ({werewolf_team_list_str}). Your fellow werewolves are: {werewolf_team_list_str}. "
                                f"The voting rules are: {self.night_voting.voting_rule}."
                                f"The tally is: {self.night_voting.get_current_tally_info(self.state)}"
                                f"Who would you like to eliminate tonight? Options: {self.night_voting.get_valid_targets()}",
                    entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                    public=False,
                    visible_to=[voter]
                )
        else: # All werewolf votes are in, or voting is otherwise complete
            # Resolve werewolf elimination and doctor save
            werewolf_target_id = self.night_voting.get_elected() # str | None
            
            # Log night events (elimination/save)
            if werewolf_target_id:
                werewolf_target_player = self.state.get_player_by_id(werewolf_target_id)
                if werewolf_target_player: # Player exists
                    if werewolf_target_id in self._night_save_queue:
                        self.state.add_history_entry(
                            description=f"Last night, P{werewolf_target_id} ({werewolf_target_player.role.name}) was attacked by werewolves but saved by the Doctor!", # Keep description
                            entry_type=HistoryEntryType.ACTION_RESULT,
                            public=True,
                            data={
                                "action_type": "heal_outcome",
                                "saved_player_id": werewolf_target_id,
                                "saved_player_role_name": werewolf_target_player.role.name,
                                "saved_by_role": RoleConst.DOCTOR.value,
                                "attacked_by_team": Team.WEREWOLVES.value,
                                "outcome": "saved"
                            }
                        )
                    else: # Not saved
                        original_role_name = werewolf_target_player.role.name 
                        self.state.eliminate_player(werewolf_target_id)
                        self.state.add_history_entry(
                            description=f"Last night, P{werewolf_target_id} was eliminated by werewolves. They were a {original_role_name}.", # Keep description
                            entry_type=HistoryEntryType.ELIMINATION,
                            public=True,
                            data={
                                "eliminated_player_id": werewolf_target_id,
                                "eliminated_player_role_name": original_role_name,
                                "elimination_reason": "werewolves",
                                "saved_by_doctor": False
                            }
                        )
                else: # Target ID from vote, but player not found
                    self.state.add_history_entry(
                        description=f"Last night, werewolves targeted P{werewolf_target_id}, but this player could not be found. No one was eliminated by werewolves.",
                        entry_type=HistoryEntryType.ERROR,
                        public=True
                    )
            else: # No one elected by werewolves
                self.state.add_history_entry(
                    description="Last night, the werewolves did not reach a consensus (or no valid target was chosen). No one was eliminated by werewolves.",
                    entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT, # This could also be a VOTE_RESULT type
                    data={
                        "outcome": "no_elimination",
                        "reason": "no_consensus_werewolves"
                    },
                    public=True
                )

            # Check for game over condition AFTER night eliminations
            if self.is_game_over():
                self.detailed_phase = DetailedPhase.GAME_OVER
            else:
                # If game is not over, then transition to day
                self.detailed_phase = DetailedPhase.DAY_START
            
            self._night_save_queue = [] # Reset for the next night
