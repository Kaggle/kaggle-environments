import random
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple

import gymnasium as gym
from gymnasium import spaces

from .actions import (
    Action, VoteAction,
    HealAction, InspectAction, ChatAction
)
from .protocols import (
    DiscussionProtocol, VotingProtocol,
    RoundRobinDiscussion, SimultaneousMajority
)
from .roles import Player, Team, Phase, RoleConst # Added HistoryEntry
from .states import GameState, HistoryEntry, HistoryEntryType


# NEW Enums for detailed phase management
class DetailedPhase(Enum):
    # Night Phases
    NIGHT_START = "NIGHT_START"
    NIGHT_AWAIT_ACTIONS = "NIGHT_AWAIT_ACTIONS"
    # Day Phases
    DAY_START = "DAY_START" # Transition state to setup day discussion/bidding
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
        self._player_history_cursors[player_id] = (updated_cursor_day, updated_cursor_idx_in_day)

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
                            description=f"Last night, P{werewolf_target_id} ({werewolf_target_player.role.name}) was attacked by werewolves but saved by the Doctor!",
                            entry_type=HistoryEntryType.ACTION_RESULT,
                            public=True
                        )
                    else: 
                        original_role_name = werewolf_target_player.role.name 
                        self.state.eliminate_player(werewolf_target_id)
                        self.state.add_history_entry(
                            description=f"Last night, P{werewolf_target_id} was eliminated by werewolves. They were a {original_role_name}.",
                            entry_type=HistoryEntryType.ELIMINATION,
                            public=True
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
                    entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                    public=True
                )

            self._night_save_queue = [] 
            if self.is_game_over():
                self.detailed_phase = DetailedPhase.GAME_OVER
            else:
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

        wolves = [p for p in self.state.alive_players() if p.role.team == Team.WEREWOLVES]
        villagers = [p for p in self.state.alive_players() if p.role.team == Team.VILLAGERS]

        winner_message = "Game Over: Undetermined."
        if not wolves and villagers: 
            winner_message = "Game Over: Villagers Win!"
        elif wolves and not villagers: 
            winner_message = "Game Over: Werewolves Win!"
        elif wolves and len(wolves) >= len(villagers): 
            winner_message = "Game Over: Werewolves Win!"
        elif not wolves and not villagers: 
            winner_message = "Game Over: Draw! No one is left."
        
        self.state.add_history_entry(
            description=winner_message,
            entry_type=HistoryEntryType.GAME_END,
            public=True
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
                        description=f"P{exiled_player_id} ({original_role_name}) was exiled by vote. They were a {original_role_name}.",
                        entry_type=HistoryEntryType.ELIMINATION,
                        public=True
                    )
            else:
                self.state.add_history_entry(
                    description="The vote resulted in no exile (e.g., a tie, no majority, or all abstained).",
                    entry_type=HistoryEntryType.VOTE_RESULT,
                    public=True
                )

            if self.is_game_over():
                self.detailed_phase = DetailedPhase.GAME_OVER
            else:
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
                            description=f"Last night, P{werewolf_target_id} ({werewolf_target_player.role.name}) was attacked by werewolves but saved by the Doctor!",
                            entry_type=HistoryEntryType.ACTION_RESULT,
                            public=True
                        )
                    else: # Not saved
                        original_role_name = werewolf_target_player.role.name 
                        self.state.eliminate_player(werewolf_target_id)
                        self.state.add_history_entry(
                            description=f"Last night, P{werewolf_target_id} was eliminated by werewolves. They were a {original_role_name}.",
                            entry_type=HistoryEntryType.ELIMINATION,
                            public=True
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
                    entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                    public=True
                )

            # Check for game over condition AFTER night eliminations
            if self.is_game_over():
                self.detailed_phase = DetailedPhase.GAME_OVER
            else:
                # If game is not over, then transition to day
                self.detailed_phase = DetailedPhase.DAY_START
            
            self._night_save_queue = [] # Reset for the next night


class WerewolfEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    MAX_VISIBLE_HISTORY_ITEMS = 20 # Max number of history items in observation

    def __init__(self, num_players=8, role_distribution=None, seed=None):
        self.rng = random.Random(seed)
        self.num_players = num_players
        # self.player_ids = [str(i) for i in range(num_players)] # Store player IDs
        # self.role_distribution = role_distribution or [WerewolfRole(), VillagerRole()] # Simplified
        self._setup_spaces()
        self.state: Optional[GameState] = None # type: ignore
        self._player_full_visible_history_cache: Dict[str, List[HistoryEntry]] = {}
        self.moderator: Optional[Moderator] = None

    def _setup_spaces(self):
        # All agents share same action space: select player or noop (-1)
        # This action space might need to be more complex if actions are structured dicts
        self.action_space = spaces.Discrete(self.num_players + 1)

        # Define the observation space for a single active player
        player_status_item_space = spaces.Dict({
            "id": spaces.Text(max_length=32),
            "is_alive": spaces.Discrete(2),
            # "role_revealed_to_me": spaces.Text(max_length=32) # Future enhancement
        })

        # Create a tuple space for all_players_status, assuming num_players is fixed at init
        # If num_players can change, this needs to be a Sequence, or handle padding.
        all_players_status_tuple = tuple(player_status_item_space for _ in range(self.num_players))

        single_player_observation_space = spaces.Dict({
            "player_id": spaces.Text(max_length=32),
            "role": spaces.Text(max_length=32),
            "team": spaces.Text(max_length=32),
            "is_alive": spaces.Discrete(2),
            "day_count": spaces.Box(low=0, high=100, shape=(), dtype=int),
            "game_phase": spaces.Text(max_length=16),
            "detailed_game_phase": spaces.Text(max_length=32),
            "all_players_status": spaces.Tuple(all_players_status_tuple),
            "visible_history_log": spaces.Tuple(tuple(spaces.Text(max_length=512) for _ in range(self.MAX_VISIBLE_HISTORY_ITEMS))),
            "action_prompt": spaces.Text(max_length=512)
        })

        self.observation_space = spaces.Dict({
            str(i): single_player_observation_space for i in range(self.num_players)
        })

    # Gymnasium API -------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)

        # Simplified role distribution for example purposes
        from .roles import Werewolf as WerewolfRole, Villager as VillagerRole, Seer as SeerRole, \
            Doctor as DoctorRole  # Avoid name clash
        available_roles = [WerewolfRole(), VillagerRole(), SeerRole(), DoctorRole(), VillagerRole(), VillagerRole(),
                           WerewolfRole(), VillagerRole()]

        players = [
            Player(id=str(i), role=self.rng.choice(available_roles[:self.num_players] or [VillagerRole()]))
            # Ensure enough roles
            for i in range(self.num_players)
        ]
        # Ensure at least one werewolf if possible for a typical game
        if self.num_players > 2 and not any(p.role.name == "Werewolf" for p in players):
            if players:  # Ensure players list is not empty
                players[0].role = WerewolfRole(descriptions="A werewolf.")  # Ensure at least one werewolf

        self.state = GameState(players=players, history=[])


        self._player_full_visible_history_cache.clear()
        # Setup protocols for the Moderator
        discussion_protocol = RoundRobinDiscussion()
        day_voting_protocol = SimultaneousMajority()
        # Night voting (werewolf elimination) also needs a VotingProtocol instance.
        # For simplicity, let's assume werewolves also use a SimultaneousMajority for their internal decision for now.
        # This would typically be a NightTeamActionProtocol like WerewolfEliminationProtocol.
        # The Moderator's night_voting parameter expects a VotingProtocol.
        night_voting_protocol_for_werewolves = SimultaneousMajority() # Placeholder
        self.moderator = Moderator(self.state, discussion=discussion_protocol, day_voting=day_voting_protocol, night_voting=night_voting_protocol_for_werewolves) # type: ignore
        obs = self._get_observations()
        info = {}
        return obs, info

    def step(self, actions: Dict[str, Any]):  # Agent ID (str) to their action choice (Any, to be parsed into Action)
        # Convert Dict[str, Any] to List[Action] - This needs proper mapping based on current phase
        # For now, assuming actions are already List[Action] or Moderator.advance is adapted
        # This is a placeholder for proper action conversion. `actions` keys are now str.
        
        # --- Placeholder for Action Parsing ---
        # This section needs to convert raw agent outputs (from `actions: Dict[str, Any]`)
        # into structured `Action` objects (e.g., VoteAction, ChatAction)
        # based on the current game phase and what the moderator expects.
        # For now, we'll assume `actions` is already `Dict[str, Action]` for the demo.
        # In a real scenario, this parsing is non-trivial.
        parsed_player_actions: Dict[str, Action] = {}
        if self.moderator and self.state:
            active_ids = self.moderator.get_active_player_ids()
            for agent_id, raw_action in actions.items():
                if agent_id in active_ids:
                    # TODO: Implement robust parsing of `raw_action` into an `Action` object
                    # This is highly dependent on how raw actions are structured and game phase.
                    # Example: if raw_action is an int for voting, convert to VoteAction(target_id=str(raw_action))
                    # For now, if it's already an Action object, use it.
                    if isinstance(raw_action, Action):
                        parsed_player_actions[agent_id] = raw_action
        # --- End Placeholder ---

        if self.moderator:
            self.moderator.advance(parsed_player_actions)
        terminated = self.moderator.is_game_over() if self.moderator else True
        obs = self._get_observations()
        rewards = self._compute_rewards(terminated)
        truncated = False  # no max steps yet
        info = {}
        return obs, rewards, terminated, truncated, info

    # --------------------------------------------------------------------- #
    def _get_observations(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """Return per-agent observations. Active players get a dict, others None."""
        if not self.moderator or not self.state:
            return {str(i): None for i in range(self.num_players)}

        observations: Dict[str, Optional[Dict[str, Any]]] = {p.id: None for p in self.state.players}
        active_player_ids = self.moderator.get_active_player_ids()

        for player in self.state.players:
            if player.id in active_player_ids: # Only generate full obs for active players
                new_history_entries, new_cursor_pos = self.moderator.get_observation(player.id)
                self._player_full_visible_history_cache.setdefault(player.id, []).extend(new_history_entries)
                # Update the moderator's cursor for this player *after* processing the observations
                self.moderator.update_player_cursor(player.id, new_cursor_pos)

                current_player_full_log = self._player_full_visible_history_cache[player.id]
                
                # Get descriptions for the observation, limited to MAX_VISIBLE_HISTORY_ITEMS
                visible_history_descriptions = [
                    entry.description for entry in current_player_full_log[-self.MAX_VISIBLE_HISTORY_ITEMS:]
                ]
                # Pad history if less than MAX_VISIBLE_HISTORY_ITEMS
                visible_history_descriptions.extend([""] * (self.MAX_VISIBLE_HISTORY_ITEMS - len(visible_history_descriptions)))

                latest_prompt = "No specific prompt. It's your turn to act."

                # Find the latest prompt for this player
                for entry in reversed(current_player_full_log): # Search in the player's own cached history
                    if entry.entry_type == HistoryEntryType.MODERATOR_ANNOUNCEMENT and player.id in entry.visible_to:
                        latest_prompt = entry.description
                        break
                    # Consider other prompt-like entries, e.g., public bidding announcements
                    elif entry.entry_type == HistoryEntryType.BIDDING_INFO and self.moderator.detailed_phase == DetailedPhase.DAY_DISCUSSION_AWAIT_CHAT and self.discussion.discussion_rule.startswith("Bidding phase"): # type: ignore
                         latest_prompt = entry.description # Generic bidding prompt
                         break

                obs_data = {
                    "player_id": player.id,
                    "role": player.role.name,
                    "team": player.role.team.value,
                    "is_alive": player.alive,
                    "day_count": self.state.day_count,
                    "game_phase": self.state.phase.value,
                    "detailed_game_phase": self.moderator.detailed_phase.value,
                    "all_players_status": [{"id": p_other.id, "is_alive": p_other.alive} for p_other in self.state.players],
                    "visible_history_log": tuple(visible_history_descriptions), # Ensure it's a tuple
                    "action_prompt": latest_prompt
                }
                # Pad all_players_status if num_players is fixed and space expects fixed tuple
                if len(obs_data["all_players_status"]) < self.num_players:
                    obs_data["all_players_status"].extend(
                        [{"id": "-1", "is_alive": False}] * (self.num_players - len(obs_data["all_players_status"]))
                    )
                obs_data["all_players_status"] = tuple(obs_data["all_players_status"])

                observations[player.id] = obs_data
        return observations

    def _compute_rewards(self, terminated):
        if not terminated:
            return {str(p.id): 0.0 for p in self.state.players} if self.state else {}

        wolves_win = any(p.alive and p.role.team == Team.WEREWOLVES for p in self.state.players) and \
                     len([p for p in self.state.alive_players() if p.role.team == Team.WEREWOLVES]) >= \
                     len([p for p in self.state.alive_players() if p.role.team == Team.VILLAGERS])
        villagers_win = not any(p.alive and p.role.team == Team.WEREWOLVES for p in self.state.players) and \
                        any(p.alive and p.role.team == Team.VILLAGERS for p in self.state.players)

        rewards = {}
        for p in self.state.players:
            if (p.role.team == Team.VILLAGERS and villagers_win) or \
                    (p.role.team == Team.WEREWOLVES and wolves_win):
                rewards[str(p.id)] = 1.0
            elif terminated:  # If game is over and player's team didn't win
                rewards[str(p.id)] = -1.0
            else:  # Game not over or player's team status unclear (e.g. tie, not explicitly handled)
                rewards[str(p.id)] = 0.0
        return rewards

    def render(self, mode="human"):
        if self.state:
            print(
                f"Day {self.state.day_count} — Phase: {self.state.phase.value} (Detailed: {self.moderator.detailed_phase.value if self.moderator else 'N/A'})")
            for p in self.state.players:
                status = "alive" if p.alive else "dead"
                print(f" P{p.id}: {p.role.name} ({p.role.team.value}) [{status}]")
            # print("History:", self.state.history) # Print last 5 history events - history is a dict
            print("-" * 20)
        else:
            print("Game state not initialized.")
