import json
from typing import List, Dict, Type, Sequence, Protocol

from .actions import Action, VoteAction, ChatAction, BidAction
from .base import BaseModerator, PlayerID
from .consts import Phase, Team, RoleConst, PhaseDivider, DetailedPhase
from .night_elimination_manager import NightEliminationManager
from .protocols.base import VotingProtocol, DiscussionProtocol
from .protocols.chat import BiddingDiscussion
from .records import (
    EventName, GameStartDataEntry, GameStartRoleDataEntry, RequestWerewolfVotingDataEntry,
    WerewolfNightEliminationElectedDataEntry, DayExileElectedDataEntry,
    GameEndResultsDataEntry
)
from .roles import Player
from .states import GameState


class ActionQueue:
    """A data structure for managing player ids in action specific queues."""

    def __init__(self):
        self._action_queue: Dict[str, List[PlayerID]] = {}

    def clear(self):
        self._action_queue = {}

    def append(self, action_cls: Type[Action], player_id: PlayerID):
        action_type = action_cls.__name__
        self._action_queue.setdefault(action_type, [])
        if player_id in self._action_queue[action_type]:
            raise ValueError(f'player {player_id} is already in the action queue. ')
        self._action_queue[action_type].append(player_id)

    def extend(self, action_cls: Type[Action], player_ids: Sequence[PlayerID]):
        for player_id in player_ids:
            self.append(action_cls, player_id)

    def get(self, action_cls: Type[Action]) -> List[str]:
        """return a list of player_id for the selected action."""
        return self._action_queue.get(action_cls.__name__, [])

    def get_active_player_ids(self) -> List[PlayerID]:
        all_players = set()
        for players in self._action_queue.values():
            all_players.update(players)
        return list(all_players)


def phase_handler(phase: DetailedPhase):
    """Decorator to register a method as a handler for a specific game phase."""
    def decorator(func):
        setattr(func, '_phase_handler_for', phase)
        return func
    return decorator


class PhaseHandler(Protocol):
    def __call__(self, player_actions: Dict[PlayerID, Action]):
        pass


class Moderator(BaseModerator):
    """Drives the finite-state machine for the game."""

    def __init__(
        self,
        state: GameState,
        discussion: DiscussionProtocol,
        day_voting: VotingProtocol,  # Renamed for clarity
        night_voting: VotingProtocol,
        allow_doctor_self_save: bool = False,  # should be set by doctor role
        reveal_night_elimination_role: bool = True,
        reveal_day_exile_role: bool = True
    ):
        self._state = state
        self.discussion = discussion
        self.day_voting = day_voting
        self.night_voting = night_voting
        self._allow_doctor_self_save = allow_doctor_self_save
        self._set_doctor_self_save()

        self._reveal_night_elimination_role = reveal_night_elimination_role
        self._reveal_day_exile_role = reveal_day_exile_role

        self._active_night_roles_queue: List[Player] = []
        self._night_elimination_manager = NightEliminationManager(self._state, self._reveal_night_elimination_role)
        self._action_queue = ActionQueue()

        # This is for registering role specific event handling
        self._register_player_handlers()

        # below is the state transition function table
        # each transition function has the signature tr_func(actions: List[Action]) where the input is a list of actions
        # with the length the same as the number of agents
        self.detailed_phase = DetailedPhase.NIGHT_START
        self._phase_handlers: Dict[DetailedPhase, PhaseHandler] = {}
        self._register_phase_handlers()

        self._make_initial_announcements()

    @property
    def state(self) -> GameState:
        return self._state

    def _make_initial_announcements(self):
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
        self.doctor_special_msg = "Doctor is allowed to save themselves during night time." if self._allow_doctor_self_save \
            else "Doctor is NOT allowed to save themselves during night time."
        day_exile_reveal_msg = "If a player is exiled in the day, their role will be revealed." if self._reveal_day_exile_role \
            else "If a player is exiled in the day, their role will NOT be revealed."
        night_elimination_reveal_msg = "If a player is eliminated at night, their role will be revealed." \
            if self._reveal_night_elimination_role else "If a player is eliminated at night, their role will NOT be revealed."

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
        self.state.push_event(
            description=description,
            event_name=EventName.MODERATOR_ANNOUNCEMENT,
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
            self.state.push_event(
                description=f'Your player id is "{data.player_id}". Your team is "{data.team}". Your role is "{data.role}".\n'
                            f"The rule of your role: {data.rule_of_role}",
                event_name=EventName.GAME_START,
                public=False,
                visible_to=[player.id],
                data=data
            )

    def _register_phase_handlers(self):
        """Collects all methods decorated with @phase_handler."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, '_phase_handler_for'):
                phase = getattr(attr, '_phase_handler_for')
                self._phase_handlers[phase] = attr

    def _set_doctor_self_save(self):
        for doctor in self.state.get_players_by_role(RoleConst.DOCTOR):
            doctor.role.allow_self_save = self._allow_doctor_self_save

    def _register_player_handlers(self):
        for player in self.state.players:
            for event_name, handlers in player.get_event_handlers(self).items():
                for handler in handlers:
                    self.state.register_event_handler(event_name, handler)

    def request_action(
            self,
            action_cls: Type[Action], player_id: PlayerID, prompt: str, data=None,
            event_name=EventName.MODERATOR_ANNOUNCEMENT
        ):
        """A public method for listeners to add a player to the action queue."""
        self._action_queue.append(action_cls, player_id)
        # Create the corresponding data entry to prompt the player
        self.state.push_event(
            description=prompt,
            event_name=event_name,
            public=False,
            visible_to=[player_id],
            data=data
        )

    def confirm_action(self, player_actions: Dict[PlayerID, Action]):
        for action in player_actions.values():
            # moderator confirming the action with players
            action.push_event(state=self.state)

    def set_next_phase(self, new_detailed_phase: DetailedPhase, add_one_day: bool = False):
        """Note: phase change is not the same as phase start, still need phase start at each block"""
        old_detailed_phase = self.detailed_phase
        self.detailed_phase = new_detailed_phase
        self.state.detailed_phase = new_detailed_phase
        new_phase = Phase.NIGHT
        if new_detailed_phase not in [DetailedPhase.NIGHT_START, DetailedPhase.NIGHT_AWAIT_ACTIONS]:
            new_phase = Phase.DAY
        self.state.phase = new_phase

        if add_one_day:
            self.state.day_count += 1

        self.state.push_event(
            description=f"Transitioning from {old_detailed_phase} to {new_detailed_phase}.",
            event_name=EventName.PHASE_CHANGE,
            public=False
        )

    def get_active_player_ids(self) -> List[PlayerID]:
        return self._action_queue.get_active_player_ids()

    def record_night_save(self, doctor_id: PlayerID, target_id: PlayerID):
        self._night_elimination_manager.record_save(doctor_id, target_id)

    def advance(self, player_actions: Dict[PlayerID, Action]):
        self.confirm_action(player_actions)
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
            self.set_next_phase(DetailedPhase.GAME_OVER)
            self._determine_and_log_winner()

    @phase_handler(DetailedPhase.NIGHT_START)
    def _handle_night_start(self, player_actions: Dict[PlayerID, Action]):
        self._action_queue.clear()
        self.state.add_phase_divider(PhaseDivider.NIGHT_START)
        self.state.push_event(
            description=f"Night {self.state.day_count} begins!",
            event_name=EventName.NIGHT_START,
            public=True
        )
        self.state.add_phase_divider(PhaseDivider.NIGHT_VOTE_START)

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
            self.state.push_event(
                description=f"Wake up Werewolves. Your fellow alive werewolves are: {data.alive_werewolve_player_ids}. "
                            f"Choose one target player to eliminate tonight. The voting rule ({data.voting_protocol_name}): {data.voting_protocol_rule} "
                            f"Who would you like to eliminate tonight? Options: {data.valid_targets}.",
                event_name=EventName.VOTE_REQUEST,
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
        self.set_next_phase(DetailedPhase.NIGHT_AWAIT_ACTIONS)

    @phase_handler(DetailedPhase.NIGHT_AWAIT_ACTIONS)
    def _handle_night_await_actions(self, player_actions: Dict[PlayerID, Action]):
        # Process werewolf votes
        werewolf_voters_expected = self._action_queue.get(VoteAction)
        if werewolf_voters_expected:
            self.night_voting.collect_votes(player_actions, self.state, werewolf_voters_expected)

        self._action_queue.clear()

        if not self.night_voting.done():
            next_ww_voters = self.night_voting.get_next_voters()
            self._action_queue.extend(VoteAction, next_ww_voters)
            vote_action_queue = self._action_queue.get(VoteAction)
            alive_werewolves_still_to_vote = [p for p in self.state.alive_players_by_role(RoleConst.WEREWOLF) if
                                              p.id in vote_action_queue]
            if alive_werewolves_still_to_vote:
                for ww_voter in alive_werewolves_still_to_vote:
                    prompt = self.night_voting.get_voting_prompt(self.state, ww_voter.id)
                    self.state.push_event(
                        description=prompt,
                        event_name=EventName.VOTE_REQUEST,
                        public=False,
                        visible_to=[ww_voter.id]
                    )
            # Stay in NIGHT_AWAIT_ACTIONS
        else:
            werewolf_target_id = self.night_voting.get_elected()

            data = WerewolfNightEliminationElectedDataEntry(elected_target_player_id=werewolf_target_id)
            self.state.push_event(
                description=f'Werewolves elected to eliminate player "{data.elected_target_player_id}".',
                event_name=EventName.VOTE_RESULT,
                public=False,
                visible_to=[p.id for p in self.state.alive_players_by_team(Team.WEREWOLVES)],
                data=data
            )

            self._night_elimination_manager.resolve_elimination(werewolf_target_id)

            self.night_voting.reset()
            self._night_elimination_manager.reset()

            self.state.add_phase_divider(PhaseDivider.NIGHT_VOTE_END)
            self.state.add_phase_divider(PhaseDivider.NIGHT_END)
            if not self.is_game_over():
                self.set_next_phase(DetailedPhase.DAY_START, add_one_day=True)

    @phase_handler(DetailedPhase.DAY_START)
    def _handle_day_start(self, player_actions: Dict[PlayerID, Action]):
        self.state.add_phase_divider(PhaseDivider.DAY_START)
        self._action_queue.clear()
        self.night_step = 0  # Reset night step counter

        self.state.push_event(
            description=f"Day {self.state.day_count} begins.",
            event_name=EventName.DAY_START,
            public=True
        )

        self.state.push_event(
            description=f"Villagers, let's decide who to exile. The discussion rule is: {self.discussion.discussion_rule}",
            event_name=EventName.MODERATOR_ANNOUNCEMENT,
            public=True,
            data={'discussion_rule': self.discussion.discussion_rule}
        )

        self.state.add_phase_divider(PhaseDivider.DAY_CHAT_START)
        self.discussion.begin(self.state)

        # Check if the protocol starts with bidding
        if isinstance(self.discussion, BiddingDiscussion):
            self.set_next_phase(DetailedPhase.DAY_BIDDING_AWAIT)
            # In bidding, all alive players can be active
            bidders = self.discussion.speakers_for_tick(self.state)
            self._action_queue.extend(BidAction, bidders)
            self.discussion.prompt_speakers_for_tick(self.state, bidders)
        else:
            # If no bidding, go straight to chatting
            self.set_next_phase(DetailedPhase.DAY_CHAT_AWAIT)
            initial_speakers = self.discussion.speakers_for_tick(self.state)
            self._action_queue.extend(ChatAction, initial_speakers)
            self.discussion.prompt_speakers_for_tick(self.state, initial_speakers)

    @phase_handler(DetailedPhase.DAY_BIDDING_AWAIT)
    def _handle_day_bidding_await(self, player_actions: Dict[PlayerID, Action]):
        current_bidders = self._action_queue.get(BidAction)
        self._action_queue.clear()

        # The protocol processes bid actions
        self.discussion.process_actions(list(player_actions.values()), current_bidders, self.state)

        # We need to explicitly check if the bidding sub-phase is over
        # This requires a reference to the bidding protocol within BiddingDiscussion
        assert isinstance(self.discussion, BiddingDiscussion)
        bidding_protocol = self.discussion.bidding
        if bidding_protocol.is_finished(self.state):
            self.state.push_event(
                description="Bidding has concluded. The discussion will now begin.",
                event_name=EventName.PHASE_CHANGE,
                public=True
            )
            # Transition to the chat phase
            self.set_next_phase(DetailedPhase.DAY_CHAT_AWAIT)

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

    @phase_handler(DetailedPhase.DAY_CHAT_AWAIT)
    def _handle_day_chat_await(self, player_actions: Dict[PlayerID, Action]):
        speaker_ids = self._action_queue.get(ChatAction)
        self._action_queue.clear()
        self.discussion.process_actions(list(player_actions.values()), speaker_ids, self.state)

        if self.discussion.is_discussion_over(self.state):
            self.state.push_event(
                description="Daytime discussion has concluded. Moving to day vote.",
                event_name=EventName.PHASE_CHANGE,
                public=True
            )
            self.discussion.reset()

            self.state.add_phase_divider(PhaseDivider.DAY_CHAT_END)
            self.state.add_phase_divider(PhaseDivider.DAY_VOTE_START)
            alive_players = self.state.alive_players()
            self.day_voting.begin_voting(self.state, alive_players, alive_players)
            self.set_next_phase(DetailedPhase.DAY_VOTING_AWAIT)
            self.state.push_event(
                description="Voting phase begins. We will decide who to exile today."
                            f"\nDay voting Rule: {self.day_voting.voting_rule}"
                            f"\nCurrent alive players are: {[player.id for player in alive_players]}",
                event_name=EventName.MODERATOR_ANNOUNCEMENT,
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
                        self.state.push_event(
                            description=prompt, event_name=EventName.VOTE_REQUEST,
                            public=False, visible_to=[voter_id]
                        )
        else:
            # Discussion is not over. Check if we need to go back to bidding action and phase.
            action_cls = ChatAction
            if isinstance(self.discussion, BiddingDiscussion) and self.discussion.is_bidding_phase():
                self.set_next_phase(DetailedPhase.DAY_BIDDING_AWAIT)
                action_cls = BidAction

            # Get the next active players (either bidders or the next speaker)
            next_actors = self.discussion.speakers_for_tick(self.state)
            self._action_queue.extend(action_cls, next_actors)
            self.discussion.prompt_speakers_for_tick(self.state, next_actors)

    @phase_handler(DetailedPhase.DAY_VOTING_AWAIT)
    def _handle_day_voting_await(self, player_actions: Dict[PlayerID, Action]):
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
                        self.state.push_event(
                            description=f'"{exiled_player_id}" in team {data.elected_player_team_name} is exiled by vote. The player is a {original_role_name}.',
                            event_name=EventName.ELIMINATION,
                            public=True,
                            data=data
                        )
                    else:
                        data = DayExileElectedDataEntry(
                            elected_player_id=exiled_player_id
                        )
                        self.state.push_event(
                            description=f'"{exiled_player_id}" in team {data.elected_player_team_name} is exiled by vote.',
                            event_name=EventName.ELIMINATION,
                            public=True,
                            data=data
                        )
            else:
                self.state.push_event(
                    description="The vote resulted in no exile (e.g., a tie, no majority, or all abstained).",
                    event_name=EventName.VOTE_RESULT,
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
                self.set_next_phase(DetailedPhase.NIGHT_START)
        else:
            next_voters_ids = self.day_voting.get_next_voters()
            self._action_queue.extend(VoteAction, next_voters_ids)
            if next_voters_ids:
                for voter_id in next_voters_ids:
                    player = self.state.get_player_by_id(voter_id)
                    if player and player.alive:
                        prompt = self.day_voting.get_voting_prompt(self.state, voter_id)
                        self.state.push_event(
                            description=prompt, event_name=EventName.VOTE_REQUEST,
                            public=False, visible_to=[voter_id]
                        )
            # Stay in DetailedPhase.DAY_VOTING_AWAIT

    def _determine_and_log_winner(self):
        # Check if a GAME_END entry already exists
        game_end_history = self.state.get_event_by_name(EventName.GAME_END)
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

        self.state.push_event(
            description=f"{winner_message}\n{reason}\nScores: {scores}\n"
                        f"Survivors: {data.survivors_until_last_round_and_role}\n"
                        f"All player roles: {data.all_players_and_role}",
            event_name=EventName.GAME_END,
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