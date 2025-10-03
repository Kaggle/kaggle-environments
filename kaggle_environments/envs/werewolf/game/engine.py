import json
from typing import Dict, List, Protocol, Sequence, Type

from .actions import Action, BidAction, ChatAction, VoteAction
from .base import BaseModerator, PlayerID
from .consts import DetailedPhase, PhaseDivider, RevealLevel, RoleConst, Team
from .night_elimination_manager import NightEliminationManager
from .protocols.base import DiscussionProtocol, VotingProtocol
from .protocols.chat import BiddingDiscussion
from .records import (
    DayExileElectedDataEntry,
    EventName,
    GameEndResultsDataEntry,
    GameStartDataEntry,
    GameStartRoleDataEntry,
    RequestWerewolfVotingDataEntry,
    WerewolfNightEliminationElectedDataEntry,
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
            raise ValueError(f"player {player_id} is already in the action queue. ")
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
        setattr(func, "_phase_handler_for", phase)
        return func

    return decorator


class PhaseHandler(Protocol):
    def __call__(self, player_actions: Dict[PlayerID, Action]) -> DetailedPhase:
        pass


class Moderator(BaseModerator):
    """Drives the finite-state machine for the game."""

    def __init__(
        self,
        state: GameState,
        discussion: DiscussionProtocol,
        day_voting: VotingProtocol,  # Renamed for clarity
        night_voting: VotingProtocol,
        night_elimination_reveal_level: RevealLevel = RevealLevel.ROLE,
        day_exile_reveal_level: RevealLevel = RevealLevel.ROLE,
    ):
        self._state = state
        self.discussion = discussion
        self.day_voting = day_voting
        self.night_voting = night_voting

        self._night_elimination_reveal_level = night_elimination_reveal_level
        self._day_exile_reveal_level = day_exile_reveal_level

        self._active_night_roles_queue: List[Player] = []
        self._night_elimination_manager = NightEliminationManager(
            self._state, reveal_level=self._night_elimination_reveal_level
        )
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
            day_discussion_display_name=self.discussion.display_name,
            day_discussion_protocol_rule=self.discussion.rule,
            night_werewolf_discussion_protocol_name=self.night_voting.__class__.__name__,
            night_werewolf_discussion_display_name=self.night_voting.display_name,
            night_werewolf_discussion_protocol_rule=self.night_voting.rule,
            day_voting_protocol_name=self.day_voting.__class__.__name__,
            day_voting_display_name=self.day_voting.display_name,
            day_voting_protocol_rule=self.day_voting.rule,
        )

        role_msg = "\n".join(
            ["The following explain the function of each role."]
            + [
                f"  * Role name {role.name.value} - team {role.team.value} - {role.descriptions}"
                for role in self.state.all_unique_roles
            ]
        )

        if self._day_exile_reveal_level == RevealLevel.ROLE:
            day_exile_reveal_msg = "If a player is exiled in the day, their role will be revealed."
        elif self._day_exile_reveal_level == RevealLevel.TEAM:
            day_exile_reveal_msg = "If a player is exiled in the day, their team will be revealed."
        elif self._day_exile_reveal_level == RevealLevel.NO_REVEAL:
            day_exile_reveal_msg = "If a player is exiled in the day, their team and role will NOT be revealed."
        else:
            raise ValueError(f"Unsupported day_exile_reveal_level = {self._day_exile_reveal_level}.")

        if self._night_elimination_reveal_level == RevealLevel.ROLE:
            night_elimination_reveal_msg = "If a player is eliminated at night, their role will be revealed."
        elif self._night_elimination_reveal_level == RevealLevel.TEAM:
            night_elimination_reveal_msg = "If a player is eliminated at night, their team will be revealed."
        elif self._night_elimination_reveal_level == RevealLevel.NO_REVEAL:
            night_elimination_reveal_msg = (
                "If a player is eliminated at night, their team and role will NOT be revealed."
            )
        else:
            raise ValueError(f"Unsupported night_elimination_reveal_level = {self._night_elimination_reveal_level}.")

        description = "\n - ".join(
            [
                "Werewolf game begins.",
                f"**Player Roster:** {data.player_ids}",
                f"**Alive Players:** {data.number_of_players}.",
                f"**Role Counts:** {data.role_counts}.",
                f"**Alive Team Member:** {data.team_member_counts}",
                f"**Day Discussion:** {data.day_discussion_display_name}. {data.day_discussion_protocol_rule}",
                f"**Day Exile Vote:** {data.day_voting_display_name}. {data.day_voting_protocol_rule}",
                f"**Night Werewolf Vote:** {data.night_werewolf_discussion_display_name}. {data.night_werewolf_discussion_protocol_rule}",
                role_msg,
                day_exile_reveal_msg,
                night_elimination_reveal_msg,
            ]
        )
        self.state.push_event(
            description=description, event_name=EventName.MODERATOR_ANNOUNCEMENT, public=True, data=data
        )
        # add role specific announcements
        for player in self.state.alive_players():
            data = GameStartRoleDataEntry(
                player_id=player.id, team=player.role.team, role=player.role.name, rule_of_role=player.role.descriptions
            )
            self.state.push_event(
                description=f'Your player id is "{data.player_id}". Your team is "{data.team}". Your role is "{data.role}".\n'
                f"The rule of your role: {data.rule_of_role}",
                event_name=EventName.GAME_START,
                public=False,
                visible_to=[player.id],
                data=data,
            )

    def _register_phase_handlers(self):
        """Collects all methods decorated with @phase_handler."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, "_phase_handler_for"):
                phase = getattr(attr, "_phase_handler_for")
                self._phase_handlers[phase] = attr

    def _register_player_handlers(self):
        for player in self.state.players:
            for event_name, handlers in player.get_event_handlers(self).items():
                for handler in handlers:
                    self.state.register_event_handler(event_name, handler)

    def request_action(
        self,
        action_cls: Type[Action],
        player_id: PlayerID,
        prompt: str,
        data=None,
        event_name=EventName.MODERATOR_ANNOUNCEMENT,
    ):
        """A public method for listeners to add a player to the action queue."""
        self._action_queue.append(action_cls, player_id)
        # Create the corresponding data entry to prompt the player
        self.state.push_event(
            description=prompt, event_name=event_name, public=False, visible_to=[player_id], data=data
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
        self.state.phase = new_detailed_phase.category

        if add_one_day:
            self.state.day_count += 1

        self.state.push_event(
            description=f"Transitioning from {old_detailed_phase} to {new_detailed_phase}.",
            event_name=EventName.PHASE_CHANGE,
            public=False,
        )

    def get_active_player_ids(self) -> List[PlayerID]:
        return self._action_queue.get_active_player_ids()

    def record_night_save(self, doctor_id: PlayerID, target_id: PlayerID):
        self._night_elimination_manager.record_save(doctor_id, target_id)

    def _call_handler(self, player_actions: Dict[PlayerID, Action]):
        current_handler = self._phase_handlers.get(self.detailed_phase)
        if current_handler:
            next_detailed_phase = current_handler(player_actions)
        else:
            raise ValueError(f"Unhandled detailed_phase: {self.detailed_phase}")
        add_one_day = True if next_detailed_phase == DetailedPhase.DAY_START else False
        self.set_next_phase(next_detailed_phase, add_one_day=add_one_day)

    def advance(self, player_actions: Dict[PlayerID, Action]):
        self.confirm_action(player_actions)
        # Process the incoming actions for the current phase.
        self._call_handler(player_actions)

        # Loop through automatic state transitions (those that don't need agent actions)
        # This continues until the game is over or requires new agent input.
        # this logic is required since Environments in core.py requires that there are some players being ACTIVE to
        # continue. Otherwise, if all INACTIVE the game is marked done.
        while not self.get_active_player_ids() and not self.is_game_over():
            self._call_handler({})

        # After all transitions, check for game over.
        if self.is_game_over() and self.detailed_phase != DetailedPhase.GAME_OVER:
            # clear action queue
            self._action_queue.clear()
            self.set_next_phase(DetailedPhase.GAME_OVER)
            self._determine_and_log_winner()

    @phase_handler(DetailedPhase.NIGHT_START)
    def _handle_night_start(self, player_actions: Dict[PlayerID, Action]) -> DetailedPhase:
        self._action_queue.clear()
        self.state.add_phase_divider(PhaseDivider.NIGHT_START)
        self.state.push_event(
            description=f"Night {self.state.day_count} begins!", event_name=EventName.NIGHT_START, public=True
        )

        # initialize werewolves voting
        self.state.add_phase_divider(PhaseDivider.NIGHT_VOTE_START)
        alive_werewolves = self.state.alive_players_by_role(RoleConst.WEREWOLF)
        alive_werewolf_ids = list({p.id for p in alive_werewolves})
        potential_targets = self.state.alive_players_by_team(Team.VILLAGERS)  # Target non-werewolves

        data = RequestWerewolfVotingDataEntry(
            valid_targets=[f"{p.id}" for p in potential_targets],
            alive_werewolve_player_ids=[f"{p.id}" for p in alive_werewolves],
            voting_protocol_name=self.night_voting.__class__.__name__,
            voting_protocol_rule=self.night_voting.rule,
            action_json_schema=json.dumps(VoteAction.schema_for_player()),
        )
        self.state.push_event(
            description=f"Wake up Werewolves. Your fellow alive werewolves are: {data.alive_werewolve_player_ids}. "
            f"Choose one target player to eliminate tonight. "
            f"The voting rule ({data.voting_protocol_name}): {data.voting_protocol_rule} "
            f"Who would you like to eliminate tonight? Options: {data.valid_targets}.",
            event_name=EventName.VOTE_REQUEST,
            public=False,
            visible_to=alive_werewolf_ids,
            data=data,
        )
        self.night_voting.begin_voting(
            state=self.state, alive_voters=alive_werewolves, potential_targets=potential_targets
        )
        return DetailedPhase.NIGHT_AWAIT_ACTIONS

    @phase_handler(DetailedPhase.NIGHT_AWAIT_ACTIONS)
    def _handle_night_await_actions(self, player_actions: Dict[PlayerID, Action]) -> DetailedPhase:
        # Process werewolf votes
        werewolf_voters_expected = self._action_queue.get(VoteAction)
        if werewolf_voters_expected:
            self.night_voting.collect_votes(player_actions, self.state, werewolf_voters_expected)

        self._action_queue.clear()

        if not self.night_voting.done():
            next_ww_voters = self.night_voting.get_next_voters()
            self._action_queue.extend(VoteAction, next_ww_voters)
            vote_action_queue = self._action_queue.get(VoteAction)
            alive_werewolves_still_to_vote = [
                p for p in self.state.alive_players_by_role(RoleConst.WEREWOLF) if p.id in vote_action_queue
            ]
            if alive_werewolves_still_to_vote:
                for ww_voter in alive_werewolves_still_to_vote:
                    prompt = self.night_voting.get_voting_prompt(self.state, ww_voter.id)
                    self.state.push_event(
                        description=prompt,
                        event_name=EventName.VOTE_REQUEST,
                        public=False,
                        visible_to=[ww_voter.id],
                        visible_in_ui=False,
                    )
            return DetailedPhase.NIGHT_AWAIT_ACTIONS
        else:
            return DetailedPhase.NIGHT_CONCLUDE

    @phase_handler(DetailedPhase.NIGHT_CONCLUDE)
    def _handle_night_conclude(self, player_actions: Dict[PlayerID, Action]) -> DetailedPhase:
        werewolf_target_id = self.night_voting.get_elected()

        data = WerewolfNightEliminationElectedDataEntry(elected_target_player_id=werewolf_target_id)
        self.state.push_event(
            description=f'Werewolves elected to eliminate player "{data.elected_target_player_id}".',
            event_name=EventName.VOTE_RESULT,
            public=False,
            visible_to=[p.id for p in self.state.alive_players_by_team(Team.WEREWOLVES)],
            data=data,
        )

        self._night_elimination_manager.resolve_elimination(werewolf_target_id)

        self.night_voting.reset()
        self._night_elimination_manager.reset()

        self.state.add_phase_divider(PhaseDivider.NIGHT_VOTE_END)
        self.state.add_phase_divider(PhaseDivider.NIGHT_END)
        return DetailedPhase.DAY_START

    @phase_handler(DetailedPhase.DAY_START)
    def _handle_day_start(self, player_actions: Dict[PlayerID, Action]) -> DetailedPhase:
        self.state.add_phase_divider(PhaseDivider.DAY_START)
        self._action_queue.clear()
        self.night_step = 0  # Reset night step counter

        self.state.push_event(
            description=f"Day {self.state.day_count} begins.", event_name=EventName.DAY_START, public=True
        )

        self.state.push_event(
            description=f"Villagers, let's decide who to exile. The discussion rule is: {self.discussion.rule}",
            event_name=EventName.MODERATOR_ANNOUNCEMENT,
            public=True,
            data={"discussion_rule": self.discussion.rule},
        )

        self.state.add_phase_divider(PhaseDivider.DAY_CHAT_START)
        self.discussion.begin(self.state)

        # Check if the protocol starts with bidding
        if isinstance(self.discussion, BiddingDiscussion):
            return DetailedPhase.DAY_BIDDING_AWAIT
        else:
            return DetailedPhase.DAY_CHAT_AWAIT

    @phase_handler(DetailedPhase.DAY_BIDDING_AWAIT)
    def _handle_day_bidding_await(self, player_actions: Dict[PlayerID, Action]) -> DetailedPhase:
        current_bidders = self._action_queue.get(BidAction)
        self._action_queue.clear()

        # The protocol processes bid actions
        self.discussion.process_actions(list(player_actions.values()), current_bidders, self.state)

        # We need to explicitly check if the bidding sub-phase is over
        # This requires a reference to the bidding protocol within BiddingDiscussion
        if self.discussion.bidding.is_finished(self.state):
            return DetailedPhase.DAY_BIDDING_CONCLUDE
        else:
            # Bidding is not over (e.g., sequential auction), get next bidders
            next_bidders = self.discussion.speakers_for_tick(self.state)
            self._action_queue.extend(BidAction, next_bidders)
            self.discussion.prompt_speakers_for_tick(self.state, next_bidders)
            return DetailedPhase.DAY_BIDDING_AWAIT

    @phase_handler(DetailedPhase.DAY_BIDDING_CONCLUDE)
    def _handle_day_bidding_conclude(self, player_actions: Dict[PlayerID, Action]) -> DetailedPhase:
        self.state.push_event(
            description="Bidding has concluded. The discussion will now begin.",
            event_name=EventName.PHASE_CHANGE,
            public=True,
        )
        self.discussion.bidding.reset()
        return DetailedPhase.DAY_CHAT_AWAIT

    @phase_handler(DetailedPhase.DAY_CHAT_AWAIT)
    def _handle_day_chat_await(self, player_actions: Dict[PlayerID, Action]) -> DetailedPhase:
        speaker_ids = self._action_queue.get(ChatAction)
        self._action_queue.clear()
        self.discussion.process_actions(list(player_actions.values()), speaker_ids, self.state)

        if self.discussion.is_discussion_over(self.state):
            return DetailedPhase.DAY_CHAT_CONCLUDE
        else:
            # Discussion is not over. Check if we need to go back to bidding action and phase.
            if isinstance(self.discussion, BiddingDiscussion) and self.discussion.is_bidding_phase():
                return DetailedPhase.DAY_BIDDING_AWAIT
            # Get the next active players (either bidders or the next speaker)
            next_actors = self.discussion.speakers_for_tick(self.state)
            self._action_queue.extend(ChatAction, next_actors)
            self.discussion.prompt_speakers_for_tick(self.state, next_actors)
            return DetailedPhase.DAY_CHAT_AWAIT

    @phase_handler(DetailedPhase.DAY_CHAT_CONCLUDE)
    def _handle_day_chat_conclude(self, player_actions: Dict[PlayerID, Action]) -> DetailedPhase:
        self.state.push_event(
            description="Daytime discussion has concluded. Moving to day vote.",
            event_name=EventName.PHASE_CHANGE,
            public=True,
        )
        self.discussion.reset()
        self.state.add_phase_divider(PhaseDivider.DAY_CHAT_END)
        return DetailedPhase.DAY_VOTING_START

    @phase_handler(DetailedPhase.DAY_VOTING_START)
    def _handle_day_voting_start(self, player_actions: Dict[PlayerID, Action]) -> DetailedPhase:
        self.state.add_phase_divider(PhaseDivider.DAY_VOTE_START)
        alive_players = self.state.alive_players()
        self.day_voting.begin_voting(self.state, alive_players, alive_players)
        self.state.push_event(
            description="Voting phase begins. We will decide who to exile today."
            f"\nDay voting Rule: {self.day_voting.rule}"
            f"\nCurrent alive players are: {[player.id for player in alive_players]}",
            event_name=EventName.MODERATOR_ANNOUNCEMENT,
            public=True,
            data={"voting_rule": self.day_voting.rule},
        )
        return DetailedPhase.DAY_VOTING_AWAIT

    @phase_handler(DetailedPhase.DAY_VOTING_AWAIT)
    def _handle_day_voting_await(self, player_actions: Dict[PlayerID, Action]) -> DetailedPhase:
        vote_queue = self._action_queue.get(VoteAction)
        self.day_voting.collect_votes(player_actions, self.state, vote_queue)
        self._action_queue.clear()  # Clear previous voters

        if self.day_voting.done():
            return DetailedPhase.DAY_VOTING_CONCLUDE
        else:
            next_voters_ids = self.day_voting.get_next_voters()
            self._action_queue.extend(VoteAction, next_voters_ids)
            if next_voters_ids:
                for voter_id in next_voters_ids:
                    player = self.state.get_player_by_id(voter_id)
                    if player and player.alive:
                        prompt = self.day_voting.get_voting_prompt(self.state, voter_id)
                        self.state.push_event(
                            description=prompt,
                            event_name=EventName.VOTE_REQUEST,
                            public=False,
                            visible_to=[voter_id],
                            visible_in_ui=False,
                        )
            return DetailedPhase.DAY_VOTING_AWAIT

    @phase_handler(DetailedPhase.DAY_VOTING_CONCLUDE)
    def _handle_day_voting_conclude(self, player_actions: Dict[PlayerID, Action]) -> DetailedPhase:
        exiled_player_id = self.day_voting.get_elected()
        if exiled_player_id:
            exiled_player = self.state.get_player_by_id(exiled_player_id)
            if exiled_player:
                self.state.eliminate_player(exiled_player_id)

                role = None
                team = None
                description = f'Player "{exiled_player_id}" is exiled by vote.'
                if self._day_exile_reveal_level == RevealLevel.ROLE:
                    role = exiled_player.role.name
                    team = exiled_player.role.team
                    description = (
                        f'Player "{exiled_player_id}" in team {team} is exiled by vote. The player is a {role}.'
                    )
                elif self._day_exile_reveal_level == RevealLevel.TEAM:
                    team = exiled_player.role.team
                    description = f'Player "{exiled_player_id}" in team {team} is exiled by vote.'

                data = DayExileElectedDataEntry(
                    elected_player_id=exiled_player_id, elected_player_role_name=role, elected_player_team_name=team
                )
                self.state.push_event(description=description, event_name=EventName.ELIMINATION, public=True, data=data)
        else:
            self.state.push_event(
                description="The vote resulted in no exile (e.g., a tie, no majority, or all abstained).",
                event_name=EventName.VOTE_RESULT,
                public=True,
                data={"vote_type": "day_exile", "outcome": "no_exile", "reason": "tie_or_no_majority"},
            )

        self.day_voting.reset()
        self.state.add_phase_divider(PhaseDivider.DAY_VOTE_END)
        self.state.add_phase_divider(PhaseDivider.DAY_END)
        return DetailedPhase.NIGHT_START

    def _determine_and_log_winner(self):
        # Check if a GAME_END entry already exists
        game_end_event = self.state.get_event_by_name(EventName.GAME_END)
        if game_end_event:
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
            all_players=[p.model_dump() for p in self.state.players],
        )

        self.state.push_event(
            description=f"{winner_message}\n{reason}\nScores: {scores}\n"
            f"Survivors: {data.survivors_until_last_round_and_role}\n"
            f"All player roles: {data.all_players_and_role}",
            event_name=EventName.GAME_END,
            public=True,
            data=data,
        )

    def is_game_over(self) -> bool:
        if self.detailed_phase == DetailedPhase.GAME_OVER:
            return True
        wolves = self.state.alive_players_by_team(Team.WEREWOLVES)
        villagers = self.state.alive_players_by_team(Team.VILLAGERS)
        if not wolves and villagers:
            return True
        if wolves and len(wolves) >= len(villagers):
            return True
        return False
