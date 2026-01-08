import random
from collections import Counter
from typing import Dict, List, Optional, Sequence

from kaggle_environments.envs.werewolf.game.actions import Action, NoOpAction, VoteAction
from kaggle_environments.envs.werewolf.game.base import PlayerID
from kaggle_environments.envs.werewolf.game.consts import EventName, Phase, StrEnum
from kaggle_environments.envs.werewolf.game.protocols.base import VotingProtocol
from kaggle_environments.envs.werewolf.game.records import (
    DayExileVoteDataEntry,
    VoteOrderDataEntry,
    WerewolfNightVoteDataEntry,
)
from kaggle_environments.envs.werewolf.game.roles import Player
from kaggle_environments.envs.werewolf.game.states import GameState

from .factory import register_protocol
from .ordering import FirstPlayerStrategy, PivotSelector


class TieBreak(StrEnum):
    RANDOM = "random"
    """Randomly select from top ties."""

    NO_EXILE = "no_elected"
    """Tie result in no one elected."""


ABSTAIN_VOTE = "-1"


class Ballot:
    def __init__(self, tie_selection: TieBreak = TieBreak.RANDOM):
        self._ballots: Dict[PlayerID, PlayerID] = {}
        self._tie_selection = tie_selection

    def reset(self):
        self._ballots = {}

    def add_vote(self, voter_id: PlayerID, target_id: PlayerID):
        """Records a vote from a voter for a target."""
        self._ballots[voter_id] = target_id

    def get_tally(self) -> Counter:
        """Returns a Counter of votes for each target, excluding abstained votes."""
        return Counter(v for v in self._ballots.values() if v is not None and v != ABSTAIN_VOTE)

    def get_elected(self, potential_targets: List[PlayerID]) -> Optional[PlayerID]:
        """
        Tallies the votes and determines the elected player based on the tie-breaking rule.
        """
        counts = self.get_tally().most_common()
        elected: Optional[PlayerID] = None

        if not counts:
            # No valid votes were cast.
            if self._tie_selection == TieBreak.RANDOM and potential_targets:
                elected = random.choice(potential_targets)
            # If NO_EXILE, elected remains None.
        else:
            _, top_votes = counts[0]
            top_candidates = [v for v, c in counts if c == top_votes]

            if len(top_candidates) == 1:
                elected = top_candidates[0]
            else:  # It's a tie.
                if self._tie_selection == TieBreak.RANDOM:
                    elected = random.choice(top_candidates)
                # If NO_EXILE, elected remains None.

        return elected

    def get_all_votes(self) -> Dict[PlayerID, PlayerID]:
        """Returns a copy of all recorded ballots."""
        return self._ballots.copy()

    def get_tie_break_description(self):
        if self._tie_selection == TieBreak.RANDOM:
            return (
                "Ties result in random selection amongst the top ties. "
                "If no valid vote available (if all casted abstained votes), "
                "will result in random elimination of one player."
            )

        elif self._tie_selection == TieBreak.NO_EXILE:
            return "Ties result in no elimination."

        else:
            raise ValueError(f"Unsupported tie_break={self._tie_selection}.")


@register_protocol()
class SimultaneousMajority(VotingProtocol):
    def __init__(self, tie_break=TieBreak.RANDOM):
        self._expected_voters: List[PlayerID] = []
        self._potential_targets: List[PlayerID] = []
        self._current_game_state: Optional[GameState] = None  # To store state from begin_voting
        self._elected: Optional[PlayerID] = None
        self._done_tallying = False
        self._tie_break = tie_break
        self._ballot = Ballot(tie_selection=self._tie_break)

        if tie_break not in TieBreak:
            raise ValueError(f"Invalid tie_break value: {tie_break}. Must be one of {TieBreak}.")

    def reset(self) -> None:
        self._ballot.reset()
        self._expected_voters = []
        self._potential_targets = []
        self._current_game_state = None
        self._elected = None
        self._done_tallying = False

    @property
    def display_name(self) -> str:
        return "Simultaneous Majority Voting"

    @property
    def rule(self) -> str:
        rule = "Player with the most votes is eliminated. "
        rule += self._ballot.get_tie_break_description()
        return rule

    def begin_voting(self, state: GameState, alive_voters: Sequence[Player], potential_targets: Sequence[Player]):
        self._ballot.reset()
        # Ensure voters and targets are alive at the start of voting
        self._expected_voters = [p.id for p in alive_voters if p.alive]
        self._potential_targets = [p.id for p in potential_targets if p.alive]
        self._current_game_state = state  # Store the game state reference

    def collect_votes(self, player_actions: Dict[PlayerID, Action], state: GameState, expected_voters: List[PlayerID]):
        for actor_id, action in player_actions.items():
            if actor_id in expected_voters:
                self.collect_vote(action, state)

        # For any expected voter who didn't act, record an abstain vote.
        all_votes = self._ballot.get_all_votes()
        for player_id in expected_voters:
            if player_id not in all_votes:
                self._ballot.add_vote(player_id, ABSTAIN_VOTE)

    def collect_vote(self, vote_action: Action, state: GameState):
        actor_player = state.get_player_by_id(vote_action.actor_id)
        if not isinstance(vote_action, VoteAction):
            state.push_event(
                description=f'Invalid vote attempt by player "{vote_action.actor_id}". '
                f"Not a VoteAction; submitted {vote_action.__class__.__name__} instead. "
                f"Cast as abstained vote.",
                event_name=EventName.ERROR,
                public=False,
                visible_to=self._expected_voters,
                data={},
            )
            self._ballot.add_vote(vote_action.actor_id, ABSTAIN_VOTE)
            return

        if state.phase == Phase.NIGHT:
            data_entry_class = WerewolfNightVoteDataEntry
        else:
            data_entry_class = DayExileVoteDataEntry

        data = data_entry_class(
            actor_id=vote_action.actor_id,
            target_id=vote_action.target_id,
            reasoning=vote_action.reasoning,
            perceived_threat_level=vote_action.perceived_threat_level,
            action=vote_action,
        )

        # Voter must be expected and alive at the moment of casting vote
        if actor_player and actor_player.alive and vote_action.actor_id in self._expected_voters:
            # Prevent re-voting
            if vote_action.actor_id in self._ballot.get_all_votes():
                state.push_event(
                    description=f'Invalid vote attempt by "{vote_action.actor_id}", already voted.',
                    event_name=EventName.ERROR,
                    public=False,
                    visible_to=self._expected_voters,
                    data=data,
                )
                return

            if vote_action.target_id in self._potential_targets:
                self._ballot.add_vote(vote_action.actor_id, vote_action.target_id)

                # Determine DataEntry type based on game phase
                state.push_event(
                    description=f'Player "{data.actor_id}" voted to eliminate "{data.target_id}". ',
                    event_name=EventName.VOTE_ACTION,
                    public=False,
                    visible_to=self._expected_voters,
                    data=data,
                    source=vote_action.actor_id,
                )
            else:
                self._ballot.add_vote(vote_action.actor_id, ABSTAIN_VOTE)
                state.push_event(
                    description=f'Invalid vote attempt by "{vote_action.actor_id}".',
                    event_name=EventName.ERROR,
                    public=False,
                    visible_to=self._expected_voters,
                    data=data,
                )
                return
        else:
            state.push_event(
                description=f"Invalid vote attempt by {vote_action.actor_id}.",
                event_name=EventName.ERROR,
                public=False,
                data=data,
            )

    def get_voting_prompt(self, state: GameState, player_id: PlayerID) -> str:
        target_options = [
            p_id
            for p_id in self._potential_targets
            if state.get_player_by_id(p_id) and state.get_player_by_id(p_id).alive
        ]
        return f'Player "{player_id}", please cast your vote. Options: {target_options} or Abstain ("{ABSTAIN_VOTE}").'

    def get_current_tally_info(self, state: GameState) -> Dict[PlayerID, int]:
        return self._ballot.get_tally()

    def get_next_voters(self) -> List[PlayerID]:
        # For simultaneous, all expected voters vote at once, and only once.
        return [voter for voter in self._expected_voters if voter not in self._ballot.get_all_votes()]

    def done(self) -> bool:
        # The voting is considered "done" after one tick where voters were requested.
        # The moderator will then call tally_votes.
        return all(voter in self._ballot.get_all_votes() for voter in self._expected_voters)

    def get_valid_targets(self) -> List[PlayerID]:
        # Return a copy of targets that were valid (alive) at the start of voting.
        return list(self._potential_targets)

    def get_elected(self) -> PlayerID | None:  # Return type matches tally_votes
        if not self.done():
            raise Exception("Voting is not done yet.")
        if self._elected is None and not self._done_tallying:
            self._elected = self._ballot.get_elected(self._potential_targets)
            self._done_tallying = True
        return self._elected


@register_protocol()
class SequentialVoting(VotingProtocol):
    """
    Players vote one by one in a sequence. Each player is shown the current
    tally before casting their vote. All players in the initial list of
    voters get a turn.
    """

    def __init__(self, first_to_vote: str = "rotate", tie_break: TieBreak = TieBreak.RANDOM):
        self._potential_targets: List[PlayerID] = []
        self._voter_queue: List[PlayerID] = []  # Order of players to vote
        self._expected_voters: List[PlayerID] = []
        self._current_voter_index: int = 0  # Index for _voter_queue
        self._current_game_state: Optional[GameState] = None  # To store state from begin_voting
        self._elected: Optional[PlayerID] = None
        self._done_tallying = False
        self.pivot_selector = PivotSelector(first_to_vote)
        self._ballot = Ballot(tie_selection=tie_break)

    def reset(self) -> None:
        self._ballot.reset()
        self._potential_targets = []
        self._expected_voters = []
        self._voter_queue = []
        self._current_voter_index = 0
        self._current_game_state = None
        self._elected = None
        self._done_tallying = False

    @property
    def display_name(self) -> str:
        return "Sequential Voting"

    @property
    def rule(self) -> str:
        rule_txt = "Players vote one by one. Player with the most votes after all have voted is eliminated. "
        strategy = self.pivot_selector.strategy
        if strategy == FirstPlayerStrategy.FIXED:
            rule_txt += "The voting order always starts from the beginning of the player list."
        elif strategy == FirstPlayerStrategy.ROTATE:
            rule_txt += "The starting voter rotates to the next player in the list for each subsequent voting phase."
        elif strategy == FirstPlayerStrategy.RANDOM:
            rule_txt += "The starting voter is chosen randomly for each voting phase."

        rule_txt += " " + self._ballot.get_tie_break_description()
        return rule_txt

    def begin_voting(self, state: GameState, alive_voters: Sequence[Player], potential_targets: Sequence[Player]):
        self._ballot.reset()
        all_ids = state.all_player_ids
        alive_voter_ids_set = set(p.id for p in alive_voters)

        # Determine pivot
        pivot = self.pivot_selector.get_pivot(all_ids, alive_voter_ids_set)

        # Construct full voter queue based on all players, then filter by alive
        # This ensures consistent rotation logic regardless of deaths
        ordered_potential_voters = PivotSelector.get_ordered_ids(all_ids, pivot)

        # Filter for actual alive voters
        self._expected_voters = [pid for pid in ordered_potential_voters if pid in alive_voter_ids_set]
        self._potential_targets = [p.id for p in potential_targets]
        self._voter_queue = list(self._expected_voters)
        self._current_voter_index = 0
        self._current_game_state = state

        if self._expected_voters:
            data = VoteOrderDataEntry(vote_order_of_player_ids=self._expected_voters)
            state.push_event(
                description=f"Voting starts from player {self._expected_voters[0]} "
                f"with the following order: {self._expected_voters}",
                event_name=EventName.VOTE_ORDER,
                public=False,
                visible_to=list(alive_voter_ids_set),
                data=data,
            )

    def get_voting_prompt(self, state: GameState, player_id: PlayerID) -> str:
        """
        Generates a prompt for the given player_id, assuming it's their turn.
        """
        current_tally = self.get_current_tally_info(state)

        # Sort for consistent display
        tally_str_parts = []
        for target_id, votes in sorted(current_tally.items(), key=lambda x: x[1], reverse=True):
            tally_str_parts.append(f"{target_id}: {votes} vote(s)")

        tally_str = "; ".join(tally_str_parts) if tally_str_parts else "No votes cast yet."

        options_str_parts = []
        for p_target in state.alive_players():  # Iterate through all alive players for options
            if p_target.id in self._potential_targets:
                options_str_parts.append(f"{p_target.id}")
        options_str = ", ".join(options_str_parts)

        return (
            f"{player_id}, it is your turn to vote. "
            f"Current tally: {tally_str}. "
            f"Options: {options_str} or Abstain (vote for {ABSTAIN_VOTE})."
        )

    def collect_votes(self, player_actions: Dict[PlayerID, Action], state: GameState, expected_voters: List[PlayerID]):
        if self.done():
            return

        # In sequential voting, expected_voters should contain exactly one player.
        if not expected_voters:
            # This case should ideally not be reached if `done()` is false.
            # However, if it happens (e.g. initialization where ActionQueue is empty),
            # we should NOT skip the turn. The Engine will query get_next_voters() again.
            return

        expected_voter_id = expected_voters[0]
        action = player_actions.get(expected_voter_id)

        if action:
            self.collect_vote(action, state)
        else:
            # This block handles timeout for the expected voter.
            # The player did not submit an action. Treat as NoOp/Abstain.
            self.collect_vote(NoOpAction(actor_id=expected_voter_id, day=state.day_count, phase=state.phase), state)

    def collect_vote(self, vote_action: Action, state: GameState):
        if not isinstance(vote_action, (VoteAction, NoOpAction)):
            # Silently ignore if not a VoteAction or NoOpAction.
            # Consider logging an "unexpected action type" error if more verbosity is needed.
            return

        if self.done():
            state.push_event(
                description=f"Action ({vote_action.kind}) received from {vote_action.actor_id}, "
                f"but voting is already complete.",
                event_name=EventName.ERROR,
                public=False,
                visible_to=[vote_action.actor_id],
            )
            return

        expected_voter_id = self._voter_queue[self._current_voter_index]
        if vote_action.actor_id != expected_voter_id:
            state.push_event(
                description=f"Action ({vote_action.kind}) received from {vote_action.actor_id}, "
                f"but it is {expected_voter_id}'s turn.",
                event_name=EventName.ERROR,
                public=False,  # Or public if strict turn enforcement is announced
                visible_to=[vote_action.actor_id, expected_voter_id],
            )
            return

        actor_player = next((p for p in state.players if p.id == vote_action.actor_id), None)
        if actor_player and actor_player.alive:
            description_for_event = ""
            involved_players_list = [vote_action.actor_id]  # Actor is always involved
            data = None
            if isinstance(vote_action, NoOpAction):
                self._ballot.add_vote(vote_action.actor_id, ABSTAIN_VOTE)  # Treat NoOp as abstain
                description_for_event = f"{vote_action.actor_id} chose to NoOp (treated as Abstain)."

            elif isinstance(vote_action, VoteAction):  # This must be true if not NoOpAction
                target_display: str
                recorded_target_id = vote_action.target_id
                if vote_action.target_id != ABSTAIN_VOTE and vote_action.target_id not in self._potential_targets:
                    # Invalid target chosen for VoteAction
                    state.push_event(
                        description=f"{vote_action.actor_id} attempted to vote for {vote_action.target_id} "
                        f"(invalid target). Vote recorded as Abstain.",
                        event_name=EventName.ERROR,
                        public=False,
                        visible_to=[vote_action.actor_id],
                    )
                    recorded_target_id = ABSTAIN_VOTE  # Treat invalid target as abstain
                    target_display = f"Invalid Target ({vote_action.target_id}), recorded as Abstain"
                elif vote_action.target_id == ABSTAIN_VOTE:
                    # Explicit Abstain via VoteAction
                    target_display = "Abstain"
                    # recorded_target_id is already ABSTAIN_VOTE
                else:
                    # Valid target chosen for VoteAction
                    target_display = f"{vote_action.target_id}"
                    involved_players_list.append(vote_action.target_id)  # Add valid target to involved

                self._ballot.add_vote(vote_action.actor_id, recorded_target_id)
                description_for_event = f"{vote_action.actor_id} has voted for {target_display}."

                # Add data entry for the vote
                data_entry_class = DayExileVoteDataEntry if state.phase == Phase.DAY else WerewolfNightVoteDataEntry
                data = data_entry_class(
                    actor_id=vote_action.actor_id,
                    target_id=recorded_target_id,
                    reasoning=vote_action.reasoning,
                    perceived_threat_level=vote_action.perceived_threat_level,
                    action=vote_action,
                )

            state.push_event(
                description=description_for_event,
                event_name=EventName.VOTE_ACTION,
                public=False,
                visible_to=self._expected_voters,
                data=data,
                source=vote_action.actor_id,
            )
            self._current_voter_index += 1
        else:  # Player not found, not alive, or (redundantly) not their turn
            state.push_event(
                description=f"Invalid action ({vote_action.kind}) attempt by {vote_action.actor_id} (player not found,"
                f" not alive, or not their turn). Action not counted.",
                event_name=EventName.ERROR,
                public=False,
                visible_to=[vote_action.actor_id],
            )
            # If voter was expected but found to be not alive, advance turn to prevent stall
            if vote_action.actor_id == expected_voter_id:  # Implies actor_player was found but not actor_player.alive
                self._current_voter_index += 1

    def get_current_tally_info(self, state: GameState) -> Dict[str, int]:
        # Returns counts of non-abstain votes for valid targets
        return self._ballot.get_tally()

    def get_next_voters(self) -> List[PlayerID]:
        if not self.done():
            # Ensure _current_voter_index is within bounds before accessing
            if self._current_voter_index < len(self._voter_queue):
                return [self._voter_queue[self._current_voter_index]]
        return []

    def done(self) -> bool:
        if not self._voter_queue:  # No voters were ever in the queue
            return True
        return self._current_voter_index >= len(self._voter_queue)

    def get_valid_targets(self) -> List[PlayerID]:
        return list(self._potential_targets)

    def get_elected(self) -> Optional[PlayerID]:
        if not self.done():
            raise Exception("Voting is not done yet.")
        if self._elected is None and not self._done_tallying:
            self._elected = self._ballot.get_elected(self._potential_targets)
            self._done_tallying = True
        return self._elected
