import itertools
import json
import random
import re
from abc import ABC, abstractmethod
from collections import Counter, deque
from typing import Dict, List, Sequence, Optional, Tuple

from .actions import EliminateProposalAction, BidAction, Action, ChatAction, VoteAction, NoOpAction
from .consts import Team, Phase
from .records import HistoryEntryType, RequestVillagerToSpeakDataEntry, DayExileVoteDataEntry, ChatDataEntry, \
    WerewolfNightVoteDataEntry, BidDataEntry, BidResultDataEntry, DiscussionOrderDataEntry, VoteOrderDataEntry
from .roles import Player
from .states import GameState


def _extract_player_ids_from_string(text: str, all_player_ids: List[str]) -> List[str]:
    """Extracts player IDs mentioned in a string."""
    if not all_player_ids:
        return []
    # Create a regex pattern to find any of the player IDs as whole words
    # Using a set for faster lookups and to handle duplicates from the regex
    pattern = r'\b(' + '|'.join(re.escape(pid) for pid in all_player_ids) + r')\b'
    # Use a set to automatically handle duplicates found by the regex
    found_ids = set(re.findall(pattern, text))
    return sorted(list(found_ids)) # sorted for deterministic order


class VotingProtocol(ABC):
    """Collects, validates, and tallies votes."""

    @property
    @abstractmethod
    def voting_rule(self) -> str:
        """A string describing the voting rule in effect."""

    @abstractmethod
    def begin_voting(self, state: GameState, alive_voters: Sequence[Player], potential_targets: Sequence[Player]):
        """Initialize for a new voting round."""

    @abstractmethod
    def get_voting_prompt(self, state: GameState, player_id: str) -> str:
        """
        Returns a string prompt for the specified player, potentially including current tally.
        """

    @abstractmethod
    def collect_vote(self, vote_action: Action, state: GameState):  # Changed to Action, will check type
        """Collect an individual vote."""

    @abstractmethod
    def collect_votes(self, player_actions: Dict[str, Action], state: GameState, expected_voters: List[str]):
        """Collect a batch of votes."""

    @abstractmethod
    def _tally_votes(self, state: GameState) -> str | None:
        """
        Return exiled `player_id`, or None if no one is exiled
        (e.g. no majority rule / tied vote behaviour).
        """

    @abstractmethod
    def get_current_tally_info(self, state: GameState) -> Dict[str, str]:
        """
        Return the current tally by a map, where key is player, value is target.
        """

    @abstractmethod
    def get_next_voters(self) -> List[str]:
        """get the next batch of voters"""

    @abstractmethod
    def done(self):
        """Check if voting is done."""

    @abstractmethod
    def get_valid_targets(self) -> List[str]:
        """get a list of targets"""

    @abstractmethod
    def get_elected(self) -> Optional[str]:
        """get the final elected individual, or None if no one was elected."""

    @abstractmethod
    def reset(self) -> None:
        """Resets the protocol to its initial state."""
        pass


class TeamDecisionProtocol(ABC):
    """
    Converts multiple proposals from the same team into ONE final action.
    """

    @abstractmethod
    def resolve(
            self,
            team_players: Sequence[Player],
            proposals: Sequence[EliminateProposalAction],
    ) -> EliminateProposalAction | None:
        """"""

    @abstractmethod
    def reset(self) -> None:
        """Resets the protocol to its initial state."""
        pass


def _find_mentioned_players(text: str, all_player_ids: List[str]) -> List[str]:
    """
    Finds player IDs mentioned in a string of text, ordered by their first appearance.
    Player IDs are treated as whole words.
    Example: "I think gpt-4 is suspicious, what do you think John?" -> ["gpt-4", "John"]
    """
    if not text or not all_player_ids:
        return []

    # Sort by length descending to handle substrings correctly.
    sorted_player_ids = sorted(all_player_ids, key=len, reverse=True)
    pattern = r'\b(' + '|'.join(re.escape(pid) for pid in sorted_player_ids) + r')\b'

    matches = re.finditer(pattern, text)

    # Deduplicate while preserving order of first appearance
    ordered_mentioned_ids = []
    seen = set()
    for match in matches:
        player_id = match.group(1)
        if player_id not in seen:
            ordered_mentioned_ids.append(player_id)
            seen.add(player_id)

    return ordered_mentioned_ids


class BiddingProtocol(ABC):
    """Drives one auction round and returns the winner(s)."""
    @property
    @abstractmethod
    def bidding_rules(self) -> str:
        """Specify the bidding rules"""

    @property
    @abstractmethod
    def bids(self) -> Dict[str, int]:
        """return a snapshot of the current bids"""

    @staticmethod
    def get_last_mentioned(state: GameState) -> Tuple[List[str], str]:
        """get the players that were mentioned in last player message."""
        last_chat_message = ""
        sorted_days = sorted(state.history.keys(), reverse=True)
        for day in sorted_days:
            for entry in reversed(state.history[day]):
                if entry.entry_type == HistoryEntryType.DISCUSSION and isinstance(entry.data, ChatDataEntry):
                    last_chat_message = entry.data.message
                    break
            if last_chat_message:
                break
        players = _find_mentioned_players(last_chat_message, state.all_player_ids)
        return players, last_chat_message

    @abstractmethod
    def begin(self, state: GameState) -> None: ...

    @abstractmethod
    def accept(self, bid: BidAction, state: GameState) -> None: ...

    @abstractmethod
    def process_incoming_bids(self, actions: List[Action], state: GameState) -> None:
        """Processes a batch of actions, handling BidActions by calling self.accept()."""

    @abstractmethod
    def is_finished(self, state: GameState) -> bool: ...

    @abstractmethod
    def outcome(self, state: GameState) -> list[str]:
        """ # Return type should be list[str] for player IDs
        Return list of player-ids, ordered by bid strength.
        Could be 1 winner (sealed-bid) or a full ranking (Dutch auction).
        """

    @abstractmethod
    def reset(self) -> None:
        """Resets the protocol to its initial state."""


class NightTeamActionProtocol(ABC):
    """
    Handles collection of proposals from a team and resolves them into a single action.
    """

    @abstractmethod
    def begin_round(self, state: GameState, team_players: Sequence[Player]):
        """Initialize for a new round of proposals from the team."""
        pass

    @abstractmethod
    def collect_proposal(self, proposal: Action, state: GameState):
        """Collect a proposal from a team member."""
        pass

    @abstractmethod
    def all_proposals_collected(self, state: GameState, team_players: Sequence[Player]) -> bool:
        """Check if all expected proposals are collected or if the phase should end."""
        pass

    @abstractmethod
    def resolve_action(self, state: GameState, team_players: Sequence[Player]) -> Optional[Action]:
        """Resolve collected proposals into a single team action."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the protocol to its initial state."""
        pass


class WerewolfEliminationProtocol(NightTeamActionProtocol):
    def __init__(self, resolver: TeamDecisionProtocol):
        self._resolver = resolver
        self._proposals: List[EliminateProposalAction] = []

    def reset(self) -> None:
        self._proposals = []

    def begin_round(self, state: GameState, team_players: Sequence[Player]):
        self._proposals = []

    def collect_proposal(self, proposal: Action, state: GameState):
        if isinstance(proposal, EliminateProposalAction):
            actor_player = next((p for p in state.players if p.id == proposal.actor_id), None)
            if actor_player and actor_player.alive and actor_player.role.team == Team.WEREWOLVES:
                self._proposals.append(proposal)

    def all_proposals_collected(self, state: GameState, team_players: Sequence[Player]) -> bool:
        # For simplicity, Moderator will decide when to call resolve. Can be enhanced.
        return True

    def resolve_action(self, state: GameState, team_players: Sequence[Player]) -> Optional[EliminateProposalAction]:
        alive_team_players = [p for p in team_players if p.alive]
        if not alive_team_players:
            return None
        return self._resolver.resolve(alive_team_players, self._proposals)


# ----------------- discussion patterns ----------------------------------- #
class DiscussionProtocol(ABC):
    """Drives the order/shape of daytime conversation."""

    @abstractmethod
    def begin(self, state: GameState) -> None:
        """Optional hook – initialise timers, round counters…"""

    @property
    @abstractmethod
    def discussion_rule(self) -> str:
        """A string describing the discussion rule in effect."""
        """Optional hook – initialise timers, round counters…"""

    @abstractmethod
    def speakers_for_tick(self, state: GameState) -> Sequence[str]:
        """
        Return the IDs that are *allowed to send a chat action* this tick.
        Return an empty sequence when the discussion phase is over.
        """

    @abstractmethod
    def is_discussion_over(self, state: GameState) -> bool:
        """Returns True if the entire discussion (including any preliminary phases like bidding) is complete."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the protocol to its initial state."""
        pass

    def process_actions(self, actions: List[Action], expected_speakers: Sequence[str], state: GameState) -> None:
        """
        Processes a batch of actions. Depending on the protocol's state (e.g., bidding or chatting),
        it will handle relevant actions (like BidAction or ChatAction) from expected_speakers.
        """
        for act in actions:
            if isinstance(act, ChatAction):
                all_player_ids = [p.id for p in state.players]
                mentioned_ids = _extract_player_ids_from_string(act.message, all_player_ids)
                if expected_speakers and act.actor_id in expected_speakers:
                    data = ChatDataEntry(
                        actor_id=act.actor_id,
                        message=act.message,
                        reasoning=act.reasoning,
                        mentioned_player_ids=mentioned_ids,
                        perceived_threat_level=act.perceived_threat_level
                    )
                    state.add_history_entry(
                        description=f'Player "{act.actor_id}" (chat): {act.message}',
                        # Make public for general discussion
                        entry_type=HistoryEntryType.DISCUSSION,
                        public=True,
                        source=act.actor_id,
                        data=data
                    )
                else:
                    state.add_history_entry(
                        description=f'Player "{act.actor_id}" (chat, out of turn): {act.message}',
                        entry_type=HistoryEntryType.DISCUSSION,  # Or a specific "INVALID_CHAT" type
                        visible_to=[act.actor_id],
                        public=False,
                        source=act.actor_id
                    )

    def call_for_actions(self, speakers: Sequence[str]) -> List[str]:
        """prepare moderator call for action for each player."""
        return [f'Player "{speaker_id}", it is your turn to speak.' for speaker_id in speakers]

    def prompt_speakers_for_tick(self, state: GameState, speakers: Sequence[str]) -> None:
        """
        Allows the protocol to make specific announcements or prompts to the current speakers for this tick.
        This method is called by the Moderator after speakers_for_tick() returns a non-empty list of speakers,
        and before process_actions().
        Implementations should use state.add_history_entry() to make announcements.
        These announcements are typically visible only to the speakers, unless they are general status updates.
        """
        call_for_actions = self.call_for_actions(speakers)
        for speaker_id, call_for_action in zip(speakers, call_for_actions):
            data = RequestVillagerToSpeakDataEntry(action_json_schema=json.dumps(ChatAction.schema_for_player()))
            state.add_history_entry(
                description=call_for_action,
                entry_type=HistoryEntryType.PROMPT_FOR_ACTION,
                public=False,
                visible_to=[speaker_id],
                data=data
            )


class RoundRobinDiscussion(DiscussionProtocol):
    def __init__(self, max_rounds: int = 1, assign_random_first_speaker: bool = True):
        """

        Args:
            max_rounds: rounds of discussion
            assign_random_first_speaker: If true, the first speaker will be determined at the beginning of
                the game randomly, while the order follow that of the player list. Otherwise, will start from the
                0th player from player list.
        """
        self.max_rounds = max_rounds
        self._queue: deque[str] = deque()
        self._assign_random_first_speaker = assign_random_first_speaker
        self._player_ids = None
        self._first_player_idx = None

    def reset(self) -> None:
        self._queue = deque()

    @property
    def discussion_rule(self) -> str:
        return f"Players speak in round-robin order for {self.max_rounds} round(s)."

    def begin(self, state):
        if self._player_ids is None:
            # initialize player_ids once.
            self._player_ids = deque(state.all_player_ids)
            if self._assign_random_first_speaker:
                self._player_ids.rotate(random.randrange(len(self._player_ids)))

        # Reset queue
        player_order = [pid for pid in self._player_ids if state.is_alive(pid)]
        self._queue = deque(player_order * self.max_rounds)
        if self.max_rounds > 0 and self._queue:
            data = DiscussionOrderDataEntry(chat_order_of_player_ids=player_order)
            state.add_history_entry(
                description="Discussion phase begins. Players will speak in round-robin order. "
                            f"Starting from player {player_order[0]} with the following order: {player_order} "
                            f"for {self.max_rounds} round(s).",
                entry_type=HistoryEntryType.DISCUSSION_ORDER,
                public=True,
                data=data
            )

    def speakers_for_tick(self, state):
        return [self._queue.popleft()] if self._queue else []

    def is_discussion_over(self, state: GameState) -> bool:
        return not self._queue  # Over if queue is empty


class RandomOrderDiscussion(DiscussionProtocol):
    def __init__(self):
        self._iters = None
        self._steps = 0

    def reset(self) -> None:
        self._iters = None
        self._steps = 0

    @property
    def discussion_rule(self) -> str:
        return "Players speak in a random order for one full round."

    def begin(self, state):
        self._iters = itertools.cycle(random.sample(
            [p.id for p in state.alive_players()],
            k=len(state.alive_players())
        ))
        self._steps = len(state.alive_players())  # one full round
        if self._steps > 0:
            state.add_history_entry(
                description="Discussion phase begins. Players will speak in random order.",
                entry_type=HistoryEntryType.PHASE_CHANGE,
                public=True
            )

    def speakers_for_tick(self, state):
        if self._steps == 0:
            return []
        self._steps -= 1
        return [next(self._iters)]

    def is_discussion_over(self, state: GameState) -> bool:
        return self._steps == 0


class ParallelDiscussion(DiscussionProtocol):
    """
    Everyone may talk for `ticks` chat turns.
    Useful when you want simultaneous / overlapping chat.
    """

    def __init__(self, ticks: int = 3):
        self.ticks = ticks
        self._remaining = 0

    def reset(self) -> None:
        self._remaining = 0

    @property
    def discussion_rule(self) -> str:
        return f"All players may speak simultaneously for {self.ticks} tick(s)."

    def begin(self, state):
        self._remaining = self.ticks
        if self.ticks > 0:
            state.add_history_entry(
                description="Parallel discussion phase begins. All players may speak.",
                entry_type=HistoryEntryType.PHASE_CHANGE,
                public=True
            )

    def speakers_for_tick(self, state):
        if self._remaining == 0:
            return []
        self._remaining -= 1
        return [p.id for p in state.alive_players()]

    def call_for_actions(self, speakers: Sequence[str]) -> List[str]:
        return [f"Parallel discussion: All designated players may speak now or remain silent. "
                f"({self._remaining + 1} speaking opportunities remaining, including this one)."] * len(speakers)

    def is_discussion_over(self, state: GameState) -> bool:
        return self._remaining == 0


class TurnByTurnBiddingDiscussion(DiscussionProtocol):
    """
    A discussion protocol where players bid for the right to speak each turn.
    This protocol manages the entire bid-speak-bid-speak loop.
    """
    SPEAKING_PHASE = 'speaking'
    BIDDING_PHASE = 'bidding'

    def __init__(self, bidding: BiddingProtocol, max_turns: int = 8, bid_result_public: bool = True):
        self.bidding = bidding
        self.max_turns = max_turns
        self._phase = self.BIDDING_PHASE  # Can be "bidding" or "speaking"
        self._turns_taken = 0
        self._speaker: Optional[str] = None
        self._all_passed = False
        self._bid_result_public = bid_result_public

    def reset(self) -> None:
        self.bidding.reset()
        self._phase = self.BIDDING_PHASE
        self._turns_taken = 0
        self._speaker = None
        self._all_passed = False

    @property
    def discussion_rule(self) -> str:
        return "\n".join([
            f"Players bid for the right to speak each turn for up to {self.max_turns} turns.",
            f"The bidding rule is: {self.bidding.__class__.__name__}.",
            self.bidding.bidding_rules,
            f"If everyone bids 0, moderator will directly move on to day voting and no one speaks."
        ])

    def begin(self, state: GameState) -> None:
        self.reset()
        self.bidding.begin(state)  # Initial setup for the first bidding round

    def is_discussion_over(self, state: GameState) -> bool:
        return self._turns_taken >= self.max_turns or self._all_passed

    def speakers_for_tick(self, state: GameState) -> Sequence[str]:
        if self.is_discussion_over(state):
            return []

        if self._phase == self.BIDDING_PHASE:
            return [p.id for p in state.alive_players()]
        elif self._phase == self.SPEAKING_PHASE:
            return [self._speaker] if self._speaker else []
        return []

    def process_actions(self, actions: List[Action], expected_speakers: Sequence[str], state: GameState) -> None:
        if self._phase == self.BIDDING_PHASE:
            self.bidding.process_incoming_bids(actions, state)

            # Handle players who didn't bid (timed out) by assuming a bid of 0
            all_alive_player_ids = [p.id for p in state.alive_players()]
            if hasattr(self.bidding, '_bids'):
                for player_id in all_alive_player_ids:
                    if player_id not in self.bidding._bids:
                        default_bid = BidAction(actor_id=player_id, amount=0, day=state.day_count, phase=state.phase.value)
                        self.bidding.accept(default_bid, state)

            bids = getattr(self.bidding, '_bids', {})
            if len(bids) >= len(all_alive_player_ids) and all(amount == 0 for amount in bids.values()):
                self._all_passed = True
                state.add_history_entry(
                    description="All players passed on speaking. Discussion ends.",
                    entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                    public=True
                )
                return

            # Once all bids are in (or a timeout, handled by moderator's single tick), determine the winner
            winner_list = self.bidding.outcome(state)
            self._speaker = winner_list[0] if winner_list else None

            if self._speaker:
                data = BidResultDataEntry(
                    winner_player_ids=[self._speaker],
                    bid_overview=self.bidding.bids,
                    mentioned_players_in_previous_turn=self.bidding.get_last_mentioned(state)[0]
                )
                overview_text = ', '.join([f'{k}: {v}' for k, v in self.bidding.bids.items()])
                state.add_history_entry(
                    description=f"Player {self._speaker} won the bid and will speak next.\n"
                                f"Bid overview - {overview_text}.",
                    entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                    public=self._bid_result_public,
                    data=data
                )
                self._phase = self.SPEAKING_PHASE
            else:
                # No one to speak, advance turn count and bid again
                self._turns_taken += 1
                if not self.is_discussion_over(state):
                    self.bidding.begin(state)  # Prepare for next bidding round

        elif self._phase == self.SPEAKING_PHASE:
            # Process the chat action from the designated speaker
            super().process_actions(actions, expected_speakers, state)
            self._turns_taken += 1

            # After speaking, transition back to bidding for the next turn
            if not self.is_discussion_over(state):
                self._phase = self.BIDDING_PHASE
                self._speaker = None
                self.bidding.begin(state)  # Reset bids and find new mentioned players

    def prompt_speakers_for_tick(self, state: GameState, speakers: Sequence[str]) -> None:
        if self._phase == self.BIDDING_PHASE:
            data = {"action_json_schema": json.dumps(BidAction.schema_for_player())}
            state.add_history_entry(
                description=(
                    f"A new round of discussion begins. Place bid for a chance to speak. "
                    f"{self.max_turns - self._turns_taken} turns left to speak."
                ),
                entry_type=HistoryEntryType.PROMPT_FOR_ACTION,
                public=True,
                data=data
            )
        elif self._phase == self.SPEAKING_PHASE and self._speaker:
            super().prompt_speakers_for_tick(state, speakers)


# ----------------- voting patterns --------------------------------------- #
class SimultaneousMajority(VotingProtocol):
    def __init__(self):
        self._ballots: Dict[str, str] = {}  # actor_id (str) -> target_id (str)
        self._expected_voters: List[str] = []
        self._potential_targets: List[str] = []
        self._current_game_state: Optional[GameState] = None  # To store state from begin_voting
        self._elected = None
        self._done_tallying = False

    def reset(self) -> None:
        self._ballots = {}
        self._expected_voters = []
        self._potential_targets = []
        self._current_game_state = None
        self._elected = None
        self._done_tallying = False

    @property
    def voting_rule(self) -> str:
        return ("Simultaneous majority vote. Player with the most votes is exiled. "
                "Ties result in random selection amongst the top ties. "
                "If no valid vote available (if all casted abstained votes), "
                "will result in random elimination of one player.")

    def begin_voting(self, state: GameState, alive_voters: Sequence[Player], potential_targets: Sequence[Player]):
        self._ballots = {}
        # Ensure voters and targets are alive at the start of voting
        self._expected_voters = [p.id for p in alive_voters if p.alive]
        self._potential_targets = [p.id for p in potential_targets if p.alive]
        self._current_game_state = state  # Store the game state reference

    def collect_votes(self, player_actions: Dict[str, Action], state: GameState, expected_voters: List[str]):
        for actor_id, action in player_actions.items():
            self.collect_vote(action, state)
        # set default for all expected voter
        for player_id in expected_voters:
            self._ballots.setdefault(player_id, "-1")

    def collect_vote(self, vote_action: Action, state: GameState):
        actor_player = state.get_player_by_id(vote_action.actor_id)
        if not isinstance(vote_action, VoteAction):
            state.add_history_entry(
                description=f'Invalid vote attempt by player "{vote_action.actor_id}". Not a VoteAction; submitted {vote_action.__class__.__name__} instead. Cast as abstained vote.',
                entry_type=HistoryEntryType.ERROR,
                public=False,
                visible_to=self._expected_voters,
                data={}
            )
            self._ballots[vote_action.actor_id] = "-1"
            return

        if state.phase == Phase.NIGHT:
            data_entry_class = WerewolfNightVoteDataEntry
        else:
            data_entry_class = DayExileVoteDataEntry

        data = data_entry_class(
            actor_id=vote_action.actor_id,
            target_id=vote_action.target_id,
            reasoning=vote_action.reasoning,
            perceived_threat_level=vote_action.perceived_threat_level
        )

        # Voter must be expected and alive at the moment of casting vote
        if actor_player and actor_player.alive and vote_action.actor_id in self._expected_voters:
            # Prevent re-voting
            if vote_action.actor_id in self._ballots:
                state.add_history_entry(
                    description=f'Invalid vote attempt by "{vote_action.actor_id}", already voted.',
                    entry_type=HistoryEntryType.ERROR,
                    public=False,
                    visible_to=self._expected_voters,
                    data=data
                )
                return

            if vote_action.target_id in self._potential_targets:
                self._ballots[vote_action.actor_id] = vote_action.target_id

                # Determine DataEntry type based on game phase
                state.add_history_entry(
                    description=f'Player "{data.actor_id}" voted to eliminate "{data.target_id}". ',
                    entry_type=HistoryEntryType.VOTE_ACTION,
                    public=False,
                    visible_to=self._expected_voters,
                    data=data
                )
            else:
                self._ballots[vote_action.actor_id] = "-1"
                state.add_history_entry(
                    description=f'Invalid vote attempt by "{vote_action.actor_id}".',
                    entry_type=HistoryEntryType.ERROR,
                    public=False,
                    visible_to=self._expected_voters,
                    data=data
                )
                return
        else:
            state.add_history_entry(
                description=f"Invalid vote attempt by {vote_action.actor_id}.",
                entry_type=HistoryEntryType.ERROR,
                public=False,
                data=data
            )

    def _tally_votes(self, state: GameState) -> str | None:
        if not self.done():
            # Voting is not yet complete for this protocol.
            raise Exception("Voting is not done yet.")

        if self._done_tallying:
            return self._elected
        self._done_tallying = True
        counts = Counter(v for v in self._ballots.values() if v is not None and v != "-1").most_common()
        if not counts:
            self._elected = random.choice(self._potential_targets)
        else:
            _, top_votes = counts[0]
            self._elected = random.choice([v for v, c in counts if c == top_votes])
        return self._elected

    def get_voting_prompt(self, state: GameState, player_id: str) -> str:
        target_options = [p_id for p_id in self._potential_targets if
                          state.get_player_by_id(p_id) and state.get_player_by_id(p_id).alive]
        return f'Player "{player_id}", please cast your vote. Options: {target_options} or Abstain ("-1").'

    def get_current_tally_info(self, state: GameState) -> Dict[str, int]:
        return Counter(self._ballots.values())

    def get_next_voters(self) -> List[str]:
        # For simultaneous, all expected voters vote at once, and only once.
        return [voter for voter in self._expected_voters if voter not in self._ballots]

    def done(self) -> bool:
        # The voting is considered "done" after one tick where voters were requested.
        # The moderator will then call tally_votes.
        return all(voter in self._ballots for voter in self._expected_voters)

    def get_valid_targets(self) -> List[str]:
        # Return a copy of targets that were valid (alive) at the start of voting.
        return list(self._potential_targets)

    def get_elected(self) -> str | None:  # Return type matches tally_votes
        if not self.done():
            raise Exception("Voting is not done yet.")
        return self._tally_votes(self._current_game_state)


class SequentialFirstToK(VotingProtocol):
    """
    Everyone votes in turn; as soon as a candidate reaches K votes he is exiled.
    """

    def __init__(self, threshold: int):
        self.threshold = threshold
        self._ballots: Dict[str, str] = {}  # actor_id (str) -> target_id (str)
        self._expected_voters: List[str] = []
        self._potential_targets: List[str] = []
        self._current_game_state: Optional[GameState] = None  # To store state from begin_voting

    def reset(self) -> None:
        self._ballots = {}
        self._expected_voters = []
        self._potential_targets = []
        self._current_game_state = None  # To store state from begin_voting

    @property
    def voting_rule(self) -> str:
        return f"Sequential voting. First player to reach {self.threshold} votes is exiled."

    def begin_voting(self, state: GameState, alive_voters: Sequence[Player], potential_targets: Sequence[Player]):
        self._ballots = {}
        self._expected_voters = [p.id for p in alive_voters]  # Or manage sequential turns
        self._potential_targets = [p.id for p in potential_targets]
        self._current_game_state = state

    def collect_votes(self, player_actions: Dict[str, Action], state: GameState, expected_voters: List[str]):
        for actor_id, action in player_actions.items():
            self.collect_vote(action, state)
        # set default for all expected voter
        for player_id in expected_voters:
            self._ballots.setdefault(player_id, "-1")

    def collect_vote(self, vote_action: Action, state: GameState):
        if not isinstance(vote_action, VoteAction): return
        # Basic validation, could be enhanced for sequential order
        if any(p.id == vote_action.actor_id and p.alive for p in state.players):
            self._ballots[vote_action.actor_id] = vote_action.target_id  # Overwrites previous vote if any

    def all_votes_collected(self, state: GameState, alive_voters: Sequence[Player]) -> bool:
        # For sequential, this might mean one player reached threshold or all voted.
        # Simplified: if tally_votes finds a winner, it's "collected" for resolution.
        return len(self._ballots) >= len(self._expected_voters)  # Or some other condition

    def _tally_votes(self, state: GameState) -> str | None:
        from collections import Counter
        tally = Counter(self._ballots.values())  # Using internally stored ballots
        for candidate, votes in tally.items():
            if votes >= self.threshold:
                return candidate
        return None

    def get_voting_prompt(self, state: GameState, player_id: str) -> str:
        # This protocol implies a turn order, which is not explicitly managed here yet.
        # Assuming the Moderator handles whose turn it is.
        current_tally = self.get_current_tally_info(state)
        tally_str_parts = []
        for target_id, votes in sorted(current_tally.items(), key=lambda x: x[1],
                                       reverse=True):  # Sort for consistent display
            tally_str_parts.append(f"{target_id}: {votes} vote(s)")
        tally_str = "; ".join(tally_str_parts) if tally_str_parts else "No votes yet."
        target_names = [f"{p.id}" for p in state.alive_players() if p.id in self._potential_targets]
        return (f"{player_id}, it's your turn to vote. Current tally: {tally_str}. "
                f"Options: {', '.join(target_names)} or Abstain ('-1'). {self.threshold} votes needed to exile.")

    def get_current_tally_info(self, state: GameState) -> Dict[str, int]:
        return Counter(tgt for tgt in self._ballots.values() if tgt != "-1")  # Excludes abstentions

    def get_next_voters(self) -> List[str]:
        # This needs more sophisticated turn management if used by Moderator.
        # For now, assuming Moderator handles turns.
        # This could return the next expected voter if not all have voted and no one reached threshold.
        return []  # Placeholder

    def done(self) -> bool:
        # Done if someone reached threshold or all expected voters have voted.
        return self._tally_votes(self._current_game_state) is not None or len(self._ballots) >= len(self._expected_voters)


# ----------------- decision protocols --------------------------------------- #

class MajorityEliminateResolver(TeamDecisionProtocol):
    def reset(self) -> None:
        pass

    def resolve(self, team_players, proposals):
        if not proposals:  # wolves forgot to act
            return None
        votes = Counter(a.target_id for a in proposals)
        top_id, top_votes = votes.most_common(1)[0]
        # tie? randomly pick among top
        top_targets = [pid for pid, v in votes.items() if v == top_votes]
        victim = random.choice(top_targets)
        # "alpha" wolf = lowest id
        alpha_id = min(p.id for p in team_players)
        return EliminateProposalAction(actor_id=alpha_id, target_id=victim)


class AlphaFirstEliminateResolver(TeamDecisionProtocol):
    def reset(self) -> None:
        pass

    def resolve(self, team_players, proposals):
        # deterministic leader
        alpha = min(team_players, key=lambda p: p.id)
        alpha_vote = next(
            (p for p in proposals if p.actor_id == alpha.id), None)
        if alpha_vote:
            return EliminateProposalAction(actor_id=alpha.id, target_id=alpha_vote.target_id)
        # fall back to majority if alpha forgot
        return MajorityEliminateResolver().resolve(team_players, proposals)


# ----------------- bidding protocols --------------------------------------- #

class UrgencyBiddingProtocol(BiddingProtocol):
    """
    A bidding protocol based on the Werewolf Arena paper.
    - Agents bid with an urgency level (0-4).
    - Highest bidder wins.
    - Ties are broken by prioritizing players mentioned in the previous turn.
    """
    @property
    def bidding_rules(self) -> str:
        return "\n".join([
            "Urgency-based bidding. Players bid with an urgency level (0-4).",
            "0: I would like to observe and listen for now.",
            "1: I have some general thoughts to share with the group.",
            "2: I have something critical and specific to contribute to this discussion.",
            "3: It is absolutely urgent for me to speak next.",
            "4: Someone has addressed me directly and I must respond.",
            "Highest bidder wins."
            "Ties are broken by the following priority: (1) players mentioned in the previous turn's chat, "
            "(2) the least spoken player, (3) round robin order of the player list."
        ])

    @property
    def bids(self) -> Dict[str, int]:
        return dict(**self._bids)

    def __init__(self):
        self._bids: Dict[str, int] = {}
        self._mentioned_last_turn: List[str] = []

    def reset(self) -> None:
        self._bids = {}
        self._mentioned_last_turn = []

    def begin(self, state: GameState) -> None:
        """Called at the start of a bidding round to identify recently mentioned players."""
        self.reset()
        # Find the very last chat entry in the history to check for mentions
        self._mentioned_last_turn, last_chat_message = self.get_last_mentioned(state)

        if last_chat_message:
            if self._mentioned_last_turn:
                state.add_history_entry(
                    description=f"Players mentioned last turn (priority in ties): {self._mentioned_last_turn}",
                    entry_type=HistoryEntryType.BIDDING_INFO,
                    public=True  # So everyone knows who has priority
                )

    def accept(self, bid: BidAction, state: GameState) -> None:
        if 0 <= bid.amount <= 4:
            self._bids[bid.actor_id] = bid.amount
            data = BidDataEntry(
                actor_id=bid.actor_id,
                reasoning=bid.reasoning,
                perceived_threat_level=bid.perceived_threat_level,
                bid_amount=bid.amount
            )
            state.add_history_entry(
                description=f"Player {bid.actor_id} submitted bid=({bid.amount}).",
                entry_type=HistoryEntryType.BIDDING_INFO,
                public=False,
                visible_to=[bid.actor_id],
                data=data
            )
        else:
            # Invalid bid amount is treated as a bid of 0
            self._bids[bid.actor_id] = 0
            state.add_history_entry(
                description=f"Player {bid.actor_id} submitted an invalid bid amount ({bid.amount}). Treated as 0.",
                entry_type=HistoryEntryType.ERROR,
                public=False,
                visible_to=[bid.actor_id]
            )

    def process_incoming_bids(self, actions: List[Action], state: GameState) -> None:
        for act in actions:
            if isinstance(act, BidAction):
                self.accept(act, state)

    def is_finished(self, state: GameState) -> bool:
        # This bidding round is considered "finished" when all alive players have bid.
        return len(self._bids) >= len(state.alive_players())

    def outcome(self, state: GameState) -> list[str]:
        if not self._bids:
            # If no one bids, deterministically pick the first alive player to speak.
            alive_players = state.alive_players()
            return [alive_players[0].id] if alive_players else []

        max_bid = max(self._bids.values())
        highest_bidders = sorted([pid for pid, amt in self._bids.items() if amt == max_bid])

        if len(highest_bidders) == 1:
            return highest_bidders

        # Tie-breaking logic
        candidates = highest_bidders

        # Rule 1: Players mentioned in the last turn
        mentioned_in_tie = [pid for pid in candidates if pid in self._mentioned_last_turn]
        if mentioned_in_tie:
            candidates = mentioned_in_tie

        if len(candidates) == 1:
            return candidates

        # Rule 2: The least spoken individual
        speech_counts = Counter(
            entry.data.actor_id
            for day_history in state.history.values()
            for entry in day_history
            if entry.entry_type == HistoryEntryType.DISCUSSION and isinstance(entry.data, ChatDataEntry)
        )

        candidate_speech_counts = {pid: speech_counts.get(pid, 0) for pid in candidates}
        min_spoken = min(candidate_speech_counts.values())
        least_spoken_candidates = sorted(
            [pid for pid, count in candidate_speech_counts.items() if count == min_spoken])

        if len(least_spoken_candidates) == 1:
            return least_spoken_candidates

        candidates = least_spoken_candidates

        # Rule 3: Round robin order of the player list in state
        for pid in state.all_player_ids:
            if pid in candidates:
                return [pid]

        # This part should be unreachable if candidates is a subset of all_player_ids
        return [candidates[0]] if candidates else []


class SequentialVoting(VotingProtocol):
    """
    Players vote one by one in a sequence. Each player is shown the current
    tally before casting their vote. All players in the initial list of
    voters get a turn.
    """

    def __init__(self, assign_random_first_voter: bool = True):
        self._ballots: Dict[str, str] = {}  # actor_id (str) -> target_id (str)
        self._potential_targets: List[str] = []
        self._voter_queue: List[str] = []  # Order of players to vote
        self._expected_voters: List[str] = []
        self._current_voter_index: int = 0  # Index for _voter_queue
        self._current_game_state: Optional[GameState] = None # To store state from begin_voting
        self._elected = None
        self._done_tallying = False
        self._assign_random_first_voter = assign_random_first_voter
        self._player_ids = None

    def reset(self) -> None:
        self._ballots = {}
        self._potential_targets = []
        self._expected_voters = []
        self._voter_queue = []
        self._current_voter_index = 0
        self._current_game_state = None
        self._elected = None
        self._done_tallying = False

    @property
    def voting_rule(self) -> str:
        return ("Sequential voting. Players vote one by one. Player with the most votes after all have voted is exiled."
                " Ties are broken randomly.")

    def begin_voting(self, state: GameState, alive_voters: Sequence[Player], potential_targets: Sequence[Player]):
        if self._player_ids is None:
            # initialize player_ids once.
            self._player_ids = deque(state.all_player_ids)
            if self._assign_random_first_voter:
                self._player_ids.rotate(random.randrange(len(self._player_ids)))
        alive_voter_ids = [p.id for p in alive_voters]
        alive_voter_ids_set = set(alive_voter_ids)
        self._ballots = {}
        self._expected_voters = [pid for pid in self._player_ids if pid in alive_voter_ids_set]
        self._potential_targets = [p.id for p in potential_targets]
        # The order of voting can be based on player ID, a random shuffle, or the order in alive_voters
        # For simplicity, using the order from alive_voters.
        self._voter_queue = list(self._expected_voters)
        self._current_voter_index = 0
        self._current_game_state = state # Store the game state reference

        if self._expected_voters:
            data = VoteOrderDataEntry(vote_order_of_player_ids=self._expected_voters)
            state.add_history_entry(
                description=f"Voting starts from player {self._expected_voters[0]} "
                            f"with the following order: {self._expected_voters}",
                entry_type=HistoryEntryType.VOTE_ORDER,
                public=False,
                visible_to=alive_voter_ids,
                data=data
            )

    def get_voting_prompt(self, state: GameState, player_id: str) -> str:
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

        return (f"{player_id}, it is your turn to vote. "
                f"Current tally: {tally_str}. "
                f"Options: {options_str} or Abstain (vote for -1).")

    def collect_votes(self, player_actions: Dict[str, Action], state: GameState, expected_voters: List[str]):
        if self.done():
            return

        # In sequential voting, expected_voters should contain exactly one player.
        if not expected_voters:
            # This case should ideally not be reached if `done()` is false.
            # If it is, advancing the turn might be a safe way to prevent a stall.
            self._current_voter_index += 1
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
            state.add_history_entry(
                description=f"Action ({vote_action.kind}) received from {vote_action.actor_id}, but voting is already complete.",
                entry_type=HistoryEntryType.ERROR,
                public=False,
                visible_to=[vote_action.actor_id]
            )
            return

        expected_voter_id = self._voter_queue[self._current_voter_index]
        if vote_action.actor_id != expected_voter_id:
            state.add_history_entry(
                description=f"Action ({vote_action.kind}) received from {vote_action.actor_id}, but it is {expected_voter_id}'s turn.",
                entry_type=HistoryEntryType.ERROR,
                public=False,  # Or public if strict turn enforcement is announced
                visible_to=[vote_action.actor_id, expected_voter_id]
            )
            return

        actor_player = next((p for p in state.players if p.id == vote_action.actor_id), None)
        if actor_player and actor_player.alive:
            description_for_history = ""
            involved_players_list = [vote_action.actor_id]  # Actor is always involved
            data = None
            if isinstance(vote_action, NoOpAction):
                self._ballots[vote_action.actor_id] = "-1"  # Treat NoOp as abstain
                description_for_history = f"{vote_action.actor_id} chose to NoOp (treated as Abstain)."

            elif isinstance(vote_action, VoteAction):  # This must be true if not NoOpAction
                target_display: str
                recorded_target_id = vote_action.target_id
                if vote_action.target_id != "-1" and vote_action.target_id not in self._potential_targets:
                    # Invalid target chosen for VoteAction
                    state.add_history_entry(
                        description=f"{vote_action.actor_id} attempted to vote for {vote_action.target_id} (invalid target). Vote recorded as Abstain.",
                        entry_type=HistoryEntryType.ERROR,
                        public=False,
                        visible_to=[vote_action.actor_id]
                    )
                    recorded_target_id = "-1"  # Treat invalid target as abstain
                    target_display = f"Invalid Target ({vote_action.target_id}), recorded as Abstain"
                elif vote_action.target_id == "-1":
                    # Explicit Abstain via VoteAction
                    target_display = "Abstain"
                    # recorded_target_id is already "-1"
                else:
                    # Valid target chosen for VoteAction
                    target_display = f"{vote_action.target_id}"
                    involved_players_list.append(vote_action.target_id)  # Add valid target to involved

                self._ballots[vote_action.actor_id] = recorded_target_id
                description_for_history = f"{vote_action.actor_id} has voted for {target_display}."

                # Add data entry for the vote
                data_entry_class = DayExileVoteDataEntry if state.phase == Phase.DAY else WerewolfNightVoteDataEntry
                data = data_entry_class(
                    actor_id=vote_action.actor_id,
                    target_id=recorded_target_id,
                    reasoning=vote_action.reasoning,
                    perceived_threat_level=vote_action.perceived_threat_level
                )

            state.add_history_entry(
                description=description_for_history,
                entry_type=HistoryEntryType.VOTE_ACTION,
                public=False,
                visible_to=self._expected_voters,
                data=data
            )
            self._current_voter_index += 1
        else:  # Player not found, not alive, or (redundantly) not their turn
            state.add_history_entry(
                description=f"Invalid action ({vote_action.kind}) attempt by {vote_action.actor_id} (player not found, not alive, or not their turn). Action not counted.",
                entry_type=HistoryEntryType.ERROR,
                public=False,
                visible_to=[vote_action.actor_id]
            )
            # If voter was expected but found to be not alive, advance turn to prevent stall
            if vote_action.actor_id == expected_voter_id:  # Implies actor_player was found but not actor_player.alive
                self._current_voter_index += 1

    def _tally_votes(self, state: GameState) -> Optional[str]:
        if not self.done():
            # Voting is not yet complete for this protocol.
            raise Exception("Voting is not done yet.")

        if self._done_tallying:
            return self._elected
        self._done_tallying = True

        counts = Counter(v for v in self._ballots.values() if v is not None and v != "-1").most_common()
        if not counts:
            self._elected = random.choice(self._potential_targets)
        else:
            _, top_votes = counts[0]
            self._elected = random.choice([v for v, c in counts if c == top_votes])
        return self._elected

    def get_current_tally_info(self, state: GameState) -> Dict[str, int]:
        # Returns counts of non-abstain votes for valid targets
        return Counter(
            target_id for target_id in self._ballots.values()
            if target_id != "-1" and target_id in self._potential_targets
        )

    def get_next_voters(self) -> List[str]:
        if not self.done():
            # Ensure _current_voter_index is within bounds before accessing
            if self._current_voter_index < len(self._voter_queue):
                return [self._voter_queue[self._current_voter_index]]
        return []

    def done(self) -> bool:
        if not self._voter_queue:  # No voters were ever in the queue
            return True
        return self._current_voter_index >= len(self._voter_queue)

    def get_valid_targets(self) -> List[str]:
        return list(self._potential_targets)

    def get_elected(self) -> Optional[str]:
        if not self.done():
            raise Exception("Voting is not done yet.")
        return self._tally_votes(self._current_game_state)
