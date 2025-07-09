from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Optional  # Added Optional
import random
from collections import Counter

from .states import GameState, HistoryEntryType
from .roles import Player, Team
from .actions import EliminateProposalAction, BidAction, Action, ChatAction, VoteAction, NoOpAction


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
    def process_actions(self, actions: List[Action], expected_speakers: Sequence[str], state: GameState) -> None:
        """
        Processes a batch of actions. Depending on the protocol's state (e.g., bidding or chatting),
        it will handle relevant actions (like BidAction or ChatAction) from expected_speakers.
        """
        pass

    @abstractmethod
    def prompt_speakers_for_tick(self, state: GameState, speakers: Sequence[str]) -> None:
        """
        Allows the protocol to make specific announcements or prompts to the current speakers for this tick.
        This method is called by the Moderator after speakers_for_tick() returns a non-empty list of speakers,
        and before process_actions().
        Implementations should use state.add_history_entry() to make announcements.
        These announcements are typically visible only to the speakers, unless they are general status updates.
        """
        pass

    @abstractmethod
    def is_discussion_over(self, state: GameState) -> bool:
        """Returns True if the entire discussion (including any preliminary phases like bidding) is complete."""
        pass


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
    def tally_votes(self, state: GameState) -> str | None:
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
    def get_elected(self) -> str:
        """get the final elected individual"""


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


class BiddingProtocol(ABC):
    """Drives one auction round and returns the winner(s)."""

    @abstractmethod
    def begin(self, state: GameState) -> None: ...

    @abstractmethod
    def accept(self, bid: BidAction, state: GameState) -> None: ...

    @abstractmethod
    def process_incoming_bids(self, actions: List[Action], state: GameState) -> None:
        """Processes a batch of actions, handling BidActions by calling self.accept()."""
        pass

    @abstractmethod
    def is_finished(self, state: GameState) -> bool: ...

    @abstractmethod
    def outcome(self, state: GameState) -> list[int]:
        """ # Return type should be list[str] for player IDs
        Return list of player-ids, ordered by bid strength.
        Could be 1 winner (sealed-bid) or a full ranking (Dutch auction).
        """


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


class WerewolfEliminationProtocol(NightTeamActionProtocol):
    def __init__(self, resolver: TeamDecisionProtocol):
        self._resolver = resolver
        self._proposals: List[EliminateProposalAction] = []

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


class RoundRobinDiscussion(DiscussionProtocol):
    def __init__(self, max_rounds: int = 1):
        self.max_rounds = max_rounds
        self._queue: list[str] = []

    @property
    def discussion_rule(self) -> str:
        return f"Players speak in round-robin order for {self.max_rounds} round(s)."

    def begin(self, state):
        # Reset queue
        self._queue = [p.id for p in state.alive_players()] * self.max_rounds
        if self.max_rounds > 0 and self._queue:
            state.add_history_entry(
                description="Discussion phase begins. Players will speak in round-robin order.",
                entry_type=HistoryEntryType.PHASE_CHANGE,
                public=True
            )

    def speakers_for_tick(self, state):
        return [self._queue.pop(0)] if self._queue else []

    def process_actions(self, actions: List[Action], expected_speakers: Sequence[str], state: GameState) -> None:
        for act in actions:
            if isinstance(act, ChatAction):
                if expected_speakers and act.actor_id in expected_speakers:
                    state.add_history_entry(
                        description=f"P{act.actor_id} (chat): {act.message}",  # Make public for general discussion
                        entry_type=HistoryEntryType.DISCUSSION,
                        public=True
                    )
                else:
                    state.add_history_entry(
                        description=f"P{act.actor_id} (chat, out of turn): {act.message}",
                        entry_type=HistoryEntryType.DISCUSSION,  # Or a specific "INVALID_CHAT" type
                        visible_to=[act.actor_id]
                    )

    def prompt_speakers_for_tick(self, state: GameState, speakers: Sequence[str]) -> None:
        if speakers:  # Typically one speaker for RoundRobin
            speaker_id = speakers[0]
            state.add_history_entry(
                description=f"P{speaker_id}, it is your turn to speak.",
                entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                public=False,
                visible_to=[speaker_id]
            )

    def is_discussion_over(self, state: GameState) -> bool:
        return not self._queue  # Over if queue is empty


class RandomOrderDiscussion(DiscussionProtocol):
    @property
    def discussion_rule(self) -> str:
        return "Players speak in a random order for one full round."

    def begin(self, state):
        import random
        import itertools
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

    def process_actions(self, actions: List[Action], expected_speakers: Sequence[str], state: GameState) -> None:
        for act in actions:
            if isinstance(act, ChatAction):
                if expected_speakers and act.actor_id in expected_speakers:
                    state.add_history_entry(
                        description=f"P{act.actor_id} (chat): {act.message}",
                        entry_type=HistoryEntryType.DISCUSSION,
                        visible_to=[act.actor_id]
                    )  # Consider making public like RoundRobin
                else:
                    state.add_history_entry(
                        description=f"P{act.actor_id} (chat, out of turn): {act.message}",
                        entry_type=HistoryEntryType.DISCUSSION,
                        visible_to=[act.actor_id]
                    )

    def prompt_speakers_for_tick(self, state: GameState, speakers: Sequence[str]) -> None:
        if speakers:  # Typically one speaker
            speaker_id = speakers[0]
            state.add_history_entry(
                description=f"P{speaker_id}, it is your turn to speak.",
                entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                public=False,
                visible_to=[speaker_id]
            )

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

    def process_actions(self, actions: List[Action], expected_speakers: Sequence[str], state: GameState) -> None:
        for act in actions:
            if isinstance(act, ChatAction):
                # In parallel, any alive player in expected_speakers can talk
                if expected_speakers and act.actor_id in expected_speakers:  # expected_speakers should be all alive players
                    state.add_history_entry(
                        description=f"P{act.actor_id} (chat): {act.message}",
                        entry_type=HistoryEntryType.DISCUSSION,
                        public=True  # Parallel chat is usually public
                    )

    def prompt_speakers_for_tick(self, state: GameState, speakers: Sequence[str]) -> None:
        if speakers:  # Indicates a speaking tick is active; speakers will be all alive players
            # The begin() method already announces the general start.
            # This prompt can confirm the active tick and remaining time.
            state.add_history_entry(
                description=f"Parallel discussion: All designated players may speak now. ({self._remaining + 1} speaking opportunities remaining, including this one).",
                entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                public=True  # General status update for parallel discussion
            )

    def is_discussion_over(self, state: GameState) -> bool:
        return self._remaining == 0


class BidDrivenDiscussion(DiscussionProtocol):
    """
    Wraps an *inner* DiscussionProtocol.
    Winner(s) of the auction decide the talk order.
    """

    def __init__(self,
                 bidding: BiddingProtocol,
                 inner: DiscussionProtocol):
        self.bidding = bidding
        self.inner = inner
        self._winners: list[str] = []
        self._is_winner_speaking_now: bool = False  # Added flag

    @property
    def discussion_rule(self) -> str:
        return f"Bidding phase to determine initial speaking rights, followed by: {self.inner.discussion_rule}"

    # -------- life-cycle ------------------------------------- #
    def begin(self, state):
        self.bidding.begin(state)
        self._winners = []  # Reset winners list
        self._is_winner_speaking_now = False
        self._phase = "bidding"
        state.add_history_entry(
            description="Bidding phase begins. Players may now submit bids.",
            entry_type=HistoryEntryType.BIDDING_INFO,
            public=True
        )

    def speakers_for_tick(self, state):
        if self._phase == "bidding":
            # If bidding is not finished, all alive players are potential bidders.
            if not self.bidding.is_finished(state):
                self._is_winner_speaking_now = False
                return [p.id for p in state.alive_players()]
            
            # Bidding is finished, transition to discussion
            self._is_winner_speaking_now = False
            # During bidding, no one is designated to "speak" via chat.
            # Bids are submitted as general actions.
            # Check if bidding is finished to potentially transition phase.
            # This check is now implicitly handled by the flow above and in process_actions.
            # This check might be better placed in process_actions after bids are processed.
            if self.bidding.is_finished(state):
                self._winners = self.bidding.outcome(state)
                self._phase = "discussion"
                state.add_history_entry(
                    description="Bidding has concluded. Discussion will proceed based on bid winners (if any), then normal discussion.",
                    entry_type=HistoryEntryType.BIDDING_INFO,
                    public=True
                )
                self.inner.begin(state)
            return [] # Bidding just finished, next tick will be winner or inner.speakers_for_tick

        # Discussion phase (after bidding)
        self._is_winner_speaking_now = False  # Reset before checking winners/inner
        if self._winners:  # If there are winners in the queue to speak
            speaker_id = self._winners.pop(0)
            self._is_winner_speaking_now = True
            return [speaker_id]  # Next winner speaks, remove from queue
        else:  # No more winners, delegate to inner protocol
            # self._is_winner_speaking_now remains False
            return self.inner.speakers_for_tick(state)

    def process_actions(self, actions: List[Action], expected_speakers: Sequence[str], state: GameState) -> None:
        if self._phase == "bidding":
            bid_actions = [act for act in actions if isinstance(act, BidAction)]
            self.bidding.process_incoming_bids(bid_actions, state)

            for act in actions:
                if not isinstance(act, BidAction):  # Log non-bid actions as errors/ignored
                    state.add_history_entry(
                        description=f"P{act.actor_id} sent a {act.kind.value} action during bidding phase. Action ignored.",
                        entry_type=HistoryEntryType.ERROR,
                        public=False,  # Or true if you want to announce ignored actions
                        visible_to=[act.actor_id]
                    )

            if self.bidding.is_finished(state):
                # Transition to discussion phase (already handled in speakers_for_tick if it's called after bids,
                # but good to ensure state consistency here too)
                if self._phase == "bidding":  # Check to prevent re-triggering if already transitioned
                    self._winners = self.bidding.outcome(state)
                    self._phase = "discussion"
                    state.add_history_entry(
                        description="Bidding has concluded. Discussion will now proceed.",
                        entry_type=HistoryEntryType.BIDDING_INFO,
                        public=True
                    )
                    self.inner.begin(state)  # Initialize the inner discussion protocol

        elif self._phase == "discussion":
            # expected_speakers would be a winner if self._winners had an item when speakers_for_tick was called.
            if expected_speakers:  # A speaker (winner or from inner protocol) was designated
                speaker_id = expected_speakers[0]
                processed_action_for_speaker = False
                for act in actions:
                    if isinstance(act, ChatAction) and act.actor_id == speaker_id:
                        # If the speaker was from the _winners list, it's a bid winner's chat.
                        # Otherwise, it's a chat for the inner protocol.
                        # The distinction in logging might be useful.
                        # For now, using generic chat processing.
                        self.inner.process_actions([act], expected_speakers,
                                                   state)  # Let inner handle its own chat logic
                        processed_action_for_speaker = True
                        break
                if not processed_action_for_speaker:
                    state.add_history_entry(
                        description=f"P{speaker_id} was expected to speak but did not provide a valid chat action.",
                        entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT, public=True
                    )
            else:  # No specific speaker expected this tick (e.g. parallel inner discussion, or end of round)
                self.inner.process_actions(actions, expected_speakers, state)

    def prompt_speakers_for_tick(self, state: GameState, speakers: Sequence[str]) -> None:
        if self._phase == "bidding":
            if speakers: # Should be all alive players if bidding is active
                state.add_history_entry(
                    description="Bidding phase is active. All players may submit a BidAction.",
                    entry_type=HistoryEntryType.BIDDING_INFO, # Or MODERATOR_ANNOUNCEMENT
                    public=True
                    # visible_to=speakers # Or public if everyone should know bidding is open
                )

        elif self._phase == "discussion":
            if speakers:  # Ensure there's actually a speaker designated for this tick
                if self._is_winner_speaking_now:
                    speaker_id = speakers[0]  # Should be the winner who was just designated
                    state.add_history_entry(
                        description=f"P{speaker_id}, as a bid winner, it is your turn to speak.",
                        entry_type=HistoryEntryType.MODERATOR_ANNOUNCEMENT,
                        public=False,
                        visible_to=[speaker_id]
                    )
                else:
                    # Delegate to inner discussion protocol's prompt method
                    self.inner.prompt_speakers_for_tick(state, speakers)

    def is_discussion_over(self, state: GameState) -> bool:
        return self._phase == "discussion" and not self._winners and self.inner.is_discussion_over(state)


# ----------------- voting patterns --------------------------------------- #
class SimultaneousMajority(VotingProtocol):
    def __init__(self):
        self._ballots: Dict[str, str] = {}  # actor_id (str) -> target_id (str)
        self._expected_voters: List[str] = []
        self._potential_targets: List[str] = []
        self._current_game_state: Optional[GameState] = None # To store state from begin_voting

    @property
    def voting_rule(self) -> str:
        return "Simultaneous majority vote. Player with the most votes is exiled. Ties result in no exile."

    def begin_voting(self, state: GameState, alive_voters: Sequence[Player], potential_targets: Sequence[Player]):
        self._ballots = {}
        # Ensure voters and targets are alive at the start of voting
        self._expected_voters = [p.id for p in alive_voters if p.alive]
        self._potential_targets = [p.id for p in potential_targets if p.alive]
        self._current_game_state = state # Store the game state reference

    def collect_vote(self, vote_action: Action, state: GameState):
        if not isinstance(vote_action, VoteAction):
            return
        
        actor_player = state.get_player_by_id(vote_action.actor_id)
        
        # Voter must be expected and alive at the moment of casting vote
        if actor_player and actor_player.alive and vote_action.actor_id in self._expected_voters:
            if vote_action.target_id == "-1":  # Abstain
                self._ballots[vote_action.actor_id] = vote_action.target_id
            elif vote_action.target_id in self._potential_targets:
                # Target was valid at the start of voting.
                # Optionally, check if target is still alive using current 'state'.
                # If target died mid-round, this vote might become an abstain.
                target_player_current = state.get_player_by_id(vote_action.target_id)
                if target_player_current and target_player_current.alive:
                    self._ballots[vote_action.actor_id] = vote_action.target_id
                else: # Target is no longer alive, treat as abstain
                    self._ballots[vote_action.actor_id] = "-1"
            else: # Invalid target (not abstain, not in original potential_targets)
                self._ballots[vote_action.actor_id] = "-1" # Treat as abstain
        else:
            state.add_history_entry(
                description=f"Invalid vote attempt by P{vote_action.actor_id}.",
                entry_type=HistoryEntryType.ERROR,
                visible_to=[vote_action.actor_id]
            )

    def tally_votes(self, state_at_begin_voting: GameState) -> str | None:
        """
        Tallies votes from self._ballots.
        Uses state_at_begin_voting to confirm validity of voters/targets if necessary,
        though primary validation happens in begin_voting and collect_vote.
        """
        from collections import Counter

        # Consider votes only from expected voters and for initially valid targets (or abstain).
        # Aliveness of voter at casting time was handled by collect_vote.
        final_votes_to_consider = [
            target_id for voter_id, target_id in self._ballots.items()
            if voter_id in self._expected_voters and \
               (target_id == "-1" or target_id in self._potential_targets)
        ]
        
        non_abstain_votes = [tgt for tgt in final_votes_to_consider if tgt != "-1"]

        if not non_abstain_votes:
            return None
        
        counts = Counter(non_abstain_votes).most_common()
        if not counts: return None
        
        top, top_votes = counts[0]
        if len(counts) > 1 and counts[1][1] == top_votes:
            return None
        return top
    def get_voting_prompt(self, state: GameState, player_id: str) -> str:
        target_options = [f"P{p_id}" for p_id in self._potential_targets if state.get_player_by_id(p_id) and state.get_player_by_id(p_id).alive]
        options_str = ", ".join(target_options) if target_options else "No valid targets"
        return f"P{player_id}, please cast your vote. Options: {options_str} or Abstain ('-1')."

    def get_current_tally_info(self, state: GameState) -> Dict[str, int]:
        return Counter(
            tgt for tgt in self._ballots.values() 
            if tgt != "-1" and tgt in self._potential_targets
        )

    def get_next_voters(self) -> List[str]:
        # For simultaneous, all expected voters vote at once.
        return [v_id for v_id in self._expected_voters if v_id not in self._ballots]

    def done(self) -> bool:
        return len(self._ballots) >= len(self._expected_voters)

    def get_valid_targets(self) -> List[str]:
        # Return a copy of targets that were valid (alive) at the start of voting.
        return list(self._potential_targets)

    def get_elected(self) -> str | None: # Return type matches tally_votes
        if self._current_game_state is None:
            # This should not happen if begin_voting was called. Log error or raise.
            return None
        return self.tally_votes(self._current_game_state)


class SequentialFirstToK(VotingProtocol):
    """
    Everyone votes in turn; as soon as a candidate reaches K votes he is exiled.
    """

    def __init__(self, threshold: int):
        self.threshold = threshold
        self._ballots: Dict[str, str] = {}  # actor_id (str) -> target_id (str)
        self._expected_voters: List[str] = []
        self._potential_targets: List[str] = []

    @property
    def voting_rule(self) -> str:
        return f"Sequential voting. First player to reach {self.threshold} votes is exiled."

    def begin_voting(self, state: GameState, alive_voters: Sequence[Player], potential_targets: Sequence[Player]):
        self._ballots = {}
        self._expected_voters = [p.id for p in alive_voters]  # Or manage sequential turns
        self._potential_targets = [p.id for p in potential_targets]

    def collect_vote(self, vote_action: Action, state: GameState):
        if not isinstance(vote_action, VoteAction): return
        # Basic validation, could be enhanced for sequential order
        if any(p.id == vote_action.actor_id and p.alive for p in state.players):
            self._ballots[vote_action.actor_id] = vote_action.target_id  # Overwrites previous vote if any

    def all_votes_collected(self, state: GameState, alive_voters: Sequence[Player]) -> bool:
        # For sequential, this might mean one player reached threshold or all voted.
        # Simplified: if tally_votes finds a winner, it's "collected" for resolution.
        return len(self._ballots) >= len(self._expected_voters)  # Or some other condition

    def tally_votes(self, state: GameState) -> str | None:
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
        for target_id, votes in current_tally.items():
            target_player = next((p for p in state.players if p.id == target_id), None)
            target_name = f"P{target_id}" if not target_player else f"P{target_id} ({target_player.role.name if hasattr(target_player.role, 'name') else 'Unknown'})"
            tally_str_parts.append(f"{target_name}: {votes} votes")
        tally_str = "; ".join(tally_str_parts) if tally_str_parts else "No votes yet."
        target_names = [f"P{p.id}" for p in state.alive_players() if p.id in self._potential_targets]
        return f"P{player_id}, it's your turn to vote. Current tally: {tally_str}. Options: {', '.join(target_names)} or Abstain ('-1'). {self.threshold} votes needed to exile."

    def get_current_tally_info(self, state: GameState) -> Dict[str, int]:
        return Counter(tgt for tgt in self._ballots.values() if tgt != "-1")  # Excludes abstentions

    def get_next_voters(self) -> List[str]:
        # This needs more sophisticated turn management if used by Moderator.
        # For now, assuming Moderator handles turns.
        # This could return the next expected voter if not all have voted and no one reached threshold.
        return []  # Placeholder

    def done(self) -> bool:
        # Done if someone reached threshold or all expected voters have voted.
        return self.tally_votes(state) is not None or len(self._ballots) >= len(self._expected_voters)


# ----------------- decision protocols --------------------------------------- #

class MajorityEliminateResolver(TeamDecisionProtocol):
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

class FirstPriceSealed(BiddingProtocol):
    def begin(self, state):
        self._bids: Dict[str, int] = {}

    def accept(self, bid, state):
        self._bids[bid.actor_id] = bid.amount

    def process_incoming_bids(self, actions: List[Action], state: GameState) -> None:
        for act in actions:
            if isinstance(act, BidAction):
                try:
                    self.accept(act, state)  # self.accept should handle adding to self._bids
                except ValueError as e:  # Or other specific exceptions from accept
                    state.add_history_entry(
                        description=f"Invalid bid by P{act.actor_id}: {e}",
                        entry_type=HistoryEntryType.ERROR  # Or BIDDING_INFO with error status
                    )

    def is_finished(self, state):
        return len(self._bids) == len(state.alive_players())

    def outcome(self, state) -> list[str]:  # Player IDs are strings
        if not self._bids:
            return []
        top = max(self._bids.values())
        winners = [pid for pid, amt in self._bids.items() if amt == top]
        return sorted(winners)  # deterministic tie-break: lowest id


class VickreyAuction(BiddingProtocol):
    def begin(self, state):
        self._bids: Dict[str, int] = {}

    def accept(self, bid, state):
        self._bids[bid.actor_id] = bid.amount

    def process_incoming_bids(self, actions: List[Action], state: GameState) -> None:
        for act in actions:
            if isinstance(act, BidAction):
                try:
                    self.accept(act, state)
                except ValueError as e:
                    state.add_history_entry(
                        description=f"Invalid bid by P{act.actor_id}: {e}",
                        entry_type=HistoryEntryType.ERROR  # Or BIDDING_INFO with error status
                    )

    def is_finished(self, state):
        return len(self._bids) == len(state.alive_players())

    def outcome(self, state) -> list[str]:  # Player IDs are strings
        if not self._bids or len(self._bids) < 1:  # Handle empty or single bid
            return list(self._bids)  # trivial case
        (w_id, w_amt), (_, second_price) = sorted(
            self._bids.items(), key=lambda x: x[1], reverse=True)[:2]
        # Charge second-highest price:
        state.wallet[w_id] -= second_price
        return [w_id]


class SequentialVoting(VotingProtocol):
    """
    Players vote one by one in a sequence. Each player is shown the current
    tally before casting their vote. All players in the initial list of
    voters get a turn.
    """

    def __init__(self):
        self._ballots: Dict[str, str] = {}  # actor_id (str) -> target_id (str)
        self._potential_targets: List[str] = []
        self._voter_queue: List[str] = []  # Order of players to vote
        self._current_voter_index: int = 0  # Index for _voter_queue
        self._current_game_state: Optional[GameState] = None # To store state from begin_voting

    @property
    def voting_rule(self) -> str:
        return "Sequential voting. Players vote one by one. Player with the most votes after all have voted is exiled. Ties are broken randomly."

    def begin_voting(self, state: GameState, alive_voters: Sequence[Player], potential_targets: Sequence[Player]):
        self._ballots = {}
        self._potential_targets = [p.id for p in potential_targets]
        # The order of voting can be based on player ID, a random shuffle, or the order in alive_voters
        # For simplicity, using the order from alive_voters.
        self._voter_queue = [p.id for p in alive_voters if p.alive]
        self._current_voter_index = 0
        self._current_game_state = state # Store the game state reference

    def get_voting_prompt(self, state: GameState, player_id: str) -> str:
        """
        Generates a prompt for the given player_id, assuming it's their turn.
        """
        current_tally = self.get_current_tally_info(state)
        tally_str_parts = []
        for target_id, votes in sorted(current_tally.items()):  # Sort for consistent display
            target_player_obj = next((p for p in state.players if p.id == target_id), None)
            target_name_display = f"P{target_id}"
            if target_player_obj and hasattr(target_player_obj.role, 'name'):
                target_name_display += f" ({target_player_obj.role.name})"
            tally_str_parts.append(f"{target_name_display}: {votes} vote(s)")

        tally_str = "; ".join(tally_str_parts) if tally_str_parts else "No votes cast yet."

        options_str_parts = []
        for p_target in state.alive_players():  # Iterate through all alive players for options
            if p_target.id in self._potential_targets:
                options_str_parts.append(f"P{p_target.id}")
        options_str = ", ".join(options_str_parts)

        return (f"P{player_id}, it is your turn to vote. "
                f"Current tally: {tally_str}. "
                f"Options: {options_str} or Abstain (vote for -1).")

    def collect_vote(self, vote_action: Action, state: GameState):
        if not isinstance(vote_action, (VoteAction, NoOpAction)):
            # Silently ignore if not a VoteAction or NoOpAction.
            # Consider logging an "unexpected action type" error if more verbosity is needed.
            return

        if self.done():
            state.add_history_entry(
                description=f"Action ({vote_action.kind}) received from P{vote_action.actor_id}, but voting is already complete.",
                entry_type=HistoryEntryType.ERROR,
                public=False,
                visible_to={vote_action.actor_id}
            )
            return

        expected_voter_id = self._voter_queue[self._current_voter_index]
        if vote_action.actor_id != expected_voter_id:
            state.add_history_entry(
                description=f"Action ({vote_action.kind}) received from P{vote_action.actor_id}, but it is P{expected_voter_id}'s turn.",
                entry_type=HistoryEntryType.ERROR,
                public=False,  # Or public if strict turn enforcement is announced
                visible_to={vote_action.actor_id, expected_voter_id}
            )
            return

        actor_player = next((p for p in state.players if p.id == vote_action.actor_id), None)
        if actor_player and actor_player.alive:
            description_for_history = ""
            involved_players_list = [vote_action.actor_id]  # Actor is always involved
            if isinstance(vote_action, NoOpAction):
                self._ballots[vote_action.actor_id] = "-1"  # Treat NoOp as abstain
                description_for_history = f"P{vote_action.actor_id} chose to NoOp (treated as Abstain)."

            elif isinstance(vote_action, VoteAction):  # This must be true if not NoOpAction
                target_display: str
                recorded_target_id = vote_action.target_id
                if vote_action.target_id != "-1" and vote_action.target_id not in self._potential_targets:
                    # Invalid target chosen for VoteAction
                    state.add_history_entry(
                        description=f"P{vote_action.actor_id} attempted to vote for P{vote_action.target_id} (invalid target). Vote recorded as Abstain.",
                        entry_type=HistoryEntryType.ERROR,
                        public=True,
                        visible_to={vote_action.actor_id}
                    )
                    recorded_target_id = "-1"  # Treat invalid target as abstain
                    target_display = f"Invalid Target (P{vote_action.target_id}), recorded as Abstain"
                elif vote_action.target_id == "-1":
                    # Explicit Abstain via VoteAction
                    target_display = "Abstain"
                    # recorded_target_id is already "-1"
                else:
                    # Valid target chosen for VoteAction
                    target_display = f"P{vote_action.target_id}"
                    involved_players_list.append(vote_action.target_id)  # Add valid target to involved

                self._ballots[vote_action.actor_id] = recorded_target_id
                description_for_history = f"P{vote_action.actor_id} has voted for {target_display}."

            state.add_history_entry(
                description=description_for_history,
                entry_type=HistoryEntryType.VOTE_RESULT,
                public=True  # Transparent voting
            )
            self._current_voter_index += 1
        else:  # Player not found, not alive, or (redundantly) not their turn
            state.add_history_entry(
                description=f"Invalid action ({vote_action.kind}) attempt by P{vote_action.actor_id} (player not found, not alive, or not their turn). Action not counted.",
                entry_type=HistoryEntryType.ERROR,
                public=True,
                visible_to={vote_action.actor_id}
            )
            # If voter was expected but found to be not alive, advance turn to prevent stall
            if vote_action.actor_id == expected_voter_id:  # Implies actor_player was found but not actor_player.alive
                self._current_voter_index += 1

    def tally_votes(self, state: GameState) -> Optional[str]:
        if not self.done():
            # Voting is not yet complete for this protocol.
            return None

        non_abstain_votes = [
            target_id for voter_id, target_id in self._ballots.items()
            if target_id != "-1" and target_id in self._potential_targets  # Ensure target is valid and not an abstain
        ]

        if not non_abstain_votes:
            return None  # No non-abstain votes cast for valid targets

        counts = Counter(non_abstain_votes).most_common()
        if not counts:  # Should be redundant if non_abstain_votes is not empty
            return None

        top_candidate, top_votes = counts[0]

        # Tie-breaking: if multiple players have top_votes, exile one of them randomly.
        if len(counts) > 1 and counts[1][1] == top_votes:
            tied_candidates = [cand_id for cand_id, num_votes in counts if num_votes == top_votes]
            if tied_candidates:
                return random.choice(tied_candidates)
            return None # Should not happen if tied_candidates is populated
        return top_candidate

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
        if self._current_game_state is None:
            # This implies begin_voting was not called or state was not set.
            return None # Or raise an error
        return self.tally_votes(self._current_game_state)
