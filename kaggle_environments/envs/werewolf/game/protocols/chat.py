import itertools
import json
import random
from abc import ABC
from collections import deque
from typing import List, Optional, Sequence

from kaggle_environments.envs.werewolf.game.actions import Action, BidAction
from kaggle_environments.envs.werewolf.game.base import PlayerID
from kaggle_environments.envs.werewolf.game.consts import EventName, StrEnum
from kaggle_environments.envs.werewolf.game.protocols.base import BiddingProtocol, DiscussionProtocol
from kaggle_environments.envs.werewolf.game.records import BidResultDataEntry, DiscussionOrderDataEntry
from kaggle_environments.envs.werewolf.game.states import GameState

from .bid import SimpleBiddingProtocol
from .factory import register_protocol
from .ordering import FirstPlayerStrategy, PivotSelector


@register_protocol(default_params={"max_rounds": 2, "first_to_speak": "rotate"})
class RoundRobinDiscussion(DiscussionProtocol):
    def __init__(self, max_rounds: int = 1, first_to_speak: str = "rotate"):
        """
        Args:
            max_rounds: rounds of discussion
            first_to_speak: Strategy to determine the first speaker for each round.
                - fixed: Always starts from the first player in the player list (if alive).
                - rotate: Rotates the starting position in the player list for each round.
                - random: Randomly selects a starting position in the player list for each round.
        """
        self.max_rounds = max_rounds
        self._queue: deque[str] = deque()
        self.pivot_selector = PivotSelector(first_to_speak)

    def reset(self) -> None:
        self._queue = deque()

    @property
    def display_name(self) -> str:
        return "Roundrobin"

    @property
    def rule(self) -> str:
        base_rule = f"Players speak in round-robin order for {self.max_rounds} round(s)."
        strategy = self.pivot_selector.strategy
        if strategy == FirstPlayerStrategy.FIXED:
            return f"{base_rule} The speaking order always starts from the beginning of the player list."
        elif strategy == FirstPlayerStrategy.ROTATE:
            return (
                f"{base_rule} The starting speaker rotates to the next player in the list for each subsequent day, "
                f"but remains the same for all rounds within that day."
            )
        elif strategy == FirstPlayerStrategy.RANDOM:
            return f"{base_rule} The starting speaker is chosen randomly for each round."
        return base_rule

    def begin(self, state: GameState):
        all_ids = state.all_player_ids
        alive_set = set(p.id for p in state.alive_players())

        rounds_queue = []

        # Calculate pivot once per day for all strategies
        pivot = self.pivot_selector.get_pivot(all_ids, alive_set)
        ordered_ids = PivotSelector.get_ordered_ids(all_ids, pivot)

        for _ in range(self.max_rounds):
            # Construct order starting from the calculated pivot
            for pid in ordered_ids:
                if pid in alive_set:
                    rounds_queue.append(pid)

        self._queue = deque(rounds_queue)

        if self.max_rounds > 0 and self._queue:
            data = DiscussionOrderDataEntry(chat_order_of_player_ids=list(self._queue))
            state.push_event(
                description="Discussion phase begins. Players will speak in round-robin order. "
                f"Full speaking order for {self.max_rounds} round(s): {list(self._queue)}",
                event_name=EventName.DISCUSSION_ORDER,
                public=True,
                data=data,
            )

    def speakers_for_tick(self, state):
        return [self._queue.popleft()] if self._queue else []

    def is_discussion_over(self, state: GameState) -> bool:
        return not self._queue  # Over if queue is empty


@register_protocol()
class RandomOrderDiscussion(DiscussionProtocol):
    def __init__(self):
        self._iters = None
        self._steps = 0

    def reset(self) -> None:
        self._iters = None
        self._steps = 0

    @property
    def display_name(self) -> str:
        return "Random Order Discussion"

    @property
    def rule(self) -> str:
        return "Players speak in a random order for one full round."

    def begin(self, state):
        self._iters = itertools.cycle(
            random.sample([p.id for p in state.alive_players()], k=len(state.alive_players()))
        )
        self._steps = len(state.alive_players())  # one full round
        if self._steps > 0:
            state.push_event(
                description="Discussion phase begins. Players will speak in random order.",
                event_name=EventName.PHASE_CHANGE,
                public=True,
            )

    def speakers_for_tick(self, state):
        if self._steps == 0:
            return []
        self._steps -= 1
        return [next(self._iters)]

    def is_discussion_over(self, state: GameState) -> bool:
        return self._steps == 0


@register_protocol()
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
    def display_name(self) -> str:
        return "Parallel Discussion"

    @property
    def rule(self) -> str:
        return f"All players may speak simultaneously for {self.ticks} tick(s)."

    def begin(self, state):
        self._remaining = self.ticks
        if self.ticks > 0:
            state.push_event(
                description="Parallel discussion phase begins. All players may speak.",
                event_name=EventName.PHASE_CHANGE,
                public=True,
            )

    def speakers_for_tick(self, state):
        if self._remaining == 0:
            return []
        self._remaining -= 1
        return [p.id for p in state.alive_players()]

    def call_for_actions(self, speakers: Sequence[str]) -> List[str]:
        return [
            f"Parallel discussion: All designated players may speak now or remain silent. "
            f"({self._remaining + 1} speaking opportunities remaining, including this one)."
        ] * len(speakers)

    def is_discussion_over(self, state: GameState) -> bool:
        return self._remaining == 0


class BiddingDiscussionPhase(StrEnum):
    BIDDING_PHASE = "bidding_phase"
    SPEAKING_PHASE = "speaking_phase"


class BiddingDiscussion(DiscussionProtocol, ABC):
    def __init__(self, bidding: Optional[BiddingProtocol] = None):
        bidding = bidding or SimpleBiddingProtocol()
        self._bidding = bidding
        self._phase = BiddingDiscussionPhase.BIDDING_PHASE

    @property
    def bidding(self):
        return self._bidding

    @property
    def phase(self):
        return self._phase

    def is_bidding_phase(self):
        return self._phase == BiddingDiscussionPhase.BIDDING_PHASE

    def is_speaking_phase(self):
        return self._phase == BiddingDiscussionPhase.SPEAKING_PHASE

    def set_phase(self, phase: BiddingDiscussionPhase):
        self._phase = phase


@register_protocol(default_params={"max_turns": 8, "bid_result_public": True})
class TurnByTurnBiddingDiscussion(BiddingDiscussion):
    """
    A discussion protocol where players bid for the right to speak each turn.
    This protocol manages the entire bid-speak-bid-speak loop.
    """

    def __init__(self, bidding: Optional[BiddingProtocol] = None, max_turns: int = 8, bid_result_public: bool = True):
        super().__init__(bidding=bidding)
        self.max_turns = max_turns
        self._turns_taken = 0
        self._speaker: Optional[str] = None
        self._all_passed = False
        self._bid_result_public = bid_result_public

    def reset(self) -> None:
        self.bidding.reset()
        self.set_phase(BiddingDiscussionPhase.BIDDING_PHASE)
        self._turns_taken = 0
        self._speaker = None
        self._all_passed = False

    @property
    def display_name(self) -> str:
        return "Turn-by-turn Bidding Driven Discussion"

    @property
    def rule(self) -> str:
        return "\n".join(
            [
                f"Players bid for the right to speak each turn for up to {self.max_turns} turns.",
                f"**Bidding Rule:** {self.bidding.display_name}. {self.bidding.rule}",
                "If everyone bids 0, moderator will directly move on to day voting and no one speaks.",
            ]
        )

    def begin(self, state: GameState) -> None:
        self.reset()
        self.bidding.begin(state)  # Initial setup for the first bidding round

    def is_discussion_over(self, state: GameState) -> bool:
        return self._turns_taken >= self.max_turns or self._all_passed

    def speakers_for_tick(self, state: GameState) -> Sequence[PlayerID]:
        if self.is_discussion_over(state):
            return []

        if self.is_bidding_phase():
            return [p.id for p in state.alive_players()]
        elif self.is_speaking_phase():
            return [self._speaker] if self._speaker else []
        return []

    def process_actions(self, actions: List[Action], expected_speakers: Sequence[PlayerID], state: GameState) -> None:
        if self.is_bidding_phase():
            self.bidding.process_incoming_bids(actions, state)

            # Handle players who didn't bid (timed out) by assuming a bid of 0
            all_alive_player_ids = [p.id for p in state.alive_players()]
            if hasattr(self.bidding, "_bids"):
                for action, player_id in zip(actions, expected_speakers):
                    if not isinstance(action, BidAction):
                        default_bid = BidAction(
                            actor_id=player_id, amount=0, day=state.day_count, phase=state.phase.value
                        )
                        self.bidding.accept(default_bid, state)

            bids = getattr(self.bidding, "_bids", {})

            if len(bids) >= len(all_alive_player_ids):
                # If all bids are in
                if all(amount == 0 for amount in bids.values()):
                    # If everyone decided to pass
                    self._all_passed = True
                    state.push_event(
                        description="All players passed on speaking. Discussion ends.",
                        event_name=EventName.MODERATOR_ANNOUNCEMENT,
                        public=True,
                    )
                    return
                else:
                    winner_list = self.bidding.outcome(state)
                    self._speaker = winner_list[0] if winner_list else None
                    if self._speaker:
                        data = BidResultDataEntry(
                            winner_player_ids=[self._speaker],
                            bid_overview=self.bidding.bids,
                            mentioned_players_in_previous_turn=self.bidding.get_last_mentioned(state)[0],
                        )
                        overview_text = ", ".join([f"{k}: {v}" for k, v in self.bidding.bids.items()])
                        state.push_event(
                            description=f"Player {self._speaker} won the bid and will speak next.\n"
                            f"Bid overview - {overview_text}.",
                            event_name=EventName.BID_RESULT,
                            public=self._bid_result_public,
                            data=data,
                        )
                        self.set_phase(BiddingDiscussionPhase.SPEAKING_PHASE)
                        return
                    else:
                        self._turns_taken += 1
                        if not self.is_discussion_over(state):
                            self.bidding.begin(state)
            # continue bidding
        elif self.is_speaking_phase():
            # Process the chat action from the designated speaker
            super().process_actions(actions, expected_speakers, state)
            self._turns_taken += 1

            # After speaking, transition back to bidding for the next turn
            if not self.is_discussion_over(state):
                self.set_phase(BiddingDiscussionPhase.BIDDING_PHASE)
                self._speaker = None
                self.bidding.begin(state)

    def prompt_speakers_for_tick(self, state: GameState, speakers: Sequence[PlayerID]) -> None:
        if self.is_bidding_phase():
            data = {"action_json_schema": json.dumps(BidAction.schema_for_player())}
            state.push_event(
                description=(
                    f"A new round of discussion begins. Place bid for a chance to speak. "
                    f"{self.max_turns - self._turns_taken} turns left to speak."
                ),
                event_name=EventName.BID_REQEUST,
                public=True,
                data=data,
                visible_in_ui=False,
            )
        elif self.is_speaking_phase() and self._speaker:
            super().prompt_speakers_for_tick(state, speakers)


@register_protocol(default_params={"max_rounds": 2, "bid_result_public": True})
class RoundByRoundBiddingDiscussion(BiddingDiscussion):
    """
    A discussion protocol where players bid at the start of each round to
    determine the speaking order for that round.

    In each of the N rounds:
    1. A bidding phase occurs where all alive players submit a bid (0-4).
    2. The speaking order is determined by sorting players by their bid amount
       (descending) and then by player ID (ascending) as a tie-breaker.
    3. A speaking phase occurs where each player speaks once according to the
       determined order.
    """

    def __init__(self, bidding: Optional[BiddingProtocol] = None, max_rounds: int = 2, bid_result_public: bool = True):
        """
        Args:
            bidding: The bidding protocol to use for determining speaking order.
            max_rounds: The total number of discussion rounds.
            bid_result_public: Whether to make the bidding results public.
        """
        super().__init__(bidding=bidding)
        self.max_rounds = max_rounds
        self._bid_result_public = bid_result_public
        self._current_round = 0
        self._speaking_queue: deque[str] = deque()
        self.reset()

    def reset(self) -> None:
        """Resets the protocol to its initial state."""
        self.bidding.reset()
        self.set_phase(BiddingDiscussionPhase.BIDDING_PHASE)
        self._current_round = 0
        self._speaking_queue = deque()

    @property
    def display_name(self) -> str:
        return "Round-by-round Bidding Driven Discussion"

    @property
    def rule(self) -> str:
        """A string describing the discussion rule in effect."""
        return "\n".join(
            [
                "Players speak in an order determined by bidding at the beginning of each round. "
                f"There will be {self.max_rounds} round(s) per day.",
                "In each round, all players may speak once.",
                f"**Bidding Rule:** {self.bidding.display_name}. {self.bidding.rule}",
            ]
        )

    def begin(self, state: GameState) -> None:
        """Initializes the protocol for the first round."""
        self.reset()
        self.bidding.begin(state)

    def is_discussion_over(self, state: GameState) -> bool:
        """Checks if all rounds have been completed."""
        return self._current_round >= self.max_rounds

    def speakers_for_tick(self, state: GameState) -> Sequence[PlayerID]:
        """Returns the players who are allowed to act in the current tick."""
        if self.is_discussion_over(state):
            return []

        if self.is_bidding_phase():
            # In the bidding phase, all alive players can bid.
            return [p.id for p in state.alive_players()]
        elif self.is_speaking_phase():
            # In the speaking phase, the next player in the queue speaks.
            return [self._speaking_queue.popleft()] if self._speaking_queue else []
        return []

    def process_actions(self, actions: List[Action], expected_speakers: Sequence[PlayerID], state: GameState) -> None:
        """Processes incoming actions from players."""
        if self.is_bidding_phase():
            self.bidding.process_incoming_bids(actions, state)

            # Assume a bid of 0 for any players who timed out.
            all_alive_player_ids = [p.id for p in state.alive_players()]
            if hasattr(self.bidding, "_bids"):
                for player_id in all_alive_player_ids:
                    if player_id not in self.bidding.bids:
                        default_bid = BidAction(
                            actor_id=player_id, amount=0, day=state.day_count, phase=state.phase.value
                        )
                        self.bidding.accept(default_bid, state)

            # Determine speaking order based on bids.
            # Sort by bid amount (desc) and then player ID (asc).
            bids = self.bidding.bids
            sorted_bidders = sorted(bids.items(), key=lambda item: (-item[1], item[0]))

            self._speaking_queue = deque([player_id for player_id, bid_amount in sorted_bidders])

            # Announce the speaking order for the round.
            data = DiscussionOrderDataEntry(chat_order_of_player_ids=list(self._speaking_queue))
            speaking_order_text = ", ".join([f"{pid} ({amount})" for pid, amount in sorted_bidders])

            state.push_event(
                description=f"Bidding for round {self._current_round + 1} has concluded. The speaking order, "
                f"with bid amounts in parentheses, is: {speaking_order_text}.",
                event_name=EventName.BID_RESULT,
                public=self._bid_result_public,
                data=data,
            )

            # Transition to the speaking phase.
            self.set_phase(BiddingDiscussionPhase.SPEAKING_PHASE)

        elif self.is_speaking_phase():
            # Process the chat action from the current speaker.
            super().process_actions(actions, expected_speakers, state)

            # Check if the round is over (i.e., the speaking queue is empty).
            if not self._speaking_queue:
                self._current_round += 1
                state.push_event(
                    description=f"End of discussion round {self._current_round}.",
                    event_name=EventName.PHASE_CHANGE,
                    public=True,
                )

                # If the game isn't over, prepare for the next round's bidding.
                if not self.is_discussion_over(state):
                    self.set_phase(BiddingDiscussionPhase.BIDDING_PHASE)
                    self.bidding.begin(state)

    def prompt_speakers_for_tick(self, state: GameState, speakers: Sequence[PlayerID]) -> None:
        """Prompts the active players for their next action."""
        if self.is_bidding_phase():
            data = {"action_json_schema": json.dumps(BidAction.schema_for_player())}
            state.push_event(
                description=(
                    f"Round {self._current_round + 1} of {self.max_rounds} begins. "
                    "Place your bid to determine speaking order."
                ),
                event_name=EventName.BID_REQEUST,
                public=True,
                data=data,
                visible_in_ui=False,
            )
        elif self.is_speaking_phase():
            # The default prompt from the base class is sufficient for speaking.
            super().prompt_speakers_for_tick(state, speakers)
