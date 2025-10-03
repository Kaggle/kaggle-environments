from collections import Counter
from typing import Dict, List

from kaggle_environments.envs.werewolf.game.actions import Action, BidAction
from kaggle_environments.envs.werewolf.game.base import PlayerID
from kaggle_environments.envs.werewolf.game.consts import EventName
from kaggle_environments.envs.werewolf.game.protocols.base import BiddingProtocol
from kaggle_environments.envs.werewolf.game.records import BidDataEntry, ChatDataEntry
from kaggle_environments.envs.werewolf.game.states import GameState

from .factory import register_protocol


@register_protocol()
class SimpleBiddingProtocol(BiddingProtocol):
    """
    A straightforward bidding protocol where speaking priority is determined
    solely by the bid amount.
    - Agents bid with a numerical amount.
    - Higher bids result in earlier speaking turns.
    - Ties are broken deterministically by player ID (ascending).
    """

    def __init__(self):
        self._bids: Dict[PlayerID, int] = {}
        self._max_bid = 4
        self.reset()

    def reset(self) -> None:
        """Resets the bids for a new round."""
        self._bids = {}

    @property
    def display_name(self):
        return "Simple Bidding"

    @property
    def rule(self) -> str:
        """Provides a description of the bidding rules."""
        return "\n".join(
            (
                "Players bid with an urgency level (0-4) to determine speaking order.",
                "0: I would like to observe and listen for now.",
                "1: I have some general thoughts to share with the group.",
                "2: I have something critical and specific to contribute to this discussion.",
                "3: It is absolutely urgent for me to speak next.",
                "4: I must respond.",
                "Higher bids speak earlier. Ties are broken by player name (A-Z).",
            )
        )

    @property
    def bids(self) -> Dict[PlayerID, int]:
        """Returns a copy of the current bids."""
        return dict(**self._bids)

    def begin(self, state: GameState) -> None:
        """Initializes a new bidding round."""
        self.reset()

    def accept(self, bid: BidAction, state: GameState) -> None:
        """Accepts and records a single bid from a player."""
        bid_amount = min(max(0, bid.amount), self._max_bid)
        self._bids[bid.actor_id] = bid_amount

        data = BidDataEntry(
            actor_id=bid.actor_id,
            reasoning=bid.reasoning,
            perceived_threat_level=bid.perceived_threat_level,
            bid_amount=bid_amount,
            action=bid,
        )
        state.push_event(
            description=f"Player {bid.actor_id} submitted a bid of {bid_amount}.",
            event_name=EventName.BID_ACTION,
            public=False,  # Bids are private until the outcome is announced
            visible_to=[bid.actor_id],
            data=data,
            source=bid.actor_id,
        )

    def process_incoming_bids(self, actions: List[Action], state: GameState) -> None:
        """Processes a list of actions, handling any BidActions."""
        for act in actions:
            if isinstance(act, BidAction):
                self.accept(act, state)

    def is_finished(self, state: GameState) -> bool:
        """
        Checks if the bidding phase is complete (i.e., all alive players have bid).
        """
        return len(self._bids) >= len(state.alive_players())

    def outcome(self, state: GameState) -> list[str]:
        """
        Determines the final speaking order based on bids.

        Returns:
            A list of player IDs sorted by bid (descending) and then player ID (ascending).
        """
        if not self._bids:
            # If no bids were made, return alive players in their default order.
            return sorted([p.id for p in state.alive_players()])

        # Sort by bid amount (descending) and then by player ID (ascending) for tie-breaking.
        sorted_bidders = sorted(self._bids.items(), key=lambda item: (-item[1], item[0]))
        return [player_id for player_id, bid_amount in sorted_bidders]


@register_protocol()
class UrgencyBiddingProtocol(BiddingProtocol):
    """
    A bidding protocol based on the Werewolf Arena paper.
    - Agents bid with an urgency level (0-4).
    - Highest bidder wins.
    - Ties are broken by prioritizing players mentioned in the previous turn.
    """

    @property
    def display_name(self) -> str:
        return "Urgency Bidding"

    @property
    def rule(self) -> str:
        return "\n".join(
            [
                "Urgency-based bidding. Players bid with an urgency level (0-4).",
                "0: I would like to observe and listen for now.",
                "1: I have some general thoughts to share with the group.",
                "2: I have something critical and specific to contribute to this discussion.",
                "3: It is absolutely urgent for me to speak next.",
                "4: Someone has addressed me directly and I must respond.",
                "Highest bidder wins."
                "Ties are broken by the following priority: (1) players mentioned in the previous turn's chat, "
                "(2) the least spoken player, (3) round robin order of the player list.",
            ]
        )

    @property
    def bids(self) -> Dict[PlayerID, int]:
        return dict(**self._bids)

    def __init__(self):
        self._bids: Dict[PlayerID, int] = {}
        self._mentioned_last_turn: List[PlayerID] = []

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
                state.push_event(
                    description=f"Players mentioned last turn (priority in ties): {self._mentioned_last_turn}",
                    event_name=EventName.BIDDING_INFO,
                    public=True,  # So everyone knows who has priority
                )

    def accept(self, bid: BidAction, state: GameState) -> None:
        if 0 <= bid.amount <= 4:
            self._bids[bid.actor_id] = bid.amount
            data = BidDataEntry(
                actor_id=bid.actor_id,
                reasoning=bid.reasoning,
                perceived_threat_level=bid.perceived_threat_level,
                bid_amount=bid.amount,
                action=bid,
            )
            state.push_event(
                description=f"Player {bid.actor_id} submitted bid=({bid.amount}).",
                event_name=EventName.BID_ACTION,
                public=False,
                visible_to=[bid.actor_id],
                data=data,
                source=bid.actor_id,
            )
        else:
            # Invalid bid amount is treated as a bid of 0
            self._bids[bid.actor_id] = 0
            state.push_event(
                description=f"Player {bid.actor_id} submitted an invalid bid amount ({bid.amount}). Treated as 0.",
                event_name=EventName.ERROR,
                public=False,
                visible_to=[bid.actor_id],
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
            for day_events in state.history.values()
            for entry in day_events
            if entry.event_name == EventName.DISCUSSION and isinstance(entry.data, ChatDataEntry)
        )

        candidate_speech_counts = {pid: speech_counts.get(pid, 0) for pid in candidates}
        min_spoken = min(candidate_speech_counts.values())
        least_spoken_candidates = sorted([pid for pid, count in candidate_speech_counts.items() if count == min_spoken])

        if len(least_spoken_candidates) == 1:
            return least_spoken_candidates

        candidates = least_spoken_candidates

        # Rule 3: Round robin order of the player list in state
        for pid in state.all_player_ids:
            if pid in candidates:
                return [pid]

        # This part should be unreachable if candidates is a subset of all_player_ids
        return [candidates[0]] if candidates else []
