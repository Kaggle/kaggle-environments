import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

from kaggle_environments.envs.werewolf.game.actions import Action, BidAction, ChatAction
from kaggle_environments.envs.werewolf.game.base import PlayerID
from kaggle_environments.envs.werewolf.game.consts import EventName
from kaggle_environments.envs.werewolf.game.records import ChatDataEntry, RequestVillagerToSpeakDataEntry
from kaggle_environments.envs.werewolf.game.roles import Player
from kaggle_environments.envs.werewolf.game.states import GameState


def _extract_player_ids_from_string(text: str, all_player_ids: List[PlayerID]) -> List[PlayerID]:
    """Extracts player IDs mentioned in a string."""
    if not all_player_ids:
        return []
    # Create a regex pattern to find any of the player IDs as whole words
    # Using a set for faster lookups and to handle duplicates from the regex
    pattern = r"\b(" + "|".join(re.escape(pid) for pid in all_player_ids) + r")\b"
    # Use a set to automatically handle duplicates found by the regex
    found_ids = set(re.findall(pattern, text))
    return sorted(list(found_ids))  # sorted for deterministic order


def _find_mentioned_players(text: str, all_player_ids: List[PlayerID]) -> List[PlayerID]:
    """
    Finds player IDs mentioned in a string of text, ordered by their first appearance.
    Player IDs are treated as whole words.
    Example: "I think gpt-4 is suspicious, what do you think John?" -> ["gpt-4", "John"]
    """
    if not text or not all_player_ids:
        return []

    # Sort by length descending to handle substrings correctly.
    sorted_player_ids = sorted(all_player_ids, key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(pid) for pid in sorted_player_ids) + r")\b"

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


class GameProtocol(ABC):
    @property
    def display_name(self) -> str:
        return self.__class__.__name__

    @property
    @abstractmethod
    def rule(self) -> str:
        """Human-readable format of rule."""


class VotingProtocol(GameProtocol):
    """Collects, validates, and tallies votes."""

    @abstractmethod
    def begin_voting(self, state: GameState, alive_voters: Sequence[Player], potential_targets: Sequence[Player]):
        """Initialize for a new voting round."""

    @abstractmethod
    def get_voting_prompt(self, state: GameState, player_id: PlayerID) -> str:
        """
        Returns a string prompt for the specified player, potentially including current tally.
        """

    @abstractmethod
    def collect_vote(self, vote_action: Action, state: GameState):  # Changed to Action, will check type
        """Collect an individual vote."""

    @abstractmethod
    def collect_votes(self, player_actions: Dict[str, Action], state: GameState, expected_voters: List[PlayerID]):
        """Collect a batch of votes."""

    @abstractmethod
    def get_current_tally_info(self, state: GameState) -> Dict[PlayerID, PlayerID]:
        """
        Return the current tally by a map, where key is player, value is target.
        """

    @abstractmethod
    def get_next_voters(self) -> List[PlayerID]:
        """get the next batch of voters"""

    @abstractmethod
    def done(self):
        """Check if voting is done."""

    @abstractmethod
    def get_valid_targets(self) -> List[PlayerID]:
        """get a list of targets"""

    @abstractmethod
    def get_elected(self) -> Optional[PlayerID]:
        """get the final elected individual, or None if no one was elected."""

    @abstractmethod
    def reset(self) -> None:
        """Resets the protocol to its initial state."""
        pass


class BiddingProtocol(GameProtocol):
    """Drives one auction round and returns the winner(s)."""

    @property
    @abstractmethod
    def bids(self) -> Dict[PlayerID, int]:
        """return a snapshot of the current bids"""

    @staticmethod
    def get_last_mentioned(state: GameState) -> Tuple[List[PlayerID], str]:
        """get the players that were mentioned in last player message."""
        last_chat_message = ""
        sorted_days = sorted(state.history.keys(), reverse=True)
        for day in sorted_days:
            for entry in reversed(state.history[day]):
                if entry.event_name == EventName.DISCUSSION and isinstance(entry.data, ChatDataEntry):
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
    def outcome(self, state: GameState) -> list[PlayerID]:
        """
        Return list of player-ids, ordered by bid strength.
        Could be 1 winner (sealed-bid) or a full ranking (Dutch auction).
        """

    @abstractmethod
    def reset(self) -> None:
        """Resets the protocol to its initial state."""


class DiscussionProtocol(GameProtocol):
    """Drives the order/shape of daytime conversation."""

    @abstractmethod
    def begin(self, state: GameState) -> None:
        """Optional hook – initialise timers, round counters…"""

    @abstractmethod
    def speakers_for_tick(self, state: GameState) -> Sequence[PlayerID]:
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

    def process_actions(self, actions: List[Action], expected_speakers: Sequence[PlayerID], state: GameState) -> None:
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
                        perceived_threat_level=act.perceived_threat_level,
                        action=act,
                    )
                    state.push_event(
                        description=f'Player "{act.actor_id}" (chat): {act.message}',
                        # Make public for general discussion
                        event_name=EventName.DISCUSSION,
                        public=True,
                        source=act.actor_id,
                        data=data,
                    )
                else:
                    state.push_event(
                        description=f'Player "{act.actor_id}" (chat, out of turn): {act.message}',
                        event_name=EventName.DISCUSSION,  # Or a specific "INVALID_CHAT" type
                        visible_to=[act.actor_id],
                        public=False,
                        source=act.actor_id,
                    )

    def call_for_actions(self, speakers: Sequence[PlayerID]) -> List[str]:
        """prepare moderator call for action for each player."""
        return [f'Player "{speaker_id}", it is your turn to speak.' for speaker_id in speakers]

    def prompt_speakers_for_tick(self, state: GameState, speakers: Sequence[PlayerID]) -> None:
        """
        Allows the protocol to make specific announcements or prompts to the current speakers for this tick.
        This method is called by the Moderator after speakers_for_tick() returns a non-empty list of speakers,
        and before process_actions().
        Implementations should use state.push_event() to make announcements.
        These announcements are typically visible only to the speakers, unless they are general status updates.
        """
        call_for_actions = self.call_for_actions(speakers)
        for speaker_id, call_for_action in zip(speakers, call_for_actions):
            data = RequestVillagerToSpeakDataEntry(action_json_schema=json.dumps(ChatAction.schema_for_player()))
            state.push_event(
                description=call_for_action,
                event_name=EventName.CHAT_REQUEST,
                public=False,
                visible_to=[speaker_id],
                data=data,
                visible_in_ui=False,
            )
