"""Structured JSON observations for Negotiation.

Negotiation is a two-player imperfect-information game where agents propose
splits of a shared item pool. Each player has private utilities for each item
type, and they alternate making proposals (and optional utterances) until an
offer is accepted or the maximum number of turns is exhausted.

The default OpenSpiel observation string is a multi-line text block; the proxy
parses it into structured JSON containing the item pool, the player's own
utilities, the full proposal/utterance history, and the current turn type.
Opponent utilities are deliberately hidden (this is imperfect information).
"""

import json
from typing import Any

import pyspiel

from ... import proxy

# These constants mirror the C++ defaults in
# open_spiel/open_spiel/games/negotiation/negotiation.h.
_MAX_QUANTITY = 5


def _decode_base(value: int, dimensions: int, base: int) -> list[int]:
    """Decode an integer as a fixed-width vector of base-N digits (MSB first)."""
    digits = [0] * dimensions
    i = dimensions - 1
    while value > 0 and i >= 0:
        digits[i] = value % base
        value //= base
        i -= 1
    return digits


class NegotiationState(proxy.State):
    """Negotiation state proxy returning structured JSON observations."""

    def _params(self) -> dict[str, Any]:
        return self.get_game().get_parameters()

    def _num_distinct_proposals(self) -> int:
        # Mirrors NegotiationGame::NumDistinctProposals: (max_quantity+1)^num_items + 1.
        num_items = int(self._params().get("num_items", 3))
        return (_MAX_QUANTITY + 1) ** num_items + 1

    def _accept_action(self) -> int:
        return self._num_distinct_proposals() - 1

    def _parse_observation(self, player: int) -> dict[str, Any]:
        """Parse the per-player observation string into structured fields."""
        raw = self.__wrapped__.observation_string(player)
        result: dict[str, Any] = {
            "max_steps": None,
            "item_pool": [],
            "my_utilities": [],
            "current_player": -1,
            "turn_type": None,
            "most_recent_proposal": None,
            "most_recent_utterance": None,
        }
        for line in raw.splitlines():
            if line.startswith("Max steps: "):
                result["max_steps"] = int(line[len("Max steps: ") :])
            elif line.startswith("Item pool: "):
                result["item_pool"] = [int(x) for x in line[len("Item pool: ") :].split()]
            elif line.startswith(f"Agent {player} util vec: "):
                prefix = f"Agent {player} util vec: "
                result["my_utilities"] = [int(x) for x in line[len(prefix) :].split()]
            elif line.startswith("Current player: "):
                result["current_player"] = int(line[len("Current player: ") :])
            elif line.startswith("Turn Type: "):
                result["turn_type"] = line[len("Turn Type: ") :].lower()
            elif line.startswith("Most recent proposal: ["):
                inner = line[len("Most recent proposal: [") : -1]
                result["most_recent_proposal"] = [int(x.strip()) for x in inner.split(",")]
            elif line.startswith("Most recent utterance: ["):
                inner = line[len("Most recent utterance: [") : -1]
                result["most_recent_utterance"] = [int(x.strip()) for x in inner.split(",")]
        return result

    def _decode_history(self) -> dict[str, Any]:
        """Walk the action history to reconstruct proposals and utterances."""
        params = self._params()
        num_items = int(params.get("num_items", 3))
        num_symbols = int(params.get("num_symbols", 5))
        utterance_dim = int(params.get("utterance_dim", 3))
        enable_utterances = bool(params.get("enable_utterances", True))
        accept_action = self._accept_action()
        num_distinct_proposals = self._num_distinct_proposals()

        proposals: list[dict[str, Any]] = []
        utterances: list[dict[str, Any]] = []
        agreement_reached = False
        proposing_player = 0
        # History[0] is the chance node; subsequent actions alternate between
        # proposal and (optional) utterance for the same player, then switch.
        history = list(self.__wrapped__.history())
        expecting_proposal = True
        for action in history[1:]:
            if expecting_proposal:
                if action == accept_action:
                    agreement_reached = True
                    proposals.append({"player": proposing_player, "accept": True})
                else:
                    decoded = _decode_base(int(action), num_items, _MAX_QUANTITY + 1)
                    proposals.append({"player": proposing_player, "items": decoded, "accept": False})
                if enable_utterances:
                    expecting_proposal = False
                else:
                    proposing_player = 1 - proposing_player
            else:
                decoded = _decode_base(int(action) - num_distinct_proposals, utterance_dim, num_symbols)
                utterances.append({"player": proposing_player, "symbols": decoded})
                proposing_player = 1 - proposing_player
                expecting_proposal = True

        return {
            "proposals": proposals,
            "utterances": utterances,
            "agreement_reached": agreement_reached,
        }

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        params = self._params()
        # Observation is per-player (imperfect info). Default to player 0 when
        # the caller does not specify a viewpoint.
        view_player = 0 if player is None else player
        obs = self._parse_observation(view_player)
        decoded = self._decode_history()

        is_terminal = self.is_terminal()
        winner: int | str | None = None
        rewards: list[float] | None = None
        if is_terminal:
            returns = list(self.returns())
            rewards = returns
            if returns[0] > returns[1]:
                winner = 0
            elif returns[1] > returns[0]:
                winner = 1
            else:
                winner = "draw"

        current_player = self.current_player()
        return {
            "current_player": current_player,
            "viewing_player": view_player,
            "turn_type": obs["turn_type"],
            "max_steps": obs["max_steps"],
            "item_pool": obs["item_pool"],
            "my_utilities": obs["my_utilities"],
            "proposals": decoded["proposals"],
            "utterances": decoded["utterances"],
            "most_recent_proposal": obs["most_recent_proposal"],
            "most_recent_utterance": obs["most_recent_utterance"],
            "agreement_reached": decoded["agreement_reached"],
            "is_terminal": is_terminal,
            "winner": winner,
            "rewards": rewards,
            "params": {
                "num_items": int(params.get("num_items", 3)),
                "num_symbols": int(params.get("num_symbols", 5)),
                "utterance_dim": int(params.get("utterance_dim", 3)),
                "enable_proposals": bool(params.get("enable_proposals", True)),
                "enable_utterances": bool(params.get("enable_utterances", True)),
                "max_quantity": _MAX_QUANTITY,
                "num_distinct_proposals": self._num_distinct_proposals(),
                "accept_action": self._accept_action(),
            },
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class NegotiationGame(proxy.Game):
    """Wraps OpenSpiel's negotiation game to use the proxy state."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("negotiation", params)
        super().__init__(
            wrapped,
            short_name="negotiation_proxy",
            long_name="Negotiation (proxy)",
            # Negotiation is kSampledStochastic upstream, which makes pyspiel's
            # C++ Game::Serialize() try to dump the underlying RNG state via
            # GetRNGState(). That virtual is not exposed through pybind11, so
            # any Python proxy subclass would raise "GetRNGState unimplemented"
            # when the kaggle framework serializes the game/state pair. The
            # proxy doesn't support deserialization anyway (see proxy.py), so
            # downgrade to kDeterministic for serialization purposes. The
            # underlying wrapped state still drives chance nodes via
            # is_chance_node()/chance_outcomes() at runtime.
            chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
        )

    def new_initial_state(self, *args) -> NegotiationState:
        return NegotiationState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(NegotiationGame().get_type(), NegotiationGame)
