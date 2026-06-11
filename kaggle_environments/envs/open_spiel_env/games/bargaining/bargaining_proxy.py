"""Structured JSON observations for Bargaining.

Bargaining is a two-player imperfect-information negotiation game (Lewis et
al. 2017, "Deal or No Deal"). A chance node samples an instance: a small pool
of three item types (Book, Hat, Basketball) plus private per-player integer
valuations. Players then alternate offers -- each offer is a proposed
allocation the proposing player keeps for themselves (opponent gets the
complement) -- until one player accepts (action id 120) or `max_turns` is
reached without agreement (both get 0).

OpenSpiel's default observation_string is a multi-line text block. The proxy
parses it into structured JSON exposing the pool, the viewing player's own
valuations, the full offer history, and the agreement flag. Opponent
valuations remain hidden (imperfect information).
"""

import json
import re
from typing import Any

import pyspiel

from ... import proxy

_ITEM_KEYS = ("book", "hat", "basketball")
_AGREE_ACTION = 120
_OFFER_RE = re.compile(r"Offer:\s*Book:\s*(\d+),\s*Hat:\s*(\d+),\s*Basketball:\s*(\d+)")
_TRIPLE_RE = re.compile(r"Book:\s*(\d+),\s*Hat:\s*(\d+),\s*Basketball:\s*(\d+)")


def _parse_triple(text: str) -> dict[str, int]:
    """Parse a 'Book: x, Hat: y, Basketball: z' fragment into a dict."""
    match = _TRIPLE_RE.search(text)
    if not match:
        return {k: 0 for k in _ITEM_KEYS}
    return {
        "book": int(match.group(1)),
        "hat": int(match.group(2)),
        "basketball": int(match.group(3)),
    }


def _parse_offer(text: str) -> dict[str, int] | None:
    """Parse an 'Offer: Book: x, Hat: y, Basketball: z' line into items."""
    match = _OFFER_RE.search(text)
    if not match:
        return None
    return {
        "book": int(match.group(1)),
        "hat": int(match.group(2)),
        "basketball": int(match.group(3)),
    }


class BargainingState(proxy.State):
    """Bargaining state proxy returning structured JSON observations."""

    def _params(self) -> dict[str, Any]:
        return self.get_game().get_parameters()

    def _parse_observation(self, player: int) -> dict[str, Any]:
        """Parse the per-player observation string into the public/private fields.

        OpenSpiel's observation string only includes the most recent offer (not
        the full history), so the offer timeline is reconstructed separately
        from `state.history()` in `_decode_history`.
        """
        raw = self.__wrapped__.observation_string(player)
        result: dict[str, Any] = {
            "pool": {k: 0 for k in _ITEM_KEYS},
            "my_values": {k: 0 for k in _ITEM_KEYS},
            "agreement_reached": False,
            "num_offers": 0,
        }
        for line in raw.splitlines():
            if line.startswith("Pool:"):
                result["pool"] = _parse_triple(line)
            elif line.startswith("My values:"):
                result["my_values"] = _parse_triple(line)
            elif line.startswith("Agreement reached?"):
                result["agreement_reached"] = line.strip().endswith("1")
            elif line.startswith("Number of offers:"):
                result["num_offers"] = int(line.split(":", 1)[1].strip())
        return result

    def _decode_history(self) -> list[dict[str, Any]]:
        """Reconstruct the full offer/agree timeline from action history.

        History[0] is the chance action that picked the instance and reveals
        nothing about either player's private valuations (the mapping from
        chance id to valuations lives in OpenSpiel's instances file).
        Subsequent actions alternate P0, P1, P0, ... -- each is either an
        offer or the Agree terminator.
        """
        actions = list(self.__wrapped__.history())[1:]
        events: list[dict[str, Any]] = []
        for i, action in enumerate(actions):
            action = int(action)
            event: dict[str, Any] = {"player": i % 2}
            if action == _AGREE_ACTION:
                event["type"] = "agree"
            else:
                items = _parse_offer(self.__wrapped__.action_to_string(action))
                event["type"] = "offer"
                event["items"] = items if items is not None else {k: 0 for k in _ITEM_KEYS}
            events.append(event)
        return events

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        # Observation is per-player (imperfect info). Default to player 0 when
        # the caller does not specify a viewpoint.
        view_player = 0 if player is None else player
        obs = self._parse_observation(view_player)
        history = self._decode_history()
        params = self._params()

        is_terminal = self.is_terminal()
        returns: list[float] | None = None
        if is_terminal:
            returns = list(self.returns())

        last_offer = None
        for event in reversed(history):
            if event["type"] == "offer":
                last_offer = event
                break

        return {
            "current_player": self.current_player(),
            "viewing_player": view_player,
            "is_terminal": is_terminal,
            "agreement_reached": obs["agreement_reached"],
            "max_turns": int(params.get("max_turns", 10)),
            "num_offers": obs["num_offers"],
            "pool": obs["pool"],
            "my_values": obs["my_values"],
            "offer_history": history,
            "last_offer": last_offer,
            "returns": returns,
            "params": {
                "max_turns": int(params.get("max_turns", 10)),
                "discount": float(params.get("discount", 1.0)),
                "prob_end": float(params.get("prob_end", 0.0)),
                "agree_action": _AGREE_ACTION,
            },
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class BargainingGame(proxy.Game):
    """Wraps OpenSpiel's bargaining game to use the proxy state."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("bargaining", params)
        super().__init__(
            wrapped,
            short_name="bargaining_proxy",
            long_name="Bargaining (proxy)",
        )

    def new_initial_state(self, *args) -> BargainingState:
        return BargainingState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(BargainingGame().get_type(), BargainingGame)
