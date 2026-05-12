"""Structured JSON observations for Gin Rummy.

OpenSpiel's default observation_string for gin_rummy is multi-line ASCII art
with header fields, two boxed hand displays, the upcard, and the discard pile.
This proxy parses that text into a structured JSON object that agents can
consume directly.

Cards use OpenSpiel's two-character notation: rank in {A,2-9,T,J,Q,K} followed
by suit in {s,c,d,h}. Each player only sees their own hand; the opponent's
hand appears as an empty list.
"""

import json
import re
from typing import Any

import pyspiel

from ... import proxy

_CARD_RE = re.compile(r"[A23456789TJQK][scdh]")
_HEADER_RE = re.compile(r"^(Knock card|Prev upcard|Repeated move|Current player|Phase): ?(.*)$")
_STOCK_RE = re.compile(r"^Stock size: (\d+)\s+Upcard: (\S+)\s*$")
_DISCARD_RE = re.compile(r"^Discard pile: ?(.*)$")
_PLAYER_DEADWOOD_RE = re.compile(r"^Player(\d): Deadwood=(\d+)\s*$")
_PLAYER_RE = re.compile(r"^Player(\d):\s*$")


def _parse_card(token: str) -> str | None:
    if token == "XX" or not token:
        return None
    return token


def _parse_observation(text: str) -> dict[str, Any]:
    """Parse the OpenSpiel gin_rummy observation_string into a dict."""
    lines = text.split("\n")
    result: dict[str, Any] = {
        "knock_card": None,
        "prev_upcard": None,
        "repeated_move": 0,
        "phase": None,
        "stock_size": 0,
        "upcard": None,
        "discard_pile": [],
        "hands": {"0": [], "1": []},
        "deadwood": {"0": None, "1": None},
    }
    current_hand: int | None = None
    for line in lines:
        m = _HEADER_RE.match(line)
        if m:
            key = m.group(1).lower().replace(" ", "_")
            value = m.group(2).strip()
            if key == "knock_card":
                result["knock_card"] = int(value) if value.isdigit() else (value or None)
            elif key == "prev_upcard":
                result["prev_upcard"] = _parse_card(value)
            elif key == "repeated_move":
                result["repeated_move"] = int(value)
            elif key == "phase":
                result["phase"] = value
            # current_player from header is redundant with state.current_player()
            continue
        m = _STOCK_RE.match(line)
        if m:
            result["stock_size"] = int(m.group(1))
            result["upcard"] = _parse_card(m.group(2))
            continue
        m = _DISCARD_RE.match(line)
        if m:
            result["discard_pile"] = _CARD_RE.findall(m.group(1))
            continue
        m = _PLAYER_DEADWOOD_RE.match(line)
        if m:
            current_hand = int(m.group(1))
            result["deadwood"][str(current_hand)] = int(m.group(2))
            continue
        m = _PLAYER_RE.match(line)
        if m:
            current_hand = int(m.group(1))
            continue
        if current_hand is not None and line.startswith("|") and line.endswith("|"):
            result["hands"][str(current_hand)].extend(_CARD_RE.findall(line))
    return result


class GinRummyState(proxy.State):
    """Gin Rummy state proxy with structured JSON observations."""

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        # Use player 0's view as the source for shared fields; the opponent's
        # hand is hidden in player 0's observation, and we mask the requesting
        # player's opponent below.
        observer = player if player is not None else 0
        parsed = _parse_observation(self.__wrapped__.observation_string(observer))

        winner: int | str | None = None
        returns_list: list[float] = []
        if self.is_terminal():
            returns_list = list(self.returns())
            if returns_list[0] > returns_list[1]:
                winner = 0
            elif returns_list[1] > returns_list[0]:
                winner = 1
            else:
                winner = "draw"

        result: dict[str, Any] = {
            "phase": parsed["phase"],
            "current_player": self.current_player(),
            "is_terminal": self.is_terminal(),
            "winner": winner,
            "returns": returns_list,
            "knock_card": parsed["knock_card"],
            "prev_upcard": parsed["prev_upcard"],
            "upcard": parsed["upcard"],
            "stock_size": parsed["stock_size"],
            "discard_pile": parsed["discard_pile"],
            "hands": parsed["hands"],
            "deadwood": parsed["deadwood"],
            "repeated_move": parsed["repeated_move"],
        }
        return result

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class GinRummyGame(proxy.Game):
    """Gin Rummy game proxy."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("gin_rummy", params)
        super().__init__(
            wrapped,
            short_name="gin_rummy_proxy",
            long_name="Gin Rummy (proxy)",
        )

    def new_initial_state(self, *args) -> GinRummyState:
        return GinRummyState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(GinRummyGame().get_type(), GinRummyGame)
