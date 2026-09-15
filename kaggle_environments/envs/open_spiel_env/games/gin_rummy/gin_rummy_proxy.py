"""Structured JSON observations for Gin Rummy.

OpenSpiel's default observation_string for gin_rummy is multi-line ASCII art.
Rather than parse that text, we delegate to the engine's native
``to_observation_struct(player)``, which returns a per-player JSON view that
already implements gin rummy's information-disclosure rules (opponent's hand
hidden until knock/layoff/game-over, opponent's deadwood revealed once they
knock, laid melds public, etc.).

Cards use OpenSpiel's two-character notation: rank in ``{A,2-9,T,J,Q,K}``
followed by suit in ``{s,c,d,h}``. The opponent's hand appears as the literal
string ``"XX"`` per card position when it is still hidden.
"""

import json
from typing import Any

import pyspiel

from ... import proxy


class GinRummyState(proxy.State):
    """Gin Rummy state proxy with structured JSON observations."""

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        observer = player if player is not None else 0
        struct_json = json.loads(self.__wrapped__.to_observation_struct(observer).to_json())

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

        # Hand lists from the struct are length-`hand_size` arrays of "XX" when
        # hidden. Normalize to {"0": [...], "1": [...]} with hidden hands as []
        # for compatibility with the harness's existing rendering.
        hands_raw = struct_json.get("hands") or [[], []]
        hands = {
            "0": [c for c in hands_raw[0] if c != "XX"] if hands_raw and hands_raw[0] else [],
            "1": [c for c in hands_raw[1] if c != "XX"] if len(hands_raw) > 1 and hands_raw[1] else [],
        }
        # Track whether each hand is fully hidden so the harness can distinguish
        # "empty after laying all melds" from "still hidden".
        hand_hidden = {
            "0": bool(hands_raw[0]) and all(c == "XX" for c in hands_raw[0]),
            "1": len(hands_raw) > 1 and bool(hands_raw[1]) and all(c == "XX" for c in hands_raw[1]),
        }

        deadwood_raw = struct_json.get("deadwood") or [None, None]
        # Engine encodes hidden deadwood as -1.
        deadwood = {
            "0": deadwood_raw[0] if deadwood_raw[0] is not None and deadwood_raw[0] >= 0 else None,
            "1": deadwood_raw[1] if len(deadwood_raw) > 1 and deadwood_raw[1] is not None and deadwood_raw[1] >= 0 else None,
        }

        layed_melds_raw = struct_json.get("layed_melds") or [[], []]
        layed_melds = {
            "0": layed_melds_raw[0] if layed_melds_raw else [],
            "1": layed_melds_raw[1] if len(layed_melds_raw) > 1 else [],
        }

        return {
            "phase": struct_json.get("phase"),
            "current_player": self.current_player(),
            "is_terminal": self.is_terminal(),
            "winner": winner,
            "returns": returns_list,
            "knock_card": struct_json.get("knock_card"),
            "prev_upcard": struct_json.get("prev_upcard"),
            "upcard": struct_json.get("upcard"),
            "stock_size": struct_json.get("stock_size", 0),
            "discard_pile": struct_json.get("discard_pile") or [],
            "hands": hands,
            "hand_hidden": hand_hidden,
            "deadwood": deadwood,
            "layed_melds": layed_melds,
            "layoffs": struct_json.get("layoffs") or [],
            "knocked": struct_json.get("knocked") or [False, False],
            "finished_layoffs": bool(struct_json.get("finished_layoffs", False)),
            "pass_on_first_upcard": struct_json.get("pass_on_first_upcard") or [False, False],
            "observing_player": observer,
        }

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
