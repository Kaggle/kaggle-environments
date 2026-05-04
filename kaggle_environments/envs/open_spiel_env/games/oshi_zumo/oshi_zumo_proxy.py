"""Structured JSON observations for Oshi-Zumo.

Oshi-Zumo is a simultaneous-move bidding game. Both players have a coin
purse and bid each round to push a wrestler toward their opponent's edge of
a field of length ``2*size + 1``. The default OpenSpiel observation string
looks like ``"Coins: 50 50, Field: #...W...#\\n"`` -- the proxy parses this
into a structured dict with the wrestler's position, each player's remaining
coins, and game metadata.
"""

import json
from typing import Any

import pyspiel

from ... import proxy

_BOUNDARY = "#"
_OPEN = "."
_WRESTLER = "W"


class OshiZumoState(proxy.State):
    """Oshi-Zumo state proxy returning structured JSON observations."""

    def _apply_actions(self, actions: list[int]) -> None:
        # Override for simultaneous-move games; the proxy base class only
        # forwards the sequential _apply_action.
        return self.__wrapped__.apply_actions(actions)

    def _player_label(self, player: int) -> int | str:
        if player < 0:
            return pyspiel.PlayerId(player).name.lower()
        return player

    def _parse_observation(self) -> dict[str, Any]:
        raw = self.__wrapped__.observation_string(0).strip()
        # Format: "Coins: <c0> <c1>, Field: <field>"
        coins_part, field_part = raw.split(", Field: ", 1)
        coins_tokens = coins_part[len("Coins: ") :].split()
        coins = [int(coins_tokens[0]), int(coins_tokens[1])]
        field = field_part
        wrestler_pos = field.index(_WRESTLER) if _WRESTLER in field else -1
        return {
            "coins": coins,
            "field": field,
            "field_size": len(field),
            "wrestler_position": wrestler_pos,
        }

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        del player
        obs = self._parse_observation()

        winner: int | str | None = None
        if self.is_terminal():
            returns = self.returns()
            if returns[0] > returns[1]:
                winner = 0
            elif returns[1] > returns[0]:
                winner = 1
            else:
                winner = "draw"

        params = self.get_game().get_parameters()

        return {
            "field": obs["field"],
            "field_size": obs["field_size"],
            "wrestler_position": obs["wrestler_position"],
            "coins": obs["coins"],
            "current_player": self._player_label(self.current_player()),
            "move_number": self.move_number(),
            "is_terminal": self.is_terminal(),
            "winner": winner,
            "params": {
                "alesia": bool(params.get("alesia", False)),
                "starting_coins": int(params.get("coins", 50)),
                "size": int(params.get("size", 3)),
                "horizon": int(params.get("horizon", 1000)),
                "min_bid": int(params.get("min_bid", 0)),
            },
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class OshiZumoGame(proxy.Game):
    """Wraps OpenSpiel's oshi_zumo game to use the proxy state."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("oshi_zumo", params)
        super().__init__(
            wrapped,
            short_name="oshi_zumo_proxy",
            long_name="Oshi Zumo (proxy)",
        )

    def new_initial_state(self, *args) -> OshiZumoState:
        return OshiZumoState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(OshiZumoGame().get_type(), OshiZumoGame)
