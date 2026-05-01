"""Structured JSON observations for Dark Hex.

Dark Hex is an imperfect-information variant of Hex: each player only sees
their own pieces. When a player attempts to play on a cell occupied by the
opponent, the move is rejected and the opponent's piece becomes visible at
that cell -- the player then takes another turn. Player 0 ('x') connects
top to bottom; player 1 ('o') connects left to right.
"""

import json
from typing import Any

import pyspiel

from ... import proxy


def _player_string(player: int) -> str:
    if player < 0:
        return pyspiel.PlayerId(player).name.lower()
    if player == 0:
        return "x"
    if player == 1:
        return "o"
    raise ValueError(f"Invalid player: {player}")


class DarkHexState(proxy.State):
    """Dark Hex state proxy."""

    def _board_from_observation(self, player: int) -> list[list[str]]:
        raw = self.__wrapped__.observation_string(player)
        return [list(row) for row in raw.split("\n") if row]

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        params = self.get_game().get_parameters()
        num_rows = params.get("num_rows", params.get("board_size", 3))
        num_cols = params.get("num_cols", params.get("board_size", 3))
        winner = None
        if self.is_terminal():
            returns = self.returns()
            if returns[0] > returns[1]:
                winner = "x"
            elif returns[1] > returns[0]:
                winner = "o"
            else:
                winner = "draw"
        result: dict[str, Any] = {
            "current_player": _player_string(self.current_player()),
            "is_terminal": self.is_terminal(),
            "winner": winner,
            "num_rows": num_rows,
            "num_cols": num_cols,
        }
        if player is None:
            result["board_player_x"] = self._board_from_observation(0)
            result["board_player_o"] = self._board_from_observation(1)
        else:
            result["board"] = self._board_from_observation(player)
        return result

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class DarkHexGame(proxy.Game):
    """Dark Hex game proxy."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("dark_hex", params)
        super().__init__(
            wrapped,
            short_name="dark_hex_proxy",
            long_name="Dark Hex (proxy)",
        )

    def new_initial_state(self, *args) -> DarkHexState:
        return DarkHexState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(DarkHexGame().get_type(), DarkHexGame)
