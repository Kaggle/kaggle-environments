"""Structured JSON observations for Havannah.

Havannah is a connection game on a hexagonal board of side length N (default 8),
played on a rhombic grid of diameter D = 2N - 1 with the two opposing corners
cut off. Player 0 ('x') and player 1 ('o') take turns placing stones. A player
wins by forming one of three structures from their own stones: a *ring* (a loop
enclosing one or more cells), a *bridge* (chain joining any two of the six
corners), or a *fork* (chain joining any three of the six edges).

Actions are encoded as ``x + y * diameter``, where ``x, y`` are 0-indexed grid
coordinates. The string form of a move is ``<col><row>`` with column letter
'a' + x and 1-indexed row y + 1 (e.g. action 0 == "a1").
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


def _action_to_coord(action: int, board_size: int) -> str:
    diameter = board_size * 2 - 1
    x = action % diameter
    y = action // diameter
    return f"{chr(ord('a') + x)}{y + 1}"


def _row_x_range(y: int, board_size: int) -> tuple[int, int]:
    """Return [start_x, end_x) of playable cells in row y."""
    diameter = board_size * 2 - 1
    if y < board_size:
        return 0, board_size + y
    return y - board_size + 1, diameter


class HavannahState(proxy.State):
    """Havannah state proxy."""

    def _board_from_history(self, board_size: int) -> list[list[str | None]]:
        diameter = board_size * 2 - 1
        rows: list[list[str | None]] = []
        for y in range(diameter):
            start_x, end_x = _row_x_range(y, board_size)
            rows.append([None] * (end_x - start_x))
        for move_index, action in enumerate(self.history()):
            x = action % diameter
            y = action // diameter
            start_x, _ = _row_x_range(y, board_size)
            rows[y][x - start_x] = "x" if move_index % 2 == 0 else "o"
        return rows

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        del player
        params = self.get_game().get_parameters()
        board_size = params.get("board_size", 8)
        history = self.history()
        last_move = _action_to_coord(history[-1], board_size) if history else None
        winner = None
        if self.is_terminal():
            returns = self.returns()
            if returns[0] > returns[1]:
                winner = "x"
            elif returns[1] > returns[0]:
                winner = "o"
            else:
                winner = "draw"
        return {
            "board": self._board_from_history(board_size),
            "board_size": board_size,
            "current_player": _player_string(self.current_player()),
            "is_terminal": self.is_terminal(),
            "winner": winner,
            "last_move": last_move,
            "move_number": len(history),
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class HavannahGame(proxy.Game):
    """Havannah game proxy."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("havannah", params)
        super().__init__(
            wrapped,
            short_name="havannah_proxy",
            long_name="Havannah (proxy)",
        )

    def new_initial_state(self, *args) -> HavannahState:
        return HavannahState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(HavannahGame().get_type(), HavannahGame)
