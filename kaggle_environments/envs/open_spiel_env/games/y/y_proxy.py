"""Structured JSON observations for Y.

Y is a connection game on a triangular board. Player 0 ('x') and player 1 ('o')
take turns placing stones. A player wins by connecting all three sides of the
triangle with a single chain of their stones. Actions are encoded as
``row * board_size + col`` (both 0-indexed). On a board of size N, row r has
N - r playable cells, indexed by columns 0..N-r-1.
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
    row = action // board_size
    col = action % board_size
    return f"{chr(ord('a') + col)}{row + 1}"


class YState(proxy.State):
    """Y state proxy."""

    def _board_from_history(self, board_size: int) -> list[list[str | None]]:
        board: list[list[str | None]] = [[None] * (board_size - r) for r in range(board_size)]
        for move_index, action in enumerate(self.history()):
            row = action // board_size
            col = action % board_size
            board[row][col] = "x" if move_index % 2 == 0 else "o"
        return board

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        del player
        params = self.get_game().get_parameters()
        board_size = params.get("board_size", 19)
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


class YGame(proxy.Game):
    """Y game proxy."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("y", params)
        super().__init__(
            wrapped,
            short_name="y_proxy",
            long_name="Y (proxy)",
        )

    def new_initial_state(self, *args) -> YState:
        return YState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(YGame().get_type(), YGame)
