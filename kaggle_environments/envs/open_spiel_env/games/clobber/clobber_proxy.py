"""Structured JSON observations for Clobber.

Clobber is a two-player combinatorial game on an N x M checkerboard.
Player 0 ('o', White) moves first; player 1 ('x', Black) moves second. On
each turn the active player picks one of their pieces and moves it onto an
orthogonally adjacent square that holds an opponent's piece, capturing it.
A player who has no legal move loses.

OpenSpiel's default observation string is the ASCII board with row labels
and a column-label footer (e.g. ``"4xoxo\\n3oxox\\n...\\n abcd\\n"``). The
proxy parses this into a structured JSON dict so agents and visualizers
can consume the board and metadata without re-parsing the ASCII.
"""

import json
from typing import Any

import pyspiel

from ... import proxy

_PIECE_WHITE = "o"
_PIECE_BLACK = "x"
_EMPTY = "."


def _player_string(player: int) -> str:
    if player < 0:
        return pyspiel.PlayerId(player).name.lower()
    if player == 0:
        return _PIECE_WHITE
    if player == 1:
        return _PIECE_BLACK
    raise ValueError(f"Invalid player: {player}")


class ClobberState(proxy.State):
    """Clobber state proxy returning structured JSON observations."""

    def _parse_board(self) -> tuple[list[list[str]], int, int]:
        """Parse the OpenSpiel ASCII observation into a 2-D grid.

        Returns ``(board, rows, columns)`` where ``board[0]`` is the top
        row of the visual board (matching how OpenSpiel renders it), each
        cell is one of ``"o"``, ``"x"``, ``"."``.
        """
        raw = self.__wrapped__.observation_string(0)
        # Strip the trailing column-label line and any trailing blank lines.
        lines = [ln for ln in raw.split("\n") if ln.strip()]
        # Last line is the column labels (" abcd"), drop it.
        body_lines = lines[:-1]
        board: list[list[str]] = []
        for line in body_lines:
            # Strip the leading row label (digits + optional padding space).
            stripped = line.lstrip(" 0123456789")
            board.append(list(stripped))
        if not board:
            return [], 0, 0
        return board, len(board), len(board[0])

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        del player
        board, rows, columns = self._parse_board()
        params = self.get_game().get_parameters()

        winner: str | None = None
        if self.is_terminal():
            returns = self.returns()
            if returns[0] > returns[1]:
                winner = _PIECE_WHITE
            elif returns[1] > returns[0]:
                winner = _PIECE_BLACK
            else:
                winner = "draw"

        history = self.history()
        last_move = (
            self.__wrapped__.action_to_string(0, history[-1]) if history else None
        )

        return {
            "board": board,
            "rows": rows,
            "columns": columns,
            "current_player": _player_string(self.current_player()),
            "is_terminal": self.is_terminal(),
            "winner": winner,
            "last_move": last_move,
            "move_number": self.move_number(),
            "params": {
                "rows": int(params.get("rows", rows)),
                "columns": int(params.get("columns", columns)),
            },
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class ClobberGame(proxy.Game):
    """Wraps OpenSpiel's clobber game to use the proxy state."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("clobber", params)
        super().__init__(
            wrapped,
            short_name="clobber_proxy",
            long_name="Clobber (proxy)",
        )

    def new_initial_state(self, *args) -> ClobberState:
        return ClobberState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(ClobberGame().get_type(), ClobberGame)
