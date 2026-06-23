"""Structured JSON observations for Breakthrough.

Breakthrough is a two-player abstract game played on an N x M board
(default 8 x 8). Player 0 ('b', Black) starts at the top of the board and
moves down; player 1 ('w', White) starts at the bottom and moves up. On
each turn the active player moves one of their pieces one cell forward,
forward-diagonal-left, or forward-diagonal-right. Diagonal moves may
capture an opposing piece; forward moves may not. A player wins by
reaching the opponent's back rank or by capturing all opposing pieces.

OpenSpiel's default observation string is an ASCII board with row labels
and a column-label footer, e.g.::

    8bbbbbbbb
    7bbbbbbbb
    6........
    5........
    4........
    3........
    2wwwwwwww
    1wwwwwwww
     abcdefgh

The proxy parses this into a structured JSON dict so agents and
visualizers can consume the board and metadata without re-parsing the
ASCII. Action strings use OpenSpiel's algebraic notation
``<from><to>[*]`` (e.g. ``"a7a6"``, ``"b2c3*"`` where ``*`` marks a
capture).
"""

import json
from typing import Any

import pyspiel

from ... import proxy

_PIECE_BLACK = "b"
_PIECE_WHITE = "w"
_EMPTY = "."


def _player_string(player: int) -> str:
    if player < 0:
        return pyspiel.PlayerId(player).name.lower()
    if player == 0:
        return _PIECE_BLACK
    if player == 1:
        return _PIECE_WHITE
    raise ValueError(f"Invalid player: {player}")


class BreakthroughState(proxy.State):
    """Breakthrough state proxy returning structured JSON observations."""

    def _parse_board(self) -> tuple[list[list[str]], int, int]:
        """Parse the OpenSpiel ASCII observation into a 2-D grid.

        Returns ``(board, rows, columns)`` where ``board[0]`` is the top
        row of the visual board (the row labeled with the highest rank,
        which is Black's home row), and each cell is one of ``"b"``,
        ``"w"``, ``"."``.
        """
        raw = self.__wrapped__.observation_string(0)
        lines = [ln for ln in raw.split("\n") if ln.strip()]
        # Last line is the column labels (" abcdefgh"); drop it.
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
                winner = _PIECE_BLACK
            elif returns[1] > returns[0]:
                winner = _PIECE_WHITE
            else:
                # Breakthrough cannot draw under standard rules, but keep
                # the branch so a future rules change doesn't silently
                # fall through to winner=None.
                winner = "draw"

        # Breakthrough has no chance phase; play actions alternate
        # starting with player 0, so the last mover's id is parity of
        # (move_number - 1).
        history = self.history()
        last_move = self.__wrapped__.action_to_string((len(history) - 1) % 2, history[-1]) if history else None

        pieces = {_PIECE_BLACK: 0, _PIECE_WHITE: 0}
        for row in board:
            for cell in row:
                if cell in pieces:
                    pieces[cell] += 1

        return {
            "board": board,
            "rows": rows,
            "columns": columns,
            "current_player": _player_string(self.current_player()),
            "is_terminal": self.is_terminal(),
            "winner": winner,
            "last_move": last_move,
            "move_number": self.move_number(),
            "pieces": pieces,
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


class BreakthroughGame(proxy.Game):
    """Wraps OpenSpiel's breakthrough game to use the proxy state."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("breakthrough", params)
        super().__init__(
            wrapped,
            short_name="breakthrough_proxy",
            long_name="Breakthrough (proxy)",
        )

    def new_initial_state(self, *args) -> BreakthroughState:
        return BreakthroughState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(BreakthroughGame().get_type(), BreakthroughGame)
