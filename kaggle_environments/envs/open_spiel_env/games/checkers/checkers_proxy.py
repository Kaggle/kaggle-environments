"""Structured JSON observations for Checkers.

Checkers (American draughts) is a two-player strategy board game played on an
8x8 board.  Player 0 ('o') starts on the bottom three rows and moves first;
player 1 ('+') starts on the top three rows.  Pieces move and capture
diagonally.  A piece reaching the opponent's back rank is promoted to a king,
which can move in all four diagonal directions.

OpenSpiel's default observation string is ASCII art with rank labels and a
file-label footer (e.g. ``"8.+.+.+.+\\n7+.+.+.+.\\n...\\n abcdefgh\\n"``).
Piece characters are ``o`` / ``+`` for regular pieces and ``8`` / ``*`` for
kings (player 0 / player 1 respectively).  The proxy parses this into a
structured JSON dict so agents and visualizers can consume the state without
re-parsing the ASCII.
"""

import json
from typing import Any

import pyspiel

from ... import proxy

# Raw OpenSpiel piece characters.
_RAW_MAN_P0 = "o"
_RAW_MAN_P1 = "+"
_RAW_KING_P0 = "8"
_RAW_KING_P1 = "*"

# JSON-friendly piece labels.
_MAN_P0 = "o"
_MAN_P1 = "+"
_KING_P0 = "O"
_KING_P1 = "*"
_EMPTY = "."

_PIECE_MAP: dict[str, str] = {
    _RAW_MAN_P0: _MAN_P0,
    _RAW_MAN_P1: _MAN_P1,
    _RAW_KING_P0: _KING_P0,
    _RAW_KING_P1: _KING_P1,
    ".": _EMPTY,
}


def _player_string(player: int) -> str:
    if player < 0:
        return pyspiel.PlayerId(player).name.lower()
    if player == 0:
        return _MAN_P0
    if player == 1:
        return _MAN_P1
    raise ValueError(f"Invalid player: {player}")


class CheckersState(proxy.State):
    """Checkers state proxy returning structured JSON observations.

    Board indexing: ``board[r][c]`` where ``r=0`` is rank 1 (bottom) and
    ``r=7`` is rank 8 (top); ``c=0`` is file ``a`` and ``c=7`` is file ``h``.
    Cells are ``"o"`` (player 0 man), ``"+"`` (player 1 man), ``"O"``
    (player 0 king), ``"*"`` (player 1 king), or ``"."`` (empty).
    """

    def _parse_board(self) -> list[list[str]]:
        """Parse the OpenSpiel ASCII observation into a 2-D grid.

        Returns ``board`` where ``board[0]`` is rank 1 (bottom row) and
        ``board[7]`` is rank 8 (top row), matching algebraic notation.
        """
        raw = self.__wrapped__.observation_string(0)
        lines = raw.strip().split("\n")
        # First 8 lines are ranks 8..1 (top to bottom), each prefixed by rank number.
        # Last line is column labels (" abcdefgh").
        ranks_top_down: list[list[str]] = []
        for line in lines[:8]:
            cells = [_PIECE_MAP.get(ch, ch) for ch in line[1:]]
            ranks_top_down.append(cells)
        # Reverse so board[0] is rank 1 (bottom).
        return list(reversed(ranks_top_down))

    def _last_move_str(self) -> str | None:
        history = self.history()
        if not history:
            return None
        last = history[-1]
        prev_player = 1 - self.current_player() if not self.is_terminal() else 0
        return self.__wrapped__.action_to_string(prev_player, last)

    def _piece_counts(self, board: list[list[str]]) -> dict[str, int]:
        counts = {_MAN_P0: 0, _MAN_P1: 0, _KING_P0: 0, _KING_P1: 0}
        for row in board:
            for cell in row:
                if cell in counts:
                    counts[cell] += 1
        return counts

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        del player
        board = self._parse_board()
        piece_counts = self._piece_counts(board)

        winner: str | None = None
        if self.is_terminal():
            returns = self.returns()
            if returns[0] > returns[1]:
                winner = _MAN_P0
            elif returns[1] > returns[0]:
                winner = _MAN_P1
            else:
                winner = "draw"

        return {
            "board": board,
            "current_player": _player_string(self.current_player()),
            "is_terminal": self.is_terminal(),
            "winner": winner,
            "move_number": self.move_number(),
            "last_move": self._last_move_str(),
            "piece_counts": piece_counts,
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class CheckersGame(proxy.Game):
    """Wraps OpenSpiel's checkers game to use the proxy state."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("checkers", params)
        super().__init__(
            wrapped,
            short_name="checkers_proxy",
            long_name="Checkers (proxy)",
        )

    def new_initial_state(self, *args) -> CheckersState:
        return CheckersState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(CheckersGame().get_type(), CheckersGame)
