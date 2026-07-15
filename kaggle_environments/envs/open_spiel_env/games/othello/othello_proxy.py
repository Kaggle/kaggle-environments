"""Structured JSON observations for Othello (Reversi).

Othello is a two-player abstract game played on an 8x8 board. Player 0
('x', Black) moves first; player 1 ('o', White) follows. Each move places
a disk of the mover's colour on an empty cell and flips every contiguous
run of opposing disks that is bounded on the far side by one of the
mover's disks. A player who has no legal placement must pass; the game
ends when neither player has a legal move (usually when the board is
full). The player with more disks at the end wins; equal counts draw.

OpenSpiel's default observation string is an ASCII board with row/column
labels, e.g.::

      a b c d e f g h
    1 - - - - - - - - 1
    2 - - - - - - - - 2
    3 - - - - - - - - 3
    4 - - - o x - - - 4
    5 - - - x o - - - 5
    6 - - - - - - - - 6
    7 - - - - - - - - 7
    8 - - - - - - - - 8
      a b c d e f g h

The proxy parses this into a structured JSON dict so agents and
visualizers can consume the board and metadata without re-parsing the
ASCII. Action strings use algebraic notation ``<file><rank>``
(e.g. ``"d3"``) with ``"pass"`` for the no-move action.
"""

import json
from typing import Any

import pyspiel

from ... import proxy

PIECE_BLACK = "x"
PIECE_WHITE = "o"
EMPTY = ""

_BOARD_SIZE = 8
_PASS_ACTION = _BOARD_SIZE * _BOARD_SIZE  # OpenSpiel encodes pass as action 64.


def _player_string(player: int) -> str:
    if player < 0:
        return pyspiel.PlayerId(player).name.lower()
    if player == 0:
        return PIECE_BLACK
    if player == 1:
        return PIECE_WHITE
    raise ValueError(f"Invalid player: {player}")


class OthelloState(proxy.State):
    """Othello state proxy returning structured JSON observations."""

    def _parse_board(self) -> list[list[str]]:
        """Parse the OpenSpiel ASCII observation into an 8x8 grid.

        ``board[0]`` is rank 1 (top of the display) and ``board[r][0]`` is
        file 'a'. Cells are one of ``"x"``, ``"o"``, ``""``.
        """
        raw = self.__wrapped__.observation_string(0)
        board: list[list[str]] = []
        for line in raw.split("\n"):
            stripped = line.lstrip()
            # Board rows begin with a rank digit ("1 - - - ..."); the
            # header ("Black (x) to play:") and column-label lines
            # ("  a b c d ...") do not.
            if not stripped or not stripped[0].isdigit():
                continue
            tokens = stripped.split()
            # Drop leading and trailing rank labels, keep the 8 cell tokens.
            cells = tokens[1:-1]
            row = [PIECE_BLACK if c == PIECE_BLACK else PIECE_WHITE if c == PIECE_WHITE else EMPTY for c in cells]
            board.append(row)
        return board

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        del player
        board = self._parse_board()

        winner: str | None = None
        if self.is_terminal():
            returns = self.returns()
            if returns[0] > returns[1]:
                winner = PIECE_BLACK
            elif returns[1] > returns[0]:
                winner = PIECE_WHITE
            else:
                winner = "draw"

        # Othello is turn-alternating with no chance nodes; passes are
        # regular actions that flip the current player like any other,
        # so action i was played by (i % 2).
        history = self.history()
        move_history: list[str] = [self.__wrapped__.action_to_string(i % 2, action) for i, action in enumerate(history)]
        last_move = move_history[-1] if move_history else None

        disks = {PIECE_BLACK: 0, PIECE_WHITE: 0}
        for row in board:
            for cell in row:
                if cell in disks:
                    disks[cell] += 1

        legal = self.legal_actions()
        must_pass = (not self.is_terminal()) and legal == [_PASS_ACTION]

        return {
            "board": board,
            "rows": _BOARD_SIZE,
            "columns": _BOARD_SIZE,
            "current_player": _player_string(self.current_player()),
            "is_terminal": self.is_terminal(),
            "winner": winner,
            "disks": disks,
            "last_move": last_move,
            "move_history": move_history,
            "move_number": self.move_number(),
            "must_pass": must_pass,
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class OthelloGame(proxy.Game):
    """Wraps OpenSpiel's othello game to use the proxy state."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("othello", params)
        super().__init__(
            wrapped,
            short_name="othello_proxy",
            long_name="Othello (proxy)",
        )

    def new_initial_state(self, *args) -> OthelloState:
        return OthelloState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(OthelloGame().get_type(), OthelloGame)
