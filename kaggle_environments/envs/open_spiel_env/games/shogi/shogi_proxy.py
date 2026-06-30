"""Structured JSON observations for Shogi.

Shogi (Japanese chess) is a two-player perfect-information game played on a
9x9 board. Each side starts with 20 pieces: a king, a rook, a bishop, two
gold generals, two silver generals, two knights, two lances, and nine
pawns. Most pieces can promote when they move into, within, or out of the
opponent's back three ranks. Captured pieces switch sides and may be
re-introduced ("dropped") onto an empty square as a friendly piece on a
later turn. Player 0 plays Sente ("Black", uppercase pieces) and moves
first; player 1 plays Gote ("White", lowercase pieces).

OpenSpiel's default observation string is a single line of SFEN, e.g.::

    lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1

That is four space-separated fields: board, side-to-move, pieces in hand,
and move number. The board is nine ``/``-separated ranks, each rank a
run-length-encoded sequence of cells (digits = empty squares, letters =
pieces, ``+`` prefix = promoted piece). The proxy parses this into a
structured JSON dict so agents and visualizers do not need to re-implement
SFEN parsing. Action strings use OpenSpiel's USI notation: ``<from><to>``
for a board move (e.g. ``"7g7f"``), with a trailing ``+`` for promotion
(e.g. ``"2c2b+"``), and ``<PIECE>*<square>`` for a drop (e.g. ``"P*5e"``).
"""

import json
from typing import Any

import pyspiel

from ... import proxy

EMPTY = "."
SENTE = "b"
GOTE = "w"


def _player_string(player: int) -> str:
    if player < 0:
        return pyspiel.PlayerId(player).name.lower()
    if player == 0:
        return SENTE
    if player == 1:
        return GOTE
    raise ValueError(f"Invalid player: {player}")


def _parse_board(board_str: str) -> list[list[str]]:
    """Parse the SFEN board field into a 9x9 grid.

    Returns a list of 9 ranks top-to-bottom (rank ``a`` first, rank ``i``
    last). Each cell is ``"."`` for an empty square, a single uppercase
    or lowercase letter for an unpromoted piece, or ``"+X"`` / ``"+x"``
    for a promoted piece (Sente / Gote respectively).
    """
    ranks = board_str.split("/")
    board: list[list[str]] = []
    for rank in ranks:
        cells: list[str] = []
        i = 0
        while i < len(rank):
            ch = rank[i]
            if ch.isdigit():
                cells.extend([EMPTY] * int(ch))
                i += 1
            elif ch == "+":
                cells.append("+" + rank[i + 1])
                i += 2
            else:
                cells.append(ch)
                i += 1
        board.append(cells)
    return board


def _parse_captured(captured_str: str) -> dict[str, dict[str, int]]:
    """Parse the SFEN pieces-in-hand field into per-side counts.

    SFEN encodes hands as an optional decimal count followed by a piece
    letter (uppercase = Sente, lowercase = Gote). ``"-"`` means both
    hands are empty. Example: ``"2PNp"`` -> Sente holds 2 pawns and 1
    knight; Gote holds 1 pawn.
    """
    hands: dict[str, dict[str, int]] = {SENTE: {}, GOTE: {}}
    if captured_str == "-":
        return hands
    i = 0
    while i < len(captured_str):
        count_str = ""
        while i < len(captured_str) and captured_str[i].isdigit():
            count_str += captured_str[i]
            i += 1
        if i >= len(captured_str):
            break
        piece = captured_str[i]
        i += 1
        count = int(count_str) if count_str else 1
        side = SENTE if piece.isupper() else GOTE
        hands[side][piece] = hands[side].get(piece, 0) + count
    return hands


class ShogiState(proxy.State):
    """Shogi state proxy returning structured JSON observations."""

    def _parse_sfen(self) -> dict[str, Any]:
        raw = self.__wrapped__.observation_string(0)
        parts = raw.split(" ")
        board_str = parts[0]
        side = parts[1] if len(parts) > 1 else SENTE
        captured_str = parts[2] if len(parts) > 2 else "-"
        move_number = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 1
        return {
            "board": _parse_board(board_str),
            "side_to_move": side,
            "captured": _parse_captured(captured_str),
            "move_number": move_number,
            "sfen": raw,
        }

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        del player
        parsed = self._parse_sfen()

        winner: str | None = None
        if self.is_terminal():
            returns = self.returns()
            if returns[0] > returns[1]:
                winner = SENTE
            elif returns[1] > returns[0]:
                winner = GOTE
            else:
                winner = "draw"

        # Shogi action strings depend on board context (e.g. piece type at
        # the from-square), so replay from the initial state to render
        # each historical action correctly.
        clone = self.get_game().__wrapped__.new_initial_state()
        move_history: list[str] = []
        for action in self.history():
            move_history.append(clone.action_to_string(action))
            clone.apply_action(action)
        last_move = move_history[-1] if move_history else None

        return {
            "board": parsed["board"],
            "current_player": _player_string(self.current_player()),
            "is_terminal": self.is_terminal(),
            "winner": winner,
            "captured": parsed["captured"],
            "move_number": parsed["move_number"],
            "last_move": last_move,
            "move_history": move_history,
            "sfen": parsed["sfen"],
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class ShogiGame(proxy.Game):
    """Wraps OpenSpiel's shogi game to use the proxy state."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("shogi", params)
        super().__init__(
            wrapped,
            short_name="shogi_proxy",
            long_name="Shogi (proxy)",
        )

    def new_initial_state(self, *args) -> ShogiState:
        return ShogiState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(ShogiGame().get_type(), ShogiGame)
