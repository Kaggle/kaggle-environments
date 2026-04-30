"""Structured JSON observations for Crazyhouse.

OpenSpiel's default observation for Crazyhouse is a Crazyhouse-FEN string,
e.g. ``rnbqkbnr/.../RNBQKBNR[Pn] w KQkq - 0 2``. Pieces are encoded as
single letters (uppercase = White, lowercase = Black) and the bracketed
section after the placement field lists each player's pocket. This proxy
parses that into a structured dict for agents.

Note: in OpenSpiel's Crazyhouse, ``current_player()`` returns ``0`` for
Black and ``1`` for White (see ``ColorToPlayer`` in crazyhouse.h).
"""

import json
from typing import Any

import pyspiel

from ... import proxy


_PLAYER_BLACK = 0
_PLAYER_WHITE = 1


def _parse_fen(fen: str) -> dict[str, Any]:
    """Parse a Crazyhouse FEN into structured fields."""
    placement_with_pockets, side, castling, en_passant, halfmove, fullmove = fen.split()

    if "[" in placement_with_pockets:
        placement, pocket_str = placement_with_pockets.split("[", 1)
        pocket_str = pocket_str.rstrip("]")
    else:
        placement, pocket_str = placement_with_pockets, ""

    # Board: rank 8 first (matches FEN order), each row is a list of 8 chars.
    board: list[list[str]] = []
    for rank in placement.split("/"):
        row: list[str] = []
        for ch in rank:
            if ch.isdigit():
                row.extend(["."] * int(ch))
            else:
                row.append(ch)
        board.append(row)

    white_pocket: dict[str, int] = {}
    black_pocket: dict[str, int] = {}
    for ch in pocket_str:
        if ch.isupper():
            white_pocket[ch] = white_pocket.get(ch, 0) + 1
        else:
            key = ch.upper()
            black_pocket[key] = black_pocket.get(key, 0) + 1

    return {
        "board": board,
        "side_to_move": side,
        "castling_rights": castling,
        "en_passant": en_passant,
        "halfmove_clock": int(halfmove),
        "fullmove_number": int(fullmove),
        "pockets": {"white": white_pocket, "black": black_pocket},
    }


class CrazyhouseState(proxy.State):
    """Wraps OpenSpiel Crazyhouse state with JSON observations."""

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        del player
        fen = self.__wrapped__.observation_string(0)
        parsed = _parse_fen(fen)

        winner: str | None = None
        if self.is_terminal():
            returns = self.returns()
            if returns[_PLAYER_WHITE] > returns[_PLAYER_BLACK]:
                winner = "white"
            elif returns[_PLAYER_BLACK] > returns[_PLAYER_WHITE]:
                winner = "black"
            else:
                winner = "draw"

        current = self.current_player()
        if current == _PLAYER_WHITE:
            current_label: Any = "white"
        elif current == _PLAYER_BLACK:
            current_label = "black"
        else:
            current_label = current

        return {
            "fen": fen,
            "board": parsed["board"],
            "side_to_move": parsed["side_to_move"],
            "castling_rights": parsed["castling_rights"],
            "en_passant": parsed["en_passant"],
            "halfmove_clock": parsed["halfmove_clock"],
            "fullmove_number": parsed["fullmove_number"],
            "pockets": parsed["pockets"],
            "current_player": current_label,
            "is_terminal": self.is_terminal(),
            "winner": winner,
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class CrazyhouseGame(proxy.Game):
    """Crazyhouse game proxy."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("crazyhouse", params)
        super().__init__(
            wrapped,
            short_name="crazyhouse_proxy",
            long_name="Crazyhouse (proxy)",
        )

    def new_initial_state(self, *args) -> CrazyhouseState:
        return CrazyhouseState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(CrazyhouseGame().get_type(), CrazyhouseGame)
