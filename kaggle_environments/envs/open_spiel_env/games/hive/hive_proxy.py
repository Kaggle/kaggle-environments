"""Structured JSON observations for Hive.

Hive is a hexagonal tile-placement game where each player tries to surround the
opponent's queen bee. OpenSpiel's default ``observation_string`` is a Universal
Hive Protocol (UHP) gamestring of the form
``GameTypeString;GameStateString;TurnString;Move1;Move2;...;MoveN``. This proxy
parses that string into JSON and additionally reconstructs the axial position of
every played tile by replaying the move list, so agents do not have to parse
UHP move notation themselves.

UHP move notation (per move):
    ``<piece>``                       -- first move of the game; placed at origin
    ``<piece> <ref>``                 -- climbed on top of <ref>
    ``<piece> /<ref>``                -- SW of <ref>
    ``<piece> -<ref>``                -- W of <ref>
    ``<piece> \\<ref>``                -- NW of <ref>
    ``<piece> <ref>/``                -- NE of <ref>
    ``<piece> <ref>-``                -- E of <ref>
    ``<piece> <ref>\\``                -- SE of <ref>
    ``pass``                          -- no legal moves available
"""

import json
from typing import Any

import pyspiel

from ... import proxy

# Axial coordinate offsets for the six neighbour directions (q, r).
_DIRECTION_OFFSETS = {
    "NE": (1, -1),
    "E": (1, 0),
    "SE": (0, 1),
    "SW": (-1, 1),
    "W": (-1, 0),
    "NW": (0, -1),
}


def _parse_uhp_move(move_str: str) -> tuple[str, str | None, str | None]:
    """Return (from_tile, ref_tile, direction).

    ``direction`` is one of the six cardinal directions, ``"Above"`` for a climb,
    or ``None`` for the first move of the game. ``ref_tile`` is ``None`` only
    for the first move.
    """
    if move_str == "pass":
        return ("pass", None, None)
    parts = move_str.split()
    from_tile = parts[0]
    if len(parts) == 1:
        return (from_tile, None, None)
    token = parts[1]
    if token.startswith("\\"):
        return (from_tile, token[1:], "NW")
    if token.startswith("/"):
        return (from_tile, token[1:], "SW")
    if token.startswith("-"):
        return (from_tile, token[1:], "W")
    if token.endswith("/"):
        return (from_tile, token[:-1], "NE")
    if token.endswith("-"):
        return (from_tile, token[:-1], "E")
    if token.endswith("\\"):
        return (from_tile, token[:-1], "SE")
    return (from_tile, token, "Above")


def _compute_tile_positions(moves: list[str]) -> dict[str, list[int]]:
    """Replay moves to compute current (q, r, h) for each played tile."""
    positions: dict[str, list[int]] = {}
    for move_str in moves:
        from_tile, ref_tile, direction = _parse_uhp_move(move_str)
        if from_tile == "pass":
            continue
        if ref_tile is None:
            positions[from_tile] = [0, 0, 0]
            continue
        ref_pos = positions.get(ref_tile)
        if ref_pos is None:
            # Shouldn't happen for a valid game; skip rather than crash.
            continue
        ref_q, ref_r, ref_h = ref_pos
        if direction == "Above":
            new_q, new_r, new_h = ref_q, ref_r, ref_h + 1
        else:
            dq, dr = _DIRECTION_OFFSETS[direction]
            new_q, new_r = ref_q + dq, ref_r + dr
            max_h = -1
            for tile, (q, r, h) in positions.items():
                if tile == from_tile:
                    continue
                if q == new_q and r == new_r and h > max_h:
                    max_h = h
            new_h = max_h + 1
        positions[from_tile] = [new_q, new_r, new_h]
    return positions


def _parse_game_type(game_type: str) -> dict[str, bool]:
    """Parse 'Base+MLP' style expansion suffix into a flag dict."""
    suffix = game_type.split("+", 1)[1] if "+" in game_type else ""
    return {
        "mosquito": "M" in suffix,
        "ladybug": "L" in suffix,
        "pillbug": "P" in suffix,
    }


def _player_string(player: int) -> str:
    if player < 0:
        return pyspiel.PlayerId(player).name.lower()
    if player == 0:
        return "white"
    if player == 1:
        return "black"
    raise ValueError(f"Invalid player: {player}")


_STATUS_TO_WINNER = {
    "WhiteWins": "white",
    "BlackWins": "black",
    "Draw": "draw",
}


class HiveState(proxy.State):
    """Hive state proxy producing structured JSON observations."""

    def _legal_moves_uhp(self) -> list[str]:
        if self.is_terminal():
            return []
        player = self.current_player()
        if player < 0:
            return []
        return [self.action_to_string(player, a) for a in self.legal_actions()]

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        del player
        uhp = self.__wrapped__.observation_string(0)
        tokens = uhp.split(";")
        game_type = tokens[0] if tokens else ""
        status = tokens[1] if len(tokens) > 1 else ""
        turn = tokens[2] if len(tokens) > 2 else ""
        moves = [m for m in tokens[3:] if m]

        params = self.get_game().get_parameters()
        positions = _compute_tile_positions(moves)

        winner: str | None = None
        if self.is_terminal():
            winner = _STATUS_TO_WINNER.get(status)
            if winner is None:
                returns = self.returns()
                if returns[0] > returns[1]:
                    winner = "white"
                elif returns[1] > returns[0]:
                    winner = "black"
                else:
                    winner = "draw"

        return {
            "game_type": game_type,
            "expansions": _parse_game_type(game_type),
            "board_radius": params.get("board_size", 8),
            "status": status,
            "turn": turn,
            "current_player": _player_string(self.current_player()),
            "move_number": len(moves),
            "moves": moves,
            "last_move": moves[-1] if moves else None,
            "legal_moves": self._legal_moves_uhp(),
            "pieces": positions,
            "is_terminal": self.is_terminal(),
            "winner": winner,
            "uhp": uhp,
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class HiveGame(proxy.Game):
    """Hive game proxy."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("hive", params)
        super().__init__(
            wrapped,
            short_name="hive_proxy",
            long_name="Hive (proxy)",
        )

    def new_initial_state(self, *args) -> HiveState:
        return HiveState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(HiveGame().get_type(), HiveGame)
