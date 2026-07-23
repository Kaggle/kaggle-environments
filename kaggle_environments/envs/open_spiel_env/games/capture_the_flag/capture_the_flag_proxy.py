"""Structured JSON observations for OpenSpiel's Capture the Flag.

Capture the Flag is a two-player grid game. Both players pick a move at the
same time each turn -- one of North, East, South, West, or Stay. If they try
to enter the same cell a coin flip decides who goes first. A player wins by
picking up the opponent's flag and carrying it back to their own base, but
only if their own flag is still sitting at home. A flag carrier gets tagged
(sent back to their base, dropping the flag) when the defender is standing
in one of the four cells directly next to them (up/down/left/right, not
diagonal) AND the carrier is inside the defender's half of the board.

OpenSpiel's default observation string is a bare ASCII grid followed by a
few lines of loose text (``Carrier(...)``, ``Score:``, ``Moves:``). This
proxy parses that text into structured JSON with the board, unit and flag
positions, who's carrying what, scores, and end-of-game info.
"""

import json
from typing import Any

import pyspiel

from ... import proxy

_ACTION_NAMES = ["North", "East", "South", "West", "Stay"]


def _parse_grid_param(grid_str: str) -> dict[str, Any]:
    """Extract dimensions, base positions, and obstacles from the grid param."""
    rows = grid_str.split("\n")
    a_base: list[int] | None = None
    b_base: list[int] | None = None
    obstacles: list[list[int]] = []
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            if ch == "a":
                a_base = [r, c]
            elif ch == "b":
                b_base = [r, c]
            elif ch == "*":
                obstacles.append([r, c])
    return {
        "num_rows": len(rows),
        "num_cols": len(rows[0]) if rows else 0,
        "a_base": a_base,
        "b_base": b_base,
        "obstacles": obstacles,
    }


class CaptureTheFlagState(proxy.State):
    """Capture the Flag state proxy with JSON observations."""

    def _params(self) -> dict[str, Any]:
        params = self.get_game().get_parameters()
        grid_str = str(params.get("grid", ""))
        grid_info = _parse_grid_param(grid_str)
        return {
            "horizon": int(params["horizon"]),
            "score_limit": int(params["score_limit"]),
            **grid_info,
        }

    def _parse_raw(self) -> dict[str, Any]:
        """Parse the C++ ToString() output into a structured dict.

        Format (from capture_the_flag.cc :: ToString):
            <num_rows> lines of grid chars: '.', '*', 'A', 'B', 'a', 'b'
            Carrier(A's flag): <int> Carrier(B's flag): <int>
            Score: A=<int> B=<int>
            Moves: <int>/<int>
            [Chance Node]      <- present only at chance nodes
        """
        raw = str(self.__wrapped__)
        params = self._params()
        num_rows = params["num_rows"]
        num_cols = params["num_cols"]

        lines = raw.split("\n")
        board_lines = lines[:num_rows]
        board: list[list[str]] = []
        for r in range(num_rows):
            line = board_lines[r] if r < len(board_lines) else ""
            row = list(line.ljust(num_cols)[:num_cols])
            board.append(row)

        # Locate players and loose flags from the parsed board.
        a_pos: list[int] | None = None
        b_pos: list[int] | None = None
        a_flag_loose: list[int] | None = None
        b_flag_loose: list[int] | None = None
        for r in range(num_rows):
            for c in range(num_cols):
                ch = board[r][c]
                if ch == "A":
                    a_pos = [r, c]
                elif ch == "B":
                    b_pos = [r, c]
                elif ch == "a":
                    a_flag_loose = [r, c]
                elif ch == "b":
                    b_flag_loose = [r, c]

        # A player standing on their own home base with their flag "loose" is
        # rendered as 'A' / 'B' (not 'a' / 'b'); recover the loose-flag position
        # from the carrier line + base positions below.

        carrier_a = -1
        carrier_b = -1
        score_a = 0
        score_b = 0
        moves = 0
        horizon = params["horizon"]
        for line in lines[num_rows:]:
            if line.startswith("Carrier(A's flag):"):
                # "Carrier(A's flag): -1 Carrier(B's flag): -1"
                parts = line.split()
                # tokens: ["Carrier(A's", "flag):", "<int>", "Carrier(B's", "flag):", "<int>"]
                try:
                    carrier_a = int(parts[2])
                    carrier_b = int(parts[5])
                except (IndexError, ValueError):
                    pass
            elif line.startswith("Score:"):
                # "Score: A=0 B=0"
                for tok in line.split()[1:]:
                    if tok.startswith("A="):
                        score_a = int(tok[2:])
                    elif tok.startswith("B="):
                        score_b = int(tok[2:])
            elif line.startswith("Moves:"):
                # "Moves: 3/1000"
                slash = line.split()[1]
                left, right = slash.split("/")
                moves = int(left)
                horizon = int(right)

        # Reconstruct flag positions.
        # Carrier == -1: flag is loose. If we spotted 'a'/'b' on the board that
        # is its position; otherwise it's on the owner's home base (the player
        # is standing on their own flag and the char shows 'A'/'B').
        # Carrier == pid: flag rides with that player's position.
        if carrier_a == -1:
            flag_a_pos = a_flag_loose if a_flag_loose is not None else params["a_base"]
        else:
            flag_a_pos = a_pos if carrier_a == 0 else b_pos
        if carrier_b == -1:
            flag_b_pos = b_flag_loose if b_flag_loose is not None else params["b_base"]
        else:
            flag_b_pos = a_pos if carrier_b == 0 else b_pos

        return {
            "board": board,
            "a_pos": a_pos,
            "b_pos": b_pos,
            "flag_a_pos": flag_a_pos,
            "flag_b_pos": flag_b_pos,
            "carrier_a": None if carrier_a == -1 else carrier_a,
            "carrier_b": None if carrier_b == -1 else carrier_b,
            "score_a": score_a,
            "score_b": score_b,
            "move_number": moves,
            "horizon": horizon,
        }

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        del player  # Perfect-information game: both players see identical state.
        params = self._params()
        parsed = self._parse_raw()

        winner: int | str | None = None
        if self.is_terminal():
            returns = list(self.returns())
            if returns[0] > returns[1]:
                winner = 0
            elif returns[1] > returns[0]:
                winner = 1
            else:
                winner = "draw"

        cur = self.current_player()
        if cur == pyspiel.PlayerId.SIMULTANEOUS:
            current_player_str: int | str = "simultaneous"
        elif cur == pyspiel.PlayerId.TERMINAL:
            current_player_str = "terminal"
        elif cur == pyspiel.PlayerId.CHANCE:
            current_player_str = "chance"
        else:
            current_player_str = cur

        return {
            "board": parsed["board"],
            "num_rows": params["num_rows"],
            "num_cols": params["num_cols"],
            "a_base": params["a_base"],
            "b_base": params["b_base"],
            "obstacles": params["obstacles"],
            "a_pos": parsed["a_pos"],
            "b_pos": parsed["b_pos"],
            "flag_a_pos": parsed["flag_a_pos"],
            "flag_b_pos": parsed["flag_b_pos"],
            "carrier_a": parsed["carrier_a"],
            "carrier_b": parsed["carrier_b"],
            "score": [parsed["score_a"], parsed["score_b"]],
            "score_limit": params["score_limit"],
            "horizon": parsed["horizon"],
            "move_number": parsed["move_number"],
            "current_player": current_player_str,
            "action_names": _ACTION_NAMES,
            "is_terminal": self.is_terminal(),
            "winner": winner,
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class CaptureTheFlagGame(proxy.Game):
    """Wraps OpenSpiel's Capture the Flag game to use the proxy state."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("capture_the_flag", params)
        super().__init__(
            wrapped,
            short_name="capture_the_flag_proxy",
            long_name="Capture the Flag (proxy)",
        )

    def new_initial_state(self, *args) -> CaptureTheFlagState:
        return CaptureTheFlagState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(CaptureTheFlagGame().get_type(), CaptureTheFlagGame)
