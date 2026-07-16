"""Structured JSON observations for Ultimate Tic Tac Toe.

Ultimate Tic-Tac-Toe is a strategic variant of Tic-Tac-Toe played on a board
consisting of nine 3x3 Tic-Tac-Toe sub-grids arranged in a larger 3x3 grid.
Player 0 ('x') moves first; player 1 ('o') follows. The sub-grid where a
player must place their piece is determined by the cell coordinates of the
opponent's previous move.

OpenSpiel's default `to_string()` observation representation for Ultimate
Tic Tac Toe is a formatted ASCII grid where sub-grids are separated by visual
grid dividers. The proxy state parses this representation into a structured
JSON dictionary so that agents and visualizers can easily consume the state:

Transformations:
- Parses the ASCII board into a list of nine 9-element lists `board`, where
  `board[subgrid_idx][cell_idx]` stores `"x"`, `"o"`, or `""` (empty).
- Evaluates and tracks `subgrid_winners` representing the state of each
  sub-grid (`"x"`, `"o"`, `"draw"`, or `""`).
- Reads `active_subgrid` (index of the sub-grid where the player must play,
  or `None` if any incomplete subgrid is allowed) and `phase` (either
  `"choose_subgrid"` or `"choose_cell"`) directly from the engine's
  "Forced board:" line in the observation string.
- Derives `board_context` (`"opening"`, `"redirected"`, `"self_selected"`,
  `"opponent_directed"`, or `None` when terminal) explaining *why* the mover
  is in this phase/board -- the harness uses this to differentiate the
  strategic framing of otherwise identical-shape decisions.
- Identifies overall game `winner` (`"x"`, `"o"`, `"draw"`, or `None` if
  ongoing).
"""

import json
import re
from typing import Any

import pyspiel

from ... import proxy

# A board row from OpenSpiel's ultimate_tic_tac_toe to_string() is exactly
# three space-separated 3-character subgrid groups of ".xo". OpenSpiel 2.0
# also appends trailing "Current player: N" / "Forced board: M" lines --
# the board-row pattern isolates the grid, and _FORCED_BOARD_RE extracts
# the engine's explicit statement of which local board the mover is
# constrained to (or "any" for a free move).
_BOARD_ROW_RE = re.compile(r"^[.xo]{3} [.xo]{3} [.xo]{3}$")
_FORCED_BOARD_RE = re.compile(r"^Forced board:\s*(\d+|any)\s*$")


def check_subgrid_winner(subgrid: list[str]) -> str:
    """Check if a 3x3 sub-grid has a winner or is a draw."""
    lines = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],  # rows
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],  # cols
        [0, 4, 8],
        [2, 4, 6],  # diagonals
    ]
    for line in lines:
        if subgrid[line[0]] != "" and subgrid[line[0]] == subgrid[line[1]] == subgrid[line[2]]:
            return subgrid[line[0]]
    if all(cell != "" for cell in subgrid):
        return "draw"
    return ""


class UltimateTicTacToeState(proxy.State):
    """Wraps OpenSpiel Ultimate Tic Tac Toe state with JSON observations."""

    def _player_string(self, player: int) -> str:
        if player < 0:
            return pyspiel.PlayerId(player).name.lower()
        elif player == 0:
            return "x"
        elif player == 1:
            return "o"
        else:
            raise ValueError(f"Invalid player: {player}")

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        del player
        state_str = self.to_string()
        board = [["" for _ in range(9)] for _ in range(9)]
        # Walk the string once: collect board rows in order and grab the
        # engine-provided "Forced board" line.
        board_rows: list[str] = []
        forced_board_raw: str | None = None
        for raw_line in state_str.strip().splitlines():
            line = raw_line.strip()
            if _BOARD_ROW_RE.match(line):
                board_rows.append(line)
                continue
            m = _FORCED_BOARD_RE.match(line)
            if m:
                forced_board_raw = m.group(1)

        for i, line in enumerate(board_rows):
            parts = line.split(" ")
            major_row = i // 3
            minor_row = i % 3
            for major_col in range(3):
                subgrid_idx = major_row * 3 + major_col
                if major_col >= len(parts):
                    continue
                subgrid_part = parts[major_col]
                for minor_col in range(3):
                    if minor_col >= len(subgrid_part):
                        continue
                    cell_idx = minor_row * 3 + minor_col
                    char = subgrid_part[minor_col]
                    if char == ".":
                        board[subgrid_idx][cell_idx] = ""
                    elif char in ("x", "o"):
                        board[subgrid_idx][cell_idx] = char

        # Calculate sub-grid winners
        subgrid_winners = [check_subgrid_winner(board[s]) for s in range(9)]

        # Determine overall board winner if terminal
        winner = None
        if self.is_terminal():
            returns = self.returns()
            if returns[0] > returns[1]:
                winner = "x"
            elif returns[1] > returns[0]:
                winner = "o"
            else:
                winner = "draw"

        if self.is_terminal():
            active_subgrid, phase = None, None
        elif forced_board_raw == "any":
            active_subgrid, phase = None, "choose_subgrid"
        elif forced_board_raw is not None and forced_board_raw.isdigit():
            active_subgrid, phase = int(forced_board_raw), "choose_cell"
        else:
            raise ValueError(
                "Unexpected ultimate_tic_tac_toe to_string(): missing or "
                f"malformed 'Forced board' line (parsed value: {forced_board_raw!r}). "
                "This proxy depends on OpenSpiel 2.0's to_string() format."
            )

        # board_context explains *why* the mover is in this phase/board.
        # This lets the harness pick a prompt that reflects the strategic
        # framing: is this a free opening choice, a free move because the
        # opponent redirected them to a completed board, a cell choice on
        # a board the mover themselves just selected, or a cell choice on
        # a board the opponent's previous cell dictated?
        #   - "opening"           : start of game, no history, choose_subgrid.
        #   - "redirected"        : choose_subgrid mid-game (opponent's last
        #                           cell mapped to an inactive local board).
        #   - "self_selected"    : choose_cell after the same player's own
        #                           immediately-preceding choose_subgrid.
        #   - "opponent_directed" : choose_cell on a board dictated by the
        #                           opponent's previous cell placement.
        board_context: str | None
        if phase is None:
            board_context = None
        else:
            full_history = self.full_history()
            if not full_history:
                # No prior actions: must be the opening choose_subgrid.
                board_context = "opening"
            else:
                last_player = int(full_history[-1].player)
                current_player = int(self.current_player())
                if phase == "choose_subgrid":
                    # Not the opening (history is non-empty); the last action
                    # was the opponent's cell that mapped to an inactive board.
                    board_context = "redirected"
                elif last_player == current_player:
                    # The previous action was this same player's own
                    # choose_subgrid; now they play the cell within it.
                    board_context = "self_selected"
                else:
                    # The previous action was the opponent's cell placement,
                    # which dictated this board.
                    board_context = "opponent_directed"

        return {
            "board": board,
            "subgrid_winners": subgrid_winners,
            "active_subgrid": active_subgrid,
            "phase": phase,
            "board_context": board_context,
            "current_player": self._player_string(self.current_player()),
            "is_terminal": self.is_terminal(),
            "winner": winner,
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self):
        return self.to_json()


class UltimateTicTacToeGame(proxy.Game):
    """Wraps the OpenSpiel Ultimate Tic Tac Toe game to use proxy state."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("ultimate_tic_tac_toe", params)
        super().__init__(
            wrapped,
            short_name="ultimate_tic_tac_toe_proxy",
            long_name="Ultimate Tic Tac Toe (proxy)",
        )

    def new_initial_state(self, *args) -> UltimateTicTacToeState:
        return UltimateTicTacToeState(self.__wrapped__.new_initial_state(*args), game=self)


# Register the proxy with OpenSpiel
pyspiel.register_game(UltimateTicTacToeGame().get_type(), UltimateTicTacToeGame)
