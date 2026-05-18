"""Structured JSON observations for Ultimate Tic Tac Toe."""

import json
from typing import Any

import pyspiel

from ... import proxy


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


def get_active_subgrid_and_phase(history: list[int]) -> tuple[int | None, str]:
    """Reconstruct the active sub-grid and turn phase by playing through history.

    Returns:
        (active_subgrid_idx, phase) where phase is 'choose_subgrid' or 'choose_cell'.
    """
    active_subgrid = None
    phase = "choose_subgrid"
    subgrid_winners = ["" for _ in range(9)]
    temp_board = [["" for _ in range(9)] for _ in range(9)]
    current_player = "x"

    i = 0
    while i < len(history):
        if active_subgrid is None:
            if i + 1 >= len(history):
                return history[i], "choose_cell"
            s = history[i]
            c = history[i + 1]
            temp_board[s][c] = current_player
            subgrid_winners[s] = check_subgrid_winner(temp_board[s])
            next_s = c
            if subgrid_winners[next_s] != "":
                active_subgrid = None
                phase = "choose_subgrid"
            else:
                active_subgrid = next_s
                phase = "choose_cell"
            current_player = "o" if current_player == "x" else "x"
            i += 2
        else:
            s = active_subgrid
            c = history[i]
            temp_board[s][c] = current_player
            subgrid_winners[s] = check_subgrid_winner(temp_board[s])
            next_s = c
            if subgrid_winners[next_s] != "":
                active_subgrid = None
                phase = "choose_subgrid"
            else:
                active_subgrid = next_s
                phase = "choose_cell"
            current_player = "o" if current_player == "x" else "x"
            i += 1
    return active_subgrid, phase


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
        lines = state_str.strip().splitlines()

        grid_row_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10]
        for i, line_idx in enumerate(grid_row_indices):
            if line_idx >= len(lines):
                continue
            line = lines[line_idx]
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

        active_subgrid, phase = get_active_subgrid_and_phase(self.history())

        return {
            "board": board,
            "subgrid_winners": subgrid_winners,
            "active_subgrid": active_subgrid,
            "phase": phase,
            "current_player": self._player_string(self.current_player()),
            "is_terminal": self.is_terminal(),
            "winner": winner,
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def observation_json(self, player: int) -> str:
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
