"""Change Go state and action string representations."""

import json
from typing import Any

import pyspiel

from ... import proxy


class GoState(proxy.State):
    """Go state proxy."""

    def _player_string(self, player: int) -> str:
        if player < 0:
            return pyspiel.PlayerId(player).name.lower()
        elif player == 0:
            return "B"
        elif player == 1:
            return "W"
        else:
            raise ValueError(f"Invalid player: {player}")

    def _board_string_to_dict(self, board_string: str) -> dict:
        lines = board_string.strip().splitlines()
        if len(lines) < 3:
            raise ValueError("Input string is too short to be a valid board.")
        # The last line contains the column labels (e.g., "ABC...")
        column_labels = lines[-1].strip()
        board_rows = lines[2:-1]
        board_size = len(column_labels)
        if len(board_rows) != board_size:
            raise ValueError(f"Board dimension mismatch: {len(column_labels)} columns but {len(board_rows)} rows.")
        grid = []
        symbol_map = {"+": ".", "X": "B", "O": "W"}
        for i, row_line in enumerate(board_rows):
            row_number = board_size - i
            try:
                board_content = row_line.split(maxsplit=1)[1]
            except IndexError:
                raise ValueError(f"Malformed board row: '{row_line}'")
            current_row_list = []
            for j, stone_char in enumerate(board_content):
                col_letter = column_labels[j]
                coordinate = f"{col_letter}{row_number}"
                # TODO
                point_dict = {coordinate: symbol_map.get(stone_char, "?")}
                current_row_list.append(point_dict)
            grid.append(current_row_list)
        return grid

    def state_dict(self) -> dict[str, Any]:
        clone_state = self.get_game().__wrapped__.new_initial_state()
        action_strs = []
        for action in self.history():
            action_strs.append(clone_state.action_to_string(action))
            clone_state.apply_action(action)
        prev_move = None if not action_strs else action_strs[-1]

        return {
            "board_size": self.get_game().get_parameters()["board_size"],
            "komi": self.get_game().get_parameters()["komi"],
            "current_player_to_move": self._player_string(self.current_player()),
            "move_number": len(self.history()) + 1,
            "previous_move_a1": prev_move,
            "board_grid": self._board_string_to_dict(self.__wrapped__.__str__()),
        }

    def to_json(self) -> str:
        return json.dumps(self.state_dict())

    def observation_string(self, player: int) -> str:
        return self.observation_json(player)

    def observation_json(self, player: int) -> str:
        del player
        return self.to_json()

    def __str__(self):
        return self.to_json()


class GoGame(proxy.Game):
    """Go game proxy."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("go", params)
        super().__init__(
            wrapped,
            short_name="go_proxy",
            long_name="Go (proxy)",
        )

    def new_initial_state(self, *args) -> GoState:
        return GoState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(GoGame().get_type(), GoGame)
