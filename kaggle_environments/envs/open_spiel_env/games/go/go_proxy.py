"""Change Go state and action string representations."""

import json
from typing import Any

import pyspiel

from ... import proxy


class GoState(proxy.State):
    """Go state proxy."""

    def _compute_tromp_taylor_score(self) -> dict:
        """Compute Tromp-Taylor area score from the current board position.

        Scores the board as-is with no dead stone removal. This gives correct
        results when agents play until all dead stones are captured before
        passing (e.g. KataGo with friendlyPassOk=false). If agents pass
        prematurely while dead stones remain, the score will be inaccurate.
        """
        board_string = self.__wrapped__.__str__()
        lines = board_string.strip().splitlines()
        column_labels = lines[-1].strip()
        board_rows = lines[2:-1]
        board_size = len(column_labels)

        # Parse board into 2D grid: 'B', 'W', or '.'
        grid = []
        symbol_map = {"+": ".", "X": "B", "O": "W"}
        for row_line in board_rows:
            board_content = row_line.split(maxsplit=1)[1]
            grid.append([symbol_map.get(c, c) for c in board_content])

        # Count stones
        black_stones = sum(row.count("B") for row in grid)
        white_stones = sum(row.count("W") for row in grid)

        # Flood-fill to classify empty regions
        visited = [[False] * board_size for _ in range(board_size)]
        black_territory = 0
        white_territory = 0
        dame = 0

        for r in range(board_size):
            for c in range(board_size):
                if grid[r][c] != "." or visited[r][c]:
                    continue
                # BFS to find connected empty region
                region = []
                reaches_black = False
                reaches_white = False
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    region.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < board_size and 0 <= nc < board_size:
                            if grid[nr][nc] == "B":
                                reaches_black = True
                            elif grid[nr][nc] == "W":
                                reaches_white = True
                            elif not visited[nr][nc]:
                                visited[nr][nc] = True
                                stack.append((nr, nc))

                size = len(region)
                if reaches_black and not reaches_white:
                    black_territory += size
                elif reaches_white and not reaches_black:
                    white_territory += size
                else:
                    dame += size

        komi = self.get_game().get_parameters()["komi"]
        black_score = float(black_stones + black_territory)
        white_score = float(white_stones + white_territory) + komi
        if black_score > white_score:
            winner = "B"
        elif white_score > black_score:
            winner = "W"
        else:
            winner = "draw"

        return {
            "black_stones": black_stones,
            "white_stones": white_stones,
            "black_territory": black_territory,
            "white_territory": white_territory,
            "dame": dame,
            "komi": komi,
            "black_score": black_score,
            "white_score": white_score,
            "winner": winner,
            "winning_margin": abs(black_score - white_score),
            "scoring_method": "tromp-taylor",
        }

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

        result = {
            "board_size": self.get_game().get_parameters()["board_size"],
            "komi": self.get_game().get_parameters()["komi"],
            "current_player_to_move": self._player_string(self.current_player()),
            "move_number": len(self.history()) + 1,
            "previous_move_a1": prev_move,
            "board_grid": self._board_string_to_dict(self.__wrapped__.__str__()),
        }
        if self.is_terminal():
            result["scoring"] = self._compute_tromp_taylor_score()
        return result

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
