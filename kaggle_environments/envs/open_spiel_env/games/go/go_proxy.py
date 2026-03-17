"""Change Go state and action string representations."""

import json
from typing import Any

import pyspiel

from ... import proxy


class GoState(proxy.State):
    """Go state proxy."""

    def _compute_tromp_taylor_score(self) -> dict:
        """Compute Tromp-Taylor area score from the current board position.

        Counts stones on the board plus empty territory enclosed by a single
        color. All stones are treated as alive (no dead stone removal).
        """
        game_params = self.get_game().get_parameters()
        board_size = game_params["board_size"]
        komi = game_params["komi"]
        handicap = game_params.get("handicap", 0)
        grid = self._parse_board_grid()

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
                # DFS to find connected empty region
                region_size = 0
                reaches_black = False
                reaches_white = False
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    region_size += 1
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

                if reaches_black and not reaches_white:
                    black_territory += region_size
                elif reaches_white and not reaches_black:
                    white_territory += region_size
                else:
                    dame += region_size

        black_score = float(black_stones + black_territory)
        white_score = float(white_stones + white_territory) + komi
        if handicap >= 2:
            white_score += handicap
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

    def _parse_board_grid(self) -> list[list[str]]:
        """Parse the OpenSpiel board string into a 2D grid.

        Returns a list of rows, each a list of 'B', 'W', or '.'.
        """
        board_string = self.__wrapped__.__str__()
        lines = board_string.strip().splitlines()
        board_rows = lines[2:-1]
        symbol_map = {"+": ".", "X": "B", "O": "W"}
        return [
            [symbol_map.get(c, c) for c in row_line.split(maxsplit=1)[1]]
            for row_line in board_rows
        ]

    def _player_string(self, player: int) -> str:
        if player < 0:
            return pyspiel.PlayerId(player).name.lower()
        elif player == 0:
            return "B"
        elif player == 1:
            return "W"
        else:
            raise ValueError(f"Invalid player: {player}")

    def _board_string_to_dict(self) -> list[list[dict]]:
        """Return the board as a grid of {coordinate: stone} dicts."""
        board_string = self.__wrapped__.__str__()
        lines = board_string.strip().splitlines()
        column_labels = lines[-1].strip()
        board_size = len(column_labels)
        grid = self._parse_board_grid()
        stone_names = {"B": "B", "W": "W", ".": "."}
        result = []
        for i, row in enumerate(grid):
            row_number = board_size - i
            result.append([
                {f"{column_labels[j]}{row_number}": stone_names.get(cell, "?")}
                for j, cell in enumerate(row)
            ])
        return result

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
            "board_grid": self._board_string_to_dict(),
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
