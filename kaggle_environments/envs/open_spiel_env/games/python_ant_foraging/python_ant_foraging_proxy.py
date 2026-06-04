"""Structured JSON observations for the OpenSpiel Python Ant Foraging game.

Ant Foraging is a cooperative grid game where ants take turns moving on an
NxN grid trying to collect food and return it to the nest. The game is
sequential (one ant moves per step) and all players share the same score
equal to the number of food items delivered to the nest.

Actions: 0=stay, 1=up, 2=down, 3=left, 4=right.

OpenSpiel's default observation is an ASCII board with a legend, which is
hard for agents to parse. This proxy exposes a structured JSON view that
includes the grid, ant positions, carry status, pheromone trails, food
locations, and game progress.
"""

import json
from typing import Any

import pyspiel

from ... import proxy

_CELL_CHARS = {0: ".", 1: "N", 2: "F", 3: "#"}
_ACTION_NAMES = {0: "stay", 1: "up", 2: "down", 3: "left", 4: "right"}


class PythonAntForagingState(proxy.State):
    """Ant Foraging state proxy.

    Grid cells are one-character strings: "." empty, "N" nest, "F" food,
    "#" obstacle. Ant positions and carry status are exposed as separate
    fields rather than overlaid on the grid so agents can distinguish
    cell terrain from dynamic ant state.
    """

    def _grid_chars(self) -> list[list[str]]:
        grid = self.__wrapped__._grid
        size = self.__wrapped__._grid_size
        return [[_CELL_CHARS.get(int(grid[r, c]), "?") for c in range(size)] for r in range(size)]

    def _pheromones(self, attr: str) -> list[list[float]]:
        arr = getattr(self.__wrapped__, attr)
        return [[round(float(v), 4) for v in row] for row in arr]

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        w = self.__wrapped__
        score = int(w._food_collected)
        return {
            "grid": self._grid_chars(),
            "grid_size": int(w._grid_size),
            "num_ants": int(w._num_ants),
            "num_food": int(w._num_food),
            "nest_position": list(w._nest_pos),
            "food_positions": [list(p) for p in w._food_positions],
            "ant_positions": [list(p) for p in w._ant_positions],
            "carrying_food": list(w._carrying_food),
            "pheromone_to_food": self._pheromones("_pheromone_to_food"),
            "pheromone_to_nest": self._pheromones("_pheromone_to_nest"),
            "food_collected": score,
            "score": score,
            "turn": int(w._turn_count),
            "max_turns": int(w._max_turns),
            "current_player": self.current_player(),
            "legal_actions": [int(a) for a in self.legal_actions()],
            "action_names": _ACTION_NAMES,
            "is_terminal": self.is_terminal(),
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class PythonAntForagingGame(proxy.Game):
    """Ant Foraging game proxy."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("python_ant_foraging", params)
        super().__init__(
            wrapped,
            short_name="python_ant_foraging_proxy",
            long_name="Python Ant Foraging (proxy)",
        )

    def new_initial_state(self, *args) -> PythonAntForagingState:
        return PythonAntForagingState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(PythonAntForagingGame().get_type(), PythonAntForagingGame)
