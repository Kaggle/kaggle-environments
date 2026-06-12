"""Ant Foraging Arena: 2v2 team variant of OpenSpiel's Python Ant Foraging.

Two teams of two LLMs each play their own private ant_foraging board in
parallel. Team A is players 0,1; team B is players 2,3. Players take
turns one-at-a-time in the order [0, 2, 1, 3] repeating, so the two
teams interleave while each team's seats alternate. Players only ever
see their own team's board.

Ant Foraging is cooperative within a team: each team's score is the
number of food items its ants deliver to the nest. Both teammates
receive the team's food count as their reward. Higher team total wins
the head-to-head; equal totals = draw.

Dynamics are SEQUENTIAL — exactly one player is active per step. The
game terminates when either board has delivered all food (that team
locks in its food race win) or after ``max_turns`` rounds elapse, where
a "round" is one move per ant per team (``num_ants_per_team *
NUM_TEAMS`` interleaved steps). Food-completion terminations are only
applied at boundaries where every team has had a matching number of
moves, so the opposing team always gets its corresponding move before
the game can end.

Both teams' boards are seeded identically from the ``seed`` parameter:
same nest, food, and ant start positions. This makes any AA-vs-BB
matchup a fair head-to-head on the same puzzle.
"""

from __future__ import annotations

import enum
import json
import random
from typing import Any

import numpy as np
import pyspiel

_NUM_TEAMS = 2
_DEFAULT_GRID_SIZE = 8
_DEFAULT_NUM_ANTS_PER_TEAM = 2
_DEFAULT_NUM_FOOD = 3
_DEFAULT_MAX_TURNS = 50
_DEFAULT_PHEROMONE_DECAY = 0.9


class Action(enum.IntEnum):
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


_ACTION_NAMES = {
    Action.STAY: "stay",
    Action.UP: "up",
    Action.DOWN: "down",
    Action.LEFT: "left",
    Action.RIGHT: "right",
}

_ACTION_DELTAS = {
    Action.STAY: (0, 0),
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


_GAME_TYPE = pyspiel.GameType(
    short_name="ant_foraging_arena",
    long_name="Ant Foraging Arena (2v2)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_TEAMS * _DEFAULT_NUM_ANTS_PER_TEAM,
    min_num_players=_NUM_TEAMS * _DEFAULT_NUM_ANTS_PER_TEAM,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    parameter_specification={
        "grid_size": _DEFAULT_GRID_SIZE,
        "num_ants_per_team": _DEFAULT_NUM_ANTS_PER_TEAM,
        "num_food": _DEFAULT_NUM_FOOD,
        "max_turns": _DEFAULT_MAX_TURNS,
        "pheromone_decay": _DEFAULT_PHEROMONE_DECAY,
        "seed": 0,
    },
)


def _team_of(player_id: int, players_per_team: int) -> int:
    return player_id // players_per_team


def _seat_of(player_id: int, players_per_team: int) -> int:
    return player_id % players_per_team


def _team_player_ids(team: int, players_per_team: int) -> list[int]:
    base = team * players_per_team
    return [base + s for s in range(players_per_team)]


class AntForagingArenaGame(pyspiel.Game):
    """OpenSpiel game: two parallel ant_foraging boards, 2 LLMs per team."""

    def __init__(self, params: dict[str, Any] | None = None):
        params = params or {}
        self.grid_size = int(params.get("grid_size", _DEFAULT_GRID_SIZE))
        self.num_ants_per_team = int(params.get("num_ants_per_team", _DEFAULT_NUM_ANTS_PER_TEAM))
        self.num_food = int(params.get("num_food", _DEFAULT_NUM_FOOD))
        self.max_turns = int(params.get("max_turns", _DEFAULT_MAX_TURNS))
        self.pheromone_decay = float(params.get("pheromone_decay", _DEFAULT_PHEROMONE_DECAY))
        self.seed = int(params.get("seed", 0))

        self.num_players_total = _NUM_TEAMS * self.num_ants_per_team
        # Total interleaved steps over both boards.
        self.total_moves = self.max_turns * self.num_ants_per_team * _NUM_TEAMS

        game_info = pyspiel.GameInfo(
            num_distinct_actions=len(Action),
            max_chance_outcomes=0,
            num_players=self.num_players_total,
            min_utility=0.0,
            max_utility=float(self.num_food),
            utility_sum=None,
            max_game_length=self.total_moves,
        )
        super().__init__(_GAME_TYPE, game_info, params)

    def new_initial_state(self):
        return AntForagingArenaState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return AntForagingArenaObserver()


class AntForagingArenaState(pyspiel.State):
    """State for Ant Foraging Arena.

    Holds two ``Board`` dicts (one per team) plus a shared move counter.
    Sequential play: players act in the order [0, 2, 1, 3] repeating,
    so the two teams interleave while each team's seats alternate.
    ``current_player()`` returns the active player id (or TERMINAL).
    """

    def __init__(self, game: AntForagingArenaGame):
        super().__init__(game)
        self._game = game
        self._players_per_team = game.num_ants_per_team
        self._num_players = game.num_players_total
        self._move_number = 0
        self._is_terminal = False
        # Per-board move history: list of (seat_index, action_name).
        self._move_history: list[list[tuple[int, str]]] = [[] for _ in range(_NUM_TEAMS)]
        # Per-board state (dict per board). Initialized in _setup.
        self._boards: list[dict[str, Any]] = []
        self._setup()

    # --- Setup -------------------------------------------------------------

    def _setup(self) -> None:
        # Both teams play on identical boards (same nest, food, ant
        # placement) so AA-vs-BB matches are fair comparisons. We achieve
        # this by re-seeding the RNG with the same seed for each team.
        grid_size = self._game.grid_size
        num_food = self._game.num_food
        ants_per_team = self._players_per_team

        nest_pos = (grid_size // 2, grid_size // 2)

        for _team in range(_NUM_TEAMS):
            rng = random.Random(self._game.seed)

            # Pick `num_food` food cells from the non-nest interior. We use
            # the same "1..grid_size-1" interior range as upstream
            # ant_foraging for parity, then fall back to the full grid if
            # we can't place enough food.
            interior_cells = [
                (r, c) for r in range(1, grid_size - 1) for c in range(1, grid_size - 1) if (r, c) != nest_pos
            ]
            rng.shuffle(interior_cells)
            food_positions: list[tuple[int, int]] = []
            for cell in interior_cells:
                if len(food_positions) == num_food:
                    break
                food_positions.append(cell)
            if len(food_positions) < num_food:
                # Pull from anywhere on the board if interior is too small.
                all_cells = [
                    (r, c)
                    for r in range(grid_size)
                    for c in range(grid_size)
                    if (r, c) != nest_pos and (r, c) not in food_positions
                ]
                rng.shuffle(all_cells)
                while len(food_positions) < num_food and all_cells:
                    food_positions.append(all_cells.pop())

            grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]
            grid[nest_pos[0]][nest_pos[1]] = "N"
            for r, c in food_positions:
                grid[r][c] = "F"

            board = {
                "nest_pos": nest_pos,
                "food_positions": list(food_positions),
                "ant_positions": [nest_pos for _ in range(ants_per_team)],
                "carrying_food": [False for _ in range(ants_per_team)],
                "pheromone_to_food": np.zeros((grid_size, grid_size), dtype=np.float32),
                "pheromone_to_nest": np.zeros((grid_size, grid_size), dtype=np.float32),
                "food_collected": 0,
                "grid": grid,
            }
            self._boards.append(board)

    # --- OpenSpiel core ----------------------------------------------------

    def _current_player_id(self) -> int:
        """Player id whose turn it is. Interleaves teams: [0, 2, 1, 3, ...]."""
        team = self._move_number % _NUM_TEAMS
        seat = (self._move_number // _NUM_TEAMS) % self._players_per_team
        return team * self._players_per_team + seat

    def current_player(self):
        if self._is_terminal:
            return pyspiel.PlayerId.TERMINAL
        return self._current_player_id()

    def _legal_actions(self, player: int):
        if self._is_terminal:
            return []
        if player != self._current_player_id():
            return []
        # Mirror upstream ant_foraging: STAY is always legal; directions
        # are only legal when they keep the ant on the board.
        seat = _seat_of(player, self._players_per_team)
        team = _team_of(player, self._players_per_team)
        board = self._boards[team]
        r, c = board["ant_positions"][seat]
        # Read grid_size from the board itself so this also works on
        # states reconstructed via pyspiel.deserialize_game_and_state
        # (where ``self._game`` may not retain Python attributes).
        grid_size = len(board["grid"])
        actions = [Action.STAY.value]
        for action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
            dr, dc = _ACTION_DELTAS[action]
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                actions.append(action.value)
        return sorted(actions)

    def _apply_action(self, action_value):
        if self._is_terminal:
            return
        pid = self._current_player_id()
        team = _team_of(pid, self._players_per_team)
        seat = _seat_of(pid, self._players_per_team)
        action = Action(action_value)
        self._step_player(team, seat, action)
        self._move_history[team].append((seat, _ACTION_NAMES[action]))

        self._move_number += 1

        # Decay pheromones after each full round across both boards.
        round_size = self._players_per_team * _NUM_TEAMS
        if self._move_number % round_size == 0:
            for board in self._boards:
                board["pheromone_to_food"] *= self._game.pheromone_decay
                board["pheromone_to_nest"] *= self._game.pheromone_decay

        # Terminal: any board has delivered all food, OR we hit the cap.
        # Food-completion can only end the game once every team has had a
        # matching number of moves — i.e. on boundaries where each team
        # has played the same number of times. Otherwise team A finishing
        # mid-step would deny team B its corresponding move.
        if self._move_number >= self._game.total_moves:
            self._is_terminal = True
        elif (
            self._move_number % _NUM_TEAMS == 0
            and any(b["food_collected"] >= self._game.num_food for b in self._boards)
        ):
            self._is_terminal = True

    def _step_player(self, team: int, seat: int, action: Action) -> None:
        board = self._boards[team]
        grid_size = self._game.grid_size
        r, c = board["ant_positions"][seat]
        dr, dc = _ACTION_DELTAS[action]
        nr, nc = r + dr, c + dc
        # _legal_actions filters off-board moves, so this is defensive
        # only (e.g. callers that bypass legal_actions).
        if not (0 <= nr < grid_size and 0 <= nc < grid_size):
            nr, nc = r, c
        board["ant_positions"][seat] = (nr, nc)

        # Food pickup: only when not already carrying.
        if not board["carrying_food"][seat] and (nr, nc) in board["food_positions"]:
            board["carrying_food"][seat] = True
            board["food_positions"].remove((nr, nc))
            board["grid"][nr][nc] = "."
            board["pheromone_to_food"][nr, nc] = 1.0

        # Food delivery: carrying ant returns to nest.
        if board["carrying_food"][seat] and (nr, nc) == board["nest_pos"]:
            board["carrying_food"][seat] = False
            board["food_collected"] += 1
            board["pheromone_to_nest"][nr, nc] = 1.0

        # Lay pheromone based on current carry state.
        if board["carrying_food"][seat]:
            board["pheromone_to_nest"][nr, nc] = min(1.0, float(board["pheromone_to_nest"][nr, nc]) + 0.3)
        elif board["pheromone_to_food"][nr, nc] > 0:
            board["pheromone_to_food"][nr, nc] = min(1.0, float(board["pheromone_to_food"][nr, nc]) + 0.1)

    def _action_to_string(self, player: int, action: int) -> str:
        return _ACTION_NAMES[Action(action)]

    def is_terminal(self) -> bool:
        return self._is_terminal

    def returns(self) -> list[float]:
        rewards = [0.0] * self._num_players
        if not self._is_terminal:
            return rewards
        for team in range(_NUM_TEAMS):
            team_score = float(self._boards[team]["food_collected"])
            for pid in _team_player_ids(team, self._players_per_team):
                rewards[pid] = team_score
        return rewards

    # --- Rendering & observations -----------------------------------------

    def _pheromone_view(self, arr: np.ndarray) -> list[list[float]]:
        return [[round(float(v), 4) for v in row] for row in arr]

    def _board_view(self, team: int) -> dict[str, Any]:
        board = self._boards[team]
        pids = _team_player_ids(team, self._players_per_team)
        return {
            "team_id": team,
            "grid": [row[:] for row in board["grid"]],
            "grid_size": self._game.grid_size,
            "num_ants": self._players_per_team,
            "num_food": self._game.num_food,
            "nest_position": list(board["nest_pos"]),
            "food_positions": [list(p) for p in board["food_positions"]],
            "ant_positions": {pid: list(board["ant_positions"][seat]) for seat, pid in enumerate(pids)},
            "carrying_food": {pid: bool(board["carrying_food"][seat]) for seat, pid in enumerate(pids)},
            "pheromone_to_food": self._pheromone_view(board["pheromone_to_food"]),
            "pheromone_to_nest": self._pheromone_view(board["pheromone_to_nest"]),
            "food_collected": int(board["food_collected"]),
            "move_history": [
                {"seat": seat, "player_id": pids[seat], "action": name} for seat, name in self._move_history[team]
            ],
        }

    def observation_dict(self, player: int | None = None) -> dict[str, Any]:
        max_turns = self._game.max_turns
        total_moves = self._game.total_moves
        if self._is_terminal:
            active_player_id = None
            active_team_id = None
            active_seat = None
        else:
            active_player_id = self._current_player_id()
            active_team_id = _team_of(active_player_id, self._players_per_team)
            active_seat = _seat_of(active_player_id, self._players_per_team)
        result: dict[str, Any] = {
            "phase": "terminal" if self._is_terminal else "play",
            "move_number": self._move_number,
            "moves_remaining": max(0, total_moves - self._move_number),
            "max_turns": max_turns,
            "active_player_id": active_player_id,
            "active_team_id": active_team_id,
            "active_seat": active_seat,
            "num_teams": _NUM_TEAMS,
            "players_per_team": self._players_per_team,
            "is_terminal": self._is_terminal,
        }
        if player is None:
            # Full reveal — used for renderer / debugging.
            result["boards"] = [self._board_view(team) for team in range(_NUM_TEAMS)]
        else:
            team = _team_of(player, self._players_per_team)
            result["your_player_id"] = player
            result["your_team_id"] = team
            result["your_seat"] = _seat_of(player, self._players_per_team)
            result["board"] = self._board_view(team)
            result["your_turn"] = not self._is_terminal and player == active_player_id
        if self._is_terminal:
            result["returns"] = self.returns()
            team_totals = [int(self._boards[team]["food_collected"]) for team in range(_NUM_TEAMS)]
            result["team_totals"] = team_totals
            best = max(team_totals)
            winners = [t for t, total in enumerate(team_totals) if total == best]
            result["winning_team"] = winners[0] if len(winners) == 1 else "draw"
            # Reveal everything at terminal regardless of viewer.
            result["boards"] = [self._board_view(team) for team in range(_NUM_TEAMS)]
        return result

    def __str__(self) -> str:
        return json.dumps(self.observation_dict(None))


class AntForagingArenaObserver:
    """Per-player JSON observer.

    We expose only the string view (LLM agents read JSON, not tensors).
    OpenSpiel still pokes at ``.tensor`` and ``.dict``, so we provide
    empty placeholders.
    """

    def __init__(self):
        self.tensor = np.zeros(0, dtype=np.float32)
        self.dict = {}

    def set_from(self, state: AntForagingArenaState, player: int) -> None:
        pass

    def string_from(self, state: AntForagingArenaState, player: int) -> str:
        return json.dumps(state.observation_dict(player))


pyspiel.register_game(_GAME_TYPE, AntForagingArenaGame)
