"""Coin Game Arena: 2v2 team variant of OpenSpiel's Coin Game.

Two teams of two LLMs each play their own private coin_game board in
parallel. Team A is players 0,1; team B is players 2,3. Each step, one
seat from each team plays simultaneously (turn 0: players 0+2, turn 1:
players 1+3, repeating). Players only ever see their own team's board.

Scoring follows the standard Coin Game rule per player:

    reward = self_pref^2 + other_pref^2 - bad_coins^2

aggregated within a team's board (other_pref counts coins of the OTHER
teammate's preference collected by anyone on that board, bad_coins are
unowned colours collected by anyone on that board). The team total is
the sum of its two players' rewards. Higher team total wins; equal
totals = draw.

Dynamics are SIMULTANEOUS so the kaggle harness can request both teams'
moves in parallel each step. Setup (preferences, player and coin
placement) is baked deterministically into the initial state via the
``seed`` parameter so paired AA-vs-BB matches can use identical boards.
"""

from __future__ import annotations

import enum
import json
import random
from typing import Any

import pyspiel

_NUM_PLAYERS = 4
_PLAYERS_PER_TEAM = 2
_NUM_TEAMS = 2
_DEFAULT_ROWS = 8
_DEFAULT_COLS = 8
_DEFAULT_EPISODE_LENGTH = 20
_DEFAULT_NUM_EXTRA_COIN_COLORS = 1
_DEFAULT_NUM_COINS_PER_COLOR = 4
_COIN_RANGE = "abcdefghijklmnopqrstuvwxyz"


class Action(enum.IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAND = 4


_ACTION_NAMES = {
    Action.UP: "up",
    Action.DOWN: "down",
    Action.LEFT: "left",
    Action.RIGHT: "right",
    Action.STAND: "stand",
}

_ACTION_DELTAS = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.STAND: (0, 0),
}


_GAME_TYPE = pyspiel.GameType(
    short_name="coin_game_arena",
    long_name="Coin Game Arena (2v2)",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    parameter_specification={
        "rows": _DEFAULT_ROWS,
        "columns": _DEFAULT_COLS,
        "episode_length": _DEFAULT_EPISODE_LENGTH,
        "num_extra_coin_colors": _DEFAULT_NUM_EXTRA_COIN_COLORS,
        "num_coins_per_color": _DEFAULT_NUM_COINS_PER_COLOR,
        "seed": 0,
    },
)


def _team_of(player_id: int) -> int:
    return player_id // _PLAYERS_PER_TEAM


def _seat_of(player_id: int) -> int:
    return player_id % _PLAYERS_PER_TEAM


def _team_player_ids(team: int) -> list[int]:
    base = team * _PLAYERS_PER_TEAM
    return [base + s for s in range(_PLAYERS_PER_TEAM)]


class CoinGameArenaGame(pyspiel.Game):
    """OpenSpiel game: two parallel coin_game boards, 2 LLMs per team."""

    def __init__(self, params: dict[str, Any] | None = None):
        params = params or {}
        self.rows = int(params.get("rows", _DEFAULT_ROWS))
        self.cols = int(params.get("columns", _DEFAULT_COLS))
        self.episode_length = int(
            params.get("episode_length", _DEFAULT_EPISODE_LENGTH)
        )
        self.num_extra_coin_colors = int(
            params.get("num_extra_coin_colors", _DEFAULT_NUM_EXTRA_COIN_COLORS)
        )
        self.num_coins_per_color = int(
            params.get("num_coins_per_color", _DEFAULT_NUM_COINS_PER_COLOR)
        )
        self.seed = int(params.get("seed", 0))

        self.num_colors_per_board = _PLAYERS_PER_TEAM + self.num_extra_coin_colors
        self.coin_colors = list(_COIN_RANGE[: self.num_colors_per_board])

        max_coins = self.num_colors_per_board * self.num_coins_per_color
        # Loose utility bounds: best case grabs everything good, worst case
        # eats every coin as bad.
        max_utility = float(max_coins * max_coins)
        min_utility = float(-(max_coins * max_coins))

        game_info = pyspiel.GameInfo(
            num_distinct_actions=len(Action),
            max_chance_outcomes=0,
            num_players=_NUM_PLAYERS,
            min_utility=min_utility,
            max_utility=max_utility,
            utility_sum=0.0,
            max_game_length=self.episode_length,
        )
        super().__init__(_GAME_TYPE, game_info, params)

    def new_initial_state(self):
        return CoinGameArenaState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return CoinGameArenaObserver()


class CoinGameArenaState(pyspiel.State):
    """State for Coin Game Arena.

    Internally holds two ``Board`` dicts (one per team) plus a shared
    move counter. ``current_player()`` always returns SIMULTANEOUS; the
    framework reads per-player ``legal_actions(pid)`` and only the seat
    whose turn it is on each board returns a non-empty list.
    """

    def __init__(self, game: CoinGameArenaGame):
        super().__init__(game)
        self._game = game
        self._move_number = 0
        self._is_terminal = False
        # Per-board move history: list of (seat_index, action_name).
        self._move_history: list[list[tuple[int, str]]] = [[] for _ in range(_NUM_TEAMS)]
        # Per-player preferences (4-tuple). Seeded deterministically below.
        self._preferences: list[str] = ["?"] * _NUM_PLAYERS
        # Per-board state (dict per board). Initialized in _setup.
        self._boards: list[dict[str, Any]] = []
        self._setup()

    # --- Setup -------------------------------------------------------------

    def _setup(self) -> None:
        rng = random.Random(self._game.seed)
        rows = self._game.rows
        cols = self._game.cols
        colors = self._game.coin_colors
        coins_per_color = self._game.num_coins_per_color

        # Assign per-player preferences. Each board's two players get
        # distinct preferences drawn from the first len(team) colors of
        # the universe (matching upstream coin_game conventions: player i
        # prefers a unique colour, extras are bad coins).
        for team in range(_NUM_TEAMS):
            shuffled = list(colors[:_PLAYERS_PER_TEAM])
            rng.shuffle(shuffled)
            for seat, pid in enumerate(_team_player_ids(team)):
                self._preferences[pid] = shuffled[seat]

        # Build each board: place 2 players + coins on disjoint cells.
        for team in range(_NUM_TEAMS):
            cells = [(r, c) for r in range(rows) for c in range(cols)]
            rng.shuffle(cells)
            player_positions = [cells.pop() for _ in range(_PLAYERS_PER_TEAM)]
            coins: dict[tuple[int, int], str] = {}
            for color in colors:
                for _ in range(coins_per_color):
                    if not cells:
                        break
                    pos = cells.pop()
                    coins[pos] = color
            collected: dict[int, dict[str, int]] = {
                pid: {color: 0 for color in colors}
                for pid in _team_player_ids(team)
            }
            self._boards.append(
                {
                    "player_positions": player_positions,  # indexed by seat (0,1)
                    "coins": coins,                        # (r,c) -> color
                    "collected": collected,                # pid -> {color: count}
                }
            )

    # --- OpenSpiel core ----------------------------------------------------

    def current_player(self):
        if self._is_terminal:
            return pyspiel.PlayerId.TERMINAL
        return pyspiel.PlayerId.SIMULTANEOUS

    def _active_seat(self) -> int:
        """Which seat (0 or 1) plays on each board this step."""
        return self._move_number % _PLAYERS_PER_TEAM

    def _legal_actions(self, player: int):
        if self._is_terminal:
            return []
        if _seat_of(player) != self._active_seat():
            return []
        return [a.value for a in Action]

    def _apply_actions(self, actions):
        """Apply one move per board (the active seat's action).

        ``actions`` is a 4-list. Inactive seats supplied INVALID_ACTION
        (they have empty legal_actions); we ignore them.
        """
        if self._is_terminal:
            return
        active_seat = self._active_seat()
        for team in range(_NUM_TEAMS):
            pid = _team_player_ids(team)[active_seat]
            action_value = actions[pid]
            if action_value == pyspiel.INVALID_ACTION:
                # Should not happen when this seat is active, but guard.
                continue
            action = Action(action_value)
            self._step_player(team, active_seat, action)
            self._move_history[team].append((active_seat, _ACTION_NAMES[action]))

        self._move_number += 1
        if self._move_number >= self._game.episode_length:
            self._is_terminal = True

    def _step_player(self, team: int, seat: int, action: Action) -> None:
        board = self._boards[team]
        rows = self._game.rows
        cols = self._game.cols
        r, c = board["player_positions"][seat]
        dr, dc = _ACTION_DELTAS[action]
        nr, nc = r + dr, c + dc
        # Out-of-bounds moves are a no-op (player stays put), matching the
        # spirit of upstream coin_game where boundary moves are blocked.
        if not (0 <= nr < rows and 0 <= nc < cols):
            nr, nc = r, c
        # Collisions with the teammate's cell are also a no-op so two
        # players can't occupy the same square.
        teammate_seat = 1 - seat
        if board["player_positions"][teammate_seat] == (nr, nc):
            nr, nc = r, c
        board["player_positions"][seat] = (nr, nc)
        # Collect any coin on the new cell.
        coin = board["coins"].pop((nr, nc), None)
        if coin is not None:
            pid = _team_player_ids(team)[seat]
            board["collected"][pid][coin] += 1

    def _action_to_string(self, player: int, action: int) -> str:
        return _ACTION_NAMES[Action(action)]

    def is_terminal(self) -> bool:
        return self._is_terminal

    def returns(self) -> list[float]:
        rewards = [0.0] * _NUM_PLAYERS
        if not self._is_terminal:
            return rewards
        for team in range(_NUM_TEAMS):
            board = self._boards[team]
            pids = _team_player_ids(team)
            # Aggregate coin counts on this board.
            total_collected: dict[str, int] = {color: 0 for color in self._game.coin_colors}
            for pid in pids:
                for color, count in board["collected"][pid].items():
                    total_collected[color] += count
            team_prefs = {self._preferences[pid] for pid in pids}
            # Coins of unowned colours collected by anyone on this board.
            bad_coins = sum(
                count for color, count in total_collected.items()
                if color not in team_prefs
            )
            for pid in pids:
                self_pref = self._preferences[pid]
                self_pref_count = sum(
                    board["collected"][p].get(self_pref, 0) for p in pids
                )
                other_pids = [p for p in pids if p != pid]
                other_pref_count = 0
                for op in other_pids:
                    op_pref = self._preferences[op]
                    other_pref_count += sum(
                        board["collected"][p].get(op_pref, 0) for p in pids
                    )
                rewards[pid] = float(
                    self_pref_count ** 2
                    + other_pref_count ** 2
                    - bad_coins ** 2
                )
        return rewards

    # --- Rendering & observations -----------------------------------------

    def _board_to_grid(self, team: int) -> list[list[str]]:
        rows = self._game.rows
        cols = self._game.cols
        grid = [["." for _ in range(cols)] for _ in range(rows)]
        board = self._boards[team]
        for (r, c), color in board["coins"].items():
            grid[r][c] = color
        for seat, (r, c) in enumerate(board["player_positions"]):
            pid = _team_player_ids(team)[seat]
            grid[r][c] = str(pid)
        return grid

    def _board_view(self, team: int, viewer_pid: int | None) -> dict[str, Any]:
        board = self._boards[team]
        pids = _team_player_ids(team)
        view: dict[str, Any] = {
            "team_id": team,
            "board": self._board_to_grid(team),
            "num_rows": self._game.rows,
            "num_columns": self._game.cols,
            "coin_colors": list(self._game.coin_colors),
            "player_positions": {
                pid: list(board["player_positions"][seat])
                for seat, pid in enumerate(pids)
            },
            "coins_collected": {
                pid: dict(board["collected"][pid]) for pid in pids
            },
            "move_history": [
                {"seat": seat, "player_id": pids[seat], "action": name}
                for seat, name in self._move_history[team]
            ],
        }
        if viewer_pid is not None and viewer_pid in pids:
            view["your_preference"] = self._preferences[viewer_pid]
        return view

    def observation_dict(self, player: int | None = None) -> dict[str, Any]:
        episode_length = self._game.episode_length
        active_seat = (
            self._active_seat() if not self._is_terminal else None
        )
        result: dict[str, Any] = {
            "phase": "terminal" if self._is_terminal else "play",
            "move_number": self._move_number,
            "moves_remaining": max(0, episode_length - self._move_number),
            "episode_length": episode_length,
            "active_seat": active_seat,
            "num_teams": _NUM_TEAMS,
            "players_per_team": _PLAYERS_PER_TEAM,
            "is_terminal": self._is_terminal,
        }
        if player is None:
            # Full reveal — used for renderer / debugging.
            result["preferences"] = {
                pid: pref for pid, pref in enumerate(self._preferences)
            }
            result["boards"] = [
                self._board_view(team, None) for team in range(_NUM_TEAMS)
            ]
        else:
            team = _team_of(player)
            result["your_player_id"] = player
            result["your_team_id"] = team
            result["your_seat"] = _seat_of(player)
            result["your_preference"] = self._preferences[player]
            result["board"] = self._board_view(team, player)
            # Convenience flag for the harness.
            result["your_turn"] = (
                not self._is_terminal and _seat_of(player) == active_seat
            )
        if self._is_terminal:
            result["returns"] = self.returns()
            team_totals = []
            for team in range(_NUM_TEAMS):
                team_totals.append(
                    sum(result["returns"][pid] for pid in _team_player_ids(team))
                )
            result["team_totals"] = team_totals
            best = max(team_totals)
            winners = [t for t, total in enumerate(team_totals) if total == best]
            result["winning_team"] = winners[0] if len(winners) == 1 else "draw"
            # Reveal everything at terminal so the renderer/visualizer
            # has the full picture regardless of viewer.
            result["preferences"] = {
                pid: pref for pid, pref in enumerate(self._preferences)
            }
            result["boards"] = [
                self._board_view(team, None) for team in range(_NUM_TEAMS)
            ]
        return result

    def __str__(self) -> str:
        return json.dumps(self.observation_dict(None))


class CoinGameArenaObserver:
    """Per-player JSON observer.

    We expose only the string view (LLM agents read JSON, not tensors).
    OpenSpiel still pokes at ``.tensor`` and ``.dict``, so we provide
    empty placeholders.
    """

    def __init__(self):
        import numpy as np
        self.tensor = np.zeros(0, dtype=np.float32)
        self.dict = {}

    def set_from(self, state: CoinGameArenaState, player: int) -> None:
        # No tensor observation — kept as a no-op for protocol compliance.
        pass

    def string_from(self, state: CoinGameArenaState, player: int) -> str:
        return json.dumps(state.observation_dict(player))


pyspiel.register_game(_GAME_TYPE, CoinGameArenaGame)
