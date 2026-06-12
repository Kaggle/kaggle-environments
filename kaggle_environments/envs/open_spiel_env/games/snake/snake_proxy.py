"""Structured JSON observations for the custom Snake game.

The Snake game (see ``snake_game.py``) is a multi-snake grid game with
simultaneous-move semantics implemented sequentially: each turn cycles
through players 0..N-1, buffering their actions, and on player N-1's
move the buffered actions are applied together. Actions are
``0=UP, 1=DOWN, 2=LEFT, 3=RIGHT``.

This proxy exposes the state as structured JSON so agents and the
visualizer don't have to parse the ASCII board produced by
``SnakeState.__str__``.
"""

import json
from typing import Any

import pyspiel

from ... import proxy

# Ensure the underlying custom game is registered before we try to load it.
from . import snake_game  # noqa: F401
from .snake_game import Action as _Action

_BODY_CHARS = ["a", "b", "c", "d"]
_HEAD_CHARS = ["A", "B", "C", "D"]
_FOOD_CHAR = "*"
_EMPTY_CHAR = "."


class SnakeState(proxy.State):
    """Snake state proxy with JSON observations."""

    def _board(self, snakes: list[list[list[int]]], is_alive: list[bool], foods: list[list[int]]) -> list[list[str]]:
        rows = self.__wrapped__.rows
        cols = self.__wrapped__.cols
        board = [[_EMPTY_CHAR] * cols for _ in range(rows)]
        for fr, fc in foods:
            board[fr][fc] = _FOOD_CHAR
        for i, snake in enumerate(snakes):
            if not is_alive[i] or not snake:
                continue
            body_char = _BODY_CHARS[i % len(_BODY_CHARS)]
            head_char = _HEAD_CHARS[i % len(_HEAD_CHARS)]
            for r, c in snake:
                board[r][c] = body_char
            hr, hc = snake[0]
            board[hr][hc] = head_char
        return board

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        del player  # Snake is perfect-information; same view for everyone.
        wrapped = self.__wrapped__
        snakes = [[[int(r), int(c)] for r, c in snake] for snake in wrapped.snakes]
        is_alive = [bool(a) for a in wrapped.is_alive]
        scores = [float(s) for s in wrapped.scores]
        foods = [[int(r), int(c)] for r, c in wrapped.foods]
        # Back-compat: expose the first food under the old singular key too.
        food = foods[0] if foods else None
        food_respawn_interval = int(getattr(wrapped, "food_respawn_interval", 0))
        turn = int(getattr(wrapped, "_steps", 0))
        if food_respawn_interval > 0:
            turns_until_respawn = food_respawn_interval - (turn % food_respawn_interval)
        else:
            turns_until_respawn = None

        is_terminal = self.is_terminal()
        winner: int | str | None = None
        if is_terminal:
            alive_ids = [i for i, a in enumerate(is_alive) if a]
            if len(alive_ids) == 1:
                winner = alive_ids[0]
            else:
                top_score = max(scores)
                top = [i for i, s in enumerate(scores) if s == top_score]
                winner = top[0] if len(top) == 1 else "draw"

        snake_objs = [
            {"player": i, "body": snakes[i], "alive": is_alive[i], "score": scores[i]}
            for i in range(wrapped.num_players)
        ]

        # While players are buffering moves for the current turn (sequential
        # implementation of simultaneous play), expose who is acting next and
        # which players have already submitted this turn.
        buffer = [a for a in getattr(wrapped, "_move_buffer", [])]
        pending = [i for i, a in enumerate(buffer) if a is not None]

        round_history = [
            [
                _Action(a).name if a is not None and 0 <= a < len(_Action) else None
                for a in round_moves
            ]
            for round_moves in getattr(wrapped, "_round_history", [])
        ]

        return {
            "board": self._board(snakes, is_alive, foods),
            "num_rows": int(wrapped.rows),
            "num_columns": int(wrapped.cols),
            "num_players": int(wrapped.num_players),
            "foods": foods,
            "food": food,  # deprecated, kept for back-compat
            "food_respawn_interval": food_respawn_interval,
            "turns_until_respawn": turns_until_respawn,
            "snakes": snake_objs,
            "scores": scores,
            "is_alive": is_alive,
            "current_player": self.current_player(),
            "pending_this_turn": pending,
            "round_history": round_history,
            "turn": turn,
            "is_terminal": is_terminal,
            "winner": winner,
            "game_over_reason": getattr(wrapped, "_game_over_reason", None),
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def information_state_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class SnakeGame(proxy.Game):
    """Snake game proxy."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("snake", params)
        super().__init__(
            wrapped,
            short_name="snake_proxy",
            long_name="Snake (proxy)",
        )

    def new_initial_state(self, *args) -> SnakeState:
        return SnakeState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(SnakeGame().get_type(), SnakeGame)
