"""Structured JSON observations for Markov Soccer.

Markov Soccer (Littman 1994) is a 2-player simultaneous-move grid game.
Players A and B move on a small grid (default 4 rows x 5 cols). A ball
spawns randomly; whoever holds it scores by walking off the opponent's
goal edge. Walking into the ball-holder steals the ball.

Cell encoding from the underlying OpenSpiel observation string:
  'a' / 'A' = player A (uppercase = holds ball)
  'b' / 'B' = player B (uppercase = holds ball)
  'O'       = loose ball on field
  '.'       = empty

Actions: 0=up, 1=down, 2=left, 3=right, 4=stand.
"""

import json
from typing import Any

import pyspiel

from ... import proxy

_ACTION_NAMES = ["up", "down", "left", "right", "stand"]


def _player_label(player: int) -> str:
    if player < 0:
        return pyspiel.PlayerId(player).name.lower()
    if player == 0:
        return "A"
    if player == 1:
        return "B"
    raise ValueError(f"Invalid player: {player}")


class MarkovSoccerState(proxy.State):
    """Markov Soccer state proxy."""

    def _parse_board(self) -> dict[str, Any]:
        raw = self.__wrapped__.observation_string(0)
        # Drop possible "Chance Node" trailer; keep non-empty grid rows.
        rows = [row for row in raw.split("\n") if row and row != "Chance Node"]
        board = [list(row) for row in rows]

        player_a_pos = None
        player_b_pos = None
        ball_pos = None
        ball_owner = None
        for r, row in enumerate(board):
            for c, ch in enumerate(row):
                if ch in ("a", "A"):
                    player_a_pos = [r, c]
                    if ch == "A":
                        ball_pos = [r, c]
                        ball_owner = "A"
                elif ch in ("b", "B"):
                    player_b_pos = [r, c]
                    if ch == "B":
                        ball_pos = [r, c]
                        ball_owner = "B"
                elif ch == "O":
                    ball_pos = [r, c]
        return {
            "board": board,
            "player_a_pos": player_a_pos,
            "player_b_pos": player_b_pos,
            "ball_pos": ball_pos,
            "ball_owner": ball_owner,
        }

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        parsed = self._parse_board()
        winner: str | None = None
        if self.is_terminal():
            returns = self.returns()
            if returns[0] > returns[1]:
                winner = "A"
            elif returns[1] > returns[0]:
                winner = "B"
            else:
                winner = "draw"
        return {
            "board": parsed["board"],
            "current_player": _player_label(self.current_player()),
            "is_terminal": self.is_terminal(),
            "winner": winner,
            "player_a_pos": parsed["player_a_pos"],
            "player_b_pos": parsed["player_b_pos"],
            "ball_pos": parsed["ball_pos"],
            "ball_owner": parsed["ball_owner"],
            "actions": _ACTION_NAMES,
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class MarkovSoccerGame(proxy.Game):
    """Markov Soccer game proxy."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("markov_soccer", params)
        super().__init__(
            wrapped,
            short_name="markov_soccer_proxy",
            long_name="Markov Soccer (proxy)",
        )

    def new_initial_state(self, *args) -> MarkovSoccerState:
        return MarkovSoccerState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(MarkovSoccerGame().get_type(), MarkovSoccerGame)
