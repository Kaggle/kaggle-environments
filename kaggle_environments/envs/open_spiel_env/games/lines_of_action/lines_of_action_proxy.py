"""Structured JSON observations for Lines of Action."""

import json
from typing import Any

import pyspiel

from ... import proxy


def _player_string(player: int) -> str:
    if player == 0:
        return "x"
    if player == 1:
        return "o"
    return pyspiel.PlayerId(player).name.lower()


class LinesOfActionState(proxy.State):
    """Lines of Action state proxy.

    Board indexing: ``board[r][c]`` where ``r=0`` is rank 1 (bottom) and
    ``r=7`` is rank 8 (top); ``c=0`` is file ``a`` and ``c=7`` is file ``h``.
    Cells are ``"x"`` (black), ``"o"`` (white), or ``"."`` (empty).
    """

    def _parse_board(self) -> list[list[str]]:
        # Raw observation lines look like:
        #   "8.xxxxxx.", "7o......o", ..., "1.xxxxxx.", " abcdefgh", "", "", "Current player: x"
        # The first 8 lines are ranks 8..1; strip the leading rank label.
        raw = self.__wrapped__.observation_string(0)
        lines = raw.split("\n")
        ranks_top_down = [list(line[1:9]) for line in lines[:8]]
        # Reverse so board[0] is rank 1 (bottom), matching algebraic notation.
        return list(reversed(ranks_top_down))

    def _last_move_str(self) -> str | None:
        history = self.history()
        if not history:
            return None
        last = history[-1]
        # Previous player made the last move; current_player has flipped.
        prev_player = 1 - self.current_player() if not self.is_terminal() else None
        # action_to_string ignores player for this game, so 0 is fine as fallback.
        return self.__wrapped__.action_to_string(prev_player if prev_player is not None else 0, last)

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        del player
        winner: str | None = None
        if self.is_terminal():
            returns = self.returns()
            if returns[0] > returns[1]:
                winner = "x"
            elif returns[1] > returns[0]:
                winner = "o"
            else:
                winner = "draw"
        return {
            "board": self._parse_board(),
            "current_player": _player_string(self.current_player()),
            "is_terminal": self.is_terminal(),
            "winner": winner,
            "move_number": self.move_number(),
            "last_move": self._last_move_str(),
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class LinesOfActionGame(proxy.Game):
    """Lines of Action game proxy."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("lines_of_action", params)
        super().__init__(
            wrapped,
            short_name="lines_of_action_proxy",
            long_name="Lines of Action (proxy)",
        )

    def new_initial_state(self, *args) -> LinesOfActionState:
        return LinesOfActionState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(LinesOfActionGame().get_type(), LinesOfActionGame)
