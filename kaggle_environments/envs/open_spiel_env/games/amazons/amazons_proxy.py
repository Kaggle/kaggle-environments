"""Structured JSON observations for Game of the Amazons.

OpenSpiel encodes one Amazons turn as three sub-actions: pick an amazon to
move (from), pick its destination (to), and pick a square to burn with an
arrow (shoot). Each sub-action is an integer ``row * 10 + col`` on the 10x10
board. The proxy exposes the board grid, whose turn it is, and which of the
three sub-actions is expected next so agents do not need to parse the ASCII
observation string.
"""

import json
from typing import Any

import pyspiel

from ... import proxy

_BOARD_SIZE = 10
_PHASES = ("from", "to", "shoot")


class AmazonsState(proxy.State):
    """Amazons state proxy returning structured JSON observations."""

    def _player_string(self, player: int) -> str:
        if player < 0:
            return pyspiel.PlayerId(player).name.lower()
        if player == 0:
            return "x"
        if player == 1:
            return "o"
        raise ValueError(f"Invalid player: {player}")

    def _board(self) -> list[list[str]]:
        rows = self.to_string().strip().split("\n")
        return [list(row) for row in rows]

    def _phase(self) -> str | None:
        if self.is_terminal():
            return None
        return _PHASES[self.move_number() % 3]

    def state_dict(self) -> dict[str, Any]:
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
            "board": self._board(),
            "board_size": _BOARD_SIZE,
            "current_player": self._player_string(self.current_player()),
            "phase": self._phase(),
            "move_number": self.move_number(),
            "is_terminal": self.is_terminal(),
            "winner": winner,
        }

    def to_json(self) -> str:
        return json.dumps(self.state_dict())

    def action_to_dict(self, action: int) -> dict[str, Any]:
        return {"row": action // _BOARD_SIZE, "col": action % _BOARD_SIZE}

    def action_to_json(self, action: int) -> str:
        return json.dumps(self.action_to_dict(action))

    def dict_to_action(self, action_dict: dict[str, Any]) -> int:
        return int(action_dict["row"]) * _BOARD_SIZE + int(action_dict["col"])

    def json_to_action(self, action_json: str) -> int:
        return self.dict_to_action(json.loads(action_json))

    def observation_string(self, player: int) -> str:
        del player
        return self.to_json()

    def __str__(self) -> str:
        return self.to_json()


class AmazonsGame(proxy.Game):
    """Wraps OpenSpiel's amazons game to use the proxy state."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("amazons", params)
        super().__init__(
            wrapped,
            short_name="amazons_proxy",
            long_name="Amazons (proxy)",
        )

    def new_initial_state(self, *args) -> AmazonsState:
        return AmazonsState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(AmazonsGame().get_type(), AmazonsGame)
