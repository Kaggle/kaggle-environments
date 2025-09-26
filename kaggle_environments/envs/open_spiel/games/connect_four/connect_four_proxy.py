"""Change Connect Four state and action string representations."""

import json
from typing import Any

import pyspiel

from ... import proxy


class ConnectFourState(proxy.State):
    """Connect Four state proxy."""

    def _player_string(self, player: int) -> str:
        if player < 0:
            return pyspiel.PlayerId(player).name.lower()
        elif player == 0:
            return "x"
        elif player == 1:
            return "o"
        else:
            raise ValueError(f"Invalid player: {player}")

    def state_dict(self) -> dict[str, Any]:
        # row 0 is now bottom row
        rows = reversed(self.to_string().strip().split("\n"))
        board = [list(row) for row in rows]
        winner = None
        if self.is_terminal():
            if self.returns()[0] > self.returns()[1]:
                winner = "x"
            elif self.returns()[1] > self.returns()[0]:
                winner = "o"
            else:
                winner = "draw"
        return {
            "board": board,
            "current_player": self._player_string(self.current_player()),
            "is_terminal": self.is_terminal(),
            "winner": winner,
        }

    def to_json(self) -> str:
        return json.dumps(self.state_dict())

    def action_to_dict(self, action: int) -> dict[str, Any]:
        return {"col": action}

    def action_to_json(self, action: int) -> str:
        return json.dumps(self.action_to_dict(action))

    def dict_to_action(self, action_dict: dict[str, Any]) -> int:
        return int(action_dict["col"])

    def json_to_action(self, action_json: str) -> int:
        action_dict = json.loads(action_json)
        return self.dict_to_action(action_dict)

    def observation_string(self, player: int) -> str:
        return self.observation_json(player)

    def observation_json(self, player: int) -> str:
        del player
        return self.to_json()

    def __str__(self):
        return self.to_json()


class ConnectFourGame(proxy.Game):
    """Connect Four game proxy."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("connect_four", params)
        super().__init__(
            wrapped,
            short_name="connect_four_proxy",
            long_name="Connect Four (proxy)",
        )

    def new_initial_state(self, *args) -> ConnectFourState:
        return ConnectFourState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(ConnectFourGame().get_type(), ConnectFourGame)
