"""Structured JSON observations for Mancala (Kalah).

OpenSpiel's mancala uses a 14-cell board laid out as::

    [ 13  12  11  10   9   8 ]   <- player 1's pits
      0                       7
    [  1   2   3   4   5   6 ]   <- player 0's pits

where index 0 is player 1's store and index 7 is player 0's store. The
default ``observation_string`` is ASCII art with hyphen separators -- the
proxy parses it into a clean dict so agents can read pit counts, scores,
and turn info directly.
"""

import json
from typing import Any

import pyspiel

from ... import proxy

_PLAYER_0_PITS = [1, 2, 3, 4, 5, 6]
_PLAYER_0_STORE = 7
_PLAYER_1_PITS = [8, 9, 10, 11, 12, 13]
_PLAYER_1_STORE = 0


class MancalaState(proxy.State):
    """Mancala state proxy returning structured JSON observations."""

    def _player_label(self, player: int) -> int | str:
        if player < 0:
            return pyspiel.PlayerId(player).name.lower()
        return player

    def _board(self) -> list[int]:
        tensor = self.__wrapped__.observation_tensor(0)
        return [int(v) for v in tensor]

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        del player
        board = self._board()

        winner: int | str | None = None
        if self.is_terminal():
            returns = self.returns()
            if returns[0] > returns[1]:
                winner = 0
            elif returns[1] > returns[0]:
                winner = 1
            else:
                winner = "draw"

        history = self.history()
        last_action = int(history[-1]) if history else None

        return {
            "board": board,
            "pits": {
                "0": [board[i] for i in _PLAYER_0_PITS],
                "1": [board[i] for i in _PLAYER_1_PITS],
            },
            "stores": {
                "0": board[_PLAYER_0_STORE],
                "1": board[_PLAYER_1_STORE],
            },
            "scores": [board[_PLAYER_0_STORE], board[_PLAYER_1_STORE]],
            "current_player": self._player_label(self.current_player()),
            "move_number": self.move_number(),
            "last_action": last_action,
            "is_terminal": self.is_terminal(),
            "winner": winner,
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class MancalaGame(proxy.Game):
    """Wraps OpenSpiel's mancala game to use the proxy state."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("mancala", params)
        super().__init__(
            wrapped,
            short_name="mancala_proxy",
            long_name="Mancala (proxy)",
        )

    def new_initial_state(self, *args) -> MancalaState:
        return MancalaState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(MancalaGame().get_type(), MancalaGame)
