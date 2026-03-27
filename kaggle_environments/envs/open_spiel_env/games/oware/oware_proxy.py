"""Structured JSON observations for Oware."""

import json
from typing import Any

import pyspiel

from ... import proxy

_PIT_NAMES_P0 = ["A", "B", "C", "D", "E", "F"]
_PIT_NAMES_P1 = ["a", "b", "c", "d", "e", "f"]


class OwareState(proxy.State):
    """Wraps OpenSpiel Oware state with JSON observations."""

    def _parse_observation(self) -> dict[str, Any]:
        """Parse the pipe-separated observation string.

        Format: "<current_player> | <score0> <score1> | <pit0> ... <pit11>"
        Pits 0-5 are player 0's (A-F), pits 6-11 are player 1's (a-f).
        """
        raw = self.__wrapped__.observation_string(0)
        parts = raw.split(" | ")
        scores = list(map(int, parts[1].split()))
        pits = list(map(int, parts[2].split()))
        return {
            "pits": [pits[:6], pits[6:]],
            "scores": scores,
        }

    def _last_action_info(self) -> dict[str, Any] | None:
        history = self.history()
        if not history:
            return None
        action = history[-1]
        # Previous player is the one who just moved
        prev_player = 1 - max(0, self.current_player()) if not self.is_terminal() else (
            0 if len(history) % 2 == 1 else 1
        )
        pit_names = _PIT_NAMES_P0 if prev_player == 0 else _PIT_NAMES_P1
        return {
            "player": prev_player,
            "pit": action,
            "pit_name": pit_names[action],
        }

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        obs = self._parse_observation()
        winner = None
        if self.is_terminal():
            returns = self.returns()
            if returns[0] > returns[1]:
                winner = 0
            elif returns[1] > returns[0]:
                winner = 1
            else:
                winner = "draw"
        return {
            "board": obs["pits"],
            "scores": obs["scores"],
            "current_player": self.current_player(),
            "is_terminal": self.is_terminal(),
            "winner": winner,
            "last_action": self._last_action_info(),
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self):
        return self.to_json()


class OwareGame(proxy.Game):
    """Wraps the OpenSpiel Oware game to use the proxy state."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("oware", params)
        super().__init__(
            wrapped,
            short_name="oware_proxy",
            long_name="Oware (proxy)",
        )

    def new_initial_state(self, *args) -> OwareState:
        return OwareState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(OwareGame().get_type(), OwareGame)
