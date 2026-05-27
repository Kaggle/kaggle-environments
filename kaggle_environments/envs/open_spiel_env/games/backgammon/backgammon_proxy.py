"""Structured JSON observations for Backgammon.

OpenSpiel's default backgammon ``ObservationString`` is ASCII art of the board
plus a few labelled lines (``Turn``, ``Dice``, ``Bar``, ``Scores``).  That's
hard for LLM agents to parse reliably, so this proxy converts the state into
a flat JSON dict with the board, dice, bar, off, and current player.

Point indexing follows OpenSpiel's internal numbering (``0..23``).  Player X
moves from low indices toward 23 and bears off past 23; player O moves from
23 toward 0 and bears off below 0.  Initial setup (OpenSpiel coords):

    X: 0(x2), 11(x5), 16(x3), 18(x5)
    O: 23(o2), 12(o5), 7(o3), 5(o5)

"""

import json
import re
from typing import Any

import pyspiel

from ... import proxy

_X = "x"
_O = "o"

# Parses lines like "Dice: 12", "Dice: 1u3", "Dice: ".  Each die is one ASCII
# digit; a trailing 'u' marks a die already used this turn.
_DIE_RE = re.compile(r"(\d)(u?)")


def _player_string(player: int) -> str:
    if player == 0:
        return _X
    if player == 1:
        return _O
    return pyspiel.PlayerId(player).name.lower()


def _parse_dice(observation: str) -> list[dict[str, Any]]:
    for line in observation.splitlines():
        if line.startswith("Dice:"):
            payload = line[len("Dice:") :].strip()
            return [{"value": int(value), "used": used == "u"} for value, used in _DIE_RE.findall(payload)]
    return []


def _parse_bar(observation: str) -> dict[str, int]:
    for line in observation.splitlines():
        if line.startswith("Bar:"):
            payload = line[len("Bar:") :]
            return {_X: payload.count(_X), _O: payload.count(_O)}
    return {_X: 0, _O: 0}


def _parse_scores(observation: str) -> dict[str, int]:
    for line in observation.splitlines():
        if line.startswith("Scores"):
            # e.g. "Scores, X: 3, O: 1"
            x_match = re.search(r"X:\s*(\d+)", line)
            o_match = re.search(r"O:\s*(\d+)", line)
            return {
                _X: int(x_match.group(1)) if x_match else 0,
                _O: int(o_match.group(1)) if o_match else 0,
            }
    return {_X: 0, _O: 0}


class BackgammonState(proxy.State):
    """Backgammon state proxy returning structured JSON observations."""

    def _board(self) -> list[dict[str, Any] | None]:
        out: list[dict[str, Any] | None] = []
        for pos in range(24):
            x_count = self.__wrapped__.board(0, pos)
            o_count = self.__wrapped__.board(1, pos)
            if x_count > 0:
                out.append({"player": _X, "count": x_count})
            elif o_count > 0:
                out.append({"player": _O, "count": o_count})
            else:
                out.append(None)
        return out

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        del player
        observation = self.__wrapped__.observation_string(0)
        board = self._board()
        bar = _parse_bar(observation)
        off = _parse_scores(observation)
        dice = _parse_dice(observation)

        winner: str | None = None
        if self.is_terminal():
            returns = self.returns()
            if returns[0] > returns[1]:
                winner = _X
            elif returns[1] > returns[0]:
                winner = _O
            else:
                winner = "draw"

        return {
            "board": board,
            "bar": bar,
            "off": off,
            "dice": dice,
            "current_player": _player_string(self.current_player()),
            "is_terminal": self.is_terminal(),
            "winner": winner,
            "move_number": self.move_number(),
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class BackgammonGame(proxy.Game):
    """Wraps OpenSpiel's backgammon game to use the proxy state."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("backgammon", params)
        super().__init__(
            wrapped,
            short_name="backgammon_proxy",
            long_name="Backgammon (proxy)",
        )

    def new_initial_state(self, *args) -> BackgammonState:
        return BackgammonState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(BackgammonGame().get_type(), BackgammonGame)
