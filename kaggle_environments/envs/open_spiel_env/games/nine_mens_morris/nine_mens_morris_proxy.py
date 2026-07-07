"""Structured JSON observations for Nine Men's Morris.

Nine Men's Morris is a two-player game on a 24-point board arranged as three
nested squares joined by four midpoint lines. Each player has nine "men" and
tries to form *mills* (three of their pieces in a row along a board line); a
mill lets the player capture one opponent piece.

Play proceeds in three phases:

1. **Placement** -- each player alternately drops one of their nine men on
   an empty point.
2. **Movement** -- once all pieces are placed, players move a piece to an
   adjacent empty point.
3. **Flying** -- when a player has exactly 3 pieces left, they may move any
   piece to any empty point.

A player loses when reduced to fewer than 3 pieces or when they have no legal
move on their turn. The engine also draws the game after ``kMaxNumTurns``.

## Point layout (24 points, indexed 0-23)

The board points, in the order used by OpenSpiel and by this proxy's ``board``
array::

     0-----------1-----------2
     |           |           |
     |   3-------4-------5   |
     |   |       |       |   |
     |   |   6---7---8   |   |
     |   |   |       |   |   |
     9--10--11      12--13--14
     |   |   |       |   |   |
     |   |  15--16--17   |   |
     |   |       |       |   |
     |  18------19------20   |
     |           |           |
    21----------22----------23

## Actions

OpenSpiel encodes actions as integers:

- ``0..23`` -- during placement, drop a piece on that point. During a pending
  capture (immediately after forming a mill), remove the opponent's piece at
  that point.
- ``24 + source*24 + dest`` -- move (phase 2 or 3) from ``source`` to ``dest``.

The proxy overrides ``_action_to_string`` so ``legalActionStrings`` disambiguate
placement, capture, and movement (OpenSpiel's default labels a capture as
``"Point N"``, which agents easily mistake for a placement).
"""

import json
import re
from typing import Any

import pyspiel

from ... import proxy

_NUM_POINTS = 24

# Row/col of each of the 24 board points inside OpenSpiel's 13-line ASCII
# rendering. Mirrors ``kPointStrCoords`` in
# ``open_spiel/games/nine_mens_morris/nine_mens_morris.cc``.
_POINT_COORDS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 7),
    (0, 14),
    (2, 2),
    (2, 7),
    (2, 12),
    (4, 4),
    (4, 7),
    (4, 10),
    (6, 0),
    (6, 2),
    (6, 4),
    (6, 10),
    (6, 12),
    (6, 14),
    (8, 4),
    (8, 7),
    (8, 10),
    (10, 2),
    (10, 7),
    (10, 12),
    (12, 0),
    (12, 7),
    (12, 14),
)

_PLAYER_LABEL = ("W", "B")


def _player_string(player: int) -> str:
    if player < 0:
        return pyspiel.PlayerId(player).name.lower()
    if 0 <= player < len(_PLAYER_LABEL):
        return _PLAYER_LABEL[player]
    raise ValueError(f"Invalid player: {player}")


def _from_move_action(action: int) -> tuple[int, int]:
    idx = action - _NUM_POINTS
    return idx // _NUM_POINTS, idx % _NUM_POINTS


class NineMensMorrisState(proxy.State):
    """Nine Men's Morris state proxy returning structured JSON observations."""

    def _parse_raw(self) -> dict[str, Any]:
        raw = self.__wrapped__.to_string()
        lines = raw.split("\n")
        # First 13 lines are the ASCII board; remainder is labeled metadata.
        board: list[str] = []
        for row, col in _POINT_COORDS:
            ch = lines[row][col] if col < len(lines[row]) else "."
            board.append(ch if ch in ("W", "B", ".") else ".")

        meta_text = "\n".join(lines[13:])

        turn_match = re.search(r"Turn number:\s*(\d+)", meta_text)
        turn_number = int(turn_match.group(1)) if turn_match else 0

        deploy_match = re.search(r"Men to deploy:\s*(\d+)\s+(\d+)", meta_text)
        if deploy_match:
            men_to_deploy = {
                _PLAYER_LABEL[0]: int(deploy_match.group(1)),
                _PLAYER_LABEL[1]: int(deploy_match.group(2)),
            }
        else:
            men_to_deploy = {_PLAYER_LABEL[0]: 0, _PLAYER_LABEL[1]: 0}

        num_match = re.search(r"Num men:\s*(\d+)\s+(\d+)", meta_text)
        if num_match:
            num_men = {
                _PLAYER_LABEL[0]: int(num_match.group(1)),
                _PLAYER_LABEL[1]: int(num_match.group(2)),
            }
        else:
            num_men = {_PLAYER_LABEL[0]: 0, _PLAYER_LABEL[1]: 0}

        capture_pending = "Capture time" in meta_text

        return {
            "board": board,
            "turn_number": turn_number,
            "men_to_deploy": men_to_deploy,
            "num_men": num_men,
            "capture_pending": capture_pending,
        }

    def _phase(self, parsed: dict[str, Any]) -> str:
        if parsed["capture_pending"]:
            return "capture"
        cur = self.current_player()
        if cur < 0:
            return "terminal"
        label = _PLAYER_LABEL[cur]
        if parsed["men_to_deploy"][label] > 0:
            return "placement"
        if parsed["num_men"][label] > 3:
            return "movement"
        return "flying"

    def _last_action_string(self) -> str | None:
        history = self.history()
        if not history:
            return None
        # We use OpenSpiel's context-free label here (e.g. "Point 5") rather
        # than the capture-aware label -- distinguishing capture from
        # placement retrospectively would require replaying history.
        return self.__wrapped__.action_to_string(history[-1])

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        del player
        parsed = self._parse_raw()
        winner: str | None = None
        if self.is_terminal():
            returns = self.returns()
            if returns[0] > returns[1]:
                winner = _PLAYER_LABEL[0]
            elif returns[1] > returns[0]:
                winner = _PLAYER_LABEL[1]
            else:
                winner = "draw"
        return {
            "board": parsed["board"],
            "current_player": _player_string(self.current_player()),
            "phase": self._phase(parsed),
            "men_to_deploy": parsed["men_to_deploy"],
            "num_men": parsed["num_men"],
            "turn_number": parsed["turn_number"],
            "is_terminal": self.is_terminal(),
            "winner": winner,
            "last_action": self._last_action_string(),
        }

    def _action_to_string(self, player: int, action: int) -> str:
        """Context-aware action label used for ``legalActionStrings``.

        OpenSpiel labels a capture (a point-index picked after forming a mill)
        as ``"Point N"`` -- identical to a placement action -- which invites
        agents to confuse the two. We distinguish them here.
        """
        del player
        if action < _NUM_POINTS:
            parsed = self._parse_raw()
            if parsed["capture_pending"]:
                return f"Capture opponent piece at point {action}"
            return f"Place at point {action}"
        src, dst = _from_move_action(action)
        return f"Move point {src} -> point {dst}"

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class NineMensMorrisGame(proxy.Game):
    """Wraps OpenSpiel's nine_mens_morris game to use the proxy state."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("nine_mens_morris", params)
        super().__init__(
            wrapped,
            short_name="nine_mens_morris_proxy",
            long_name="Nine men's morris (proxy)",
        )

    def new_initial_state(self, *args) -> NineMensMorrisState:
        return NineMensMorrisState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(NineMensMorrisGame().get_type(), NineMensMorrisGame)
