"""Structured JSON observations for Quoridor.

Quoridor is a 2- or 4-player abstract game on a square board (default 9x9).
Each turn a player either moves their pawn one square (orthogonally, with
jump rules) or places a 2-cell wall between rows or columns to obstruct
opponents. The first player to reach the opposite side of the board wins.

OpenSpiel encodes the board as a ``diameter x diameter`` grid where
``diameter = 2 * board_size - 1``. Even/even coordinates are cells, odd/even
are vertical walls, even/odd are horizontal walls, and odd/odd are wall
corners. Pawn-move actions are encoded relative to the current player's
location; wall-placement actions are absolute. Move strings use algebraic
notation: ``e5`` for a pawn at column e, row 5; ``a1v`` for a vertical wall
with its top half at a1; ``a1h`` for a horizontal wall just below row 1
covering columns a and b.
"""

import json
from typing import Any, Mapping

import pyspiel

from ... import proxy

_PAWN_CHARS = {"0": 0, "@": 1, "#": 2, "%": 3}
_PLAYER_LABELS = ("x", "o", "n", "s")
# OpenSpiel's Quoridor caps board_size at 25 (column letters a..y).
_MAX_BOARD_SIZE = 25


def _player_string(player: int, num_players: int) -> str:
    if player < 0:
        return pyspiel.PlayerId(player).name.lower()
    if 0 <= player < num_players:
        return _PLAYER_LABELS[player]
    raise ValueError(f"Invalid player: {player}")


def _coord_to_label(col: int, row: int) -> str:
    return f"{chr(ord('a') + col)}{row + 1}"


def _default_wall_count(params: Mapping[str, Any]) -> int:
    """Quoridor's per-player wall budget.

    OpenSpiel's parameter_specification claims the default is 0 but the C++
    constructor falls back to board_size^2 / 8 only when wall_count is absent
    from the param dict -- not when it's explicitly 0. Both the proxy's
    ``load_game`` path and the wall-counter need the formula.
    """
    raw = int(params.get("wall_count", 0))
    if raw > 0:
        return raw
    board_size = int(params.get("board_size", 9))
    return (board_size * board_size) // 8


class QuoridorState(proxy.State):
    """Quoridor state proxy returning structured JSON observations."""

    def _params(self) -> dict[str, Any]:
        return self.get_game().get_parameters()

    def _board_size(self) -> int:
        return int(self._params().get("board_size", 9))

    def _num_players(self) -> int:
        return int(self._params().get("players", 2))

    def _parse_observation(self) -> dict[str, Any]:
        """Parse the ASCII observation into structured pawn/wall data.

        Quoridor renders each cell as 3 characters (`` . ``, `` @ ``, ...)
        and each vertical wall as a single ``|`` or space, with 2-character
        leading row labels. The wall rows between cells use ``---`` for
        horizontal walls and ``+`` for corners.
        """
        obs = self.__wrapped__.observation_string(0)
        board_size = self._board_size()
        lines = obs.split("\n")

        # Column positions in cell rows: cell content at index 3 + 4*c,
        # vertical wall at 5 + 4*c. In wall rows the leading label is replaced
        # with ``  `` (2 chars), so a horizontal wall starts at index 2 + 4*c
        # and spans 3 chars.
        def cell_x(c: int) -> int:
            return 3 + 4 * c

        def wall_x(c: int) -> int:
            return 5 + 4 * c

        def hwall_start(c: int) -> int:
            return 2 + 4 * c

        pawns: dict[str, str] = {}
        cells: list[list[int | None]] = [[None] * board_size for _ in range(board_size)]
        v_walls: list[tuple[int, int]] = []
        h_walls: list[tuple[int, int]] = []
        v_seen: set[tuple[int, int]] = set()
        h_seen: set[tuple[int, int]] = set()

        # Lines 0 (header), 1 (column header), then 2*board_size + 1 more.
        # Cell rows at index 2 + 2*r; wall rows between them at 3 + 2*r.
        for r in range(board_size):
            cell_line = lines[2 + 2 * r]
            for c in range(board_size):
                ch = cell_line[cell_x(c)]
                if ch in _PAWN_CHARS:
                    p = _PAWN_CHARS[ch]
                    cells[r][c] = p
                    pawns[_PLAYER_LABELS[p]] = _coord_to_label(c, r)
            for c in range(board_size - 1):
                if cell_line[wall_x(c)] == "|":
                    if (c, r - 1) not in v_seen:
                        v_walls.append((c, r))
                    v_seen.add((c, r))
            if r < board_size - 1:
                wall_line = lines[3 + 2 * r]
                for c in range(board_size):
                    start = hwall_start(c)
                    if wall_line[start : start + 3] == "---":
                        if (c - 1, r) not in h_seen:
                            h_walls.append((c, r))
                        h_seen.add((c, r))

        # Sanity check: every wall occupies two adjacent cell rows / columns,
        # so each parsed wall must have both halves visible. If only one half
        # showed up the dedup logic above silently undercounts -- raise so a
        # future regression in OpenSpiel's no-overlap rule is loud.
        for c, r in v_walls:
            if (c, r + 1) not in v_seen:
                raise ValueError(f"Vertical wall {_coord_to_label(c, r)}v missing its bottom half in observation")
        for c, r in h_walls:
            if (c + 1, r) not in h_seen:
                raise ValueError(f"Horizontal wall {_coord_to_label(c, r)}h missing its right half in observation")

        return {
            "cells": cells,
            "pawns": pawns,
            "vertical_walls": [_coord_to_label(c, r) + "v" for c, r in v_walls],
            "horizontal_walls": [_coord_to_label(c, r) + "h" for c, r in h_walls],
        }

    def _walls_remaining(self) -> list[int]:
        """Count walls remaining per player by replaying the history.

        OpenSpiel's ``observation_string`` reports buggy wall counts after any
        wall placement. We can't classify a history action by its raw id
        either: pawn-move ids are encoded relative to the mover's location, so
        a step like "down by one" can map to an id whose diameter coordinates
        look like a wall's. Walk the history and let OpenSpiel disambiguate
        for us via ``action_to_string``, where wall placements end in ``v`` /
        ``h``.
        """
        params = self._params()
        num_players = int(params.get("players", 2))
        wall_count = _default_wall_count(params)
        placed = [0] * num_players
        state = self.get_game().__wrapped__.new_initial_state()
        for action in self.history():
            player = state.current_player()
            if 0 <= player < num_players:
                label = state.action_to_string(player, action)
                if label.endswith(("v", "h")):
                    placed[player] += 1
            state.apply_action(action)
        return [wall_count - p for p in placed]

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        del player
        num_players = self._num_players()
        parsed = self._parse_observation()
        winner: str | None = None
        if self.is_terminal():
            returns = self.returns()
            max_return = max(returns)
            winners = [i for i, r in enumerate(returns) if r == max_return]
            if len(winners) == 1 and max_return > 0:
                winner = _PLAYER_LABELS[winners[0]]
            else:
                winner = "draw"

        walls_remaining = {_PLAYER_LABELS[i]: w for i, w in enumerate(self._walls_remaining())}
        legal_actions: list[str] = []
        if not self.is_terminal():
            legal_actions = [self.__wrapped__.action_to_string(self.current_player(), a) for a in self.legal_actions()]
        return {
            "board_size": self._board_size(),
            "num_players": num_players,
            "cells": parsed["cells"],
            "pawns": parsed["pawns"],
            "vertical_walls": parsed["vertical_walls"],
            "horizontal_walls": parsed["horizontal_walls"],
            "walls_remaining": walls_remaining,
            "current_player": _player_string(self.current_player(), num_players),
            "is_terminal": self.is_terminal(),
            "winner": winner,
            "legal_actions": legal_actions,
            "move_number": self.move_number(),
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class QuoridorGame(proxy.Game):
    """Wraps OpenSpiel's quoridor game to use the proxy state."""

    def __init__(self, params: Any | None = None):
        params = dict(params) if params else {}
        board_size = int(params.get("board_size", 9))
        if board_size > _MAX_BOARD_SIZE:
            # The proxy renders column letters via chr('a' + col); OpenSpiel
            # itself caps the board at 25x25 for the same reason.
            raise ValueError(f"board_size must be <= {_MAX_BOARD_SIZE}, got {board_size}")
        # Backfill the wall_count formula default: parameter_specification
        # resolves to 0, which OpenSpiel takes literally instead of computing
        # board_size^2 / 8. See _default_wall_count for the full story.
        if int(params.get("wall_count", 0)) == 0:
            params["wall_count"] = _default_wall_count(params)
        wrapped = pyspiel.load_game("quoridor", params)
        super().__init__(
            wrapped,
            short_name="quoridor_proxy",
            long_name="Quoridor (proxy)",
        )

    def new_initial_state(self, *args) -> QuoridorState:
        return QuoridorState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(QuoridorGame().get_type(), QuoridorGame)
