"""Structured JSON observations for Dots and Boxes.

OpenSpiel encodes a Dots and Boxes action as an integer that selects either a
horizontal or vertical line on a (num_rows+1) x (num_cols+1) lattice of dots.
The proxy decodes this into ``{"orientation": "h"|"v", "row", "col"}`` and
exposes the full line grid, owned boxes, and per-player scores so agents and
the visualizer never have to parse OpenSpiel's box-drawing ASCII.
"""

import json
from typing import Any

import pyspiel

from ... import proxy

_DEFAULT_NUM_ROWS = 2
_DEFAULT_NUM_COLS = 2


def _player_label(player: int) -> str:
    if player == 0:
        return "1"
    if player == 1:
        return "2"
    return ""


class DotsAndBoxesState(proxy.State):
    """Dots and Boxes state proxy returning structured JSON observations."""

    def _dimensions(self) -> tuple[int, int]:
        params = self.get_game().get_parameters()
        return (
            int(params.get("num_rows", _DEFAULT_NUM_ROWS)),
            int(params.get("num_cols", _DEFAULT_NUM_COLS)),
        )

    def _decode_action(self, action: int, num_rows: int, num_cols: int) -> tuple[str, int, int]:
        max_h = (num_rows + 1) * num_cols
        if action < max_h:
            return "h", action // num_cols, action % num_cols
        action -= max_h
        return "v", action // (num_cols + 1), action % (num_cols + 1)

    def _encode_action(self, orientation: str, row: int, col: int, num_rows: int, num_cols: int) -> int:
        if orientation == "h":
            return row * num_cols + col
        return (num_rows + 1) * num_cols + row * (num_cols + 1) + col

    def _replay(self, num_rows: int, num_cols: int) -> dict[str, Any]:
        h_lines = [[0] * num_cols for _ in range(num_rows + 1)]
        v_lines = [[0] * (num_cols + 1) for _ in range(num_rows)]
        boxes = [[0] * num_cols for _ in range(num_rows)]
        scores = [0, 0]
        current = 0
        last_action: dict[str, Any] | None = None

        for action in self.history():
            orientation, row, col = self._decode_action(action, num_rows, num_cols)
            mark = current + 1
            won_box = False
            if orientation == "h":
                h_lines[row][col] = mark
                # Box above this line
                if row > 0 and self._box_complete(h_lines, v_lines, row - 1, col, num_rows, num_cols):
                    boxes[row - 1][col] = mark
                    scores[current] += 1
                    won_box = True
                # Box below this line
                if row < num_rows and self._box_complete(h_lines, v_lines, row, col, num_rows, num_cols):
                    boxes[row][col] = mark
                    scores[current] += 1
                    won_box = True
            else:
                v_lines[row][col] = mark
                # Box left of this line
                if col > 0 and self._box_complete(h_lines, v_lines, row, col - 1, num_rows, num_cols):
                    boxes[row][col - 1] = mark
                    scores[current] += 1
                    won_box = True
                # Box right of this line
                if col < num_cols and self._box_complete(h_lines, v_lines, row, col, num_rows, num_cols):
                    boxes[row][col] = mark
                    scores[current] += 1
                    won_box = True

            last_action = {
                "orientation": orientation,
                "row": row,
                "col": col,
                "player": _player_label(current),
            }
            if not won_box:
                current = 1 - current

        return {
            "h_lines": h_lines,
            "v_lines": v_lines,
            "boxes": boxes,
            "scores": scores,
            "last_action": last_action,
        }

    @staticmethod
    def _box_complete(
        h_lines: list[list[int]],
        v_lines: list[list[int]],
        row: int,
        col: int,
        num_rows: int,
        num_cols: int,
    ) -> bool:
        del num_rows, num_cols
        return (
            h_lines[row][col] != 0
            and h_lines[row + 1][col] != 0
            and v_lines[row][col] != 0
            and v_lines[row][col + 1] != 0
        )

    def state_dict(self) -> dict[str, Any]:
        num_rows, num_cols = self._dimensions()
        replayed = self._replay(num_rows, num_cols)

        winner: str | None = None
        if self.is_terminal():
            scores = replayed["scores"]
            if scores[0] > scores[1]:
                winner = "1"
            elif scores[1] > scores[0]:
                winner = "2"
            else:
                winner = "draw"

        return {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "h_lines": replayed["h_lines"],
            "v_lines": replayed["v_lines"],
            "boxes": replayed["boxes"],
            "scores": replayed["scores"],
            "current_player": _player_label(self.current_player()) if not self.is_terminal() else "",
            "is_terminal": self.is_terminal(),
            "winner": winner,
            "last_action": replayed["last_action"],
        }

    def to_json(self) -> str:
        return json.dumps(self.state_dict())

    def action_to_dict(self, action: int) -> dict[str, Any]:
        num_rows, num_cols = self._dimensions()
        orientation, row, col = self._decode_action(action, num_rows, num_cols)
        return {"orientation": orientation, "row": row, "col": col}

    def action_to_json(self, action: int) -> str:
        return json.dumps(self.action_to_dict(action))

    def dict_to_action(self, action_dict: dict[str, Any]) -> int:
        num_rows, num_cols = self._dimensions()
        return self._encode_action(
            str(action_dict["orientation"]),
            int(action_dict["row"]),
            int(action_dict["col"]),
            num_rows,
            num_cols,
        )

    def json_to_action(self, action_json: str) -> int:
        return self.dict_to_action(json.loads(action_json))

    def observation_string(self, player: int) -> str:
        del player
        return self.to_json()

    def __str__(self) -> str:
        return self.to_json()


class DotsAndBoxesGame(proxy.Game):
    """Wraps OpenSpiel's dots_and_boxes game to use the proxy state."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("dots_and_boxes", params)
        super().__init__(
            wrapped,
            short_name="dots_and_boxes_proxy",
            long_name="Dots and Boxes (proxy)",
        )

    def new_initial_state(self, *args) -> DotsAndBoxesState:
        return DotsAndBoxesState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(DotsAndBoxesGame().get_type(), DotsAndBoxesGame)
