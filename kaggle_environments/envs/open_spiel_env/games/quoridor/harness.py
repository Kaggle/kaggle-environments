"""LLM harness for OpenSpiel Quoridor.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
# A Quoridor move is either a cell ("e5") or a wall ("a1v"/"a1h"). Column is a
# single letter; row is 1-2 digits; an optional trailing v/h marks the wall.
_MOVE_RE = re.compile(r"\b([a-y])(\d{1,2})([vh])?\b", re.IGNORECASE)


QUORIDOR_PROMPT_TEMPLATE = """Let's play Quoridor.

Rules: Quoridor is a 2-player abstract strategy game on a {board_size}x{board_size} grid.
Each player has one pawn and {wall_count} walls. Player x starts on the bottom
row (row {board_size}) and wins by reaching the top row (row 1). Player o starts on
the top row and wins by reaching the bottom row. Starting pawn positions are
shown on the board below.

On each turn a player MUST do exactly one of the following:

  1. Move their pawn one square orthogonally (up, down, left, or right) into an
     empty adjacent square that is not blocked by a wall.
  2. If the opponent's pawn is in an adjacent square, jump straight over it to
     the square beyond. If a wall (or the board edge) is directly behind the
     opponent, the jumping pawn may instead step diagonally to either square
     beside the opponent.
  3. Place one of their remaining walls between two rows or two columns. Each
     wall is 2 squares long. Walls may not overlap, cross another wall, or
     extend off the board, and a wall may NOT be placed if it would completely
     cut off either player from their goal row.

If a player is completely boxed in and has no legal pawn move and no walls
left to place, their only legal action is to stay in place; in that case
``legal_actions`` will contain exactly one entry equal to their current
square. The game also ends in a draw if no winner emerges within
{max_moves} total moves.

Coordinates use the format "<column><row>" with column letters starting at "a"
(left) and row numbers starting at 1 (top). Move notation:

  - Pawn move: the destination square, e.g. "e8" means move your pawn to e8.
  - Vertical wall: "<col><row>v" places a 2-cell vertical wall whose top half
    sits between the named square and the square to its right; the wall also
    covers the same column gap in the next row down. Example: "a1v" blocks the
    gap between a1 and b1 (and between a2 and b2).
  - Horizontal wall: "<col><row>h" places a 2-cell horizontal wall just below
    the named row, covering the named column and the column to its right.
    Example: "a1h" blocks the gap between a1 and a2 (and between b1 and b2).

Current game state (parsed from the engine):
{state_str}

Your remaining walls: {own_walls}. Opponent's remaining walls: {opp_walls}.
Walls already on the board:
  Vertical:   {vertical_walls}
  Horizontal: {horizontal_walls}

Your moves so far: {move_history}

You are playing as player {player_code} ({goal_description}).
It is now your turn. Play your strongest legal move.

Your response should include the reasoning that led you to your move, and
conclude with your final move as JSON formatted exactly like:

```json
{{
  "move": "<move>"
}}
```

Where `<move>` is a single cell coordinate (pawn move) or a wall coordinate
ending in `v` or `h`. Failure to output your final answer in the specified
format will result in a loss.
Begin!
"""


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested move "{previous_action}" but this is not a legal move. Possible
reasons: the destination is off the board or blocked, you cannot jump because
no opponent is adjacent, the wall would overlap an existing wall, the wall
would completely block one player from reaching their goal row, or you have no
walls left. Reconsider the state and play a legal move.
"""


def _iter_json_objects(text: str) -> list[str]:
    """Yield top-level balanced ``{...}`` substrings from ``text``.

    Skips brace-like characters inside double-quoted strings (including
    escaped quotes) so e.g. ``{"move": "}"}`` is handled correctly. Regex
    can't do this on its own because JSON objects can nest arbitrarily.
    """
    out: list[str] = []
    depth = 0
    start = -1
    i = 0
    in_str = False
    escape = False
    while i < len(text):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start : i + 1])
                    start = -1
        i += 1
    return out


def _extract_move_from_json(response: str) -> str | None:
    """Try to extract a move string from a JSON code block, then any JSON object.

    The fenced `````json`` block is the primary path. As a fallback,
    walk every balanced ``{...}`` in the response, parse it with ``json.loads``,
    and return the value at key ``move`` from the first one that parses.
    """
    match = _JSON_BLOCK_RE.search(response)
    if match:
        try:
            data = json.loads(match.group(1))
            move = str(data.get("move", "")).strip()
            if move:
                return move
        except json.JSONDecodeError:
            pass

    for candidate in _iter_json_objects(response):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "move" in data:
            move = str(data["move"]).strip()
            if move:
                return move
    return None


def _normalize(move: str) -> str:
    return move.replace(" ", "").lower()


def _match_move_to_legal(
    move: str,
    legal_moves: Sequence[str],
) -> str | None:
    """Match a move string (e.g. "e5" or "a1v") to a legal move string."""
    candidate = _normalize(move)
    if not candidate:
        return None
    for legal in legal_moves:
        if _normalize(legal) == candidate:
            return legal
    return None


def get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str]:
    """Return ``{action_id: action_string}`` for the current state."""
    legal_actions = observation.get("legalActions")
    legal_action_strings = observation.get("legalActionStrings")
    if legal_actions and legal_action_strings:
        return dict(zip(legal_actions, legal_action_strings))

    serialized = observation.get("serializedGameAndState", "")
    if not serialized:
        return {}
    _, state = pyspiel.deserialize_game_and_state(serialized)
    actions = state.legal_actions()
    return {a: state.action_to_string(a) for a in actions}


def _render_board(parsed: Mapping[str, Any]) -> str:
    """Render the proxy's JSON cells/walls as an ASCII board.

    The layout mirrors OpenSpiel's ``ToString`` so coordinates in the prompt
    line up with what a player would read on a Quoridor board: columns labeled
    a..<letter> across the top and bottom, rows numbered 1..N down the sides.
    """
    board_size = int(parsed.get("board_size") or 9)
    cells = parsed.get("cells") or [[None] * board_size for _ in range(board_size)]
    v_walls = set(parsed.get("vertical_walls") or [])
    h_walls = set(parsed.get("horizontal_walls") or [])

    def label(col: int, row: int) -> str:
        return f"{chr(ord('a') + col)}{row + 1}"

    glyph_of = {0: "x", 1: "o", 2: "n", 3: "s"}

    col_letters = "   ".join(chr(ord("a") + c) for c in range(board_size))
    lines: list[str] = [f"   {col_letters}"]
    for r in range(board_size):
        # Cell row: row label, cells, vertical walls between cells.
        row_label = f"{r + 1:>2}"
        parts = [row_label]
        for c in range(board_size):
            cell = cells[r][c] if r < len(cells) and c < len(cells[r]) else None
            parts.append(f" {glyph_of.get(cell, '.')} ")
            if c < board_size - 1:
                # Wall belongs to this row if its top half OR bottom half covers
                # the current row at the same column gap.
                has_wall = label(c, r) + "v" in v_walls or (r > 0 and label(c, r - 1) + "v" in v_walls)
                parts.append("|" if has_wall else " ")
        parts.append(f" {row_label}")
        lines.append("".join(parts))
        # Wall row between this row and the next.
        if r < board_size - 1:
            wall_parts = ["  "]
            for c in range(board_size):
                # Wall belongs in this gap if its top column OR left column
                # covers it.
                has_h = label(c, r) + "h" in h_walls or (c > 0 and label(c - 1, r) + "h" in h_walls)
                wall_parts.append("---" if has_h else "   ")
                if c < board_size - 1:
                    # Corner marker if either neighbour places a wall touching
                    # this corner.
                    corner = label(c, r) + "v" in v_walls or label(c, r) + "h" in h_walls
                    wall_parts.append("+" if corner else " ")
            lines.append("".join(wall_parts))
    lines.append(f"   {col_letters}")
    return "\n".join(lines) + "\n"


def _format_state(observation: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    """Return (ASCII board, parsed observation dict).

    Raises if ``observationString`` is missing or empty -- silently degrading
    to a board-less prompt would let the LLM hallucinate state.
    """
    raw = observation.get("observationString", "")
    if not raw:
        raise ValueError("Quoridor harness received empty observationString")
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw if raw.endswith("\n") else raw + "\n", {}
    return _render_board(parsed), parsed


def _goal_description(player_code: str, board_size: int) -> str:
    if player_code == "x":
        return f"starts at the center of row {board_size}; reach row 1 to win"
    return f"starts at the center of row 1; reach row {board_size} to win"


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current Quoridor state."""
    state_str, parsed = _format_state(observation)
    board_size = int(parsed.get("board_size") or 9)
    walls_remaining = parsed.get("walls_remaining") or {}
    player_id = observation.get("playerId", 0)
    player_code = "x" if player_id == 0 else "o"
    opp_code = "o" if player_code == "x" else "x"
    own_walls = walls_remaining.get(player_code, "?")
    opp_walls = walls_remaining.get(opp_code, "?")
    wall_count = own_walls if isinstance(own_walls, int) else (board_size * board_size) // 8

    vertical_walls = parsed.get("vertical_walls") or []
    horizontal_walls = parsed.get("horizontal_walls") or []
    move_history_str = " ".join(move_history) if move_history else "None"

    # OpenSpiel terminates a stalled game at 4 * board_size^2 total moves.
    max_moves = 4 * board_size * board_size

    prompt = QUORIDOR_PROMPT_TEMPLATE.format(
        board_size=board_size,
        wall_count=wall_count,
        max_moves=max_moves,
        state_str=state_str,
        own_walls=own_walls,
        opp_walls=opp_walls,
        vertical_walls=", ".join(vertical_walls) if vertical_walls else "(none)",
        horizontal_walls=", ".join(horizontal_walls) if horizontal_walls else "(none)",
        move_history=move_history_str,
        player_code=player_code,
        goal_description=_goal_description(player_code, board_size),
    )

    if previous_response is not None:
        prompt += RETHINK_SUFFIX.format(
            previous_response=previous_response[:500],
            previous_action=previous_action or "(could not parse)",
        )

    return prompt


def parse_response(
    response: str,
    legal_action_strings: Sequence[str],
) -> ParseResult:
    """Extract a legal Quoridor move from the model response.

    Tries to extract the move from a JSON block first, then falls back to
    scanning the response text for any move-shaped token (cell or wall).
    """
    raw = _extract_move_from_json(response)
    if raw is not None:
        matched = _match_move_to_legal(raw, legal_action_strings)
        if matched is not None:
            return ParseResult(legal_action=matched, raw_action=raw)

    # Text-scan fallback. Walk matches in order and keep the LAST legal one --
    # models tend to brainstorm earlier candidates and state their final answer
    # near the end of the response (or in an "I'll play X" tail line).
    last_match: tuple[str, str] | None = None
    for m in _MOVE_RE.finditer(response):
        candidate = m.group(0)
        matched = _match_move_to_legal(candidate, legal_action_strings)
        if matched is not None:
            last_match = (matched, candidate)
    if last_match is not None:
        matched, candidate = last_match
        return ParseResult(legal_action=matched, raw_action=raw or candidate)

    return ParseResult(legal_action=None, raw_action=raw)
