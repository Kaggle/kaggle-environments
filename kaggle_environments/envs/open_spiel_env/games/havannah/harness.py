"""LLM harness for OpenSpiel Havannah.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult, extract_last_json_object

_COORD_RE = re.compile(r"\b([a-o])(\d{1,2})\b", re.IGNORECASE)


# --- Prompt ---


HAVANNAH_PROMPT_TEMPLATE = """Let's play Havannah.

Rules: Havannah is a connection game on a hexagonal board of side length {board_size}
(so {diameter} cells across the widest row, and the two opposite corners of the
rhombic grid are cut off so the board has 6 corners and 6 edges).
Players alternate placing one stone per turn on any empty cell. Stones never move
or get captured. A player wins immediately by completing any one of three
structures with their own stones:

  - Ring:   a closed loop of connected stones enclosing one or more cells
            (the enclosed cells may be empty or occupied by either player).
  - Bridge: a connected chain joining any 2 of the 6 corner cells.
  - Fork:   a connected chain touching any 3 of the 6 edges
            (corner cells do NOT count as edges).

Two stones are "connected" if they sit on adjacent hex cells and share the same
color, or are joined through a path of same-colored stones via adjacency.
If the board fills up without a win, the game is a draw.

Coordinates use the format "<column><row>" with column letters starting at "a"
and row numbers starting at 1. The board state below uses "X" for player x's
stones, "O" for player o's stones, and "." for empty cells. The leading and
trailing letters/numbers on each row are coordinate labels, NOT stones.

The current game state is:
{state_str}
The moves played so far are:
{move_history}
You are playing as player {player_code} ({player_name}).
It is now your turn. Play your strongest move.
The move MUST be legal (target an empty cell on the board).
Your response should include the reasoning that led you to your move, and
conclude with your final move as a JSON formatted as follows:

```json
{{
  "move": "<coord>"
}}
```

Where coord is a single cell coordinate like "a1", "g4", or "h8".
Failure to output your final answer in the specified format will result in a loss.
Begin!
"""


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested move "{previous_action}" but this is not a legal move (the cell
must be empty and on the board). Reconsider the state and play a legal move.
"""


# --- Helpers ----------------------------------------------------------------


def _extract_move_from_json(response: str) -> str | None:
    """Pull the move string out of the LAST JSON object in the response."""
    data = extract_last_json_object(response, required_keys=("move",))
    if data is None:
        return None
    move = str(data.get("move") or "").strip()
    return move or None


def _normalize(move: str) -> str:
    return move.replace(" ", "").lower()


def _match_move_to_legal(
    move: str,
    legal_moves: Sequence[str],
) -> str | None:
    """Match a move string (e.g. "a1") to a legal move string."""
    candidate = _normalize(move)
    if not candidate:
        return None
    for legal in legal_moves:
        if _normalize(legal) == candidate:
            return legal
    return None


# --- Public functions (called by main.py) -----------------------------------


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
    """Render the proxy's JSON board as the human-readable ASCII grid.

    Mirrors the OpenSpiel ToString() layout so the prompt matches the coordinate
    conventions players already know.
    """
    board = parsed.get("board") or []
    board_size = int(parsed.get("board_size") or len(board) // 2 + 1)
    diameter = board_size * 2 - 1

    lines: list[str] = []
    # Top column labels: a..<letter at board_size-1>
    top_indent = " " * (board_size + 3)
    top_labels = " ".join(chr(ord("a") + x) for x in range(board_size))
    lines.append(f"{top_indent} {top_labels}")

    for y in range(diameter):
        row_cells = board[y] if y < len(board) else []
        glyphs = []
        for cell in row_cells:
            if cell == "x":
                glyphs.append("X")
            elif cell == "o":
                glyphs.append("O")
            else:
                glyphs.append(".")
        leading_spaces = abs(board_size - 1 - y) + 1 + (0 if y + 1 >= 10 else 1)
        prefix = " " * leading_spaces + str(y + 1)
        body = " " + " ".join(glyphs)
        if y < board_size - 1:
            # Trailing right-column label, matching the OpenSpiel layout.
            suffix = " " + chr(ord("a") + board_size + y)
            lines.append(prefix + body + suffix)
        else:
            lines.append(prefix + body)
    return "\n".join(lines) + "\n"


def _format_state(observation: Mapping[str, Any]) -> tuple[str, int]:
    """Format the observation as an ASCII board plus return the board size."""
    raw = observation.get("observationString", "")
    if not raw:
        return "", 0
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Fall back to the raw string if it isn't JSON.
        return raw if raw.endswith("\n") else raw + "\n", 0
    board_size = int(parsed.get("board_size") or 0)
    return _render_board(parsed), board_size


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current game state."""
    state_str, board_size = _format_state(observation)
    if board_size <= 0:
        board_size = 8  # OpenSpiel default
    diameter = board_size * 2 - 1
    player_id = observation.get("playerId", 0)
    player_code = "x" if player_id == 0 else "o"
    player_name = "first to move" if player_id == 0 else "second to move"

    move_history_str = " ".join(move_history) if move_history else "None"

    prompt = HAVANNAH_PROMPT_TEMPLATE.format(
        board_size=board_size,
        diameter=diameter,
        state_str=state_str,
        move_history=move_history_str,
        player_code=player_code,
        player_name=player_name,
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
    """Extract a legal Havannah move from the model response.

    If the model gave an explicit JSON move, trust it: when it's legal we
    submit it, and when it's illegal we surface ``legal_action=None`` so
    the rethink loop can ask the model to correct itself. Only when no JSON
    is present do we fall back to scanning for a bare coordinate.
    """
    raw = _extract_move_from_json(response)
    if raw is not None:
        matched = _match_move_to_legal(raw, legal_action_strings)
        return ParseResult(legal_action=matched, raw_action=raw)

    # No JSON answer at all -- best-effort: pick the last coord-like token,
    # since models typically enumerate rejected options before stating the
    # final move.
    for m in reversed(list(_COORD_RE.finditer(response))):
        candidate = m.group(0)
        matched = _match_move_to_legal(candidate, legal_action_strings)
        if matched is not None:
            return ParseResult(legal_action=matched, raw_action=candidate)

    return ParseResult(legal_action=None, raw_action=None)
