"""LLM harness for OpenSpiel Dots and Boxes.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.

Dots and Boxes is a two-player game on a grid of dots. Players alternate
drawing a single horizontal or vertical edge between two adjacent dots. When
a player draws the fourth edge of a unit box they claim that box (marked with
their player number) and immediately take another turn. When every edge has
been drawn the player with more boxes wins.

The harness prompt asks the LLM to respond with a JSON
``{"move": "h r c"}`` shorthand payload (e.g. ``h 0 1`` for the horizontal
edge between dots (0,1) and (0,2)), which is normalized and matched against
the legal action strings produced by OpenSpiel.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult, parse_json_action

# Matches the shorthand "h 0 1" / "v 2 0" the LLM is asked to produce
# (whitespace or comma separators). A leading lookbehind rejects matches
# inside ``P1(h,0,1)`` or other wrapped forms.
_MOVE_TOKEN_RE = re.compile(
    r"(?<![(\w])([hv])[\s,]+(\d+)[\s,]+(\d+)",
    re.IGNORECASE,
)
# Matches the canonical OpenSpiel form ``P1(h,0,1)`` for normalizing the
# legal-action list produced by pyspiel.
_OPENSPIEL_LEGAL_RE = re.compile(
    r"P\d+\(([hv]),(\d+),(\d+)\)",
    re.IGNORECASE,
)


# --- Prompt -----------------------------------------------------------------


DOTS_AND_BOXES_PROMPT_TEMPLATE = """Let's play Dots and Boxes.

Rules: The board is a {num_rows}x{num_cols} grid of unit boxes formed by a
({rows_plus}x{cols_plus}) lattice of dots. Players alternate drawing one
edge per turn between two horizontally- or vertically-adjacent dots. When
you draw the fourth edge of a unit box you claim it (marked with your
player number) and MUST take another turn immediately. The game ends when
every edge has been drawn; the player with more boxes wins.

Coordinates: rows are numbered 0 (top) to {num_rows} (bottom); columns are
0 (left) to {num_cols} (right). A horizontal edge ``h(r,c)`` connects dot
(r,c) to (r,c+1); a vertical edge ``v(r,c)`` connects dot (r,c) to (r+1,c).
Valid ranges: horizontal r in 0..{num_rows}, c in 0..{cols_minus};
vertical r in 0..{rows_minus}, c in 0..{num_cols}.

Board (``+`` = dot, ``-``/``|`` = drawn edge, ``.`` = open edge; box cells
show the owning player number when claimed, ``.`` when still open):
{board_ascii}

Score: Player 1 = {p1_score}, Player 2 = {p2_score}. Boxes remaining:
{boxes_remaining}.

You are Player {player_label}.
{last_move_label}: {last_move}
Moves you have played so far: {move_history}

Action notation: ``<h|v> <row> <col>`` (e.g. ``h 0 1`` or ``v 2 0``). Only
open edges (shown as ``.`` on the board) are legal.

Respond with your reasoning followed by your final move in a JSON block:

```json
{{
  "move": "<your move>"
}}
```

Failure to output your final answer in the specified format, or selecting
an illegal move, will result in a loss.
"""


RETHINK_ILLEGAL = """

You suggested move "{previous_action}" but this is not a legal move.
Reconsider the rules and the current state, then pick a legal move.

(Keep using the same JSON output format as before -- only the move value needs to change.)
"""

RETHINK_UNPARSABLE = """

Your previous response ended with:
{previous_response}

No JSON answer could be parsed from that. Conclude your response
with your final move as JSON in a ```json fenced block, exactly
as the original instructions required:

```json
{{"move": "<orientation row col>"}}
```

For example: `{{"move": "h 0 1"}}`

The move you choose must also be legal in the current state.
"""


# --- Helpers ----------------------------------------------------------------


def _parse_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the structured dots-and-boxes state dict out of the observation."""
    raw = observation.get("observationString", "") or ""
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    serialized = observation.get("serializedGameAndState", "")
    if serialized:
        _, state = pyspiel.deserialize_game_and_state(serialized)
        try:
            return json.loads(state.observation_string(0))
        except (json.JSONDecodeError, RuntimeError):
            pass
    return {}


def _format_board_ascii(state: Mapping[str, Any]) -> str:
    """Render the board as an ASCII grid with edges and box owners.

    Rows of dots and horizontal edges interleave with rows of vertical
    edges and box cells. Open edges render as ``.`` so the model can see
    candidate moves at a glance; drawn edges show ``---`` / ``|``; box
    cells show the owning player digit or ``.`` for an unclaimed box.
    """
    num_rows = int(state.get("num_rows", 0))
    num_cols = int(state.get("num_cols", 0))
    h_lines = state.get("h_lines") or []
    v_lines = state.get("v_lines") or []
    boxes = state.get("boxes") or []
    if not (num_rows and num_cols and h_lines and v_lines):
        return "(unavailable)"

    lines: list[str] = []
    for r in range(num_rows + 1):
        # Dots + horizontal edges row.
        parts: list[str] = []
        for c in range(num_cols):
            parts.append("+")
            owner = h_lines[r][c] if r < len(h_lines) and c < len(h_lines[r]) else 0
            parts.append("---" if owner else " . ")
        parts.append("+")
        lines.append("".join(parts))

        if r >= num_rows:
            break

        # Vertical edges + box owners row.
        parts = []
        for c in range(num_cols + 1):
            owner = v_lines[r][c] if r < len(v_lines) and c < len(v_lines[r]) else 0
            parts.append("|" if owner else ".")
            if c < num_cols:
                box_owner = boxes[r][c] if r < len(boxes) and c < len(boxes[r]) else 0
                parts.append(f" {box_owner} " if box_owner else " . ")
        lines.append("".join(parts))

    return "\n".join(lines)


def _boxes_remaining(state: Mapping[str, Any]) -> int:
    boxes = state.get("boxes") or []
    return sum(1 for row in boxes for cell in row if not cell)


def _normalize_move(raw: str) -> str | None:
    """Normalize a move string to the canonical ``h r c`` / ``v r c`` form."""
    if not raw:
        return None
    m = _MOVE_TOKEN_RE.search(raw)
    if not m:
        return None
    orientation = m.group(1).lower()
    return f"{orientation} {int(m.group(2))} {int(m.group(3))}"


def _normalize_legal(action_string: str) -> str | None:
    """Normalize an OpenSpiel ``P1(h,0,1)`` string to ``h 0 1``."""
    m = _OPENSPIEL_LEGAL_RE.search(action_string or "")
    if not m:
        return None
    return f"{m.group(1).lower()} {int(m.group(2))} {int(m.group(3))}"


def _match_to_legal(
    raw: str | None,
    legal_action_strings: Sequence[str],
) -> str | None:
    """Match a (possibly messy) move string against the legal action list."""
    if raw is None:
        return None
    normalized = _normalize_move(raw)
    if normalized is None:
        return None
    for legal in legal_action_strings:
        if _normalize_legal(legal) == normalized:
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
    player_id = observation.get("playerId", state.current_player())
    actions = state.legal_actions()
    return {a: state.action_to_string(player_id, a) for a in actions}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current dots-and-boxes state."""
    state = _parse_observation_payload(observation)
    player_id = observation.get("playerId", 0)
    player_label = 1 if player_id == 0 else 2

    num_rows = int(state.get("num_rows", 0))
    num_cols = int(state.get("num_cols", 0))
    scores = state.get("scores") or [0, 0]
    last_action = state.get("last_action")
    if last_action:
        last_move = (
            f"{last_action.get('orientation', '?')} "
            f"{last_action.get('row', '?')} {last_action.get('col', '?')}"
        )
        last_move_label = (
            "Your previous move (you completed a box, so it is your turn again)"
            if str(last_action.get("player")) == str(player_label)
            else "Opponent's last move"
        )
    else:
        last_move = "(none yet)"
        last_move_label = "Previous move"

    move_history_str = ", ".join(move_history) if move_history else "None"

    prompt = DOTS_AND_BOXES_PROMPT_TEMPLATE.format(
        num_rows=num_rows,
        num_cols=num_cols,
        rows_plus=num_rows + 1,
        cols_plus=num_cols + 1,
        rows_minus=max(num_rows - 1, 0),
        cols_minus=max(num_cols - 1, 0),
        board_ascii=_format_board_ascii(state),
        p1_score=scores[0] if len(scores) > 0 else 0,
        p2_score=scores[1] if len(scores) > 1 else 0,
        boxes_remaining=_boxes_remaining(state),
        player_label=player_label,
        last_move_label=last_move_label,
        last_move=last_move,
        move_history=move_history_str,
    )

    if previous_response is not None:
        if previous_action:
            prompt += RETHINK_ILLEGAL.format(previous_action=previous_action)
        else:
            prompt += RETHINK_UNPARSABLE.format(
                previous_response=(previous_response or "")[-500:],
            )

    return prompt


def parse_response(
    response: str, legal_action_strings: Sequence[str],
) -> ParseResult:
    """Trust the model's JSON answer; let the rethink loop fix anything else."""
    return parse_json_action(response, legal_action_strings, matcher=_match_to_legal)
