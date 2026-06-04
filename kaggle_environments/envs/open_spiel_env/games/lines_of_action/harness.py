"""LLM harness for OpenSpiel Lines of Action.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult, parse_json_action, render_rethink_suffix

# Lines of Action notation: "<file><rank><sep><file><rank>" where sep is '-' (move)
# or 'x' (capture). Files are a-h, ranks are 1-8. Use [ \t]* (not \s*) so the
# pattern cannot bridge a newline and stitch unrelated tokens into a fake move.
_MOVE_RE = re.compile(r"\b([a-h])([1-8])[ \t]*([-x])[ \t]*([a-h])([1-8])\b", re.IGNORECASE)


# --- Prompt -----------------------------------------------------------------


LOA_PROMPT_TEMPLATE = """Let's play Lines of Action.

Rules: 8x8 board with files a-h and ranks 1-8. You play 12 pieces; the
opponent plays 12 pieces. Black (X) starts on the top and bottom rows
(except the corners); White (O) starts on the left and right columns
(except the corners). Black moves first.

A piece moves in a straight line (horizontally, vertically, or diagonally)
a number of squares EXACTLY equal to the total number of pieces (both
colors) on that line. A piece may jump over its own pieces but NOT over
opponent pieces. A piece may not land on one of its own pieces; landing on
an opponent's piece captures it.

You win by connecting all of your remaining pieces into a single group
(connectivity is 8-directional: horizontal, vertical, or diagonal
neighbours count as connected). If your move connects both your pieces
and your opponent's pieces simultaneously, you (the moving player) win.
A player who has no legal moves on their turn loses. The game is drawn
if the same position (with the same player to move) occurs for the
second time, or if 1000 moves are played without a winner.

Board ('.' = empty, 'x' = Black, 'o' = White). Each rank has its total
piece count on the right ("row"); each file's total is below ("col"):
{board_ascii}

Line counts for each of your pieces (use these to pick the move
distance — your piece moves EXACTLY this many squares along the chosen
line: row=horizontal, col=vertical, /=NE-SW diagonal, \\=NW-SE diagonal):
{piece_line_counts}

Move number: {move_number}
Last move played: {last_move}
Moves played so far: {move_history}

You are playing as {player_name} ({player_code}).
It is now your turn. Play your strongest move.
The move MUST be legal.

Action notation: ``<from><sep><to>`` -- e.g. ``b1-h1`` (slide b1 to h1)
or ``c3xa3`` (capture moving from c3 onto an opponent piece at a3). Use
'-' for a normal move and 'x' for a capture. Files are lowercase a-h;
ranks are 1-8.

Respond with your reasoning followed by your final move in a JSON block:

```json
{{
  "move": "<move>"
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
{{"move": "<from><sep><to>"}}
```

For example: `{{"move": "b1-h1"}}`

The move you choose must also be legal in the current state.
"""


# --- Helpers ----------------------------------------------------------------


def _parse_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the structured LoA state dict out of the observation."""
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


def _format_board_ascii(board: Sequence[Sequence[str]]) -> str:
    """Render the 8x8 board with rank labels on the left and files on top.

    ``board[0]`` is rank 1 (bottom row); ``board[7]`` is rank 8 (top). We
    print ranks top-down so the visual board matches standard orientation,
    and add row/column piece counts in the margins so the model doesn't
    have to count pieces on each line itself.
    """
    if not board:
        return "(unavailable)"
    n = len(board[0])
    file_header = "    " + " ".join(chr(ord("a") + c) for c in range(n)) + "   row"
    lines = [file_header]
    for r in range(len(board) - 1, -1, -1):
        row = board[r]
        row_count = sum(1 for cell in row if cell != ".")
        lines.append(f"  {r + 1} " + " ".join(row) + f"   {row_count}")
    col_counts = [
        sum(1 for r in range(len(board)) if board[r][c] != ".")
        for c in range(n)
    ]
    lines.append("col   " + " ".join(str(c) for c in col_counts))
    return "\n".join(lines)


def _format_piece_line_counts(
    board: Sequence[Sequence[str]], my_piece: str,
) -> str:
    """For each of the player's pieces, list piece counts on its 4 lines.

    A LoA piece moves EXACTLY this many squares along the chosen line, so
    pre-computing these counts saves the model from a tedious step that's
    easy to get wrong.
    """
    if not board:
        return "(unavailable)"
    n = len(board[0])
    # Row r (1-indexed), column f (0=a). board[r-1][f].
    row_count = [sum(1 for cell in board[r] if cell != ".") for r in range(n)]
    col_count = [
        sum(1 for r in range(n) if board[r][c] != ".") for c in range(n)
    ]
    ne_count: dict[int, int] = {}  # key = rank - file (constant on '/' diagonal)
    nw_count: dict[int, int] = {}  # key = rank + file (constant on '\' diagonal)
    for r in range(n):
        for c in range(n):
            if board[r][c] == ".":
                continue
            ne_count[(r + 1) - (c + 1)] = ne_count.get((r + 1) - (c + 1), 0) + 1
            nw_count[(r + 1) + (c + 1)] = nw_count.get((r + 1) + (c + 1), 0) + 1

    lines = []
    for r in range(n - 1, -1, -1):
        for c in range(n):
            if board[r][c] != my_piece:
                continue
            sq = f"{chr(ord('a') + c)}{r + 1}"
            row = row_count[r]
            col = col_count[c]
            ne = ne_count.get((r + 1) - (c + 1), 0)
            nw = nw_count.get((r + 1) + (c + 1), 0)
            lines.append(f"  {sq}: row={row}, col={col}, /={ne}, \\={nw}")
    return "\n".join(lines) if lines else "  (no pieces)"


def _normalize(move: str) -> str:
    """Strip whitespace and lowercase a candidate move string."""
    return re.sub(r"\s+", "", move).lower()


def _match_move_to_legal(
    move: str,
    legal_moves: Sequence[str],
) -> str | None:
    """Match a move string against the legal-move list, ignoring case/whitespace.

    Also accepts a move whose separator differs from what the engine reports
    (e.g. the model said "b1-c2" but the actual legal move is "b1xc2"
    because c2 holds an opponent piece).
    """
    target = _normalize(move)
    if not target:
        return None

    legal_normalized = {_normalize(legal): legal for legal in legal_moves}
    if target in legal_normalized:
        return legal_normalized[target]

    # Fall back to matching just the from/to coordinates (ignore separator).
    m = _MOVE_RE.fullmatch(target)
    if not m:
        return None
    coords = (m.group(1).lower(), m.group(2), m.group(4).lower(), m.group(5))
    for legal in legal_moves:
        lm = _MOVE_RE.fullmatch(_normalize(legal))
        if lm and (lm.group(1).lower(), lm.group(2), lm.group(4).lower(), lm.group(5)) == coords:
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


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current game state."""
    state = _parse_observation_payload(observation)
    player_id = observation.get("playerId", 0)
    player_name = "Black" if player_id == 0 else "White"
    player_code = "X" if player_id == 0 else "O"

    board = state.get("board") or []
    move_number = state.get("move_number", len(move_history))
    last_move = state.get("last_move") or "(none yet)"

    my_piece = "x" if player_id == 0 else "o"
    move_history_str = " ".join(move_history) if move_history else "None"

    prompt = LOA_PROMPT_TEMPLATE.format(
        board_ascii=_format_board_ascii(board),
        piece_line_counts=_format_piece_line_counts(board, my_piece),
        move_number=move_number,
        last_move=last_move,
        move_history=move_history_str,
        player_name=player_name,
        player_code=player_code,
    )

    prompt += render_rethink_suffix(
        RETHINK_ILLEGAL, RETHINK_UNPARSABLE,
        previous_response, previous_action,
    )

    return prompt


def parse_response(
    response: str, legal_action_strings: Sequence[str],
) -> ParseResult:
    """Trust the model's JSON answer; let the rethink loop fix anything else."""
    return parse_json_action(response, legal_action_strings, matcher=_match_move_to_legal)
