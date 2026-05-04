"""LLM harness for OpenSpiel Clobber.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.

Clobber is a sequential two-player combinatorial game on an N x M
checkerboard. Player 0 ('o', White) moves first; Player 1 ('x', Black) moves
second. On each turn the active player picks one of their pieces and moves
it onto an orthogonally adjacent square that holds an opponent's piece,
capturing it. A player who has no legal move loses.

Action strings are 4-character ``"<from><to>"`` coordinates like ``"a1b1"``.
Files are letters ``a..`` (left to right); ranks are digits ``1..N``
(bottom to top).
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(
    r"\{[^{}]*\"move\"\s*:\s*\"([a-z]\d+[a-z]\d+)\"[^{}]*\}", re.DOTALL
)
_MOVE_RE = re.compile(r"\b([a-z]\d+[a-z]\d+)\b")


# --- Prompt -----------------------------------------------------------------


CLOBBER_PROMPT_TEMPLATE = """Let's play Clobber.

Rules: Two players take turns on an {rows}x{columns} checkerboard. White
('o') moves first; Black ('x') moves second. On your turn pick one of YOUR
pieces and move it onto an orthogonally adjacent square (up/down/left/right)
that holds an OPPONENT's piece, capturing it. The first player with no
legal move LOSES.

Coordinates: files are letters ``a..{last_file}`` left-to-right; ranks are
digits ``1..{rows}`` bottom-to-top. A move is written ``<from><to>``, e.g.
``a1b1`` moves the piece on a1 to b1 (capturing whatever is on b1).

Board (top to bottom; '.' = empty):
{board_ascii}

You are Player {player_label} ('{my_piece}').
Move number: {move_number}
Last move played: {last_move}

You MUST pick one of the legal moves: {legal_moves}.

Respond with your reasoning followed by your final move in a JSON block:

```json
{{
  "move": "<one of the legal moves>"
}}
```

Failure to output your final answer in the specified format, or selecting a
move that is not in the legal list, will result in a loss.
"""


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested move "{previous_action}" but it is NOT in the legal move list.
Reconsider and pick one of the legal moves exactly.
"""


# --- Helpers ----------------------------------------------------------------


def _parse_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the structured clobber state dict out of the observation."""
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


def _format_board_ascii(board: Sequence[Sequence[str]], rows: int, columns: int) -> str:
    """Render the board with rank labels on the left and file labels on top.

    Top row of ``board`` is the highest rank (top of the visual board); we
    label it ``rows`` and count down to 1 at the bottom.
    """
    if not board or not rows or not columns:
        return "(unavailable)"
    width = max(len(str(rows)), 1)
    lines = []
    file_header = " " * (width + 1) + " ".join(
        chr(ord("a") + c) for c in range(columns)
    )
    lines.append(file_header)
    for r, row in enumerate(board):
        rank_label = str(rows - r).rjust(width)
        lines.append(f"{rank_label} " + " ".join(row))
    return "\n".join(lines)


def _extract_move_from_json(response: str) -> str | None:
    """Pull the move from a ```json``` block or a bare ``{"move": "..."}``."""
    match = _JSON_BLOCK_RE.search(response)
    if match:
        try:
            data = json.loads(match.group(1))
            move = data.get("move")
            if move is None:
                return None
            return str(move).strip()
        except json.JSONDecodeError:
            pass
    bare = _BARE_JSON_RE.search(response)
    if bare:
        return bare.group(1).strip()
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
    player_id = observation.get("playerId", 0)
    actions = state.legal_actions(player_id)
    return {a: state.action_to_string(player_id, a) for a in actions}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current clobber state."""
    del move_history  # The full board state is sufficient context for clobber.
    state = _parse_observation_payload(observation)
    player_id = observation.get("playerId", 0)

    rows = int(state.get("rows") or 0)
    columns = int(state.get("columns") or 0)
    board = state.get("board") or []
    move_number = state.get("move_number", 0)
    last_move = state.get("last_move") or "(none yet)"
    my_piece = "o" if player_id == 0 else "x"
    last_file = chr(ord("a") + max(0, columns - 1))

    legal_action_strings = observation.get("legalActionStrings") or []
    if not legal_action_strings:
        legal_action_strings = list(get_legal_moves(observation).values())
    legal_moves_str = ", ".join(sorted(legal_action_strings)) or "(none)"

    prompt = CLOBBER_PROMPT_TEMPLATE.format(
        rows=rows,
        columns=columns,
        last_file=last_file,
        board_ascii=_format_board_ascii(board, rows, columns),
        player_label=player_id,
        my_piece=my_piece,
        move_number=move_number,
        last_move=last_move,
        legal_moves=legal_moves_str,
    )

    if previous_response is not None:
        prompt += RETHINK_SUFFIX.format(
            previous_response=previous_response[:500],
            previous_action=previous_action or "(could not parse)",
        )

    return prompt


def parse_response(
    response: str, legal_action_strings: Sequence[str],
) -> ParseResult:
    """Extract a legal move from the LLM response.

    Tries a ``json`` block first, then a bare ``{"move": "..."}``, then falls
    back to scanning for any ``[a-z]\\d+[a-z]\\d+`` token that matches a
    legal move.
    """
    legal_set = set(legal_action_strings)

    raw = _extract_move_from_json(response)
    if raw is not None and raw in legal_set:
        return ParseResult(legal_action=raw, raw_action=raw)

    for token in _MOVE_RE.findall(response):
        if token in legal_set:
            return ParseResult(legal_action=token, raw_action=raw or token)

    return ParseResult(legal_action=None, raw_action=raw)
