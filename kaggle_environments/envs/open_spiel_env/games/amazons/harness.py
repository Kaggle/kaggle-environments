"""LLM harness for OpenSpiel Game of the Amazons.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.

Amazons turns decompose into three sub-actions: pick an amazon (``from``),
move it like a queen (``to``), then shoot an arrow from the new square
(``shoot``). Each sub-action is one LLM call. The proxy exposes the current
phase and board in ``observationString`` as JSON.
"""

from __future__ import annotations

import json
import re
import string
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult

# Importing the proxy registers the ``amazons_proxy`` pyspiel game so that
# ``deserialize_game_and_state`` can rebuild it from the obs. Wrapped in
# try/except because the harness can be dropped into a notebook where the
# proxy module isn't on the path; in that case we rely on ``legalActions``
# being included in the observation directly.
try:
    from kaggle_environments.envs.open_spiel_env.games.amazons import (  # noqa: F401
        amazons_proxy,
    )
except Exception:
    pass

_DEFAULT_BOARD_SIZE = 10  # only used when an observation lacks board dims

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{[^{}]*\"move\"\s*:\s*\"([^\"]+)\"[^{}]*\}", re.DOTALL)
# Matches algebraic cell names like "a1", "z26". The legal-action set bounds
# what counts as a real cell; this regex just needs to extract candidates.
_CELL_RE = re.compile(r"\b([a-zA-Z])\s*([1-9][0-9]?)\b")


# --- Coordinate helpers -----------------------------------------------------


def _cell_to_algebraic(row: int, col: int) -> str:
    """Convert 0-indexed (row, col) to algebraic notation, e.g. (0, 0) -> 'a1'."""
    return f"{string.ascii_lowercase[col]}{row + 1}"


def _algebraic_to_cell(text: str) -> tuple[int, int] | None:
    """Convert 'a1' / 'J10' to 0-indexed (row, col), or None if unparseable."""
    match = _CELL_RE.search(text)
    if not match:
        return None
    col = string.ascii_lowercase.index(match.group(1).lower())
    row = int(match.group(2)) - 1
    return row, col


def _action_to_algebraic(action: int, num_cols: int) -> str:
    """Convert a pyspiel action id to algebraic notation.

    Amazons encodes every sub-action (from / to / shoot) as
    ``row * num_cols + col`` (0-indexed), so the same conversion works in
    all three phases regardless of the action_to_string format pyspiel
    happens to use. ``num_cols`` is read from the live observation rather
    than hardcoded because OpenSpiel ships different default sizes by build.
    """
    return _cell_to_algebraic(action // num_cols, action % num_cols)


def _algebraic_to_action(algebraic: str, num_cols: int) -> int | None:
    """Inverse of ``_action_to_algebraic`` ('a7' -> 60 on a 10-wide board)."""
    cell = _algebraic_to_cell(algebraic)
    if cell is None:
        return None
    row, col = cell
    return row * num_cols + col


# --- Board rendering --------------------------------------------------------


def _render_board(board: Sequence[Sequence[str]]) -> str:
    """Render the board with column letters and row numbers around the edges."""
    rows_count = len(board)
    cols_count = len(board[0]) if rows_count else 0
    header = "   " + " ".join(string.ascii_lowercase[:cols_count])
    rows: list[str] = [header]
    for r in range(rows_count):
        rows.append(f"{r + 1:>2} " + " ".join(board[r]))
    return "\n".join(rows)


def _parse_state(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Parse the proxy's JSON observation, returning ``{}`` on any failure."""
    raw = observation.get("observationString", "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _board_dims(state: Mapping[str, Any]) -> tuple[int, int]:
    """Return ``(num_rows, num_cols)`` from a parsed state dict.

    Prefers the actual board grid (always accurate), falls back to explicit
    dimension fields, and finally to a 10x10 default when the obs is empty.
    """
    board = state.get("board") or []
    if board:
        return len(board), len(board[0])
    nr = state.get("num_rows") or state.get("board_size") or _DEFAULT_BOARD_SIZE
    nc = state.get("num_cols") or state.get("board_size") or _DEFAULT_BOARD_SIZE
    return int(nr), int(nc)


# --- Prompt -----------------------------------------------------------------


_PHASE_INSTRUCTION = {
    "from": (
        "Choose which of your amazons to move. Each legal move below names a "
        "square that currently holds one of your amazons."
    ),
    "to": (
        "You picked up an amazon. Choose where to move it. Amazons move like "
        "chess queens (any number of empty squares in a straight or diagonal "
        "line) and cannot pass through arrows or other pieces."
    ),
    "shoot": (
        "Now shoot an arrow from your amazon's new square. The arrow flies "
        "like a queen's move from that square. The square it lands on is "
        "burned for the rest of the game and no piece may enter or cross it."
    ),
}


AMAZONS_PROMPT_TEMPLATE = """Let's play the Game of the Amazons on a {num_rows}x{num_cols} board.

Pieces: X = Black amazons, O = White amazons, # = burned (arrow) squares,
. = empty. Each turn has three steps: move one of your amazons like a chess
queen (from -> to), then shoot an arrow from its new square (also like a
queen). A player who cannot move on their turn loses.

You are playing as {player_name} ({player_glyph}).
Current sub-action: {phase_upper}

Board:
{board}

Move history (last {history_max}):
{move_history}

{phase_instruction}

Legal moves you may play (algebraic, column letter + row number):
{legal_moves}

It is your turn. Think briefly about the position, then choose your move.
The move MUST be exactly one of the legal squares listed above.

Respond with your reasoning followed by your final answer in a JSON block:

```json
{{
  "move": "<square from the legal list, e.g. a1>"
}}
```

Failure to output your final answer in the specified format, or selecting a
square that is not in the legal list, will result in a loss.
"""


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested move "{previous_action}" but it is NOT in the legal list above.
Reconsider and pick a square that appears verbatim in the legal moves list.
"""


_HISTORY_MAX = 12


def _format_history(move_history: list[str]) -> str:
    if not move_history:
        return "(no moves yet)"
    return ", ".join(move_history[-_HISTORY_MAX:])


# --- Public functions (called by main.py) -----------------------------------


def get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str]:
    """Return ``{action_id: algebraic_string}`` for the current sub-action.

    The strings are algebraic cell names like ``"a7"`` so the model can
    reason in board coordinates. The action int is computed directly from
    the pyspiel encoding (``row * num_cols + col``, 0-indexed) so this
    works regardless of how the runtime's pyspiel formats action strings.
    ``num_cols`` is read from the observation so the conversion stays
    correct on OpenSpiel builds whose amazons defaults aren't 10x10.
    """
    state = _parse_state(observation)
    _, num_cols = _board_dims(state)

    legal_actions = observation.get("legalActions")

    if not legal_actions:
        serialized = observation.get("serializedGameAndState", "")
        if not serialized:
            return {}
        try:
            _, sp_state = pyspiel.deserialize_game_and_state(serialized)
        except Exception:
            return {}
        legal_actions = sp_state.legal_actions()

    return {a: _action_to_algebraic(a, num_cols) for a in legal_actions}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current sub-action."""
    state = _parse_state(observation)
    num_rows, num_cols = _board_dims(state)
    board = state.get("board") or [["." for _ in range(num_cols)] for _ in range(num_rows)]
    phase = state.get("phase") or "from"
    current = state.get("current_player", "x")
    player_name = "Black" if current == "x" else "White"
    player_glyph = "X" if current == "x" else "O"

    legal_moves = get_legal_moves(observation)
    legal_strings = list(legal_moves.values())

    prompt = AMAZONS_PROMPT_TEMPLATE.format(
        num_rows=num_rows,
        num_cols=num_cols,
        player_name=player_name,
        player_glyph=player_glyph,
        phase_upper=phase.upper(),
        board=_render_board(board),
        history_max=_HISTORY_MAX,
        move_history=_format_history(move_history),
        phase_instruction=_PHASE_INSTRUCTION.get(phase, _PHASE_INSTRUCTION["from"]),
        legal_moves=", ".join(legal_strings) if legal_strings else "(none)",
    )

    if previous_response is not None:
        prompt += RETHINK_SUFFIX.format(
            previous_response=previous_response[:500],
            previous_action=previous_action or "(could not parse)",
        )

    return prompt


def _extract_move_from_json(response: str) -> str | None:
    """Pull the move string out of a ```json``` block, or a bare JSON object."""
    match = _JSON_BLOCK_RE.search(response)
    if match:
        try:
            data = json.loads(match.group(1))
            move = str(data.get("move", "")).strip()
            if move:
                return move
        except json.JSONDecodeError:
            pass

    bare = _BARE_JSON_RE.search(response)
    if bare:
        return bare.group(1).strip()

    return None


def _normalize_cell(text: str) -> str | None:
    """Normalize free-form text to canonical 'a7'-style notation."""
    cell = _algebraic_to_cell(text)
    if cell is None:
        return None
    row, col = cell
    return _cell_to_algebraic(row, col)


def parse_response(
    response: str, legal_action_strings: Sequence[str],
) -> ParseResult:
    """Extract a legal cell from the model response.

    Tries the JSON block first, then a bare ``{"move": "..."}`` object,
    then a fallback scan for any legal cell mentioned anywhere in the
    response. Match is on canonical algebraic form so 'A7', 'a7 ', etc.
    all map to 'a7'.
    """
    legal_set = set(legal_action_strings)

    raw = _extract_move_from_json(response)
    if raw is not None:
        canonical = _normalize_cell(raw)
        if canonical is not None and canonical in legal_set:
            return ParseResult(legal_action=canonical, raw_action=raw)

    for match in _CELL_RE.finditer(response):
        canonical = _normalize_cell(match.group(0))
        if canonical is not None and canonical in legal_set:
            return ParseResult(legal_action=canonical, raw_action=raw or canonical)

    return ParseResult(legal_action=None, raw_action=raw)
