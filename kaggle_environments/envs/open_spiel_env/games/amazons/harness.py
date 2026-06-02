"""LLM harness for OpenSpiel Game of the Amazons.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.

Amazons turns decompose into three sub-actions: pick a queen (``from``),
move it like a chess queen (``to``), then place a barrier reachable from the
new square (``shoot`` in the engine, presented to the model as a barrier
placement). Each sub-action is one LLM call. The proxy exposes the current
phase, board, and (for in-progress turns) the queen's source/destination in
``observationString`` as JSON.

The prompt avoids weapon-adjacent vocabulary so that model APIs with strict
safety filters don't refuse to play: queens move, then place a barrier on a
square that becomes blocked. The game mechanics are unchanged.
"""

from __future__ import annotations

import json
import re
import string
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult, parse_json_action, render_rethink_suffix

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

# Matches algebraic cell names like "a1", "z26". The legal-action set bounds
# what counts as a real cell; this regex just needs to extract candidates.
_CELL_RE = re.compile(r"\b([a-zA-Z])[ \t]*([1-9][0-9]?)\b")


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


# Phase labels shown to the model. Sterilized from the engine's internal
# names so safety filters don't flag the prompt ("shoot" -> "place barrier").
_PHASE_LABEL = {
    "from": "PICK QUEEN",
    "to": "MOVE QUEEN",
    "shoot": "PLACE BARRIER",
}


def _phase_instruction(
    phase: str,
    from_sq: str | None,
    to_sq: str | None,
) -> str:
    """Per-phase instruction text. Names the in-progress squares when known.

    The TO/SHOOT phases are the high-stakes ones: the engine has already
    cleared the queen's source cell, so the board no longer reveals which
    queen the player picked up. Without telling the model, it has to infer
    the source from the move-history list -- which it usually gets wrong,
    burning retries. We name ``from_sq`` / ``to_sq`` explicitly here.
    """
    if phase == "to":
        if from_sq is not None:
            return (
                f"You picked up the queen previously at {from_sq} -- that "
                f"square is now empty on the board. Choose where to move "
                f"it: any empty square reachable from {from_sq} by a "
                "queen-move (any number of empty squares in a straight or "
                "diagonal line; the path may not cross another queen or a "
                "blocked square)."
            )
        return (
            "You picked up one of your queens earlier this turn -- its "
            "previous square is now empty on the board. Choose where to "
            "move it: queens move any number of empty squares in a "
            "straight or diagonal line and cannot pass through other "
            "pieces or blocked squares."
        )
    if phase == "shoot":
        if from_sq is not None and to_sq is not None:
            return (
                f"Your queen moved from {from_sq} to {to_sq} this turn "
                f"and is now sitting at {to_sq}. Place a barrier on any "
                f"empty square reachable from {to_sq} by a queen-move "
                "(any number of empty squares in a straight or diagonal "
                "line). The chosen square is then permanently blocked "
                "for the rest of the game: no piece may enter or cross "
                "it."
            )
        return (
            "Place a barrier from the queen you just moved. The barrier "
            "travels any number of empty squares from that queen's new "
            "square in a straight or diagonal line; the square it lands "
            "on is permanently blocked for the rest of the game."
        )
    # FROM phase
    return (
        "Choose which of your queens to move. You may only pick a queen "
        "that has at least one empty neighbouring square -- a queen with "
        "no empty neighbour cannot move and is not a legal selection."
    )


AMAZONS_PROMPT_TEMPLATE = """Let's play a strategy game on a {num_rows}x{num_cols} grid.

Pieces: X = Player 1's queens, O = Player 2's queens, # = blocked squares,
. = empty. Each turn has three steps: move one of your queens like a chess
queen (from -> to), then place a barrier on an empty square reachable by
another queen-move from its new position. Barriers permanently block their
square: no piece may enter or cross. A player who cannot move on their
turn loses.

You are playing as {player_name} ({player_glyph}).
Current sub-action: {phase_label}

Board:
{board}

Move history (last {history_max_turns} completed turns):
{move_history}

{phase_instruction}

It is your turn. Think briefly about the position, then choose your move.
Use algebraic notation: column letter + row number, e.g. ``a1`` or ``j10``.

Respond with your reasoning followed by your final answer in a JSON block:

```json
{{
  "move": "<square in algebraic notation>"
}}
```

For example: `{{"move": "a1"}}`

Failure to output your final answer in the specified format, or selecting an
illegal square, will result in a loss.
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
{{"move": "<algebraic square>"}}
```

For example: `{{"move": "a1"}}`

The move you choose must also be legal in the current state.
"""


_HISTORY_MAX_TURNS = 8


def _format_history(move_history: list[str]) -> str:
    """Group the flat action list into 3-action turns labelled by player.

    The framework stores ``move_history`` as a flat list of algebraic cells,
    one per sub-action. Without grouping the model sees ``a7, d7, e7, c6,
    c5, c4`` and has to remember "every three cells is one turn, players
    alternate, X started" -- which is undocumented and easy to get wrong.
    Render each completed turn as ``X: a7 -> d7, barrier e7`` instead. Skip
    any partial trailing turn (its squares are surfaced by the prompt's
    source-square disclosure instead, so showing it here would duplicate).
    """
    if len(move_history) < 3:
        return "(no completed turns yet)"
    full_turns = len(move_history) // 3
    start = max(0, full_turns - _HISTORY_MAX_TURNS)
    lines = []
    for t in range(start, full_turns):
        i = t * 3
        from_sq, to_sq, barrier_sq = move_history[i:i + 3]
        player = "X" if t % 2 == 0 else "O"
        lines.append(f"{player}: {from_sq} -> {to_sq}, barrier {barrier_sq}")
    return "\n".join(lines)


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


def _sub_turn_squares(
    state: Mapping[str, Any],
    move_history: list[str],
    phase: str,
    num_cols: int,
) -> tuple[str | None, str | None]:
    """Return ``(from_sq, to_sq)`` algebraic for the in-progress turn.

    Prefers the action ids surfaced by the updated proxy (always accurate);
    falls back to the trailing entries of ``move_history`` so the harness
    still works when dropped into a notebook where the proxy module isn't
    on the path (per ``deserialize_game_and_state`` comment at the top of
    this file).
    """
    from_sq: str | None = None
    to_sq: str | None = None
    from_id = state.get("from_action")
    to_id = state.get("to_action")
    if isinstance(from_id, int):
        from_sq = _action_to_algebraic(from_id, num_cols)
    if isinstance(to_id, int):
        to_sq = _action_to_algebraic(to_id, num_cols)
    if from_sq is None and phase == "to" and move_history:
        from_sq = move_history[-1]
    elif from_sq is None and phase == "shoot" and len(move_history) >= 2:
        from_sq = move_history[-2]
        to_sq = to_sq if to_sq is not None else move_history[-1]
    return from_sq, to_sq


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
    player_name = "Player 1" if current == "x" else "Player 2"
    player_glyph = "X" if current == "x" else "O"
    from_sq, to_sq = _sub_turn_squares(state, move_history, phase, num_cols)

    prompt = AMAZONS_PROMPT_TEMPLATE.format(
        num_rows=num_rows,
        num_cols=num_cols,
        player_name=player_name,
        player_glyph=player_glyph,
        phase_label=_PHASE_LABEL.get(phase, phase.upper()),
        board=_render_board(board),
        history_max_turns=_HISTORY_MAX_TURNS,
        move_history=_format_history(move_history),
        phase_instruction=_phase_instruction(phase, from_sq, to_sq),
    )

    prompt += render_rethink_suffix(
        RETHINK_ILLEGAL, RETHINK_UNPARSABLE,
        previous_response, previous_action,
    )

    return prompt


def _match_cell_to_legal(
    raw: str, legal_action_strings: Sequence[str],
) -> str | None:
    """Normalize free-form text to canonical 'a7'-style and match a legal."""
    cell = _algebraic_to_cell(raw)
    if cell is None:
        return None
    canonical = _cell_to_algebraic(*cell)
    return canonical if canonical in set(legal_action_strings) else None


def parse_response(
    response: str, legal_action_strings: Sequence[str],
) -> ParseResult:
    """Trust the model's JSON answer; let the rethink loop fix anything else."""
    return parse_json_action(response, legal_action_strings, matcher=_match_cell_to_legal)
