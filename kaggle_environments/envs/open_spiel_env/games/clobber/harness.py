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

from kaggle_environments.core_harness import ParseResult, parse_json_action, render_rethink_suffix


# --- Prompt -----------------------------------------------------------------


CLOBBER_PROMPT_TEMPLATE = """Let's play Clobber.

Rules: Two players take turns on a {rows}x{columns} checkerboard. White
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
Moves played so far (both players, oldest first): {move_history_str}
Last move played: {last_move}

Choose your move. It must move one of your pieces onto an orthogonally
adjacent square that holds an opponent's piece.

Respond with your reasoning followed by your final move in a JSON block:

```json
{{
  "move": "<from><to>"
}}
```

For example: `{{"move": "a1b1"}}`

Failure to output your final answer in the specified format, or selecting an
illegal move, will result in a loss.
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
{{"move": "<from><to>"}}
```

For example: `{{"move": "a1b1"}}`

The move you choose must also be legal in the current state.
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


def _board_dims(state: Mapping[str, Any]) -> tuple[int, int]:
    """Return ``(rows, columns)`` from a parsed state dict.

    Prefers the actual board grid (always accurate for the current state),
    falls back to the explicit ``rows``/``columns`` fields the proxy emits.
    Returns ``(0, 0)`` only when nothing is available.
    """
    board = state.get("board") or []
    if board:
        return len(board), len(board[0])
    rows = state.get("rows") or 0
    columns = state.get("columns") or 0
    return int(rows), int(columns)


def _reconstruct_move_history(observation: Mapping[str, Any]) -> list[str]:
    """Rebuild the full-game move history from the serialized pyspiel state.

    Used only when the proxy state dict didn't surface ``move_history`` (e.g.
    older replays). Clobber has no chance phase, so play actions alternate
    starting with player 0.
    """
    serialized = observation.get("serializedGameAndState", "")
    if not serialized:
        return []
    _, state = pyspiel.deserialize_game_and_state(serialized)
    return [
        state.action_to_string(idx % 2, action)
        for idx, action in enumerate(state.history())
    ]


def _format_move_history(moves: Sequence[str]) -> str:
    return ", ".join(moves) if moves else "(none yet)"


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
    # The per-agent `move_history` argument only contains this agent's own
    # actions. We need both players' moves, so we source the full game
    # history from the proxy's state_dict (with a serialized-state fallback).
    del move_history
    state = _parse_observation_payload(observation)
    player_id = observation.get("playerId", 0)

    rows, columns = _board_dims(state)
    board = state.get("board") or []
    full_moves = state.get("move_history")
    if not isinstance(full_moves, list):
        full_moves = _reconstruct_move_history(observation)
    last_move = state.get("last_move") or (full_moves[-1] if full_moves else None)
    last_move_str = last_move or "(none yet)"
    my_piece = "o" if player_id == 0 else "x"
    last_file = chr(ord("a") + max(0, columns - 1))

    prompt = CLOBBER_PROMPT_TEMPLATE.format(
        rows=rows,
        columns=columns,
        last_file=last_file,
        board_ascii=_format_board_ascii(board, rows, columns),
        player_label=player_id,
        my_piece=my_piece,
        move_history_str=_format_move_history(full_moves),
        last_move=last_move_str,
    )

    prompt += render_rethink_suffix(
        RETHINK_ILLEGAL, RETHINK_UNPARSABLE,
        previous_response, previous_action,
    )

    return prompt


_NOTATION_NOISE_RE = re.compile(r"[\s\-x>]+", re.IGNORECASE)


def _normalize_move(token: str) -> str:
    return _NOTATION_NOISE_RE.sub("", token.lower())


def _match_move(raw: str, legal: Sequence[str]) -> str | None:
    """Tolerate the capture/coord separators models naturally write.

    Clobber is a capture game and the engine emits bare ``<from><to>``
    strings, but chess-trained models routinely add ``-`` (``a1-b1``),
    ``->`` (``a1->b1``), or the SAN capture marker ``x`` (``a1xb1``).
    Strip those and the surrounding whitespace before comparing.
    """
    target = _normalize_move(raw)
    if not target:
        return None
    for legal_str in legal:
        if _normalize_move(legal_str) == target:
            return legal_str
    return None


def parse_response(
    response: str, legal_action_strings: Sequence[str],
) -> ParseResult:
    """Trust the model's JSON answer; let the rethink loop fix anything else."""
    return parse_json_action(
        response, legal_action_strings, matcher=_match_move,
    )
