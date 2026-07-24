"""LLM harness for OpenSpiel Breakthrough.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``. The
game rules and action-notation conventions live in
``BREAKTHROUGH_PROMPT_TEMPLATE`` below -- the model sees them, so it's the
single source of truth.
"""

from __future__ import annotations

import json
import string
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import (
    ParseResult,
    parse_json_action,
    render_rethink_suffix,
)
from kaggle_environments.envs.open_spiel_env.games.breakthrough.breakthrough_proxy import (
    PIECE_BLACK,
    PIECE_WHITE,
)

# --- Prompt -----------------------------------------------------------------


BREAKTHROUGH_PROMPT_TEMPLATE = """Let's play Breakthrough on a {rows}x{columns} board.
Files {file_range} left-to-right; ranks 1-{rows} bottom-to-top.

Rules: Each turn move one of your pieces exactly one square forward:
straight (empty square only, NEVER a capture), forward-diagonal-left, or
forward-diagonal-right (each may land on an empty square OR capture an
adjacent opponent piece diagonally). No sideways, backward, or multi-square
moves. Captures are optional. Win by reaching the opponent's back rank OR
capturing all opponent pieces. No draws.

Notation: ``<from><to>`` for slides (``a7a6``), ``<from><to>*`` for diagonal
captures (``b2c3*`` = piece on b2 captures diagonally to c3).

Common illegal-move trap: a straight-forward slide can NEVER capture. If
the square directly ahead of your piece holds an opponent, you must either
approach it from a diagonal (a neighbouring file, one rank behind) or move
a different piece.

Board ('.' = empty, 'b' = Black, 'w' = White):
{board_ascii}

Pieces: Black='b' ({black_count}), White='w' ({white_count}).
You are Player {player_label} ('{my_piece}'), moving toward rank {forward_rank}.
Your pieces: {my_squares}
Opponent pieces: {opp_squares}

Move number: {move_number}
Last move: {last_move}
Full move history (both players, oldest first): {move_history}

Respond with your reasoning, then your final move in a JSON block:

```json
{{"move": "<from><to>"}}
```

Examples: `{{"move": "a7a6"}}` (slide), `{{"move": "b2c3*"}}` (capture).

Failure to output a legal move in this format results in a loss.
"""


RETHINK_ILLEGAL = """

You suggested move "{previous_action}" but this is not a legal move.
Reconsider the rules and the current board state, then pick a legal move.
Remember: straight-forward moves cannot capture, diagonal moves must land
on an empty square or capture an opponent piece, and a diagonal capture
must be written with a trailing ``*`` (e.g. ``b2c3*``).

(Keep using the same JSON output format as before -- only the move value
needs to change.)
"""

RETHINK_UNPARSABLE = """

Your previous response ended with:
{previous_response}

No JSON answer could be parsed. Conclude your response with the final move
as JSON in a ```json fenced block:

```json
{{"move": "<from><to>"}}
```

Examples: `{{"move": "a7a6"}}` (slide) or `{{"move": "b2c3*"}}` (capture).
The move must also be legal in the current state.
"""


# --- Helpers ----------------------------------------------------------------


def _parse_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the structured breakthrough state dict out of the observation."""
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
    """Render the board with rank labels on the left and files on top.

    The proxy returns ``board[0]`` as the top visual row (rank == rows),
    descending to ``board[rows-1]`` as rank 1. We print the rows in the
    same order so the visual orientation matches the proxy.
    """
    if not board:
        return "(unavailable)"
    rows = len(board)
    cols = len(board[0]) if board[0] else 0
    file_header = "  " + " ".join(chr(ord("a") + c) for c in range(cols))
    lines = [file_header]
    for r in range(rows):
        rank_label = rows - r
        lines.append(f"{rank_label:>2} " + " ".join(board[r]))
    return "\n".join(lines)


def _list_player_squares(board: Sequence[Sequence[str]], piece_char: str) -> list[str]:
    """Return algebraic squares (e.g. 'a7') holding ``piece_char`` pieces.

    ``board[0]`` is the top visual row (rank == len(board)).
    """
    rows = len(board)
    squares: list[str] = []
    for r, row in enumerate(board):
        for c, cell in enumerate(row):
            if cell == piece_char:
                squares.append(f"{chr(ord('a') + c)}{rows - r}")
    return squares


def _normalize_move(raw: str) -> str:
    """Lowercase, strip whitespace, and remove obvious wrappers."""
    s = raw.strip().lower()
    # Strip surrounding quotes/brackets and trailing punctuation a model
    # might add (e.g. `{"move": "a7a6."}` or `{"move": "a7a6,"}`).
    s = s.strip("`'\"<>[](){}.,!? \t\n")
    # Remove any internal whitespace ("a7 a6" or "a7\ta6") that some models
    # insert between the from/to squares.
    s = "".join(s.split())
    # Some models write moves with separators OpenSpiel doesn't use: a dash
    # ("a7-a6"), an 'x' ("b2xc3"), or an arrow ("a7->a6"). Drop them so the
    # from/to squares concatenate. "->" first so the arrow is removed as a
    # unit rather than leaving a stray ">".
    s = s.replace("->", "").replace("-", "").replace("x", "")
    return s


def _match_move_to_legal(raw: str, legal_action_strings: Sequence[str]) -> str | None:
    """Match ``raw`` to a legal action string, tolerating common drift.

    Models routinely (a) drop the trailing ``*`` capture marker or
    (b) add one to a non-capture. Try the literal normalization first,
    then try toggling the trailing ``*``.
    """
    if not legal_action_strings:
        return None
    legal_set = set(legal_action_strings)
    candidate = _normalize_move(raw)
    if candidate in legal_set:
        return candidate
    # Try adding a trailing '*' (model forgot the capture marker).
    if not candidate.endswith("*") and f"{candidate}*" in legal_set:
        return f"{candidate}*"
    # Try removing a trailing '*' (model added one to a non-capture).
    if candidate.endswith("*") and candidate[:-1] in legal_set:
        return candidate[:-1]
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
    _move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current breakthrough state.

    Ignores the framework-provided per-agent ``_move_history`` -- the proxy
    surfaces a full both-player ``move_history`` in the state payload, which
    is what the model needs to reason about the position.
    """
    state = _parse_observation_payload(observation)
    player_id = observation.get("playerId", 0)

    board = state.get("board") or []
    move_number = state.get("move_number", 0)
    last_move = state.get("last_move") or "(none yet)"
    pieces = state.get("pieces") or {}
    full_history = state.get("move_history") or []

    my_piece = PIECE_BLACK if player_id == 0 else PIECE_WHITE
    opp_piece = PIECE_WHITE if player_id == 0 else PIECE_BLACK

    my_squares_list = _list_player_squares(board, my_piece)
    opp_squares_list = _list_player_squares(board, opp_piece)
    my_squares = ", ".join(my_squares_list) if my_squares_list else "(none)"
    opp_squares = ", ".join(opp_squares_list) if opp_squares_list else "(none)"

    move_history_str = ", ".join(full_history) if full_history else "None"

    # Derive board dimensions from the live state so the prompt stays
    # accurate if the game is loaded with a non-default `rows`/`columns`.
    params = state.get("params") or {}
    rows = int(params.get("rows") or state.get("rows") or len(board) or 8)
    columns = int(
        params.get("columns")
        or state.get("columns")
        or (len(board[0]) if board else 8)
    )
    if columns <= 0 or columns > 26:
        columns = max(1, min(26, columns or 8))
    file_letters = string.ascii_lowercase[:columns]
    file_range = f"{file_letters[0]}-{file_letters[-1]}" if columns > 1 else file_letters

    # Black ('b') moves toward rank 1; White ('w') moves toward rank `rows`.
    forward_rank = 1 if player_id == 0 else rows

    prompt = BREAKTHROUGH_PROMPT_TEMPLATE.format(
        board_ascii=_format_board_ascii(board),
        black_count=pieces.get(PIECE_BLACK, 0),
        white_count=pieces.get(PIECE_WHITE, 0),
        player_label=player_id,
        my_piece=my_piece,
        opp_piece=opp_piece,
        my_squares=my_squares,
        opp_squares=opp_squares,
        forward_rank=forward_rank,
        move_number=move_number,
        last_move=last_move,
        move_history=move_history_str,
        rows=rows,
        columns=columns,
        file_range=file_range,
    )

    prompt += render_rethink_suffix(
        RETHINK_ILLEGAL,
        RETHINK_UNPARSABLE,
        previous_response,
        previous_action,
    )

    return prompt


def parse_response(response: str, legal_action_strings: Sequence[str]) -> ParseResult:
    """Trust the model's JSON answer; let the rethink loop fix anything else."""
    return parse_json_action(
        response,
        legal_action_strings,
        matcher=_match_move_to_legal,
    )
