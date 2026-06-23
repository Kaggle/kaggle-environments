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


BREAKTHROUGH_PROMPT_TEMPLATE = """Let's play Breakthrough.

Rules: 8x8 board with files a-h (left-to-right) and ranks 1-8
(bottom-to-top). Player 0 ('b', Black) starts on ranks 7-8 and moves
first toward rank 1. Player 1 ('w', White) starts on ranks 1-2 and moves
toward rank 8. Each turn a player moves exactly one of their own pieces
one square in one of three forward directions:

- straight forward (into an empty square),
- forward-diagonal-left (into an empty square OR onto an opposing piece,
  capturing it),
- forward-diagonal-right (into an empty square OR onto an opposing piece,
  capturing it).

Straight-forward moves may NEVER capture. Diagonal moves capture by
displacing the opposing piece, which is removed from the board. Pieces
never move sideways or backward, and there is no en-passant or any other
special rule. Captures are NOT mandatory.

Win conditions (no draws are possible):
- reach the opponent's back rank with any one of your pieces (rank 1 for
  Black, rank 8 for White); OR
- capture all of the opponent's pieces (leaving them with none).

Board (rank labels on the left, file labels on top; '.' = empty,
'b' = Black piece, 'w' = White piece):
{board_ascii}

Piece counts: Black ('b') = {black_count}, White ('w') = {white_count}.

You are Player {player_label} ('{my_piece}'); the opponent is '{opp_piece}'.
Your pieces are at: {my_squares}
Opponent pieces are at: {opp_squares}
"Forward" for you means toward rank {forward_rank} (your goal rank).

Move number: {move_number}
Last move played: {last_move}
Moves played so far this game (both players, oldest first): {move_history}

Action notation: a 4-character string ``<from><to>`` denotes "the piece at
<from> moves to <to>", where each square is lowercase file+rank (e.g.
``a7`` or ``e4``). When the move is a diagonal capture, append a single
``*`` to mark it as a capture, giving a 5-character string such as
``b2c3*``. Examples: ``a7a6`` means "the piece on a7 slides straight
forward to a6" (only legal when a6 is empty); ``b2c3*`` means "the White
piece on b2 captures the Black piece on c3 diagonally". The from-square
must hold one of your own pieces and the to-square must be adjacent and
in one of your three forward directions.

It is your turn. Choose one legal move.

Respond with your reasoning followed by your final move in a JSON block:

```json
{{
  "move": "<from><to>"
}}
```

For example: `{{"move": "a7a6"}}` (slide) or `{{"move": "b2c3*"}}` (capture).

Failure to output your final answer in the specified format, or selecting
an illegal move, will result in a loss.
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

No JSON answer could be parsed from that. Conclude your response with
your final move as JSON in a ```json fenced block, exactly as the original
instructions required:

```json
{{"move": "<from><to>"}}
```

For example: `{{"move": "a7a6"}}` (slide) or `{{"move": "b2c3*"}}` (capture).

The move you choose must also be legal in the current state.
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
    # Strip surrounding quotes/brackets a model might add.
    s = s.strip("`'\"<>[](){} ")
    # Some models write moves with a dash ("a7-a6") or an 'x' ("b2xc3").
    # OpenSpiel uses neither; drop them so the from/to squares concatenate.
    s = s.replace("-", "").replace("x", "")
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

    # Black ('b') moves toward rank 1; White ('w') moves toward rank 8.
    forward_rank = 1 if player_id == 0 else 8

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
