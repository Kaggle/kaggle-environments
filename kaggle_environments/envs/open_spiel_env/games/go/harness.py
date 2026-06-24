"""LLM harness for OpenSpiel Go.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.
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

_GTP_COLUMNS = "abcdefghjklmnopqrstuvwxyz"

# --- Prompt ---


GO_PROMPT_TEMPLATE = """Let's play Go.

Rules: Tromp-Taylor scoring (area scoring — count stones on the board plus
empty territory enclosed by a single color; all stones are treated as alive).
Komi is given in the game state JSON below. Suicide is illegal: you may not
place a stone that would be immediately captured unless it captures enemy
stones first. Immediate single-stone ko recapture is illegal: after a move
captures exactly one enemy stone in a ko shape, the opponent cannot play on
the point vacated by that captured stone on the very next move. A legal
non-pass move that repeats an earlier board position ends the game in a draw.
The game ends when both players pass consecutively.

The current game state JSON is:
{state_str}

ASCII board for the same position (X=Black, O=White, +=empty; board labels may be uppercase):
{ascii_board}

The full game move history is:
{move_history}

You are playing as player {player_name} ({player_code}).
It is now your turn. Play your strongest move.
The move MUST be legal.
Your response should include the reasoning that led you to your move, and
conclude with your final move as a JSON formatted as follows:

```json
{{
  "move": "<move>"
}}
```

Where move is the coordinate only (e.g. "A1", "B2", "E5") or "PASS" if you wish to pass.
{coordinate_guidance}
The final JSON move must be the coordinate only, without the player prefix.
Failure to output your final answer in the specified format will result in a loss.
Begin!
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
{{"move": "<coordinate>"}}
```

For example: `{{"move": "A1"}}`

The move you choose must also be legal in the current state.
"""


# --- Helpers ----------------------------------------------------------------


def _parse_state(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Parse the proxy JSON observation, returning ``{}`` on failure."""
    raw = observation.get("observationString", "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _gtp_column_range(board_size: int | None) -> str:
    """Return compact GTP column guidance for a square Go board."""
    if board_size is None or board_size <= 0:
        return "the columns shown in the board state"
    columns = _GTP_COLUMNS[:board_size]
    if board_size <= 8:
        return f"A-{columns[-1].upper()}"
    if board_size == 9:
        return "A-H,J"
    return f"A-H,J-{columns[-1].upper()}"


def _coordinate_guidance(board_size: int | None) -> str:
    if board_size is None or board_size <= 0:
        return (
            "Coordinates use GTP notation: column letters shown in "
            'the board state, skipping "i", followed by row numbers starting '
            'from 1.'
        )
    return (
        f"Coordinates use GTP notation for this {board_size}x{board_size} board: "
        f"columns are {_gtp_column_range(board_size)} (the letter \"I\" is "
        f"skipped), and rows are 1-{board_size}."
    )


def _board_size_from_state(state: Mapping[str, Any]) -> int | None:
    board_size = state.get("board_size")
    if isinstance(board_size, int):
        return board_size
    if isinstance(board_size, str) and board_size.isdigit():
        return int(board_size)
    board_grid = state.get("board_grid")
    if isinstance(board_grid, list) and board_grid:
        return len(board_grid)
    return None


def _format_full_move_history(state: Mapping[str, Any]) -> str:
    history = state.get("move_history")
    if not history:
        return "None"
    if isinstance(history, list):
        return " ".join(str(move) for move in history) or "None"
    return str(history)


def _ascii_board_from_state(state: Mapping[str, Any]) -> str:
    board = state.get("ascii_board")
    if isinstance(board, str) and board.strip():
        return board
    return "Not available in this observation."


def _match_move_to_legal(
    move: str,
    legal_moves: Sequence[str],
) -> str | None:
    """Match a move string (e.g. "e5", "PASS") to a legal move string."""
    move_lower = move.lower()

    if move_lower == "pass":
        for legal in legal_moves:
            if legal.upper().endswith("PASS"):
                return legal
        return None

    for legal in legal_moves:
        parts = legal.split()
        if len(parts) == 2 and parts[1].lower() == move_lower:
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
    obs_string = observation.get("observationString", "")
    state = _parse_state(observation)
    board_size = _board_size_from_state(state)
    player_id = observation.get("playerId", 0)
    player_name = "Black" if player_id == 0 else "White"
    player_code = "B" if player_id == 0 else "W"

    del move_history

    prompt = GO_PROMPT_TEMPLATE.format(
        state_str=obs_string,
        ascii_board=_ascii_board_from_state(state),
        move_history=_format_full_move_history(state),
        player_name=player_name,
        player_code=player_code,
        coordinate_guidance=_coordinate_guidance(board_size),
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
