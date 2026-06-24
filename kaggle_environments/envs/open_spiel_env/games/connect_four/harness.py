"""LLM harness for OpenSpiel Connect Four.

Migrated from Google DeepMind's GameArena connect_four harness.
The prompt template is preserved exactly from the original.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult

# ---------------------------------------------------------------------------
# Prompt templates — exact copies from GameArena
# ---------------------------------------------------------------------------

CONNECT_X_PROMPT = """
You are a world-class Connect X AI. Your task is to analyze the current game state
and make the optimal move.

I. Game Rules & Configuration

A. Game Name: Connect X (Generalized Connect Four).
B. Board Size: The board has {rows} rows and {columns} columns.
C. Gravity: Disks fall to the lowest empty spot in a column.
D. Win Condition: The first player to get {in_a_row} of their pieces in a row (horizontally, vertically, or diagonally) wins.
E. Legal Moves: **You cannot put your piece in a column if the top row (Row 0) is occupied.**
F. Column Indices: Columns are 0-indexed from left to right (0 to {max_column_index}).
G. You are playing as player {player_name}.

II. Input Data Format

Current Board State:

{visual_board_state}

III. Required Final Answer Format

All responses MUST start with your **reasoning** and conclude with the final
answer.
The final answer MUST be on a single, final, new line.
The final answer line MUST be in the precise format:

Final Answer: <column_index>

Where <column_index> is a single integer representing the **zero-based column
index**.
Action is on you (Player {player_name}). Choose the optimal column.
{rethink_prompt}
""".strip()

CONNECTX_RETHINK = """
A legal action (a single integer column index) could not be parsed from your previous response.
Think carefully and respond with a legal, optimal column index.
Remember to include the final answer on the final line of your response.
It must EXACTLY follow the specified final answer format:
Final Answer: <column_index>

Your previous response concluded with:
{generation}
""".strip()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PLAYER_MAP = {0: "x", 1: "o"}
_FINAL_ANSWER_RE = re.compile(r"Final\s+Answer\s*:\s*(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


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
    serialized = observation.get("serializedGameAndState", "")
    _, state = pyspiel.deserialize_game_and_state(serialized)

    visual_board_state = state.to_string()
    params = state.get_game().get_parameters()
    rows = int(params.get("rows", 6))
    columns = int(params.get("columns", 7))
    in_a_row = int(params.get("x_in_row", 4))

    player_id = observation.get("playerId", 0)
    player_name = _PLAYER_MAP[player_id]

    rethink_prompt = ""
    if previous_response is not None:
        rethink_prompt = CONNECTX_RETHINK.format(
            generation=previous_response,
        )

    return CONNECT_X_PROMPT.format(
        rows=rows,
        columns=columns,
        in_a_row=in_a_row,
        visual_board_state=visual_board_state,
        player_name=player_name,
        max_column_index=columns - 1,
        rethink_prompt=rethink_prompt,
    )


def parse_response(
    response: str,
    legal_action_strings: Sequence[str],
) -> ParseResult:
    """Extract a legal Connect Four move from the model response.

    Multi-stage parser:
    1. Look for ``Final Answer: <column>``
    2. Scan for last digit in the response matching a legal column
    """
    # Stage 1: "Final Answer: <digit>" -- use the LAST occurrence (matching
    # GameArena's ``parse_move_from_response`` which uses ``rfind`` on the
    # action tag). Models that consider then revise their answer will
    # restate the final answer; the trailing one is the intent.
    matches = list(_FINAL_ANSWER_RE.finditer(response))
    match = matches[-1] if matches else None
    raw = match.group(1) if match else None
    if raw is not None:
        matched = _match_column_to_legal(raw, legal_action_strings)
        if matched is not None:
            return ParseResult(legal_action=matched, raw_action=raw)

    # Stage 2: scan for digits from the end of the response
    for digit_match in reversed(list(re.finditer(r"\d+", response))):
        column = digit_match.group()
        matched = _match_column_to_legal(column, legal_action_strings)
        if matched is not None:
            return ParseResult(legal_action=matched, raw_action=column)

    return ParseResult(raw_action=raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _match_column_to_legal(
    column: str,
    legal_action_strings: Sequence[str],
) -> str | None:
    """Match a column number string to a legal action string."""
    for legal in legal_action_strings:
        if legal.endswith(column):
            return legal
    return None


