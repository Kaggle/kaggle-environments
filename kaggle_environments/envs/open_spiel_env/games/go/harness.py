"""LLM harness for OpenSpiel Go.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult, parse_json_action

# --- Prompt ---


GO_PROMPT_TEMPLATE = """Let's play Go.

Rules: Tromp-Taylor scoring (area scoring — count stones on the board plus
empty territory enclosed by a single color; all stones are treated as alive).
Komi is given in the game state below. Two differences from standard
Tromp-Taylor: (1) suicide is illegal — you may not place a stone that would
be immediately captured unless it captures enemy stones first, and
(2) positional superko violations end the game as a draw rather than simply
making the move illegal. The game ends when both players pass consecutively.

The current game state is:
{state_str}
The moves played so far are:
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

Where move is the coordinate only (e.g. "a1", "b2", "e5") or "PASS" if you wish to pass.
Coordinates use GTP notation: columns are lowercase letters a-h, j (the letter
"i" is skipped to avoid confusion with "l"), rows are numbers starting from 1.
For example on a 9x9 board, columns are a-h,j and rows are 1-9.
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

For example: `{{"move": "a1"}}`

The move you choose must also be legal in the current state.
"""


# --- Helpers ----------------------------------------------------------------


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
    player_id = observation.get("playerId", 0)
    player_name = "Black" if player_id == 0 else "White"
    player_code = "B" if player_id == 0 else "W"

    move_history_str = " ".join(move_history) if move_history else "None"

    prompt = GO_PROMPT_TEMPLATE.format(
        state_str=obs_string,
        move_history=move_history_str,
        player_name=player_name,
        player_code=player_code,
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
    return parse_json_action(response, legal_action_strings, matcher=_match_move_to_legal)
