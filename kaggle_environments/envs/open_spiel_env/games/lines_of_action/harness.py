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

from kaggle_environments.core_harness import ParseResult

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{[^{}]*\"move\"\s*:\s*\"([^\"]+)\"[^{}]*\}", re.DOTALL)
# Lines of Action notation: "<file><rank><sep><file><rank>" where sep is '-' (move)
# or 'x' (capture). Files are a-h, ranks are 1-8.
_MOVE_RE = re.compile(r"\b([a-h])([1-8])\s*([-x])\s*([a-h])([1-8])\b", re.IGNORECASE)


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
neighbours count as connected). If your move connects both your pieces and
your opponent's pieces simultaneously, your opponent wins. There are no
draws under normal play.

The current game state is:
{state_str}
The moves played so far are:
{move_history}
You are playing as {player_name} ({player_code}).
It is now your turn. Play your strongest move.
The move MUST be legal.
Your response should include the reasoning that led you to your move, and
conclude with your final move as a JSON formatted as follows:

```json
{{
  "move": "<move>"
}}
```

Where move uses the notation "<from><sep><to>" -- for example "b1-h1" for
a non-capturing move from b1 to h1, or "c3xa3" for a capture moving from
c3 onto an opponent piece at a3. Use '-' for a normal move and 'x' for a
capture. Files are lowercase a-h; ranks are 1-8.

Failure to output your final answer in the specified format will result in
a loss.
Begin!
"""


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested move "{previous_action}" but this is not in the legal moves list.
Reconsider and play a legal move.
"""


# --- Helpers ----------------------------------------------------------------


def _normalize(move: str) -> str:
    """Strip whitespace and lowercase a candidate move string."""
    return re.sub(r"\s+", "", move).lower()


def _extract_move_from_json(response: str) -> str | None:
    """Try to extract a move string from a JSON code block or bare JSON."""
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
    obs_string = observation.get("observationString", "")
    player_id = observation.get("playerId", 0)
    player_name = "Black" if player_id == 0 else "White"
    player_code = "X" if player_id == 0 else "O"

    move_history_str = " ".join(move_history) if move_history else "None"

    prompt = LOA_PROMPT_TEMPLATE.format(
        state_str=obs_string,
        move_history=move_history_str,
        player_name=player_name,
        player_code=player_code,
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
    """Extract a legal Lines of Action move from the model response.

    Tries to extract a move from a JSON block first, then falls back to
    scanning the response for a coordinate pair like ``b1-h1`` or ``c3xa3``.
    """
    raw = _extract_move_from_json(response)
    if raw is not None:
        matched = _match_move_to_legal(raw, legal_action_strings)
        if matched is not None:
            return ParseResult(legal_action=matched, raw_action=raw)

    # Fallback: scan the response text for any move-shaped token and try it.
    for m in _MOVE_RE.finditer(response):
        candidate = m.group(0)
        matched = _match_move_to_legal(candidate, legal_action_strings)
        if matched is not None:
            return ParseResult(legal_action=matched, raw_action=raw or candidate)

    return ParseResult(legal_action=None, raw_action=raw)
