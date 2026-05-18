"""LLM harness for the OpenSpiel Coin Game.

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
# Coin Game action names: up | down | left | right | stand.
_MOVE_RE = re.compile(r"\b(up|down|left|right|stand)\b", re.IGNORECASE)

_VALID_MOVES = ("up", "down", "left", "right", "stand")


# --- Prompt -----------------------------------------------------------------


COIN_PROMPT_TEMPLATE = """Let's play the Coin Game.

Rules: {rows}x{cols} grid. Each player has been privately assigned a
preferred coin colour (you only know your own). On each turn a player
picks one of {{up, down, left, right, stand}}; moving onto a coin
collects it. The game lasts {episode_length} moves total (alternating
between players). At the end, your reward is

    self_pref^2 + other_pref^2 - bad_coins^2

where ``self_pref`` counts coins of YOUR preferred colour you collected,
``other_pref`` counts coins of OTHER players' preferences collected by
ANYONE, and ``bad_coins`` are coins of unowned colours collected by
anyone. So picking up your own preference is good; ignoring colours that
are nobody's preference is best (since collecting them hurts everyone).

Coordinates are ``[row, column]`` with ``row=0`` at the top. Cells are
``"."`` for empty, digits for players, lowercase letters for coin
colours.

The current game state is:
{state_str}

Your player id is {player_id}. Your preferred coin colour is "{your_pref}".
The moves played so far are:
{move_history}

It is now your turn. Choose your move.
The move MUST be one of: up, down, left, right, stand.
Your response should include the reasoning that led you to your move,
and conclude with your final move as a JSON formatted as follows:

```json
{{
  "move": "<move>"
}}
```

Failure to output your final answer in the specified format will result
in a loss.
Begin!
"""


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested move "{previous_action}" but this is not in the legal
moves list. Reconsider and play a legal move from {{up, down, left,
right, stand}}.
"""


# --- Helpers ----------------------------------------------------------------


def _normalize(move: str) -> str:
    return re.sub(r"\s+", "", move).lower()


def _extract_move_from_json(response: str) -> str | None:
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
    """Match ``move`` against the legal-move list, ignoring case/whitespace."""
    target = _normalize(move)
    if not target:
        return None
    legal_normalized = {_normalize(legal): legal for legal in legal_moves}
    return legal_normalized.get(target)


def _extract_preference(observation: Mapping[str, Any]) -> str:
    """Pull the player's private preference out of the observation JSON."""
    obs_str = observation.get("observationString", "")
    if not obs_str:
        return "?"
    try:
        parsed = json.loads(obs_str)
    except (json.JSONDecodeError, TypeError):
        return "?"
    return str(parsed.get("your_preference", "?"))


def _extract_dims(observation: Mapping[str, Any]) -> tuple[int, int, int]:
    """Return ``(rows, cols, episode_length)`` from the observation JSON."""
    obs_str = observation.get("observationString", "")
    if not obs_str:
        return (8, 8, 20)
    try:
        parsed = json.loads(obs_str)
    except (json.JSONDecodeError, TypeError):
        return (8, 8, 20)
    return (
        int(parsed.get("num_rows", 8)),
        int(parsed.get("num_columns", 8)),
        int(parsed.get("episode_length", 20)),
    )


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
    your_pref = _extract_preference(observation)
    rows, cols, episode_length = _extract_dims(observation)

    move_history_str = " ".join(move_history) if move_history else "None"

    prompt = COIN_PROMPT_TEMPLATE.format(
        rows=rows,
        cols=cols,
        episode_length=episode_length,
        state_str=obs_string,
        player_id=player_id,
        your_pref=your_pref,
        move_history=move_history_str,
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
    """Extract a legal Coin Game move from the model response.

    Tries to extract the move from a JSON block first, then falls back to
    scanning the response text for the first action keyword.
    """
    raw = _extract_move_from_json(response)
    if raw is not None:
        matched = _match_move_to_legal(raw, legal_action_strings)
        if matched is not None:
            return ParseResult(legal_action=matched, raw_action=raw)

    # Fallback: scan the response text for any action keyword.
    for m in _MOVE_RE.finditer(response):
        candidate = m.group(0)
        matched = _match_move_to_legal(candidate, legal_action_strings)
        if matched is not None:
            return ParseResult(legal_action=matched, raw_action=raw or candidate)

    return ParseResult(legal_action=None, raw_action=raw)
