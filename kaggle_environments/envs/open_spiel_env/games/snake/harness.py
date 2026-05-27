"""LLM harness for the OpenSpiel Snake game.

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
_BARE_JSON_RE = re.compile(r'\{[^{}]*"move"\s*:\s*"([^"]+)"[^{}]*\}', re.DOTALL)
# Snake action names are uppercase: UP | DOWN | LEFT | RIGHT.
_MOVE_RE = re.compile(r"\b(up|down|left|right)\b", re.IGNORECASE)


# --- Prompt -----------------------------------------------------------------


SNAKE_PROMPT_TEMPLATE = """Let's play Snake.

Rules: {rows}x{cols} grid with {num_players} snakes moving simultaneously.
Each turn every snake picks one of {{UP, DOWN, LEFT, RIGHT}} and all
snakes move one cell in the chosen direction at the same time. A snake
dies if its new head:
  - leaves the grid,
  - lands on any snake's post-move body (including its own), or
  - collides head-to-head with another snake (both die).
A snake that moves onto a food cell ("*") grows by one and earns one
point. Food spawns in 180°-rotationally-symmetric pairs (one cell and
its mirror through the board center), so both starting positions are
equidistant from the nearest food. Every {food_respawn_interval} turns
a fresh pair is spawned and any uneaten food is removed — there is no
respawn when food is eaten. Snakes that do NOT eat lose their tail on
the same turn. The game ends when at most one snake is alive (in a
multi-player game) or after {max_turns} turns; the last snake standing
wins, otherwise the highest score wins.

Coordinates are ``[row, column]`` with ``row=0`` at the top and
``column=0`` on the left. The board uses these characters:
  ``.`` empty, ``*`` food, lowercase letter ({your_body_char}/etc.) snake
  body, uppercase letter ({your_head_char}/etc.) snake head. Your snake
  uses letter "{your_letter}".

The current game state is:
{state_str}

You are player {player_id}. Your snake is at {your_body}{alive_note}.
Your score: {your_score}. Food at: {food_str}.
Next food respawn in {turns_until_respawn} turn(s).

Moves you have played so far:
{move_history}

It is now your turn. Choose your move.
The move MUST be one of: UP, DOWN, LEFT, RIGHT.
Your response should include the reasoning that led you to your move,
and conclude with your final move as a JSON formatted as follows:

```json
{{
  "move": "<UP|DOWN|LEFT|RIGHT>"
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
moves list. Reconsider and play a legal move from {{UP, DOWN, LEFT,
RIGHT}}.
"""


# --- Helpers ----------------------------------------------------------------


def _normalize(move: str) -> str:
    return re.sub(r"\s+", "", move).upper()


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


def _match_move_to_legal(move: str, legal_moves: Sequence[str]) -> str | None:
    """Match ``move`` against the legal-move list, ignoring case/whitespace."""
    target = _normalize(move)
    if not target:
        return None
    legal_normalized = {_normalize(legal): legal for legal in legal_moves}
    return legal_normalized.get(target)


def _parse_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Parse the snake proxy's JSON observation, returning ``{}`` on error."""
    obs_str = observation.get("observationString", "")
    if not obs_str:
        return {}
    try:
        return json.loads(obs_str)
    except (json.JSONDecodeError, TypeError):
        return {}


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
    """Build the LLM prompt for the current snake game state."""
    obs_string = observation.get("observationString", "")
    player_id = int(observation.get("playerId", 0))
    parsed = _parse_observation(observation)

    rows = int(parsed.get("num_rows", 10))
    cols = int(parsed.get("num_columns", 10))
    num_players = int(parsed.get("num_players", 2))
    max_turns = rows * cols * 2

    body_chars = ["a", "b", "c", "d"]
    head_chars = ["A", "B", "C", "D"]
    your_letter = body_chars[player_id % len(body_chars)]
    your_body_char = body_chars[player_id % len(body_chars)]
    your_head_char = head_chars[player_id % len(head_chars)]

    snakes = parsed.get("snakes") or []
    your_snake = next((s for s in snakes if int(s.get("player", -1)) == player_id), None)
    your_body = your_snake["body"] if your_snake else "(unknown)"
    your_score = your_snake["score"] if your_snake else 0
    alive = bool(your_snake.get("alive", True)) if your_snake else True
    alive_note = "" if alive else " (DEAD — you are out of the game)"

    foods = parsed.get("foods")
    if foods is None:
        # Back-compat: older proxies emitted a single "food" key.
        single = parsed.get("food")
        foods = [single] if single else []
    food_str = ", ".join(str(f) for f in foods) if foods else "(no food on board)"

    food_respawn_interval = int(parsed.get("food_respawn_interval") or 10)
    turn = int(parsed.get("turn", 0))
    turns_until_respawn = parsed.get("turns_until_respawn")
    if turns_until_respawn is None and food_respawn_interval > 0:
        turns_until_respawn = food_respawn_interval - (turn % food_respawn_interval)

    move_history_str = " ".join(move_history) if move_history else "None"

    prompt = SNAKE_PROMPT_TEMPLATE.format(
        rows=rows,
        cols=cols,
        num_players=num_players,
        max_turns=max_turns,
        food_respawn_interval=food_respawn_interval,
        turns_until_respawn=turns_until_respawn,
        your_letter=your_letter,
        your_body_char=your_body_char,
        your_head_char=your_head_char,
        state_str=obs_string,
        player_id=player_id,
        your_body=your_body,
        your_score=your_score,
        alive_note=alive_note,
        food_str=food_str,
        move_history=move_history_str,
    )

    if previous_response is not None:
        prompt += RETHINK_SUFFIX.format(
            previous_response=previous_response[:500],
            previous_action=previous_action or "(could not parse)",
        )

    return prompt


def parse_response(
    response: str,
    legal_action_strings: Sequence[str],
) -> ParseResult:
    """Extract a legal Snake move from the model response.

    Tries to extract the move from a JSON block first, then falls back to
    scanning the response text for the first direction keyword.
    """
    raw = _extract_move_from_json(response)
    if raw is not None:
        matched = _match_move_to_legal(raw, legal_action_strings)
        if matched is not None:
            return ParseResult(legal_action=matched, raw_action=raw)

    for m in _MOVE_RE.finditer(response):
        candidate = m.group(0)
        matched = _match_move_to_legal(candidate, legal_action_strings)
        if matched is not None:
            return ParseResult(legal_action=matched, raw_action=raw or candidate)

    return ParseResult(legal_action=None, raw_action=raw)
