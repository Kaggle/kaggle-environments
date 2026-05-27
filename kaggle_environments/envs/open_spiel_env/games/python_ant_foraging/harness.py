"""LLM harness for the OpenSpiel Python Ant Foraging game.

The Kaggle agent script imports three module-level functions from here:
``get_legal_moves``, ``generate_prompt``, and ``parse_response``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{[^{}]*\"move\"\s*:\s*\"([^\"]+)\"[^{}]*\}", re.DOTALL)
# Plain direction words for the text fallback. Matches the action set
# {stay, up, down, left, right}.
_DIRECTION_RE = re.compile(r"\b(stay|up|down|left|right)\b", re.IGNORECASE)


# --- Prompt -----------------------------------------------------------------


ANT_PROMPT_TEMPLATE = """\
You are playing Ant Foraging, a cooperative grid game.

Rules:
- The world is a {grid_size}x{grid_size} grid. ``N`` marks the single nest;
  ``F`` marks each remaining food source; ``.`` is an empty cell.
- {num_ants} ants share one team score. The game ends when all {num_food}
  food items have been delivered to the nest, or after {max_turns} full
  rounds, whichever comes first.
- Each round every ant takes one turn in order (ant 0, then ant 1, ...).
  On its turn an ant picks one of {{stay, up, down, left, right}}. The
  move must stay on the board.
- Stepping onto an ``F`` cell automatically picks up that food (only one
  food at a time per ant). Returning to ``N`` while carrying drops it off
  and increases the team score by 1.
- Ants leave decaying pheromone trails. ``pheromone_to_food`` is laid
  near remembered food; ``pheromone_to_nest`` is laid by ants carrying
  food on the way home. Both grids decay each round, so fresh trails are
  more reliable than faint ones.

Coordinates: positions are ``[row, column]`` with ``row=0`` at the top and
``column=0`` on the left. ``up`` decreases row, ``down`` increases row,
``left`` decreases column, ``right`` increases column.

Action format: legal moves are written as ``ant<i>:<direction>``. Only
``ant{player_id}:...`` actions belong to you this turn -- choose a single
direction word and the framework will pair it with your ant id.

Current state (JSON):
{state_str}

Your player id (ant id) is {player_id}. Your ant is currently at
{your_position} and is {carry_status}. Score so far: {score} of
{num_food} food delivered, round {turn} of {max_turns}.

Moves taken so far this game: {move_history}

It is now your turn. Choose your move.
Your response should include your reasoning, then conclude with your
final move as JSON formatted exactly as follows:

```json
{{
  "move": "<stay|up|down|left|right>"
}}
```

Failure to output your final answer in this exact format will result in
a wasted turn.
Begin!
"""


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested move "{previous_action}" but this is not a legal move from
the current position. Reconsider the rules and the board, then pick a
legal move from {{stay, up, down, left, right}} that keeps your ant on
the board.
"""


# --- Helpers ----------------------------------------------------------------


def _parse_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    obs_str = observation.get("observationString", "")
    if not obs_str:
        return {}
    try:
        return json.loads(obs_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def _direction_only(action_string: str) -> str:
    """Strip the ``ant<i>:`` prefix from a legal action string.

    For ``ant0:up`` returns ``up``. If the action is already a bare
    direction, returns it unchanged.
    """
    if ":" in action_string:
        return action_string.split(":", 1)[1]
    return action_string


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
    legal_action_strings: Sequence[str],
) -> str | None:
    """Match the model's move (a bare direction or full ``ant<i>:dir``) to
    one of the supplied legal action strings.
    """
    target = _normalize(move)
    if not target:
        return None
    for legal in legal_action_strings:
        candidates = {_normalize(legal), _normalize(_direction_only(legal))}
        if target in candidates:
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
    player = state.current_player()
    return {a: state.action_to_string(player, a) for a in state.legal_actions()}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current game state."""
    parsed = _parse_observation(observation)
    player_id = observation.get("playerId", 0)

    grid_size = int(parsed.get("grid_size", 8))
    num_ants = int(parsed.get("num_ants", 2))
    num_food = int(parsed.get("num_food", 3))
    max_turns = int(parsed.get("max_turns", 50))
    turn = int(parsed.get("turn", 0))
    score = int(parsed.get("food_collected", parsed.get("score", 0)))

    ant_positions = parsed.get("ant_positions") or []
    carrying = parsed.get("carrying_food") or []
    if 0 <= player_id < len(ant_positions):
        your_position = ant_positions[player_id]
    else:
        your_position = "unknown"
    carry_status = (
        "carrying food back to the nest"
        if (0 <= player_id < len(carrying) and carrying[player_id])
        else "searching for food"
    )

    move_history_str = ", ".join(move_history) if move_history else "None"

    # Send the structured observation as-is — it already contains the grid,
    # food positions, ant positions, carrying state, and pheromones.
    state_str = json.dumps(parsed, indent=2) if parsed else (observation.get("observationString", ""))

    prompt = ANT_PROMPT_TEMPLATE.format(
        grid_size=grid_size,
        num_ants=num_ants,
        num_food=num_food,
        max_turns=max_turns,
        turn=turn,
        score=score,
        player_id=player_id,
        your_position=your_position,
        carry_status=carry_status,
        state_str=state_str,
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
    """Extract a legal Ant Foraging move from the model response.

    Accepts either a bare direction (``"up"``) or the fully-qualified
    legal action (``"ant0:up"``). Falls back to scanning the response
    text for the first direction keyword that matches a legal move.
    """
    raw = _extract_move_from_json(response)
    if raw is not None:
        matched = _match_move_to_legal(raw, legal_action_strings)
        if matched is not None:
            return ParseResult(legal_action=matched, raw_action=raw)

    # Fallback: scan the response text for any direction keyword that
    # corresponds to a legal action.
    for m in _DIRECTION_RE.finditer(response):
        candidate = m.group(0)
        matched = _match_move_to_legal(candidate, legal_action_strings)
        if matched is not None:
            return ParseResult(legal_action=matched, raw_action=raw or candidate)

    return ParseResult(legal_action=None, raw_action=raw)
