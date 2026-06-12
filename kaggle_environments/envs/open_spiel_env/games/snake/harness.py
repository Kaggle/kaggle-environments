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

from kaggle_environments.core_harness import ParseResult, parse_json_action, render_rethink_suffix

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
its mirror through the board center). Every {food_respawn_interval}
turns a fresh pair is spawned and any uneaten food on the board is
removed -- there is NO respawn when food is eaten. Snakes that do NOT
eat lose their tail on the same turn. The game ends when at most one
snake is alive, or when the board has no room left for a new food pair.

Your goal is to maximize your food score (1 point per food eaten).
There is NO bonus for being the last snake standing -- surviving only
ends the game; it does not boost your score.

Coordinates are ``[row, column]`` with ``row=0`` at the top and
``column=0`` on the left. The board uses these characters:
  ``.`` empty, ``*`` food, lowercase letter ({your_body_char}/etc.) snake
  body, uppercase letter ({your_head_char}/etc.) snake head. Your snake
  uses letter "{your_letter}".

Current board:
{board_str}

You are player {player_id}. Your snake body is at {your_body}{alive_note}
(the first coordinate is your head).
Your score: {your_score}. Food at: {food_str}.
Next food respawn in {turns_until_respawn} turn(s).

Recent rounds (most recent last; one line per simultaneous round):
{round_history_str}

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
{{"move": "<DIRECTION>"}}
```

For example: `{{"move": "UP"}}`

The move you choose must also be legal in the current state.
"""


# --- Helpers ----------------------------------------------------------------


_RECENT_ROUNDS_LIMIT = 10


def _parse_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Parse the snake proxy's JSON observation, returning ``{}`` on error."""
    obs_str = observation.get("observationString", "")
    if not obs_str:
        return {}
    try:
        return json.loads(obs_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def _render_board(board: list[list[str]] | None) -> str:
    """Render the proxy's 2D board array as a single ASCII grid."""
    if not board:
        return "(no board)"
    return "\n".join("".join(row) for row in board)


def _render_round_history(
    round_history: list[list[str | None]] | None,
    num_players: int,
) -> str:
    """Render the per-round action log shared by all players.

    Lines are 1-indexed by round number, capped at the most recent
    ``_RECENT_ROUNDS_LIMIT`` rounds so the prompt doesn't grow without
    bound. ``None`` in a slot means the player didn't supply a move that
    round (e.g. they were already dead).
    """
    if not round_history:
        return "(no moves yet)"
    total = len(round_history)
    recent = round_history[-_RECENT_ROUNDS_LIMIT:]
    start_idx = total - len(recent) + 1
    lines = []
    for i, round_moves in enumerate(recent):
        round_num = start_idx + i
        parts = [
            f"P{p}={(round_moves[p] if p < len(round_moves) else None) or '-'}"
            for p in range(num_players)
        ]
        lines.append(f"Round {round_num}: " + ", ".join(parts))
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
    actions = state.legal_actions()
    return {a: state.action_to_string(a) for a in actions}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current snake game state."""
    del move_history  # We render the proxy's full per-round history instead.
    player_id = int(observation.get("playerId", 0))
    parsed = _parse_observation(observation)

    rows = int(parsed.get("num_rows", 10))
    cols = int(parsed.get("num_columns", 10))
    num_players = int(parsed.get("num_players", 2))

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
    alive_note = "" if alive else " (DEAD -- you are out of the game)"

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

    board_str = _render_board(parsed.get("board"))
    round_history_str = _render_round_history(
        parsed.get("round_history"), num_players,
    )

    prompt = SNAKE_PROMPT_TEMPLATE.format(
        rows=rows,
        cols=cols,
        num_players=num_players,
        food_respawn_interval=food_respawn_interval,
        turns_until_respawn=turns_until_respawn,
        your_letter=your_letter,
        your_body_char=your_body_char,
        your_head_char=your_head_char,
        board_str=board_str,
        player_id=player_id,
        your_body=your_body,
        your_score=your_score,
        alive_note=alive_note,
        food_str=food_str,
        round_history_str=round_history_str,
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
    return parse_json_action(response, legal_action_strings)
