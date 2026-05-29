"""LLM harness for Ant Foraging Arena (2v2 team variant of Ant Foraging).

Drop the body of this file into the notebook attached to the competition
via HarnessKernelId. The auto-generated ``main.py`` calls these three
module-level functions: ``get_legal_moves``, ``generate_prompt``,
``parse_response``.

The arena observation is a per-player JSON view that includes only the
calling player's team's board, the player's ant position and carry
status, and the full move history on that board (so a player can see
what their teammate just played).
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult, parse_json_action, render_rethink_suffix


# --- Prompt -----------------------------------------------------------------


ARENA_PROMPT_TEMPLATE = """Let's play Ant Foraging Arena (2v2 cooperative ant foraging).

Setup: 2 teams of {players_per_team} ants each. Each team plays its own
private {grid_size}x{grid_size} grid in parallel (the other team's grid
is HIDDEN from you). Within your team, you and your teammate take turns
moving — on each step one seat moves on each board, in the order
[team A seat 0, team B seat 0, team A seat 1, team B seat 1, ...].

Important: every ant on your team is another instance of YOU (same
model, same submission), and the opposing team is two instances of a
single different agent. There is NO in-game communication, so you must
coordinate with your teammate purely by reasoning about what "another
copy of me" would do given the same board. The only thing that
distinguishes you from your teammate is your seat (and therefore the
order in which you move).

World: ``N`` marks the single nest in the centre; ``F`` marks each
remaining food source; ``.`` is an empty cell.

Actions: {{stay, up, down, left, right}}. Moves that would step off the
board are silently blocked (you stay put). Stepping onto an ``F`` cell
automatically picks up that food (one food at a time per ant). Returning
to ``N`` while carrying drops it off and increases your team's score
by 1.

Pheromones: ants leave decaying ``pheromone_to_food`` (near remembered
food) and ``pheromone_to_nest`` (laid while carrying). Both grids decay
each round, so fresh trails are more reliable than faint ones.

Scoring: your team's score = food delivered to your nest. Higher team
score wins. The game ends as soon as either team delivers all
{num_food} food items OR after {max_turns} rounds elapse.

Coordinates: positions are ``[row, column]`` with ``row=0`` at the top
and ``column=0`` on the left. ``up`` decreases row, ``down`` increases
row, ``left`` decreases column, ``right`` increases column.

Your team id is {team_id}. You are player {player_id} (seat {seat} on
your team's board). Your teammate is player {teammate_id} (seat
{teammate_seat}). Your ant is currently at {your_position} and is
{carry_status}. Team food so far: {food_collected} of {num_food}.

Current state of your team's board (JSON):
{board_str}

Move history on your team's board so far (most recent last):
{move_history_str}

It is now your turn. Choose your move.
The move MUST be one of: up, down, left, right, stay.
Your response should include the reasoning that led to your move, and
conclude with your final move as JSON formatted exactly as follows:

```json
{{
  "move": "<move>"
}}
```

Failure to output your final answer in the specified format will result
in a wasted turn.
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
{{"move": "<direction>"}}
```

For example: `{{"move": "up"}}`

The move you choose must also be legal in the current state.
"""


# --- Helpers ----------------------------------------------------------------


def _parse_obs(observation: Mapping[str, Any]) -> dict[str, Any]:
    obs_str = observation.get("observationString", "")
    if not obs_str:
        return {}
    try:
        return json.loads(obs_str)
    except (json.JSONDecodeError, TypeError):
        return {}


# --- Public functions (called by main.py) -----------------------------------


def get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str]:
    """Return ``{action_id: action_string}`` for the current state.

    Returns ``{}`` when this player has no legal actions (another seat
    is the active one this step).
    """
    legal_actions = observation.get("legalActions")
    legal_action_strings = observation.get("legalActionStrings")
    if legal_actions and legal_action_strings:
        return dict(zip(legal_actions, legal_action_strings))
    if legal_actions == [] or legal_action_strings == []:
        return {}
    serialized = observation.get("serializedGameAndState", "")
    if not serialized:
        return {}
    _, state = pyspiel.deserialize_game_and_state(serialized)
    player = observation.get("playerId", 0)
    actions = state.legal_actions(player)
    return {a: state.action_to_string(player, a) for a in actions}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current arena state.

    The ``move_history`` parameter (this player's own past moves, supplied
    by core_harness) is ignored; the arena observation already exposes
    the full per-board history including the teammate's plays.
    """
    del move_history
    obs = _parse_obs(observation)
    player_id = observation.get("playerId", obs.get("your_player_id", 0))
    team_id = obs.get("your_team_id", 0)
    seat = obs.get("your_seat", 0)
    players_per_team = int(obs.get("players_per_team", 2))
    board = obs.get("board", {})
    grid_size = int(board.get("grid_size", 8))
    num_food = int(board.get("num_food", 3))
    food_collected = int(board.get("food_collected", 0))
    max_turns = int(obs.get("max_turns", 50))

    teammate_seat = (seat + 1) % players_per_team
    teammate_id = team_id * players_per_team + teammate_seat

    ant_positions = board.get("ant_positions") or {}
    your_position = ant_positions.get(str(player_id), "unknown")
    carrying = board.get("carrying_food") or {}
    is_carrying = bool(carrying.get(str(player_id), False))
    carry_status = "carrying food back to the nest" if is_carrying else "searching for food"

    # Emit a compact subset of the board view to the model.
    board_view = {
        "grid": board.get("grid"),
        "nest_position": board.get("nest_position"),
        "food_positions": board.get("food_positions"),
        "ant_positions": board.get("ant_positions"),
        "carrying_food": board.get("carrying_food"),
        "pheromone_to_food": board.get("pheromone_to_food"),
        "pheromone_to_nest": board.get("pheromone_to_nest"),
        "moves_remaining": obs.get("moves_remaining"),
    }
    board_str = json.dumps(board_view, indent=2)

    history = board.get("move_history") or []
    if history:
        move_history_str = "\n".join(
            f"  move {idx + 1}: player {entry.get('player_id')} (seat {entry.get('seat')}) -> {entry.get('action')}"
            for idx, entry in enumerate(history)
        )
    else:
        move_history_str = "  (no moves yet)"

    prompt = ARENA_PROMPT_TEMPLATE.format(
        grid_size=grid_size,
        players_per_team=players_per_team,
        num_food=num_food,
        max_turns=max_turns,
        team_id=team_id,
        player_id=player_id,
        seat=seat,
        teammate_id=teammate_id,
        teammate_seat=teammate_seat,
        your_position=your_position,
        carry_status=carry_status,
        food_collected=food_collected,
        board_str=board_str,
        move_history_str=move_history_str,
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
