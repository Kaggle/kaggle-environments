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
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult, parse_json_action, render_rethink_suffix


# --- Prompt -----------------------------------------------------------------


ARENA_PROMPT_TEMPLATE = """Let's play Ant Foraging Arena (2v2 cooperative ant foraging).

Setup: 2 teams of {players_per_team} ants each. Each team plays its own
private {grid_size}x{grid_size} grid in parallel; the other team's board
is HIDDEN from you. Within your team, seat 0 moves first, then seat 1.

Important: every ant on your team is another instance of YOU (same
model, same submission), and the opposing team is two instances of a
single different agent. There is NO in-game communication, so you must
coordinate with your teammate purely by reasoning about what "another
copy of me" would do given the same board.

World: ``N`` marks the single nest in the centre; ``F`` marks each
remaining food source; ``.`` is an empty cell.

Actions: {{stay, up, down, left, right}}. Off-board moves are not legal
-- if your ant is on an edge, your legal set excludes any direction
that would step off the board (``stay`` is always legal). Stepping
onto an ``F`` cell automatically picks up that food (one food at a
time per ant). Returning to ``N`` while carrying drops it off and
increases your team's score by 1.

Pheromones: ants leave decaying ``to_food`` (near remembered food) and
``to_nest`` (laid while carrying) trails. Both decay each round, so
fresh trails are more reliable than faint ones.

Scoring: your team's score = food delivered to your nest. Higher team
score wins. The game ends as soon as either team delivers all
{num_food} food items OR after {max_turns} rounds elapse (where a round
is one move per ant per team).

Coordinates: positions are ``[row, column]`` with ``row=0`` at the top
and ``column=0`` on the left. ``up`` decreases row, ``down`` increases
row, ``left`` decreases column, ``right`` increases column.

Your team id is {team_id}. You are player {player_id} (seat {seat} on
your team's board). Your teammate is player {teammate_id} (seat
{teammate_seat}). Your ant is currently at {your_position} and is
{carry_status}. Team food so far: {food_collected} of {num_food}.
Game progress: round {current_round} of {max_turns}.

Your team's board (terrain with both team ants overlaid; digit = ant id
modulo team, capital letter = that ant is carrying food):
{grid_ascii}

Pheromone trails on your team's board (sparse view; only cells above
{pher_threshold} shown):
  to_food: {pher_food}
  to_nest: {pher_nest}

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


_NUM_TEAMS = 2
_PHEROMONE_THRESHOLD = 0.05
_MOVE_HISTORY_TAIL = 16


# --- Helpers ----------------------------------------------------------------


def _parse_obs(observation: Mapping[str, Any]) -> dict[str, Any]:
    obs_str = observation.get("observationString", "")
    if not obs_str:
        return {}
    try:
        return json.loads(obs_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def _ant_glyph(seat: int, carrying: bool) -> str:
    """Glyph for a teammate's ant within the team-local view.

    Uses the seat index (0..players_per_team-1) so both teams render
    identically regardless of which global player ids they happen to
    hold. Searching ants are digits; carrying ants are capital letters.
    """
    if not carrying:
        return str(seat) if 0 <= seat < 10 else "?"
    return chr(ord("A") + seat) if 0 <= seat < 26 else "?"


def _render_grid_ascii(
    grid: list[list[str]],
    ant_positions_by_pid: Mapping[str, Any] | None,
    carrying_by_pid: Mapping[str, Any] | None,
    players_per_team: int,
    team_id: int,
) -> str:
    """Render the team board with the team's ants overlaid on terrain.

    First-seat-wins when multiple ants share a cell (matches upstream
    ant_foraging's __str__ convention). The prose elsewhere lists each
    ant's position separately, so any stacking lost here is recoverable.
    """
    rows = len(grid) if grid else 0
    cols = len(grid[0]) if rows else 0
    ant_positions_by_pid = ant_positions_by_pid or {}
    carrying_by_pid = carrying_by_pid or {}

    # Seat -> (row, col) using team-relative seat indices so the renderer
    # stays symmetric across the two teams.
    seat_at: dict[tuple[int, int], int] = {}
    for seat in range(players_per_team):
        pid = team_id * players_per_team + seat
        pos = ant_positions_by_pid.get(str(pid))
        if not pos or len(pos) < 2:
            continue
        cell = (int(pos[0]), int(pos[1]))
        if cell not in seat_at:
            seat_at[cell] = seat

    header = "    " + " ".join(str(c) for c in range(cols))
    lines = [header]
    for r in range(rows):
        row_chars = []
        for c in range(cols):
            if (r, c) in seat_at:
                seat = seat_at[(r, c)]
                pid = team_id * players_per_team + seat
                carrying = bool(carrying_by_pid.get(str(pid), False))
                row_chars.append(_ant_glyph(seat, carrying))
            else:
                row_chars.append(grid[r][c])
        lines.append(f"{r:>2}  " + " ".join(row_chars))
    return "\n".join(lines)


def _sparse_pheromone(
    pheromone: list[list[float]] | None,
    threshold: float = _PHEROMONE_THRESHOLD,
) -> str:
    if not pheromone:
        return "(none)"
    items: list[str] = []
    for r, row in enumerate(pheromone):
        for c, v in enumerate(row):
            if float(v) >= threshold:
                items.append(f"[{r},{c}]={float(v):.2f}")
    return ", ".join(items) if items else "(none)"


def _format_move_history(history: list[dict[str, Any]] | None) -> str:
    if not history:
        return "  (no moves yet)"
    tail = history[-_MOVE_HISTORY_TAIL:]
    return "\n".join(
        f"  move {idx + 1}: player {entry.get('player_id')} "
        f"(seat {entry.get('seat')}) -> {entry.get('action')}"
        for idx, entry in enumerate(tail, start=len(history) - len(tail))
    )


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
    move_number = int(obs.get("move_number", 0))

    # Normalize to per-team-round units so the model isn't comparing
    # interleaved-step counts against round-based max_turns. Display is
    # 1-indexed so the final move reads "round 50 of 50"; the engine's
    # 0-indexed count would read "round 49 of 50" on the last move, and
    # models systematically misread that as "one round still remains".
    round_size = players_per_team * _NUM_TEAMS
    current_round = (move_number // round_size) + 1 if round_size else 1

    teammate_seat = (seat + 1) % players_per_team
    teammate_id = team_id * players_per_team + teammate_seat

    ant_positions = board.get("ant_positions") or {}
    your_position = ant_positions.get(str(player_id), "unknown")
    carrying = board.get("carrying_food") or {}
    is_carrying = bool(carrying.get(str(player_id), False))
    carry_status = "carrying food back to the nest" if is_carrying else "searching for food"

    grid_ascii = _render_grid_ascii(
        board.get("grid") or [],
        ant_positions,
        carrying,
        players_per_team,
        team_id,
    )
    pher_food = _sparse_pheromone(board.get("pheromone_to_food"))
    pher_nest = _sparse_pheromone(board.get("pheromone_to_nest"))
    move_history_str = _format_move_history(board.get("move_history"))

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
        current_round=current_round,
        grid_ascii=grid_ascii,
        pher_threshold=f"{_PHEROMONE_THRESHOLD:.2f}",
        pher_food=pher_food,
        pher_nest=pher_nest,
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
