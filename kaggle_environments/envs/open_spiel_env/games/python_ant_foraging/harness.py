"""LLM harness for the OpenSpiel Python Ant Foraging game.

The Kaggle agent script imports three module-level functions from here:
``get_legal_moves``, ``generate_prompt``, and ``parse_response``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult, parse_json_action, render_rethink_suffix

# --- Prompt -----------------------------------------------------------------


ANT_PROMPT_TEMPLATE = """\
You are playing Ant Foraging, a cooperative grid game.

Rules:
- The world is a {grid_size}x{grid_size} grid. ``N`` marks the single
  nest; ``F`` marks each remaining food source; ``.`` is an empty cell.
- {num_ants} ants share one team score. The game ends when all
  {num_food} food items have been delivered to the nest, or after
  {max_turns} full rounds, whichever comes first.
- Each round every ant takes one turn in order (ant 0, then ant 1, ...).
  On its turn an ant picks one of {{stay, up, down, left, right}}. The
  move must stay on the board (off-board moves are not legal).
- Stepping onto an ``F`` cell automatically picks up that food (only one
  food at a time per ant). Returning to ``N`` while carrying drops it off
  and increases the team score by 1.
- Ants leave decaying pheromone trails. ``to_food`` pheromone is laid
  near remembered food; ``to_nest`` pheromone is laid by ants carrying
  food on the way home. Both decay each round, so fresh trails are more
  reliable than faint ones.

Coordinates: positions are ``[row, column]`` with ``row=0`` at the top
and ``column=0`` on the left. ``up`` decreases row, ``down`` increases
row, ``left`` decreases column, ``right`` increases column.

Board (terrain with your team's ants overlaid; digit = ant id,
capital letter = that ant is carrying food):
{grid_ascii}

Pheromone trails (sparse view; only cells above {pher_threshold} shown):
  to_food: {pher_food}
  to_nest: {pher_nest}

Game progress: round {turn} of {max_turns}, score {score} of {num_food}
food delivered.

Ants on the board:
{ant_summary}

Moves taken so far this game: {move_history_str}

You are ant {player_id}, currently at {your_position} and {carry_status}.

Action format: legal moves are written as ``ant<i>:<direction>``. Only
``ant{player_id}:...`` actions belong to you this turn -- choose a single
direction word (``stay``, ``up``, ``down``, ``left``, or ``right``) and
the framework will pair it with your ant id.

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


_PHEROMONE_THRESHOLD = 0.05
_MOVE_HISTORY_TAIL = 16


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


def _ant_glyph(ant_id: int, carrying: bool) -> str:
    """Single-char glyph for an ant. Digit if searching, capital A/B/...
    if carrying food. Falls back to ``?`` if ant_id is out of A-Z range.
    """
    if not carrying:
        return str(ant_id) if 0 <= ant_id < 10 else "?"
    return chr(ord("A") + ant_id) if 0 <= ant_id < 26 else "?"


def _render_grid_ascii(
    grid: list[list[str]],
    ant_positions: list[list[int]],
    carrying_food: list[bool],
) -> str:
    """Render the board with ants overlaid on terrain.

    First-ant-wins when multiple ants share a cell (matches the engine's
    own __str__). The prose elsewhere in the prompt lists every ant's
    position separately, so any stacking lost here is recoverable.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    ant_at: dict[tuple[int, int], int] = {}
    for i, pos in enumerate(ant_positions):
        if not pos or len(pos) < 2:
            continue
        cell = (int(pos[0]), int(pos[1]))
        if cell not in ant_at:
            ant_at[cell] = i

    header = "    " + " ".join(str(c) for c in range(cols))
    lines = [header]
    for r in range(rows):
        row_chars = []
        for c in range(cols):
            if (r, c) in ant_at:
                i = ant_at[(r, c)]
                carrying = bool(carrying_food[i]) if i < len(carrying_food) else False
                row_chars.append(_ant_glyph(i, carrying))
            else:
                row_chars.append(grid[r][c])
        lines.append(f"{r:>2}  " + " ".join(row_chars))
    return "\n".join(lines)


def _sparse_pheromone(
    pheromone: list[list[float]] | None,
    threshold: float = _PHEROMONE_THRESHOLD,
) -> str:
    """List cells whose pheromone value is at least ``threshold``."""
    if not pheromone:
        return "(none)"
    items: list[str] = []
    for r, row in enumerate(pheromone):
        for c, v in enumerate(row):
            if float(v) >= threshold:
                items.append(f"[{r},{c}]={float(v):.2f}")
    return ", ".join(items) if items else "(none)"


def _format_ant_summary(
    ant_positions: list[list[int]],
    carrying_food: list[bool],
) -> str:
    if not ant_positions:
        return "  (none)"
    lines = []
    for i, pos in enumerate(ant_positions):
        carrying = bool(carrying_food[i]) if i < len(carrying_food) else False
        status = "carrying food" if carrying else "searching"
        lines.append(f"  ant {i}: at [{int(pos[0])},{int(pos[1])}], {status}")
    return "\n".join(lines)


def _format_move_history(
    proxy_history: list[dict[str, Any]] | None,
    fallback: list[str],
) -> str:
    """Render the game-wide history if the proxy provides it, otherwise
    fall back to the per-agent history list supplied by core_harness.
    """
    if proxy_history:
        entries = [
            f"ant{int(entry.get('seat', 0))}:{entry.get('action', '?')}"
            for entry in proxy_history[-_MOVE_HISTORY_TAIL:]
        ]
        return ", ".join(entries) if entries else "None"
    if fallback:
        return ", ".join(fallback[-_MOVE_HISTORY_TAIL:])
    return "None"


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
    # Display is 1-indexed so the final round reads "round 50 of 50"; the
    # engine's 0-indexed ``turn`` would render "round 49 of 50" on the
    # last round, which models systematically misread as "one round still
    # remains". Mirrors the arena harness.
    display_round = int(parsed.get("turn", 0)) + 1
    score = int(parsed.get("food_collected", parsed.get("score", 0)))

    ant_positions = parsed.get("ant_positions") or []
    carrying = parsed.get("carrying_food") or []
    grid = parsed.get("grid") or [["." for _ in range(grid_size)] for _ in range(grid_size)]

    if 0 <= player_id < len(ant_positions):
        your_position = ant_positions[player_id]
    else:
        your_position = "unknown"
    carry_status = (
        "carrying food back to the nest"
        if (0 <= player_id < len(carrying) and carrying[player_id])
        else "searching for food"
    )

    grid_ascii = _render_grid_ascii(grid, ant_positions, carrying)
    pher_food = _sparse_pheromone(parsed.get("pheromone_to_food"))
    pher_nest = _sparse_pheromone(parsed.get("pheromone_to_nest"))
    ant_summary = _format_ant_summary(ant_positions, carrying)
    move_history_str = _format_move_history(parsed.get("move_history"), move_history)

    prompt = ANT_PROMPT_TEMPLATE.format(
        grid_size=grid_size,
        num_ants=num_ants,
        num_food=num_food,
        max_turns=max_turns,
        turn=display_round,
        score=score,
        player_id=player_id,
        your_position=your_position,
        carry_status=carry_status,
        grid_ascii=grid_ascii,
        pher_threshold=f"{_PHEROMONE_THRESHOLD:.2f}",
        pher_food=pher_food,
        pher_nest=pher_nest,
        ant_summary=ant_summary,
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
    return parse_json_action(response, legal_action_strings, matcher=_match_move_to_legal)
