"""LLM harness for the OpenSpiel Coin Game.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult, parse_json_action, render_rethink_suffix


# --- Prompt -----------------------------------------------------------------


COIN_PROMPT_TEMPLATE = """Let's play the Coin Game.

Rules: {rows}x{cols} grid. Each player has been privately assigned a
preferred coin colour (you only know your own). Players take turns in
fixed order starting with player 0. On each turn a player picks one of
{{up, down, left, right, stand}}; moving onto a coin collects it.
Moves that would leave the board, or move onto another player's cell,
are no-ops — you stay in place (this is not illegal). The game lasts
{episode_length} moves total (alternating between players). At the end,
each player's reward is

    self_pref^2 + other_pref^2 - bad_coins^2

where ``self_pref`` counts coins of YOUR preferred colour collected by
ANYONE, ``other_pref`` counts coins of OTHER players' preferences
collected by ANYONE, and ``bad_coins`` are coins of unowned colours
(nobody's preference) collected by anyone.

Coordinates are ``[row, column]`` with ``row=0`` at the top. Cells are
``"."`` for empty, digits for players (the player ids shown below, not
seat indices), lowercase letters for coin colours.

Your player id is {player_id}. Your preferred coin colour is "{your_pref}".

Current board:
{board_str}

Other state (JSON):
{state_str}

Move history so far (most recent last):
{move_history_str}

{moves_remaining} of {episode_length} moves remain in the game.

It is now your turn. Choose your move.
The move MUST be one of: up, down, left, right, stand.
Your response should include the reasoning that led to your move, and
conclude with your final move as a JSON formatted as follows:

```json
{{
  "move": "<move>"
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


def _render_board(board: Any) -> str:
    """Render the proxy's board (list of rows of single-char cells) as ASCII."""
    if not isinstance(board, list) or not board:
        return "(board unavailable)"
    return "\n".join("".join(row) for row in board)


def _render_move_history(history: Any) -> str:
    """Render the per-board move log; arena-style annotated lines."""
    if not history:
        return "  (no moves yet)"
    return "\n".join(
        f"  move {entry.get('move_number')}: player {entry.get('player_id')} -> {entry.get('action')}"
        for entry in history
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
    """Build the LLM prompt for the current game state.

    The ``move_history`` parameter (this player's own past moves, supplied
    by core_harness) is ignored; the proxy now exposes a full per-board
    history so we can show all players' moves, not just our own.
    """
    del move_history  # see docstring
    parsed = _parse_obs(observation)
    player_id = int(observation.get("playerId", parsed.get("your_player_id", 0)))

    rows = int(parsed.get("num_rows", 8))
    cols = int(parsed.get("num_columns", 8))
    episode_length = int(parsed.get("episode_length", 20))
    moves_remaining = int(
        parsed.get(
            "moves_remaining",
            episode_length - int(parsed.get("move_number", 0)),
        )
    )
    your_pref = str(parsed.get("your_preference", "?"))

    board_str = _render_board(parsed.get("board"))

    # Curated subset of the observation — the model already has the board
    # and preference above; this slice carries only the per-player state
    # that's useful for planning.
    other_state = {
        "player_positions": parsed.get("player_positions"),
        "coins_collected": parsed.get("coins_collected"),
        "coin_colors": parsed.get("coin_colors"),
    }
    state_str = json.dumps(other_state, indent=2)

    move_history_str = _render_move_history(parsed.get("move_history"))

    prompt = COIN_PROMPT_TEMPLATE.format(
        rows=rows,
        cols=cols,
        episode_length=episode_length,
        moves_remaining=moves_remaining,
        player_id=player_id,
        your_pref=your_pref,
        board_str=board_str,
        state_str=state_str,
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
