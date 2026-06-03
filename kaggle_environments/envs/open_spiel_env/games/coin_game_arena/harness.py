"""LLM harness for Coin Game Arena (2v2 team variant of Coin Game).

Drop the body of this file into the notebook attached to the competition
via HarnessKernelId. The auto-generated ``main.py`` calls these three
module-level functions: ``get_legal_moves``, ``generate_prompt``,
``parse_response``.

The arena observation is a per-player JSON view that includes only the
calling player's team's board, the player's preferred coin colour, and
the full move history on that board (so a player can see what their
teammate just played).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult, parse_json_action, render_rethink_suffix


# --- Prompt -----------------------------------------------------------------


ARENA_PROMPT_TEMPLATE = """Let's play Coin Game Arena (2v2 team coin game).

Setup: 2 teams of 2 players each. Each team plays its own private
{rows}x{cols} board (the other team's board is HIDDEN from you). The
two boards advance in strict alternation: each global step, exactly
ONE player moves, and the next global step the other team's board
moves. Within a team, the two seats also alternate, so your team's
sequence is [seat 0, seat 1, seat 0, ...] on your board.

Important: every player on your team is another instance of YOU
(same model, same submission), and the opposing team is two instances
of a single different agent. There is NO in-game communication, so you
must coordinate with your teammate purely by reasoning about what
"another copy of me" would do given the same board and preferences.
The only thing that distinguishes you from your teammate is your
preferred coin colour and your seat (and therefore the order in which
you move).

Actions: {{up, down, left, right, stand}}. Moving onto a coin collects
it. Moves that would leave the board, or move onto your teammate's
cell, are no-ops — you stay in place (this is not illegal). The
episode runs for {episode_length} moves total per board.

Scoring (per player on YOUR board, summed across the team):

    reward = self_pref^2 + other_pref^2 - bad_coins^2

where self_pref = coins of YOUR preferred colour collected by anyone on
your board; other_pref = coins of your TEAMMATE's preferred colour
collected by anyone on your board; bad_coins = coins of unowned colours
(not preferred by either teammate) collected by anyone on your board.
The team with the higher total reward wins. Ties are draws.

Cells are ``"."`` for empty, digits for players (the player ids shown
below, not the seat indices), lowercase letters for coin colours.
Coordinates are ``[row, column]`` with ``row=0`` at the top.

Your team id is {team_id}. You are player {player_id} (seat {seat} on
your team's board). Your teammate is player {teammate_id} (seat
{teammate_seat}). Your preferred coin colour is "{your_pref}".

Current state of your team's board (JSON):
{board_str}

Move history on your team's board so far (most recent last):
{move_history_str}

{moves_remaining_this_board} of {episode_length} moves remain on
your team's board.

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


# --- Public functions (called by main.py) -----------------------------------


def get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str]:
    """Return ``{action_id: action_string}`` for the current state.

    Returns ``{}`` when this player has no legal actions (their teammate's
    seat is the active one this step).
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
    your_pref = obs.get("your_preference", "?")
    rows = int(obs.get("board", {}).get("num_rows", 8))
    cols = int(obs.get("board", {}).get("num_columns", 8))
    episode_length = int(obs.get("episode_length", 20))
    players_per_team = int(obs.get("players_per_team", 2))

    teammate_seat = (seat + 1) % players_per_team
    teammate_id = team_id * players_per_team + teammate_seat

    board = obs.get("board", {})
    history = board.get("move_history") or []
    # Per-board moves remaining (engine's obs.moves_remaining counts BOTH
    # boards' steps, which mixes units with episode_length above). One
    # entry in history per move taken on this board, so subtract.
    moves_remaining_this_board = max(0, episode_length - len(history))
    # Emit a compact subset of the board view to the model. No
    # moves_remaining here — it's surfaced as a separate sentence below
    # to avoid a unit mismatch with episode_length.
    board_view = {
        "board": board.get("board"),
        "player_positions": board.get("player_positions"),
        "coins_collected": board.get("coins_collected"),
        "coin_colors": board.get("coin_colors"),
    }
    board_str = json.dumps(board_view, indent=2)

    if history:
        move_history_str = "\n".join(
            f"  move {idx + 1}: player {entry.get('player_id')} (seat {entry.get('seat')}) -> {entry.get('action')}"
            for idx, entry in enumerate(history)
        )
    else:
        move_history_str = "  (no moves yet)"

    prompt = ARENA_PROMPT_TEMPLATE.format(
        rows=rows,
        cols=cols,
        episode_length=episode_length,
        team_id=team_id,
        player_id=player_id,
        seat=seat,
        teammate_id=teammate_id,
        teammate_seat=teammate_seat,
        your_pref=your_pref,
        board_str=board_str,
        move_history_str=move_history_str,
        moves_remaining_this_board=moves_remaining_this_board,
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
