"""LLM harness for OpenSpiel Markov Soccer.

Markov Soccer (Littman 1994) is a 2-player simultaneous-move grid game. On
the default 4 row x 5 col field, Player A starts at (2, 1) and Player B at
(1, 3); a loose ball spawns at random in column 2. Each round both players
simultaneously choose one of ``up / down / left / right / stand``. After the
choices are revealed, a chance node randomly picks which player's move is
resolved first; then the other's. Walking onto the loose ball ('O') picks it
up. The ball-holder loses possession to a defender by walking into the
defender's cell -- the reverse direction does nothing. A player holding the
ball wins by walking off the opponent's goal edge from row 1 or row 2; A
scores on the right edge, B on the left. If the horizon (default 100) is
reached without a goal, the game is a draw.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import (
    ParseResult,
    create_agent_fn,
    parse_json_action,
    render_rethink_suffix,
)

MARKOV_SOCCER_PROMPT_TEMPLATE = """Markov Soccer (Littman 1994): 2-player simultaneous-move grid game on a {num_rows} x {num_cols} field.
Rows 0 (top) to {max_row} (bottom); columns 0 (left) to {max_col} (right).
Each round both players SIMULTANEOUSLY pick one of five actions:
  up = row-1, down = row+1, left = col-1, right = col+1, stand = no move.
After both moves are revealed, a hidden coin flip picks whose move resolves first; you cannot know the order in advance.

Board pieces: 'a'/'A' = Player A (uppercase = holds ball), 'b'/'B' = Player B, 'O' = loose ball, '.' = empty.

Mechanics (per player in initiative order):
  - Moving into '.' moves you there. Moving into 'O' picks up the ball (piece becomes uppercase).
  - Ball-holder walking into the OPPONENT: possession transfers, neither piece moves.
  - A player WITHOUT the ball walking into another player is a no-op. You cannot steal by walking into the ball-holder; you must wait for the ball-holder to walk into you.
  - Moving off the edge is a no-op EXCEPT: a ball-holder in row 1 or row 2 stepping off the opponent's goal edge SCORES and wins immediately.

Scoring edges: A scores by stepping RIGHT off column {max_col}; B by stepping LEFT off column 0 (from row 1 or 2, while holding the ball). Winner +1, loser -1. Draw at {max_rounds} rounds with no goal.

Current board (row 0 on top; columns labelled 0..{max_col}):
{board_ascii}

Player positions: A at {a_pos}, B at {b_pos}.
Ball: {ball_status}

You are Player {player_label} ('{my_piece_lower}' without ball, '{my_piece_upper}' with ball). {ball_for_you}
{your_goal_sentence}
{opponent_goal_sentence}

Round: {move_number}
{move_history_block}

Your turn. Choose one of: up, down, left, right, stand.

Respond with your reasoning, then end your response with JSON:

```json
{{"move": "<up|down|left|right|stand>"}}
```
"""


RETHINK_ILLEGAL = """

You suggested move "{previous_action}" but this is not a legal move.
The only legal moves are: up, down, left, right, stand.

(Keep using the same JSON output format as before -- only the move value needs to change.)
"""

RETHINK_UNPARSABLE = """

Your previous response ended with:
{previous_response}

No JSON answer could be parsed from that. Conclude your response with your
final move as JSON in a ```json fenced block, exactly as the original
instructions required:

```json
{{"move": "<up|down|left|right|stand>"}}
```

For example: `{{"move": "right"}}`

The move you choose must also be one of: up, down, left, right, stand.
"""

def _parse_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the structured markov-soccer state dict out of the observation."""
    raw = observation.get("observationString", "") or ""
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    serialized = observation.get("serializedGameAndState", "")
    if serialized:
        _, state = pyspiel.deserialize_game_and_state(serialized)
        try:
            return json.loads(state.observation_string(0))
        except (json.JSONDecodeError, RuntimeError):
            pass
    return {}


def _format_board_ascii(board: Sequence[Sequence[str]]) -> str:
    """Render the board with a column header and row labels on the left."""
    if not board:
        return "(unavailable)"
    num_cols = len(board[0])
    header = "    " + " ".join(str(c) for c in range(num_cols))
    lines = [header]
    for r, row in enumerate(board):
        lines.append(f"  {r} " + " ".join(row))
    return "\n".join(lines)


def _pos_str(pos: Any) -> str:
    """Render a [row, col] position as ``(r, c)`` or ``(unknown)``."""
    if isinstance(pos, (list, tuple)) and len(pos) == 2:
        return f"(row {pos[0]}, col {pos[1]})"
    return "(unknown)"


_ACTION_NAMES = ["up", "down", "left", "right", "stand"]


def _render_full_history(observation: Mapping[str, Any]) -> str | None:
    """Reconstruct per-round move history (both players + initiative outcome).

    OpenSpiel's ``state.full_history()`` records, in order: the initial
    ball-spawn chance outcome, then for each completed round Player A's
    move, Player B's move, and the initiative chance outcome (0 = A's
    move resolved first, 1 = B's move resolved first). Both players see
    the same history.

    Returns a multi-line string ready to drop into the prompt, or
    ``None`` when the serialized state isn't available or no full round
    has completed yet (so callers can fall back to a simpler line).
    """
    serialized = observation.get("serializedGameAndState", "")
    if not serialized:
        return None
    try:
        _, state = pyspiel.deserialize_game_and_state(serialized)
    except Exception:
        return None

    rounds: list[tuple[str, str, str]] = []
    a_move: str | None = None
    b_move: str | None = None
    saw_ball_spawn = False
    for h in state.full_history():
        if h.player == 0:
            a_move = _ACTION_NAMES[h.action]
        elif h.player == 1:
            b_move = _ACTION_NAMES[h.action]
        elif h.player == -1:
            if not saw_ball_spawn:
                saw_ball_spawn = True
                continue
            init = "A" if h.action == 0 else "B"
            rounds.append((a_move or "?", b_move or "?", init))
            a_move = b_move = None

    if not rounds:
        return None
    lines = ["Move history so far (both players, oldest first):"]
    for i, (a, b, init) in enumerate(rounds, start=1):
        lines.append(f"  Round {i}: A={a}, B={b} ({init}'s move resolved first)")
    return "\n".join(lines)

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
    player_id = observation.get("playerId", 0)
    actions = state.legal_actions(player_id)
    return {a: state.action_to_string(player_id, a) for a in actions}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current markov-soccer state.

    ``move_history`` contains this agent's own past moves; we ignore it in
    favour of reconstructing both players' move sequences plus initiative
    outcomes from ``state.full_history()`` of the deserialized pyspiel
    state, which is public information in this perfect-info game. The
    per-agent argument is used as a fallback only if the serialized state
    is unavailable.
    """
    state = _parse_observation_payload(observation)
    player_id = observation.get("playerId", 0)

    board = state.get("board") or []
    num_rows = len(board) if board else 4
    num_cols = len(board[0]) if board and board[0] else 5
    max_row = max(num_rows - 1, 0)
    max_col = max(num_cols - 1, 0)

    a_pos_str = _pos_str(state.get("player_a_pos"))
    b_pos_str = _pos_str(state.get("player_b_pos"))
    ball_owner = state.get("ball_owner")
    ball_pos = state.get("ball_pos")
    if ball_owner is None:
        ball_status = f"loose at {_pos_str(ball_pos)} (walk onto it to pick it up)"
    else:
        ball_status = f"held by Player {ball_owner} at {_pos_str(ball_pos)}"

    is_player_a = player_id == 0
    player_label = "A" if is_player_a else "B"
    my_piece_lower = "a" if is_player_a else "b"
    my_piece_upper = "A" if is_player_a else "B"

    if ball_owner == player_label:
        ball_for_you = "YOU currently hold the ball."
    elif ball_owner is None:
        ball_for_you = "Neither player holds the ball yet."
    else:
        ball_for_you = "Your opponent currently holds the ball."

    if is_player_a:
        your_goal_sentence = (
            f"Your goal: walk RIGHT off column {max_col} (the right edge) from row 1 or row 2 while holding the ball."
        )
        opponent_goal_sentence = (
            "Opponent's goal: walk LEFT off column 0 (the left edge) from row 1 or row 2 while holding the ball."
        )
    else:
        your_goal_sentence = (
            "Your goal: walk LEFT off column 0 (the left edge) from row 1 or row 2 while holding the ball."
        )
        opponent_goal_sentence = (
            "Opponent's goal: walk RIGHT off column "
            f"{max_col} (the right edge) from row 1 or row 2 while holding "
            "the ball."
        )

    horizon = 100
    serialized = observation.get("serializedGameAndState", "")
    if serialized:
        try:
            game, _ = pyspiel.deserialize_game_and_state(serialized)
            params = game.get_parameters()
            horizon = int(params.get("horizon", horizon))
        except Exception:
            pass
    # The engine increments total_moves on the initial ball-spawn chance
    # node too, so only horizon-1 simultaneous-move rounds actually run.
    max_rounds = max(horizon - 1, 0)

    move_number = len(move_history) + 1
    move_history_block = _render_full_history(observation)
    if move_history_block is None:
        if move_history:
            move_history_block = (
                f"Your past moves (oldest first): {', '.join(move_history)}"
            )
        else:
            move_history_block = "No moves have been played yet."

    prompt = MARKOV_SOCCER_PROMPT_TEMPLATE.format(
        num_rows=num_rows,
        num_cols=num_cols,
        max_row=max_row,
        max_col=max_col,
        max_rounds=max_rounds,
        board_ascii=_format_board_ascii(board),
        a_pos=a_pos_str,
        b_pos=b_pos_str,
        ball_status=ball_status,
        player_label=player_label,
        my_piece_lower=my_piece_lower,
        my_piece_upper=my_piece_upper,
        ball_for_you=ball_for_you,
        your_goal_sentence=your_goal_sentence,
        opponent_goal_sentence=opponent_goal_sentence,
        move_number=move_number,
        move_history_block=move_history_block,
    )

    prompt += render_rethink_suffix(
        RETHINK_ILLEGAL,
        RETHINK_UNPARSABLE,
        previous_response,
        previous_action,
    )

    return prompt


def parse_response(
    response: str,
    legal_action_strings: Sequence[str],
) -> ParseResult:
    """Trust the model's JSON answer; let the rethink loop fix anything else."""
    return parse_json_action(response, legal_action_strings)

class _MarkovSoccerHarness:
    """Adapts module-level harness functions to the GameHarness protocol."""

    def get_legal_moves(self, observation):
        return get_legal_moves(observation)

    def make_prompt(
        self,
        observation,
        move_history,
        previous_response=None,
        previous_action=None,
    ):
        return generate_prompt(
            observation,
            move_history,
            previous_response=previous_response,
            previous_action=previous_action,
        )

    def parse_response(self, response, legal_action_strings, *, observation):
        del observation  # unused — Markov Soccer parses purely from the response.
        return parse_response(response, legal_action_strings)


agent_fn = create_agent_fn(_MarkovSoccerHarness())
