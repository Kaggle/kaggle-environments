"""Prompt variants for the Markov Soccer ablation study.

Each variant implements the :class:`core_harness.GameHarness` protocol so the
ablation runner can do ``create_agent_fn(variant)`` directly. The production
prompt lives in :mod:`harness`; this file is the experimental surface.

Conventions enforced by ``python -m kaggle_environments.ablation check``:
  * The ``baseline`` variant in ``VARIANTS`` must produce a prompt
    BYTE-IDENTICAL to ``harness.generate_prompt`` for the same observation.
  * Every variant must render without error on five seeded observations.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import (
    ParseResult,
    parse_json_action,
    render_rethink_suffix,
)


# ---------------------------------------------------------------------------
# Baseline template (byte-identical to harness.py)
# ---------------------------------------------------------------------------

_BASELINE_TEMPLATE = """Markov Soccer (Littman 1994): 2-player simultaneous-move grid game on a {num_rows} x {num_cols} field.
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


_RETHINK_ILLEGAL = """

You suggested move "{previous_action}" but this is not a legal move.
The only legal moves are: up, down, left, right, stand.

(Keep using the same JSON output format as before -- only the move value needs to change.)
"""

_RETHINK_UNPARSABLE = """

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


# ---------------------------------------------------------------------------
# Helpers shared across variants (ported verbatim from harness.py)
# ---------------------------------------------------------------------------

_ACTION_NAMES = ["up", "down", "left", "right", "stand"]


def _parse_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
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
    if not board:
        return "(unavailable)"
    num_cols = len(board[0])
    header = "    " + " ".join(str(c) for c in range(num_cols))
    lines = [header]
    for r, row in enumerate(board):
        lines.append(f"  {r} " + " ".join(row))
    return "\n".join(lines)


def _pos_str(pos: Any) -> str:
    if isinstance(pos, (list, tuple)) and len(pos) == 2:
        return f"(row {pos[0]}, col {pos[1]})"
    return "(unknown)"


def _render_full_history(observation: Mapping[str, Any]) -> str | None:
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


def _get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str]:
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


def _compute_prompt_fields(
    observation: Mapping[str, Any],
    move_history: list[str],
) -> dict[str, Any]:
    """Compute the per-state substitution fields shared by every variant.

    Variants differ only in their template text; the field values match
    harness.generate_prompt exactly so byte-identity holds for baseline.
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

    return dict(
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


# ---------------------------------------------------------------------------
# BASELINE
# ---------------------------------------------------------------------------


class BaselineVariant:
    """Byte-identical port of harness.py. DO NOT paraphrase."""

    name = "baseline"
    template = _BASELINE_TEMPLATE

    def get_legal_moves(self, observation):
        return _get_legal_moves(observation)

    def make_prompt(
        self,
        observation,
        move_history,
        previous_response=None,
        previous_action=None,
    ):
        fields = _compute_prompt_fields(observation, move_history)
        body = self.template.format(**fields)
        body += render_rethink_suffix(
            _RETHINK_ILLEGAL,
            _RETHINK_UNPARSABLE,
            previous_response,
            previous_action,
        )
        return body

    def parse_response(self, response, legal_action_strings, *, observation=None):
        del observation
        return parse_json_action(response, legal_action_strings)


# ---------------------------------------------------------------------------
# NULL — byte-identical duplicate of baseline; calibrates the noise floor
# ---------------------------------------------------------------------------


class NullVariant(BaselineVariant):
    """Second copy of baseline. The runner schedules independent cells for
    it, so the difference between null-vs-baseline rankings measures pure
    LLM-sampling noise (the permutation test's noise floor)."""

    name = "null"


# ---------------------------------------------------------------------------
# NO_GOAL_HINT — baseline minus the perspective-restated goal sentences
# ---------------------------------------------------------------------------

_NO_GOAL_HINT_TEMPLATE = """Markov Soccer (Littman 1994): 2-player simultaneous-move grid game on a {num_rows} x {num_cols} field.
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

Round: {move_number}
{move_history_block}

Your turn. Choose one of: up, down, left, right, stand.

Respond with your reasoning, then end your response with JSON:

```json
{{"move": "<up|down|left|right|stand>"}}
```
"""


class NoGoalHintVariant(BaselineVariant):
    """Baseline with the two perspective-restated goal sentences removed
    (the 'Your goal: walk RIGHT off column X ...' and 'Opponent's goal:
    ...' lines directly under the ball-for-you cue). The mechanics-level
    'Scoring edges' paragraph still states which player scores where, so
    the model still knows the win condition; this strips only the
    per-player strategy nudge. Tests whether that nudge is doing work
    independent of the rule statement."""

    name = "no_goal_hint"
    template = _NO_GOAL_HINT_TEMPLATE


# ---------------------------------------------------------------------------
# VERBOSE — the previous baseline (long-form prose) before the compact rewrite
# ---------------------------------------------------------------------------

_VERBOSE_TEMPLATE = """Let's play Markov Soccer (Littman 1994).

Rules: Two players (A and B) move on a {num_rows} row x {num_cols} col grid.
Rows are numbered 0 (top) to {max_row} (bottom); columns are numbered 0
(left) to {max_col} (right). Each round both players SIMULTANEOUSLY choose
one of five actions:

  - ``up``    -> row index decreases by 1 (toward the top)
  - ``down``  -> row index increases by 1 (toward the bottom)
  - ``left``  -> column index decreases by 1
  - ``right`` -> column index increases by 1
  - ``stand`` -> no movement

After both choices are revealed, a hidden coin flip picks which player's move
is resolved first; the other player's move resolves immediately after. You
cannot know the resolution order in advance.

Piece encoding on the board:
  - ``a`` or ``A`` = Player A   (uppercase means A is holding the ball)
  - ``b`` or ``B`` = Player B   (uppercase means B is holding the ball)
  - ``O``         = loose ball on the field (no one is holding it)
  - ``.``         = empty cell

Movement and ball mechanics (applied per player in initiative order):

  1. ``stand`` keeps the player in place.
  2. Moving into an empty cell ('.') moves the player there.
  3. Moving into a cell with the loose ball ('O') picks it up; the player becomes
     uppercase ('a' -> 'A' or 'b' -> 'B') and now holds the ball.
  4. THE BALL-HOLDER moving into the OPPONENT's cell loses the ball: the
     ball-holder stays in place but becomes lowercase, and the opponent
     becomes uppercase (now holds the ball) without moving. Neither piece
     changes square; only possession transfers.
  5. ANY OTHER attempt to move into an occupied cell silently fails. The
     mover stays where they are and nothing else changes. In particular,
     a player WITHOUT the ball who tries to walk into the ball-holder
     does NOT steal the ball; the move just fails. The only way to take
     the ball from the opponent is to wait for the ball-holder to walk
     into you.
  6. Trying to move off the edge of the field normally does nothing (the
     mover stays in place) EXCEPT for scoring: a ball-holder in row 1 or
     row 2 who tries to step off the opponent's goal edge SCORES and wins
     immediately. Stepping off any other edge (or any edge while not
     holding the ball) is a no-op. Only rows 1 and 2 are valid goal rows.

Scoring edges:
  - Player A scores by stepping RIGHT off column {max_col} (i.e. moving
    ``right`` from column {max_col}) while holding the ball, from row 1 or
    row 2.
  - Player B scores by stepping LEFT off column 0 (i.e. moving ``left``
    from column 0) while holding the ball, from row 1 or row 2.

The game ends as soon as either player scores (winner gets +1, loser -1).
If {max_rounds} rounds pass with no goal, the game is a draw (0 / 0).
There are no other terminal conditions.

Current board (row 0 on top; columns labelled 0..{max_col} on top):
{board_ascii}

Player positions: A at {a_pos}, B at {b_pos}.
Ball: {ball_status}

You are Player {player_label} ('{my_piece_lower}' without ball, '{my_piece_upper}' with ball). {ball_for_you}
{your_goal_sentence}
{opponent_goal_sentence}

Round: {move_number}
{move_history_block}

It is your turn. Choose one of: up, down, left, right, stand.

Respond with your reasoning followed by your final move in a JSON block:

```json
{{
  "move": "<up|down|left|right|stand>"
}}
```

For example: `{{"move": "right"}}`

Failure to output your final answer in the specified format, or selecting
an illegal move, will result in a loss.
"""


class VerboseVariant(BaselineVariant):
    """The previous long-form baseline before the compact rewrite. Numbered
    step-by-step mechanics, an explicit piece-encoding table, a worked JSON
    example, and a loss-threat closing. Kept for A/B comparison against the
    compact baseline."""

    name = "verbose"
    template = _VERBOSE_TEMPLATE


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


VARIANTS = {
    "baseline": BaselineVariant(),
    "null": NullVariant(),
    "no_goal_hint": NoGoalHintVariant(),
    "verbose": VerboseVariant(),
}
