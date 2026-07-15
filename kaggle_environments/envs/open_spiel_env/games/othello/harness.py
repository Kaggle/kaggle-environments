"""LLM harness for OpenSpiel Reversi (registered under the OpenSpiel game
name ``othello``; this harness refers to the game as Reversi throughout).

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.

Reversi is a two-player game on an 8x8 board. Player 0 ('x', Black) moves
first; Player 1 ('o', White) follows. Each move places one disk of the
mover's colour on an empty cell; that cell must flank one or more
contiguous runs of opponent disks in at least one of the eight directions
(horizontal, vertical, diagonal), bounded on the far side by another disk
of the mover's colour. Every flanked opponent disk is flipped. If the
mover has no such placement they must pass. The game ends when neither
player has a legal placement (usually when the board fills). The player
with more disks wins; equal counts is a draw.

Action strings are algebraic ``<file><rank>`` coordinates such as ``"d3"``
(file letters ``a..h`` left-to-right, rank digits ``1..8`` top-to-bottom
in the rendered board), or the literal ``"pass"`` when the mover has no
legal placement.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult, parse_json_action, render_rethink_suffix

# --- Prompt -----------------------------------------------------------------


REVERSI_PROMPT_TEMPLATE = """Let's play Reversi.

Rules: 8x8 board with files a-h (left-to-right) and ranks 1-8
(top-to-bottom in the rendered board). Player 0 ('x', Black) moves first;
Player 1 ('o', White) follows.

Each move places one disk of your colour on an empty cell. That cell must
"flank" one or more contiguous runs of opponent disks in at least one of
the eight directions (horizontal, vertical, diagonal) -- a run bounded on
the far side by another disk of YOUR colour. Every flanked opponent disk
is then flipped to your colour. A placement that flips nothing is not
legal. If you have no legal placement anywhere on the board you must pass
(submit the move "pass"); your opponent then moves.

The game ends when neither player has a legal placement (usually when the
board is full). The player with more disks wins; equal counts is a draw.

Board (rank labels on the left, file labels on top; '.' = empty,
'x' = Black disk, 'o' = White disk):
{board_ascii}

Disk counts: Black ('x') = {black_count}; White ('o') = {white_count}.

You are Player {player_id} ('{my_piece}', {my_colour}).
Your opponent is Player {opp_player_id} ('{opp_piece}', {opp_colour}).{pass_note}
Move number: {move_number}
Last move played: {last_move}
Moves played so far this game (both players, oldest first): {move_history}

Action notation: a two-character string ``<file><rank>`` denotes the empty
square where you place your disk (e.g. ``"d3"`` means "place your disk at
column d, row 3"). File letters are lowercase a-h; rank digits are 1-8
(rank 1 is the top row of the rendered board, rank 8 is the bottom row).
If (and only if) you have no legal placement, submit the literal move
``"pass"``.

It is your turn. Choose a legal move.

Respond with your reasoning followed by your final move in a JSON block:

```json
{{
  "move": "<file><rank>"
}}
```

For example: `{{"move": "d3"}}` -- or `{{"move": "pass"}}` if you have no
legal placement.

Failure to output your final answer in the specified format, or selecting
an illegal move, will result in a loss.
"""


RETHINK_ILLEGAL = """

You suggested move "{previous_action}" but this is not a legal move.
Reconsider the rules and the current state, then pick a legal move.
Remember: your placement must flank at least one contiguous run of
opponent disks bounded on the far side by another of your own disks; a
placement that flips nothing is illegal. If (and only if) you have no
legal placement anywhere, the only legal move is "pass".

(Keep using the same JSON output format as before -- only the move value needs to change.)
"""

RETHINK_UNPARSABLE = """

Your previous response ended with:
{previous_response}

No JSON answer could be parsed from that. Conclude your response
with your final move as JSON in a ```json fenced block, exactly
as the original instructions required:

```json
{{"move": "<file><rank>"}}
```

For example: `{{"move": "d3"}}` -- or `{{"move": "pass"}}` if you have no
legal placement.

The move you choose must also be legal in the current state.
"""


# --- Helpers ----------------------------------------------------------------


def _parse_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the structured Reversi state dict out of the observation."""
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
    """Render the 8x8 board with file labels on top and rank labels on the left.

    ``board[0]`` is rank 1 (top row of the display); ``board[r][0]`` is file
    'a'. Empty cells render as ``.``.
    """
    if not board:
        return "(board state unavailable)"
    num_cols = len(board[0])
    file_header = "  " + " ".join(chr(ord("a") + c) for c in range(num_cols))
    lines = [file_header]
    for r, row in enumerate(board):
        cells = [c if c else "." for c in row]
        lines.append(f"{r + 1} " + " ".join(cells))
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
    player_id = observation.get("playerId", state.current_player())
    actions = state.legal_actions()
    return {a: state.action_to_string(player_id, a) for a in actions}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current Reversi state.

    ``move_history`` (the per-agent framework argument) is intentionally
    ignored; the proxy's ``state_dict()["move_history"]`` gives us the
    full both-player move sequence, which is what the model needs to
    reason about the position.
    """
    del move_history  # sourced from the proxy instead — see docstring.

    state = _parse_observation_payload(observation)
    player_id = observation.get("playerId", 0)

    board = state.get("board") or []
    disks = state.get("disks") or {"x": 0, "o": 0}
    move_number = state.get("move_number", 0)
    last_move_raw = state.get("last_move")
    last_move = last_move_raw or "(none yet)"
    proxy_history = state.get("move_history") or []
    must_pass = bool(state.get("must_pass"))

    my_piece = "x" if player_id == 0 else "o"
    opp_piece = "o" if player_id == 0 else "x"
    my_colour = "Black" if player_id == 0 else "White"
    opp_colour = "White" if player_id == 0 else "Black"
    opp_player_id = 1 - player_id

    pass_note = (
        "\n\nYou currently have no legal placements on the board; the only legal move is `pass`."
        if must_pass
        else ""
    )

    move_history_str = ", ".join(proxy_history) if proxy_history else "None"

    prompt = REVERSI_PROMPT_TEMPLATE.format(
        board_ascii=_format_board_ascii(board),
        black_count=disks.get("x", 0),
        white_count=disks.get("o", 0),
        player_id=player_id,
        my_piece=my_piece,
        my_colour=my_colour,
        opp_piece=opp_piece,
        opp_colour=opp_colour,
        opp_player_id=opp_player_id,
        pass_note=pass_note,
        move_number=move_number,
        last_move=last_move,
        move_history=move_history_str,
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
