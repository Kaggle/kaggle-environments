"""LLM harness for OpenSpiel Capture the Flag.

Capture the Flag is a 2-player simultaneous-move grid game. Each round both
players pick one of five actions -- North, East, South, West, Stay -- and
then a hidden coin flip decides whose move resolves first. A player picks up
the opponent's flag by stepping onto it at the opponent's base while it is
loose; scores by carrying it back to their own base while their own flag is
sitting at home; and gets tagged (respawn at own base, drop the flag) when
the opposing defender is standing in one of the four cells directly next
to them (up/down/left/right, not diagonal) and the carrier is inside the
defender's home territory. Territory is split by column: A owns columns
strictly left of centre, B owns columns strictly right of centre; on an
odd-width grid the centre column is neutral. First to ``score_limit``
captures wins; the game draws at ``horizon`` rounds.
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

_ACTION_NAMES = ["North", "East", "South", "West", "Stay"]


CAPTURE_THE_FLAG_PROMPT_TEMPLATE = """Capture the Flag: 2-player simultaneous-move grid game on a {num_rows} x {num_cols} grid.
Rows 0 (top) to {max_row} (bottom); columns 0 (left) to {max_col} (right).
Each round both players SIMULTANEOUSLY pick one of five actions:
  North = row-1, South = row+1, East = col+1, West = col-1, Stay = no move.
After both moves are revealed, a hidden coin flip picks whose move resolves first; you cannot know the order in advance. Consequence: if you both target the same empty cell, only whoever resolves first lands there -- the other stays put.

Move rules:
  - Moving off-grid{obstacle_move_clause} or into the OTHER player's cell is a no-op (you stay put; no tag from bumping).
  - Stepping onto the opponent's LOOSE flag AT the opponent's base picks it up; you now carry it and it moves with you.
  - Moving onto your OWN base while carrying the opponent's flag means you win -- BUT only if your own flag is still sitting at your home base AT THAT MOMENT. Standing still (Stay) never triggers a score; you must step onto the base. If your own flag is loose or held by the opponent, arriving at your base does nothing.

Post-turn resolution (after BOTH moves have applied, using final positions):
  - Tagging: any carrier who is directly next to the flag's owner (in one of the four cells up, down, left, or right -- not diagonal) AND standing inside the flag-owner's home territory is tagged -- respawn at own base, and the flag returns to its owner's base. Because this uses final positions, a defender who moves next to you during the same turn tags you even if you weren't next to them when the turn started.

Territory split: A owns columns 0..{a_territory_max}; B owns columns {b_territory_min}..{max_col}.{neutral_note}

Bases: A base at {a_base}, B base at {b_base}.

Board pieces: '.' = empty,{obstacle_legend} 'A'/'B' = players, 'a'/'b' = loose flag at that cell.
A player standing on their own home base with their own flag still home renders as 'A' or 'B' (the flag is under the player).

Current board (row 0 on top; columns labelled 0..{max_col}):
{board_ascii}

Positions: A at {a_pos}, B at {b_pos}.
Flag A (belongs to A): {flag_a_status}
Flag B (belongs to B): {flag_b_status}

Score: A={score_a}, B={score_b} (first to {score_limit} wins; draw at {horizon} rounds).

You are Player {player_label}. Your flag is Flag {player_label}; your goal is to carry Flag {opponent_label} to your base at {your_base}.

Round {round_number} of at most {horizon}.
{move_history_block}

Your turn. Choose one of: North, East, South, West, Stay.

Respond with your reasoning, then end your response with JSON:

```json
{{"move": "<North|East|South|West|Stay>"}}
```
"""


RETHINK_ILLEGAL = """

You suggested move "{previous_action}" but this is not a legal move.
The only legal moves are: North, East, South, West, Stay.

(Keep using the same JSON output format as before -- only the move value needs to change.)
"""


RETHINK_UNPARSABLE = """

Your previous response ended with:
{previous_response}

No JSON answer could be parsed from that. Conclude your response with your
final move as JSON in a ```json fenced block, exactly as the original
instructions required:

```json
{{"move": "<North|East|South|West|Stay>"}}
```

For example: `{{"move": "East"}}`

The move you choose must also be one of: North, East, South, West, Stay.
"""


def _parse_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the structured CTF state dict out of the observation."""
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
    if isinstance(pos, (list, tuple)) and len(pos) == 2:
        return f"(row {pos[0]}, col {pos[1]})"
    return "(unknown)"


def _flag_status(carrier: Any, flag_pos: Any, base_pos: Any) -> str:
    if carrier is None:
        at_home = (
            isinstance(flag_pos, (list, tuple))
            and isinstance(base_pos, (list, tuple))
            and list(flag_pos) == list(base_pos)
        )
        return "at home base" if at_home else f"loose at {_pos_str(flag_pos)}"
    carrier_label = "A" if carrier == 0 else "B"
    return f"carried by Player {carrier_label}, currently at {_pos_str(flag_pos)}"


def _render_full_history(observation: Mapping[str, Any]) -> str | None:
    """Reconstruct per-round move history from the serialized pyspiel state.

    Each completed round in CTF appears in ``full_history()`` as three
    entries: Player A's move, Player B's move, and the initiative chance
    outcome (0 = A resolved first, 1 = B resolved first).
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
    for h in state.full_history():
        if h.player == 0:
            a_move = _ACTION_NAMES[h.action] if 0 <= h.action < len(_ACTION_NAMES) else str(h.action)
        elif h.player == 1:
            b_move = _ACTION_NAMES[h.action] if 0 <= h.action < len(_ACTION_NAMES) else str(h.action)
        elif h.player == -1:
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
    """Build the LLM prompt for the current CTF state."""
    del move_history  # Full history is reconstructed from the pyspiel state.
    state = _parse_observation_payload(observation)
    player_id = observation.get("playerId", 0)

    board = state.get("board") or []
    num_rows = state.get("num_rows") or (len(board) if board else 5)
    num_cols = state.get("num_cols") or (len(board[0]) if board and board[0] else 7)
    max_row = max(num_rows - 1, 0)
    max_col = max(num_cols - 1, 0)

    centre = num_cols // 2
    if num_cols % 2 == 1:
        a_territory_max = centre - 1
        b_territory_min = centre + 1
        neutral_note = f" Column {centre} is neutral (belongs to neither)."
    else:
        a_territory_max = centre - 1
        b_territory_min = centre
        neutral_note = ""

    obstacles = state.get("obstacles") or []
    if obstacles:
        obstacle_legend = " '*' = obstacle (impassable),"
        obstacle_move_clause = ", into an obstacle,"
    else:
        obstacle_legend = ""
        obstacle_move_clause = ""

    is_player_a = player_id == 0
    player_label = "A" if is_player_a else "B"
    opponent_label = "B" if is_player_a else "A"
    your_base = _pos_str(state.get("a_base") if is_player_a else state.get("b_base"))

    flag_a_status = _flag_status(
        state.get("carrier_a"),
        state.get("flag_a_pos"),
        state.get("a_base"),
    )
    flag_b_status = _flag_status(
        state.get("carrier_b"),
        state.get("flag_b_pos"),
        state.get("b_base"),
    )

    score = state.get("score") or [0, 0]
    horizon = state.get("horizon", 300)
    move_number = state.get("move_number", 0)
    round_number = int(move_number) + 1

    move_history_block = _render_full_history(observation)
    if move_history_block is None:
        move_history_block = "No rounds have been played yet."

    prompt = CAPTURE_THE_FLAG_PROMPT_TEMPLATE.format(
        num_rows=num_rows,
        num_cols=num_cols,
        max_row=max_row,
        max_col=max_col,
        a_territory_max=a_territory_max,
        b_territory_min=b_territory_min,
        neutral_note=neutral_note,
        a_base=_pos_str(state.get("a_base")),
        b_base=_pos_str(state.get("b_base")),
        obstacle_legend=obstacle_legend,
        obstacle_move_clause=obstacle_move_clause,
        board_ascii=_format_board_ascii(board),
        a_pos=_pos_str(state.get("a_pos")),
        b_pos=_pos_str(state.get("b_pos")),
        flag_a_status=flag_a_status,
        flag_b_status=flag_b_status,
        score_a=score[0],
        score_b=score[1],
        score_limit=state.get("score_limit", 1),
        horizon=horizon,
        player_label=player_label,
        opponent_label=opponent_label,
        your_base=your_base,
        round_number=round_number,
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
