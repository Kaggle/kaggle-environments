"""LLM harness for OpenSpiel Nine Men's Morris.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.

Nine Men's Morris is a two-player game on a 24-point board arranged as three
nested squares joined by four midpoint lines. Each player has nine "men" and
tries to form *mills* -- three of their own pieces on any of the 16 lines --
which lets them capture one opponent piece. Play proceeds in three phases
(placement, movement, flying) and terminates when a player is reduced to
fewer than 3 pieces or has no legal move.

Action encoding: point indices 0-23 mean "place at this point" during
placement OR "remove this opponent piece" after forming a mill; movement
actions 24-599 are ``24 + source*24 + dest``. The proxy's ``_action_to_string``
renders these as ``"Place at point N"``, ``"Capture opponent piece at point
N"``, and ``"Move point A -> point B"`` respectively.
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

# --- Board constants (mirror open_spiel/games/nine_mens_morris/*) -----------


_NUM_POINTS = 24

# Each point's neighbors along drawn board lines. Movement in phase 2 is
# restricted to these adjacencies; -1 in the C++ source is dropped.
_POINT_NEIGHBORS: dict[int, tuple[int, ...]] = {
    0: (1, 9),
    1: (0, 2, 4),
    2: (1, 14),
    3: (4, 10),
    4: (1, 3, 5, 7),
    5: (4, 13),
    6: (7, 11),
    7: (4, 6, 8),
    8: (7, 12),
    9: (0, 10, 21),
    10: (3, 9, 11, 18),
    11: (6, 10, 15),
    12: (8, 13, 17),
    13: (5, 12, 14, 20),
    14: (2, 13, 23),
    15: (11, 16),
    16: (15, 17, 19),
    17: (12, 16),
    18: (10, 19),
    19: (16, 18, 20, 22),
    20: (13, 19),
    21: (9, 22),
    22: (19, 21, 23),
    23: (14, 22),
}

# All 16 possible mills. Duplicates the visualizer's list; keep the two in
# sync when either changes.
_MILLS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (15, 16, 17),
    (18, 19, 20),
    (21, 22, 23),
    (0, 9, 21),
    (3, 10, 18),
    (6, 11, 15),
    (8, 12, 17),
    (5, 13, 20),
    (2, 14, 23),
    (1, 4, 7),
    (9, 10, 11),
    (12, 13, 14),
    (16, 19, 22),
)

# A single ASCII board template with point-index placeholders (##). Each
# placeholder is exactly 2 chars wide so pieces or point numbers fit
# without shifting neighbouring characters.
_BOARD_TEMPLATE = (
    "00-----------01-----------02\n"
    " |            |            |\n"
    " |   03-------04-------05  |\n"
    " |    |       |        |   |\n"
    " |    |   06--07--08   |   |\n"
    " |    |    |      |    |   |\n"
    "09---10---11     12---13---14\n"
    " |    |    |      |    |   |\n"
    " |    |   15--16--17   |   |\n"
    " |    |       |        |   |\n"
    " |   18-------19-------20  |\n"
    " |            |            |\n"
    "21-----------22-----------23"
)


# --- Helpers ----------------------------------------------------------------


def _parse_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the structured proxy state dict out of the observation."""
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


def _from_move_action(action: int) -> tuple[int, int]:
    """Decode a movement action into (source, dest) point indices."""
    idx = action - _NUM_POINTS
    return idx // _NUM_POINTS, idx % _NUM_POINTS


def _format_board(board: Sequence[str]) -> str:
    """Overlay pieces on the ASCII board template.

    Each two-character point label (``00`` .. ``23``) becomes either the
    point index (empty) or ``W ``/``B `` when occupied.
    """
    if len(board) != _NUM_POINTS:
        return "(unavailable)"
    out = _BOARD_TEMPLATE
    for p in range(_NUM_POINTS):
        cell = board[p]
        # Two-char cell so pieces align with the point-number placeholders.
        marker = f"{cell} " if cell in ("W", "B") else f"{p:02d}"
        out = out.replace(f"{p:02d}", marker, 1)
    return out


def _mills_of(board: Sequence[str], piece: str) -> list[tuple[int, int, int]]:
    return [m for m in _MILLS if all(board[p] == piece for p in m)]


def _build_move_history_lines(serialized: str) -> list[str]:
    """Reconstruct a labeled per-move history from serialized state.

    We replay the raw pyspiel game rather than the proxy so this file has
    no cross-module dependency at deploy time (harness ships standalone).
    Labeling captures apart from placements uses the fact that whenever an
    ``action < 24`` leaves ``current_player`` unchanged, a mill has just
    formed -- so the next ``action < 24`` by the same player is the
    capture step.
    """
    if not serialized:
        return []
    _, state = pyspiel.deserialize_game_and_state(serialized)
    history = state.history()
    if not history:
        return []
    game = pyspiel.load_game("nine_mens_morris")
    tmp = game.new_initial_state()
    lines: list[str] = []
    capture_pending = False
    for action in history:
        cp = tmp.current_player()
        player = "W" if cp == 0 else "B"
        if action >= _NUM_POINTS:
            src, dst = _from_move_action(action)
            tmp.apply_action(action)
            formed_mill = tmp.current_player() == cp
            suffix = " (mill!)" if formed_mill else ""
            lines.append(f"{player}: move {src}->{dst}{suffix}")
            capture_pending = formed_mill
        elif capture_pending:
            tmp.apply_action(action)
            lines.append(f"{player}: capture point {action}")
            capture_pending = False
        else:
            tmp.apply_action(action)
            formed_mill = tmp.current_player() == cp
            suffix = " (mill!)" if formed_mill else ""
            lines.append(f"{player}: place point {action}{suffix}")
            capture_pending = formed_mill
    return lines


# --- Prompt -----------------------------------------------------------------


NINE_MENS_MORRIS_PROMPT_TEMPLATE = """Let's play Nine Men's Morris (also called Merels or Mill).

Rules: Two-player game on a 24-point board arranged as three nested
squares connected by four midpoint lines. Each player has 9 pieces
("men"). White ('W', Player 0) moves first; Black ('B', Player 1)
follows. The game runs in three phases:

1. Placement (phase 1): players alternate placing one of their 9 men
   on any EMPTY point.
2. Movement (phase 2): once a player has placed all 9 of their men,
   that player moves one of their pieces to an ADJACENT empty point
   along a drawn board line.
3. Flying (phase 3): when a player has been reduced to EXACTLY 3
   pieces, that player may move any of their pieces to ANY empty
   point on the board (adjacency no longer applies).

A "mill" is three of your own pieces along one of the 16 board lines.
Forming a mill in ANY phase lets you IMMEDIATELY remove one of your
opponent's pieces. You may NOT remove an opponent piece that is part
of an opponent mill unless ALL of your opponent's pieces are in mills.
Breaking and re-forming a mill on a later turn re-triggers a capture.

You LOSE if (a) you are reduced to fewer than 3 pieces, or (b) it is
your turn and you have no legal move. The game is a DRAW if the
200-turn hard cap is reached.

Point indices (0-23) on the board:

{board_ascii}

Adjacency for movement (phase 2) follows the drawn lines above. Each
point's neighbors are:
{adjacency}

The 16 mills (three-in-a-row lines):
- Squares (horizontal top/bottom of each nested square): (0,1,2), (3,4,5), (6,7,8), (15,16,17), (18,19,20), (21,22,23)
- Squares (vertical left/right of each nested square):   (0,9,21), (3,10,18), (6,11,15), (8,12,17), (5,13,20), (2,14,23)
- Crosses (midpoint-to-midpoint through each board side): (1,4,7), (9,10,11), (12,13,14), (16,19,22)

Current state:
{state_summary}

You are playing {my_side} ('{my_piece}'). Opponent ({opp_side}, '{opp_piece}').
Active mills -- yours: {my_mills}; opponent's: {opp_mills}.

Current phase: {phase_description}
{phase_instructions}

Move number: {turn_number}
Move history so far (both players, oldest first):
{move_history_block}

It is your turn. Choose a legal action and respond with your reasoning
followed by your final answer in a JSON block. Use this concise notation
for the "move" value:

- Placement (phase 1) or Capture (mill formed): a single point index
  0-23, e.g. `{{"move": "4"}}` to place at (or capture) point 4. The
  current phase determines whether the integer means placement or
  capture -- both cannot be legal at the same time.
- Movement (phase 2/3): source and destination point indices separated
  by `-`, e.g. `{{"move": "3-10"}}` to move point 3 to point 10.

```json
{{
  "move": "<your action>"
}}
```

Failure to output your final answer in the specified format, or
selecting an illegal move, will result in a loss.
"""


RETHINK_ILLEGAL = """

You suggested move "{previous_action}" but this is not a legal move.
Reconsider the current phase (placement / capture / movement / flying),
the board state, and the mill rules, then pick a legal move.

(Keep using the same JSON output format as before -- only the move value needs to change.)
"""

RETHINK_UNPARSABLE = """

Your previous response ended with:
{previous_response}

No JSON answer could be parsed from that. Conclude your response
with your final move as JSON in a ```json fenced block, exactly
as the original instructions required:

```json
{{"move": "<your action>"}}
```

For example: `{{"move": "4"}}` for a placement or capture at point 4,
or `{{"move": "3-10"}}` for a movement from point 3 to point 10.

The move you choose must also be legal in the current state.
"""


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
    """Build the LLM prompt for the current nine-men's-morris state."""
    del move_history  # per-agent history is a subset of what we render below

    state = _parse_observation_payload(observation)
    player_id = observation.get("playerId", 0)

    board: list[str] = list(state.get("board") or ["."] * _NUM_POINTS)
    phase: str = state.get("phase", "placement")
    men_to_deploy = state.get("men_to_deploy") or {"W": 0, "B": 0}
    num_men = state.get("num_men") or {"W": 0, "B": 0}
    turn_number = state.get("turn_number", 0)

    my_piece = "W" if player_id == 0 else "B"
    opp_piece = "B" if player_id == 0 else "W"
    my_side = "WHITE" if player_id == 0 else "BLACK"
    opp_side = "BLACK" if player_id == 0 else "WHITE"

    my_mills = _mills_of(board, my_piece)
    opp_mills = _mills_of(board, opp_piece)

    my_reserve = int(men_to_deploy.get(my_piece, 0))

    # Always render White first, then Black, so orientation is stable across
    # per-player prompts (the "my"/"opp" swap only affects the labels below).
    w_reserve = int(men_to_deploy.get("W", 0))
    b_reserve = int(men_to_deploy.get("B", 0))
    w_total = int(num_men.get("W", 0))
    b_total = int(num_men.get("B", 0))
    state_summary = (
        f"- White (W): {max(0, w_total - w_reserve)} on board, "
        f"{w_reserve} still to place, {w_total} not yet captured (of 9).\n"
        f"- Black (B): {max(0, b_total - b_reserve)} on board, "
        f"{b_reserve} still to place, {b_total} not yet captured (of 9)."
    )

    if phase == "capture":
        phase_description = (
            "capture. You just formed a mill on your previous placement/move and must now REMOVE one opponent piece."
        )
        phase_instructions = (
            "Choose an opponent piece to capture. You cannot pick a piece "
            "that is part of an opponent mill unless ALL opponent pieces are "
            "currently in mills."
        )
    elif phase == "placement":
        phase_description = f"placement. You have {my_reserve} of your 9 men still to place."
        phase_instructions = (
            "Place one of your men on any empty point. If placing forms a "
            "mill, you will be prompted next turn to capture."
        )
    elif phase == "movement":
        phase_description = (
            "movement. You've finished placing all 9 of your men (some "
            "may since have been captured). Move one piece to an ADJACENT "
            "empty point."
        )
        phase_instructions = (
            "Move one of your pieces along a drawn board line to an empty "
            "adjacent point. If the move forms a mill, you will capture next "
            "turn."
        )
    elif phase == "flying":
        phase_description = (
            "flying. You have only 3 pieces left, so you may move any of them to ANY empty point on the board."
        )
        phase_instructions = (
            "Move one of your three remaining pieces to any empty point. One more piece lost and you lose the game."
        )
    else:
        phase_description = phase
        phase_instructions = ""

    adjacency_lines = "\n".join(f"  {p}: " + ", ".join(str(n) for n in _POINT_NEIGHBORS[p]) for p in range(_NUM_POINTS))

    history_lines = _build_move_history_lines(observation.get("serializedGameAndState", "") or "")
    move_history_block = (
        "\n".join(f"  {i + 1}. {line}" for i, line in enumerate(history_lines)) if history_lines else "  (none yet)"
    )

    def _fmt_mills(mills: Sequence[tuple[int, int, int]]) -> str:
        return ", ".join(f"({a},{b},{c})" for a, b, c in mills) if mills else "(none)"

    prompt = NINE_MENS_MORRIS_PROMPT_TEMPLATE.format(
        board_ascii=_format_board(board),
        adjacency=adjacency_lines,
        state_summary=state_summary,
        my_side=my_side,
        my_piece=my_piece,
        opp_side=opp_side,
        opp_piece=opp_piece,
        my_mills=_fmt_mills(my_mills),
        opp_mills=_fmt_mills(opp_mills),
        phase_description=phase_description,
        phase_instructions=phase_instructions,
        turn_number=turn_number,
        move_history_block=move_history_block,
    )

    prompt += render_rethink_suffix(
        RETHINK_ILLEGAL,
        RETHINK_UNPARSABLE,
        previous_response,
        previous_action,
    )

    return prompt


def _match_action(raw: str, legals: Sequence[str]) -> str | None:
    """Map the model's concise move string to a canonical proxy legal.

    Accepted shapes (see prompt): ``N`` for placement or capture (the two
    are phase-exclusive, so only one canonical form ever appears in the
    current legals for a given N) and ``A-B`` (with optional ``>``, i.e.
    ``A->B``) for movement.
    """
    s = raw.strip().rstrip(".")
    if "-" in s:
        src, _, dst = s.partition("-")
        dst = dst.lstrip(">")
        try:
            a, b = int(src.strip()), int(dst.strip())
        except ValueError:
            return None
        target = f"Move point {a} -> point {b}"
        return target if target in legals else None
    try:
        n = int(s)
    except ValueError:
        return None
    for candidate in (f"Place at point {n}", f"Capture opponent piece at point {n}"):
        if candidate in legals:
            return candidate
    return None


def parse_response(
    response: str,
    legal_action_strings: Sequence[str],
) -> ParseResult:
    """Trust the model's JSON answer; let the rethink loop fix anything else."""
    return parse_json_action(response, legal_action_strings, matcher=_match_action)
