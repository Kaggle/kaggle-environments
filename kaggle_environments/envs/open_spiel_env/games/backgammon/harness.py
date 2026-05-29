"""LLM harness for OpenSpiel Backgammon.

Drop the body of this file into the notebook attached to the competition via
``HarnessKernelId``. The auto-generated ``main.py`` calls these three
module-level functions: ``get_legal_moves``, ``generate_prompt``,
``parse_response``.

Backgammon is a two-player race. Player 0 ('x') and Player 1 ('o') each have
fifteen checkers. The board has 24 points plus a bar and a bear-off tray.
Both dice are rolled by the game (the framework hides chance nodes from
agents), and on each turn the active player consumes both dice values --
each die moves one checker that many pips toward home. Whichever player
bears off all fifteen checkers first wins.

OpenSpiel exposes legal moves with action strings of the form
``"<action_id> - <notation>"`` (for example ``"648 - Bar/21 Bar/20"`` or
``"0 - 24/23 24/22"``). Notation is *player-relative*: each player counts
from their own 1-point. The proxy observation, by contrast, is in
*absolute* OpenSpiel coordinates (``board[0..23]``). The harness translates
between the two so the prompt and the action notation use the same
numbering as the legal-move strings.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult, extract_last_json_object

# Players: 0 = X, 1 = O (matches the proxy's "x"/"o" labels).
_PLAYER_LABELS = {0: "x", 1: "o"}


# --- Coordinate conversion --------------------------------------------------


def _abs_to_point(player_id: int, abs_pos: int) -> int:
    """OpenSpiel absolute pos (0..23) -> player-relative point (1..24).

    Player O numbers points 1..24 in increasing OpenSpiel order; player X
    uses the mirror, so their 1-point is OpenSpiel pos 23.
    """
    if player_id == 1:
        return abs_pos + 1
    return 24 - abs_pos


# --- Prompt -----------------------------------------------------------------


BACKGAMMON_PROMPT_TEMPLATE = """Let's play Backgammon.

Rules: Each player has 15 checkers. Player 0 plays 'x' and player 1 plays
'o'. The board has 24 points numbered 1..24 from your own perspective:
points 1-6 are your home board (where you bear off from), 7-12 your outer
board, 13-18 your opponent's outer board, and 19-24 your opponent's home
board. Your checkers move from high-numbered points down toward your
1-point, then off the board. Your opponent moves the opposite way on the
same board.

The two dice are already rolled for you. On your turn you must use BOTH
dice if possible -- each die moves one of your checkers that many pips
toward your 1-point. You may move two different checkers, or use both dice
on the same checker. A move is legal only if the destination point is (a)
empty, (b) occupied by your own checkers, or (c) occupied by exactly one
opposing checker (a "blot") -- landing on a blot sends it to the bar and
the destination becomes yours. A point with two or more opposing checkers
is blocked.

If you have any checker on the bar you MUST re-enter it before making any
other move. Bar entries go to your opponent's home board: a die value of N
brings the checker to your point (25 - N). Once every checker is in your
home board (points 1-6) you may bear off: a die of N bears off the checker
on your N-point. If no checker sits on your N-point, you may bear off the
checker furthest from home, but only when N exceeds the highest occupied
point. If you cannot use either die you must Pass.

You win by bearing off all fifteen of your checkers first.

Dice rolled this turn: {dice_str}
Move number: {move_number}
Bar -- yours: {my_bar}, opponent's: {opp_bar}
Borne off -- yours: {my_off}, opponent's: {opp_off}

Your checkers (point: count):
{my_points}

Opponent's checkers (point: count):
{opp_points}

Moves you have played so far: {move_history}

You are Player {player_id} ('{my_piece}'). Choose a single legal move that
uses BOTH dice when possible.

Action notation: use the part AFTER the ``" - "`` in legal-move strings.
Examples (your own perspective in every case):
* ``24/23 24/22`` -- two different checkers, one die each.
* ``13/11 13/8`` -- two different starting points.
* ``24/22/21`` -- same checker uses both dice in sequence (24 -> 22 -> 21).
* ``13/11(2)`` -- doubles: the same start/end is played twice.
* ``Bar/22`` -- enter from the bar onto your 22-point.
* ``6/Off`` -- bear off the checker on your 6-point.
* ``13/8*`` -- a hit (the ``*`` is added automatically by the game).
* ``Pass`` -- only when no legal move exists.

Tradition is to list moves with the higher starting point first.

Respond with your reasoning followed by your move in a JSON block:

```json
{{
  "move": "<notation>, e.g. 24/23 24/22"
}}
```

Failure to output your final answer in the specified format, or selecting
an illegal move, will result in a loss.
"""


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested move "{previous_action}" but it is not a legal move.
Reconsider the dice and your checker positions, then pick a legal move
using the notation after the ``" - "`` in legal-move strings.
"""


# --- Helpers ----------------------------------------------------------------


def _parse_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the structured backgammon state dict out of the observation."""
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


def _format_points_for_player(
    board: Sequence[Mapping[str, Any] | None],
    player_id: int,
    want_label: str,
) -> str:
    """Render occupied points (for one side) in player-relative numbering."""
    entries: list[tuple[int, int]] = []  # (player_point, count)
    for abs_pos, slot in enumerate(board):
        if not slot:
            continue
        if slot.get("player") != want_label:
            continue
        point = _abs_to_point(player_id, abs_pos)
        entries.append((point, int(slot.get("count", 0))))
    if not entries:
        return "  (none on the board)"
    entries.sort(key=lambda t: -t[0])  # high to low, matches play order
    return "\n".join(f"  point {pt:>2}: {ct}" for pt, ct in entries)


def _format_dice(dice: Sequence[Mapping[str, Any]]) -> str:
    """Render the dice as ``"3, 5"`` (omit dice already consumed)."""
    if not dice:
        return "(none rolled)"
    remaining = [str(d.get("value")) for d in dice if not d.get("used")]
    if not remaining:
        return "(both dice already used)"
    return ", ".join(remaining)


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
    """Build the LLM prompt for the current backgammon state."""
    state = _parse_observation_payload(observation)
    player_id = int(observation.get("playerId", 0))
    my_label = _PLAYER_LABELS.get(player_id, "x")
    opp_label = "o" if my_label == "x" else "x"

    board = state.get("board") or [None] * 24
    bar = state.get("bar") or {"x": 0, "o": 0}
    off = state.get("off") or {"x": 0, "o": 0}
    dice = state.get("dice") or []
    move_number = state.get("move_number", 0)

    move_history_str = ", ".join(move_history) if move_history else "None"

    prompt = BACKGAMMON_PROMPT_TEMPLATE.format(
        dice_str=_format_dice(dice),
        move_number=move_number,
        my_bar=bar.get(my_label, 0),
        opp_bar=bar.get(opp_label, 0),
        my_off=off.get(my_label, 0),
        opp_off=off.get(opp_label, 0),
        my_points=_format_points_for_player(board, player_id, my_label),
        opp_points=_format_points_for_player(board, player_id, opp_label),
        move_history=move_history_str,
        player_id=player_id,
        my_piece=my_label,
    )

    if previous_response is not None:
        prompt += RETHINK_SUFFIX.format(
            previous_response=previous_response[:500],
            previous_action=previous_action or "(could not parse)",
        )

    return prompt


# --- Parsing ----------------------------------------------------------------


def _strip_action_id_prefix(action_string: str) -> str:
    """``"648 - Bar/21 Bar/20"`` -> ``"Bar/21 Bar/20"``.

    Legal-move strings from OpenSpiel always start with the action id and a
    ``" - "`` separator; the LLM is asked to write only the right-hand side.
    """
    sep = " - "
    idx = action_string.find(sep)
    return action_string[idx + len(sep) :] if idx >= 0 else action_string


def _normalize_notation(notation: str) -> str:
    """Lowercase + collapse whitespace so model and reference compare equal."""
    return re.sub(r"\s+", " ", notation.strip().lower())


def _extract_move_from_json(response: str) -> str | None:
    """Pull the move string out of the LAST JSON object in the response."""
    data = extract_last_json_object(response, required_keys=("move",))
    if data is None:
        return None
    move = str(data.get("move") or "").strip()
    return move or None


def parse_response(
    response: str,
    legal_action_strings: Sequence[str],
) -> ParseResult:
    """Extract a legal Backgammon move from the LLM response.

    Tries a ```json``` block first, then a bare ``{"move": "..."}``, then
    falls back to scanning the raw response for any legal move notation.
    Matching is case-insensitive and tolerant of whitespace.
    """
    # Build {normalized_notation: original_legal_string} for matching.
    legal_by_notation: dict[str, str] = {}
    for legal in legal_action_strings:
        notation = _strip_action_id_prefix(legal)
        legal_by_notation[_normalize_notation(notation)] = legal

    raw = _extract_move_from_json(response)
    if raw is not None:
        normalized = _normalize_notation(raw)
        if normalized in legal_by_notation:
            return ParseResult(legal_action=legal_by_notation[normalized], raw_action=raw)

    # Fallback: scan the response for a legal notation substring. Pick the
    # one whose rightmost occurrence is latest (models enumerate rejected
    # options before stating their final move). Tie-break on length so that
    # a longer notation beats a shorter prefix like ``Pass``.
    normalized_response = _normalize_notation(response)
    best_end = -1
    best_notation: str | None = None
    for notation in legal_by_notation:
        pos = normalized_response.rfind(notation)
        if pos < 0:
            continue
        end = pos + len(notation)
        if end > best_end or (end == best_end and len(notation) > len(best_notation or "")):
            best_end = end
            best_notation = notation
    if best_notation is not None:
        return ParseResult(
            legal_action=legal_by_notation[best_notation],
            raw_action=raw or best_notation,
        )

    return ParseResult(legal_action=None, raw_action=raw)
