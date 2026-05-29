"""LLM harness for OpenSpiel Checkers.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.

Checkers (American draughts) is a two-player game on an 8x8 board. Player 0
('o', moves first) starts on the bottom three rows; Player 1 ('+') starts on
the top three rows. Pieces move and capture diagonally one square at a time;
captures jump over an adjacent opponent onto the empty square beyond. A piece
that reaches the opponent's back rank is promoted to a king ('O' for player 0,
'*' for player 1) and may move and capture in all four diagonal directions.

Action strings are 4-character ``"<from><to>"`` coordinates such as ``"a3b4"``
or ``"d6b4"`` (a capture jumps two squares). Files are letters ``a..h``
(left-to-right); ranks are digits ``1..8`` (bottom-to-top).
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult, extract_last_json_object

_MOVE_RE = re.compile(r"\b([a-h][1-8][a-h][1-8])\b", re.IGNORECASE)


# --- Prompt -----------------------------------------------------------------


CHECKERS_PROMPT_TEMPLATE = """Let's play Checkers (American draughts).

Rules: 8x8 board with files a-h (left-to-right) and ranks 1-8
(bottom-to-top). Player 0 ('o') starts on ranks 1-3 and moves first;
Player 1 ('+') starts on ranks 6-8. Pieces move diagonally one square
forward to an empty square. A capture jumps diagonally over an adjacent
opponent piece onto the empty square beyond, removing the jumped piece.

If any capture is available, you MUST take a capture. When a piece
reaches the opponent's back rank (rank 8 for Player 0, rank 1 for Player 1)
it is promoted to a king ('O' for Player 0, '*' for Player 1) and may move
and capture in all four diagonal directions, one square at a time.

A multi-jump (capturing several opponent pieces in one turn) is represented
by separate moves on consecutive turns of the same player. On each
continuation turn only the piece that just captured may move, and it must
capture again; if more than one continuation jump is available you may
choose any one. The sequence ends as soon as the moving piece is crowned,
even if further captures would otherwise be possible.

You win by capturing all of your opponent's pieces, or by leaving them with
no legal move on their turn. The game is drawn if 40 consecutive plies
(counting both players' moves) pass without any capture; every capture
resets that counter to zero. There is no other draw condition.

Board (rank labels on the left, file labels on top; '.' = empty,
'o' = Player 0 man, 'O' = Player 0 king, '+' = Player 1 man,
'*' = Player 1 king):
{board_ascii}

Piece counts: Player 0 men={p0_men}, kings={p0_kings}; Player 1 men={p1_men},
kings={p1_kings}.

You are Player {player_label} ('{my_piece}').
Your men ('{my_piece}') are at: {my_men_squares}
Your kings ('{my_king}') are at: {my_king_squares}
Opponent men ('{opp_piece}') are at: {opp_men_squares}
Opponent kings ('{opp_king}') are at: {opp_king_squares}
Captures available this turn: {captures_flag}{capture_reminder}

Move number: {move_number}
Last move played: {last_move}
Moves you have played so far: {move_history}

Action notation: a four-character string ``<from><to>`` denotes "piece at
<from> moves to <to>"; both squares are lowercase file+rank such as ``a3``
or ``e5``. For example, ``a3b4`` means "the piece on a3 moves to b4" (a
one-square diagonal slide), and ``c3e5`` means "the piece on c3 jumps two
squares diagonally to e5" -- which is only a legal capture when an
opponent occupies the intermediate square d4 and e5 is empty. As Player
{player_label}, "forward" for your men means toward rank {forward_rank}.
Kings may move and capture in any diagonal direction.
{continuation_note}
It is your turn. Choose a legal move for one of your pieces, remembering
that if any capture is available you MUST take a capture.

Respond with your reasoning followed by your final move in a JSON block:

```json
{{
  "move": "<from><to>, e.g. a3b4"
}}
```

Failure to output your final answer in the specified format, or selecting
an illegal move, will result in a loss.
"""


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested move "{previous_action}" but it is not a legal move.
Reconsider and pick a legal move for one of your pieces (taking a capture
if any are available).
"""


# --- Helpers ----------------------------------------------------------------


def _parse_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the structured checkers state dict out of the observation."""
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
    """Render the 8x8 board with rank labels on the left and files on top.

    ``board[0]`` is rank 1 (bottom row); ``board[7]`` is rank 8 (top). We
    print ranks top-down so the visual board matches standard orientation.
    """
    if not board:
        return "(unavailable)"
    file_header = "  " + " ".join(chr(ord("a") + c) for c in range(len(board[0])))
    lines = [file_header]
    for r in range(len(board) - 1, -1, -1):
        lines.append(f"{r + 1} " + " ".join(board[r]))
    return "\n".join(lines)


def _list_pieces_of(
    board: Sequence[Sequence[str]], player_id: int
) -> tuple[list[str], list[str]]:
    """Return (men_squares, king_squares) in algebraic notation for player."""
    man_char = "o" if player_id == 0 else "+"
    king_char = "O" if player_id == 0 else "*"
    men: list[str] = []
    kings: list[str] = []
    for r, row in enumerate(board):
        for c, cell in enumerate(row):
            square = f"{chr(ord('a') + c)}{r + 1}"
            if cell == man_char:
                men.append(square)
            elif cell == king_char:
                kings.append(square)
    return men, kings


def _is_capture(action_string: str) -> bool:
    """A checkers capture jumps two ranks (e.g. d6b4); a slide moves one."""
    if len(action_string) != 4:
        return False
    try:
        return abs(int(action_string[1]) - int(action_string[3])) == 2
    except ValueError:
        return False


def _extract_move_from_json(response: str) -> str | None:
    """Pull the move string out of the LAST JSON object in the response."""
    data = extract_last_json_object(response, required_keys=("move",))
    if data is None:
        return None
    move = str(data.get("move") or "").strip().lower()
    return move or None


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
    """Build the LLM prompt for the current checkers state."""
    state = _parse_observation_payload(observation)
    player_id = observation.get("playerId", 0)

    board = state.get("board") or []
    move_number = state.get("move_number", 0)
    last_move_raw = state.get("last_move")
    last_move = last_move_raw or "(none yet)"
    piece_counts = state.get("piece_counts") or {}
    my_piece = "o" if player_id == 0 else "+"
    my_king = "O" if player_id == 0 else "*"
    opp_piece = "+" if player_id == 0 else "o"
    opp_king = "*" if player_id == 0 else "O"

    men, kings = _list_pieces_of(board, player_id)
    my_men_squares = ", ".join(men) if men else "(none)"
    my_king_squares = ", ".join(kings) if kings else "(none)"
    opp_men, opp_kings = _list_pieces_of(board, 1 - player_id)
    opp_men_squares = ", ".join(opp_men) if opp_men else "(none)"
    opp_king_squares = ", ".join(opp_kings) if opp_kings else "(none)"

    legal_moves = get_legal_moves(observation)
    captures_available = any(_is_capture(s) for s in legal_moves.values())
    captures_flag = "yes" if captures_available else "no"
    capture_reminder = (
        " (you MUST take a capture this turn)" if captures_available else ""
    )

    continuation_note = ""
    if (
        move_history
        and last_move_raw
        and _is_capture(move_history[-1])
        and move_history[-1].lower() == last_move_raw.lower()
    ):
        landed_square = move_history[-1][2:4].lower()
        continuation_note = (
            f"\nMulti-jump in progress: your previous capture "
            f"({move_history[-1]}) landed on {landed_square}. You must "
            f"capture again with the piece now on {landed_square} -- no "
            "other piece may move this turn. If more than one continuation "
            "jump is available, you may choose any of them.\n"
        )

    forward_rank = 8 if player_id == 0 else 1

    move_history_str = ", ".join(move_history) if move_history else "None"

    prompt = CHECKERS_PROMPT_TEMPLATE.format(
        board_ascii=_format_board_ascii(board),
        p0_men=piece_counts.get("o", 0),
        p0_kings=piece_counts.get("O", 0),
        p1_men=piece_counts.get("+", 0),
        p1_kings=piece_counts.get("*", 0),
        player_label=player_id,
        my_piece=my_piece,
        my_king=my_king,
        my_men_squares=my_men_squares,
        my_king_squares=my_king_squares,
        opp_piece=opp_piece,
        opp_king=opp_king,
        opp_men_squares=opp_men_squares,
        opp_king_squares=opp_king_squares,
        captures_flag=captures_flag,
        capture_reminder=capture_reminder,
        forward_rank=forward_rank,
        move_number=move_number,
        last_move=last_move,
        move_history=move_history_str,
        continuation_note=continuation_note,
    )

    if previous_response is not None:
        prompt += RETHINK_SUFFIX.format(
            previous_response=previous_response[:500],
            previous_action=previous_action or "(could not parse)",
        )

    return prompt


def parse_response(
    response: str,
    legal_action_strings: Sequence[str],
) -> ParseResult:
    """Extract a legal Checkers move from the LLM response.

    Tries a ```json``` block first, then a bare ``{"move": "..."}``, then
    falls back to scanning for any ``[a-h][1-8][a-h][1-8]`` token that
    matches a legal move.
    """
    legal_set = {legal.lower(): legal for legal in legal_action_strings}

    raw = _extract_move_from_json(response)
    if raw is not None and raw in legal_set:
        return ParseResult(legal_action=legal_set[raw], raw_action=raw)

    # Iterate in reverse so the *last* token mentioned wins -- models
    # typically enumerate rejected options before stating the final move.
    for token in reversed(_MOVE_RE.findall(response)):
        tok = token.lower()
        if tok in legal_set:
            return ParseResult(legal_action=legal_set[tok], raw_action=raw or tok)

    return ParseResult(legal_action=None, raw_action=raw)
