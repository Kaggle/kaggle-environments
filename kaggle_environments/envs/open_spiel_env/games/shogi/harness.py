"""LLM harness for OpenSpiel Shogi.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.

Shogi (Japanese chess) is a two-player perfect-information game on a 9x9
board. Player 0 plays Sente ("Black", uppercase pieces) and moves first;
Player 1 plays Gote ("White", lowercase pieces). Captured pieces switch
sides and may be re-introduced ("dropped") onto the board on a later turn.

Action strings use OpenSpiel's USI notation: ``<from><to>`` for a board
move (e.g. ``"7g7f"``), with a trailing ``+`` for promotion
(e.g. ``"8h2b+"``), and ``<PIECE>*<square>`` for a drop (e.g. ``"P*5e"``).
Files are digits ``1..9`` (right-to-left from Sente's view); ranks are
letters ``a..i`` (top-to-bottom, so ``a`` is Gote's back rank and ``i`` is
Sente's back rank).
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

# --- Prompt -----------------------------------------------------------------


SHOGI_PROMPT_TEMPLATE = """Let's play Shogi (Japanese chess).

Rules: 9x9 board. Files are numbered 1-9 from right to left as Sente
(Player 0) looks at the board; ranks are lettered a-i from top to bottom,
so rank a is Gote's back rank and rank i is Sente's back rank. Sente
(Player 0, "Black") uses uppercase pieces and moves first; Gote (Player 1,
"White") uses lowercase pieces. A square is written as ``<file><rank>``,
e.g. ``7g`` or ``5e``.

Piece movement:
- K/k King: one square in any direction.
- G/g Gold general: one square in any direction except the two backward
  diagonals.
- S/s Silver general: one square straight forward or any of the four
  diagonals.
- N/n Knight: jumps to one of the two squares two ranks forward and one
  file to either side (only forward, cannot move sideways or backward).
- L/l Lance: any number of empty squares straight forward.
- B/b Bishop: any number of empty squares diagonally.
- R/r Rook: any number of empty squares orthogonally.
- P/p Pawn: one square straight forward. Pawns capture the same way they
  move (unlike chess).

"Forward" for Sente is toward rank a (upward on the board); for Gote it is
toward rank i (downward).

Promotion: the promotion zone is the opponent's three back ranks -- ranks
a, b, c for Sente; ranks g, h, i for Gote. When a piece moves into,
within, or out of the promotion zone you may choose to promote it (append
``+`` to the move). Promotion is compulsory when a non-promoted piece
would otherwise have no legal move next turn (a pawn or lance on the
opponent's back rank; a knight on either of the opponent's last two
ranks). Promoted pieces are shown with a
``+`` prefix on the board (e.g. ``+P`` is a promoted pawn). Promoted
pawn, lance, knight, and silver all move like a gold general. Promoted
bishop ("horse") moves like a bishop and also one square in any
orthogonal direction. Promoted rook ("dragon") moves like a rook and also
one square in any diagonal direction. Kings and golds never promote.

Captures and drops: when you capture an opponent piece it joins your
"hand" as an unpromoted piece of your colour and may later be dropped
onto any empty square on your turn instead of moving a piece on the
board. Drop notation is ``<PIECE>*<square>`` using the uppercase piece
letter regardless of colour, e.g. ``P*5e`` drops a pawn on 5e. Drop
restrictions:
- Pawn, lance, and knight may not be dropped where they would have no
  legal move next turn (a pawn or lance on the opponent's back rank; a
  knight on either of the opponent's last two ranks).
- You may not drop a pawn on a file that already contains one of your
  own unpromoted pawns ("nifu").
- You may not deliver immediate checkmate by dropping a pawn
  ("uchifuzume"). Delivering mate by dropping any other piece, or by a
  regular pawn move, is allowed.

Game end (there are five terminal conditions the engine enforces; the
king is never actually captured, because any move that would leave your
own king under attack is filtered out of your legal moves before you
see them):
- You LOSE the moment you have no legal move on your turn. This
  covers both checkmate and stalemate -- shogi has no stalemate-draw
  concept; running out of legal replies always loses.
- PERPETUAL CHECK is a LOSS for the checking side, not a draw. If the
  same position repeats a fourth time AND either side has delivered
  at least 6 checks in a row (the engine tracks consecutive checks
  per side and resets the counter to zero the moment the run of
  checks is broken), the side responsible for the checks loses. You
  cannot force a draw by chasing the opponent's king with endless
  checks -- you will lose.
- FOURFOLD REPETITION with no perpetual-check pattern is a DRAW: the
  same board, the same hands (pieces-in-hand count toward the
  position fingerprint), and the same side to move recur for the
  fourth time, and neither side has a run of 6+ consecutive checks.
- ENTERING KING is an automatic WIN for the moving side: if, after
  your move, your king sits inside your promotion zone (the
  opponent's three back ranks) AND your material points total at
  least 28, you win immediately -- no declaration required. Material
  points are counted as follows: every one of your pieces sitting
  inside your promotion zone contributes (rook, bishop, promoted
  rook, promoted bishop = 5 each; every other non-king piece = 1);
  every piece in your hand contributes the same way; the king
  itself and pieces still outside the enemy camp count zero.
- MUTUAL ENTERING KINGS is a DRAW when both kings sit inside their
  respective opponent's back three ranks after your move AND you
  did not meet the 28-point ENTERING KING threshold above. (The
  win check runs first, so a just-moved side with 28+ material
  wins outright even when the opposing king is also in their camp.)

Board (files 9-1 across the top, ranks a-i down the left side; '.' =
empty, uppercase = Sente, lowercase = Gote, '+X'/'+x' = promoted):
{board_ascii}

SFEN (Shogi Forsyth-Edwards Notation) for the same position: {sfen}
The four SFEN fields are: board (nine ``/``-separated ranks a..i, each
rank run-length-encoded where digits count empty squares, letters are
pieces, and a ``+`` prefix marks a promoted piece), side to move (``b``
= Sente, ``w`` = Gote), pieces in hand (``-`` if both empty; otherwise
concatenated ``<count><PIECE>`` entries, uppercase for Sente and
lowercase for Gote, count omitted when it is 1), and the SFEN full-move
counter (Sente + Gote reply = 1 full move).

Pieces in hand (rendered with uppercase piece letters for both sides
because USI drop notation uses ``<UPPERCASE_PIECE>*<square>``
regardless of the dropping side; the SFEN pieces-in-hand field above
still uses SFEN's uppercase-Sente / lowercase-Gote convention):
- Sente: {sente_hand}
- Gote: {gote_hand}

You are Player {player_label} ({side_label}, {piece_case} pieces).

Move number: {move_number}
Last move played: {last_move}
Moves played so far this game (both players, oldest first): {full_history}

Action notation reminder: a board move is ``<from><to>``, e.g. ``7g7f``
means "the piece on 7g moves to 7f". Append ``+`` to promote when the
move enters, stays within, or leaves the promotion zone: e.g. ``8h2b+``
means "the piece on 8h moves to 2b and promotes". A drop is
``<PIECE>*<square>`` using the uppercase piece letter, e.g. ``P*5e``
drops a pawn from hand onto 5e.

It is your turn. Choose a legal move.

Respond with your reasoning followed by your final move in a JSON block:

```json
{{
  "move": "<your_move>"
}}
```

For example: `{{"move": "7g7f"}}`

Failure to output your final answer in the specified format, or selecting
an illegal move, will result in a loss.
"""


RETHINK_ILLEGAL = """

You suggested move "{previous_action}" but this is not a legal move.
Reconsider the rules (piece movement, promotion zone, drop restrictions
including nifu and uchifuzume) and the current board, then pick a legal
move.

(Keep using the same JSON output format as before -- only the move value
needs to change.)
"""

RETHINK_UNPARSABLE = """

Your previous response ended with:
{previous_response}

No JSON answer could be parsed from that. Conclude your response with
your final move as JSON in a ```json fenced block, exactly as the
original instructions required:

```json
{{"move": "<your_move>"}}
```

For example: `{{"move": "7g7f"}}` (board move), `{{"move": "8h2b+"}}`
(board move with promotion), or `{{"move": "P*5e"}}` (drop from hand).

The move you choose must also be legal in the current state.
"""


# --- Helpers ----------------------------------------------------------------


_FILE_LABELS = "987654321"  # column 0 is file 9, column 8 is file 1
_RANK_LABELS = "abcdefghi"  # row 0 is rank a, row 8 is rank i


def _parse_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the structured shogi state dict out of the observation."""
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
    """Render the 9x9 board with file labels on top and rank labels on the left.

    ``board[0]`` is rank ``a`` (top, Gote's back rank); ``board[8]`` is
    rank ``i`` (bottom, Sente's back rank). Column 0 is file 9, column 8
    is file 1 (files are numbered right-to-left from Sente's view).
    Cells are 2 characters wide so that promoted pieces (``+X``) align
    with unpromoted pieces (``X ``).
    """
    if not board:
        return "(unavailable)"
    file_header = "   " + " ".join(f"{f:>2}" for f in _FILE_LABELS)
    lines = [file_header]
    for r, row in enumerate(board):
        cells = " ".join(f"{cell:>2}" for cell in row)
        lines.append(f" {_RANK_LABELS[r]} {cells}")
    return "\n".join(lines)


def _format_hand(hand: Mapping[str, int]) -> str:
    """Format a per-side captured-piece dict as e.g. ``"2P, N"`` or ``"(empty)"``.

    Piece letters are always rendered in uppercase, regardless of which
    side's hand this is, to match USI drop notation (drops always use
    ``<UPPERCASE_PIECE>*<square>``). The section label ("Sente: ..."
    vs "Gote: ...") already disambiguates ownership.
    """
    if not hand:
        return "(empty)"
    order = ["R", "B", "G", "S", "N", "L", "P"]
    items: list[str] = []
    for key in order:
        for piece, count in hand.items():
            if piece.upper() == key:
                letter = key
                items.append(f"{count}{letter}" if count > 1 else letter)
    for piece, count in hand.items():
        if piece.upper() not in order:
            letter = piece.upper()
            items.append(f"{count}{letter}" if count > 1 else letter)
    return ", ".join(items) if items else "(empty)"


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
    """Build the LLM prompt for the current shogi position."""
    del move_history  # Per-agent history omits opponent moves; use full history from proxy.
    state = _parse_observation_payload(observation)
    player_id = observation.get("playerId", 0)

    board = state.get("board") or []
    move_number = state.get("move_number", 1)
    last_move_raw = state.get("last_move")
    last_move = last_move_raw or "(none yet)"
    captured = state.get("captured") or {}
    sente_hand = _format_hand(captured.get("b") or {})
    gote_hand = _format_hand(captured.get("w") or {})
    sfen = state.get("sfen") or "(unavailable)"

    full_history_list = state.get("move_history") or []
    full_history = ", ".join(full_history_list) if full_history_list else "None"

    side_label = "Sente" if player_id == 0 else "Gote"
    piece_case = "uppercase" if player_id == 0 else "lowercase"

    prompt = SHOGI_PROMPT_TEMPLATE.format(
        board_ascii=_format_board_ascii(board),
        sfen=sfen,
        sente_hand=sente_hand,
        gote_hand=gote_hand,
        player_label=player_id,
        side_label=side_label,
        piece_case=piece_case,
        move_number=move_number,
        last_move=last_move,
        full_history=full_history,
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
