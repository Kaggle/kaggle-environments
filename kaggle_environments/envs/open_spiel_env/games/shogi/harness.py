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

# Importing the proxy has the side effect of registering ShogiGame with
# pyspiel, so `deserialize_game_and_state` in the fallback path below
# returns a ShogiState whose `observation_string(0)` emits structured
# JSON (not bare SFEN). Without this import, a stripped-down invocation
# context could silently fall through to `return {}` and hand the model
# a prompt with "(unavailable)" board / empty hands / no history.
from kaggle_environments.envs.open_spiel_env.games.shogi import shogi_proxy  # noqa: F401

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

Your pieces on the board (source-square list for board moves; unpromoted
letters, ``+`` prefix marks promoted): {own_roster}

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
{diagnosis}
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

# Iteration order for hand / roster displays (major -> minor).
_PIECE_ORDER = ["K", "R", "B", "G", "S", "N", "L", "P"]

EMPTY_CELL = "."


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


def _square_label(row: int, col: int) -> str:
    """Board index (row 0-8 top-down, col 0-8 left-to-right) -> USI square."""
    return f"{_FILE_LABELS[col]}{_RANK_LABELS[row]}"


def _parse_square(square: str) -> tuple[int, int] | None:
    """USI square (e.g. ``"7g"``) -> ``(row, col)`` board indices, or ``None``."""
    if len(square) != 2:
        return None
    try:
        col = _FILE_LABELS.index(square[0])
        row = _RANK_LABELS.index(square[1])
    except ValueError:
        return None
    return row, col


def _format_own_roster(board: Sequence[Sequence[str]], player_id: int) -> str:
    """List the current player's on-board pieces grouped by type.

    Rendered as e.g. ``"K:5i, G:4i,6i, ..."``. The point is to give the
    model an explicit source-square list so it stops trying to move
    opponent squares or empty squares -- the two most common illegal
    board moves in the replay archive. Piece letters are always shown
    uppercase (matching USI drop notation and the hand rendering) and
    promoted pieces are prefixed with ``+``.
    """
    if not board:
        return "(unavailable)"
    want_uppercase = player_id == 0
    grouped: dict[str, list[str]] = {}
    for row_idx, row in enumerate(board):
        for col_idx, cell in enumerate(row):
            if cell == EMPTY_CELL:
                continue
            letter = cell[-1]  # last char; strips optional '+'
            is_sente = letter.isupper()
            if is_sente != want_uppercase:
                continue
            key = ("+" if cell.startswith("+") else "") + letter.upper()
            grouped.setdefault(key, []).append(_square_label(row_idx, col_idx))
    if not grouped:
        return "(none)"
    ordered: list[str] = []
    for base in _PIECE_ORDER:
        for prefix in ("", "+"):
            key = prefix + base
            if key in grouped:
                ordered.append(f"{key}:{','.join(grouped[key])}")
    return "; ".join(ordered) if ordered else "(none)"


def _diagnose_illegal_move(
    move: str,
    board: Sequence[Sequence[str]],
    captured: Mapping[str, Mapping[str, int]],
    player_id: int,
) -> str:
    """Explain WHY ``move`` is illegal from the given position.

    Returns a short one-sentence hint. Falls back to a generic string
    when the move syntax is unrecognized -- the engine already rejected
    it, so this only needs to help common cases (empty source,
    opponent-source, out-of-hand drop, occupied-drop, own-square
    capture, non-promotable geometry).
    """
    if not move:
        return ""
    side_label = "Sente" if player_id == 0 else "Gote"
    want_uppercase = player_id == 0

    # Drop: <UPPERCASE_PIECE>*<square>
    if "*" in move:
        parts = move.split("*")
        if len(parts) != 2 or len(parts[0]) != 1 or not parts[0].isalpha():
            return "Reason: drop notation must be <PIECE>*<square>, e.g. P*5e."
        piece = parts[0].upper()
        sq = _parse_square(parts[1])
        if sq is None:
            return f"Reason: {parts[1]!r} is not a valid square (files 1-9, ranks a-i)."
        row, col = sq
        # Hand check
        side_key = "b" if player_id == 0 else "w"
        # SFEN hand keys are uppercase for Sente, lowercase for Gote.
        hand = captured.get(side_key) or {}
        hand_letter = piece if want_uppercase else piece.lower()
        if hand.get(hand_letter, 0) <= 0:
            return f"Reason: you have no {piece} in hand to drop."
        # Occupied target
        if board and board[row][col] != EMPTY_CELL:
            return f"Reason: {parts[1]} is occupied by {board[row][col]!r}; drops must target empty squares."
        # Back-rank restrictions
        rank_letter = _RANK_LABELS[row]
        if piece == "P" or piece == "L":
            if (player_id == 0 and rank_letter == "a") or (player_id == 1 and rank_letter == "i"):
                return f"Reason: dropping a {piece} on the opponent's back rank leaves it with no legal move."
        if piece == "N":
            if (player_id == 0 and rank_letter in ("a", "b")) or (player_id == 1 and rank_letter in ("h", "i")):
                return f"Reason: dropping a knight this deep leaves it with no legal forward move."
        # Nifu (pawn drop into a file already containing your unpromoted pawn)
        if piece == "P" and board:
            own_pawn = "P" if want_uppercase else "p"
            for r in range(9):
                if board[r][col] == own_pawn:
                    return (
                        f"Reason: nifu -- file {_FILE_LABELS[col]} already contains "
                        f"one of your unpromoted pawns ({own_pawn} on {_square_label(r, col)})."
                    )
        return "Reason: this drop violates a drop rule (nifu, uchifuzume, or a back-rank restriction)."

    # Board move: <from><to> with optional trailing '+'.
    promotes = move.endswith("+")
    core = move[:-1] if promotes else move
    if len(core) != 4:
        return "Reason: board moves are <from><to>, four characters like 7g7f (add + to promote)."
    fr_sq = _parse_square(core[:2])
    to_sq = _parse_square(core[2:4])
    if fr_sq is None or to_sq is None:
        return "Reason: coordinates must use files 1-9 (right to left) and ranks a-i (top to bottom)."
    if fr_sq == to_sq:
        return "Reason: from-square and to-square are the same."
    if not board:
        return ""
    fr_row, fr_col = fr_sq
    to_row, to_col = to_sq
    piece = board[fr_row][fr_col]
    if piece == EMPTY_CELL:
        return f"Reason: {core[:2]} is empty; there is no piece to move."
    piece_letter = piece[-1]
    is_sente_piece = piece_letter.isupper()
    if is_sente_piece != want_uppercase:
        owner = "Sente" if is_sente_piece else "Gote"
        return f"Reason: the piece on {core[:2]} ({piece!r}) is {owner}'s -- you are {side_label} and may only move your own pieces."
    dest = board[to_row][to_col]
    if dest != EMPTY_CELL:
        dest_is_sente = dest[-1].isupper()
        if dest_is_sente == want_uppercase:
            return f"Reason: {core[2:4]} is occupied by your own {dest!r}; you cannot capture your own pieces."
    if promotes:
        # Promotion requires the piece to touch the promotion zone and to
        # be a promotable type (not K/G and not already promoted).
        base = piece_letter.upper()
        if piece.startswith("+"):
            return f"Reason: {piece!r} is already promoted; promotions cannot stack."
        if base in ("K", "G"):
            return f"Reason: {base} (king/gold) never promotes."
        promo_rows = (0, 1, 2) if player_id == 0 else (6, 7, 8)
        if fr_row not in promo_rows and to_row not in promo_rows:
            zone = "a, b, c" if player_id == 0 else "g, h, i"
            return (
                f"Reason: promotion requires the move to start in, end in, or leave "
                f"your promotion zone (ranks {zone}); {core[:2]}->{core[2:4]} touches neither."
            )
    return ""


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
        own_roster=_format_own_roster(board, player_id),
        move_number=move_number,
        last_move=last_move,
        full_history=full_history,
    )

    # Pre-fill {diagnosis} on the ILLEGAL template so render_rethink_suffix
    # (which only knows how to substitute {previous_action}) doesn't need
    # to grow shogi-specific parameters. Escape any braces in the
    # diagnosis before splicing -- the diagnosis text can contain
    # unvalidated model input (e.g. an unparseable drop square echoed
    # back), and render_rethink_suffix runs .format() on the whole
    # template afterwards, which would otherwise raise on a stray '{'.
    diagnosis = ""
    if previous_action:
        diagnosis = _diagnose_illegal_move(
            previous_action, board, captured, player_id
        )
    escaped_diagnosis = diagnosis.replace("{", "{{").replace("}", "}}")
    illegal_template = RETHINK_ILLEGAL.replace(
        "{diagnosis}", escaped_diagnosis + ("\n" if diagnosis else "")
    )

    prompt += render_rethink_suffix(
        illegal_template,
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
