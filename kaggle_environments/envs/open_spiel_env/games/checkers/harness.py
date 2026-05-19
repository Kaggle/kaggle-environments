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

from kaggle_environments.core_harness import ParseResult

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(
    r"\{[^{}]*\"move\"\s*:\s*\"([a-h][1-8][a-h][1-8])\"[^{}]*\}",
    re.DOTALL | re.IGNORECASE,
)
_MOVE_RE = re.compile(r"\b([a-h][1-8][a-h][1-8])\b", re.IGNORECASE)


# --- Prompt -----------------------------------------------------------------


CHECKERS_PROMPT_TEMPLATE = """Let's play Checkers (American draughts).

Rules: 8x8 board with files a-h (left-to-right) and ranks 1-8
(bottom-to-top). Player 0 ('o') starts on ranks 1-3 and moves first;
Player 1 ('+') starts on ranks 6-8. Pieces move diagonally one square
forward to an empty square. A capture jumps diagonally over an adjacent
opponent piece onto the empty square beyond, removing the jumped piece.

If any capture is available, you MUST take a capture (a man may capture
either forward or backward). When a piece reaches the opponent's back rank
(rank 8 for Player 0, rank 1 for Player 1) it is promoted to a king ('O'
for Player 0, '*' for Player 1) and may move and capture in all four
diagonal directions, one square at a time.

A multi-jump (capturing several opponent pieces in one turn) is represented
by separate moves on consecutive turns of the same player; the legal-move
list will only show the next single jump available.

You win by capturing all of your opponent's pieces, or by leaving them with
no legal move on their turn.

Board (rank labels on the left, file labels on top; '.' = empty,
'o' = Player 0 man, 'O' = Player 0 king, '+' = Player 1 man,
'*' = Player 1 king):
{board_ascii}

Piece counts: Player 0 men={p0_men}, kings={p0_kings}; Player 1 men={p1_men},
kings={p1_kings}.

You are Player {player_label} ('{my_piece}').
Move number: {move_number}
Last move played: {last_move}
Moves you have played so far: {move_history}

Action notation: ``<from><to>`` -- four lowercase characters, e.g.
``a3b4`` (slide a3 to b4) or ``d6b4`` (capture jump from d6 to b4 over
the opponent piece on c5).

You MUST pick one of the legal moves listed below: {legal_moves}.

Respond with your reasoning followed by your final move in a JSON block:

```json
{{
  "move": "<one of the legal moves>"
}}
```

Failure to output your final answer in the specified format, or selecting a
move that is not in the legal list, will result in a loss.
"""


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested move "{previous_action}" but it is NOT in the legal move list.
Reconsider and pick one of the legal moves exactly.
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


def _extract_move_from_json(response: str) -> str | None:
    """Pull the move from a ```json``` block or a bare ``{"move": "..."}``."""
    match = _JSON_BLOCK_RE.search(response)
    if match:
        try:
            data = json.loads(match.group(1))
            move = data.get("move")
            if move is None:
                return None
            return str(move).strip().lower()
        except json.JSONDecodeError:
            pass
    bare = _BARE_JSON_RE.search(response)
    if bare:
        return bare.group(1).strip().lower()
    return None


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
    last_move = state.get("last_move") or "(none yet)"
    piece_counts = state.get("piece_counts") or {}
    my_piece = "o" if player_id == 0 else "+"

    legal_action_strings = observation.get("legalActionStrings") or []
    if not legal_action_strings:
        legal_action_strings = list(get_legal_moves(observation).values())
    legal_moves_str = ", ".join(sorted(legal_action_strings)) or "(none)"

    move_history_str = ", ".join(move_history) if move_history else "None"

    prompt = CHECKERS_PROMPT_TEMPLATE.format(
        board_ascii=_format_board_ascii(board),
        p0_men=piece_counts.get("o", 0),
        p0_kings=piece_counts.get("O", 0),
        p1_men=piece_counts.get("+", 0),
        p1_kings=piece_counts.get("*", 0),
        player_label=player_id,
        my_piece=my_piece,
        move_number=move_number,
        last_move=last_move,
        move_history=move_history_str,
        legal_moves=legal_moves_str,
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

    for token in _MOVE_RE.findall(response):
        tok = token.lower()
        if tok in legal_set:
            return ParseResult(legal_action=legal_set[tok], raw_action=raw or tok)

    return ParseResult(legal_action=None, raw_action=raw)
