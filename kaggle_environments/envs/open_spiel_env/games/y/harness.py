"""LLM harness for OpenSpiel Y.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.

Y is a connection game on a triangular hex board. Each player tries to form
a single chain of their stones that touches all three sides of the triangle.
The proxy in ``y_proxy.py`` emits a JSON observation describing the board.
Action ids are encoded as ``row * board_size + col`` (0-indexed); the
algebraic action string is ``<col_letter><row_number>`` with row numbers
1-indexed (so action 0 is ``a1``).
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult

# Importing the proxy registers the ``y_proxy`` pyspiel game so that
# ``deserialize_game_and_state`` can rebuild it from the obs. Wrapped in
# try/except because the harness can be dropped into a notebook where the
# proxy module isn't on the path; in that case we rely on ``legalActions``
# being included in the observation directly.
try:
    from kaggle_environments.envs.open_spiel_env.games.y import (  # noqa: F401
        y_proxy,
    )
except Exception:
    pass

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{[^{}]*\"move\"\s*:\s*\"([^\"]+)\"[^{}]*\}", re.DOTALL)
_COORD_RE = re.compile(r"\b([a-z])\s*([0-9]+)\b", re.IGNORECASE)


# --- Prompt -----------------------------------------------------------------


Y_PROMPT_TEMPLATE = """Let's play Y on a triangular hex board of side length {board_size}.

Rules:
- The board is a triangle. Row 1 is the longest edge ({board_size} cells:
  a1..{last_col}1). Row {board_size} is the apex (a single cell, a{board_size}).
  Row r contains {board_size} - r + 1 cells, columns a..(letter for column
  {board_size} - r).
- Each cell is a hex with up to six neighbours. Two stones are connected if
  they sit on adjacent hexes. A "group" is a maximal set of one player's
  stones connected through their own stones.
- The three sides of the triangle are:
    * BOTTOM side: row 1 (a1, b1, c1, ..., {last_col}1).
    * LEFT side:   the leftmost cell of each row (a1, a2, a3, ..., a{board_size}).
    * RIGHT side:  the rightmost cell of each row -- this is the diagonal
      hypotenuse running from {last_col}1 up to a{board_size}.
  The three corner cells (a1, {last_col}1, a{board_size}) belong to two sides
  each.
- You play as {player_name} ({player_glyph}). On each turn you place one stone
  on any empty cell. You win as soon as one of your groups touches ALL THREE
  sides of the triangle simultaneously. There are no draws and the game
  cannot end in a stalemate.

Board (each row is indented so neighbour relationships are visually
preserved; '.' = empty, 'x' = Player X stone, 'o' = Player O stone):

{board_render}

Last move: {last_move_line}

Move history (oldest first, alternating X then O):
{move_history}

Legal moves you may play (every empty cell):
{legal_moves}

It is your turn. Think briefly about how to extend or defend your chain so
that it eventually touches all three sides, then choose your move. The move
MUST be exactly one of the legal moves listed above.

Respond with your reasoning followed by your final move in a JSON block:

```json
{{
  "move": "<coordinate from the legal list, e.g. a1, c3, d5>"
}}
```

Failure to output your final answer in the specified format, or selecting a
move that is not in the legal list, will result in a loss.
"""


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested move "{previous_action}" but it is NOT in the legal moves list.
Reconsider and pick a coordinate that appears verbatim in the legal moves above.
"""


# --- Helpers ----------------------------------------------------------------


def _parse_observation(observation: Mapping[str, Any]) -> dict[str, Any] | None:
    """Pull the JSON observation emitted by ``YState.observation_string``."""
    raw = observation.get("observationString")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None


def _player_info(player_id: int) -> tuple[str, str]:
    """Return (display_name, board_glyph) for a player id."""
    if player_id == 0:
        return "Player X", "x"
    return "Player O", "o"


def _render_board(board: list[list[str | None]], board_size: int) -> str:
    """Render the triangular board with the apex at the top.

    The proxy stores ``board[r]`` for algebraic row ``r + 1``, so ``board[0]``
    is the longest (bottom) row and ``board[board_size - 1]`` is the apex.
    We render top-down (apex first) so the bottom edge appears at the bottom
    of the rendered text, matching how a physical Y board is usually drawn.
    """
    if not board:
        return "(board unavailable)"
    lines: list[str] = []
    for r in range(board_size - 1, -1, -1):
        row = board[r]
        # Each step up the triangle shifts the row half a cell right; we use
        # one space of indent per step so neighbours line up under each other.
        indent = " " * (board_size - 1 - r)
        cells = " ".join(cell if cell is not None else "." for cell in row)
        lines.append(f"{indent}row {r + 1:>2}  {cells}")
    # Footer with column letters underneath the longest (bottom) row.
    col_letters = " ".join(chr(ord("a") + c) for c in range(board_size))
    lines.append(f"{' ' * (board_size - 1)}        {col_letters}")
    return "\n".join(lines)


def _format_move_history(move_history: list[str]) -> str:
    if not move_history:
        return "(no moves yet)"
    return ", ".join(move_history)


def _last_move_line(state: Mapping[str, Any]) -> str:
    last = state.get("last_move")
    if not last:
        return "(none -- this is the first move)"
    return str(last)


def _extract_move_from_json(response: str) -> str | None:
    match = _JSON_BLOCK_RE.search(response)
    if match:
        try:
            data = json.loads(match.group(1))
            move = str(data.get("move", "")).strip()
            if move:
                return move
        except json.JSONDecodeError:
            pass
    bare = _BARE_JSON_RE.search(response)
    if bare:
        return bare.group(1).strip()
    return None


def _normalize(move: str) -> str:
    """Lowercase and strip whitespace/punctuation so 'A1' / '(a, 1)' match."""
    cleaned = re.sub(r"[\s()\[\],.]", "", move).lower()
    return cleaned


def _match_move_to_legal(move: str, legal_moves: Sequence[str]) -> str | None:
    if not move:
        return None
    if move in legal_moves:
        return move
    target = _normalize(move)
    for legal in legal_moves:
        if _normalize(legal) == target:
            return legal
    return None


# --- Public functions (called by main.py) -----------------------------------


def get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str]:
    """Return ``{action_id: coordinate_string}`` for the current state."""
    legal_actions = observation.get("legalActions")
    legal_action_strings = observation.get("legalActionStrings")
    if legal_actions and legal_action_strings:
        return dict(zip(legal_actions, legal_action_strings))

    serialized = observation.get("serializedGameAndState", "")
    if not serialized:
        return {}
    try:
        _, state = pyspiel.deserialize_game_and_state(serialized)
    except Exception:
        return {}
    return {a: state.action_to_string(a) for a in state.legal_actions()}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current state."""
    obs_json = _parse_observation(observation) or {}
    board_size = obs_json.get("board_size", 8)
    board = obs_json.get("board") or []

    player_id = observation.get("playerId", 0)
    player_name, player_glyph = _player_info(player_id)

    legal_action_strings = observation.get("legalActionStrings") or []
    if not legal_action_strings:
        legal_action_strings = list(get_legal_moves(observation).values())

    last_col = chr(ord("a") + board_size - 1)

    prompt = Y_PROMPT_TEMPLATE.format(
        board_size=board_size,
        last_col=last_col,
        player_name=player_name,
        player_glyph=player_glyph,
        board_render=_render_board(board, board_size),
        last_move_line=_last_move_line(obs_json),
        move_history=_format_move_history(move_history),
        legal_moves=", ".join(legal_action_strings) if legal_action_strings else "(none)",
    )

    if previous_response is not None:
        prompt += RETHINK_SUFFIX.format(
            previous_response=previous_response[:500],
            previous_action=previous_action or "(could not parse)",
        )

    return prompt


def parse_response(
    response: str, legal_action_strings: Sequence[str],
) -> ParseResult:
    """Extract a legal coordinate (e.g. ``a1``) from the model response."""
    raw = _extract_move_from_json(response)
    if raw is not None:
        matched = _match_move_to_legal(raw, legal_action_strings)
        if matched is not None:
            return ParseResult(legal_action=matched, raw_action=raw)

    # Fallback: scan the prose for any "<letter><digits>" coordinate token.
    for match in _COORD_RE.finditer(response):
        candidate = f"{match.group(1).lower()}{match.group(2)}"
        matched = _match_move_to_legal(candidate, legal_action_strings)
        if matched is not None:
            return ParseResult(legal_action=matched, raw_action=raw or candidate)

    return ParseResult(legal_action=None, raw_action=raw)
