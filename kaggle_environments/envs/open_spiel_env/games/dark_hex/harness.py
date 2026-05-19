"""LLM harness for OpenSpiel Dark Hex.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.

The proxy in ``dark_hex_proxy.py`` emits a JSON observation per player; this
harness parses that JSON to render the player's *own view* of the board.
Cells the player has not yet learned about appear as ``.`` (could be empty or
hide an opponent piece).
"""

import json
import re
import sys
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{[^{}]*\"move\"\s*:\s*\"([^\"]+)\"[^{}]*\}", re.DOTALL)
_COORD_RE = re.compile(r"\b([a-z])\s*([0-9]+)\b", re.IGNORECASE)


# --- Prompt -----------------------------------------------------------------


DARK_HEX_PROMPT_TEMPLATE = """Let's play Dark Hex (imperfect-information Hex).

Rules:
- The board is a {num_rows}-row by {num_cols}-column hex grid. Adjacency
  follows standard Hex (each cell touches up to six neighbours).
- You play as {player_name} ({player_code}). Your goal is to form an unbroken
  chain of your stones from {connect_goal}.
- This is the IMPERFECT-INFORMATION variant. You can only see:
    * your own stones (shown as '{player_code}')
    * any opponent stones you have discovered by attempting to play on them
      (a "collision" -- the move is wasted but the cell becomes visible)
  Cells shown as '.' are UNKNOWN to you: they may be empty, or they may hide
  an opponent stone you have not yet bumped into.
- On each turn you nominate any cell that is not already known to be yours.
  If the cell is empty, your stone is placed there and it becomes the
  opponent's turn. If the cell is already occupied by an opponent stone, no
  stone is placed -- the cell is revealed to you and it REMAINS YOUR TURN
  (you will move again). The opponent learns nothing from your collision.
  Use collisions deliberately as a probing/scouting tool.

Your current view of the board ({player_code} = your stones,
opponent symbol shown only on revealed cells, '.' = unknown):

{board_render}

Move history (your nominated moves only -- you do not see the opponent's):
{move_history}

{last_move_line}
Legal moves you may play (already excludes cells you know are yours;
collision attempts on unknown cells are still legal):
{legal_moves}

It is your turn. Think about which cell most advances your connection
across {connect_goal} (or which probe most reduces your uncertainty), then
choose a move. The move MUST be exactly one of the legal moves listed above.

Respond with your reasoning followed by your final move in a JSON block:

```json
{{
  "move": "<coordinate from the legal list, e.g. a1, b3, d2>"
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
    """Pull the JSON observation emitted by ``DarkHexState.observation_string``."""
    raw = observation.get("observationString")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None


def _player_info(player_id: int) -> tuple[str, str, str]:
    """Return (display_name, board_code, connect_goal_text)."""
    if player_id == 0:
        return "Player X", "x", "the TOP edge to the BOTTOM edge"
    return "Player O", "o", "the LEFT edge to the RIGHT edge"


def _render_board(board: list[list[str]], num_cols: int) -> str:
    """Render the hex board with column letters, row numbers, and indentation.

    Each row is shifted half a cell to the right of the row above, mirroring
    the parallelogram layout of a Hex grid.
    """
    if not board:
        return "(board unavailable)"
    col_header = "    " + " ".join(chr(ord("a") + c) for c in range(num_cols))
    lines = [col_header]
    for r, row in enumerate(board):
        indent = " " * r
        cells = " ".join(row)
        lines.append(f"{indent}{r + 1:>2}  {cells}")
    return "\n".join(lines)


def _format_move_history(move_history: list[str]) -> str:
    if not move_history:
        return "(no moves yet)"
    return ", ".join(move_history)


def _last_move_line(move_history: list[str]) -> str:
    if not move_history:
        return "This is your first move."
    return f"Your most recent nominated move was: {move_history[-1]}"


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
    """Lowercase, strip whitespace and parentheses (so 'A1' or '(a, 1)' work)."""
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
        try:
            keys = list(observation.keys())
        except Exception:
            keys = ["<unkeyable>"]
        print(
            f"[dark_hex harness] get_legal_moves: empty obs. keys={keys}",
            file=sys.stderr,
        )
        return {}
    _, state = pyspiel.deserialize_game_and_state(serialized)
    actions = state.legal_actions()
    if not actions:
        print(
            f"[dark_hex harness] get_legal_moves: deserialized state has no "
            f"legal actions. current_player={state.current_player()} "
            f"is_terminal={state.is_terminal()}",
            file=sys.stderr,
        )
    return {a: state.action_to_string(a) for a in actions}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt rendered from the player's *own* view."""
    obs_json = _parse_observation(observation) or {}
    num_rows = obs_json.get("num_rows", 4)
    num_cols = obs_json.get("num_cols", 4)
    board = obs_json.get("board") or []

    player_id = observation.get("playerId", 0)
    player_name, player_code, connect_goal = _player_info(player_id)

    legal_action_strings = observation.get("legalActionStrings") or []
    if not legal_action_strings:
        legal_action_strings = list(get_legal_moves(observation).values())

    prompt = DARK_HEX_PROMPT_TEMPLATE.format(
        num_rows=num_rows,
        num_cols=num_cols,
        player_name=player_name,
        player_code=player_code,
        connect_goal=connect_goal,
        board_render=_render_board(board, num_cols),
        move_history=_format_move_history(move_history),
        last_move_line=_last_move_line(move_history),
        legal_moves=", ".join(legal_action_strings),
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
