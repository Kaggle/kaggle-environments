"""LLM harness for OpenSpiel Hive.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The framework calls the three module-level functions
``get_legal_moves``, ``generate_prompt`` and ``parse_response``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult, create_agent_fn

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r'\{[^{}]*"move"\s*:\s*"([^"]+)"[^{}]*\}', re.DOTALL)


# --- Prompt -----------------------------------------------------------------


HIVE_PROMPT_TEMPLATE = """Let's play Hive.

Hive is a tile-placement and movement game played on a hexagonal grid with no
fixed board -- the playing surface is the tiles themselves. The objective is to
completely surround the opponent's Queen Bee (all six neighbouring positions
occupied by any pieces of either colour). If both Queens are surrounded on the
same move, the game is a draw.

PIECES (per player, prefixed ``w`` for White / ``b`` for Black):
  Q   Queen Bee       x1  -- moves 1 hex around the hive
  A1..A3  Soldier Ant x3  -- slides any number of hexes around the hive
  G1..G3  Grasshopper x3  -- jumps in a straight line over >=1 contiguous pieces
  S1..S2  Spider      x2  -- slides exactly 3 hexes
  B1..B2  Beetle      x2  -- moves 1 hex; may climb on top of other pieces
  M       Mosquito    x1  -- copies the movement of any adjacent piece
  L       Ladybug     x1  -- moves exactly 2 on top then 1 down
  P       Pillbug     x1  -- moves 1 hex; may "throw" an adjacent piece to an
                              adjacent empty hex (uses the player's turn)

PLACEMENT RULES:
  - White's first move is at the origin; Black's first move must touch it.
  - Each subsequent placement must touch at least one friendly piece AND must
    NOT touch any opposing piece.
  - The Queen Bee may NOT be placed on a player's very first move
    (Tournament Opening Rule -- this applies to both White's first move and
    Black's first move), but it MUST be placed by each player's 4th move at
    the latest.

MOVEMENT RULES:
  - Pieces may only move once their player has placed their Queen.
  - The One Hive rule: at no point during a move may the hive be split into
    disconnected groups. Pieces "pinned" by this rule cannot move.
  - The Freedom to Move rule: a sliding piece must be able to physically slide
    between its current and target hex (the two adjacent hexes shared with the
    move cannot both be occupied).
  - Beetles, Mosquitoes acting as Beetles, and Ladybugs can stack; the
    top-most tile at a coordinate is the one that "counts" for neighbouring
    and surround calculations.
  - If a player has no legal move on their turn, they must pass.

UHP MOVE NOTATION (the format used in this game):
  - First move of the game:        ``wA1``  (just the piece name)
  - Place / move adjacent to ref:  ``<piece> <position>`` where ``<position>``
    encodes the direction relative to a reference tile:
        ``ref/``   target is NE of ref
        ``ref-``   target is E  of ref
        ``ref\\``   target is SE of ref
        ``/ref``   target is SW of ref
        ``-ref``   target is W  of ref
        ``\\ref``   target is NW of ref
        ``ref``    target is ON TOP OF ref (climbing piece only)
  - Pass (only when no legal moves):  ``pass``

Examples:
    ``wA1``          White Ant 1 placed at the origin (first move only).
    ``wA1 wQ-``      White Ant 1 placed east of the White Queen.
    ``bB1 wA1``      Black Beetle 1 climbs on top of White Ant 1.
    ``wA1 -bQ``      White Ant 1 moves to west of Black Queen.

CURRENT GAME STATE (JSON from the game engine):
{state_str}

PIECE POSITIONS (axial coordinates [q, r, h]; h>0 means stacked on top):
{pieces_str}

MOVES PLAYED SO FAR:
{move_history}

You are playing as {player_color}. It is now your turn.
Pick the move you believe is strongest. Your move MUST be one of the legal UHP
move strings available in the current position.

Your response should include the reasoning that led you to your move, and
conclude with your final answer as a JSON formatted as follows:

```json
{{
  "move": "<UHP move>"
}}
```

Where ``<UHP move>`` is a string like ``wQ``, ``wA1 wQ-``, ``bB1 wA1``, or
``pass``. Failure to output your final answer in the specified format will
result in a loss. Begin!
"""


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested move "{previous_action}" but it is not a legal move in the
current position. Reconsider the rules (placement adjacency, queen-by-move-4,
the One Hive rule, freedom to move, and UHP notation) and pick a legal move.
"""


# --- Helpers ----------------------------------------------------------------


def _extract_move_from_json(response: str) -> str | None:
    """Try to extract a move string from a JSON code block or bare JSON."""
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
    """Collapse internal whitespace; preserve direction characters."""
    return re.sub(r"\s+", " ", move.strip())


def _match_move_to_legal(
    move: str,
    legal_moves: Sequence[str],
) -> str | None:
    """Match a UHP move string to one of the legal moves (case-insensitive)."""
    candidate = _normalize(move)
    if not candidate:
        return None
    candidate_lower = candidate.lower()
    candidate_compact = candidate_lower.replace(" ", "")
    for legal in legal_moves:
        legal_norm = _normalize(legal)
        if legal_norm.lower() == candidate_lower:
            return legal
        if legal_norm.lower().replace(" ", "") == candidate_compact:
            return legal
    return None


def _format_pieces(pieces: Mapping[str, Sequence[int]]) -> str:
    if not pieces:
        return "(none placed)"
    # Stable output: sort by colour then piece name.
    items = sorted(pieces.items(), key=lambda kv: (kv[0][0], kv[0][1:]))
    return "\n".join(f"  {tile}: [{q}, {r}, {h}]" for tile, (q, r, h) in items)


def _format_state(observation: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return (compact_state_json, parsed_dict)."""
    raw = observation.get("observationString", "")
    if not raw:
        return "", {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw, {}
    # Strip large arrays from the JSON dump shown in the prompt; the prompt
    # presents pieces and history in dedicated sections.
    shown = {k: v for k, v in parsed.items() if k not in {"pieces", "moves", "legal_moves", "uhp"}}
    return json.dumps(shown, indent=2), parsed


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
    actions = state.legal_actions()
    return {a: state.action_to_string(a) for a in actions}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current game state."""
    state_str, parsed = _format_state(observation)
    pieces_str = _format_pieces(parsed.get("pieces") or {})

    player_id = observation.get("playerId", 0)
    player_color = "White (w)" if player_id == 0 else "Black (b)"

    move_history_str = ", ".join(move_history) if move_history else "None"

    prompt = HIVE_PROMPT_TEMPLATE.format(
        state_str=state_str or "(empty)",
        pieces_str=pieces_str,
        move_history=move_history_str,
        player_color=player_color,
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
    """Extract a legal Hive UHP move from the model response.

    Stages:
      1. JSON code block ``{"move": "..."}``.
      2. Bare JSON ``{"move": "..."}`` anywhere in the response.
      3. Fallback scan (only when no JSON was extracted): longest legal-move
         string that appears in the response surrounded by word boundaries.

    If JSON *was* extracted but didn't match a legal move, we do NOT fall
    back to scanning the prose -- the model explicitly stated its intent and
    we should let the rethink loop handle it rather than silently guessing a
    different move from the surrounding text.
    """
    raw = _extract_move_from_json(response)
    if raw is not None:
        matched = _match_move_to_legal(raw, legal_action_strings)
        if matched is not None:
            return ParseResult(legal_action=matched, raw_action=raw)
        return ParseResult(legal_action=None, raw_action=raw)

    # No JSON found -- scan the prose for a legal UHP move. Use word-boundary
    # matching (the surrounding chars cannot be alphanumeric) so short tokens
    # like ``wQ`` don't match inside larger identifiers, and skip ``pass``
    # entirely because it's a common English word that would otherwise match
    # phrases like "I'll pass on the spider".
    for legal in sorted(legal_action_strings, key=len, reverse=True):
        if legal == "pass":
            continue
        pattern = r"(?<![A-Za-z0-9])" + re.escape(legal) + r"(?![A-Za-z0-9])"
        if re.search(pattern, response, re.IGNORECASE):
            return ParseResult(legal_action=legal, raw_action=legal)

    return ParseResult(legal_action=None, raw_action=None)


# --- Adapter and agent ------------------------------------------------------


class _HiveHarness:
    """Adapts module-level functions to the GameHarness protocol."""

    def get_legal_moves(self, observation):
        return get_legal_moves(observation)

    def make_prompt(
        self,
        observation,
        move_history,
        previous_response=None,
        previous_action=None,
    ):
        return generate_prompt(
            observation,
            move_history,
            previous_response,
            previous_action,
        )

    def parse_response(self, response, legal_action_strings):
        return parse_response(response, legal_action_strings)


agent_fn = create_agent_fn(_HiveHarness())
