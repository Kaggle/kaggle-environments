"""LLM harness for OpenSpiel Crazyhouse.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.

Crazyhouse is a chess variant: captured pieces switch colour and enter the
captor's *pocket*, from which they can later be *dropped* onto any empty
square (pawns may not drop on the 1st or 8th rank). OpenSpiel exposes moves
in Standard Algebraic Notation, e.g. ``e4``, ``Nf3``, ``Bxd5``, ``O-O``, and
drops as ``P@e4`` / ``N@d5`` (uppercase piece, ``@``, target square).
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult

try:
    from kaggle_environments.envs.open_spiel_env.games.crazyhouse import (  # noqa: F401
        crazyhouse_proxy,
    )
except Exception:
    pass


_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{[^{}]*\"move\"\s*:\s*\"([^\"]+)\"[^{}]*\}", re.DOTALL)

# A SAN-or-drop token. Conservative: piece letter (PNBRQK) or pawn file (a-h),
# optional disambiguation, optional capture x, target square, optional
# promotion (=Q etc.), optional check/mate marker. Castling and drops are
# handled separately.
_SAN_TOKEN_RE = re.compile(
    r"\b("
    r"O-O-O[+#]?|O-O[+#]?"                               # castling
    r"|[PNBRQK]@[a-h][1-8]"                              # drop
    r"|[NBRQK][a-h]?[1-8]?x?[a-h][1-8][+#]?"             # piece move
    r"|[a-h]x?[a-h][1-8](?:=[NBRQ])?[+#]?"               # pawn capture / push (file form)
    r"|[a-h][1-8](?:=[NBRQ])?[+#]?"                      # pawn push (target only)
    r")"
)


# --- Prompt -----------------------------------------------------------------


CRAZYHOUSE_PROMPT_TEMPLATE = """Let's play Crazyhouse.

Rules: Crazyhouse is chess with one twist — when you capture an opponent's
piece, it switches colour and enters your *pocket*. Instead of a normal move
you may *drop* any piece from your pocket onto any empty square, with these
restrictions:
- Pawns cannot be dropped on the 1st or 8th rank.
- Promoted pieces revert to pawns when captured.
- A drop that gives checkmate IS legal (no rule against drop-mate here).
All other rules are standard chess: castling, en passant, promotion, and
the king cannot remain in check after your move.

Move notation (Standard Algebraic Notation):
- Pawn push: ``e4``, ``d5``. Pawn capture: ``exd5``. Promotion: ``e8=Q``.
- Piece move: ``Nf3``, ``Bxc4``, ``Qd1``. Disambiguation when needed:
  ``Nbd2`` (file) or ``R1e2`` (rank).
- Castling: ``O-O`` (kingside), ``O-O-O`` (queenside).
- Drop: ``P@e4``, ``N@d5``, ``B@h6``, ``R@e1``, ``Q@d4`` (uppercase piece,
  ``@``, target square). Drops use uppercase regardless of which side
  you play.
Suffixes ``+`` (check) and ``#`` (mate) may appear in legal moves; copy
them exactly when present.

Current game state (JSON; ``pockets`` lists piece counts available to drop,
both sides shown with uppercase letters):
{state_str}

The moves played so far are:
{move_history}

Your pocket: {my_pocket}
Opponent pocket: {opp_pocket}

You are playing as {player_name} ({player_code}). It is now your turn.
Pick your strongest move from the legal list. The move MUST appear
verbatim (including any ``+``/``#`` suffix and the exact letter case) in
the legal moves list — there are too many legal moves to enumerate them
all here, so consult the position above.

Respond with your reasoning followed by your final move in a JSON block:

```json
{{
  "move": "<move>"
}}
```

Failure to output your final answer in the specified format, or selecting
a move that is not legal, will result in a loss.
Begin!
"""


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested move "{previous_action}" but this is not in the legal moves
list. Reconsider the position and play a legal move. Remember that drops
use uppercase piece letters with ``@`` (e.g. ``P@e4``) and that check/mate
suffixes (``+``/``#``) must match the legal move string exactly.
"""


# --- Helpers ----------------------------------------------------------------


def _parse_state_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the structured crazyhouse state dict out of the observation."""
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
        try:
            _, state = pyspiel.deserialize_game_and_state(serialized)
            return json.loads(state.observation_string(0))
        except (json.JSONDecodeError, RuntimeError):
            pass
    return {}


def _format_pocket(pocket: Mapping[str, int] | None) -> str:
    """Render a pocket like ``{"P": 2, "N": 1}`` as ``2xP, 1xN`` (or empty)."""
    if not pocket:
        return "(empty)"
    return ", ".join(f"{count}x{piece}" for piece, count in sorted(pocket.items()))


def _normalize_san(token: str) -> str:
    """Normalize an SAN token for tolerant matching.

    Strips trailing ``+``/``#`` markers and lower-cases pawn-only moves so
    ``E4`` and ``e4`` are treated the same. Piece moves keep their leading
    uppercase letter so ``Nf3`` doesn't collide with a hypothetical ``nf3``.
    """
    t = token.strip().rstrip("+#")
    if not t:
        return t
    if "@" in t:
        # Drop: P@e4 — uppercase piece letter, lowercase square.
        head, _, tail = t.partition("@")
        return head.upper() + "@" + tail.lower()
    if t.startswith(("O-O", "0-0")):
        return t.replace("0", "O")
    first = t[0]
    if first in "PNBRQK":
        return first + t[1:].lower()
    return t.lower()


def _build_legal_index(legal_moves: Sequence[str]) -> dict[str, str]:
    """Map normalised SAN -> the exact legal string the engine expects."""
    index: dict[str, str] = {}
    for legal in legal_moves:
        index.setdefault(_normalize_san(legal), legal)
    return index


def _match_move_to_legal(
    move: str,
    legal_moves: Sequence[str],
) -> str | None:
    """Match a candidate move string to one of the legal action strings."""
    if not move:
        return None
    candidate = _normalize_san(move)
    index = _build_legal_index(legal_moves)
    if candidate in index:
        return index[candidate]
    # Try with the check/mate suffix the model may have omitted: a unique
    # legal move whose stripped form matches counts as a hit.
    stripped_hits = [legal for legal in legal_moves if _normalize_san(legal) == candidate]
    if len(stripped_hits) == 1:
        return stripped_hits[0]
    return None


def _extract_move_from_json(response: str) -> str | None:
    """Pull the move string from a ```json``` block or a bare JSON object."""
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
    player_id = state.current_player()
    if player_id < 0:
        return {}
    actions = state.legal_actions(player_id)
    return {a: state.action_to_string(player_id, a) for a in actions}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current crazyhouse state."""
    state = _parse_state_payload(observation)
    # Crazyhouse uses player 1 = White, 0 = Black (see crazyhouse_proxy.py).
    player_id = observation.get("playerId", 1)
    player_name = "White" if player_id == 1 else "Black"
    player_code = "W" if player_id == 1 else "B"

    pockets = state.get("pockets") or {}
    my_pocket = pockets.get(player_name.lower(), {})
    opp_pocket = pockets.get("black" if player_id == 1 else "white", {})

    move_history_str = " ".join(move_history) if move_history else "None"

    prompt = CRAZYHOUSE_PROMPT_TEMPLATE.format(
        state_str=observation.get("observationString", "") or json.dumps(state),
        move_history=move_history_str,
        my_pocket=_format_pocket(my_pocket),
        opp_pocket=_format_pocket(opp_pocket),
        player_name=player_name,
        player_code=player_code,
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
    """Extract a legal crazyhouse move from the LLM response.

    Tries a ```json``` block first, then a bare ``{"move": "..."}``, then
    scans the response text for any token that matches a legal SAN move.
    """
    raw = _extract_move_from_json(response)
    if raw is not None:
        matched = _match_move_to_legal(raw, legal_action_strings)
        if matched is not None:
            return ParseResult(legal_action=matched, raw_action=raw)

    # Fallback: scan response for any legal SAN/drop token. Iterate in the
    # order the model wrote them so the first plausible move wins.
    legal_index = _build_legal_index(legal_action_strings)
    for match in _SAN_TOKEN_RE.finditer(response):
        token = match.group(1)
        canonical = _normalize_san(token)
        if canonical in legal_index:
            return ParseResult(
                legal_action=legal_index[canonical],
                raw_action=raw or token,
            )

    return ParseResult(legal_action=None, raw_action=raw)
