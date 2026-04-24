"""LLM harness for OpenSpiel Gin Rummy.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.

Gin Rummy turns walk through several phases (decide on the first upcard, draw,
discard, optionally knock, lay off onto the knocker's melds). Each sub-action
is one LLM call. The proxy in ``gin_rummy_proxy.py`` exposes the current phase
plus the player's hand, the upcard, the discard pile, the stock size, and the
knock card in ``observationString`` as JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult

# Importing the proxy registers the ``gin_rummy_proxy`` pyspiel game so that
# ``deserialize_game_and_state`` can rebuild it from the obs. Wrapped in
# try/except because the harness can be dropped into a notebook where the
# proxy module isn't on the path; in that case we rely on ``legalActions``
# being included in the observation directly.
try:
    from kaggle_environments.envs.open_spiel_env.games.gin_rummy import (  # noqa: F401
        gin_rummy_proxy,
    )
except Exception:
    pass


_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{[^{}]*\"move\"\s*:\s*\"([^\"]+)\"[^{}]*\}", re.DOTALL)
# Strip OpenSpiel's "Player: 0 Action: Draw upcard" wrapper.
_ACTION_PREFIX_RE = re.compile(r"^Player:\s*\d+\s+Action:\s*", re.IGNORECASE)
# A canonical card token, e.g. "As", "Td", "9h".
_CARD_RE = re.compile(r"\b([A23456789TJQK])([scdh])\b")
# Sort cards by suit then by rank so hands display in a stable, readable order.
_RANK_ORDER = {r: i for i, r in enumerate("A23456789TJQK")}
_SUIT_ORDER = {s: i for i, s in enumerate("scdh")}
_SUIT_NAME = {"s": "Spades", "c": "Clubs", "d": "Diamonds", "h": "Hearts"}


# --- Helpers ----------------------------------------------------------------


def _parse_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Parse the JSON observation_string emitted by the gin_rummy proxy."""
    raw = observation.get("observationString", "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}


def _strip_prefix(action_str: str) -> str:
    """Drop OpenSpiel's 'Player: <id> Action: ' wrapper from an action string."""
    return _ACTION_PREFIX_RE.sub("", action_str).strip()


def _sort_cards(cards: Sequence[str]) -> list[str]:
    """Sort cards by suit then rank for stable display."""
    return sorted(
        (c for c in cards if _CARD_RE.fullmatch(c)),
        key=lambda c: (_SUIT_ORDER.get(c[1], 99), _RANK_ORDER.get(c[0], 99)),
    )


def _format_hand(hand: Sequence[str]) -> str:
    """Group cards by suit, e.g. 'Spades: As 5s 9s | Hearts: 2h Th'."""
    if not hand:
        return "(empty)"
    groups: dict[str, list[str]] = {"s": [], "c": [], "d": [], "h": []}
    for card in _sort_cards(hand):
        groups[card[1]].append(card)
    parts: list[str] = []
    for suit in "scdh":
        if groups[suit]:
            parts.append(f"{_SUIT_NAME[suit]}: " + " ".join(groups[suit]))
    return " | ".join(parts) if parts else "(empty)"


def _format_discard_pile(pile: Sequence[str]) -> str:
    """Show the discard pile bottom -> top, marking the top (= upcard)."""
    if not pile:
        return "(empty)"
    cards = list(pile)
    top = cards[-1]
    if len(cards) == 1:
        return f"{top}  (top)"
    return " ".join(cards[:-1]) + f"  [top: {top}]"


def _format_history(move_history: list[str], limit: int = 16) -> str:
    if not move_history:
        return "(no moves yet)"
    if len(move_history) <= limit:
        return ", ".join(move_history)
    return "... " + ", ".join(move_history[-limit:])


def _player_glyph(player_id: int) -> str:
    return f"Player {player_id}"


# --- Phase-specific instructions --------------------------------------------


_PHASE_INSTRUCTION = {
    "FirstUpcard": (
        "This is the first decision of the hand. You may either take the "
        "upcard (it goes into your hand and you must then discard) or pass. "
        "If both players pass on the first upcard, the dealer's opponent then "
        "draws normally."
    ),
    "Deal": (
        "Decide whether to take the current upcard. Taking it adds the card "
        "to your hand and you must then discard; passing lets your opponent "
        "decide on it."
    ),
    "Draw": (
        "Choose where to draw your card from: 'Draw upcard' takes the visible "
        "top of the discard pile, 'Draw stock' draws an unknown card from the "
        "stock. After drawing you will discard."
    ),
    "Discard": (
        "You just drew a card. Choose one card from your hand to discard. "
        "If your remaining deadwood would be at most the knock card, you may "
        "instead knock to end the hand (use 'Knock' on the action list)."
    ),
    "Knock": (
        "You have declared knock. Now declare each meld in your hand "
        "(runs of 3+ in a suit, or sets of 3-4 of the same rank). Choose one "
        "of the listed meld groups; you'll be asked again for any remaining "
        "melds, and unmelded cards become your deadwood."
    ),
    "Layoff": (
        "Your opponent knocked. You may lay off any of your unmatched cards "
        "onto their declared melds to reduce your deadwood. Choose a card to "
        "lay off, or pass if no card extends one of their melds."
    ),
    "GameOver": "The hand is over.",
}


def _instruction_for_phase(phase: str | None, legal_strings: Sequence[str]) -> str:
    if phase and phase in _PHASE_INSTRUCTION:
        return _PHASE_INSTRUCTION[phase]
    # Fallback: infer from the legal moves on offer.
    legals = set(legal_strings)
    if "Draw upcard" in legals or "Draw stock" in legals:
        return _PHASE_INSTRUCTION["Draw"]
    if "Knock" in legals and any(_CARD_RE.fullmatch(s) for s in legals):
        return _PHASE_INSTRUCTION["Discard"]
    if "Pass" in legals and "Draw upcard" not in legals:
        return _PHASE_INSTRUCTION["Layoff"]
    return (
        "Choose one of the legal actions listed below. Your move MUST be "
        "exactly one of those strings."
    )


# --- Prompt -----------------------------------------------------------------


GIN_RUMMY_PROMPT_TEMPLATE = """Let's play Gin Rummy (OpenSpiel rules, knock card = {knock_card}).

Card notation: rank in {{A,2-9,T,J,Q,K}} followed by suit in {{s,c,d,h}}.
Goal: arrange your 10-card hand into melds (runs of 3+ in a suit, sets of 3-4
of the same rank). Cards not in any meld are 'deadwood' (face value, faces=10,
Ace=1). When your deadwood is at or below the knock card you may knock to end
the hand. Going gin (zero deadwood) and undercutting the knocker pay bonuses.

You are {player_glyph}.
Phase: {phase}
Knock card: {knock_card}    Stock remaining: {stock_size}
Upcard (top of discard, available to draw): {upcard}
Discard pile (oldest -> newest): {discard_pile}

Your hand ({hand_count} cards): {hand}
Your current deadwood: {deadwood}

Move history (last 16 actions, oldest -> newest):
{move_history}

{phase_instruction}

Legal moves you may play:
{legal_moves}

It is your turn. Think briefly about the position, then choose your move.
The move MUST be exactly one of the legal moves listed above (use the exact
string, e.g. 'Draw upcard', 'Knock', '7h', 'As2s3s').

Respond with your reasoning followed by your final answer in a JSON block:

```json
{{
  "move": "<one of the legal moves above>"
}}
```

Failure to output your final answer in the specified format, or selecting a
move that is not in the legal list, will result in a loss.
"""


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested move "{previous_action}" but it is NOT in the legal moves list.
Reconsider and pick a move that appears verbatim in the legal moves above.
"""


# --- Public functions (called by main.py) -----------------------------------


def get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str]:
    """Return ``{action_id: action_string}`` for the current sub-action.

    Strings are the cleaned OpenSpiel action names (no 'Player: X Action:'
    prefix), e.g. ``"Draw upcard"``, ``"Knock"``, ``"7h"``, ``"As2s3s"``.
    """
    legal_actions = observation.get("legalActions")
    legal_action_strings = observation.get("legalActionStrings")
    if legal_actions and legal_action_strings:
        return {a: _strip_prefix(s) for a, s in zip(legal_actions, legal_action_strings)}

    serialized = observation.get("serializedGameAndState", "")
    if not serialized:
        return {}
    try:
        _, state = pyspiel.deserialize_game_and_state(serialized)
    except Exception:
        return {}
    actions = state.legal_actions()
    return {a: _strip_prefix(state.action_to_string(a)) for a in actions}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current sub-action."""
    obs = _parse_observation(observation)

    player_id = observation.get("playerId", obs.get("current_player", 0)) or 0
    hand = obs.get("hands", {}).get(str(player_id), []) or []
    deadwood = (obs.get("deadwood") or {}).get(str(player_id))
    phase = obs.get("phase") or "Unknown"
    knock_card = obs.get("knock_card", "?")
    stock_size = obs.get("stock_size", "?")
    upcard = obs.get("upcard") or "(none)"
    discard_pile = obs.get("discard_pile") or []

    legal_map = get_legal_moves(observation)
    legal_strings = list(legal_map.values())

    prompt = GIN_RUMMY_PROMPT_TEMPLATE.format(
        player_glyph=_player_glyph(player_id),
        phase=phase,
        knock_card=knock_card,
        stock_size=stock_size,
        upcard=upcard,
        discard_pile=_format_discard_pile(discard_pile),
        hand_count=len(hand),
        hand=_format_hand(hand),
        deadwood=deadwood if deadwood is not None else "?",
        move_history=_format_history(move_history),
        phase_instruction=_instruction_for_phase(phase, legal_strings),
        legal_moves=", ".join(legal_strings) if legal_strings else "(none)",
    )

    if previous_response is not None:
        prompt += RETHINK_SUFFIX.format(
            previous_response=previous_response[:500],
            previous_action=previous_action or "(could not parse)",
        )

    return prompt


def _extract_move_from_json(response: str) -> str | None:
    """Pull the move string out of a ```json``` block, or a bare JSON object."""
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
    """Lowercase and drop whitespace/punctuation for forgiving comparisons."""
    return re.sub(r"[\s,.\-_'\"`()\[\]]", "", move).lower()


def _match_move_to_legal(move: str, legal_moves: Sequence[str]) -> str | None:
    if not move:
        return None
    if move in legal_moves:
        return move
    # Tolerate the model echoing the OpenSpiel "Player: X Action: ..." wrapper.
    stripped = _strip_prefix(move)
    if stripped in legal_moves:
        return stripped
    target = _normalize(stripped)
    for legal in legal_moves:
        if _normalize(legal) == target:
            return legal
    return None


def parse_response(
    response: str, legal_action_strings: Sequence[str],
) -> ParseResult:
    """Extract a legal action string from the model response.

    Tries the JSON block first, then a bare ``{"move": "..."}`` object, then a
    fallback whole-token scan for any legal action string mentioned in the
    response. Match is case- and punctuation-insensitive (so 'Draw Upcard',
    'draw-upcard', '"As2s3s"' all map back to the canonical legal string).
    """
    raw = _extract_move_from_json(response)
    if raw is not None:
        matched = _match_move_to_legal(raw, legal_action_strings)
        if matched is not None:
            return ParseResult(legal_action=matched, raw_action=raw)

    # Fallback: scan the response text for any legal action token. Try the
    # longest legal strings first so 'AsAcAdAh' matches before 'AsAcAd'.
    sorted_legals = sorted(legal_action_strings, key=len, reverse=True)
    for legal in sorted_legals:
        pattern = r"(?<![A-Za-z0-9])" + re.escape(legal) + r"(?![A-Za-z0-9])"
        if re.search(pattern, response):
            return ParseResult(legal_action=legal, raw_action=raw or legal)

    return ParseResult(legal_action=None, raw_action=raw)
