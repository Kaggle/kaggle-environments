"""LLM harness for OpenSpiel Gin Rummy.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.

Gin Rummy turns walk through several phases (decide on the first upcard, draw,
discard, optionally knock, lay off onto the knocker's melds). Each sub-action
is one LLM call. The proxy in ``gin_rummy_proxy.py`` exposes the current phase
plus the player's hand, the upcard, the discard pile, the stock size, the
knock card, and (once a knock has happened) the knocker's laid melds, full
hand and deadwood -- all in ``observationString`` as JSON.

The env loads ``gin_rummy`` with default params (``oklahoma=false``,
``knock_card=10``, ``gin_bonus=25``, ``undercut_bonus=25``).
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult, extract_last_json_object

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


def _format_history(move_history: list[str]) -> str:
    if not move_history:
        return "(no moves yet)"
    return ", ".join(move_history)


def _player_glyph(player_id: int) -> str:
    return f"Player {player_id}"


def _format_meld(meld: Sequence[str]) -> str:
    """Render a single meld as space-separated card tokens."""
    return " ".join(meld)


def _format_layed_melds(melds: Sequence[Sequence[str]]) -> str:
    if not melds:
        return "(none yet)"
    return "; ".join(_format_meld(m) for m in melds)


def _format_opponent_block(obs: dict[str, Any], opp_id: int) -> str:
    """Render whatever's visible about the opponent for this prompt.

    Before a knock the opponent's hand and deadwood are hidden. After the
    opponent knocks, the engine reveals their laid melds, remaining hand
    (= deadwood cards), and deadwood total -- the harness must show all of it
    so the player can play layoffs intelligently.
    """
    opp_key = str(opp_id)
    layed = (obs.get("layed_melds") or {}).get(opp_key) or []
    layoffs = obs.get("layoffs") or []
    hand = (obs.get("hands") or {}).get(opp_key) or []
    hidden = (obs.get("hand_hidden") or {}).get(opp_key, True)
    deadwood = (obs.get("deadwood") or {}).get(opp_key)
    knocked = (obs.get("knocked") or [False, False])[opp_id]

    lines: list[str] = []
    if knocked:
        lines.append("Opponent has KNOCKED.")
    lines.append(f"Opponent laid melds: {_format_layed_melds(layed)}")
    if layoffs:
        lines.append(f"Cards layed off onto opponent's melds: {' '.join(layoffs)}")
    if hidden or not hand:
        if knocked:
            lines.append("Opponent remaining hand (deadwood cards): (none)")
        else:
            lines.append("Opponent hand: (hidden)")
    else:
        lines.append(f"Opponent remaining hand ({len(hand)} cards): {_format_hand(hand)}")
    if deadwood is not None:
        lines.append(f"Opponent deadwood: {deadwood}")
    return "\n".join(lines)


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
        "You have declared knock. First (if your hand is 11 cards) you must "
        "discard one card from the action list; that's the card you choose to "
        "keep out of your melds. Then you'll be asked, one at a time, to "
        "declare each meld in your hand (runs of 3+ in a suit, or sets of "
        "3-4 of the same rank). Choose 'Pass' once your remaining unmelded "
        "cards (your deadwood) total at most the knock card and you're done."
    ),
    "Layoff": (
        "Your opponent knocked. The opponent's laid melds are listed above. "
        "You may extend any of their melds with your own unmatched cards to "
        "reduce your deadwood: a layoff card must continue a run in the same "
        "suit or complete a set of the same rank. Choose a card to lay off, "
        "or 'Pass' if no card extends one of their melds. After laying off "
        "individual cards, you'll then declare your own melds (one at a "
        "time, then 'Pass') to subtract them from your deadwood. Note: if "
        "the opponent went gin (deadwood 0) you cannot lay off."
    ),
    "Wall": (
        "Only 2 cards remain in the stock (this is 'the wall'). You may "
        "'Pass' (the hand ends with no winner -- both players score 0), or "
        "if your current best meld arrangement plus the upcard would leave "
        "you at or below the knock card you may declare 'Knock' (the upcard "
        "is added to your hand automatically)."
    ),
    "GameOver": "The hand is over.",
}


def _instruction_for_phase(phase: str | None, legal_strings: Sequence[str]) -> str:
    if phase and phase in _PHASE_INSTRUCTION:
        return _PHASE_INSTRUCTION[phase]
    # Fallback: infer from the legal moves on offer. Note: Wall and Layoff
    # both expose only ['Pass', 'Knock']/cards; we can't tell them apart from
    # the legals alone, so we lean on the phase string above whenever it's
    # available and only hit this fallback for genuinely unknown phases.
    legals = set(legal_strings)
    if "Draw upcard" in legals or "Draw stock" in legals:
        return _PHASE_INSTRUCTION["Draw"]
    if "Knock" in legals and any(_CARD_RE.fullmatch(s) for s in legals):
        return _PHASE_INSTRUCTION["Discard"]
    return (
        "Choose a legal action for this phase. Use an exact OpenSpiel "
        "action string."
    )


# --- Prompt -----------------------------------------------------------------


GIN_RUMMY_PROMPT_TEMPLATE = """Let's play Gin Rummy (OpenSpiel default rules; knock card = {knock_card}).

Card notation: rank in {{A,2-9,T,J,Q,K}} followed by suit in {{s,c,d,h}}.
Goal: arrange your 10-card hand into melds (runs of 3+ in a suit, e.g.
'5s6s7s'; sets of 3-4 of the same rank, e.g. '7s7c7d'). Cards not in any
meld are 'deadwood' worth their face value (Ace=1, 2-9=pip, T/J/Q/K=10).

When your deadwood is at or below the knock card ({knock_card}) you may knock
to end the hand. The knocker scores (opponent deadwood - knocker deadwood).
Going gin (deadwood == 0) adds a +{gin_bonus} bonus and the opponent cannot
lay off. If the non-knocker ends with deadwood <= knocker's after laying off
they 'undercut' and instead receive (knocker deadwood - non-knocker deadwood)
+ {undercut_bonus} bonus.

Other ways the hand can end:
  - The stock runs down to 2 cards (the 'wall'): the next player may pass
    (hand ends with both scoring 0) or knock if legal.
  - If both players draw the same upcard and immediately discard it back in
    succession, the hand ends with both scoring 0.

You are {player_glyph}.
Phase: {phase}
Knock card: {knock_card}    Stock remaining: {stock_size}
Upcard (top of discard, available to draw): {upcard}
Discard pile (oldest -> newest): {discard_pile}

Your hand ({hand_count} cards): {hand}
Your current deadwood: {deadwood}
Your laid melds: {your_layed_melds}

{opponent_block}

Your previous actions (oldest -> newest):
{move_history}

{phase_instruction}

It is your turn. Think briefly about the position, then choose your move.
Use the exact OpenSpiel action string (e.g. 'Draw upcard', 'Draw stock',
'Knock', 'Pass', a single card like '7h', or a meld as concatenated cards
like 'As2s3s').

Respond with your reasoning followed by your final answer in a JSON block:

```json
{{
  "move": "<exact OpenSpiel action string>"
}}
```

Failure to output your final answer in the specified format, or selecting an
illegal move, will result in a loss.
"""


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested move "{previous_action}" but it is not a legal move.
Reconsider the position and pick a legal action for this phase.
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


_DEFAULT_GIN_BONUS = 25
_DEFAULT_UNCUT_BONUS = 25


def _game_param(observation: Mapping[str, Any], name: str, default: int) -> int:
    """Look up a numeric engine param via the serialized game string.

    Without the serialized state we can't recover non-default params, so we
    fall back to the OpenSpiel default. This is good enough for the prompt's
    rule wording -- the env loads with defaults today.
    """
    serialized = observation.get("serializedGameAndState", "")
    if not serialized:
        return default
    try:
        game, _ = pyspiel.deserialize_game_and_state(serialized)
        return int(game.get_parameters().get(name, default))
    except Exception:
        return default


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current sub-action."""
    obs = _parse_observation(observation)

    player_id = observation.get("playerId", obs.get("current_player", 0)) or 0
    opp_id = 1 - player_id
    hand = obs.get("hands", {}).get(str(player_id), []) or []
    deadwood = (obs.get("deadwood") or {}).get(str(player_id))
    phase = obs.get("phase") or "Unknown"
    knock_card = obs.get("knock_card", "?")
    stock_size = obs.get("stock_size", "?")
    upcard = obs.get("upcard") or "(none)"
    discard_pile = obs.get("discard_pile") or []
    my_layed = (obs.get("layed_melds") or {}).get(str(player_id)) or []

    gin_bonus = _game_param(observation, "gin_bonus", _DEFAULT_GIN_BONUS)
    undercut_bonus = _game_param(observation, "undercut_bonus", _DEFAULT_UNCUT_BONUS)

    legal_map = get_legal_moves(observation)
    legal_strings = list(legal_map.values())

    prompt = GIN_RUMMY_PROMPT_TEMPLATE.format(
        player_glyph=_player_glyph(player_id),
        phase=phase,
        knock_card=knock_card,
        gin_bonus=gin_bonus,
        undercut_bonus=undercut_bonus,
        stock_size=stock_size,
        upcard=upcard,
        discard_pile=_format_discard_pile(discard_pile),
        hand_count=len(hand),
        hand=_format_hand(hand),
        deadwood=deadwood if deadwood is not None else "?",
        your_layed_melds=_format_layed_melds(my_layed),
        opponent_block=_format_opponent_block(obs, opp_id),
        move_history=_format_history(move_history),
        phase_instruction=_instruction_for_phase(phase, legal_strings),
    )

    if previous_response is not None:
        prompt += RETHINK_SUFFIX.format(
            previous_response=previous_response[:500],
            previous_action=previous_action or "(could not parse)",
        )

    return prompt


def _extract_move_from_json(response: str) -> str | None:
    """Pull the move string out of the LAST JSON object in the response."""
    data = extract_last_json_object(response, required_keys=("move",))
    if data is None:
        return None
    move = str(data.get("move") or "").strip()
    return move or None


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

    # Fallback: scan the response for any legal-action token. Pick the legal
    # whose rightmost occurrence is latest (models enumerate rejected options
    # before stating their final move). Tie-break on length so longer tokens
    # like 'AsAcAdAh' beat shorter prefixes like 'AsAcAd' at the same position.
    best_end = -1
    best_legal: str | None = None
    for legal in legal_action_strings:
        pattern = r"(?<![A-Za-z0-9])" + re.escape(legal) + r"(?![A-Za-z0-9])"
        matches = list(re.finditer(pattern, response))
        if not matches:
            continue
        end = matches[-1].end()
        if end > best_end or (end == best_end and len(legal) > len(best_legal or "")):
            best_end = end
            best_legal = legal
    if best_legal is not None:
        return ParseResult(legal_action=best_legal, raw_action=raw or best_legal)

    return ParseResult(legal_action=None, raw_action=raw)
