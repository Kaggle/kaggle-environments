"""LLM harness for Bridge Arena (2v2 team variant of OpenSpiel Bridge).

Two teams of two LLMs play contract bridge at one table. Team A sits
N/S (external player ids 0 and 1); team B sits E/W (external player ids
2 and 3). Partners may only "communicate" through their public bids and
plays -- there is no side channel between teammates.

The arena observation is a per-player JSON view that includes the
viewer's seat / partner / opponents, the dealer and vulnerability, the
full auction so far (each call attributed to a player id and table
position), the cards played, and the raw OpenSpiel bridge observation
text (which shows the viewer's hand during the auction and all four
hands once play begins -- standard bridge convention).
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(
    r"\{[^{}]*\"(?:move|bid|call|action)\"\s*:\s*\"([^\"]+)\"[^{}]*\}",
    re.DOTALL,
)

# Suit letter -> unicode glyph used by OpenSpiel's bridge action strings.
_SUIT_LETTER_TO_GLYPH = {"C": "♣", "D": "♦", "H": "♥", "S": "♠"}
_SUIT_GLYPH_TO_LETTER = {v: k for k, v in _SUIT_LETTER_TO_GLYPH.items()}


# --- Prompt -----------------------------------------------------------------


ARENA_PROMPT_TEMPLATE = """\
Let's play Bridge Arena (2v2 contract bridge).

Setup: 4 players at one table; partners sit opposite. The table reads
clockwise N -> E -> S -> W. Team A holds N and S; team B holds E and W.

Important: your partner is another instance of YOU (same model, same
submission). The opposing team is two instances of a single different
agent. There is NO side channel -- you and your partner can only
"communicate" through the public bidding and card play that every seat
at the table sees.

Phases:
  1. AUCTION. Starting with the dealer and rotating clockwise, each
     player makes one CALL on their turn: "Pass", a bid like "1NT" or
     "4♠" (level 1-7, denomination ♣/♦/♥/♠/NT), "Dbl" of the opponents'
     last bid, or "RDbl" after a double. The auction ends after three
     consecutive passes following any bid (or four passes from the start,
     which scores 0 for both sides). The highest bid becomes the
     CONTRACT; whichever side first named the contract's denomination is
     the declaring side, and on that side the player who first named the
     denomination is the DECLARER. The contract sets the trump suit (or
     no-trump) and the number of tricks the declarer must take beyond 6.
  2. PLAY. The opening lead comes from the player to declarer's left.
     Dummy (declarer's partner) lays their hand face up after the
     opening lead; declarer plays both their own hand and dummy's. Each
     trick: four cards, one per player clockwise. You must follow suit
     if you have one; otherwise you may discard or trump. The highest
     trump in a trick wins, else the highest card of the led suit. The
     trick winner leads the next.

Scoring is standard contract bridge (duplicate, no honors). Both
players on a team always get the same final score. The team with the
higher score wins. A passed-out auction scores 0 for both teams.

Action format. The legal actions surfaced to you come from OpenSpiel
bridge and use these strings:
  * Calls: "Pass", "Dbl", "RDbl", and "<level><denom>" with denom in
    {{♣, ♦, ♥, ♠, NT}} -- e.g., "1♣", "3NT", "4♠", "7♥".
  * Cards: "<suit><rank>" with suit in {{♣, ♦, ♥, ♠}} and rank in
    {{2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K, A}} -- e.g., "♠A", "♥T".
You may write the suit as the unicode glyph or the letter C/D/H/S; the
parser accepts either.

Identity:
  * You are player {player_id}, seated at {your_table_position}.
  * Your partner is player {partner_id}, seated at {partner_table_position}.
  * Your opponents are players {opp0_id} ({opp0_pos}) and {opp1_id} ({opp1_pos}).
  * You are on team {team_id}; {team_label}.
  * Dealer: player {dealer_id} ({dealer_pos}).
  * Phase: {phase}.

Bridge state (raw OpenSpiel observation for your seat):
{raw_observation}

Auction so far (one entry per call, in order):
{auction_str}

Cards played so far:
{plays_str}

It is now your turn. Choose your action.
Your response should include the reasoning that led to your choice,
and conclude with your final answer as JSON formatted as follows:

```json
{{
  "bid": "<action>"
}}
```

Use the field name "bid" for both calls and card plays. Failure to
output your final answer in the specified format will be treated as an
illegal move.
Begin!
"""


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested action "{previous_action}" but this is not a legal
action in the current state. Reconsider the rules and the current
state, then pick a legal call (during the auction) or card (during
play). Output your final answer as `{{"bid": "<action>"}}` inside a
```json``` block.
"""


# --- Helpers ----------------------------------------------------------------


def _parse_obs(observation: Mapping[str, Any]) -> dict[str, Any]:
    obs_str = observation.get("observationString", "")
    if not obs_str:
        return {}
    try:
        return json.loads(obs_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def _normalize_action_string(s: str) -> str:
    """Canonical form for matching bridge action strings.

    Operates entirely in lower case, with whitespace removed and suit
    glyphs collapsed to single letters c/d/h/s. Card rank "10" collapses
    to "t" so OpenSpiel's ``"♥10"`` and a model's ``"H10"`` or ``"HT"``
    all match. Synonyms: ``double``/``x`` -> ``dbl``; ``redouble``/``xx``
    -> ``rdbl``; ``no trump``/``notrump``/``<n>n`` -> ``<n>nt``. A
    leading ``bid ``/``call ``/``play ``/``card `` token is dropped.
    """
    if not s:
        return ""
    # Glyph -> letter first (case-insensitive), then lowercase + collapse
    # whitespace so every later step operates on a single canonical form.
    out = s
    for glyph, letter in _SUIT_GLYPH_TO_LETTER.items():
        out = out.replace(glyph, letter)
    out = re.sub(r"\s+", " ", out).strip().lower()
    if not out:
        return ""
    for prefix in ("bid ", "call ", "play ", "card "):
        if out.startswith(prefix):
            out = out[len(prefix):].lstrip()
            break
    out = out.replace(" ", "")
    if out in ("double", "x"):
        return "dbl"
    if out in ("redouble", "xx"):
        return "rdbl"
    if out == "pass":
        return "pass"
    out = out.replace("notrump", "nt").replace("no-trump", "nt")
    # Card rank: "10" collapses to "t" so "h10", "♥10", and "HT" all
    # normalize to "ht". Safe because "10" never appears in a bid form
    # (bid levels are 1-7).
    out = out.replace("10", "t")
    if re.fullmatch(r"[1-7]n", out):
        out = out[:-1] + "nt"
    return out


def _match_action_to_legal(raw: str, legal_action_strings: Sequence[str]) -> str | None:
    target = _normalize_action_string(raw)
    if not target:
        return None
    for legal in legal_action_strings:
        if _normalize_action_string(legal) == target:
            return legal
    return None


def _extract_move_from_json(response: str) -> str | None:
    """Pull a 'bid'/'move'/'call'/'action' value out of the model response."""
    match = _JSON_BLOCK_RE.search(response)
    if match:
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            for key in ("bid", "move", "call", "action"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    bare = _BARE_JSON_RE.search(response)
    if bare:
        return bare.group(1).strip()
    return None


def _format_auction(auction: Sequence[Mapping[str, Any]]) -> str:
    if not auction:
        return "  (no calls yet)"
    return "\n".join(
        f"  {idx + 1}. player {entry.get('player_id')}"
        f" ({entry.get('table_position')}, team {entry.get('team_id')}):"
        f" {entry.get('call')}"
        for idx, entry in enumerate(auction)
    )


def _format_plays(plays: Sequence[Mapping[str, Any]]) -> str:
    if not plays:
        return "  (no cards played yet)"
    # Group into tricks of 4 for readability.
    lines = []
    for trick_idx in range(0, len(plays), 4):
        trick = plays[trick_idx : trick_idx + 4]
        cards = ", ".join(entry.get("card", "?") for entry in trick)
        lines.append(f"  trick {trick_idx // 4 + 1}: {cards}")
    return "\n".join(lines)


# --- Public functions (called by main.py) -----------------------------------


def get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str]:
    """Return ``{action_id: action_string}`` for the current state.

    Returns ``{}`` when this player has no legal actions (another seat
    at the table is on call).
    """
    legal_actions = observation.get("legalActions")
    legal_action_strings = observation.get("legalActionStrings")
    if legal_actions and legal_action_strings:
        return dict(zip(legal_actions, legal_action_strings))
    if legal_actions == [] or legal_action_strings == []:
        return {}
    serialized = observation.get("serializedGameAndState", "")
    if not serialized:
        return {}
    _, state = pyspiel.deserialize_game_and_state(serialized)
    player = observation.get("playerId", 0)
    actions = state.legal_actions(player)
    return {a: state.action_to_string(player, a) for a in actions}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current bridge arena state.

    ``move_history`` (this player's own past actions, supplied by the
    framework) is ignored: the arena observation already includes the
    full table-visible auction and play history attributed by player id.
    """
    del move_history
    obs = _parse_obs(observation)
    player_id = observation.get("playerId", obs.get("your_player_id", 0))
    your_table_position = obs.get("your_table_position", "?")
    partner_id = obs.get("your_partner_player_id", "?")
    partner_table_position = obs.get("partner_table_position", "?")
    opponent_ids = obs.get("opponent_player_ids", [])
    table_seating = obs.get("table_seating", {})
    team_id = obs.get("your_team_id", 0)
    team_label = "team A sits N/S" if team_id == 0 else "team B sits E/W"
    dealer_id = obs.get("dealer_player_id", "?")
    dealer_pos = obs.get("dealer_table_position", "?")
    phase = obs.get("phase", "?")
    raw_observation = obs.get("raw_observation", "(no raw observation)")
    auction_str = _format_auction(obs.get("auction", []))
    plays_str = _format_plays(obs.get("plays", []))

    opp0_id = opponent_ids[0] if len(opponent_ids) >= 1 else "?"
    opp1_id = opponent_ids[1] if len(opponent_ids) >= 2 else "?"
    opp0_pos = table_seating.get(str(opp0_id), "?")
    opp1_pos = table_seating.get(str(opp1_id), "?")

    prompt = ARENA_PROMPT_TEMPLATE.format(
        player_id=player_id,
        your_table_position=your_table_position,
        partner_id=partner_id,
        partner_table_position=partner_table_position,
        opp0_id=opp0_id,
        opp0_pos=opp0_pos,
        opp1_id=opp1_id,
        opp1_pos=opp1_pos,
        team_id=team_id,
        team_label=team_label,
        dealer_id=dealer_id,
        dealer_pos=dealer_pos,
        phase=phase,
        raw_observation=raw_observation,
        auction_str=auction_str,
        plays_str=plays_str,
    )

    if previous_response is not None:
        prompt += RETHINK_SUFFIX.format(
            previous_response=previous_response[:500],
            previous_action=previous_action or "(could not parse)",
        )
    return prompt


def parse_response(response: str, legal_action_strings: Sequence[str]) -> ParseResult:
    """Extract a legal bridge call/card from the model's response."""
    raw = _extract_move_from_json(response)
    if raw is not None:
        matched = _match_action_to_legal(raw, legal_action_strings)
        if matched is not None:
            return ParseResult(legal_action=matched, raw_action=raw)

    # Fallback: scan the response text for any legal action string.
    # Prefer the LAST match in the response -- LLMs typically state
    # their final answer at the end after their reasoning, and the
    # rethink prompt also echoes the previous (wrong) answer near the
    # start of the next response. Last-occurrence wins so a model that
    # reconsiders mid-response doesn't get pinned to its first guess.
    legal_norm = {legal: _normalize_action_string(legal) for legal in legal_action_strings}
    best: tuple[int, str, str] | None = None  # (position, legal, token)
    for match in re.finditer(r"\S+", response):
        token = match.group()
        cleaned = token.strip(".,:;()'\"`*")
        norm = _normalize_action_string(cleaned)
        if not norm:
            continue
        for legal, normalized_legal in legal_norm.items():
            if norm == normalized_legal:
                best = (match.start(), legal, cleaned)
                break  # one legal action per token; keep scanning for later tokens
    if best is not None:
        _, legal, token = best
        return ParseResult(legal_action=legal, raw_action=token)

    return ParseResult(legal_action=None, raw_action=raw)
