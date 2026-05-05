"""LLM harness for OpenSpiel Oshi-Zumo.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.

Oshi-Zumo is a simultaneous-move bidding game: each round both players choose
an integer bid in ``[min_bid, my_coins]``; the higher bid pushes a wrestler
one square toward the opponent's edge of a length ``2*size+1`` field; equal
bids leave the wrestler stationary. Both bids are deducted regardless. The
game ends when the wrestler is pushed off an edge, both players are out of
coins, or the horizon is reached.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{[^{}]*\"bid\"\s*:\s*(-?\d+)[^{}]*\}", re.DOTALL)
_BID_PREFIX_RE = re.compile(r"\[P\d+\]Bid:\s*(\d+)")


# --- Prompt -----------------------------------------------------------------


OSHI_ZUMO_PROMPT_TEMPLATE = """Let's play Oshi-Zumo (sumo push).

Rules: Two wrestlers (you and the opponent) start with the same number of
coins and try to push a token off either edge of a 1D field of length
{field_size}. Each round both players SIMULTANEOUSLY choose an integer bid
of at least {min_bid} coins, up to their current coin total. The higher
bidder pushes the token one square toward the opponent's edge; equal bids
leave it stationary. Both bids are deducted regardless of who won. The
game ends when (a) the token is pushed off an edge — that side loses, or
(b) both players run out of coins, or (c) the {horizon}-round horizon is
reached. At the end, the side the token sits on loses; exactly center is
a draw.

Field (W is the token, # are the off-edge cells):
  {field}
  index: {index_row}

Token position:    {wrestler_position} (center is {center}; lower is your goal, higher is the opponent's)
Your coins:        {my_coins}
Opponent coins:    {opp_coins}
Round:             {move_number}

Your past bids:        {my_history}

You are Player {player_label}. Choose your bid for this round.
You MUST pick one of the legal bids: {legal_bids}.

Respond with your reasoning followed by your final bid in a JSON block:

```json
{{
  "bid": <integer from the legal list>
}}
```

Failure to output your final answer in the specified format, or selecting a
bid that is not in the legal list, will result in a loss.
"""


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested bid "{previous_action}" but it is NOT in the legal bid list.
Reconsider and pick one of the legal bids exactly.
"""


# --- Helpers ----------------------------------------------------------------


def _bid_from_action_string(action_string: str) -> int | None:
    """Extract the integer bid from an action string like ``[P0]Bid: 5``."""
    m = _BID_PREFIX_RE.search(action_string)
    if m:
        return int(m.group(1))
    try:
        return int(action_string.strip())
    except ValueError:
        return None


def _parse_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the structured oshi-zumo state dict out of the observation."""
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


def _format_field_index_row(field_size: int) -> str:
    """Index row aligned under the field, single character per cell."""
    return "".join(str(i % 10) for i in range(field_size))


def _extract_bid_from_json(response: str) -> str | None:
    """Pull the bid from a ```json``` block or a bare ``{"bid": N}``."""
    match = _JSON_BLOCK_RE.search(response)
    if match:
        try:
            data = json.loads(match.group(1))
            bid = data.get("bid")
            if bid is None:
                return None
            return str(bid).strip()
        except json.JSONDecodeError:
            pass
    bare = _BARE_JSON_RE.search(response)
    if bare:
        return bare.group(1).strip()
    return None


def _match_bid_to_legal(
    bid: str,
    legal_action_strings: Sequence[str],
) -> str | None:
    """Match a bid (raw integer string) to one of the legal action strings."""
    if bid is None:
        return None
    try:
        bid_int = int(str(bid).strip())
    except ValueError:
        return None
    for legal in legal_action_strings:
        if _bid_from_action_string(legal) == bid_int:
            return legal
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
    player_id = observation.get("playerId", 0)
    actions = state.legal_actions(player_id)
    return {a: state.action_to_string(player_id, a) for a in actions}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current oshi-zumo state.

    ``move_history`` contains the agent's own past bids (one entry per round
    it acted in). The opponent's past coin counts are encoded as a hidden
    suffix on each entry so the prompt can show the opponent's bid history
    too, since core_harness doesn't track per-round game state.
    """
    state = _parse_observation_payload(observation)
    player_id = observation.get("playerId", 0)
    coins = state.get("coins") or [0, 0]
    my_coins = coins[player_id] if player_id in (0, 1) else coins[0]
    opp_coins = coins[1 - player_id] if player_id in (0, 1) else coins[1]
    params = state.get("params") or {}

    field = state.get("field", "")
    field_size = state.get("field_size", len(field))
    wrestler_position = state.get("wrestler_position", -1)
    center = (field_size - 1) // 2
    move_number = state.get("move_number", 0)
    min_bid = int(params.get("min_bid", 0))
    horizon = int(params.get("horizon", 0))

    legal_action_strings = observation.get("legalActionStrings") or []
    if not legal_action_strings:
        legal_action_strings = list(get_legal_moves(observation).values())
    legal_bids = sorted(
        {b for b in (_bid_from_action_string(s) for s in legal_action_strings) if b is not None}
    )
    legal_bids_str = ", ".join(str(b) for b in legal_bids) or "(none)"

    my_bids = [_bid_from_action_string(s) for s in move_history]
    my_history_str = (
        ", ".join(str(b) for b in my_bids if b is not None)
        or "(none yet)"
    )

    prompt = OSHI_ZUMO_PROMPT_TEMPLATE.format(
        field_size=field_size,
        min_bid=min_bid,
        horizon=horizon,
        field=field or "(unavailable)",
        index_row=_format_field_index_row(field_size) if field_size else "",
        wrestler_position=wrestler_position,
        center=center,
        my_coins=my_coins,
        opp_coins=opp_coins,
        move_number=move_number,
        my_history=my_history_str,
        player_label=player_id,
        legal_bids=legal_bids_str,
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
    """Extract a legal bid from the LLM response.

    Tries a ``json`` block first, then a bare ``{"bid": N}``, then falls
    back to scanning for any legal ``[Pk]Bid: N`` substring or a standalone
    integer that matches a legal bid.
    """
    raw = _extract_bid_from_json(response)
    if raw is not None:
        matched = _match_bid_to_legal(raw, legal_action_strings)
        if matched is not None:
            return ParseResult(legal_action=matched, raw_action=raw)

    for legal in legal_action_strings:
        if legal in response:
            return ParseResult(legal_action=legal, raw_action=raw or legal)

    legal_bids = {
        _bid_from_action_string(s): s for s in legal_action_strings
    }
    legal_bids.pop(None, None)
    for n in sorted(legal_bids.keys(), reverse=True):
        pattern = r"(?<!\d)" + re.escape(str(n)) + r"(?!\d)"
        if re.search(pattern, response):
            return ParseResult(legal_action=legal_bids[n], raw_action=raw or str(n))

    return ParseResult(legal_action=None, raw_action=raw)
