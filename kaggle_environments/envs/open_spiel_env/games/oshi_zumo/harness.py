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

from kaggle_environments.core_harness import ParseResult, parse_json_action

_BID_PREFIX_RE = re.compile(r"\[P\d+\]Bid:\s*(\d+)")


# --- Prompt -----------------------------------------------------------------


OSHI_ZUMO_PROMPT_TEMPLATE = """Let's play Oshi-Zumo (sumo push).

Rules: Two wrestlers (you and the opponent) start with the same number of
coins and try to push a token off either edge of a 1D field of length
{field_size}. Each round both players SIMULTANEOUSLY choose an integer bid
of at least {min_bid} coins, up to their current coin total. The higher
bidder pushes the token one cell toward the opponent's edge; equal bids
leave it stationary. Both bids are deducted regardless of who won.

The game ends when (a) the token is pushed off either edge — the player
whose edge it falls off LOSES, or (b) both players run out of coins, or
(c) the {horizon}-round horizon is reached. If the game ends by (b) or
(c), the player whose half of the field the token is currently on LOSES;
if the token is exactly at the center, it is a draw.

Field (W is the token, # are the off-edge cells):
  {field}
  index: {index_row}

You are Player {player_label}.
  - You WIN if the token is pushed off index {opp_edge_index} (the opponent's edge).
  - You LOSE if the token is pushed off index {your_edge_index} (your edge).
  - To push the token toward index {opp_edge_index}, you need to OUT-BID the opponent.

Token position:    {wrestler_position} (center is {center})
Your coins:        {my_coins}
Opponent coins:    {opp_coins}
Round:             {move_number}

Your past bids:    {my_history}

Choose your bid for this round (an integer at least {min_bid} and at
most your current coin total).

Respond with your reasoning followed by your final bid in a JSON block:

```json
{{
  "bid": <integer>
}}
```

Failure to output your final answer in the specified format, or selecting an
illegal bid, will result in a loss.
"""


RETHINK_SUFFIX = """

Your previous response was:
{previous_response}

You suggested bid "{previous_action}" but it is not a legal bid.
Reconsider your coin total and the minimum bid, then pick a legal integer bid.
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
    it acted in). The opponent's per-round bid history is not available to
    the harness (core_harness only forwards this agent's own action history);
    only the opponent's current coin total appears in the prompt.
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

    # Engine: wrestler_pos == 0 -> P1 wins; wrestler_pos == field_size-1 -> P0 wins.
    # So P0's losing edge is index 0 and winning edge is field_size-1; vice versa for P1.
    your_edge_index = 0 if player_id == 0 else max(field_size - 1, 0)
    opp_edge_index = max(field_size - 1, 0) if player_id == 0 else 0

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
        your_edge_index=your_edge_index,
        opp_edge_index=opp_edge_index,
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
    """Trust the model's JSON answer; let the rethink loop fix anything else."""
    return parse_json_action(
        response, legal_action_strings,
        json_key="bid",
        matcher=_match_bid_to_legal,
    )
