"""LLM harness for the OpenSpiel Negotiation game.

Negotiation is a two-player imperfect-information game. On each round, a
player either makes a structured proposal (a per-item allocation that they
would keep) or accepts the opponent's most recent proposal. When utterances
are enabled, each proposal turn is followed by an utterance turn for the
same player -- a vector of symbols with no game effect, used as a private
communication channel.

The proxy state encodes:
  * action ids ``0 .. (max_quantity+1)^num_items - 1`` -- proposals (base-N
    digits of the kept allocation, big-endian)
  * action id ``num_distinct_proposals - 1`` -- accept (legal only after at
    least one proposal exists)
  * action ids ``num_distinct_proposals .. + num_distinct_utterances`` --
    utterances (base num_symbols digits of the utterance vector)

The harness asks the LLM for structured JSON instead of a raw integer:
  * Proposal turn:  ``{"action": "propose", "keep": [a, b, c]}``
                    ``{"action": "accept"}``
  * Utterance turn: ``{"symbols": [x, y, z]}``

We then encode the JSON back to an action id and confirm it is in
``legalActions``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from kaggle_environments.core_harness import ParseResult, extract_last_json_object

_INT_LIST_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


# --- Prompt -----------------------------------------------------------------


PROPOSAL_PROMPT_TEMPLATE = """Let's play Negotiation.

Two players split a shared pool of items. There are {num_items} item types
with item indexes 0..{num_items_m1}. The pool currently holds:
{pool_lines}

Each player has a PRIVATE utility per item (the opponent's utilities are
hidden from you). Your reward at the end is the dot product of your
utility vector and the items you actually receive; the opponent's reward
uses their own utility vector. The game is general-sum: you can both do
well or both do poorly.

Your utility vector (item index : utility per unit):
{my_util_lines}

How a round works:
  * It's a "proposal" turn: you propose an allocation that YOU would keep.
    The opponent would then receive the remainder of the pool. A proposal
    is just a vector ``[k_0, k_1, ...]`` with ``0 <= k_i <= pool[i]``.
  * If a previous opponent proposal exists, you may instead ACCEPT it.
    Acceptance ends the game immediately: the most-recent proposer keeps
    the items they offered themselves; the other player receives the
    remainder of the pool.
  * If neither side accepts within {max_steps} proposals, the game ends
    with zero reward for both.{utterance_note}

History so far (most recent last):
{history_str}

You are Player {player_label} (id {player_id}). It is your PROPOSAL turn.
{accept_help}

Respond with your reasoning, then conclude with a JSON block of EITHER
form:

```json
{{"action": "propose", "keep": [<int>, <int>, ...]}}
```
or
```json
{{"action": "accept"}}
```

Failure to output legal JSON in the specified format will result in a loss.
"""


UTTERANCE_PROMPT_TEMPLATE = """Let's play Negotiation -- utterance turn.

You just made a {last_action_desc}. The rules force you to emit an
utterance immediately afterward: a vector of {utterance_dim} symbols from
the alphabet ``0..{num_symbols_m1}``. Utterances are PRIVATE to the two
players and have no mechanical effect on rewards -- they are a free
channel for signalling intent.

Your private utility vector: {my_util_compact}
Current item pool: {pool_compact}

Your most recent proposal (what you offered to keep): {last_proposal}

History so far (most recent last):
{history_str}

Respond with your reasoning, then conclude with a JSON block:

```json
{{"symbols": [<int>, <int>, ...]}}
```

The list must contain exactly {utterance_dim} integers, each between 0
and {num_symbols_m1} inclusive.
"""


RETHINK_ILLEGAL = """

Your previous response was:
{previous_response}

You suggested action "{previous_action}" but it is not legal in this
state. Re-read the rules and the current state, then pick a legal action
in the required JSON format.

(Keep using the same JSON output format as before -- only the action value needs to change.)
"""

RETHINK_UNPARSABLE = """

Your previous response could not be parsed -- no JSON action answer
was found. Conclude your response with your final answer as JSON in
a ```json fenced block, exactly as the original instructions required:

```json
{"action": "propose", "keep": [<int>, <int>, ...]}
```

The action you choose must also be legal in the current state.
"""


# --- Constants from the proxy ----------------------------------------------


_MAX_QUANTITY = 5  # mirrors negotiation.h kMaxQuantity


def _encode_base(digits: Sequence[int], base: int) -> int:
    value = 0
    for d in digits:
        value = value * base + int(d)
    return value


def _decode_base(value: int, dimensions: int, base: int) -> list[int]:
    digits = [0] * dimensions
    i = dimensions - 1
    while value > 0 and i >= 0:
        digits[i] = value % base
        value //= base
        i -= 1
    return digits


# --- Observation parsing ---------------------------------------------------


def _parse_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    raw = observation.get("observationString", "") or ""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {}


def _format_history(state: Mapping[str, Any]) -> str:
    proposals = state.get("proposals") or []
    utterances = state.get("utterances") or []
    enable_utt = state.get("params", {}).get("enable_utterances", True)
    pool = state.get("item_pool") or []
    lines: list[str] = []
    for i, p in enumerate(proposals):
        who = f"Player {int(p.get('player', 0)) + 1}"
        if p.get("accept"):
            lines.append(f"{who}: ACCEPTS")
            continue
        kept = p.get("items") or []
        if pool and len(pool) == len(kept):
            offered = [max(0, pool[j] - kept[j]) for j in range(len(pool))]
            lines.append(f"{who}: proposes keep={kept}, offer={offered}")
        else:
            lines.append(f"{who}: proposes keep={kept}")
        if enable_utt and i < len(utterances):
            symbols = utterances[i].get("symbols") or []
            lines.append(f"{who}: utters [{', '.join(str(s) for s in symbols)}]")
    return "\n".join(lines) if lines else "(empty)"


# --- Public functions ------------------------------------------------------


def get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str]:
    """Return ``{action_id: action_string}`` for the current state.

    Falls back to a derived set when ``legalActions`` is missing: during
    proposal turns we enumerate every legal split of the pool (plus the
    accept action if a previous proposal exists); during utterance turns we
    enumerate every legal utterance vector. Either is bounded by the small
    default parameters (3 items at quantity 5, or 5^3 utterances).
    """
    legal_actions = observation.get("legalActions")
    legal_action_strings = observation.get("legalActionStrings")
    if legal_actions and legal_action_strings:
        return dict(zip(legal_actions, legal_action_strings))

    state = _parse_observation_payload(observation)
    if not state:
        return {}
    params = state.get("params") or {}
    num_items = int(params.get("num_items", 3))
    num_distinct_proposals = int(params.get("num_distinct_proposals", 217))
    accept_action = int(params.get("accept_action", num_distinct_proposals - 1))
    num_symbols = int(params.get("num_symbols", 5))
    utterance_dim = int(params.get("utterance_dim", 3))
    enable_utt = bool(params.get("enable_utterances", True))
    turn_type = state.get("turn_type", "proposal")
    pool = state.get("item_pool") or [_MAX_QUANTITY] * num_items
    proposals = state.get("proposals") or []

    moves: dict[int, str] = {}
    if turn_type == "proposal":
        digits = [0] * num_items
        while True:
            if all(d <= pool[i] for i, d in enumerate(digits)):
                action_id = _encode_base(digits, _MAX_QUANTITY + 1)
                moves[action_id] = f"Proposal: [{', '.join(str(d) for d in digits)}]"
            i = num_items - 1
            while i >= 0:
                digits[i] += 1
                if digits[i] <= _MAX_QUANTITY:
                    break
                digits[i] = 0
                i -= 1
            if i < 0:
                break
        if proposals:
            moves[accept_action] = "Proposal: Agreement reached!"
    elif turn_type == "utterance" and enable_utt:
        total = num_symbols**utterance_dim
        for v in range(total):
            digits = _decode_base(v, utterance_dim, num_symbols)
            action_id = num_distinct_proposals + v
            moves[action_id] = f", Utterance: [{', '.join(str(d) for d in digits)}]"
    return moves


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    state = _parse_observation_payload(observation)
    player_id = int(observation.get("playerId", 0))
    params = state.get("params") or {}
    num_items = int(params.get("num_items", 3))
    num_symbols = int(params.get("num_symbols", 5))
    utterance_dim = int(params.get("utterance_dim", 3))
    enable_utt = bool(params.get("enable_utterances", True))
    pool = state.get("item_pool") or [0] * num_items
    my_util = state.get("my_utilities") or [0] * num_items
    proposals = state.get("proposals") or []
    turn_type = state.get("turn_type", "proposal")
    max_steps = int(state.get("max_steps") or 7)

    pool_lines = "\n".join(f"  item {i}: {qty} units in pool" for i, qty in enumerate(pool))
    my_util_lines = "\n".join(
        f"  item {i}: {u} per unit (so a unit of item {i} is worth {u} to you)" for i, u in enumerate(my_util)
    )
    my_util_compact = "[" + ", ".join(str(u) for u in my_util) + "]"
    pool_compact = "[" + ", ".join(str(q) for q in pool) + "]"
    history_str = _format_history(state)

    if turn_type == "utterance" and enable_utt:
        # Identify the proposal this player just made (the last entry whose
        # player matches us). If it was an accept, frame it differently.
        last_proposal = "(none)"
        last_action_desc = "proposal"
        for p in reversed(proposals):
            if int(p.get("player", -1)) == player_id:
                if p.get("accept"):
                    last_proposal = "you ACCEPTED the opponent's last offer"
                    last_action_desc = "decision to ACCEPT"
                else:
                    items = p.get("items") or []
                    last_proposal = f"keep={items}"
                prompt = UTTERANCE_PROMPT_TEMPLATE.format(
                    num_items=num_items,
                    utterance_dim=utterance_dim,
                    num_symbols_m1=num_symbols - 1,
                    my_util_compact=my_util_compact,
                    pool_compact=pool_compact,
                    last_proposal=last_proposal,
                    history_str=history_str,
                    last_action_desc=last_action_desc,
                )
                break
        else:
            prompt = UTTERANCE_PROMPT_TEMPLATE.format(
                num_items=num_items,
                utterance_dim=utterance_dim,
                num_symbols_m1=num_symbols - 1,
                my_util_compact=my_util_compact,
                pool_compact=pool_compact,
                last_proposal=last_proposal,
                history_str=history_str,
                last_action_desc=last_action_desc,
            )
    else:
        # Proposal turn.
        has_open_offer = any(not p.get("accept") and int(p.get("player", -1)) != player_id for p in proposals)
        accept_help = (
            'You MAY accept the opponent\'s last proposal with `{"action": "accept"}`.'
            if has_open_offer
            else "There is no opponent proposal to accept yet -- you must propose."
        )
        utterance_note = (
            f"\n  * After each proposal you also emit a private utterance (a vector"
            f" of {utterance_dim} symbols in 0..{num_symbols - 1}). It has no"
            f" mechanical effect."
            if enable_utt
            else ""
        )
        prompt = PROPOSAL_PROMPT_TEMPLATE.format(
            num_items=num_items,
            num_items_m1=num_items - 1,
            pool_lines=pool_lines,
            my_util_lines=my_util_lines,
            max_steps=max_steps,
            utterance_note=utterance_note,
            history_str=history_str,
            player_label=player_id + 1,
            player_id=player_id,
            accept_help=accept_help,
        )

    # Move history (the harness framework already tracks this player's own
    # past action strings); fold it into a compact suffix.
    if move_history:
        prompt += "\nYour own past submissions: " + " | ".join(move_history[-6:])

    if previous_response is not None:
        if previous_action:
            prompt += RETHINK_ILLEGAL.format(
                previous_response=previous_response[:500],
                previous_action=previous_action,
            )
        else:
            prompt += RETHINK_UNPARSABLE

    return prompt


# --- Response parsing ------------------------------------------------------


_PAYLOAD_KEYS = (
    "action", "accept", "keep", "items", "proposal", "symbols", "utterance",
)


def _extract_payload(response: str) -> dict[str, Any] | None:
    """Pull the LAST JSON object that carries a negotiation action field."""
    return extract_last_json_object(response, required_keys=_PAYLOAD_KEYS)


def _parse_int_list(value: Any) -> list[int] | None:
    if isinstance(value, list):
        try:
            return [int(x) for x in value]
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        match = _INT_LIST_RE.search(value)
        if match:
            try:
                return [int(x.strip()) for x in match.group(1).split(",")]
            except ValueError:
                return None
    return None


def _candidate_action_strings(payload: dict[str, Any]) -> list[str]:
    """Convert a parsed JSON dict into candidate legal-action strings."""
    candidates: list[str] = []
    action = str(payload.get("action", "")).strip().lower() if payload.get("action") else ""

    if action == "accept" or payload.get("accept") is True:
        candidates.append("Proposal: Agreement reached!")
        return candidates

    # Proposal: items either at "keep", "items", or top-level list.
    keep = payload.get("keep") or payload.get("items") or payload.get("proposal")
    items = _parse_int_list(keep)
    if items is not None and action in ("", "propose", "proposal"):
        candidates.append(f"Proposal: [{', '.join(str(i) for i in items)}]")

    # Utterance: "symbols" or "utterance" key.
    symbols = payload.get("symbols") or payload.get("utterance")
    syms = _parse_int_list(symbols)
    if syms is not None:
        candidates.append(f", Utterance: [{', '.join(str(i) for i in syms)}]")

    return candidates


def parse_response(
    response: str,
    legal_action_strings: Sequence[str] | None,
) -> ParseResult:
    if not legal_action_strings:
        return ParseResult(raw_action=response[:200])

    legal_set = set(legal_action_strings)
    payload = _extract_payload(response)

    if payload is None:
        # The model didn't give a structured answer at all. Return None
        # so the rethink loop asks for one rather than guessing at the
        # intent from bracket-lists or stray "accept" keywords in the
        # prose (both of which silently substituted moves the model
        # never chose).
        return ParseResult(legal_action=None, raw_action=None)

    # The model gave a structured answer. Trust it: submit if any
    # derived candidate is legal, otherwise surface raw_action with
    # legal_action=None so the rethink loop fires.
    raw_repr = json.dumps(payload, separators=(",", ":"))
    for candidate in _candidate_action_strings(payload):
        if candidate in legal_set:
            return ParseResult(legal_action=candidate, raw_action=raw_repr)
    return ParseResult(legal_action=None, raw_action=raw_repr)


# --- GameHarness adapter ----------------------------------------------------


class _NegotiationHarness:
    def get_legal_moves(self, observation: Mapping[str, Any]) -> dict[int, str]:
        return get_legal_moves(observation)

    def make_prompt(
        self,
        observation: Mapping[str, Any],
        move_history: list[str],
        previous_response: str | None = None,
        previous_action: str | None = None,
    ) -> str:
        return generate_prompt(observation, move_history, previous_response, previous_action)

    def parse_response(self, response: str, legal_action_strings: Sequence[str] | None) -> ParseResult:
        return parse_response(response, legal_action_strings)


# Lazy import so the module can be imported without litellm available
# (the framework requires it when actually running an agent).
try:
    from kaggle_environments.core_harness import create_agent_fn

    agent_fn = create_agent_fn(_NegotiationHarness())
except ImportError:  # pragma: no cover - import-time fallback
    agent_fn = None
