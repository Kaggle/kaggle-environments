"""LLM harness for the OpenSpiel Bargaining game (Lewis et al. 2017).

Two-player imperfect-information negotiation over a small pool of three item
types (Book, Hat, Basketball). A chance node samples an instance: the pool
quantities (each 0..5, total 5..7) and per-player integer valuations (each
0..10, with each player's vector summing to 10). Players then alternate
offers -- every offer is an allocation the proposing player wants to KEEP for
themselves (the opponent would receive the complement). On any turn after at
least one offer exists, a player may instead ACCEPT the opponent's most
recent offer; acceptance ends the game and each player scores the dot
product of their private valuations with the items they actually receive.
If ``max_turns`` offers are made with no acceptance, both players score 0.

The harness asks the LLM for structured JSON:
  * Offer turn:  ``{"action": "offer", "keep": {"book": A, "hat": B, "basketball": C}}``
  * Accept turn: ``{"action": "agree"}``

We then encode the JSON back to the canonical OpenSpiel action string
(``"Offer: Book: A, Hat: B, Basketball: C"`` or ``"Agree"``) and confirm it
is in ``legalActions``.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import (
    ParseResult,
    extract_last_json_object,
    render_rethink_suffix,
)

_ITEM_KEYS = ("book", "hat", "basketball")
_ITEM_LABELS = {"book": "Book", "hat": "Hat", "basketball": "Basketball"}
_AGREE_ACTION_STRING = "Agree"


# --- Prompt -----------------------------------------------------------------


BARGAINING_PROMPT_TEMPLATE = """Let's play Bargaining (a.k.a. "Deal or No Deal").

Two players negotiate over a shared pool of three item types: Book, Hat,
Basketball. The pool for this game contains:
{pool_lines}

Each player has a PRIVATE integer valuation per item (the opponent's values
are HIDDEN from you). Your reward at the end of the game is the dot product
of your valuation vector with the items you actually receive; the opponent's
reward is computed from their own (hidden) valuation vector. The game is
general-sum -- both players can do well, or both poorly.

Your private valuations (per unit):
{my_value_lines}

How a turn works:
  * On an OFFER turn, you propose an allocation that YOU would keep. The
    opponent would then receive the complement (pool minus your kept items).
    Each kept count must be an integer between 0 and the pool quantity for
    that item, inclusive. You may propose to keep nothing or to keep
    everything -- those are legal offers, just usually bad ones.
  * If a previous opponent offer exists, you may instead ACCEPT it.
    Acceptance ENDS the game immediately: you receive the items the
    opponent offered you (= the complement of what they wanted to keep),
    and the opponent receives the items they wanted to keep. Each player
    then scores their own private dot product over what they received.
  * If {max_turns} offers go by without acceptance, the game ends with
    ZERO reward for both players (no deal is the worst outcome unless your
    private utility from the alternative is positive, which it never is).

Offers made so far (most recent last; each shows what the OFFERING player
wanted to keep -- the other player would receive the complement):
{history_str}

Offers used so far: {num_offers} of {max_turns}. Offers remaining
(including this one if you OFFER): {turns_left}.

You are Player {player_label} (id {player_id}). It is your turn.
{accept_help}

Respond with your reasoning, then conclude with a JSON block of EITHER form:

```json
{{"action": "offer", "keep": {{"book": <int>, "hat": <int>, "basketball": <int>}}}}
```
or
```json
{{"action": "agree"}}
```

For example: `{{"action": "offer", "keep": {{"book": 1, "hat": 0, "basketball": 2}}}}`

The "keep" counts describe what YOU retain -- the opponent receives the
pool minus your kept items. Each kept value must satisfy
``0 <= keep[item] <= pool[item]``.

Failure to output your final answer in the specified JSON format, or
choosing an illegal allocation, will result in a loss.
"""


RETHINK_ILLEGAL = """

You suggested action "{previous_action}" but this is not legal in the
current state. Either the kept counts exceed the pool, or you tried to
ACCEPT when no opponent offer exists yet. Reconsider the pool quantities
and the offer history, then pick a legal action.

(Keep using the same JSON output format as before -- only the action value needs to change.)
"""

RETHINK_UNPARSABLE = """

Your previous response ended with:
{previous_response}

No valid action JSON could be extracted from that. Conclude your response
with your final action as JSON in a ```json fenced block, exactly as the
original instructions required:

```json
{{"action": "offer", "keep": {{"book": <int>, "hat": <int>, "basketball": <int>}}}}
```
or
```json
{{"action": "agree"}}
```

For example: `{{"action": "offer", "keep": {{"book": 1, "hat": 0, "basketball": 2}}}}`

The action you choose must also be legal in the current state.
"""


# --- Observation parsing ----------------------------------------------------


def _parse_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the structured bargaining state dict out of the observation."""
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
        player_id = int(observation.get("playerId", 0))
        try:
            return json.loads(state.observation_string(player_id))
        except (json.JSONDecodeError, RuntimeError):
            pass
    return {}


def _format_items_dict(items: Mapping[str, int]) -> str:
    """Render a {book, hat, basketball} dict as ``Book=A, Hat=B, Basketball=C``."""
    return ", ".join(f"{_ITEM_LABELS[k]}={int(items.get(k, 0))}" for k in _ITEM_KEYS)


def _complement(items: Mapping[str, int], pool: Mapping[str, int]) -> dict[str, int]:
    """Pool minus items, floored at 0."""
    return {k: max(0, int(pool.get(k, 0)) - int(items.get(k, 0))) for k in _ITEM_KEYS}


def _format_history(state: Mapping[str, Any]) -> str:
    """Render the offer/agree timeline as a numbered list."""
    history = state.get("offer_history") or []
    pool = state.get("pool") or {}
    if not history:
        return "(no offers yet -- you are opening the negotiation)"
    lines: list[str] = []
    for i, event in enumerate(history, start=1):
        player = int(event.get("player", 0))
        who = f"Player {player + 1}"
        if event.get("type") == "agree":
            lines.append(f"  {i}. {who} ACCEPTS the previous offer (game ends).")
            continue
        items = event.get("items") or {}
        offered = _complement(items, pool)
        lines.append(
            f"  {i}. {who} offers: keep [{_format_items_dict(items)}] / opponent gets [{_format_items_dict(offered)}]"
        )
    return "\n".join(lines)


# --- Public functions -------------------------------------------------------


def get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str]:
    """Return ``{action_id: action_string}`` for the current state.

    Falls back to deserializing the pyspiel state if the harness obs dict
    omits ``legalActions`` (e.g. when called from tests).
    """
    legal_actions = observation.get("legalActions")
    legal_action_strings = observation.get("legalActionStrings")
    if legal_actions and legal_action_strings:
        return dict(zip(legal_actions, legal_action_strings))

    serialized = observation.get("serializedGameAndState", "")
    if not serialized:
        return {}
    _, state = pyspiel.deserialize_game_and_state(serialized)
    return {a: state.action_to_string(a) for a in state.legal_actions()}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current bargaining state."""
    del move_history  # full offer history (both players) is in the proxy state.
    state = _parse_observation_payload(observation)
    player_id = int(observation.get("playerId", 0))

    pool = state.get("pool") or {k: 0 for k in _ITEM_KEYS}
    my_values = state.get("my_values") or {k: 0 for k in _ITEM_KEYS}
    params = state.get("params") or {}
    max_turns = int(params.get("max_turns", state.get("max_turns", 10)))
    num_offers = int(state.get("num_offers", 0))
    turns_left = max(0, max_turns - num_offers)
    offer_history = state.get("offer_history") or []

    def _unit_word(n: int) -> str:
        return "unit" if n == 1 else "units"

    pool_lines = "\n".join(
        f"  {_ITEM_LABELS[k]}: {int(pool.get(k, 0))} {_unit_word(int(pool.get(k, 0)))}" for k in _ITEM_KEYS
    )
    my_value_lines = "\n".join(f"  {_ITEM_LABELS[k]}: {int(my_values.get(k, 0))}" for k in _ITEM_KEYS)
    history_str = _format_history(state)

    # Acceptance is legal only when the opponent has an open offer on the
    # table (i.e. the last event in history was their offer).
    last_offer_event = offer_history[-1] if offer_history else None
    can_accept = bool(
        last_offer_event
        and last_offer_event.get("type") == "offer"
        and int(last_offer_event.get("player", -1)) != player_id
    )
    if can_accept:
        accepted_items = last_offer_event.get("items") or {}
        you_would_receive = _complement(accepted_items, pool)
        accept_help = (
            "You MAY accept the opponent's most recent offer with"
            ' `{"action": "agree"}`. If you accept, you would receive '
            f"[{_format_items_dict(you_would_receive)}] (their offer to you)."
        )
    else:
        # Default-config bargaining alternates players strictly, so the only
        # way to land here is an empty offer_history (the opening turn).
        accept_help = (
            "No opponent offer exists yet -- you are opening, so you MUST"
            " make an offer (you cannot accept on the first turn)."
        )

    prompt = BARGAINING_PROMPT_TEMPLATE.format(
        pool_lines=pool_lines,
        my_value_lines=my_value_lines,
        max_turns=max_turns,
        num_offers=num_offers,
        turns_left=turns_left,
        history_str=history_str,
        player_label=player_id + 1,
        player_id=player_id,
        accept_help=accept_help,
    )

    prompt += render_rethink_suffix(
        RETHINK_ILLEGAL,
        RETHINK_UNPARSABLE,
        previous_response,
        previous_action,
    )

    return prompt


# --- Response parsing ------------------------------------------------------


_PAYLOAD_KEYS = ("action", "keep", "items", "offer", "agree")
# Tolerate case + pluralization variants for the keep-dict keys: models often
# mirror the prompt's display capitalization ("Book") or add an English plural
# ("books"). Without normalization those silently fall through to 0 in the
# count lookup, producing a legal-but-unintended keep-nothing offer.
_ITEM_KEY_ALIASES = {
    "book": "book", "books": "book",
    "hat": "hat", "hats": "hat",
    "basketball": "basketball", "basketballs": "basketball",
    "ball": "basketball", "balls": "basketball",
}


def _normalize_keep(keep_obj: Mapping[str, Any]) -> dict[str, Any]:
    """Map keep-dict keys to canonical lowercase item keys.

    Later occurrences win on alias collision (e.g. both ``Book`` and ``book``
    present) so the model's last-written value applies.
    """
    out: dict[str, Any] = {}
    for k, v in keep_obj.items():
        canonical = _ITEM_KEY_ALIASES.get(str(k).strip().lower())
        if canonical is not None:
            out[canonical] = v
    return out


def _payload_to_action_string(payload: Mapping[str, Any]) -> str | None:
    """Convert a parsed JSON dict into a canonical legal-action string.

    Returns ``None`` if the dict doesn't look like a valid bargaining
    action (missing keys, non-integer counts, etc.). Returns a string
    even if the action would be illegal in the current state -- the
    caller verifies legality against ``legal_action_strings``.
    """
    action = str(payload.get("action", "")).strip().lower() if payload.get("action") is not None else ""

    if action == "agree" or action == "accept" or payload.get("agree") is True:
        return _AGREE_ACTION_STRING

    keep_obj = payload.get("keep") or payload.get("items") or payload.get("offer")
    if not isinstance(keep_obj, Mapping):
        return None
    normalized = _normalize_keep(keep_obj)
    try:
        counts = {k: int(normalized.get(k, 0)) for k in _ITEM_KEYS}
    except (TypeError, ValueError):
        return None
    if any(v < 0 for v in counts.values()):
        return None
    return f"Offer: Book: {counts['book']}, Hat: {counts['hat']}, Basketball: {counts['basketball']}"


def parse_response(
    response: str,
    legal_action_strings: Sequence[str] | None,
) -> ParseResult:
    """Extract the model's chosen action and match it to a legal one.

    Single intent surface: the LAST JSON object in the response that
    carries any of the bargaining payload keys. No prose-scan fallback --
    if the model didn't write a structured answer, we return
    ``legal_action=None`` so the rethink loop asks for one.
    """
    if not legal_action_strings:
        return ParseResult(raw_action=None)

    payload = extract_last_json_object(response, required_keys=_PAYLOAD_KEYS)
    if payload is None:
        return ParseResult(legal_action=None, raw_action=None)

    candidate = _payload_to_action_string(payload)
    if candidate is None:
        # JSON parsed but didn't decode to any valid action shape (unknown
        # verb, no keep dict, non-int counts, etc.). Route to the
        # unparsable rethink rather than the illegal one, since the illegal
        # template's diagnosis ("kept counts exceed the pool, or Agree with
        # no offer") wouldn't fit a shape failure.
        return ParseResult(legal_action=None, raw_action=None)

    raw_repr = json.dumps(payload, separators=(",", ":"))
    legal_set = set(legal_action_strings)
    if candidate in legal_set:
        return ParseResult(legal_action=candidate, raw_action=raw_repr)
    return ParseResult(legal_action=None, raw_action=raw_repr)


# --- GameHarness adapter ----------------------------------------------------


class _BargainingHarness:
    """Adapts the module-level functions to the ``GameHarness`` protocol."""

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

    agent_fn = create_agent_fn(_BargainingHarness())
except ImportError:  # pragma: no cover - import-time fallback
    agent_fn = None
