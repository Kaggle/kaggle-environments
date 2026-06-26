"""Bargaining prompt variants for prompt-sensitivity ablation studies.

The canonical production prompt lives in :mod:`harness`. This module holds
alternative variants so :mod:`harness_experiment` can A/B their effect on
agent behavior without touching the deployed file.

Each variant is a :class:`PromptVariant` carrying:

* ``item_labels`` -- display labels for the three items (e.g.
  ``Book/Hat/Basketball`` or ``A/B/C``).
* ``schema_keys`` -- the JSON keys the prompt asks the model to use in its
  ``"keep"`` dict (matches the display labels' lowercase form for the
  baseline; diverges for the GENERIC_NAMES variant).
* ``aliases`` -- the full parser alias map mapping any JSON key the model
  might emit (case-folded, stripped) back to a canonical item key
  (``book``/``hat``/``basketball``). Each variant ships a complete map; the
  experiment harness picks it up directly.
* ``build_body`` -- a callable that renders the per-state prompt body
  (everything before the rethink suffix).
* ``rethink_illegal`` / ``rethink_unparsable`` -- the templates appended by
  :func:`core_harness.render_rethink_suffix` on retry. The unparsable
  template embeds the JSON schema, so it has to vary with ``schema_keys``.

Variants shipped:

* ``BASELINE`` -- byte-identical to ``harness.py`` (control arm).
* ``COMPACT`` -- same information, terser language; rules and reminders are
  collapsed and the worked example / loss-threat are dropped.
* ``MINIMAL`` -- maximally stripped: just pool, values, history, schema.
  Tests how much of the game mechanics the model already knows.
* ``NO_ACCEPT_PREVIEW`` -- baseline minus the "you would receive [...]"
  preview on the accept branch. Tests whether models can compute the
  acceptance allocation themselves.
* ``GENERIC_NAMES`` -- baseline with items renamed Book/Hat/Basketball ->
  A/B/C (in display labels, history, and JSON schema). Tests whether the
  real names carry semantic priors that bias allocations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import (
    ParseResult,
    extract_last_json_object,
    render_rethink_suffix,
)

_ITEM_KEYS = ("book", "hat", "basketball")
_AGREE_ACTION_STRING = "Agree"
# JSON keys we accept as carriers of a bargaining action (used to filter
# stray JSON objects in the model's reasoning).
_PAYLOAD_KEYS = ("action", "keep", "items", "offer", "agree")


# ---------------------------------------------------------------------------
# Shared rendering helpers
# ---------------------------------------------------------------------------


def parse_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the structured bargaining state dict out of the observation.

    Same fallback chain as :mod:`harness`: prefer the proxy's JSON
    ``observationString``; if absent, deserialize the pyspiel state and ask
    it directly.
    """
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


def _complement(items: Mapping[str, int], pool: Mapping[str, int]) -> dict[str, int]:
    """Pool minus items, floored at 0."""
    return {k: max(0, int(pool.get(k, 0)) - int(items.get(k, 0))) for k in _ITEM_KEYS}


def _format_items_dict(items: Mapping[str, int], labels: Mapping[str, str]) -> str:
    """Render a {book, hat, basketball} dict as ``L1=A, L2=B, L3=C``."""
    return ", ".join(f"{labels[k]}={int(items.get(k, 0))}" for k in _ITEM_KEYS)


def _unit_word(n: int) -> str:
    return "unit" if n == 1 else "units"


def _format_history_rich(state: Mapping[str, Any], labels: Mapping[str, str]) -> str:
    """Numbered timeline showing both ``keep`` and the complement per offer."""
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
            f"  {i}. {who} offers: keep [{_format_items_dict(items, labels)}]"
            f" / opponent gets [{_format_items_dict(offered, labels)}]"
        )
    return "\n".join(lines)


def _format_history_minimal(state: Mapping[str, Any], labels: Mapping[str, str]) -> str:
    """Terser history: just ``Px keep: L1=A L2=B L3=C`` per row, no complement."""
    history = state.get("offer_history") or []
    if not history:
        return "  (none)"
    lines: list[str] = []
    for event in history:
        player = int(event.get("player", 0))
        who = f"P{player + 1}"
        if event.get("type") == "agree":
            lines.append(f"  {who} agree")
            continue
        items = event.get("items") or {}
        parts = " ".join(f"{labels[k]}={int(items.get(k, 0))}" for k in _ITEM_KEYS)
        lines.append(f"  {who} keep: {parts}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-state context passed to each variant's build function
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptContext:
    """Everything build_body needs that's derived from the live observation."""

    state: Mapping[str, Any]
    player_id: int
    pool: Mapping[str, int]
    my_values: Mapping[str, int]
    max_turns: int
    discount: float
    num_offers: int
    turns_left: int
    offer_history: list
    can_accept: bool
    last_offer_event: Mapping[str, Any] | None


def build_context(observation: Mapping[str, Any]) -> PromptContext:
    """Parse the observation into a :class:`PromptContext`.

    Lives here (not in harness_experiment.py) so variant build functions
    can also be exercised directly from tests or a sweep runner.
    """
    state = parse_observation_payload(observation)
    player_id = int(observation.get("playerId", 0))

    pool = state.get("pool") or {k: 0 for k in _ITEM_KEYS}
    my_values = state.get("my_values") or {k: 0 for k in _ITEM_KEYS}
    params = state.get("params") or {}
    max_turns = int(params.get("max_turns", state.get("max_turns", 10)))
    discount = float(params.get("discount", 1.0))
    num_offers = int(state.get("num_offers", 0))
    turns_left = max(0, max_turns - num_offers)
    offer_history = state.get("offer_history") or []

    last_offer_event = offer_history[-1] if offer_history else None
    can_accept = bool(
        last_offer_event
        and last_offer_event.get("type") == "offer"
        and int(last_offer_event.get("player", -1)) != player_id
    )

    return PromptContext(
        state=state,
        player_id=player_id,
        pool=pool,
        my_values=my_values,
        max_turns=max_turns,
        discount=discount,
        num_offers=num_offers,
        turns_left=turns_left,
        offer_history=offer_history,
        can_accept=can_accept,
        last_offer_event=last_offer_event,
    )


# ---------------------------------------------------------------------------
# PromptVariant
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptVariant:
    """A self-contained Bargaining prompt configuration.

    Implements the :class:`core_harness.GameHarness` protocol directly so
    the ablation runner can do ``create_agent_fn(variant)`` with no wrapper.
    Production stays on :mod:`harness`; this is the experimental surface.
    """

    name: str
    item_labels: Mapping[str, str]
    schema_keys: tuple[str, str, str]
    aliases: Mapping[str, str]
    build_body: Callable[[PromptContext, "PromptVariant"], str]
    rethink_illegal: str
    rethink_unparsable: str

    # -- GameHarness protocol ------------------------------------------------

    def get_legal_moves(self, observation: Mapping[str, Any]) -> dict[int, str]:
        """Return ``{action_id: action_string}`` for the current state.

        Legality is a property of the OpenSpiel state, not the prompt, so
        every variant uses the same lookup path.
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

    def make_prompt(
        self,
        observation: Mapping[str, Any],
        move_history: list[str],
        previous_response: str | None = None,
        previous_action: str | None = None,
    ) -> str:
        """Render the prompt body via this variant + the variant's rethink suffix."""
        del move_history  # offer history lives in the proxy state
        ctx = build_context(observation)
        body = self.build_body(ctx, self)
        body += render_rethink_suffix(
            self.rethink_illegal,
            self.rethink_unparsable,
            previous_response,
            previous_action,
        )
        return body

    def parse_response(
        self,
        response: str,
        legal_action_strings: Sequence[str] | None,
        *,
        observation: Mapping[str, Any] | None = None,
    ) -> ParseResult:
        """Extract the model's chosen action and match it to a legal one.

        Same single-intent-surface contract as ``harness.parse_response``:
        the LAST JSON object carrying any bargaining payload key, normalized
        via this variant's alias map.
        """
        del observation
        if not legal_action_strings:
            return ParseResult(raw_action=None)
        payload = extract_last_json_object(response, required_keys=_PAYLOAD_KEYS)
        if payload is None:
            return ParseResult(legal_action=None, raw_action=None)
        candidate = self._payload_to_action_string(payload)
        if candidate is None:
            return ParseResult(legal_action=None, raw_action=None)
        raw_repr = json.dumps(payload, separators=(",", ":"))
        if candidate in set(legal_action_strings):
            return ParseResult(legal_action=candidate, raw_action=raw_repr)
        return ParseResult(legal_action=None, raw_action=raw_repr)

    # -- helpers used by parse_response --------------------------------------

    def _payload_to_action_string(self, payload: Mapping[str, Any]) -> str | None:
        """Convert a parsed JSON dict into the canonical OpenSpiel action string.

        Output is always ``"Offer: Book: A, Hat: B, Basketball: C"`` /
        ``"Agree"`` regardless of variant -- only the input JSON keys vary,
        and the variant's alias map normalizes those.
        """
        action_val = payload.get("action")
        action = str(action_val).strip().lower() if action_val is not None else ""
        if action in ("agree", "accept") or payload.get("agree") is True:
            return _AGREE_ACTION_STRING

        keep_obj = payload.get("keep") or payload.get("items") or payload.get("offer")
        if not isinstance(keep_obj, Mapping):
            return None
        normalized: dict[str, Any] = {}
        for k, v in keep_obj.items():
            canonical = self.aliases.get(str(k).strip().lower())
            if canonical is not None:
                normalized[canonical] = v
        try:
            counts = {k: int(normalized.get(k, 0)) for k in _ITEM_KEYS}
        except (TypeError, ValueError):
            return None
        if any(v < 0 for v in counts.values()):
            return None
        return (
            f"Offer: Book: {counts['book']}, "
            f"Hat: {counts['hat']}, "
            f"Basketball: {counts['basketball']}"
        )


# ---------------------------------------------------------------------------
# Label sets and parser aliases
# ---------------------------------------------------------------------------


_BASELINE_LABELS: Mapping[str, str] = {
    "book": "Book",
    "hat": "Hat",
    "basketball": "Basketball",
}
_GENERIC_LABELS: Mapping[str, str] = {
    "book": "A",
    "hat": "B",
    "basketball": "C",
}

_BASELINE_ALIASES: Mapping[str, str] = {
    "book": "book", "books": "book",
    "hat": "hat", "hats": "hat",
    "basketball": "basketball", "basketballs": "basketball",
    "ball": "basketball", "balls": "basketball",
}
# GENERIC_NAMES intentionally still accepts the real names. The ablation is
# about what's in the PROMPT, not about parser tolerance -- if the model
# leaks "book" we want the move to land so the run isn't lost to a parse
# failure that confounds the signal.
_GENERIC_ALIASES: Mapping[str, str] = {
    **_BASELINE_ALIASES,
    "a": "book",
    "b": "hat",
    "c": "basketball",
}


# ---------------------------------------------------------------------------
# Rethink templates
# ---------------------------------------------------------------------------


# The illegal-action template is schema-agnostic: it points the model at
# the pool / offer-history constraint, not at the schema. Shared verbatim
# across all variants.
_RETHINK_ILLEGAL = """

You suggested action "{previous_action}" but this is not legal in the
current state. Either the kept counts exceed the pool, or you tried to
ACCEPT when no opponent offer exists yet. Reconsider the pool quantities
and the offer history, then pick a legal action.

(Keep using the same JSON output format as before -- only the action value needs to change.)
"""


def _make_rethink_unparsable(schema_keys: tuple[str, str, str]) -> str:
    """Build an unparsable-rethink template carrying this variant's schema."""
    a, b, c = schema_keys
    return f"""

Your previous response ended with:
{{previous_response}}

No valid action JSON could be extracted from that. Conclude your response
with your final action as JSON in a ```json fenced block, exactly as the
original instructions required:

```json
{{{{"action": "offer", "keep": {{{{"{a}": <int>, "{b}": <int>, "{c}": <int>}}}}}}}}
```
or
```json
{{{{"action": "agree"}}}}
```

For example: `{{{{"action": "offer", "keep": {{{{"{a}": 1, "{b}": 0, "{c}": 2}}}}}}}}`

The action you choose must also be legal in the current state.
"""


# ---------------------------------------------------------------------------
# Variant: BASELINE  (byte-identical to harness.py)
# ---------------------------------------------------------------------------


_BASELINE_TEMPLATE = """Let's play Bargaining (a.k.a. "Deal or No Deal").

Two players negotiate over a shared pool of three item types: Book, Hat,
Basketball. The pool for this game contains:
{pool_lines}

Each player has a PRIVATE integer valuation per item (the opponent's values are
HIDDEN from you). Your reward at the end of the game is the dot product of your
valuation vector with the items you actually receive; the opponent's reward is
computed from their own (hidden) valuation vector. Each per-item valuation is
between 0 and 10 inclusive, and the values are sampled such that each player's
dot product with the pool equals 10 (their maximum possible reward).

Your goal is to end the game with a higher reward than your opponent.

Your private valuations (per unit):
{my_value_lines}

How a turn works:
  * On an OFFER turn, you propose an allocation that YOU would keep. The
    opponent would then receive the complement (pool minus your kept items).
    Each kept count must be an integer between 0 and the pool quantity for
    that item, inclusive.
  * If a previous opponent offer exists, you may instead ACCEPT it.
    Acceptance ENDS the game immediately: you receive the items the
    opponent offered you (= the complement of what they wanted to keep),
    and the opponent receives the items they wanted to keep. Each player
    then scores their own private dot product over what they received.
  * If {max_turns} offers go by without acceptance, the game ends with
    ZERO reward for both players (this is a tie).
{discount_note}
Offers made so far (most recent last; each shows what the OFFERING player
wanted to keep -- the other player would receive the complement):
{history_str}

Offers used so far: {num_offers} of {max_turns}. Offers remaining
(including this one if you OFFER): {turns_left}.

You are Player {player_label}. It is your turn.
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


def _discount_note_baseline(discount: float) -> str:
    """Same wording as harness.py; suppressed entirely when gamma == 1."""
    if discount >= 1.0:
        return ""
    return (
        f"  * Payoffs are discounted by a factor of {discount} per"
        " additional offer past the first. Accepting the very first"
        " offer is UNDISCOUNTED; if agreement is reached only after a"
        f" 2nd offer has been made, both players' payoffs are multiplied"
        f" by {discount}; after a 3rd offer, by {discount}^2; in"
        f" general, after the Nth offer, by {discount}^(N-1). Earlier"
        " acceptance preserves more reward.\n"
    )


def _accept_help_with_preview(ctx: PromptContext, labels: Mapping[str, str]) -> str:
    accepted_items = (ctx.last_offer_event or {}).get("items") or {}
    you_would_receive = _complement(accepted_items, ctx.pool)
    return (
        "You MAY accept the opponent's most recent offer with"
        ' `{"action": "agree"}`. If you accept, you would receive '
        f"[{_format_items_dict(you_would_receive, labels)}] (their offer to you)."
    )


def _accept_help_no_preview() -> str:
    return (
        "You MAY accept the opponent's most recent offer with"
        ' `{"action": "agree"}`.'
    )


def _accept_help_opening() -> str:
    return (
        "No opponent offer exists yet -- you are opening, so you MUST"
        " make an offer (you cannot accept on the first turn)."
    )


def _build_baseline(ctx: PromptContext, variant: PromptVariant) -> str:
    labels = variant.item_labels
    pool_lines = "\n".join(
        f"  {labels[k]}: {int(ctx.pool.get(k, 0))} {_unit_word(int(ctx.pool.get(k, 0)))}"
        for k in _ITEM_KEYS
    )
    my_value_lines = "\n".join(
        f"  {labels[k]}: {int(ctx.my_values.get(k, 0))}" for k in _ITEM_KEYS
    )
    history_str = _format_history_rich(ctx.state, labels)
    accept_help = (
        _accept_help_with_preview(ctx, labels) if ctx.can_accept else _accept_help_opening()
    )
    return _BASELINE_TEMPLATE.format(
        pool_lines=pool_lines,
        my_value_lines=my_value_lines,
        max_turns=ctx.max_turns,
        discount_note=_discount_note_baseline(ctx.discount),
        num_offers=ctx.num_offers,
        turns_left=ctx.turns_left,
        history_str=history_str,
        player_label=ctx.player_id + 1,
        accept_help=accept_help,
    )


# ---------------------------------------------------------------------------
# Variant: COMPACT  (same info, terser)
# ---------------------------------------------------------------------------


_COMPACT_TEMPLATE = """Bargaining (Deal or No Deal). You are Player {player_label}.

Pool: {pool_inline}
Your valuations (private): {values_inline}
(Each player's valuations sum to 10 over the pool. The opponent's are
hidden from you.)

Each turn, either OFFER an allocation you keep (the opponent gets the
complement) or AGREE to the opponent's last offer. Agreement ends the
game; each player scores the dot product of their valuations with what
they receive. If {max_turns} total offers (combined from both players)
pass without agreement, both score 0. Win by ending with a higher score
than the opponent.
{discount_note}
History ({num_offers}/{max_turns} offers used):
{history_str}
{accept_help}

Respond with your reasoning, then end your response with JSON:

```json
{{"action": "offer", "keep": {{"book": <int>, "hat": <int>, "basketball": <int>}}}}
```
or
```json
{{"action": "agree"}}
```

Each keep count must satisfy 0 <= keep[item] <= pool[item].
"""


def _build_compact(ctx: PromptContext, variant: PromptVariant) -> str:
    labels = variant.item_labels
    pool_inline = ", ".join(f"{labels[k]} {int(ctx.pool.get(k, 0))}" for k in _ITEM_KEYS)
    values_inline = ", ".join(
        f"{labels[k]} {int(ctx.my_values.get(k, 0))}" for k in _ITEM_KEYS
    )
    history_str = _format_history_rich(ctx.state, labels)
    accept_help = (
        _accept_help_with_preview(ctx, labels) if ctx.can_accept else _accept_help_opening()
    )
    return _COMPACT_TEMPLATE.format(
        player_label=ctx.player_id + 1,
        pool_inline=pool_inline,
        values_inline=values_inline,
        max_turns=ctx.max_turns,
        discount_note=_discount_note_baseline(ctx.discount),
        num_offers=ctx.num_offers,
        history_str=history_str,
        accept_help=accept_help,
    )


# ---------------------------------------------------------------------------
# Variant: MINIMAL  (maximally stripped)
# ---------------------------------------------------------------------------


_MINIMAL_TEMPLATE = """Bargaining. Player {player_label}.

Pool: {pool_inline}
Your private values (opponent has different, hidden values): {values_inline}
Reward = your values dotted with the items you receive. Max possible reward = 10 for each player (your values dot the pool sum to 10). If {max_turns} total offers (combined from both players) pass without agreement, both score 0.
Each turn: OFFER = what YOU keep (opponent gets the complement). AGREE = accept opponent's last offer, ending the game.

History:
{history_str}

Respond with your reasoning, then end your response with JSON, one of:
{{"action": "offer", "keep": {{"book": N, "hat": N, "basketball": N}}}}
{{"action": "agree"}}
"""


def _build_minimal(ctx: PromptContext, variant: PromptVariant) -> str:
    # Strips decoration (worked examples, rules paragraphs, accept-help
    # branch, discount note, offers-remaining counter, complement-rendered
    # history) but keeps every piece of game-mechanics info -- values are
    # private, action semantics, reward formula, tie condition. Without
    # those, performance would conflate "model didn't understand the
    # game" with "model is good at this specific prompt design".
    labels = variant.item_labels
    pool_inline = " ".join(f"{labels[k]}={int(ctx.pool.get(k, 0))}" for k in _ITEM_KEYS)
    values_inline = " ".join(
        f"{labels[k]}={int(ctx.my_values.get(k, 0))}" for k in _ITEM_KEYS
    )
    history_str = _format_history_minimal(ctx.state, labels)
    return _MINIMAL_TEMPLATE.format(
        player_label=ctx.player_id + 1,
        pool_inline=pool_inline,
        values_inline=values_inline,
        max_turns=ctx.max_turns,
        history_str=history_str,
    )


# ---------------------------------------------------------------------------
# Variant: MINIMAL_WITH_GOAL
# ---------------------------------------------------------------------------
#
# Same as MINIMAL plus the one "win condition" sentence from baseline.
# Tests the prior hypothesis that the goal sentence is load-bearing
# (models that maximize their own score in isolation play very differently
# from models that frame the task as "win the negotiation"). If this
# variant's pair-win% on equal-skill matchups shifts vs MINIMAL, the goal
# framing was doing real work even when nothing else from baseline is
# present.


_MINIMAL_WITH_GOAL_TEMPLATE = """Bargaining. Player {player_label}.

Your goal is to end the game with a higher reward than your opponent.

Pool: {pool_inline}
Your private values (opponent has different, hidden values): {values_inline}
Reward = your values dotted with the items you receive. Max possible reward = 10 for each player (your values dot the pool sum to 10). If {max_turns} total offers (combined from both players) pass without agreement, both score 0.
Each turn: OFFER = what YOU keep (opponent gets the complement). AGREE = accept opponent's last offer, ending the game.

History:
{history_str}

Respond with your reasoning, then end your response with JSON, one of:
{{"action": "offer", "keep": {{"book": N, "hat": N, "basketball": N}}}}
{{"action": "agree"}}
"""


def _build_minimal_with_goal(ctx: PromptContext, variant: PromptVariant) -> str:
    labels = variant.item_labels
    pool_inline = " ".join(f"{labels[k]}={int(ctx.pool.get(k, 0))}" for k in _ITEM_KEYS)
    values_inline = " ".join(
        f"{labels[k]}={int(ctx.my_values.get(k, 0))}" for k in _ITEM_KEYS
    )
    history_str = _format_history_minimal(ctx.state, labels)
    return _MINIMAL_WITH_GOAL_TEMPLATE.format(
        player_label=ctx.player_id + 1,
        pool_inline=pool_inline,
        values_inline=values_inline,
        max_turns=ctx.max_turns,
        history_str=history_str,
    )


# ---------------------------------------------------------------------------
# Variant: NO_ACCEPT_PREVIEW  (baseline minus the receive-preview)
# ---------------------------------------------------------------------------


def _build_no_accept_preview(ctx: PromptContext, variant: PromptVariant) -> str:
    # Same as baseline, except the can_accept branch drops the
    # "you would receive [...]" line. The model still knows accepting is
    # legal; it has to compute the resulting allocation itself.
    labels = variant.item_labels
    pool_lines = "\n".join(
        f"  {labels[k]}: {int(ctx.pool.get(k, 0))} {_unit_word(int(ctx.pool.get(k, 0)))}"
        for k in _ITEM_KEYS
    )
    my_value_lines = "\n".join(
        f"  {labels[k]}: {int(ctx.my_values.get(k, 0))}" for k in _ITEM_KEYS
    )
    history_str = _format_history_rich(ctx.state, labels)
    accept_help = _accept_help_no_preview() if ctx.can_accept else _accept_help_opening()
    return _BASELINE_TEMPLATE.format(
        pool_lines=pool_lines,
        my_value_lines=my_value_lines,
        max_turns=ctx.max_turns,
        discount_note=_discount_note_baseline(ctx.discount),
        num_offers=ctx.num_offers,
        turns_left=ctx.turns_left,
        history_str=history_str,
        player_label=ctx.player_id + 1,
        accept_help=accept_help,
    )


# ---------------------------------------------------------------------------
# Variant: GENERIC_NAMES  (Book/Hat/Basketball -> A/B/C, schema uses a/b/c)
# ---------------------------------------------------------------------------


_GENERIC_TEMPLATE = """Let's play Bargaining (a.k.a. "Deal or No Deal").

Two players negotiate over a shared pool of three item types: A, B, C.
The pool for this game contains:
{pool_lines}

Each player has a PRIVATE integer valuation per item (the opponent's values are
HIDDEN from you). Your reward at the end of the game is the dot product of your
valuation vector with the items you actually receive; the opponent's reward is
computed from their own (hidden) valuation vector. Each per-item valuation is
between 0 and 10 inclusive, and the values are sampled such that each player's
dot product with the pool equals 10 (their maximum possible reward).

Your goal is to end the game with a higher reward than your opponent.

Your private valuations (per unit):
{my_value_lines}

How a turn works:
  * On an OFFER turn, you propose an allocation that YOU would keep. The
    opponent would then receive the complement (pool minus your kept items).
    Each kept count must be an integer between 0 and the pool quantity for
    that item, inclusive.
  * If a previous opponent offer exists, you may instead ACCEPT it.
    Acceptance ENDS the game immediately: you receive the items the
    opponent offered you (= the complement of what they wanted to keep),
    and the opponent receives the items they wanted to keep. Each player
    then scores their own private dot product over what they received.
  * If {max_turns} offers go by without acceptance, the game ends with
    ZERO reward for both players (this is a tie).
{discount_note}
Offers made so far (most recent last; each shows what the OFFERING player
wanted to keep -- the other player would receive the complement):
{history_str}

Offers used so far: {num_offers} of {max_turns}. Offers remaining
(including this one if you OFFER): {turns_left}.

You are Player {player_label}. It is your turn.
{accept_help}

Respond with your reasoning, then conclude with a JSON block of EITHER form:

```json
{{"action": "offer", "keep": {{"a": <int>, "b": <int>, "c": <int>}}}}
```
or
```json
{{"action": "agree"}}
```

For example: `{{"action": "offer", "keep": {{"a": 1, "b": 0, "c": 2}}}}`

The "keep" counts describe what YOU retain -- the opponent receives the
pool minus your kept items. Each kept value must satisfy
``0 <= keep[item] <= pool[item]``.

Failure to output your final answer in the specified JSON format, or
choosing an illegal allocation, will result in a loss.
"""


def _build_generic_names(ctx: PromptContext, variant: PromptVariant) -> str:
    labels = variant.item_labels  # A / B / C
    pool_lines = "\n".join(
        f"  {labels[k]}: {int(ctx.pool.get(k, 0))} {_unit_word(int(ctx.pool.get(k, 0)))}"
        for k in _ITEM_KEYS
    )
    my_value_lines = "\n".join(
        f"  {labels[k]}: {int(ctx.my_values.get(k, 0))}" for k in _ITEM_KEYS
    )
    history_str = _format_history_rich(ctx.state, labels)
    accept_help = (
        _accept_help_with_preview(ctx, labels) if ctx.can_accept else _accept_help_opening()
    )
    return _GENERIC_TEMPLATE.format(
        pool_lines=pool_lines,
        my_value_lines=my_value_lines,
        max_turns=ctx.max_turns,
        discount_note=_discount_note_baseline(ctx.discount),
        num_offers=ctx.num_offers,
        turns_left=ctx.turns_left,
        history_str=history_str,
        player_label=ctx.player_id + 1,
        accept_help=accept_help,
    )


# ---------------------------------------------------------------------------
# Public variants
# ---------------------------------------------------------------------------


BASELINE = PromptVariant(
    name="baseline",
    item_labels=_BASELINE_LABELS,
    schema_keys=("book", "hat", "basketball"),
    aliases=_BASELINE_ALIASES,
    build_body=_build_baseline,
    rethink_illegal=_RETHINK_ILLEGAL,
    rethink_unparsable=_make_rethink_unparsable(("book", "hat", "basketball")),
)

# Null variant: a second copy of BASELINE with a different name so the
# runner schedules independent cells for it. Rendered prompts are
# byte-identical to baseline, so any observed Σ|Δrank| between null and
# baseline is pure LLM sampling noise. Use it as the noise floor in
# permutation tests on the real variants.
NULL = PromptVariant(
    name="null",
    item_labels=_BASELINE_LABELS,
    schema_keys=("book", "hat", "basketball"),
    aliases=_BASELINE_ALIASES,
    build_body=_build_baseline,
    rethink_illegal=_RETHINK_ILLEGAL,
    rethink_unparsable=_make_rethink_unparsable(("book", "hat", "basketball")),
)

COMPACT = PromptVariant(
    name="compact",
    item_labels=_BASELINE_LABELS,
    schema_keys=("book", "hat", "basketball"),
    aliases=_BASELINE_ALIASES,
    build_body=_build_compact,
    rethink_illegal=_RETHINK_ILLEGAL,
    rethink_unparsable=_make_rethink_unparsable(("book", "hat", "basketball")),
)

MINIMAL = PromptVariant(
    name="minimal",
    item_labels=_BASELINE_LABELS,
    schema_keys=("book", "hat", "basketball"),
    aliases=_BASELINE_ALIASES,
    build_body=_build_minimal,
    rethink_illegal=_RETHINK_ILLEGAL,
    rethink_unparsable=_make_rethink_unparsable(("book", "hat", "basketball")),
)

MINIMAL_WITH_GOAL = PromptVariant(
    name="minimal_with_goal",
    item_labels=_BASELINE_LABELS,
    schema_keys=("book", "hat", "basketball"),
    aliases=_BASELINE_ALIASES,
    build_body=_build_minimal_with_goal,
    rethink_illegal=_RETHINK_ILLEGAL,
    rethink_unparsable=_make_rethink_unparsable(("book", "hat", "basketball")),
)

NO_ACCEPT_PREVIEW = PromptVariant(
    name="no_accept_preview",
    item_labels=_BASELINE_LABELS,
    schema_keys=("book", "hat", "basketball"),
    aliases=_BASELINE_ALIASES,
    build_body=_build_no_accept_preview,
    rethink_illegal=_RETHINK_ILLEGAL,
    rethink_unparsable=_make_rethink_unparsable(("book", "hat", "basketball")),
)

GENERIC_NAMES = PromptVariant(
    name="generic_names",
    item_labels=_GENERIC_LABELS,
    schema_keys=("a", "b", "c"),
    aliases=_GENERIC_ALIASES,
    build_body=_build_generic_names,
    rethink_illegal=_RETHINK_ILLEGAL,
    rethink_unparsable=_make_rethink_unparsable(("a", "b", "c")),
)


VARIANTS: dict[str, PromptVariant] = {
    BASELINE.name: BASELINE,
    NULL.name: NULL,
    COMPACT.name: COMPACT,
    MINIMAL.name: MINIMAL,
    MINIMAL_WITH_GOAL.name: MINIMAL_WITH_GOAL,
    NO_ACCEPT_PREVIEW.name: NO_ACCEPT_PREVIEW,
    GENERIC_NAMES.name: GENERIC_NAMES,
}
