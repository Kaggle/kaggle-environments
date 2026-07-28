"""Cost-reduced LLM harness for OpenSpiel repeated_poker.

Same game and parsing pipeline as :mod:`harness`, but instead of re-sending the
full text of every previously played hand on every decision (which grows the
prompt quadratically over a 100-hand session), this harness replaces the past
hands with a compact, deterministic **opponent model**:

  1. A global HUD computed from the opponent's actions across all prior hands
     (VPIP / PFR / 3bet preflop, postflop aggression, showdown frequency, net).
  2. A **context-aware** section that summarizes the opponent's tendencies on
     the *current street* -- split by whether they were the aggressor or facing
     a bet -- so the model gets the read that is most relevant to the decision
     actually in front of it.
  3. Full ACPC-style renders of only the last few hands, for concrete recency.

All of this is computed by the harness from the structured hand histories, so it
costs no LLM tokens to produce and stays constant-size as the session grows.

The prompt template, legal-move extraction, and response parsing are imported
verbatim from :mod:`harness` -- the game mechanics are identical, only the
``{readable_state_str}`` construction differs.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import pyspiel

from kaggle_environments.envs.open_spiel_env.games.repeated_poker import (
    hand_history_utils as hh_utils,
)

# Reused verbatim from the full-history harness -- identical game mechanics.
from kaggle_environments.envs.open_spiel_env.games.repeated_poker.harness import (  # noqa: F401
    POKER_RETHINK,
    _config_for_state,
    _deserialize_state,
    _extract_move_from_response,
    _render_past_hand,
    _soft_parse_poker_action,
    get_legal_moves,
    parse_response,
    parse_response_with_state,
)

# ---------------------------------------------------------------------------
# Tightened prompt template.
#
# The full-history harness re-sends a ~2.8k-char GTO preamble on every prompt.
# Once the history is compressed to a HUD, that fixed boilerplate dominates the
# prompt (~60%), so we trim it here. The load-bearing pieces are preserved so
# the response parser is unaffected:
#   - the literal "Final Answer:" tag (RuleBasedMoveParser splits on it),
#   - the five action words fold/check/call/bet/raise,
#   - the "total for the round" bet-sizing convention (soft parser assumes it).
# The GTO prose, the concept checklist, and the duplicated examples are dropped.
# ---------------------------------------------------------------------------

REPEATED_POKER = """
You are a world-class Heads-Up No-Limit Texas Hold'em poker AI. Play to maximize
EV: use GTO as a baseline and deviate to exploit the opponent's tendencies (see
the HUD and per-street reads below). Consider ranges, board texture, position,
pot odds, and stack-to-pot ratio.

Keep your reasoning short: a few sentences at most stating your key beliefs and
confidence. Do not write long analyses. Then end with the final answer.
The final answer MUST be the last line, in exactly this format:
Final Answer: <action> <size-if-bet-or-raise>
where <action> is one of: fold, check, call, bet, raise. No other text or
punctuation on that line (not "**Final Answer:** call", not "final answer - bet").
Valid: "Final Answer: fold" / "Final Answer: check" / "Final Answer: call" /
"Final Answer: bet 100" / "Final Answer: raise 100".

For bet/raise, <size> is the TOTAL chips committed that round, not the increment.
E.g. facing a bet of 100, reply "raise 200" to raise by 100 more; "raise 100" is
invalid. Sizes are in chips, not big blinds.

Hand to analyze:

{readable_state_str}

Action is on you. Format your response correctly.

{rethink_prompt}
""".strip()

# Number of most-recent hands to render in full (concrete recency anchor).
_NUM_RECENT_HANDS = 2

Street = hh_utils.Street
ActionKind = hh_utils.ActionKind

# Actions taken while facing an outstanding wager this street.
_FACING_KINDS = (ActionKind.CALL, ActionKind.RAISE, ActionKind.FOLD)
# Actions taken with no wager to call (first-in or checked-to).
_UNFACED_KINDS = (ActionKind.BET, ActionKind.CHECK)
_AGGRESSIVE_KINDS = (ActionKind.BET, ActionKind.RAISE)
_POSTFLOP_STREETS = (Street.FLOP, Street.TURN, Street.RIVER)


# ---------------------------------------------------------------------------
# Opponent model -- an online-poker HUD computed deterministically from the
# structured hand histories. All stats are standard tracker definitions:
#   VPIP           voluntarily put chips in pot preflop
#   PFR            raised preflop
#   3Bet           reraised facing an open raise / times facing an open
#   Fold-to-3Bet   folded as the opener facing a 3bet / times opener faced a 3bet
#   CBet           bet flop as preflop aggressor / times saw flop as aggressor
#   Fold-to-CBet   folded to a flop c-bet / times faced a flop c-bet
#   WTSD           went to showdown / times saw the flop
#   AF             (postflop bets + raises) / postflop calls
# ---------------------------------------------------------------------------


class _OpponentStats:
    """Deterministic HUD accumulated from an opponent's actions across hands."""

    def __init__(self) -> None:
        self.hands = 0
        self.net = 0
        # Each stat is a [numerator, denominator] pair.
        self.vpip = [0, 0]
        self.pfr = [0, 0]
        self.threebet = [0, 0]
        self.fold_to_3bet = [0, 0]
        self.cbet = [0, 0]
        self.fold_to_cbet = [0, 0]
        self.wtsd = [0, 0]
        # Aggression factor components (postflop).
        self.af_aggressive = 0
        self.af_calls = 0
        # Per-street facing/unfaced distribution for the context-aware block.
        self.by_street: dict[Street, dict[str, int]] = {}

    def _street(self, street: Street) -> dict[str, int]:
        return self.by_street.setdefault(
            street,
            {
                "faced": 0,
                "fold": 0,
                "call": 0,
                "raise": 0,
                "unfaced": 0,
                "bet": 0,
                "check": 0,
            },
        )


def _accumulate_hand(stats: _OpponentStats, hand: hh_utils.Hand, opp: int) -> None:
    """Fold one parsed past hand's opponent actions into the HUD counters."""
    stats.hands += 1
    if opp < len(hand.profits):
        stats.net += hand.profits[opp]

    events = hand.events
    preflop = [e for e in events if e.street is Street.PREFLOP]

    # --- Preflop: VPIP / PFR ---
    stats.vpip[1] += 1
    stats.pfr[1] += 1
    opp_pf = [e for e in preflop if e.actor == opp]
    if any(e.kind in (ActionKind.CALL, *_AGGRESSIVE_KINDS) for e in opp_pf):
        stats.vpip[0] += 1
    if any(e.kind in _AGGRESSIVE_KINDS for e in opp_pf):
        stats.pfr[0] += 1

    # --- Preflop raise walk: 3Bet and Fold-to-3Bet ---
    num_raises = 0
    last_raiser: int | None = None
    opp_opened = False
    for e in preflop:
        if e.actor == opp:
            # Facing exactly one (villain) open raise -> a 3bet opportunity.
            if num_raises == 1 and last_raiser != opp:
                stats.threebet[1] += 1
                if e.kind in _AGGRESSIVE_KINDS:
                    stats.threebet[0] += 1
            # Opener facing a reraise -> a fold-to-3bet opportunity.
            elif opp_opened and num_raises >= 2 and last_raiser != opp:
                stats.fold_to_3bet[1] += 1
                if e.kind is ActionKind.FOLD:
                    stats.fold_to_3bet[0] += 1
        if e.kind in _AGGRESSIVE_KINDS:
            if e.actor == opp and num_raises == 0:
                opp_opened = True
            num_raises += 1
            last_raiser = e.actor
    pf_aggressor = last_raiser  # last preflop raiser, or None in a limped pot

    # --- Saw flop / showdown / WTSD ---
    saw_flop = len(hand.community) >= 1 and len(hand.community[0]) == 3
    went_to_showdown = not any(e.kind is ActionKind.FOLD for e in events)
    if saw_flop:
        stats.wtsd[1] += 1
        if went_to_showdown:
            stats.wtsd[0] += 1

    # --- Flop: C-Bet and Fold-to-C-Bet ---
    flop = [e for e in events if e.street is Street.FLOP]
    if saw_flop and pf_aggressor is not None:
        if pf_aggressor == opp:
            stats.cbet[1] += 1
            if any(e.actor == opp and e.kind is ActionKind.BET for e in flop):
                stats.cbet[0] += 1
        else:
            aggressor_cbet = any(e.actor == pf_aggressor and e.kind is ActionKind.BET for e in flop)
            if aggressor_cbet:
                stats.fold_to_cbet[1] += 1
                response = next(
                    (e for e in flop if e.actor == opp and e.kind in _FACING_KINDS),
                    None,
                )
                if response is not None and response.kind is ActionKind.FOLD:
                    stats.fold_to_cbet[0] += 1

    # --- Postflop aggression factor + per-street distribution ---
    for e in events:
        if e.actor != opp:
            continue
        if e.street in _POSTFLOP_STREETS:
            if e.kind in _AGGRESSIVE_KINDS:
                stats.af_aggressive += 1
            elif e.kind is ActionKind.CALL:
                stats.af_calls += 1
        bucket = stats._street(e.street)
        if e.kind in _FACING_KINDS:
            bucket["faced"] += 1
            if e.kind is ActionKind.FOLD:
                bucket["fold"] += 1
            elif e.kind is ActionKind.CALL:
                bucket["call"] += 1
            elif e.kind is ActionKind.RAISE:
                bucket["raise"] += 1
        elif e.kind in _UNFACED_KINDS:
            bucket["unfaced"] += 1
            if e.kind is ActionKind.BET:
                bucket["bet"] += 1
            else:
                bucket["check"] += 1


def _pct(pair: list[int]) -> str:
    num, den = pair
    if den <= 0:
        return "n/a"
    return f"{round(100 * num / den)}%"


def _af(stats: _OpponentStats) -> str:
    if stats.af_calls == 0:
        return "inf" if stats.af_aggressive > 0 else "n/a"
    return f"{stats.af_aggressive / stats.af_calls:.1f}"


def _render_opponent_model(stats: _OpponentStats, opp: int) -> str:
    """A HUD line block, mirroring an online-poker tracker overlay."""
    sign = "+" if stats.net >= 0 else ""
    lines = [
        f"=== HUD for Player{opp} (opponent) -- {stats.hands} hands, net {sign}{stats.net} chips vs you ===",
        f"VPIP {_pct(stats.vpip)} ({stats.vpip[0]}/{stats.vpip[1]})  |  "
        f"PFR {_pct(stats.pfr)} ({stats.pfr[0]}/{stats.pfr[1]})  |  "
        f"3Bet {_pct(stats.threebet)} ({stats.threebet[0]}/{stats.threebet[1]})  |  "
        f"Fold-to-3Bet {_pct(stats.fold_to_3bet)} "
        f"({stats.fold_to_3bet[0]}/{stats.fold_to_3bet[1]})",
        f"CBet {_pct(stats.cbet)} ({stats.cbet[0]}/{stats.cbet[1]})  |  "
        f"Fold-to-CBet {_pct(stats.fold_to_cbet)} "
        f"({stats.fold_to_cbet[0]}/{stats.fold_to_cbet[1]})  |  "
        f"WTSD {_pct(stats.wtsd)} ({stats.wtsd[0]}/{stats.wtsd[1]})  |  "
        f"AF {_af(stats)}",
        "Legend: pct (made/opportunities). VPIP=voluntary play%, PFR=raise% "
        "preflop; 3Bet/Fold-to-3Bet=preflop reraise made/faced; "
        "CBet/Fold-to-CBet=flop continuation-bet made/faced; WTSD=showdown% when "
        "saw flop; AF=postflop aggression (bets+raises per call). Small samples "
        "are noisy -- weight by opportunities.",
    ]
    return "\n".join(lines)


def _render_street_tendencies(stats: _OpponentStats, street: Street, opp: int) -> str:
    """Context-aware block: opponent's behavior on the *current* street."""
    bucket = stats.by_street.get(street)
    label = street.name.capitalize()
    if not bucket or (bucket["faced"] == 0 and bucket["unfaced"] == 0):
        return (
            f"=== Opponent tendencies on the {label} ===\n"
            f"No prior {label.lower()} actions observed for Player{opp} yet."
        )
    lines = [f"=== Opponent tendencies on the {label} (Player{opp}) ==="]
    if bucket["unfaced"] > 0:
        lines.append(
            f"First to act / checked to ({bucket['unfaced']}x): "
            f"bet {_pct([bucket['bet'], bucket['unfaced']])}, "
            f"check {_pct([bucket['check'], bucket['unfaced']])}."
        )
    if bucket["faced"] > 0:
        lines.append(
            f"Facing a bet/raise ({bucket['faced']}x): "
            f"fold {_pct([bucket['fold'], bucket['faced']])}, "
            f"call {_pct([bucket['call'], bucket['faced']])}, "
            f"raise {_pct([bucket['raise'], bucket['faced']])}."
        )
    return "\n".join(lines)


def _render_standing(state_dict: dict, cur: int) -> str:
    """Render the model's current standing on the scored metric.

    The episode is scored on cumulative chip profit summed across all hands
    (zero-sum in heads-up). A model's optimal risk tolerance depends on whether
    it is ahead or behind and how many hands remain, so we surface both.
    """
    hand_returns = state_dict.get("hand_returns", [])
    cur_net = int(sum(r[cur] for r in hand_returns if len(r) > cur))
    hand_number = state_dict["hand_number"]
    max_num_hands = state_dict["max_num_hands"]
    hands_left = max_num_hands - hand_number
    if cur_net > 0:
        standing = f"AHEAD by {cur_net} chips"
    elif cur_net < 0:
        standing = f"BEHIND by {-cur_net} chips"
    else:
        standing = "EVEN"
    return (
        f"=== Standing (scored on cumulative chip profit) ===\n"
        f"{standing}, hand {hand_number + 1}/{max_num_hands} ({hands_left} left). "
        "Adjust risk accordingly."
    )


# ---------------------------------------------------------------------------
# Readable-state builder (replaces harness._render_readable_state)
# ---------------------------------------------------------------------------


def _render_readable_state(pyspiel_state: pyspiel.State) -> str:
    """Build the ``{readable_state_str}`` with a compact opponent model instead
    of the full transcript of every prior hand."""
    state_dict = json.loads(str(pyspiel_state))
    cfg = _config_for_state(pyspiel_state)
    cur = pyspiel_state.current_player()
    opp = 1 - cur

    # Parse the current (in-progress) hand to learn the current street.
    players = [f"Player{i}" for i in range(pyspiel_state.num_players())]
    up_state_dict = json.loads(state_dict["current_universal_poker_json"])
    acpc_state_str = up_state_dict["acpc_state"].split("\n")[0]
    if not acpc_state_str.startswith("STATE:"):
        raise ValueError(f"Expected ACPC state to start with STATE:, got {acpc_state_str}")
    acpc_state_str_full = acpc_state_str + "::" + "|".join(players)
    cur_hand, cur_parse_state = hh_utils.parse_acpc_line(
        acpc_state_str_full,
        cfg=cfg,
        policy=hh_utils.ButtonPolicy(),
        button_index=(state_dict["hand_number"] % 2) + 1,
        hand_id_override=str(state_dict["hand_number"]),
    )
    current_street = cur_parse_state.street

    # Accumulate the opponent model over all completed hands, and keep the
    # parsed objects for the last few so we can render them in full.
    stats = _OpponentStats()
    recent_hands: list[hh_utils.Hand] = []
    acpc_hhs = list(pyspiel_state.acpc_hand_histories())
    for i, acpc_hh in enumerate(acpc_hhs):
        button_index = (i % 2) + 1
        hand, _ = hh_utils.parse_acpc_line(acpc_hh, cfg=cfg, policy=hh_utils.ButtonPolicy(), button_index=button_index)
        _accumulate_hand(stats, hand, opp)
        recent_hands.append(hand)

    if len(acpc_hhs) != state_dict["hand_number"]:
        raise ValueError(
            f"Number of past hands {len(acpc_hhs)} does not match number of"
            f" hands in state (current hand={state_dict['hand_number']})."
        )

    sections: list[str] = [f"You are Player{cur}."]
    sections.append(_render_standing(state_dict, cur))

    if stats.hands > 0:
        sections.append(_render_opponent_model(stats, opp))
        sections.append(_render_street_tendencies(stats, current_street, opp))

        recent = acpc_hhs[-_NUM_RECENT_HANDS:]
        start = len(acpc_hhs) - len(recent)
        rendered_recent = [_render_past_hand(acpc_hh, ((start + j) % 2) + 1, cfg) for j, acpc_hh in enumerate(recent)]
        sections.append(f"=== Most recent {len(recent)} hand(s) in full ===\n\n" + "\n\n".join(rendered_recent))
    else:
        sections.append("This is the first hand of the session; no history yet.")

    observer_id = f"Player{cur}"
    current_hand_str = hh_utils.render_pokersite(hand=cur_hand, observer_id=observer_id, sitename="")
    sections.append("=== Current hand ===\n\n" + current_hand_str)

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Public functions (called by core_harness) -- mirror harness.py signatures.
# ---------------------------------------------------------------------------


def generate_prompt_from_state(
    state: pyspiel.State,
    previous_response: str | None = None,
) -> str:
    """Build the LLM prompt from a pre-deserialized pyspiel state, using the
    compact opponent-model rendering."""
    readable_state_str = _render_readable_state(state)

    if previous_response is None:
        rethink_prompt = ""
    else:
        if not previous_response:
            generation = "NO RESPONSE RECEIVED"
        else:
            generation = "\n".join(previous_response.split("\n")[-5:])
        rethink_prompt = POKER_RETHINK.format(generation=generation)

    return REPEATED_POKER.format(
        readable_state_str=readable_state_str,
        rethink_prompt=rethink_prompt,
    )


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt with the compact opponent model."""
    del move_history, previous_action  # not used in repeated_poker prompts
    state = _deserialize_state(observation)
    if state is None:
        raise ValueError("Observation is missing serializedGameAndState.")
    return generate_prompt_from_state(state, previous_response=previous_response)
