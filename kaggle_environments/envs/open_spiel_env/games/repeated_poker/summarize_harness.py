"""Cost-reduced LLM harness for OpenSpiel repeated_poker.

Same game and parsing pipeline as :mod:`harness`, but instead of re-sending the
full text of every previously played hand on every decision (which grows the
prompt quadratically over a 100-hand session), this harness replaces the past
hands with a compact, deterministic **opponent model**:

  1. A global HUD computed from the opponent's actions across all prior hands
     (VPIP / PFR / 3B / F3B preflop, CB / FCB on the flop, WTSD, and postflop
     aggression factor). Preflop and flop tendencies are captured directly;
     turn/river tendencies are only partially captured -- the aggression factor
     lumps flop/turn/river together and there are no street-specific turn or
     river stats, so later-street reads come from (2) below instead.
  2. Full ACPC-style renders of only the most recent hands that reached the
     turn or a showdown -- these are the ones where later-street lines and (at
     showdown) actual holdings are revealed, which the HUD cannot summarize.
     Hands decided preflop or on the flop are already covered by the HUD.

All of this is computed by the harness from the structured hand histories, so it
costs no LLM tokens to produce and stays constant-size as the session grows.

The game mechanics are identical to :mod:`harness`; the ``{readable_state_str}``
construction differs (compact opponent model instead of full hand histories),
and the response format uses the shared JSON answer structure
(```json {"move": ...}```) that the other game harnesses use rather than the
GameArena ``Final Answer:`` tag convention.
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
from kaggle_environments.envs.open_spiel_env.games.repeated_poker import (
    hand_history_utils as hh_utils,
)

# Reused from the full-history harness -- identical game mechanics.
# NOTE: we deliberately do NOT import harness._render_past_hand -- it renders
# with observer_id=None (both players' hole cards visible), which would leak the
# opponent's cards for hands that never reached showdown. We render past hands
# locally with the correct observer instead (see _render_recent_hand). We also
# do NOT reuse harness's response parsing: that harness follows the GameArena
# "Final Answer: <action>" tag convention, whereas this harness uses the shared
# JSON answer structure (```json {"move": ...}```) that the other game harnesses
# use, so it parses the move out of the JSON object instead.
from kaggle_environments.envs.open_spiel_env.games.repeated_poker.harness import (
    _config_for_state,
    _deserialize_state,
    _soft_parse_poker_action,
    get_legal_moves,  # noqa: F401  (re-exported for main.py)
)

# ---------------------------------------------------------------------------
# Tightened prompt template.
#
# The full-history harness re-sends a ~2.8k-char GTO preamble on every prompt.
# Once the history is compressed to a HUD, that fixed boilerplate dominates the
# prompt (~60%), so we trim it here. Unlike the byte-parity harness (which uses
# the GameArena "Final Answer:" tag), this harness concludes with the shared
# ```json {"move": ...}``` answer block used by the other game harnesses. The
# load-bearing pieces the soft parser assumes are preserved:
#   - the five action words fold/check/call/bet/raise,
#   - the "total for the round" bet-sizing convention.
# The GTO prose, the concept checklist, and the duplicated examples are dropped.
# ---------------------------------------------------------------------------

REPEATED_POKER = """
Let's play No-Limit Texas Hold'em. Play to maximize EV: use GTO as a baseline
and deviate to exploit the opponent's tendencies (see the HUD below). Consider
ranges, board texture, position, pot odds, and stack-to-pot ratio.

{readable_state_str}

Reason briefly (a few sentences: key beliefs and your confidence; no long
analyses), then conclude with your move as JSON:

```json
{{"move": "<action> <size-if-bet-or-raise>"}}
```

where <action> is one of: fold, check, call, bet, raise. For bet/raise, <size>
is the TOTAL chips committed that round, not the increment. E.g. facing a bet of
100, reply {{"move": "raise 200"}} to raise by 100 more; {{"move": "raise 100"}}
is invalid. Sizes are in chips, not big blinds.
{rethink_prompt}
""".strip()

# Rethink suffix for when a move WAS parsed but is not a legal action here.
RETHINK_ILLEGAL = """

You suggested move "{previous_action}" but this is not a legal action in this
spot. Choose a legal action and keep the same JSON output format:

```json
{{"move": "<action> <size-if-bet-or-raise>"}}
```
"""

# Rethink suffix for when no JSON move could be parsed from the response.
RETHINK_UNPARSABLE = """

Your previous response ended with:
{previous_response}

No JSON move could be parsed from that. Conclude your response with your move as
JSON in a ```json fenced block, exactly as the original instructions required:

```json
{{"move": "<action> <size-if-bet-or-raise>"}}
```

For example: `{{"move": "call"}}` or `{{"move": "raise 200"}}`.
"""

# Number of most-recent hands to render in full (concrete recency anchor).
_NUM_RECENT_HANDS = 2

Street = hh_utils.Street
ActionKind = hh_utils.ActionKind

# Actions taken while facing an outstanding wager this street.
_FACING_KINDS = (ActionKind.CALL, ActionKind.RAISE, ActionKind.FOLD)
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


def _accumulate_hand(stats: _OpponentStats, hand: hh_utils.Hand, opp: int) -> None:
    """Fold one parsed past hand's opponent actions into the HUD counters."""
    stats.hands += 1

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
    # Gate postflop stats on there being at least one voluntary postflop action.
    # When both players are all-in preflop the flop (and often turn/river) is
    # still dealt, but neither player had any decision on those streets -- the
    # board just runs out. Counting those hands would score the opponent as
    # "went to showdown" (they never folded) and, below, as "declined to c-bet"
    # (they never got to bet), badly skewing WTSD up and CB down.
    saw_flop = len(hand.community) >= 1 and len(hand.community[0]) == 3
    has_postflop_action = any(e.street in _POSTFLOP_STREETS for e in events)
    saw_flop_with_action = saw_flop and has_postflop_action
    went_to_showdown = not any(e.kind is ActionKind.FOLD for e in events)
    if saw_flop_with_action:
        stats.wtsd[1] += 1
        if went_to_showdown:
            stats.wtsd[0] += 1

    # --- Flop: C-Bet and Fold-to-C-Bet ---
    flop = [e for e in events if e.street is Street.FLOP]
    if saw_flop_with_action and pf_aggressor is not None:
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

    # --- Postflop aggression factor ---
    for e in events:
        if e.actor != opp or e.street not in _POSTFLOP_STREETS:
            continue
        if e.kind in _AGGRESSIVE_KINDS:
            stats.af_aggressive += 1
        elif e.kind is ActionKind.CALL:
            stats.af_calls += 1


def _pct(pair: list[int]) -> str:
    num, den = pair
    if den <= 0:
        return "n/a"
    return f"{round(100 * num / den)}%"


def _af(stats: _OpponentStats) -> str:
    frac = f"({stats.af_aggressive}(b+r)/{stats.af_calls}c)"
    if stats.af_calls == 0:
        return f"{'inf' if stats.af_aggressive > 0 else 'n/a'} {frac}"
    return f"{stats.af_aggressive / stats.af_calls:.1f} {frac}"


def _reached_turn_or_showdown(hand: hh_utils.Hand) -> bool:
    """True if the hand went past the flop with real postflop play -- i.e. a
    turn card was dealt (so there was turn/river action) or it ended in a
    showdown. These are the hands where full replay adds information the HUD
    can't capture; hands decided preflop or on the flop are already summarized
    by the stats.

    Requires at least one voluntary postflop action so all-in-preflop runouts
    (where the board is dealt out with no further decisions) don't waste a
    recency slot -- those carry no postflop line to learn from."""
    # community is [flop(3), turn(1), river(1)] with later streets possibly empty.
    turn_dealt = len(hand.community) >= 2 and len(hand.community[1]) >= 1
    showdown = not any(e.kind is ActionKind.FOLD for e in hand.events)
    saw_flop = len(hand.community) >= 1 and len(hand.community[0]) == 3
    has_postflop_action = any(e.street in _POSTFLOP_STREETS for e in hand.events)
    return has_postflop_action and (turn_dealt or (showdown and saw_flop))


def _reached_showdown(hand: hh_utils.Hand) -> bool:
    """True if the hand went to showdown (nobody folded). Only then were both
    players' hole cards revealed at the table -- so only then may we render the
    opponent's cards. Hands that ended in a fold keep the folder's cards
    hidden, exactly as they were during play."""
    return not any(e.kind is ActionKind.FOLD for e in hand.events)


def _button_index_for_hand(hand_index: int, cur_hand: int, cur_dealer: int, rotate: bool) -> int:
    """Return the 0-based seat holding the button for a past hand.

    ``cur_dealer`` is the dealer of the current (in-progress) hand ``cur_hand``,
    read from the live state. With ``rotate_dealer`` the button alternates each
    hand, so a hand played ``cur_hand - hand_index`` hands ago sat on the other
    button iff that gap is odd; without rotation the dealer never moves. This is
    correct for both ``rotate_dealer`` settings, unlike assuming alternation.
    """
    if not rotate:
        return cur_dealer
    return cur_dealer ^ ((cur_hand - hand_index) % 2)


def _render_recent_hand(hand: hh_utils.Hand, cur: int) -> str:
    """Render a completed past hand for the current player. The opponent's hole
    cards are shown only if the hand reached showdown (they were revealed at the
    table); otherwise they stay masked, exactly as during play."""
    observer_id = None if _reached_showdown(hand) else f"Player{cur}"
    return hh_utils.render_pokersite(hand=hand, observer_id=observer_id, sitename="")


def _render_opponent_model(stats: _OpponentStats, opp: int) -> str:
    """A HUD line block, mirroring an online-poker tracker overlay."""
    lines = [
        f"=== HUD for Player{opp} (opponent) -- {stats.hands} hands ===",
        f"VPIP {_pct(stats.vpip)} ({stats.vpip[0]}/{stats.vpip[1]})  |  "
        f"PFR {_pct(stats.pfr)} ({stats.pfr[0]}/{stats.pfr[1]})  |  "
        f"3B {_pct(stats.threebet)} ({stats.threebet[0]}/{stats.threebet[1]})  |  "
        f"F3B {_pct(stats.fold_to_3bet)} ({stats.fold_to_3bet[0]}/{stats.fold_to_3bet[1]})",
        f"CB {_pct(stats.cbet)} ({stats.cbet[0]}/{stats.cbet[1]})  |  "
        f"FCB {_pct(stats.fold_to_cbet)} ({stats.fold_to_cbet[0]}/{stats.fold_to_cbet[1]})  |  "
        f"WTSD {_pct(stats.wtsd)} ({stats.wtsd[0]}/{stats.wtsd[1]})  |  "
        f"AF {_af(stats)}",
        "Legend (percentages shown as pct (made/opportunities)): "
        "VPIP=Voluntary Put In Pot, PFR=Pre-Flop Raise, 3B=3-Bet, "
        "F3B=Fold to 3-Bet, CB=flop Continuation Bet, FCB=Fold to Continuation "
        "Bet, WTSD=Went To Showdown (of hands that saw the flop), "
        "AF=Aggression Factor = postflop (bets+raises)/call. "
        "Small samples are noisy -- weight by opportunities.",
    ]
    return "\n".join(lines)


def _render_standing(state_dict: dict, cur: int) -> str:
    """Render the model's current standing on the scored metric.

    The episode is scored on cumulative chip profit summed across all hands
    (zero-sum in heads-up). Report it in chips and big blinds, alongside how
    many hands remain, so the model has its full standing.
    """
    hand_returns = state_dict.get("hand_returns", [])
    cur_net = int(sum(r[cur] for r in hand_returns if len(r) > cur))
    hand_number = state_dict["hand_number"]
    max_num_hands = state_dict["max_num_hands"]
    big_blind = state_dict.get("big_blind", 0) or 0
    if cur_net > 0:
        standing = f"AHEAD by {cur_net} chips"
    elif cur_net < 0:
        standing = f"BEHIND by {-cur_net} chips"
    else:
        standing = "EVEN"
    if big_blind:
        standing += f" ({cur_net / big_blind:+.1f} BB)"
    return f"=== Standing (scored on cumulative chip profit) ===\n{standing}, hand {hand_number + 1}/{max_num_hands}."


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

    # Button assignment. ``dealer`` is the current hand's button seat; whether it
    # rotates each hand comes from the game params. Both are needed to place the
    # button correctly for every past hand (assuming plain alternation is wrong
    # when rotate_dealer=False, which corrupts positional stats).
    cur_hand_number = state_dict["hand_number"]
    cur_dealer = state_dict["dealer"]
    rotate = bool(pyspiel_state.get_game().get_parameters().get("rotate_dealer"))

    # Parse the current (in-progress) hand.
    players = [f"Player{i}" for i in range(pyspiel_state.num_players())]
    up_state_dict = json.loads(state_dict["current_universal_poker_json"])
    acpc_state_str = up_state_dict["acpc_state"].split("\n")[0]
    if not acpc_state_str.startswith("STATE:"):
        raise ValueError(f"Expected ACPC state to start with STATE:, got {acpc_state_str}")
    acpc_state_str_full = acpc_state_str + "::" + "|".join(players)
    cur_hand, _ = hh_utils.parse_acpc_line(
        acpc_state_str_full,
        cfg=cfg,
        policy=hh_utils.ButtonPolicy(),
        button_index=cur_dealer,
        hand_id_override=str(cur_hand_number),
    )

    # Accumulate the opponent model over all completed hands. Preflop and flop
    # tendencies are captured by the HUD; only hands that reached the turn or a
    # showdown are worth replaying in full (later-street lines and revealed
    # holdings can't be summarized statistically).
    stats = _OpponentStats()
    acpc_hhs = list(pyspiel_state.acpc_hand_histories())
    parsed_hands: list[hh_utils.Hand] = []
    deep_hand_indices: list[int] = []
    for i, acpc_hh in enumerate(acpc_hhs):
        button_index = _button_index_for_hand(i, cur_hand_number, cur_dealer, rotate)
        hand, _ = hh_utils.parse_acpc_line(acpc_hh, cfg=cfg, policy=hh_utils.ButtonPolicy(), button_index=button_index)
        parsed_hands.append(hand)
        _accumulate_hand(stats, hand, opp)
        if _reached_turn_or_showdown(hand):
            deep_hand_indices.append(i)

    if len(acpc_hhs) != cur_hand_number:
        raise ValueError(
            f"Number of past hands {len(acpc_hhs)} does not match number of"
            f" hands in state (current hand={cur_hand_number})."
        )

    sections: list[str] = [f"You are Player{cur}."]
    sections.append(_render_standing(state_dict, cur))

    if stats.hands > 0:
        sections.append(_render_opponent_model(stats, opp))

        recent_idx = deep_hand_indices[-_NUM_RECENT_HANDS:]
        if recent_idx:
            rendered_recent = [_render_recent_hand(parsed_hands[i], cur) for i in recent_idx]
            sections.append(
                f"=== Most recent {len(recent_idx)} hand(s) that reached the turn "
                "or later, in full ===\n\n" + "\n\n".join(rendered_recent)
            )
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
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt from a pre-deserialized pyspiel state, using the
    compact opponent-model rendering."""
    readable_state_str = _render_readable_state(state)
    rethink_prompt = render_rethink_suffix(
        RETHINK_ILLEGAL,
        RETHINK_UNPARSABLE,
        previous_response,
        previous_action,
    )
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
    del move_history  # not used in repeated_poker prompts
    state = _deserialize_state(observation)
    if state is None:
        raise ValueError("Observation is missing serializedGameAndState.")
    return generate_prompt_from_state(
        state,
        previous_response=previous_response,
        previous_action=previous_action,
    )


def parse_response_with_state(
    response: str,
    legal_action_strings: Sequence[str],
    state: pyspiel.State,
) -> ParseResult:
    """Parse with a pre-deserialized state. Same as ``parse_response`` but
    skips deserialization -- exposed for the verify script.

    Two-stage pipeline: pull the ``"move"`` string out of the last JSON object
    in the response (the shared answer structure), then soft-match it against
    the legal moves with the stateful poker parser, which resolves the
    "total for the round" bet-sizing convention against the live state.
    """
    data = extract_last_json_object(response, required_keys=("move",))
    if data is None:
        return ParseResult(legal_action=None, raw_action=None)
    raw_value = data.get("move")
    if raw_value is None:
        return ParseResult(legal_action=None, raw_action=None)
    raw = str(raw_value).strip()
    if not raw:
        return ParseResult(legal_action=None, raw_action=None)
    player_number = state.current_player()
    matched = _soft_parse_poker_action(raw, legal_action_strings, state, player_number)
    if matched is not None and matched in legal_action_strings:
        return ParseResult(legal_action=matched, raw_action=raw)
    return ParseResult(legal_action=None, raw_action=raw)


def parse_response(
    response: str,
    legal_action_strings: Sequence[str],
    *,
    observation: Mapping[str, Any] | None = None,
) -> ParseResult:
    """Extract a legal poker action from the model response.

    Uses the shared JSON answer structure (```json {"move": ...}```) that the
    other game harnesses use, then soft-matches the extracted move against the
    legal actions using the live pyspiel state (needed for bet-size math).
    """
    data = extract_last_json_object(response, required_keys=("move",))
    if data is None:
        return ParseResult(legal_action=None, raw_action=None)
    raw_value = data.get("move")
    raw = None if raw_value is None else str(raw_value).strip()
    if not raw:
        return ParseResult(legal_action=None, raw_action=None)
    if observation is None:
        # Without state context we can only return what we extracted; the
        # framework will treat this as an illegal-move retry.
        return ParseResult(legal_action=None, raw_action=raw)
    state = _deserialize_state(observation)
    if state is None:
        return ParseResult(legal_action=None, raw_action=raw)
    return parse_response_with_state(response, legal_action_strings, state)
