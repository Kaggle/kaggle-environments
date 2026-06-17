"""LLM harness for OpenSpiel repeated_poker (Heads-Up No-Limit Texas Hold'em).

Migrated from Google DeepMind's GameArena poker harness. The prompts produced
here are intended to be byte-identical to those produced by
``game_arena.harness.games.poker.poker_agents.RepeatedPokerRethinkAgent`` with
``REPEATED_POKER`` template + ``RETHINK_REPEATED_POKER`` strategy.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult
from kaggle_environments.envs.open_spiel_env.games.repeated_poker import (
    hand_history_utils as hh_utils,
)

# ---------------------------------------------------------------------------
# Prompt templates -- verbatim ports of GameArena's REPEATED_POKER / POKER_RETHINK.
# Whitespace, punctuation, and the trailing ``.strip()`` are preserved exactly.
# ---------------------------------------------------------------------------

REPEATED_POKER = """
You are a world-class Heads-Up No-Limit Texas Hold'em (HU NLHE) poker AI. Your task is to analyze the given hand and determine the optimal play based on the following framework.

I. Guiding Principles

A. Primary Objective: Maximize Expected Value (EV).
Every decision must be justified by its overall profitability.

B. Core Strategy: GTO-Informed Exploitation.
Use Game Theory Optimal (GTO) play as your baseline, but deviate to maximally exploit your opponent's specific tendencies and mistakes.

II. Decision-Making Framework

A. Your analysis should:
- Clearly outline the current situation.
- Consider key concepts in modern poker theory.

B. Key concepts to consider:
- Opponent Profile: What are their stats and tendencies (e.g., aggressive, passive, sticky, tight)?
- Ranges: Who has the range and nut advantage on this board? What is my opponent's likely hand range given the action and our positions? What is my range in the current situation?
- Board Texture: Is the board static/dynamic, wet/dry? How does it interact with the ranges?
- Pot Dynamics: What is the Stack-to-Pot Ratio (SPR)? What are the pot odds and implied odds?
- Line & Sizing: What is my overall plan for the hand? Justify chosen bet sizes.
- Game Rules: E.g., what are the minimum bets allowed in a given situation?
- Anything else that you think is particularly relevant.

B. Mandatory information to include in your response:
Beliefs: Explicitly state any key beliefs that factor into your decision.
Confidence: Rate your certainty in these beliefs and in your chosen action (e.g. high, medium, low).
Final Answer: <action> <size-if-bet-or-raise>

III. Required Final Answer Format

All responses MUST start with your reasoning and conclude with the final answer.
The final answer MUST be on a single, final, new line.
The final answer line MUST be in the precise format:
Final Answer: <action> <size-if-bet-or-raise>
Where <action> is one of the following: fold, check, call, bet, raise
DO NOT include extra punctuation or text on the same line as the final answer.

Example invalid final answer lines:
final answer - bet 100  (missing capitalization and colon)
**Final Answer:** call  (extraneous punctuation)

Example valid final answer lines:
Final Answer: fold
Final Answer: check
Final Answer: call
Final Answer: bet 100
Final Answer: raise 100

If betting or raising, the size MUST refer to the total bet amount for the action that round.
- E.g. if your opponent has bet 100, you must respond "raise 200" to indicate a raise of 100 additional chips; "raise 100" is not a valid response.
- The amount MUST be in chips, not big blinds.

Below is the hand history you are tasked with analyzing.

{readable_state_str}

Action is on you. Remember to format your response correctly. Good luck!

{rethink_prompt}
""".strip()

POKER_RETHINK = """
A legal action could not be parsed from your previous response.
Think carefully and respond with a legal action.
Remember to include the final answer on the final line of your response.
It must EXACTLY follow the specified final answer format:
Final Answer: <action> <size-if-bet-or-raise>

Your previous response concluded with:
{generation}
""".strip()


# ---------------------------------------------------------------------------
# Helpers shared by the prompt builder and the verify script
# ---------------------------------------------------------------------------


def _config_for_state(pyspiel_state: pyspiel.State) -> hh_utils.Config:
    """Build the hh_utils.Config the upstream agent hard-coded."""
    # TODO: extract small_blind / big_blind / starting_stacks from the state's
    # universal_poker game params. Upstream has the same TODO and hard-codes
    # these for HU NLHE 1/2 with 200/200 stacks; preserving for prompt parity.
    return hh_utils.Config(
        seats=pyspiel_state.num_players(),
        small_blind=1,
        big_blind=2,
        starting_stacks=[200] * pyspiel_state.num_players(),
    )


# Cache for past-hand renders. Each hand history is deterministic for a given
# (acpc_hh, button_index, cfg) tuple, and past hands never change once the hand
# ends -- so caching avoids the O(n^2) cost of re-rendering all prior hands at
# every new prompt across a long session.
_PAST_HAND_RENDER_CACHE: dict[tuple[str, int, int, int, int, int], str] = {}


def _render_past_hand(acpc_hh: str, button_index: int, cfg: hh_utils.Config) -> str:
    key = (
        acpc_hh,
        button_index,
        cfg.seats,
        cfg.small_blind,
        cfg.big_blind,
        tuple(cfg.starting_stacks).__hash__(),
    )
    cached = _PAST_HAND_RENDER_CACHE.get(key)
    if cached is not None:
        return cached
    hh, _ = hh_utils.parse_acpc_line(
        acpc_hh,
        cfg=cfg,
        policy=hh_utils.ButtonPolicy(),
        button_index=button_index,
    )
    rendered = hh_utils.render_pokersite(hand=hh, observer_id=None, sitename="")
    _PAST_HAND_RENDER_CACHE[key] = rendered
    return rendered


def _render_readable_state(pyspiel_state: pyspiel.State) -> str:
    """Build the multi-hand session view that goes into ``{readable_state_str}``.

    Replicates ``RepeatedPokerRethinkAgent._get_prompt_substitutions``.
    """
    state_dict = json.loads(str(pyspiel_state))
    cfg = _config_for_state(pyspiel_state)

    past_hhs: list[str] = []
    for i, acpc_hh in enumerate(pyspiel_state.acpc_hand_histories()):
        past_hhs.append(_render_past_hand(acpc_hh, (i % 2) + 1, cfg))
    if len(past_hhs) != state_dict["hand_number"]:
        raise ValueError(
            f"Number of past hands {len(past_hhs)} does not match number of"
            f" hands in state (current hand={state_dict['hand_number']})."
        )
    past_hhs_str = "\n\n".join(past_hhs)

    players = [f"Player{i}" for i in range(pyspiel_state.num_players())]
    up_state_dict = json.loads(state_dict["current_universal_poker_json"])
    acpc_state_str = up_state_dict["acpc_state"].split("\n")[0]
    if not acpc_state_str.startswith("STATE:"):
        raise ValueError(f"Expected ACPC state to start with STATE:, got {acpc_state_str}")
    # Pluribus-style player suffix.
    acpc_state_str = acpc_state_str + "::" + "|".join(players)

    hh, _ = hh_utils.parse_acpc_line(
        acpc_state_str,
        cfg=cfg,
        policy=hh_utils.ButtonPolicy(),
        button_index=(state_dict["hand_number"] % 2) + 1,
        hand_id_override=str(state_dict["hand_number"]),
    )
    observer_id = f"Player{pyspiel_state.current_player()}"
    current_hand_str = hh_utils.render_pokersite(hand=hh, observer_id=observer_id, sitename="")
    current_hand_str = f"You are Player{pyspiel_state.current_player()}.\n\n{current_hand_str}"
    return (
        f"You are Player{pyspiel_state.current_player()}.\n\n"
        + "Previously played hands this session:\n\n"
        + past_hhs_str
        + "\n\n"
        + "Current hand:\n\n"
        + current_hand_str
    )


def _deserialize_state(observation: Mapping[str, Any]) -> pyspiel.State | None:
    serialized = observation.get("serializedGameAndState", "")
    if not serialized:
        return None
    _, state = pyspiel.deserialize_game_and_state(serialized)
    return state


# ---------------------------------------------------------------------------
# RuleBasedMoveParser port -- find "Final Answer:" tag, extract suffix.
# Upstream uses additional_tags=[] so only "Final Answer:" splits.
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<.*?>")


def _extract_move_from_response(response: str, action_tag: str = "Final Answer:") -> str | None:
    """Faithful port of ``parse_move_from_response`` from upstream parsers.py."""
    if response is None:
        return None
    idx = response.rfind(action_tag)
    if idx == -1:
        return None
    suffix = response[idx + len(action_tag) :]
    move_str = (
        suffix.strip(" .")
        .replace("$", "")
        .replace("\\boxed{", "")
        .replace("\\text{", "")
        .replace("\boxed{", "")
        .replace("\text{", "")
        .replace("}", "")
        .replace("*", "")
        .replace(" ", "")
        .replace("`", "")
        .replace("\n", "")
    )
    move_str = _HTML_TAG_RE.sub("", move_str)
    if not move_str:
        return None
    return move_str


# ---------------------------------------------------------------------------
# Soft parser (port of repeated_poker_soft_parser).
# ---------------------------------------------------------------------------


def _soft_parse_poker_action(
    selected_action: str,
    legal_moves: Sequence[str],
    pyspiel_state: pyspiel.State,
    player_number: int,
) -> str | None:
    """Map a free-text action (``fold``, ``call``, ``raise 80``, ...) to a legal
    ACPC-style action string (``player=N move=Bet80``).

    Direct port of ``repeated_poker_soft_parser``.
    """
    if player_number >= 2:
        raise ValueError("More than 2 players not currently supported.")
    state_dict = json.loads(str(pyspiel_state))
    up_state_dict = json.loads(state_dict["current_universal_poker_json"])
    acpc_state_str = up_state_dict["acpc_state"].split("\n")[0]
    if not acpc_state_str.startswith("STATE:"):
        raise ValueError(f"Expected ACPC state to start with STATE:, got {acpc_state_str}")
    starting_stacks = up_state_dict.get("starting_stacks", [])
    num_players = len(starting_stacks)
    if not num_players:
        raise ValueError(f"No starting stacks found in {state_dict}.")
    players = [f"Player{i}" for i in range(num_players)]
    acpc_state_str_full = acpc_state_str + "::" + "|".join(players)
    cfg = hh_utils.Config(
        seats=num_players,
        small_blind=state_dict["small_blind"],
        big_blind=state_dict["big_blind"],
        starting_stacks=starting_stacks,
    )
    hand, parse_state = hh_utils.parse_acpc_line(
        acpc_state_str_full,
        cfg=cfg,
        policy=hh_utils.ButtonPolicy(),
        button_index=state_dict["hand_number"] % 2 + 1,
    )
    most_recent_cur_player_event_this_street = None
    if parse_state.street == hh_utils.Street.PREFLOP:
        for event in hand.events:
            if event.actor == player_number:
                most_recent_cur_player_event_this_street = event
    else:
        for event in hand.events:
            if event.street != parse_state.street:
                continue
            elif event.actor == player_number:
                most_recent_cur_player_event_this_street = event
    if most_recent_cur_player_event_this_street is None:
        contrib_street = 0
    else:
        contrib_street = most_recent_cur_player_event_this_street.to_amount or 0
    contrib_total = parse_state.contrib_total[player_number]
    contrib_prev = contrib_total - contrib_street

    selected_lower = selected_action.lower()
    number_match = re.findall(r"\d+", selected_action)
    if "fold" in selected_lower:
        poker_move = "Fold"
    elif "check" in selected_lower:
        poker_move = "Call"
    elif "call" in selected_lower:
        poker_move = "Call"
    elif "all in" in selected_lower or "all-in" in selected_lower:
        return legal_moves[-1]
    elif number_match:
        parsed_amount = int(number_match[-1])
        if parsed_amount <= 0:
            return selected_action  # Illegal move (caller will treat as None).
        bet_size = parsed_amount + contrib_prev

        legal_bet_moves = [a for a in legal_moves if "Bet" in a]
        legal_bet_sizes = [int(a.split("Bet")[1]) for a in legal_bet_moves]
        if bet_size in legal_bet_sizes:
            poker_move = f"Bet{bet_size}"
        else:
            # Map under-sized to smallest legal, over-sized to largest legal,
            # no-bet-legal to Call.
            if not legal_bet_moves:
                poker_move = "Call"
            else:
                poker_move = f"Bet{max(legal_bet_sizes)}"
                for legal_bet_size in legal_bet_sizes:
                    if legal_bet_size >= bet_size:
                        poker_move = f"Bet{legal_bet_size}"
                        break
    else:
        return None

    candidate = f"player={player_number} move={poker_move}"
    if candidate not in legal_moves:
        return None
    return candidate


# ---------------------------------------------------------------------------
# Public functions (called by core_harness)
# ---------------------------------------------------------------------------


def get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str]:
    """Return ``{action_id: action_string}`` for the current state."""
    legal_actions = observation.get("legalActions")
    legal_action_strings = observation.get("legalActionStrings")
    if legal_actions and legal_action_strings:
        return dict(zip(legal_actions, legal_action_strings))
    state = _deserialize_state(observation)
    if state is None:
        return {}
    return {a: state.action_to_string(state.current_player(), a) for a in state.legal_actions()}


def generate_prompt_from_state(
    state: pyspiel.State,
    previous_response: str | None = None,
) -> str:
    """Build the LLM prompt from a pre-deserialized pyspiel state.

    Exposed so the verify script can replay the game forward, deserializing
    once instead of per-prompt.
    """
    readable_state_str = _render_readable_state(state)

    if previous_response is None:
        rethink_prompt = ""
    else:
        # Upstream POKER_RETHINK uses the last 5 lines of the prior generation,
        # or "NO RESPONSE RECEIVED" if empty.
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
    """Build the LLM prompt. Output is byte-identical to upstream's
    ``RepeatedPokerRethinkAgent`` + ``REPEATED_POKER`` template +
    ``RETHINK_REPEATED_POKER`` strategy.
    """
    del move_history, previous_action  # not used in repeated_poker prompts
    state = _deserialize_state(observation)
    if state is None:
        raise ValueError("Observation is missing serializedGameAndState.")
    return generate_prompt_from_state(state, previous_response=previous_response)


def parse_response_with_state(
    response: str,
    legal_action_strings: Sequence[str],
    state: pyspiel.State,
) -> ParseResult:
    """Parse with a pre-deserialized state. Same as ``parse_response`` but
    skips deserialization -- exposed for the verify script."""
    raw = _extract_move_from_response(response)
    if raw is None:
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

    Two-stage pipeline matching upstream:
    1. ``RuleBasedMoveParser`` -- extract suffix after the last "Final Answer:".
    2. ``PokerSoftParser`` -- soft-match against legal moves, handling the
       street-total vs ACPC cross-street-total bet-sizing convention.
    """
    raw = _extract_move_from_response(response)
    if raw is None:
        return ParseResult(legal_action=None, raw_action=None)
    if observation is None:
        # Without state context we can only return what we extracted; the
        # framework will treat this as an illegal-move retry.
        return ParseResult(legal_action=None, raw_action=raw)
    state = _deserialize_state(observation)
    if state is None:
        return ParseResult(legal_action=None, raw_action=raw)
    return parse_response_with_state(response, legal_action_strings, state)


