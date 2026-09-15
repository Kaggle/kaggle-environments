"""LLM harness for Hanabi Arena (2v2 team variant of OpenSpiel Hanabi).

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.

Base Hanabi is fully cooperative, so it has no winner to rank by. The arena
manufactures a head-to-head: two teams of two each play their own private
Hanabi table, both dealt from the *same shuffled deck*, and the higher final
score wins. A player sees their own table only -- their teammate's hand
face-up, their own as hint knowledge, and nothing at all of the opposing
table.

The arena observation carries a per-table ``move_history`` in which the env
has already annotated each decision with the facts it made public -- the card
a play or discard removed, whether that play advanced a firework, and which
slots a hint pointed at. Only this player's own table appears there, so the
opposing table is never rendered.

That annotation is deliberately done env-side rather than by deserializing
``serializedGameAndState`` here: the serialized state reconstructs every
hidden hand, so ``hanabi_arena`` withholds it from agent observations
entirely. This harness therefore has no serialized-state path at all.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from kaggle_environments.core_harness import ParseResult, parse_json_action, render_rethink_suffix

# The engine's own defaults, mirrored from ``hanabi_arena_game``'s parameter
# specification. They are the fallbacks the prompt renders when a value is
# missing from the observation: every rule line the prompt states is an
# assertion about the game being played, and a wrong constant is worse than a
# missing one because the model cannot tell it was misinformed.
_DEFAULT_COLORS = 5
_DEFAULT_RANKS = 5
_DEFAULT_HAND_SIZE = 5
_DEFAULT_MAX_LIFE_TOKENS = 3
_DEFAULT_MAX_INFO_TOKENS = 8
# "RYGWB" -- OpenSpiel's ColorIndexToChar order (hanabi_lib/util.cc).
_DEFAULT_COLOR_LETTERS = ["R", "Y", "G", "W", "B"]

# --- Prompt -----------------------------------------------------------------


HANABI_ARENA_PROMPT_TEMPLATE = """Let's play Hanabi Arena, a 2v2 team version of \
the cooperative card game Hanabi.

Two teams of {players_per_team} each play their own private Hanabi table. Both \
tables are dealt from the same shuffled deck -- identical starting hands, \
identical draw order -- so the two teams face the same puzzle. The team with \
the higher final score wins; equal scores are a draw. You never see the \
opposing table.

Your teammate is playing the same way you are, so you can expect them to read \
the table as you would. Hints are the only communication there is, and they \
only reach your own table.

Rules:
- The deck has {num_colors} colors ({color_list}) and ranks 1-{num_ranks}, \
{deck_total} cards in total: per color, {deck_composition}.
- Each player holds up to {hand_size} cards and can see every other player's \
cards at their table but not their own. Once the deck is empty there are no \
replacements, so hands shrink as cards are played or discarded.
- The team builds one firework stack per color in ascending order 1 to \
{num_ranks}. The score is the sum of the stack heights, at most {max_score}.
- On your turn you take exactly one action:
  * Play a card from one of your slots. If its rank is exactly one above its \
color's current stack height it joins that stack; otherwise it is discarded \
and the team loses a life token.
  * Discard a card from one of your slots. It is lost and the team regains one \
info token. Illegal while info tokens are at the maximum of {max_info_tokens}.
  * Hint another player at your table one color or one rank. Every card of that \
color/rank in their hand is pointed out; the rest are thereby excluded from it. \
Costs one info token, so it is illegal at 0 info tokens, and the color or rank \
you name must match at least one card in that player's hand.
- Every hint is public at your table: all your team hears who was told what. \
The opposing team hears nothing.
- After you play or discard, your remaining cards shift down one slot and, if \
the deck is not empty, a replacement is drawn into your highest slot. Slot 0 is \
therefore always your oldest card.
- Completing a color's stack at rank {num_ranks} regains one info token. Info \
tokens never exceed {max_info_tokens}.
- Your table ends when your team loses its last life token (the score becomes 0 \
no matter how many fireworks were played), when every stack is complete \
({max_score} points), or when the deck runs out -- after the deck empties each \
player takes exactly one more turn.
- The two tables take turns one move at a time. Once a table finishes, the \
other keeps playing alone until it finishes too. Within your table, turns pass \
in seat order, wrapping from Player {last_player_id} back to Player \
{first_player_id}.

You are Player {player_id} on Team {team_id}, seat {seat} at your team's table. \
Your teammate is Player {teammate_id}.

Your team's fireworks:
{fireworks_block}
Score: {score}/{max_score}
Tokens: {life_tokens}/{max_life_tokens} life, {info_tokens}/{max_info_tokens} info
Deck: {deck_size}{final_turns}
Discarded: {discards}

Your own hand, which you cannot see ("told" is what hints have said about the \
card out loud; "possible" is what is still consistent with every hint, \
including hints that skipped this card):
{own_hand_block}
{other_hands_block}
Moves played at your table so far (all seats, oldest first):
{move_history}

Action notation -- your answer must be exactly one of these forms:
  (Play N)                     play the card in your slot N
  (Discard N)                  discard the card in your slot N
  (Reveal player +K color C)   hint color C to the player K seats after you
  (Reveal player +K rank R)    hint rank R to the player K seats after you
Slots are numbered from 0. Colors are the single letters {color_list}. Hint \
targets are written as an offset from your own seat: {offset_map}.

It is your turn. Reason step by step about what your teammate knows, what your \
own cards could be, and which action scores your team the most, then give your \
final answer in a JSON block:

```json
{{
  "move": "<action>"
}}
```

For example: `{{"move": "{example_move}"}}`

Failure to output your final answer in the specified format, or choosing an \
illegal action, will result in a loss.
"""


RETHINK_ILLEGAL = """

You suggested move "{previous_action}" but this is not a legal move.
Reconsider the rules and the current state, then pick a legal move.

(Keep using the same JSON output format as before -- only the move value needs to change.)
"""

# A separate template because the correction is a different one. When the
# answer named more than one move, the move itself may be perfectly legal and
# telling the model otherwise sends it to re-examine the board -- the one
# thing that is not wrong -- instead of its phrasing. See
# ``_is_undecided_answer``.
RETHINK_UNDECIDED = """

Your answer "{previous_action}" names more than one move, so it is not clear
which one you chose. The "move" value must be a single action and nothing
else -- put any comparison with the moves you rejected in your reasoning above
the JSON, not inside it.

(Keep using the same JSON output format as before -- only the move value needs to change.)
"""

RETHINK_UNPARSABLE = """

Your previous response ended with:
{previous_response}

No JSON answer could be parsed from that. Conclude your response
with your final move as JSON in a ```json fenced block, exactly
as the original instructions required:

```json
{{"move": "<action>"}}
```

For example: `{{"move": "(Play 0)"}}`

The move you choose must also be legal in the current state.
"""


# --- Observation helpers ----------------------------------------------------


def _parse_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the JSON arena view emitted by ``HanabiArenaObserver``."""
    raw = observation.get("observationString")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _deck_composition(num_ranks: int) -> str:
    """Describe one color's copies, mirroring OpenSpiel's deck.

    Three copies of rank 1, one copy of the top rank, two of everything
    between. ``NumberCardInstances`` tests the bottom rank *before* the top
    one, so in the degenerate ``ranks=1`` game -- where the single rank is
    both -- the bottom-rank count wins and there are three copies, not one.
    """
    if num_ranks <= 1:
        return "three 1s"
    if num_ranks == 2:
        return "three 1s and one 2"
    middle = "two 2s" if num_ranks == 3 else f"two each of 2-{num_ranks - 1}"
    return f"three 1s, {middle}, and one {num_ranks}"


def _deck_total(num_colors: int, num_ranks: int) -> int:
    """Deck size implied by ``_deck_composition`` -- NOT ``colors * ranks``.

    Three copies of rank 1, one of the top rank, two of each rank between, so
    a standard 5x5 deck holds 50 cards rather than 25. Only used as a fallback
    when the observation does not carry ``deck_total``; it must agree with the
    composition line the prompt prints beside it.
    """
    if num_ranks <= 1:
        return 3 * num_colors
    return num_colors * (3 + 2 * (num_ranks - 2) + 1)


def _color_letters(table: Mapping[str, Any]) -> list[str]:
    """Color letters in engine order, from the fireworks the engine reported."""
    fireworks = table.get("fireworks") or {}
    return list(fireworks.keys())


def _card_text(card: Mapping[str, Any] | None) -> str:
    if not card:
        return "??"
    return f"{card.get('color', '?')}{card.get('rank', '?')}"


def _told_text(card: Mapping[str, Any]) -> str:
    """What the holder has been told out loud about this card."""
    parts = []
    if card.get("hinted_color"):
        parts.append(f"color {card['hinted_color']}")
    if card.get("hinted_rank"):
        parts.append(f"rank {card['hinted_rank']}")
    return " and ".join(parts) if parts else "nothing"


def _possible_text(card: Mapping[str, Any]) -> str:
    """What is still consistent with every hint the holder has received."""
    colors = "".join(card.get("plausible_colors") or []) or "-"
    ranks = "".join(str(r) for r in card.get("plausible_ranks") or []) or "-"
    return f"colors {colors}, ranks {ranks}"


def _hand_of_seat(table: Mapping[str, Any], seat: int) -> dict[str, Any] | None:
    for hand in table.get("hands") or []:
        if hand.get("player") == seat:
            return hand
    return None


def _render_own_hand(cards: Sequence[Mapping[str, Any]]) -> str:
    """Own slots as hint knowledge -- never the faces.

    ``hanabi_arena`` exposes no ``observation_type`` parameter, so unlike base
    Hanabi there is no "seer" variant here and a seat never sees its own
    cards mid-game. The faces DO appear in the terminal observation, which is
    not a turn the prompt is ever built for.
    """
    if not cards:
        return "  (no cards)"
    return "\n".join(
        f"  slot {i}: told {_told_text(card)}; possible {_possible_text(card)}" for i, card in enumerate(cards)
    )


def _render_other_hands(table: Mapping[str, Any], seat: int, players_per_team: int, team_base: int) -> str:
    """Every other seat at this table face-up, labelled with its hint offset.

    Seats are relabelled with their arena-wide player ids so the model can
    talk about "Player 3" rather than "the other seat", while the hint
    offsets stay table-relative because that is what the engine's action
    strings use.
    """
    blocks = []
    for offset in range(1, players_per_team):
        other_seat = (seat + offset) % players_per_team
        hand = _hand_of_seat(table, other_seat)
        player_id = (hand or {}).get("player_id", team_base + other_seat)
        cards = (hand or {}).get("cards") or []
        header = f"Player {player_id}'s hand (hint target +{offset}):"
        if not cards:
            blocks.append(f"{header}\n  (no cards)")
            continue
        lines = [
            f"  slot {i}: {_card_text(c.get('card'))} -- told {_told_text(c)}; possible {_possible_text(c)}"
            for i, c in enumerate(cards)
        ]
        blocks.append(header + "\n" + "\n".join(lines))
    return "\n".join(blocks)


def _render_fireworks(table: Mapping[str, Any]) -> str:
    fireworks = table.get("fireworks")
    num_ranks = int(table.get("ranks", 5) or 5)
    if fireworks is None:
        return "  (unavailable -- the board could not be read this turn)"
    if not fireworks:
        return "  (none started)"
    lines = []
    for color, height in fireworks.items():
        if height >= num_ranks:
            lines.append(f"  {color}: complete (up to {color}{num_ranks})")
        elif height == 0:
            lines.append(f"  {color}: empty, next playable {color}1")
        else:
            lines.append(f"  {color}: up to {color}{height}, next playable {color}{height + 1}")
    return "\n".join(lines)


def _render_discards(table: Mapping[str, Any]) -> str:
    """Discards grouped as ``R1 x2`` so duplicate-tracking is readable.

    "(none)" is reserved for a genuinely empty pile. Counting discards is how
    a Hanabi player knows a card is dead, so an unreadable pile rendered as
    empty would have the model treat every 5 as still saveable.
    """
    discards = table.get("discards")
    if discards is None:
        return "(unavailable)"
    if not discards:
        return "(none)"
    counts: dict[str, int] = {}
    for card in discards:
        counts[_card_text(card)] = counts.get(_card_text(card), 0) + 1
    order = {letter: i for i, letter in enumerate(_color_letters(table))}
    tokens = sorted(counts.items(), key=lambda kv: (order.get(kv[0][0], 99), kv[0][1:]))
    return ", ".join(name if n == 1 else f"{name} x{n}" for name, n in tokens)


def _offset_map(seat: int, players_per_team: int, team_base: int) -> str:
    return ", ".join(
        f"+{offset} = Player {team_base + (seat + offset) % players_per_team}" for offset in range(1, players_per_team)
    )


# --- Move history -----------------------------------------------------------


_PLAY_ACTION_RE = re.compile(r"^\(Play (\d+)\)$")
_DISCARD_ACTION_RE = re.compile(r"^\(Discard (\d+)\)$")
_REVEAL_COLOR_ACTION_RE = re.compile(r"^\(Reveal player \+(\d+) color (\w)\)$")
_REVEAL_RANK_ACTION_RE = re.compile(r"^\(Reveal player \+(\d+) rank (\d+)\)$")


def _render_hint_slots(entry: Mapping[str, Any]) -> str:
    """Where a hint's cards are NOW, and where they were when it was given.

    Slot numbers are positions, not identities: every play or discard slides
    the cards above it down one, so the slots a hint pointed at stop naming
    those cards almost immediately. The env keeps both readings -- ``slots``
    walked forward to the present, ``slots_when_given`` as the public record
    -- and both belong in the prompt, because they answer different
    questions. Rendering only the original is the bug this replaced: the
    holder cannot see the faces, so a history line saying "rank 2 -- slot 4"
    beside an own-hand block showing nothing known about slot 4 is a
    contradiction they have no way to resolve.

    The two are printed together only when they differ; a hint whose cards
    have not moved reads as plainly as it ever did.
    """
    given = entry.get("slots_when_given")
    if given is None:
        # Which slots a hint pointed at is the substance of the hint. A guess
        # is worse than an omission, and "no slots" would be a lie -- the
        # engine only allows hints that match at least one card.
        return ""
    current = entry.get("slots", given)
    noun = "slot" if len(given) == 1 else "slots"
    if not given:
        return " -- no slots"
    rendered = f" -- {noun} {', '.join(str(s) for s in given)}"
    if list(current) == list(given):
        return rendered
    if not current:
        return f"{rendered} at the time, all since played or discarded"
    moved = "slot" if len(current) == 1 else "slots"
    suffix = "" if len(current) == len(given) else " (the rest since played or discarded)"
    return f"{rendered} at the time, now {moved} {', '.join(str(s) for s in current)}{suffix}"


def _describe_decision(entry: Mapping[str, Any], team_base: int) -> str:
    """One history line: the action plus the public facts it revealed.

    Every fact read here was recorded by the env at the moment the action
    resolved, when it became common knowledge at this table (see
    ``_public_facts`` in ``hanabi_arena_game``). Nothing still face-down is
    read, and the opposing table never appears in this list at all.
    """
    player_id = entry.get("player_id", team_base)
    label = entry.get("label", "?")

    match = _PLAY_ACTION_RE.match(label) or _DISCARD_ACTION_RE.match(label)
    if match:
        slot = int(match.group(1))
        card = _card_text(entry.get("card"))
        if label.startswith("(Discard"):
            return f"P{player_id} discarded slot {slot} ({card})"
        advanced = entry.get("advanced")
        if advanced is None:
            return f"P{player_id} played slot {slot} ({card})"
        outcome = "firework advanced" if advanced else "misplay, life lost"
        return f"P{player_id} played slot {slot} ({card}) -- {outcome}"

    if entry.get("hint_kind") is not None:
        target_id = entry.get("target_player_id", team_base)
        kind, value = entry["hint_kind"], entry.get("hint_value")
        return f"P{player_id} hinted P{target_id} {kind} {value}{_render_hint_slots(entry)}"

    return f"P{player_id} {label}"


def _move_history_lines(table: Mapping[str, Any], team_base: int) -> list[str] | None:
    """This table's moves, annotated with what each one made public.

    ``None`` when the observation carries no history at all, so the caller
    can say so rather than render an empty log as "nothing has happened".
    """
    history = table.get("move_history")
    if history is None:
        return None
    return [_describe_decision(entry, team_base) for entry in history]


def _readable(table: Mapping[str, Any], key: str) -> str:
    """A live counter, or ``unavailable`` when the view could not be read.

    Zero is a real and dangerous reading for every one of these -- 0 lives is
    one misplay from a bomb-out, 0 info tokens forbids hinting, 0 cards left
    means the final round is running -- so a missing field must never render
    as one. The move log already draws this distinction; the state block
    stating a confident falsehood beside it was the gap.
    """
    value = table.get(key)
    return "unavailable" if value is None else str(value)


def _render_deck_size(table: Mapping[str, Any]) -> str:
    deck_size = table.get("deck_size")
    return "unavailable" if deck_size is None else f"{deck_size} card(s) left"


def _render_final_turns(remaining: int | None) -> str:
    if remaining is None:
        return ""
    if remaining == 1:
        return " -- final round: this is the last turn at your table"
    return f" -- final round: {remaining} turns remain at your table, including yours"


def _format_move_history(lines: Sequence[str] | None) -> str:
    """Render the history, or say plainly that it is unavailable.

    ``None`` (no history could be read at all) and ``[]`` (nobody has moved
    yet) must render differently: a model told "this is the first turn"
    forty moves in has no way to notice, whereas one told the log is missing
    knows to fall back on the hint knowledge rendered above.
    """
    if lines is None:
        return "  (unavailable -- the move log could not be reconstructed this turn)"
    if not lines:
        return "(none yet -- this is the first turn at your table)"
    return "\n".join(f"  {i + 1}. {line}" for i, line in enumerate(lines))


# --- Answer matching --------------------------------------------------------
#
# The engine's action strings are parenthesised ("(Play 0)", "(Reveal player +1
# color R)"). Models routinely drop the parentheses, say "hint" instead of
# "reveal", write a color name instead of its letter, omit the "color"/"rank"
# keyword, or append a note about the card they think is in the slot. All of
# those are notation drift on an action the model explicitly chose, so the
# matcher canonicalizes them rather than paying a rethink round-trip.
#
# It never infers an action the model did not write. Two rules keep that
# honest, and both matter specifically in Hanabi:
#
# - The filler allowed between the verb and the slot number is an explicit
#   whitelist, not "any non-digit". The prompt renders teammates' cards as
#   `R1`/`B3` tokens, so a model naming the *card* rather than the *slot* is
#   the single most likely notation slip -- and a permissive gap silently
#   turns "Play B3" into slot 3, an action the model never chose. Spelled-out
#   ranks are held back from the slot reading for the same reason: "play the
#   one" names a card in Hanabi prose at least as readily as it names a slot.
# - A hint target written without a "+" is ambiguous between an offset and a
#   player id. It is resolved against the arena player ids this seat's prompt
#   actually printed, and otherwise only when the legal set leaves exactly one
#   possible target, so the two readings cannot disagree.

_VERB_ALIASES = {"hint": "reveal", "tell": "reveal", "clue": "reveal", "reveal": "reveal"}
_COLOR_WORDS = {
    "red": "r",
    "yellow": "y",
    "green": "g",
    "white": "w",
    "blue": "b",
}
_RANK_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}

# Words a model may put between the verb and the slot number. Deliberately
# closed: a color letter must NOT be absorbed here.
_FILLER = r"(?:slot|card|my|the|from|in|number)"

# Only a digit is a slot. A spelled-out rank is deliberately NOT accepted:
# "play the one" reads as the rank-1 card at least as naturally as slot 1, and
# the whole point of the closed _FILLER list is that "play the <card>" must not
# resolve to a slot. Normalizing rank words globally would hand that back --
# "the" is filler and "one" would have become "1" -- so rank words are expanded
# only in the hint-value position below, where a rank is the only reading.
_RAW_SLOT_RE = re.compile(rf"^(play|discard)(?:\s+{_FILLER})*\s+(\d+)\b(.*)$")

# A rank written as a digit or as a word, for the hint-value position.
_RANK_TOKEN = r"(?:\d+|one|two|three|four|five)"

# Filler a model may put between the hint target and the value it names
# ("tell player +1 that they have a 3"). Closed for the same reason as
# _FILLER, though the exposure is milder here: the value slot only accepts a
# digit or a single letter, and a letter that is not a color simply fails to
# match any legal move.
_HINT_FILLER = r"(?:about|that|they|their|has|have|holds?|is|are|all|of|with|an?|the|cards?)"

# Rank alternates before color so a digit is read as a rank, never as a stray
# letter match; the two branches are disjoint on their value character. The
# target allows "player1" as well as "player 1" (\s* rather than \s+), and the
# value allows a plural "1s" -- "hint player +1 their 1s" is how the hint
# actually gets said, and the rank is no less unambiguous for the "s".
_RAW_REVEAL_RE = re.compile(
    rf"^reveal (?:player\s*|p)?(\+?)(\d+)(?:\s+{_HINT_FILLER})*"
    rf"\s+(?:(?:rank )?({_RANK_TOKEN})s?|(?:colou?r )?([a-z]))\b(.*)$"
)
# "Reveal color R to player +1" -- the same move with target and value
# swapped. The literal "to" is required: without it "reveal r 1" is ambiguous
# between this order and the one above.
_RAW_REVEAL_TO_RE = re.compile(
    rf"^reveal (?:(?:rank )?({_RANK_TOKEN})s?|(?:colou?r )?([a-z]))\s+to\s+(?:player\s*|p)?(\+?)(\d+)\b(.*)$"
)

# A trailing note is set off by a bracket or a separator. The lookbehind
# requires preceding content so a fully parenthesised "(Play 2)" is not
# mistaken for its own annotation.
_ANNOTATION_RE = re.compile(r"(?<=\S)\s*(?:\(|\[|--|—|–|,|;|\s-\s)(.*)$", re.DOTALL)
# Unwraps "(Play 2)" only when nothing inside closes early, so "(Play 2) (W1)"
# is left for the annotation stripper rather than swallowed whole.
_WRAPPED_RE = re.compile(r"^\(([^)]*)\)$")

# A second *action*, or an undecided "A or B", in text trailing a complete
# action. Either means the model named more than one move. Case-insensitive:
# _is_annotation runs on the RAW tail as often as on a normalized one, so
# "Play 0 -- Or maybe Discard 1" must be caught as readily as its lowercase
# twin.
#
# Matching a verb ALONE would be wrong here, because Hanabi's strategic
# vocabulary is its verb list: "Play 0 (safest play)", "(better than a hint
# right now)", "(sets up a play)" and "(this clue is best)" are commentary on
# the chosen move, not a second one. A second action needs an operand -- a
# slot number after play/discard, or a target or value after a hint verb.
#
# The hint operand must accept a BARE COLOR LETTER, not just the spelled-out
# word. Two things go wrong without it, and both end the same way -- the
# engine takes a move the model did not settle on, no rethink fires, and the
# model never learns half its answer was dropped:
#
#   - "Play 0 or hint R" is how a model abbreviates the alternative it is
#     still weighing, and the prompt itself teaches the single letters
#     ("Colors are the single letters R/Y/G/W/B"), so this is the spelling to
#     expect rather than an exotic one.
#   - "Play 0 (or reveal red)" IS caught on the raw tail, but this pattern is
#     also applied downstream to _normalize'd text, where _COLOR_WORDS has
#     already rewritten "red" to "r". A guard that recognizes only the
#     pre-normalization spelling silently stops guarding after normalization.
_ACTION_PHRASE = (
    rf"(?:play|discard)\s+(?:{_FILLER}\s+)*\#?\d"
    r"|(?:reveal|hint|tell|clue)\s+(?:player\s*|p)?"
    r"(?:\+?\d|colou?rs?\b|ranks?\b|red\b|yellow\b|green\b|white\b|blue\b|[rygwb]\b)"
)

# ...and an operand is still not enough, because a fully-specified move named
# in the tail is usually one the model REJECTED or one it expects a teammate
# to make next -- which is what Hanabi reasoning consists of. "Play 0 (discard
# 1 is worse)", "Reveal player +1 rank 1 (they will then play slot 1)",
# "Play 0 rather than Discard 3" each name exactly one move for THIS turn and
# annotate it with another; refusing them taxes a retry on the models that
# reason best, and (because raw_action is set) the rethink they get wrongly
# tells them their legal move was illegal.
#
# So the guard fires only when an *undecided* marker introduces the second
# action -- "or", "/", "either", "maybe", "otherwise" -- or a *sequencing* one
# does ("then", "followed by"), which claims two moves on one turn. Those are
# the words of an answer that did not settle on one action. The rejection
# markers -- "rather than", "better than", "not", "instead of", "worse",
# "considered" -- are commentary on a settled one, and pass.
#
# The marker must be within a short reach of the action phrase. Unbounded, any
# "or" anywhere in a long tail ("it is W1 or B1, so I play 0") would arm the
# guard against an action phrase it has nothing to do with.
_INDECISION = r"(?:\bor\b|/|\beither\b|\bmaybe\b|\bpossibly\b|\bperhaps\b|\botherwise\b|\belse\b)"
_SEQUENCING = r"(?:\bthen\b|\bfollowed by\b|\bafter that\b)"
_SECOND_ACTION_RE = re.compile(
    rf"{_INDECISION}[^.;]{{0,20}}?\b(?:{_ACTION_PHRASE})",
    re.IGNORECASE,
)
_SEQUENCED_ACTION_RE = re.compile(
    rf"{_SEQUENCING}[^.;]{{0,20}}?\b(?:{_ACTION_PHRASE})",
    re.IGNORECASE,
)

# A third shape, where the marker TRAILS the second action rather than
# introducing it: "Play 0 (Discard 1 is also fine)", "(Reveal player +1 rank 2
# works equally well)". Endorsing the alternative is as undecided as offering
# it with "or" -- the model has named two acceptable moves and left the choice
# open. The mirror-image phrasings, "(Discard 1 is worse)" and "(Discard 1
# loses tempo)", reject it and stay commentary, so the marker list is
# endorsements only.
_ENDORSED_ACTION_RE = re.compile(
    rf"\b(?:{_ACTION_PHRASE})[^.;]{{0,30}}?"
    r"\b(?:also (?:fine|good|ok|okay|works|reasonable|viable)"
    r"|(?:works|is fine|is good|is ok) too"
    r"|equally (?:good|fine|viable|strong)"
    r"|just as (?:good|fine|strong))",
    re.IGNORECASE,
)

# Sequencing alone over-fires, because the single most common thing a Hanabi
# player says about their move is what it lets their PARTNER do next: "Play 0
# (my teammate will then play slot 1)", "Reveal player +1 rank 1 -- P3 can
# then play slot 3". Those name one move for this turn and predict another for
# somebody else's, so they are commentary. What separates them from "Play 1
# then discard 2" is a subject or a modal ahead of the sequencing word -- the
# grammar of a prediction rather than of a second instruction.
#
# Scanned over the text BEFORE the marker, so a prediction anywhere ahead of
# it disarms the sequencing branch. The undecided branch is untouched by this:
# "or" means undecided regardless of who the sentence is about.
_PREDICTION_RE = re.compile(
    r"\b(?:will|would|should|shall|can|could|may|might|they|he|she|partner|teammate|p\d+)\b",
    re.IGNORECASE,
)

# An undecided tail: either it is nothing but connectors and numbers ("or 1",
# "/ 4", ", 1"), or it opens with a connector and reaches a *value* -- a rank
# ("or maybe 1") or a color ("or Y", "or maybe white"). Both name an
# alternative move rather than describing the chosen one.
#
# The value alternative must cover colors, not just digits. A hint's value is
# a color at least as often as a rank, and "Reveal player +1 color R or Y" has
# no second verb for _SECOND_ACTION_RE to catch -- so a digit-only reading
# accepts it and silently submits whichever color was named first. That is the
# confident-failure shape: the engine takes the move, no rethink fires, and the
# model never learns half its answer was dropped.
#
# The first branch is also what refuses a bare-digit tail ("Play 0-1",
# "Play 0, 1"): a separator followed by nothing but a digit reads at least as
# naturally as a second slot than as the rank of the card in the first, and
# guessing would submit a move the model did not settle on.
#
# That branch requires the tail to actually REACH a digit or an "or", via the
# lookahead. Without it the branch matches any tail made only of punctuation,
# which refuses "Play 0!", "Play 0?", "**Play 0**" and "Play 0 ✓" -- a
# decorated but perfectly decided answer, costing a rethink for punctuation
# that names no alternative at all.
#
# Every branch is anchored, which is the whole point. An unanchored "or" or
# "/" fires on ordinary Hanabi prose -- "(it is W1 or B1)", "(2/3 chance)",
# "(dead card, R/Y both done)" -- and rejecting those costs a rethink for a
# model that did exactly what it was asked. Anchoring keeps the rejection to
# tails that OPEN with the connector, which is where an undecided answer lives.
#
# "alternatively" and "instead" get the same anchoring as everything else. A
# tail OPENING with one ("Play 0 -- alternatively, a hint") is an answer still
# weighing two moves; the same words mid-tail are how a decided answer names
# what it passed up ("Play 0 (R1 instead of R2 is what I need)"), and refusing
# those is the same false rejection the second-action guard above was narrowed
# to avoid.
#
# "instead OF" is excluded even at the front, because the preposition inverts
# the word: bare "instead" offers an alternative ("Play 0 -- instead, a hint"),
# while "instead of X" names the alternative being REJECTED ("Play 0 -- instead
# of burning a token"). The second is a settled answer with its rationale
# attached, which is the most natural way to write one down.
_COLOR_VALUE = r"(?:red|yellow|green|white|blue|colou?rs?\b|[rygwb]\b)"
_UNDECIDED_RE = re.compile(
    r"^(?=.*(?:\d|\bor\b))(?:\W|\d|\bor\b)+$"
    rf"|^\W*(?:or\b|/)(?:[^\d]*\d|\W*(?:{_COLOR_VALUE}|maybe\b|possibly\b|perhaps\b|even\b|nothing\b))"
    r"|^\W*(?:instead\b(?!\s+of\b)|alternatively\b)",
    re.IGNORECASE,
)

# A slot written as a negative index -- "Play -1". _normalize drops the hyphen
# (which is what makes "Discard slot-2" work), so without this guard the move
# silently resolves to slot 1. A model writing -1 most plausibly means a
# Python-style index from the END of the hand, which is the opposite end from
# slot 1 -- the engine would take a move the model did not choose and no
# rethink would fire. Anchored at the verb so only the operand position counts:
# "Discard slot-2" and a trailing "Play 0 - 2 lives left" are untouched.
_NEGATIVE_SLOT_RE = re.compile(r"^\W*(?:play|discard)\s+-\s*\d", re.IGNORECASE)


def _normalize(text: str) -> str:
    """Lowercase, unwrap punctuation, and canonicalize verbs and color words.

    Color words are expanded here because "red" is never anything but a color.
    Rank words are NOT: "one" is a rank in the hint-value position but a card
    in "play the one", and expanding it globally would let the slot pattern
    read that as slot 1. The hint patterns accept the word directly instead.

    Markdown emphasis and backticks are stripped along with the rest: a model
    told to answer with "(Play 0)" writes "**(Play 0)**" or "`(Play 0)`" often
    enough, and the asterisks change nothing about which move it named.
    """
    cleaned = re.sub(r"[()\[\]{}\"',.:\-_#*`]+", " ", text.lower())
    tokens = cleaned.split()
    if tokens:
        tokens[0] = _VERB_ALIASES.get(tokens[0], tokens[0])
    tokens = [_COLOR_WORDS.get(token, token) for token in tokens]
    return " ".join(tokens)


def _is_annotation(rest: str) -> bool:
    """True when text trailing a complete action is commentary, not part of it.

    A remainder naming a second action, or offering a second slot, means the
    model did not settle on one -- better to rethink than to guess which half
    it meant. Everything else is commentary and must be allowed through: in
    Hanabi the model is reasoning about probabilities and about the other
    moves it rejected, so "(2/3 chance)", "(safest play)" and "(better than a
    hint)" are all ordinary ways to annotate a move it did choose.
    """
    if _SECOND_ACTION_RE.search(rest) or _ENDORSED_ACTION_RE.search(rest):
        return False
    # Sequencing is only a second instruction when the sentence is about the
    # speaker; ahead of a prediction it is a forecast of the partner's turn.
    match = _SEQUENCED_ACTION_RE.search(rest)
    if match and not _PREDICTION_RE.search(rest[: match.start()]):
        return False
    return not _UNDECIDED_RE.search(rest)


def _split_annotation(raw: str) -> tuple[str, bool]:
    """Drop a trailing note like " (likely W1)" or " - my W1".

    Only commentary is dropped. If the tail names another action the text is
    returned untouched, so an undecided answer fails to match and the model
    gets a rethink instead of whichever half happened to come first.

    The second return value says whether a tail was refused that way. The
    caller needs it to tell the two rethinks apart: an answer refused for
    naming two moves usually named a perfectly legal one first, and must not
    be told its move was illegal.
    """
    text = raw.strip()
    blocked = False
    for _ in range(2):
        wrapped = _WRAPPED_RE.match(text)
        if wrapped:
            text = wrapped.group(1).strip()
        match = _ANNOTATION_RE.search(text)
        if not match:
            break
        if not _is_annotation(match.group(1)):
            blocked = True
            break
        text = text[: match.start()].strip()
    return text, blocked


def _strip_annotation(raw: str) -> str:
    return _split_annotation(raw)[0]


def _is_undecided_answer(raw: str) -> bool:
    """True when an answer was refused for naming more than one move.

    Distinguishes the two ways a parse can fail. An answer the matcher
    rejected because its text named a second move is a *phrasing* problem and
    the move it led with is usually legal; an answer that simply named an
    illegal move is a *game-state* problem. Telling a model the first is the
    second points it at the board, which was never the issue, and models that
    reason well about Hanabi conventions hit the first case most often --
    naming the move they rejected, or the one they expect a teammate to make,
    is how that reasoning is written down.

    Structural only, so it needs no legal-move list: it asks whether the text
    parses as a complete action whose trailing tail the guard refused.
    """
    if not raw:
        return False
    text, blocked = _split_annotation(raw)
    if blocked:
        return True
    normalized = _normalize(text)
    match = _RAW_SLOT_RE.match(normalized)
    if match:
        return not _is_annotation(match.group(3))
    match = _RAW_REVEAL_RE.match(normalized) or _RAW_REVEAL_TO_RE.match(normalized)
    return bool(match) and not _is_annotation(match.group(5))


def _reveal_offsets(legal_moves: Sequence[str]) -> set[int]:
    """Hint offsets that are legal right now, read off the engine's labels.

    Reuses the history renderer's patterns so one engine label format is
    parsed by one pair of regexes; both capture the offset in group 1.
    """
    matches = (_REVEAL_COLOR_ACTION_RE.match(move) or _REVEAL_RANK_ACTION_RE.match(move) for move in legal_moves)
    return {int(m.group(1)) for m in matches if m}


def _player_id_offsets(observation: Mapping[str, Any] | None) -> dict[int, int]:
    """``{arena player id: hint offset}`` for the other seats at this table.

    The arena prompt names teammates by their arena-wide player id -- "Your
    teammate is Player 3", "+1 = Player 3" -- because that is how the rest of
    the observation labels them. A model that writes the id the prompt taught
    it is spelling the engine's offset in the arena's own vocabulary, so this
    is the mapping needed to read it back.

    Deriving it per seat is what keeps the two teams symmetric. Team 0 seat 0
    is the one seat whose teammate id (1) happens to equal the offset (+1);
    without this map that seat parses "Reveal player 1 ..." and the other
    three seats do not, which is a handicap on one side of a head-to-head
    matchup rather than a uniform tolerance gap.
    """
    if not observation:
        return {}
    state = _parse_observation(observation)
    if not state:
        return {}
    table = state.get("table") or {}
    try:
        players_per_team = int(state.get("players_per_team", table.get("num_players", 2)) or 2)
        player_id = int(observation.get("playerId", state.get("your_player_id", 0)) or 0)
        team_id = int(state.get("your_team_id", player_id // players_per_team))
        seat = int(state.get("your_seat", player_id % players_per_team))
    except (TypeError, ValueError):
        return {}
    if players_per_team < 2:
        return {}
    team_base = team_id * players_per_team
    return {team_base + (seat + offset) % players_per_team: offset for offset in range(1, players_per_team)}


def _resolve_offset(
    sign: str,
    number: int,
    legal_moves: Sequence[str],
    player_id_offsets: Mapping[int, int] | None = None,
) -> int | None:
    """Turn a written hint target into an offset, or ``None`` if ambiguous.

    An explicit "+K" is an offset by construction. A bare "K" has two possible
    readings, and both are things the prompt actually shows the model:

    - an arena player id, which is how the prompt names every seat; or
    - a table-relative offset, which is what the engine's action strings use.

    Each reading is resolved independently and they must not disagree. The
    player-id reading is only taken when that id is a seat at this table and
    the offset it implies is legal right now. The offset reading keeps its
    original guard -- accepted only when exactly one target is hintable at all
    -- so a bare number can never pick between two legal targets.

    At this game's two-seat tables the readings never both resolve to
    different offsets, since offset 2 is not legal there. The disagreement
    check is kept general so a wider table cannot quietly break it.
    """
    if sign == "+":
        return number
    offsets = _reveal_offsets(legal_moves)

    as_player_id = (player_id_offsets or {}).get(number)
    if as_player_id is not None and as_player_id not in offsets:
        as_player_id = None
    as_offset = number if offsets == {number} else None

    if as_player_id is not None and as_offset is not None and as_player_id != as_offset:
        return None
    return as_player_id if as_player_id is not None else as_offset


def _canonical_forms(
    raw: str,
    legal_moves: Sequence[str],
    player_id_offsets: Mapping[int, int] | None = None,
) -> list[str]:
    """Engine-shaped action strings the model's text could mean."""
    # _split_annotation already decided whether the tail names a second move,
    # judging it on the RAW text. Honour that verdict instead of re-deriving
    # it below: everything downstream reads _normalize'd text, where a color
    # word has become a single letter and a tail the guard refused can read
    # as harmless commentary. Recomputing a refusal on lossy text is how an
    # answer gets blocked by one code path and submitted by the next.
    text, blocked = _split_annotation(raw)
    if blocked:
        return []
    normalized = _normalize(text)

    match = _RAW_SLOT_RE.match(normalized)
    if match and _is_annotation(match.group(3)):
        verb, slot = match.group(1), int(match.group(2))
        return [f"({verb.capitalize()} {slot})"]

    match = _RAW_REVEAL_RE.match(normalized)
    if match:
        sign, number, rank, color, rest = match.groups()
    else:
        match = _RAW_REVEAL_TO_RE.match(normalized)
        if not match:
            return []
        rank, color, sign, number, rest = match.groups()
    if not _is_annotation(rest):
        return []
    offset = _resolve_offset(sign, int(number), legal_moves, player_id_offsets)
    if offset is None:
        return []
    # Rank words survive to here because _normalize no longer expands them;
    # the hint-value position is the one place a rank word is unambiguous.
    value = f"rank {_RANK_WORDS.get(rank, rank)}" if rank is not None else f"color {color.upper()}"
    return [f"(Reveal player +{offset} {value})"]


def _match_move_to_legal(
    raw: str,
    legal_moves: Sequence[str],
    player_id_offsets: Mapping[int, int] | None = None,
) -> str | None:
    if not raw:
        return None
    if raw in legal_moves:
        return raw
    # Before anything normalizes the sign away. Every route below runs through
    # _normalize, which drops the hyphen, so this has to gate all of them.
    if _NEGATIVE_SLOT_RE.match(raw.strip()):
        return None
    target = _normalize(raw)
    for legal in legal_moves:
        if _normalize(legal) == target:
            return legal
    legal_set = set(legal_moves)
    for candidate in _canonical_forms(raw, legal_moves, player_id_offsets):
        if candidate in legal_set:
            return candidate
    return None


# --- Public functions (called by main.py) -----------------------------------


def get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str]:
    """Return ``{action_id: action_string}`` for the current state."""
    legal_actions = observation.get("legalActions")
    legal_action_strings = observation.get("legalActionStrings")
    if legal_actions and legal_action_strings:
        return dict(zip(legal_actions, legal_action_strings))

    # The arena view publishes id + label pairs for the seat on the clock, and
    # only for that seat -- the legal hints against a hand are exactly the
    # colors and ranks IN it, so another seat's move list would spell out cards
    # the reader must not see.
    #
    # There is deliberately no serialized-state fallback below this. Every
    # other OpenSpiel harness deserializes `serializedGameAndState` when the
    # observation is thin, but `hanabi_arena` withholds that blob from agents
    # precisely because it reconstructs every hidden hand; reaching for it here
    # would be asking for the thing the env refused to send. If neither source
    # above has a move list, {} costs the turn -- which is the same cost the
    # fallback carried when the blob was unreadable.
    state = _parse_observation(observation)
    table_legals = (state.get("table") or {}).get("legal_actions")
    if table_legals:
        return {entry["action"]: entry["label"] for entry in table_legals}
    return {}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt from this player's own (partial) view."""
    del move_history  # Per-agent only; the table's full history is built below.
    state = _parse_observation(observation)
    table = state.get("table") or {}
    players_per_team = int(state.get("players_per_team", table.get("num_players", 2)) or 2)
    player_id = int(observation.get("playerId", state.get("your_player_id", 0)) or 0)
    team_id = int(state.get("your_team_id", player_id // players_per_team))
    seat = int(state.get("your_seat", player_id % players_per_team))
    team_base = team_id * players_per_team
    teammate_id = state.get("teammate_player_id")
    if teammate_id is None:
        teammate_id = team_base + (seat + 1) % players_per_team

    # Defaults matter here: with a malformed or absent observation the prompt
    # still renders, and every rule line it states must stay true of the game
    # actually being played. The engine's own defaults are therefore the only
    # safe fallbacks -- deck_total is the deck the composition line describes
    # (3 + 2*(ranks-2) + 1 per color, not colors*ranks), and hand_size falls
    # back to the engine's 5 rather than to however many cards happen to be
    # readable this turn. A wrong constant is worse than a missing one: the
    # model has no way to tell it was told the wrong deck.
    num_colors = int(table.get("colors", _DEFAULT_COLORS) or _DEFAULT_COLORS)
    num_ranks = int(table.get("ranks", _DEFAULT_RANKS) or _DEFAULT_RANKS)

    own_hand = _hand_of_seat(table, seat)
    own_cards = (own_hand or {}).get("cards") or []

    # The example action has to be legal notation at any hand size, and slot 0
    # exists whenever the player holds a card at all.
    example_move = "(Play 0)" if own_cards else "(Reveal player +1 rank 1)"

    history = _move_history_lines(table, team_base)

    max_score = num_colors * num_ranks
    prompt = HANABI_ARENA_PROMPT_TEMPLATE.format(
        players_per_team=players_per_team,
        num_colors=num_colors,
        color_list="/".join(_color_letters(table)) or "/".join(_DEFAULT_COLOR_LETTERS[:num_colors]),
        num_ranks=num_ranks,
        deck_total=table.get("deck_total", _deck_total(num_colors, num_ranks)),
        deck_composition=_deck_composition(num_ranks),
        hand_size=table.get("hand_size", _DEFAULT_HAND_SIZE),
        max_score=state.get("max_score", table.get("max_score", max_score)),
        max_info_tokens=table.get("max_info_tokens", _DEFAULT_MAX_INFO_TOKENS),
        max_life_tokens=table.get("max_life_tokens", _DEFAULT_MAX_LIFE_TOKENS),
        first_player_id=team_base,
        last_player_id=team_base + players_per_team - 1,
        player_id=player_id,
        team_id=team_id,
        seat=seat,
        teammate_id=teammate_id,
        fireworks_block=_render_fireworks(table),
        score=_readable(table, "score"),
        life_tokens=_readable(table, "life_tokens"),
        info_tokens=_readable(table, "info_tokens"),
        deck_size=_render_deck_size(table),
        final_turns=_render_final_turns(table.get("final_turns_remaining")),
        discards=_render_discards(table),
        own_hand_block=_render_own_hand(own_cards),
        other_hands_block=_render_other_hands(table, seat, players_per_team, team_base),
        move_history=_format_move_history(history),
        offset_map=_offset_map(seat, players_per_team, team_base),
        example_move=example_move,
    )

    # Three failure modes, not two. render_rethink_suffix partitions on
    # "was an action string extracted", which lumps "named an illegal move"
    # together with "named a legal move and then talked about another one" --
    # and the second is the common one for a model reasoning well about
    # conventions. Telling it the move was illegal sends it to re-examine the
    # board instead of its phrasing.
    if previous_action and _is_undecided_answer(previous_action):
        prompt += RETHINK_UNDECIDED.format(previous_action=previous_action)
    else:
        prompt += render_rethink_suffix(
            RETHINK_ILLEGAL,
            RETHINK_UNPARSABLE,
            previous_response,
            previous_action,
        )

    return prompt


def parse_response(
    response: str,
    legal_action_strings: Sequence[str],
    *,
    observation: Mapping[str, Any] | None = None,
) -> ParseResult:
    """Trust the model's JSON answer; let the rethink loop fix anything else.

    ``observation`` is optional and only sharpens one case: it supplies the
    arena player ids this seat's prompt printed, so a hint written as "Player
    3" resolves to the offset the engine wants. Without it the parser falls
    back to the offset-only reading, which is correct but stricter.
    """
    player_id_offsets = _player_id_offsets(observation)

    def matcher(raw: str, legals: Sequence[str]) -> str | None:
        return _match_move_to_legal(raw, legals, player_id_offsets)

    return parse_json_action(response, legal_action_strings, matcher=matcher)
