"""LLM harness for OpenSpiel Hanabi.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.

Hanabi is fully cooperative -- every seat receives the same return -- and
imperfect-information in an unusual shape: a player sees every hand *except*
their own. The proxy in ``hanabi_proxy.py`` emits one JSON observation per
player reflecting exactly that, so the prompt renders other players' cards
face-up and the agent's own hand as knowledge only (what it has been told, and
what remains plausible after every hint's negative information).

Hint knowledge is public -- every player hears every hint -- so the whole move
history can be shown to the model without leaking anything. The proxy does not
expose a history field, so it is reconstructed by replaying
``serializedGameAndState`` and annotating each decision with the card it
touched (all of which becomes public the moment the action resolves).
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult, parse_json_action, render_rethink_suffix

# --- Prompt -----------------------------------------------------------------


HANABI_PROMPT_TEMPLATE = """Let's play Hanabi, a fully cooperative card game. All \
{num_players} players share one score; you all win or lose together.

Rules:
- The deck has {num_colors} colors ({color_list}) and ranks 1-{num_ranks}, \
{deck_total} cards in total: per color, {deck_composition}.
- Each player holds up to {hand_size} cards{visibility_rule}. Once the deck is \
empty there are no replacements, so hands shrink as cards are played or \
discarded.
- The team builds one firework stack per color in ascending order 1 to \
{num_ranks}. The score is the sum of the stack heights, at most {max_score}.
- On your turn you take exactly one action:
  * Play a card from one of your slots. If its rank is exactly one above its \
color's current stack height it joins that stack; otherwise it is discarded \
and the team loses a life token.
  * Discard a card from one of your slots. It is lost and the team regains one \
info token. Illegal while info tokens are at the maximum of {max_info_tokens}.
  * Hint another player one color or one rank. Every card of that color/rank in \
their hand is pointed out; the rest are thereby excluded from it. Costs one \
info token, so it is illegal at 0 info tokens, and the color or rank you name \
must match at least one card in that player's hand.
- Every hint is public: all players hear who was told what.
- After you play or discard, your remaining cards shift down one slot and, if \
the deck is not empty, a replacement is drawn into your highest slot. Slot 0 is \
therefore always your oldest card.
- Completing a color's stack at rank {num_ranks} regains one info token. Info \
tokens never exceed {max_info_tokens}.
- The game ends when the team loses its last life token (the score becomes 0 \
no matter how many fireworks were played), when every stack is complete \
({max_score} points), or when the deck runs out -- after the deck empties each \
player takes exactly one more turn.
- Turns pass in seat order, wrapping from Player {last_seat} back to Player 0.

Fireworks:
{fireworks_block}
Tokens: {life_tokens}/{max_life_tokens} life, {info_tokens}/{max_info_tokens} info
Deck: {deck_size} card(s) left{final_turns}
Discarded: {discards}

You are Player {player_id}. Your own hand{own_hand_caveat}
{own_hand_block}
{other_hands_block}
Moves played so far (all players, oldest first):
{move_history}

Action notation -- your answer must be exactly one of these forms:
  (Play N)                     play the card in your slot N
  (Discard N)                  discard the card in your slot N
  (Reveal player +K color C)   hint color C to the player K seats after you
  (Reveal player +K rank R)    hint rank R to the player K seats after you
Slots are numbered from 0. Colors are the single letters {color_list}. Hint \
targets are written as an offset from your own seat: {offset_map}.

It is your turn. Reason step by step about what your teammates know, what your \
own cards could be, and which action helps the team most, then give your final \
answer in a JSON block:

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
    """Pull the JSON state emitted by ``HanabiState.observation_string``."""
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


def _color_letters(state: Mapping[str, Any]) -> list[str]:
    """Color letters in engine order, from the fireworks the engine reported."""
    fireworks = state.get("fireworks") or {}
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


def _render_own_hand(cards: Sequence[Mapping[str, Any]]) -> str:
    """Own slots as knowledge only -- unless the engine has revealed them.

    Under ``observation_type=seer`` the engine shows the observer their own
    cards face-up, and the proxy passes that through. Dropping the face then
    would hide from the model information the game gave it.
    """
    if not cards:
        return "  (no cards)"
    lines = []
    for i, card in enumerate(cards):
        face = card.get("card")
        prefix = f"  slot {i}: " + (f"{_card_text(face)} -- " if face else "")
        lines.append(f"{prefix}told {_told_text(card)}; possible {_possible_text(card)}")
    return "\n".join(lines)


def _render_other_hands(state: Mapping[str, Any], player_id: int, num_players: int) -> str:
    """Every other seat's cards face-up, labelled with its hint offset."""
    blocks = []
    for offset in range(1, num_players):
        seat = (player_id + offset) % num_players
        hand = next((h for h in state.get("hands") or [] if h.get("player") == seat), None)
        cards = (hand or {}).get("cards") or []
        header = f"Player {seat}'s hand (hint target +{offset}):"
        if not cards:
            blocks.append(f"{header}\n  (no cards)")
            continue
        lines = [
            f"  slot {i}: {_card_text(c.get('card'))} -- told {_told_text(c)}; possible {_possible_text(c)}"
            for i, c in enumerate(cards)
        ]
        blocks.append(header + "\n" + "\n".join(lines))
    return "\n".join(blocks)


def _render_fireworks(state: Mapping[str, Any]) -> str:
    fireworks = state.get("fireworks") or {}
    num_ranks = int(state.get("ranks", 5) or 5)
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


def _render_discards(state: Mapping[str, Any]) -> str:
    """Discards grouped as ``R1 x2`` so duplicate-tracking is readable."""
    discards = state.get("discards") or []
    if not discards:
        return "(none)"
    counts: dict[str, int] = {}
    for card in discards:
        counts[_card_text(card)] = counts.get(_card_text(card), 0) + 1
    order = {letter: i for i, letter in enumerate(_color_letters(state))}
    tokens = sorted(counts.items(), key=lambda kv: (order.get(kv[0][0], 99), kv[0][1:]))
    return ", ".join(name if n == 1 else f"{name} x{n}" for name, n in tokens)


def _offset_map(player_id: int, num_players: int) -> str:
    return ", ".join(f"+{offset} = Player {(player_id + offset) % num_players}" for offset in range(1, num_players))


# --- Move history -----------------------------------------------------------


_PLAY_ACTION_RE = re.compile(r"^\(Play (\d+)\)$")
_DISCARD_ACTION_RE = re.compile(r"^\(Discard (\d+)\)$")
_REVEAL_COLOR_ACTION_RE = re.compile(r"^\(Reveal player \+(\d+) color (\w)\)$")
_REVEAL_RANK_ACTION_RE = re.compile(r"^\(Reveal player \+(\d+) rank (\d+)\)$")


def _hand_of(state: Any, owner: int, num_players: int) -> list[dict[str, Any]] | None:
    """Face-up cards in ``owner``'s hand, read from a teammate's view.

    A player's own view hides their own cards, so the hand is read through the
    next seat -- which sees it in full. Returns ``None`` when the hand cannot
    be read at all (the state is not a proxy state, so it has no
    ``state_dict``), as distinct from ``[]`` for a genuinely empty hand. The
    two must not collapse: an unreadable hand means "we do not know which
    slots a hint touched", while an empty one means "none", and rendering the
    first as the second states a falsehood about a public fact.
    """
    state_dict = getattr(state, "state_dict", None)
    if state_dict is None:
        return None
    for hand in state_dict((owner + 1) % num_players).get("hands") or []:
        if hand.get("player") == owner:
            return list(hand.get("cards") or [])
    return None


def _describe_decision(before: Any, after: Any, player: int, label: str, num_players: int) -> str:
    """One history line: the action plus the public facts it revealed."""
    match = _PLAY_ACTION_RE.match(label) or _DISCARD_ACTION_RE.match(label)
    if match:
        slot = int(match.group(1))
        hand = _hand_of(before, player, num_players) or []
        card = _card_text(hand[slot].get("card")) if slot < len(hand) else "??"
        if label.startswith("(Discard"):
            return f"P{player} discarded slot {slot} ({card})"
        before_dict = getattr(before, "state_dict", None)
        after_dict = getattr(after, "state_dict", None)
        if before_dict is None or after_dict is None:
            return f"P{player} played slot {slot} ({card})"
        # Stack heights, not the banked score: score is zeroed by a bomb-out,
        # which would read as "no firework advanced" on the very play that
        # caused it. Same field the visualizer uses for this question.
        if after_dict(0).get("fireworks_total", 0) > before_dict(0).get("fireworks_total", 0):
            return f"P{player} played slot {slot} ({card}) -- firework advanced"
        return f"P{player} played slot {slot} ({card}) -- misplay, life lost"

    match = _REVEAL_COLOR_ACTION_RE.match(label)
    kind, value = None, None
    if match:
        kind, value = "color", match.group(2)
    else:
        match = _REVEAL_RANK_ACTION_RE.match(label)
        if match:
            kind, value = "rank", int(match.group(2))
    if match and kind is not None:
        target = (player + int(match.group(1))) % num_players
        hand = _hand_of(before, target, num_players)
        if hand is None:
            # Which slots a hint pointed at is the substance of the hint; a
            # guess is worse than an omission. "no slots" would also be a lie
            # -- the engine only allows hints that match at least one card.
            return f"P{player} hinted P{target} {kind} {value}"
        touched = [str(i) for i, c in enumerate(hand) if (c.get("card") or {}).get(kind) == value]
        noun = "slot" if len(touched) == 1 else "slots"
        slots = f"{noun} {', '.join(touched)}" if touched else "no slots"
        return f"P{player} hinted P{target} {kind} {value} -- {slots}"

    return f"P{player} {label}"


def _build_move_history(observation: Mapping[str, Any], num_players: int) -> list[str] | None:
    """Reconstruct every player's moves by replaying the serialized state.

    The proxy exposes no history field, so the full game is replayed from the
    root. Chance nodes (the deal and each draw) are applied but not reported --
    they are not decisions any player made, and their outcomes are already
    visible in the rendered hands.

    Returns ``None`` if the replay is unavailable (no serialized state, or a
    deserialize the local pyspiel cannot do). That is distinct from ``[]``,
    which means the game genuinely has no moves yet: rendering a failed
    replay as "this is the first turn" would tell a model forty moves deep
    that nothing had happened, and it has no way to detect the lie.
    """
    serialized = observation.get("serializedGameAndState")
    if not serialized:
        return None
    try:
        game, final = pyspiel.deserialize_game_and_state(serialized)
    except (pyspiel.SpielError, RuntimeError, ValueError):
        return None
    replay = game.new_initial_state()
    lines: list[str] = []
    for item in final.full_history():
        if item.player < 0:
            replay.apply_action(item.action)
            continue
        label = replay.action_to_string(item.player, item.action)
        before = replay.clone()
        replay.apply_action(item.action)
        lines.append(_describe_decision(before, replay, item.player, label, num_players))
    return lines


def _render_final_turns(remaining: int | None) -> str:
    if remaining is None:
        return ""
    if remaining == 1:
        return " -- final round: this is the last turn of the game"
    return f" -- final round: {remaining} turns remain, including yours"


def _format_move_history(lines: Sequence[str] | None) -> str:
    """Render the replayed history, or say plainly that it is unavailable.

    ``None`` (the replay failed) and ``[]`` (nobody has moved yet) must render
    differently: a model told "this is the first turn" forty moves in has no
    way to notice, whereas one told the log is missing knows to fall back on
    the hint knowledge rendered above.
    """
    if lines is None:
        return "  (unavailable -- the move log could not be reconstructed this turn)"
    if not lines:
        return "(none yet -- this is the first turn)"
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
#   turns "Play B3" into slot 3, an action the model never chose.
# - A hint target written without a "+" is ambiguous between an offset and an
#   absolute seat. It is only resolved when the legal set leaves exactly one
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
# One lookup table; the split above is documentation, not precedence.
_VALUE_WORDS = {**_COLOR_WORDS, **_RANK_WORDS}

# Words a model may put between the verb and the slot number. Deliberately
# closed: a color letter must NOT be absorbed here.
_FILLER = r"(?:slot|card|my|the|from|in|number)"

_RAW_SLOT_RE = re.compile(rf"^(play|discard)(?:\s+{_FILLER})*\s+(\d+)\b(.*)$")

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
    rf"^reveal (?:player\s*|p)?(\+?)(\d+)(?:\s+{_HINT_FILLER})*\s+(?:(?:rank )?(\d+)s?|(?:colou?r )?([a-z]))\b(.*)$"
)
# "Reveal color R to player +1" -- the same move with target and value
# swapped. The literal "to" is required: without it "reveal r 1" is ambiguous
# between this order and the one above.
_RAW_REVEAL_TO_RE = re.compile(
    r"^reveal (?:(?:rank )?(\d+)s?|(?:colou?r )?([a-z]))\s+to\s+(?:player\s*|p)?(\+?)(\d+)\b(.*)$"
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
# _is_annotation runs on the RAW tail (before _normalize lowercases anything),
# so "Play 0 -- Or maybe Discard 1" must be caught as readily as its lowercase
# twin.
#
# Matching a verb ALONE would be wrong here, because Hanabi's strategic
# vocabulary is its verb list: "Play 0 (safest play)", "(better than a hint
# right now)", "(sets up a play)" and "(this clue is best)" are commentary on
# the chosen move, not a second one. A second action needs an operand -- a
# slot number after play/discard, or a target or value after a hint verb.
_SECOND_ACTION_RE = re.compile(
    rf"\b(?:play|discard)\s+(?:{_FILLER}\s+)*\#?\d"
    r"|\b(?:reveal|hint|tell|clue)\s+(?:player\s*|p)?"
    r"(?:\+?\d|colou?rs?\b|ranks?\b|red\b|yellow\b|green\b|white\b|blue\b)",
    re.IGNORECASE,
)

# An undecided tail: either it is nothing but connectors and numbers ("or 1",
# "/ 4", ", 1"), or it opens with a connector and reaches a number ("or maybe
# 1"). Both name alternative slots rather than describing the chosen card.
#
# The first branch is also what refuses a bare-digit tail ("Play 0-1",
# "Play 0, 1"): a separator followed by nothing but a digit reads at least as
# naturally as a second slot than as the rank of the card in the first, and
# guessing would submit a move the model did not settle on.
#
# Both branches are anchored, which is the whole point. An unanchored "or" or
# "/" fires on ordinary Hanabi prose -- "(it is W1 or B1)", "(2/3 chance)",
# "(dead card, R/Y both done)" -- and rejecting those costs a rethink for a
# model that did exactly what it was asked.
_UNDECIDED_RE = re.compile(
    r"^(?:\W|\d|\bor\b)+$|^\W*(?:or\b|/)[^\d]*\d|\b(?:instead|alternatively)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """Lowercase, unwrap punctuation, and canonicalize verbs and value words."""
    cleaned = re.sub(r"[()\[\]{}\"',.:\-_#]+", " ", text.lower())
    tokens = cleaned.split()
    if tokens:
        tokens[0] = _VERB_ALIASES.get(tokens[0], tokens[0])
    tokens = [_VALUE_WORDS.get(token, token) for token in tokens]
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
    if _SECOND_ACTION_RE.search(rest):
        return False
    return not _UNDECIDED_RE.search(rest)


def _strip_annotation(raw: str) -> str:
    """Drop a trailing note like " (likely W1)" or " - my W1".

    Only commentary is dropped. If the tail names another action the text is
    returned untouched, so an undecided answer fails to match and the model
    gets a rethink instead of whichever half happened to come first.
    """
    text = raw.strip()
    for _ in range(2):
        wrapped = _WRAPPED_RE.match(text)
        if wrapped:
            text = wrapped.group(1).strip()
        match = _ANNOTATION_RE.search(text)
        if not match or not _is_annotation(match.group(1)):
            break
        text = text[: match.start()].strip()
    return text


def _reveal_offsets(legal_moves: Sequence[str]) -> set[int]:
    """Hint offsets that are legal right now, read off the engine's labels.

    Reuses the history renderer's patterns so one engine label format is
    parsed by one pair of regexes; both capture the offset in group 1.
    """
    matches = (_REVEAL_COLOR_ACTION_RE.match(move) or _REVEAL_RANK_ACTION_RE.match(move) for move in legal_moves)
    return {int(m.group(1)) for m in matches if m}


def _resolve_offset(sign: str, number: int, legal_moves: Sequence[str]) -> int | None:
    """Turn a written hint target into an offset, or ``None`` if ambiguous.

    An explicit "+K" is an offset by construction. A bare "K" could be either
    an offset or an absolute seat -- different teammates from three players
    up, where guessing wrong hints the wrong one. It is accepted only when
    exactly one target is hintable at all and "K" is that offset.

    The two readings still name the same seat only when the writer sits at
    seat 0. What that guard actually buys is weaker but sufficient: whenever
    they diverge, the absolute reading needs an offset that is *not* in the
    legal set, so it was never an available move. (Proof by exhaustion over
    every seat and target for 2-5 players: divergence requires
    ``(K - me) % n != K``, and the legal set is ``{K}``, so the offset the
    absolute reading needs is illegal.) Resolving to the one hintable target
    is therefore a charitable reading of an otherwise-illegal move, not a
    coin-flip between two valid ones -- and when two targets *are* hintable
    the ambiguity is real and this returns ``None``.
    """
    if sign == "+":
        return number
    offsets = _reveal_offsets(legal_moves)
    return number if offsets == {number} else None


def _canonical_forms(raw: str, legal_moves: Sequence[str]) -> list[str]:
    """Engine-shaped action strings the model's text could mean."""
    normalized = _normalize(_strip_annotation(raw))

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
    offset = _resolve_offset(sign, int(number), legal_moves)
    if offset is None:
        return []
    value = f"rank {int(rank)}" if rank is not None else f"color {color.upper()}"
    return [f"(Reveal player +{offset} {value})"]


def _match_move_to_legal(raw: str, legal_moves: Sequence[str]) -> str | None:
    if not raw:
        return None
    if raw in legal_moves:
        return raw
    target = _normalize(raw)
    for legal in legal_moves:
        if _normalize(legal) == target:
            return legal
    legal_set = set(legal_moves)
    for candidate in _canonical_forms(raw, legal_moves):
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

    # The proxy already publishes id + label pairs; prefer them over paying for
    # a full deserialize.
    state = _parse_observation(observation)
    proxy_legals = state.get("legal_actions")
    if proxy_legals:
        return {entry["action"]: entry["label"] for entry in proxy_legals}

    serialized = observation.get("serializedGameAndState", "")
    if not serialized:
        return {}
    try:
        _, os_state = pyspiel.deserialize_game_and_state(serialized)
    except (pyspiel.SpielError, RuntimeError, ValueError):
        # The serialized blob names the proxy game ("hanabi_proxy"), so this
        # raises wherever that game is not registered. Returning {} lets the
        # framework report "no legal actions" for the turn; letting the
        # SpielError escape would void the whole episode.
        return {}
    if os_state.is_terminal() or os_state.is_chance_node():
        return {}
    player_id = observation.get("playerId", os_state.current_player())
    # Per observer, not the bare call: the legal hints against a hand are
    # exactly the colors and ranks IN it, so the actor's move list handed to a
    # non-actor spells out that non-actor's own cards. The engine returns []
    # for a non-actor, which is what we want here too.
    return {a: os_state.action_to_string(player_id, a) for a in os_state.legal_actions(player_id)}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt from this player's own (partial) view."""
    del move_history  # Per-agent only; the full history is rebuilt below.
    state = _parse_observation(observation)
    player_id = int(observation.get("playerId", state.get("observer", 0)) or 0)
    num_players = int(state.get("num_players", 2) or 2)
    num_colors = int(state.get("colors", 5) or 5)
    num_ranks = int(state.get("ranks", 5) or 5)

    own_hand = next((h for h in state.get("hands") or [] if h.get("player") == player_id), None)
    own_cards = (own_hand or {}).get("cards") or []

    # The example action has to be legal notation at any hand size, and slot 0
    # exists whenever the player holds a card at all.
    example_move = "(Play 0)" if own_cards else "(Reveal player +1 rank 1)"

    # Under observation_type=seer the engine hands the observer their own
    # cards face-up. That is a different game -- claiming otherwise would be
    # telling the model it is blind while showing it the answer.
    sees_own_hand = any(card.get("card") for card in own_cards)
    knowledge_gloss = (
        '("told" is what hints have said about the card out loud; "possible" is what is still '
        "consistent with every hint, including hints that skipped this card)"
    )
    if sees_own_hand:
        visibility_rule = " and can see every player's cards, including their own"
        own_hand_caveat = ", which you can see face-up in this variant:"
    else:
        visibility_rule = " and can see every other player's cards but not their own"
        own_hand_caveat = f", which you cannot see {knowledge_gloss}:"

    prompt = HANABI_PROMPT_TEMPLATE.format(
        num_players=num_players,
        visibility_rule=visibility_rule,
        own_hand_caveat=own_hand_caveat,
        num_colors=num_colors,
        color_list="/".join(_color_letters(state)) or "?",
        num_ranks=num_ranks,
        deck_total=state.get("deck_total", num_colors * num_ranks),
        deck_composition=_deck_composition(num_ranks),
        hand_size=state.get("hand_size", len(own_cards)),
        max_score=state.get("max_score", num_colors * num_ranks),
        max_info_tokens=state.get("max_info_tokens", 8),
        max_life_tokens=state.get("max_life_tokens", 3),
        last_seat=num_players - 1,
        fireworks_block=_render_fireworks(state),
        life_tokens=state.get("life_tokens", 0),
        info_tokens=state.get("info_tokens", 0),
        deck_size=state.get("deck_size", 0),
        final_turns=_render_final_turns(state.get("final_turns_remaining")),
        discards=_render_discards(state),
        player_id=player_id,
        own_hand_block=_render_own_hand(own_cards),
        other_hands_block=_render_other_hands(state, player_id, num_players),
        move_history=_format_move_history(_build_move_history(observation, num_players)),
        offset_map=_offset_map(player_id, num_players),
        example_move=example_move,
    )

    prompt += render_rethink_suffix(
        RETHINK_ILLEGAL,
        RETHINK_UNPARSABLE,
        previous_response,
        previous_action,
    )

    return prompt


def parse_response(response: str, legal_action_strings: Sequence[str]) -> ParseResult:
    """Trust the model's JSON answer; let the rethink loop fix anything else."""
    return parse_json_action(response, legal_action_strings, matcher=_match_move_to_legal)
