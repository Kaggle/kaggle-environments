"""LLM harness for OpenSpiel Go Fish.

Drop the body of this file into the notebook attached to the competition via
HarnessKernelId. The auto-generated ``main.py`` calls these three module-level
functions: ``get_legal_moves``, ``generate_prompt``, ``parse_response``.

Go Fish is an imperfect-information card game. Each player holds a hand of
cards drawn from a deck of ``ranks`` distinct ranks with ``suits`` copies of
each (the default is a standard 52-card deck: 13 ranks, 4 copies each). On your
turn you ASK another player for a rank you already hold at least one of. If
they have any cards of that rank they hand them all over and you ask again; if
they have none you "go fish" (draw one card from the pool) -- if the drawn card
is the rank you just asked for you take another turn, otherwise your turn ends.
Collecting all ``suits`` copies of a rank completes a "book"; the player with
the most books when the deck and all hands are exhausted wins.

The only decision a player makes is the ASK: which opponent to ask and for
which rank. The engine handles every draw and deal automatically as a chance
event, so this harness is only ever invoked on an Ask turn.

Action strings are two characters, ``"<target><letter>"``: the target player's
number followed by a rank letter. Ranks are letter-coded ``a,b,c,...`` in rank
order, so for the standard deck ``a``=A, ``b``=2, ..., ``i``=9, ``j``=10,
``k``=J, ``l``=Q, ``m``=K. For example ``"1a"`` means "ask Player 1 for Aces".
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import pyspiel

from kaggle_environments.core_harness import ParseResult, parse_json_action, render_rethink_suffix

_STANDARD_RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]


# --- Prompt -----------------------------------------------------------------


GO_FISH_PROMPT_TEMPLATE = """Let's play Go Fish.

Rules: The deck has {num_ranks} ranks with {num_suits} copies of each. On your
turn you ASK one other player for a rank -- you may only ask for a rank you
already hold at least one card of. If that player has any cards of that rank,
they give you all of them and you take another turn (ask again). If they have
none, you "go fish": you draw one card from the pool. If the drawn card is the
very rank you asked for, you take another turn; otherwise your turn ends.
Collecting all {num_suits} copies of a rank completes a book, which is set
aside and scored. The game ends when every card is in a book; the player with
the most books wins.

You only see information revealed to you: your own hand, every player's public
card count and book count, and the events that happened since your last turn.
You do NOT see other players' hands.

Rank letters: each rank is written as a letter in rank order -- {rank_legend}.

You are Player {player_id}. You currently have {my_books} book(s).

Your hand (rank: count, ask-letter):
{hand_lines}

Other players (public info):
{other_players}

Events since your last turn (oldest first):
{events}

Moves you have played so far: {move_history}

It is your turn to ask. Choose a target player who still has cards and a rank
you hold, then respond with your reasoning followed by your move in a JSON
block. The move is the two-character string "<target><letter>": the target
player's number followed by the ask-letter for the rank.

```json
{{
  "move": "<target><letter>"
}}
```

For example: `{{"move": "{example_move}"}}` (ask Player {example_target} for the
rank written "{example_letter}").

Failure to output your final answer in the specified format, or selecting an
illegal move, will result in a loss.
"""


RETHINK_ILLEGAL = """

You suggested move "{previous_action}" but this is not a legal move.
Remember: the target must be another player who still has cards, and the rank
letter must be one you hold in your hand. Re-read your hand and the ask-letters,
then pick a legal move.

(Keep using the same JSON output format as before -- only the move value needs to change.)
"""

RETHINK_UNPARSABLE = """

Your previous response ended with:
{previous_response}

No JSON answer could be parsed from that. Conclude your response with your final
move as JSON in a ```json fenced block, exactly as the original instructions
required:

```json
{{"move": "<target><letter>"}}
```

For example: `{{"move": "1a"}}` (ask Player 1 for the rank written "a").

The move you choose must also be legal: the target must be a player with cards,
and the letter must be a rank you hold.
"""


# --- Helpers ----------------------------------------------------------------


def _num_ranks(observation: Mapping[str, Any]) -> int:
    """Number of distinct ranks in this game's deck (default 13)."""
    serialized = observation.get("serializedGameAndState", "")
    if serialized:
        try:
            game, _ = pyspiel.deserialize_game_and_state(serialized)
            return int(game.get_parameters().get("ranks", 13))
        except Exception:  # noqa: BLE001 -- fall back to the standard deck
            pass
    return 13


def _num_suits(observation: Mapping[str, Any]) -> int:
    """Number of copies of each rank (a completed book needs all of them)."""
    serialized = observation.get("serializedGameAndState", "")
    if serialized:
        try:
            game, _ = pyspiel.deserialize_game_and_state(serialized)
            return int(game.get_parameters().get("suits", 4))
        except Exception:  # noqa: BLE001
            pass
    return 4


def _rank_label(rank_index: int, num_ranks: int) -> str:
    """Human label for a rank index, matching the proxy's mapping."""
    if num_ranks == 13 and 0 <= rank_index < 13:
        return _STANDARD_RANKS[rank_index]
    return chr(ord("a") + rank_index)


def _label_to_letter(num_ranks: int) -> dict[str, str]:
    """Map each rank's human label to its action letter (index order)."""
    return {_rank_label(i, num_ranks): chr(ord("a") + i) for i in range(num_ranks)}


def _parse_observation_payload(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the structured Go Fish state dict out of the observation."""
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
        try:
            _, state = pyspiel.deserialize_game_and_state(serialized)
            player = observation.get("playerId", 0)
            return json.loads(state.observation_string(player))
        except (json.JSONDecodeError, RuntimeError):
            pass
    return {}


def _format_hand(hand: Mapping[str, int], label_to_letter: Mapping[str, str]) -> str:
    """One line per rank held, sorted by rank order, annotated with letters."""
    if not hand:
        return "(you have no cards)"
    order = list(label_to_letter.keys())
    items = sorted(hand.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99)
    lines = []
    for label, count in items:
        letter = label_to_letter.get(label, label)
        lines.append(f"  {label}: {count}  (ask-letter '{letter}')")
    return "\n".join(lines)


def _format_other_players(players: Sequence[Mapping[str, Any]], player_id: int) -> str:
    """Card and book counts for every player other than us."""
    lines = []
    for p in players:
        pid = p.get("player")
        if pid == player_id:
            continue
        cards = p.get("cards", 0)
        books = p.get("books", 0)
        note = "" if cards > 0 else "  (no cards -- cannot be asked)"
        lines.append(f"  Player {pid}: {cards} card(s), {books} book(s){note}")
    return "\n".join(lines) if lines else "(none)"


def _format_events(events: Sequence[Mapping[str, Any]]) -> str:
    """Render recent events oldest-first (the proxy lists them newest-first)."""
    if not events:
        return "(nothing has happened since your last turn)"
    lines = []
    for ev in reversed(list(events)):
        player = ev.get("player")
        label = ev.get("rank_label")
        booked = ev.get("booked")
        if ev.get("type") == "draw":
            line = f"  Player {player} drew the {label} from the pool"
            if booked:
                line += f" and completed a book of {label}"
        else:
            target = ev.get("target")
            received = ev.get("received", 0)
            line = f"  Player {player} asked Player {target} for {label}"
            if received > 0:
                line += f" and received {received} card(s)"
                if booked:
                    line += f", completing a book of {label}"
            else:
                line += " and got none"
        lines.append(line)
    return "\n".join(lines)


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
    actions = state.legal_actions()
    return {a: state.action_to_string(a) for a in actions}


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current Go Fish state."""
    state = _parse_observation_payload(observation)
    player_id = observation.get("playerId", 0)

    num_ranks = _num_ranks(observation)
    num_suits = _num_suits(observation)
    label_to_letter = _label_to_letter(num_ranks)

    hand = state.get("hand") or {}
    players = state.get("players") or []
    events = state.get("recent_events") or []

    my_books = 0
    for p in players:
        if p.get("player") == player_id:
            my_books = p.get("books", 0)
            break

    rank_legend = ", ".join(f"{letter}={label}" for label, letter in label_to_letter.items())

    # A concrete example drawn from the model's own hand when possible, so the
    # format line is never illegal advice. Fall back to a generic ask.
    example_letter = "a"
    if hand:
        first_label = next((lbl for lbl in label_to_letter if lbl in hand), next(iter(hand)))
        example_letter = label_to_letter.get(first_label, "a")
    example_target = next((p.get("player") for p in players if p.get("player") != player_id), 1)
    example_move = f"{example_target}{example_letter}"

    move_history_str = ", ".join(move_history) if move_history else "None"

    prompt = GO_FISH_PROMPT_TEMPLATE.format(
        num_ranks=num_ranks,
        num_suits=num_suits,
        rank_legend=rank_legend,
        player_id=player_id,
        my_books=my_books,
        hand_lines=_format_hand(hand, label_to_letter),
        other_players=_format_other_players(players, player_id),
        events=_format_events(events),
        move_history=move_history_str,
        example_move=example_move,
        example_target=example_target,
        example_letter=example_letter,
    )

    prompt += render_rethink_suffix(
        RETHINK_ILLEGAL,
        RETHINK_UNPARSABLE,
        previous_response,
        previous_action,
    )

    return prompt


# --- Parsing ----------------------------------------------------------------


_STD_LABEL_TO_LETTER = {label: chr(ord("a") + i) for i, label in enumerate(_STANDARD_RANKS)}
_SEPARATORS_RE = re.compile(r"[\s,:;._\-]")


def _match_move_to_legal(raw: str, legal_action_strings: Sequence[str]) -> str | None:
    """Match a model's move to a legal ``<target><letter>`` action string.

    Tolerates common drift: uppercase letters (``"1A"``), stray separators
    (``"1, a"``, ``"1-a"``), and the model naming the rank by its human label
    instead of the action letter (``"1K"`` -> ask for Kings -> letter ``m``;
    ``"110"`` / ``"1 10"`` -> ask for tens -> letter ``j``). Lowercase single
    letters are treated as literal action letters; uppercase or multi-character
    rank tokens are treated as human rank labels.
    """
    compact = _SEPARATORS_RE.sub("", raw.strip())
    if not compact:
        return None

    # The engine encodes the target as a single character ('0'+target), so a
    # legal action string is always <single-digit-target><rank-letter>.
    m = re.match(r"^(\d)(.+)$", compact)
    if m:
        target, tok = m.group(1), m.group(2)
        # Human-label interpretation (numeric like "10", or uppercase like "K").
        if tok.isdigit() or tok.isupper() or len(tok) > 1:
            letter = _STD_LABEL_TO_LETTER.get(tok.upper())
            if letter is not None:
                matched = _default_match(f"{target}{letter}", legal_action_strings)
                if matched:
                    return matched
        # Literal action-letter interpretation (e.g. lowercase "a").
        matched = _default_match(compact, legal_action_strings)
        if matched:
            return matched

    return _default_match(compact, legal_action_strings)


def _default_match(raw: str, legals: Sequence[str]) -> str | None:
    """Case-insensitive, whitespace-stripped exact match against legals."""
    target = "".join(raw.split()).lower()
    if not target:
        return None
    for legal in legals:
        if "".join(legal.split()).lower() == target:
            return legal
    return None


def parse_response(
    response: str,
    legal_action_strings: Sequence[str],
) -> ParseResult:
    """Trust the model's JSON answer; let the rethink loop fix anything else."""
    return parse_json_action(
        response,
        legal_action_strings,
        matcher=_match_move_to_legal,
    )
