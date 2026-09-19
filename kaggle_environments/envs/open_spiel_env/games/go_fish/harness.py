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
they give you all of them and you take another turn (ask again) -- unless that
leaves every other player with no cards, in which case there is nobody left to
ask and play passes on. If they have none, you "go fish": you draw one card from
the pool. If the drawn card is the rank you asked for, you take another turn;
otherwise your turn ends. If the pool is empty when you get a miss, there is no
card to draw and your turn simply ends. Collecting all {num_suits} copies of a
rank completes a book, which is set aside and scored. Running out of cards does
NOT eliminate you: if it is your turn with an empty hand you draw a card from
the pool and ask with it, and you rejoin normally whenever a card comes your
way. But if your hand AND the pool are both empty when your turn comes up, there
is nothing to draw and no card to ask with, so play skips past you to the next
player who still holds cards. The game ends when every card is in a book; the
player with the most books wins.

You only see information revealed to you: your own hand, every player's public
card count and book count, the size of the pool, which ranks are already booked,
and the events that happened since your last turn. You do NOT see other players'
hands, and you do NOT see which card anyone draws from the pool.

Rank letters: each rank is written as a letter in rank order -- {rank_legend}.

You are Player {player_id}. You currently have {my_books} book(s).

Your hand (rank: count, ask-letter):
{hand_lines}

Pool: {pool_line}

Ranks already booked (gone from play -- nobody can be asked for these):
{booked_line}

Other players (public info):
{other_players}

What you know about opponents' cards (deduced from every ask so far -- asks are
public: asking reveals the asker holds that rank, and after any ask the target
holds none of it, whether they handed cards over or had none to give):
{deductions}

Events since your last turn (oldest first):
{events}

Your own past asks and how each turned out ("received N" = they handed over N
cards of that rank; "go fish" = they had none): {move_history}

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
rank written "a"). This is only a format illustration, not a suggestion -- it is
not necessarily legal here.

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

For example: `{{"move": "{example_move}"}}` (ask Player {example_target} for the
rank written "a").

The move you choose must also be legal: the target must be a player with cards,
and the letter must be a rank you hold.
"""


# --- Helpers ----------------------------------------------------------------


def _deck_params(observation: Mapping[str, Any], state: Mapping[str, Any]) -> tuple[int, int]:
    """Return ``(num_ranks, num_suits)`` for this game's deck (default 13x4).

    Prefer the values the proxy already embeds in the observation payload so we
    don't deserialize the game a second time; fall back to the serialized game
    only when the payload lacks them (e.g. a hand-built observation in tests).
    """
    num_ranks = state.get("num_ranks")
    num_suits = state.get("num_suits")
    if isinstance(num_ranks, int) and isinstance(num_suits, int):
        return num_ranks, num_suits

    serialized = observation.get("serializedGameAndState", "")
    if serialized:
        try:
            game, _ = pyspiel.deserialize_game_and_state(serialized)
            params = game.get_parameters()
            return int(params.get("ranks", 13)), int(params.get("suits", 4))
        except (RuntimeError, ValueError):  # malformed serialization -> defaults
            pass
    return 13, 4


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
        except (json.JSONDecodeError, RuntimeError, ValueError):
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


def _example_move(player_id: int) -> tuple[str, int]:
    """Return ``(example_move, example_target)`` for the prompt's format example.

    The target must not be the observer: ``GenerateAsks`` skips
    ``target == player_id`` (go_fish.cc), so a self-ask is never legal. A single
    hardcoded ``"1a"`` was therefore structurally impossible for Player 1 --
    about half of all turns in a 2-player game -- which teaches the format with
    a move the model could never legally make.

    Only the *target* varies; the rank letter stays a fixed ``"a"``. Deriving the
    letter from the hand would turn the example back into a per-turn suggestion
    (the bug that motivated making it static in the first place) and would leak
    hand contents into a section that is meant to be pure format illustration.
    ``"a"`` is a valid letter for any deck with at least one rank, so the example
    is always well-formed even when the observer happens not to hold that rank.
    """
    target = 0 if player_id != 0 else 1
    return f"{target}a", target


def _format_pool(pool_size: int | None) -> str:
    """Describe the pool, spelling out what an empty pool means for a miss.

    Pool size drives the two rules the model most often gets wrong: a miss only
    draws a card while the pool has one, and the game is over when the pool and
    every hand are exhausted. The consequence is stated inline rather than left
    for the model to connect back to the rules paragraph.
    """
    if pool_size is None:
        return "(unknown)"
    if pool_size <= 0:
        return "empty -- a miss draws nothing and simply ends your turn"
    return f"{pool_size} card(s) left to draw"


def _format_booked(booked: Sequence[str]) -> str:
    """List ranks already scored into books, which are dead for asking."""
    if not booked:
        return "  (none yet)"
    return "  " + ", ".join(booked)


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


def _format_deductions(deductions: Sequence[Mapping[str, Any]], player_id: int) -> str:
    """Render the game-long public deduction table for every opponent.

    This is the accumulated common knowledge from *all* asks so far -- not just
    the events since our last turn. It is the game's core deduction signal:
    every ask publicly reveals that the asker holds a rank and (on a miss) that
    the target holds none of it. ``recent_events`` alone would drop everything
    older than the current observation window, so we surface the distilled
    standing facts here.

    ``wanted`` is filtered against the ranks already in ``known_has``. The proxy
    emits both, and ~93% of ``wanted`` entries name a rank the same row already
    reports a count for -- "known to hold 9>=3; has asked for 9" says nothing
    the first clause did not say more precisely. This is the densest line in the
    prompt, so the duplication is most of what the model reads there.

    What survives the filter is a distinct fact, not a weaker restatement: a
    rank the player was asked for and emptied of, but has drawn since, so they
    may hold it again. ``known_has`` cannot express that -- its floor is 0 -- and
    the ask is what makes the maybe worth acting on.

    That reading is structural, not just observed, which is why it is safe to
    state as a gloss: asking for a rank publicly sets ``player_min`` to >=1, so
    a surviving entry (``did_ask > 0`` with no ``known_has``) can only mean the
    floor fell back to 0, i.e. they were emptied of it. Being emptied requires
    having been asked, and the proxy already drops the entry as ``known_void``
    unless they have drawn since. Booking the rank themselves is the other way
    the floor drops, and the proxy retires booked ranks first. Confirmed with
    zero exceptions across the default and three off-default configs.

    Filtering here rather than in the proxy: these entries are true, merely
    redundant, so dropping them is presentation. The proxy's two filters retire
    facts the public record has *invalidated*, and its output is pinned to the
    observation tensor by a parity test.
    """
    lines = []
    for d in deductions:
        pid = d.get("player")
        if pid == player_id:
            continue
        parts = []
        known_has = d.get("known_has") or []
        known_void = d.get("known_void") or []
        # known_has entries are "<label>>=<n>"; compare on the label alone.
        has_labels = {str(h).split(">=", 1)[0] for h in known_has}
        wanted = [w for w in (d.get("wanted") or []) if w not in has_labels]
        if known_has:
            parts.append(f"known to hold {', '.join(known_has)}")
        if known_void:
            parts.append(f"known to have none of {', '.join(known_void)}")
        if wanted:
            parts.append(f"has asked for {', '.join(wanted)} (emptied of it, but has drawn since)")
        detail = "; ".join(parts) if parts else "nothing deduced yet"
        lines.append(f"  Player {pid}: {detail}")
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
            # A drawn card's rank is hidden information -- naming it here would
            # leak an opponent's hand, contradicting the prompt's "you do NOT
            # see other players' hands" and mooting the deduction block. The
            # engine's canonical public encoding (go_fish.cc observation tensor)
            # likewise records only that a draw happened, never its rank. Only a
            # completed book is genuinely public, so keep that clause.
            line = f"  Player {player} drew a card from the pool"
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


_OWN_COUNTS_RE = re.compile(r"^player (\d+) cards (\d+) books (\d+)", re.MULTILINE)


def _own_counts(obs_text: str, player_id: int) -> tuple[int, int]:
    """Read ``(total_cards, books)`` for ``player_id`` from an observation string.

    Two formats reach this function, and it must handle both:

    * The proxy's JSON payload, which carries a ``players`` list of
      ``{"player", "cards", "books"}``. This is the production format --
      ``open_spiel_env`` swaps in ``go_fish_proxy`` before serializing, so a
      state replayed out of ``serializedGameAndState`` is a proxy state whose
      ``observation_string`` is JSON, not OpenSpiel's text.
    * OpenSpiel's raw text, which carries a ``player <id> cards <c> books <b>``
      line per player. Reachable when the harness is pointed at an unwrapped
      ``go_fish`` game.

    Reading only the text form silently returned the ``(0, 0)`` default on every
    production call, which made every own-ask delta zero and mislabelled every
    hit as "go fish".
    """
    try:
        payload = json.loads(obs_text)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(payload, dict):
            for p in payload.get("players") or []:
                if p.get("player") == player_id:
                    return int(p.get("cards", 0)), int(p.get("books", 0))
            return 0, 0
    for m in _OWN_COUNTS_RE.finditer(obs_text):
        if int(m.group(1)) == player_id:
            return int(m.group(2)), int(m.group(3))
    return 0, 0


def _annotate_move_history(
    observation: Mapping[str, Any],
    move_history: Sequence[str],
    player_id: int,
    num_suits: int,
) -> list[str] | None:
    """Attach the outcome of each of the observer's own asks to its move string.

    The model is never shown the result of its OWN asks: OpenSpiel's observation
    only lists opponent events since your last turn, and it stops the event walk
    at your own most recent action (go_fish.cc ObservationString), so a player's
    asks never appear in their own ``recent_events``. Without this the model asks
    into the void -- it can't tell a hit from a "go fish" after the fact.

    We recover each own-ask outcome deterministically by replaying the game's
    full action history (recoverable from ``serializedGameAndState``) and reading
    the observer's own card/book delta across just that ask:

        received = (cards_after - cards_before) + num_suits * (books_after - books_before)

    The ``num_suits`` correction is essential: a hit that completes a book zeroes
    that rank out of the hand, so a naive card delta would under-count by exactly
    one book's worth of cards. ``booked`` is simply ``books_after > books_before``.

    ``move_history`` is treated as a SUFFIX of the observer's asks -- it ends at
    the most recent one but need not start at the first -- so outcomes are
    matched from the end.

    Returns an annotated copy of ``move_history`` (bare strings for asks whose
    outcome could not be reconstructed), or ``None`` if there is no serialized
    state to replay or the two sequences cannot be aligned -- callers fall back
    to the bare history in that case.
    """
    serialized = observation.get("serializedGameAndState", "")
    if not serialized or not move_history:
        return None
    try:
        game, _ = pyspiel.deserialize_game_and_state(serialized)
        _, final_state = pyspiel.deserialize_game_and_state(serialized)
        history = final_state.history()
        replay = game.new_initial_state()
    except (RuntimeError, ValueError):
        return None

    outcomes: list[str | None] = []
    for action in history:
        is_chance = replay.is_chance_node()
        mover = None if is_chance else replay.current_player()
        if mover == player_id:
            before = _own_counts(replay.observation_string(player_id), player_id)
            try:
                replay.apply_action(action)
            except (RuntimeError, ValueError):
                return None
            after = _own_counts(replay.observation_string(player_id), player_id)
            received = (after[0] - before[0]) + num_suits * (after[1] - before[1])
            booked = after[1] > before[1]
            if received > 0:
                note = f"received {received}" + (", completed a book" if booked else "")
            else:
                note = "go fish"
            outcomes.append(note)
        else:
            try:
                replay.apply_action(action)
            except (RuntimeError, ValueError):
                return None

    # ``move_history`` holds only the observer's own asks, in order, and ends at
    # the most recent one -- so it is a SUFFIX of ``outcomes``, not necessarily
    # the whole thing. Anchor from the end: a positional zip from index 0 would
    # pair move_history[0] with the observer's first ask of the game, silently
    # attributing every entry to the wrong ask. That is worse than showing
    # nothing, because the model builds its opponent model on inverted data with
    # no signal that anything is off. A short move_history is reachable whenever
    # the agent process restarts mid-episode (create_agent_fn's move_history
    # closure starts empty while the engine state is mid-game), or if a caller
    # ever windows the history to bound prompt length.
    if len(move_history) > len(outcomes):
        # More moves than the replay accounts for (e.g. a rethink appended a
        # move the engine never applied). The two sequences cannot be anchored
        # at either end, so there is no alignment to trust -- fall back to the
        # bare history rather than guess.
        return None
    offset = len(outcomes) - len(move_history)
    return [f"{mv} ({outcomes[offset + i]})" if outcomes[offset + i] else mv for i, mv in enumerate(move_history)]


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current Go Fish state."""
    state = _parse_observation_payload(observation)
    player_id = observation.get("playerId", 0)

    num_ranks, num_suits = _deck_params(observation, state)
    label_to_letter = _label_to_letter(num_ranks)

    hand = state.get("hand") or {}
    players = state.get("players") or []
    events = state.get("recent_events") or []
    deductions = state.get("deductions") or []
    # Absent (not 0) when the payload predates these fields or is hand-built in
    # a test -- _format_pool renders that as "(unknown)" rather than "empty",
    # which would be an actively wrong claim about the game state.
    pool_size = state.get("pool_size")
    booked = state.get("booked") or []

    my_books = 0
    for p in players:
        if p.get("player") == player_id:
            my_books = p.get("books", 0)
            break

    rank_legend = ", ".join(f"{letter}={label}" for label, letter in label_to_letter.items())
    example_move, example_target = _example_move(player_id)

    # Annotate each of our own past asks with its outcome (received N / go fish),
    # which the raw observation never reveals for our own moves. Falls back to
    # bare move strings when the history can't be replayed.
    annotated_history = _annotate_move_history(observation, move_history, player_id, num_suits)
    display_history = annotated_history if annotated_history is not None else move_history
    move_history_str = ", ".join(display_history) if display_history else "None"

    prompt = GO_FISH_PROMPT_TEMPLATE.format(
        num_ranks=num_ranks,
        num_suits=num_suits,
        rank_legend=rank_legend,
        example_move=example_move,
        example_target=example_target,
        player_id=player_id,
        my_books=my_books,
        hand_lines=_format_hand(hand, label_to_letter),
        pool_line=_format_pool(pool_size),
        booked_line=_format_booked(booked),
        other_players=_format_other_players(players, player_id),
        deductions=_format_deductions(deductions, player_id),
        events=_format_events(events),
        move_history=move_history_str,
    )

    # render_rethink_suffix substitutes only {previous_response}, so the example
    # placeholders are filled in beforehand. Use replace() rather than format():
    # the template's JSON block is literal output with doubled braces awaiting
    # that single later format() pass, and a pre-pass with format() would consume
    # the escaping and then have to re-add it.
    unparsable = RETHINK_UNPARSABLE.replace("{example_move}", example_move).replace(
        "{example_target}", str(example_target)
    )

    prompt += render_rethink_suffix(
        RETHINK_ILLEGAL,
        unparsable,
        previous_response,
        previous_action,
    )

    return prompt


# --- Parsing ----------------------------------------------------------------


_SEPARATORS_RE = re.compile(r"[\s,:;._\-]")


def _match_move_to_legal(raw: str, legal_action_strings: Sequence[str]) -> str | None:
    """Match a model's move to a legal ``<target><letter>`` action string.

    Go Fish uses a SINGLE move namespace: the action letters ``a..m`` shown in
    the hand lines and rank legend. The parser accepts ONLY that namespace --
    there is no separate human-rank-label reading. This removes the dual-encoding
    collision an earlier version had, where accepting labels as a fallback meant
    ``"1K"`` had two simultaneously-legal readings (action letter ``k`` vs label
    King=``m``) and the harness silently picked one with no rethink and no log.

    We tolerate cosmetic drift -- stray separators (``"1, a"``, ``"1-a"``) and
    case (matching is case-insensitive, so ``"1A"``/``"1J"`` resolve to action
    letters ``a``/``j``). A human rank label that is NOT also an action letter
    (``"1Q"``, ``"1, 10"``) finds no match and defers to the rethink loop.

    Residual caveat of case-insensitive matching under a single namespace: an
    uppercase card label whose glyph coincides with a legal *action letter*
    still matches that letter. E.g. ``"1K"`` -> ``1k``: if a model wrote it
    meaning King (label ``K``) but ``1k`` (=Jack) is legal, it asks for Jack.
    This is an accepted trade-off -- the prompt only ever teaches action letters
    and shows King as ask-letter ``m`` -- but it is a genuine remaining edge, not
    an impossibility. Making matching case-sensitive would close it at the cost
    of rejecting a model that merely uppercased the letter it was told to use.
    """
    compact = _SEPARATORS_RE.sub("", raw.strip())
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
