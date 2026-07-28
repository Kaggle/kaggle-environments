"""LLM harness for Word Art.

Implements the ``GameHarness`` protocol:

- ``get_legal_moves(observation)`` -- always returns ``None`` (both phases
  are free-form text: the artist submits ASCII art, the guesser submits a
  guessed word). The ``freeForm`` config flag on word_art is ``True`` by
  default.
- ``generate_prompt(observation, move_history, ...)`` -- dispatches on
  ``observation.role`` (``"artist"`` vs ``"guesser"``).
- ``parse_response(response, legal_action_strings, *, observation=None)``
  -- extracts the answer from the last role-appropriate answer marker in
  the model response and returns it as a free-form ``submission``.

Output formats differ by role:

- **Artist** writes prose reasoning, then wraps the drawing in
  ``<art>...</art>`` tags. Tags -- not JSON -- because ASCII art is
  full of newlines, backslashes, and quotes that would need escaping
  inside a JSON string. In practice models routinely forget to escape
  those, which forced ~1% of turns into an avoidable retry when this
  harness used JSON for the art payload.
- **Guesser** writes prose reasoning, then a JSON object
  ``{"guess": "..."}``. Single-word answers don't have the escaping
  problem, and JSON keeps the guesser consistent with the rest of the
  repo's harnesses.

Word Art is 2v2: agents 0/1 are Team Blue, agents 2/3 are Team Yellow.
Each round, one teammate on each team draws ASCII art for a secret word
and passes it to their teammate, who has up to ``max_attempts`` guesses.
A correct guess on attempt ``i`` scores ``guess_points[i-1]`` (per-attempt
reward table, configurable, may be fractional); failing all attempts
scores 0. After ``num_rounds`` rounds the higher total wins. Roles within
each team swap every round.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from kaggle_environments.core_harness import (
    ParseResult,
    extract_last_json_object_with_position,
)
from kaggle_environments.envs.word_art.word_art import check_art

# Matches every <art>...</art> block. Case-insensitive and tolerant of
# whitespace around the tag names so `<Art>`, `< art >`, `</ART >` all
# match. DOTALL so the tag contents can span newlines -- essential for
# multi-line ASCII art. Last-wins: if the model rethinks and emits a
# second <art> block, the trailing one is the intent.
#
# The body is tempered so it cannot span another opening `<art>`: without
# that, a stray unpaired `<art>` in the prose ("I'll use <art> tags") binds
# to the closing tag of the real drawing and swallows the reasoning into it.
_ART_TAG_RE = re.compile(
    r"<\s*art\s*>((?:(?!<\s*art\s*>).)*?)<\s*/\s*art\s*>",
    re.DOTALL | re.IGNORECASE,
)

# Stand-in `raw_action` for an <art> block that was present but empty.
# Must stay empty: on the `illegalMoveForfeit` path core_harness copies the
# last `raw_action` into `actionString`, which the env submits as the
# drawing. Any non-empty sentinel gets scored as real art -- a literal
# "<art></art>" trips the any-word check on 'art' and tells the guesser
# their teammate drew a text label when the artist in fact drew nothing.
# `generate_prompt` still distinguishes it from "no <art> tag at all",
# which parses to `raw_action=None`.
_EMPTY_ART_MARKER = ""


def _slice_thoughts(response: str, answer_start: int) -> str | None:
    """Return the prose reasoning that precedes the answer marker, or
    ``None`` if there's nothing meaningful before it (in which case
    core_harness falls back to storing the full raw response, which is
    still the useful thing to log)."""
    if answer_start <= 0:
        return None
    prose = response[:answer_start].strip()
    return prose or None

# --- Helpers ----------------------------------------------------------------


_DISQ_REASON_TEXT = {
    "target_word": "contained the target word",
    "contains_words": "contained a text label",
}


def _format_history(history: Sequence[Mapping[str, Any]], include_art: bool = True) -> str:
    """Render a compact, human-readable view of completed rounds.

    Each entry summarises the word, both teams' art, every guess, and the
    points scored. We use a labelled prose block instead of dumping raw
    JSON because the ASCII art channel is multi-line and JSON-escaping
    obliterates it.

    Disqualified art is labelled explicitly with the reason the engine
    rejected it: the env preserves the raw submission in history for
    replay transparency, but the guesser ONLY saw a placeholder at game
    time. Rendering the raw art without that annotation would mislead
    the model into thinking the teammate successfully communicated
    something.

    ``include_art`` toggles whether the ASCII art bodies of past rounds
    are included. When False, the raw art body is dropped but the
    disqualification annotation is kept: the annotation is a
    game-mechanics signal (why the team scored 0), not art content, so
    an artist who tripped the check on a past round still learns to
    avoid it. Word / guesses / points always render.
    """
    if not history:
        return "No rounds completed yet."
    lines: list[str] = []
    for i, entry in enumerate(history):
        word = entry.get("word", "?")
        blue_art = entry.get("blue_art", "")
        yellow_art = entry.get("yellow_art", "")
        blue_reason = entry.get("blue_art_disqualification_reason")
        yellow_reason = entry.get("yellow_art_disqualification_reason")
        blue_guesses = entry.get("blue_guesses", []) or []
        yellow_guesses = entry.get("yellow_guesses", []) or []
        blue_points = entry.get("blue_points", 0)
        yellow_points = entry.get("yellow_points", 0)
        lines.append(f"Round {i + 1}: word was '{word}'.")
        lines.extend(_render_team_history_entry("Blue", blue_art, blue_reason, include_art))
        lines.append(f"  Blue guesses: {blue_guesses!r} -> {_format_points(blue_points)}")
        lines.extend(_render_team_history_entry("Yellow", yellow_art, yellow_reason, include_art))
        lines.append(f"  Yellow guesses: {yellow_guesses!r} -> {_format_points(yellow_points)}")
    return "\n".join(lines)


def _render_team_history_entry(
    team_label: str, art: str, disq_reason: str | None, include_art: bool,
) -> list[str]:
    """Render one team's art (or disq annotation) for a completed round.

    Four states:
      * ``include_art=True``, no disqualification: header + indented art body.
      * ``include_art=True``, disqualified: header naming the check that
        fired + indented raw art body.
      * ``include_art=False``, no disqualification: single "art omitted" line
        so the model knows something WAS drawn (vs. artist forfeit).
      * ``include_art=False``, disqualified: single "DISQUALIFIED -- reason"
        line so the model still learns what tripped the check.
    """
    if include_art:
        if disq_reason:
            why = _DISQ_REASON_TEXT.get(disq_reason, "was disqualified")
            return [
                f"  {team_label} art: (DISQUALIFIED -- {why}; "
                "the guesser saw a placeholder, not the raw drawing below)",
                _indent(art or "(empty)", 4),
            ]
        return [
            f"  {team_label} art:",
            _indent(art or "(empty)", 4),
        ]
    if disq_reason:
        why = _DISQ_REASON_TEXT.get(disq_reason, "was disqualified")
        return [
            f"  {team_label} art: DISQUALIFIED -- {why}. Teammate saw a placeholder. (art body omitted)"
        ]
    if not art:
        return [f"  {team_label} art: (nothing submitted)"]
    return [f"  {team_label} art: (omitted for brevity)"]


def _format_points(points: Any) -> str:
    """Render a scoring value for prose lines.

    Integer-valued floats print without a trailing `.0` (so guess_points
    like `[2, 1, 1]` and `[2.0, 1.0, 1.0]` produce identical text);
    fractional floats print naturally (`1.5`).
    """
    if isinstance(points, bool):  # bool is-a int; treat as-is
        n = 1 if points else 0
    elif isinstance(points, (int, float)):
        n = points
    else:
        return f"{points} pts"
    label = "pt" if n == 1 else "pts"
    if isinstance(n, float) and n.is_integer():
        return f"{int(n)} {label}"
    return f"{n} {label}"


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines()) or pad


def _team_label(team: str) -> str:
    return "Blue" if team == "blue" else "Yellow"


def _points_for_attempt(guess_points: Sequence[float], attempt_number: int) -> float:
    # Clamped rather than indexed directly: prompt building happens outside
    # core_harness's retry try-block, so an IndexError here escapes agent_fn
    # and marks the seat ERROR for the rest of the episode.
    if not guess_points:
        return 1
    index = min(max(attempt_number, 1), len(guess_points)) - 1
    return guess_points[index]


def _scoring_block(max_attempts: int, guess_points: Sequence[float]) -> str:
    # Enumerate every attempt explicitly. Cheaper for the model than
    # asking it to interpolate a table, and it makes the reward gradient
    # obvious when values are fractional (e.g. [2, 1.5, 1] reads as three
    # distinct lines, not "1 + bonus" arithmetic).
    lines = ["Scoring (per round, per team):"]
    for i in range(max_attempts):
        lines.append(
            f"  - Correct on attempt {i + 1}: "
            f"{_format_points(_points_for_attempt(guess_points, i + 1))}"
        )
    lines.append(f"  - No correct guess within {max_attempts} attempts: 0 pts")
    lines.append(
        "Both teams play the same secret word each round in parallel; your "
        "score is independent of the other team's outcome for the round. "
        "After all rounds are played, the team with the higher total wins; "
        "equal totals are a tie."
    )
    return "\n".join(lines)


def _format_score(value: Any) -> str:
    """Score as a bare number (no unit), collapsing integer-valued floats.
    Keeps the running-score line consistent with `_format_points`, which
    already strips `.0` from history-line and scoring-table numbers.
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _round_status_block(observation: Mapping[str, Any]) -> str:
    rnd = observation.get("current_round", 0)
    n = observation.get("num_rounds", 0)
    blue_score = _format_score(observation.get("blue_score", 0))
    yellow_score = _format_score(observation.get("yellow_score", 0))
    return f"This is round {rnd + 1} of {n}. Current score: Blue {blue_score} - Yellow {yellow_score}."


# --- Rethink templates ------------------------------------------------------


# Free-form means there's no "illegal action" case (any string is a legal
# submission), but parse failure comes in two flavours per role that need
# different corrections:
#   NO_ANSWER  -> the response had no answer marker at all (no <art> tag
#                 for artists, no JSON with a "guess" key for guessers).
#                 Show the last 500 chars of the response so the model can
#                 see how its answer trailed off, and restate the format.
#   EMPTY      -> a marker was present but its value was missing / empty /
#                 whitespace-only. Show the offending marker back so the
#                 model sees exactly what got rejected instead of guessing.
RETHINK_ARTIST_NO_ANSWER = """

Your previous response did not contain a parseable <art>...</art> block.
Last 500 characters of your previous response:
{previous_response}

Re-read the output format above and respond again. Wrap your drawing in a
single <art>...</art> block; anything outside the block is treated as
reasoning and ignored."""


RETHINK_ARTIST_EMPTY = """

Your previous response included an <art>...</art> block but its contents
were empty or whitespace-only, so nothing was submitted.

Re-read the output format above and respond again. The <art>...</art>
block must contain the actual ASCII drawing."""


# REJECTED is not a parse failure -- the <art> block was well-formed, but
# running the engine's own checks against it says the drawing would be
# swapped for a placeholder and score ~0. Catching it here converts a
# silent post-hoc zero into a correctable retry, so the message must name
# the exact offending run: telling the model only "you included text"
# leaves it guessing which glyphs to change.
RETHINK_ARTIST_REJECTED = """

Your previous drawing would be REJECTED by the engine: it contains
{detail}. Your teammate would see a placeholder instead of the drawing
and your team would score 0 this round. Your submitted drawing was:
{previous_action}

Redraw it so no run of letters spells anything -- delete the offending
text outright rather than disguising it (spacing it out, punctuating
between the letters, reversing it, or stacking it down a column all trip
the same checks). Re-read the two engine checks above, then respond again
with reasoning followed by a single <art>...</art> block."""


RETHINK_GUESSER_NO_ANSWER = """

Your previous response did not contain a parseable JSON object with a
"guess" key. Last 500 characters of your previous response:
{previous_response}

Re-read the output format above and respond again. End your response with
a JSON object of the form {{"guess": "SINGLEWORD"}} (parseable: no
comments, no trailing commas)."""


RETHINK_GUESSER_BAD_VALUE = """

Your previous response included a JSON object but the "guess" key was
missing or had an invalid value (must be a non-empty string). Your
submitted JSON was:
{previous_action}

Re-read the output format above and respond again. The JSON must include a
"guess" key with a non-empty single-word string value."""


# --- Public functions (GameHarness protocol) --------------------------------


def get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str] | None:
    """Always ``None`` -- both artist (ASCII art) and guesser (free-form word)
    submissions are open-ended text. The env config sets ``freeForm: true``."""
    return None


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],  # noqa: ARG001 -- protocol arg; history shown via observation.history
    previous_response: str | None = None,
    previous_action: str | None = None,
) -> str:
    """Build the LLM prompt for the current turn."""
    role = observation.get("role", "")
    team = observation.get("team", "")
    team_label = _team_label(team)
    max_attempts = observation.get("max_attempts", 3)
    # The env surfaces these config knobs on the observation at init time.
    # The fallback defaults match the env spec defaults and only fire on
    # a malformed obs (e.g. a unit test that hand-rolls one).
    raw_guess_points = list(observation.get("guess_points") or [])
    guess_points = raw_guess_points if raw_guess_points else [1] * max_attempts
    include_art_history = bool(observation.get("include_art_history", True))
    max_art_chars = observation.get("max_art_chars", 4000)
    status_line = _round_status_block(observation)
    history_text = _format_history(observation.get("history", []), include_art_history)
    scoring = _scoring_block(max_attempts, guess_points)

    if role == "artist":
        prompt = _build_artist_prompt(
            observation,
            team_label,
            status_line,
            history_text,
            scoring,
            max_attempts,
            max_art_chars,
        )
    elif role == "guesser":
        prompt = _build_guesser_prompt(
            observation,
            team_label,
            status_line,
            history_text,
            scoring,
            max_attempts,
            guess_points,
        )
    else:
        # No active role yet (e.g. very first probe before init populates the
        # observation). Return a placeholder; core_harness will treat the
        # agent as inactive on an empty obs anyway.
        prompt = "Word Art has not started this round yet. Wait for your role (artist or guesser) to be assigned."

    # core_harness sets previous_action to the parser's raw_action on failure:
    # None when no answer marker was found, populated when a marker was found
    # but its value was missing/empty. Branch on role AND on that flag so the
    # model sees a correction tailored to what actually broke.
    if role == "artist":
        if previous_action is not None:
            # Two shapes of `previous_action` land here: the empty-block
            # sentinel, and a well-formed drawing that `parse_response`
            # refused because it would be disqualified. Re-derive the
            # verdict instead of smuggling it through the protocol -- the
            # check is deterministic, so this reproduces exactly what the
            # parser saw.
            verdict = (
                None if previous_action == _EMPTY_ART_MARKER
                else check_art(previous_action, observation.get("target_word", ""), max_art_chars)
            )
            if verdict is None:
                prompt += RETHINK_ARTIST_EMPTY
            else:
                prompt += RETHINK_ARTIST_REJECTED.format(
                    detail=verdict[1], previous_action=previous_action,
                )
        elif previous_response is not None:
            prompt += RETHINK_ARTIST_NO_ANSWER.format(previous_response=previous_response[-500:])
    elif role == "guesser":
        if previous_action is not None:
            prompt += RETHINK_GUESSER_BAD_VALUE.format(previous_action=previous_action)
        elif previous_response is not None:
            prompt += RETHINK_GUESSER_NO_ANSWER.format(previous_response=previous_response[-500:])

    return prompt


def _build_artist_prompt(
    observation: Mapping[str, Any],
    team_label: str,
    status_line: str,
    history_text: str,
    scoring: str,
    max_attempts: int,
    max_art_chars: int,
) -> str:
    target_word = observation.get("target_word", "")
    return f"""You are the ARTIST on Team {team_label} in Word Art (a 2v2 game).

{status_line}

Rules:
- You see a secret word; your teammate (the guesser) sees only your
  drawing, never the word. The opposing team plays the same secret
  word in parallel; during the live round neither team sees the
  other's drawing. Past rounds are shared in the history block below.
  Roles swap each round.
- The guesser has up to {max_attempts} attempts. Matching is
  case-insensitive with leading/trailing whitespace trimmed and accepts
  singular/plural equivalents (CAT/CATS both count, CHILD/CHILDREN both
  count). Synonyms, tenses, and other spelling variants don't count.

{scoring}

The point of the game is to convey the WORD through the IMAGE.
DO NOT INCLUDE ANY WORDS IN YOUR ART. Letters are fine as visual
elements (an 'O' for an eye, a 'V' for a beak, '|||' as columns, 'OOO'
as wheels); spelling out words -- targets, synonyms, labels, captions,
section headers, arrow annotations, NATO-alphabet, translations, rhymes
-- is not.

CRITICAL (engine-enforced): TWO mechanical checks run on your art. If
either fires, your teammate sees a placeholder instead of your drawing
(no info, almost certainly 0 points on this round).

  1. TARGET-WORD check. The engine strips every non-alphanumeric
     character and lowercases the result, then looks for the target as
     a substring forwards OR reversed. So 'cat', 'CAT', 'C A T',
     'C-A-T', 'C.A.T', 'C\\nA\\nT', 'TAC', and any of these padded with
     extra punctuation all trip it -- including annotations like
     '(scale: CAT)', arrow labels like '<- CAT', or section headers
     like 'CAT close-up:'.

  2. ANY-WORD check. The same two patterns fire in two directions:
     ROW-wise (along a single line):
       * Consecutive letters: a run of 3+ with 2+ distinct chars
         disqualifies ('top', 'HOUSE', 'grid', 'axe' all trip).
       * Spaced-out letters: breaking a word up does not help. The
         engine strips the separators and reads the letters straight
         through, so 'A R O U N D', 'T.O.P.', 'H-O-U-S-E',
         'H|O|U|S|E' and 'grid_view' all trip exactly like the
         unbroken word.
     COLUMN-wise (letters that line up down the SAME column of your
     drawing, whether on adjacent rows or with non-letter cells
     between them): a run of 4+ letters with 4+ distinct chars
     disqualifies. Catches 'F' / 'L' / 'A' / 'G' stacked one per
     line inside a box, a rainbow row of same-letter rows
     ('RRRRR' / 'OOOOO' / 'YYYYY' / 'GGGGG' -> 'ROYG' down column),
     and the same labels with blank rows or decorative dividers
     between letters.
     Repeating a SINGLE letter is always safe on a row: 'OOO' as
     wheels, 'IIIII' as railings, 'vvvvv' as grass. What trips the ROW
     check is mixing distinct letters into one run -- 'ABAB', 'VWVW'
     and 'ABBA' are each built from only two letters, but they still
     read as 3+ character runs and get rejected. Stacking rows of
     letter texture is fine so long as no column ends up with four
     different letters in it.

Your art is truncated at {max_art_chars} chars. Non-monospace
characters and non-ASCII letters (Cyrillic, Greek, accented, circled,
fullwidth, math-styled) are stripped BEFORE the checks run -- they
punch holes in your drawing rather than evading detection.
Box-drawing, blocks, arrows, and Braille survive.

The secret word you must depict is: '{target_word}'.

Past rounds in this game so far:
{history_text}

Think step by step about how to depict the word visually, writing
your reasoning as ordinary prose. Before you commit to a drawing,
re-scan it once across each row and once down each column and delete
any letter run that trips the checks above. Then end your response
with your final drawing wrapped in a single <art>...</art> block. Everything
inside the block is taken verbatim -- literal newlines are fine, no
escaping, no markdown -- and everything outside is treated as
reasoning and ignored. The example below shows the OUTPUT FORMAT, not
a template to imitate -- your drawing should depict your own word.

I'll draw a snowflake using radial symmetry -- arms meeting at a
central plus, with stars at the tips. All non-letter glyphs so
nothing can trip the any-word check.

<art>
   *
 \\ | /
---+---
 / | \\
   *
</art>"""


def _build_guesser_prompt(
    observation: Mapping[str, Any],
    team_label: str,
    status_line: str,
    history_text: str,
    scoring: str,
    max_attempts: int,
    guess_points: Sequence[float],
) -> str:
    teammate_art = observation.get("teammate_art", "")
    previous_guesses = list(observation.get("previous_guesses", []) or [])
    attempts_remaining = observation.get("attempts_remaining", max_attempts)
    attempt_number = max_attempts - attempts_remaining + 1

    if previous_guesses:
        prev_block = f"Your previous guesses this round (all wrong): {previous_guesses!r}"
    else:
        prev_block = "This is your first guess this round."

    # Pitch the current attempt in terms of its actual point value from
    # guess_points, so the model sees the same numbers as the scoring
    # table above.
    current_pts_str = _format_points(_points_for_attempt(guess_points, attempt_number))
    if attempt_number == 1:
        attempt_pitch = (
            f"This is attempt 1 of {max_attempts} in the current round. "
            f"A correct guess NOW scores {current_pts_str}."
        )
    else:
        attempt_pitch = (
            f"This is attempt {attempt_number} of {max_attempts} in the "
            f"current round. You have {attempts_remaining} attempt(s) left "
            f"(including this one). A correct guess NOW scores {current_pts_str}."
        )

    return f"""You are the GUESSER on Team {team_label} in Word Art (a 2v2 game).

{status_line}

Rules:
- Your teammate (the artist) saw a secret word and drew the ASCII art
  below; you don't see the word. Roles swap each round.
- You have up to {max_attempts} guesses. Matching is case-insensitive
  with leading/trailing whitespace trimmed and accepts singular/plural
  equivalents (CAT/CATS both count, CHILD/CHILDREN both count). Synonyms,
  tenses, and other spelling variants don't count.
- The opposing team plays the same secret word each round in parallel;
  during the live round they don't see your art or guesses.
- Art containing the target word or any run of letter-text is
  replaced by a placeholder marker; past disqualified rounds are
  labelled "DISQUALIFIED" in the history.

{scoring}

Past rounds in this game so far:
{history_text}

{attempt_pitch}
{prev_block}

Your teammate's drawing (monospace):
{teammate_art if teammate_art else "(your teammate submitted nothing)"}

Write your reasoning as prose about what the art depicts. Then end
your response with a JSON object containing your final answer as a
SINGLE WORD (no spaces, no punctuation, no articles). Example:

Four-legged animal with a tail and pointy ears; the 'meow'-like
whiskers suggest CAT.

{{"guess": "CAT"}}"""


def parse_response(
    response: str,
    legal_action_strings: Sequence[str] | None,
    *,
    observation: Mapping[str, Any] | None = None,
) -> ParseResult:
    """Extract the artist's art or the guesser's word from the LLM response.

    Both phases are free-form, so ``legal_action_strings`` is always
    ``None``. Dispatch is role-strict:

    - **Artist**: extracts the contents of the LAST ``<art>...</art>``
      block (case-insensitive, tolerant of whitespace inside the tag
      names). If no block matches, returns ``ParseResult(raw_action=None)``
      -- categorized as UNPARSABLE in telemetry. If a block matches but
      its contents are empty/whitespace-only, returns
      ``ParseResult(raw_action=<the empty tag>)`` so the rethink prompt
      can quote it back. A well-formed block is additionally run through
      the engine's own art checks (``word_art.check_art``); art that
      would be disqualified is withheld from ``submission`` so the retry
      loop fires. Without this the violation is only caught at step time,
      by which point the turn is committed and the round is worth 0 --
      and past-round annotations in the history block demonstrably do not
      teach models to stop.

    - **Guesser**: extracts the LAST parseable JSON object containing a
      ``"guess"`` key. Same two failure modes (no JSON vs. present but
      bad value) map to the same two ``raw_action`` outcomes.

    Missing / unrecognized role: returns ``ParseResult(raw_action=None)``
    without submitting. In production ``core_harness`` always forwards
    ``observation``, so this branch only fires from ad-hoc test callers.

    ``thoughts`` carries the prose reasoning that precedes the answer
    marker in the response -- everything before the last ``<art>`` /
    JSON block, whitespace-stripped. When the model wrote no prose (or
    the parser found no answer marker at all) ``thoughts`` is left
    ``None`` and ``core_harness`` falls back to logging the full raw
    response, which is still the useful thing to keep in the replay.
    """
    obs = observation or {}
    role = obs.get("role", "")

    if role == "artist":
        matches = list(_ART_TAG_RE.finditer(response))
        if not matches:
            return ParseResult(raw_action=None)
        last = matches[-1]
        raw = last.group(1)
        thoughts = _slice_thoughts(response, last.start())
        if raw.strip() == "":
            # Empty <art> block -- record the (empty) tag for the rethink
            # prompt to quote back so the model sees what got rejected.
            return ParseResult(raw_action=_EMPTY_ART_MARKER, thoughts=thoughts)
        if check_art(raw, obs.get("target_word", ""), obs.get("max_art_chars", 4000)):
            # Well-formed but would be disqualified by the engine. Withhold
            # the submission so the retry loop fires: the artist gets one
            # more shot at a clean drawing instead of a silent 0.
            return ParseResult(raw_action=raw, thoughts=thoughts)
        return ParseResult(submission=raw, raw_action=raw, thoughts=thoughts)

    if role == "guesser":
        parsed, start = extract_last_json_object_with_position(
            response, required_keys=("guess",),
        )
        if parsed is None:
            return ParseResult(raw_action=None)
        thoughts = _slice_thoughts(response, start)
        value = parsed.get("guess")
        if not isinstance(value, str) or value.strip() == "":
            # Cap the dumped-JSON preview so a runaway payload can't bloat
            # telemetry or the rethink prompt; the answer we tried to
            # extract is what matters here.
            return ParseResult(raw_action=json.dumps(parsed)[:500], thoughts=thoughts)
        return ParseResult(submission=value, raw_action=value, thoughts=thoughts)

    # Unknown / missing role -- refuse to submit; test-only path in
    # practice since core_harness forwards `observation`.
    return ParseResult(raw_action=None)
