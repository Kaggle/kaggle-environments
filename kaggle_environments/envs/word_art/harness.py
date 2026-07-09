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
First-try correct scores 1 + ``first_try_bonus`` points; later-attempt
correct scores 1; failing all attempts scores 0. After ``num_rounds``
rounds the higher score wins. Roles within each team swap every round.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from kaggle_environments.core_harness import (
    ParseResult,
    extract_last_json_object_with_position,
)

# Matches every <art>...</art> block. Case-insensitive and tolerant of
# whitespace around the tag names so `<Art>`, `< art >`, `</ART >` all
# match. DOTALL so the tag contents can span newlines -- essential for
# multi-line ASCII art. Last-wins: if the model rethinks and emits a
# second <art> block, the trailing one is the intent.
_ART_TAG_RE = re.compile(
    r"<\s*art\s*>(.*?)<\s*/\s*art\s*>",
    re.DOTALL | re.IGNORECASE,
)


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
    "contains_words": "contained text (a run of 3+ letters with 2+ distinct chars)",
}


def _format_history(history: Sequence[Mapping[str, Any]]) -> str:
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
        lines.extend(_render_team_history_art("Blue", blue_art, blue_reason))
        lines.append(f"  Blue guesses: {blue_guesses!r} -> {blue_points} pt{'s' if blue_points != 1 else ''}")
        lines.extend(_render_team_history_art("Yellow", yellow_art, yellow_reason))
        lines.append(f"  Yellow guesses: {yellow_guesses!r} -> {yellow_points} pt{'s' if yellow_points != 1 else ''}")
    return "\n".join(lines)


def _render_team_history_art(team_label: str, art: str, disq_reason: str | None) -> list[str]:
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


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines()) or pad


def _team_label(team: str) -> str:
    return "Blue" if team == "blue" else "Yellow"


def _scoring_block(max_attempts: int, first_try_bonus: int) -> str:
    base = 1
    first_try_total = base + first_try_bonus
    return (
        f"Scoring (per round, per team):\n"
        f"  - Correct on attempt 1: {first_try_total} points "
        f"(1 base + {first_try_bonus} first-try bonus)\n"
        f"  - Correct on attempt 2 through {max_attempts}: 1 point\n"
        f"  - No correct guess within {max_attempts} attempts: 0 points\n"
        "Both teams play the same secret word each round in parallel; your "
        "score is independent of the other team's outcome for the round. "
        "After all rounds are played, the team with the higher total wins; "
        "equal totals are a tie."
    )


def _round_status_block(observation: Mapping[str, Any]) -> str:
    rnd = observation.get("current_round", 0)
    n = observation.get("num_rounds", 0)
    blue_score = observation.get("blue_score", 0)
    yellow_score = observation.get("yellow_score", 0)
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
were empty or whitespace-only. Your submitted block was:
{previous_action}

Re-read the output format above and respond again. The <art>...</art>
block must contain the actual ASCII drawing."""


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
    first_try_bonus = observation.get("first_try_bonus", 1)
    max_art_chars = observation.get("max_art_chars", 4000)
    status_line = _round_status_block(observation)
    history_text = _format_history(observation.get("history", []))
    scoring = _scoring_block(max_attempts, first_try_bonus)

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
            prompt += RETHINK_ARTIST_EMPTY.format(previous_action=previous_action)
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
  drawing, never the word. The opposing team sees neither. Roles swap
  each round.
- The guesser has up to {max_attempts} attempts. Matching is
  case-insensitive with leading/trailing whitespace trimmed; only the
  exact word counts (no plurals, synonyms, partial matches, or other
  spelling variants).

{scoring}

The point of the game is to convey the WORD through the IMAGE.
DO NOT INCLUDE ANY WORDS IN YOUR ART. Letters are fine as visual
elements (an 'O' for an eye, a 'V' for a beak, '|||' as columns, 'OOO'
as wheels); spelling out words -- targets, synonyms, labels, captions,
section headers, arrow annotations, NATO-alphabet, translations, rhymes
-- is not.

CRITICAL (engine-enforced): TWO mechanical checks run on your art. If
either fires, your teammate sees a placeholder instead of your drawing
(no info, no first-try bonus, almost certainly 0 points).

  1. TARGET-WORD check. The engine strips every non-alphanumeric
     character and lowercases the result, then looks for the target as
     a substring forwards OR reversed. So 'cat', 'CAT', 'C A T',
     'C-A-T', 'C.A.T', 'C\\nA\\nT', 'TAC', and any of these padded with
     extra punctuation all trip it -- including annotations like
     '(scale: CAT)', arrow labels like '<- CAT', or section headers
     like 'CAT close-up:'.

  2. ANY-WORD check. Any run of 3+ consecutive letters with 2+ distinct
     characters (case-insensitive) disqualifies the drawing. Words like
     'top', 'the', 'HOUSE', 'MINERAL', 'grid', 'axe' all trip it. Same-
     character runs pass -- 'OOO' (eyes), 'III' (columns), 'TTT'
     (texture) are all fine -- as are 1- and 2-letter clusters like
     'V', 'OO', 'H2'. Break letters up with spaces, punctuation, or
     newlines to avoid tripping this check.

Your art is silently sanitized before scoring: combining marks, wide
characters (CJK, most emoji), and other non-single-cell Unicode are
dropped so monospace alignment holds for the guesser. It's then
truncated at {max_art_chars} characters -- keep it compact.

The secret word you must depict is: '{target_word}'.

Past rounds in this game so far:
{history_text}

Think step by step about how to depict the word visually, writing
your reasoning as ordinary prose. Then end your response with your
final drawing wrapped in a single <art>...</art> block. Everything
inside the block is taken verbatim -- literal newlines are fine, no
escaping, no markdown -- and everything outside is treated as
reasoning and ignored. Example:

I'll draw a cat face using basic ASCII characters -- pointy ears with
slashes, round eyes, a nose. Keeping runs of letters to 2 or fewer to
stay clear of the any-word check.

<art>
 /\\_/\\
( o.o )
 > ^ <
</art>"""


def _build_guesser_prompt(
    observation: Mapping[str, Any],
    team_label: str,
    status_line: str,
    history_text: str,
    scoring: str,
    max_attempts: int,
) -> str:
    teammate_art = observation.get("teammate_art", "")
    previous_guesses = list(observation.get("previous_guesses", []) or [])
    attempts_remaining = observation.get("attempts_remaining", max_attempts)
    attempt_number = max_attempts - attempts_remaining + 1

    if previous_guesses:
        prev_block = f"Your previous guesses this round (all wrong): {previous_guesses!r}"
    else:
        prev_block = "This is your first guess this round."

    if attempt_number == 1:
        attempt_pitch = f"This is attempt 1 of {max_attempts}. A correct guess NOW earns the first-try bonus."
    else:
        attempt_pitch = (
            f"This is attempt {attempt_number} of {max_attempts}. You have "
            f"{attempts_remaining} attempt(s) left (including this one). No "
            "bonus is available now, but a correct guess still scores 1 point."
        )

    return f"""You are the GUESSER on Team {team_label} in Word Art (a 2v2 game).

{status_line}

Rules:
- Your teammate (the artist) saw a secret word and drew the ASCII art
  below; you don't see the word. Roles swap each round.
- You have up to {max_attempts} guesses. Matching is case-insensitive
  with leading/trailing whitespace trimmed; only the exact word counts
  (no plurals, synonyms, partial matches, or other spelling variants).
- The opposing team plays the same secret word each round in parallel
  and cannot see your art or guesses.
- The engine mechanically disqualifies art that contains either the
  target word or any run of 3+ letters with 2+ distinct characters
  (labels, captions, headings). When that happens you'll see a
  placeholder marker instead of a picture. Past rounds in the history
  below are likewise labelled "DISQUALIFIED" when this happened.

{scoring}

{attempt_pitch}
{prev_block}

Past rounds in this game so far:
{history_text}

Your teammate's drawing (be aware that monospace alignment matters):
{teammate_art if teammate_art else "(your teammate submitted nothing)"}

Think step by step about what the art depicts (letter shapes, spatial
layout, any annotations), writing your reasoning as ordinary prose.
Then end your response with a JSON object containing your final answer
as a SINGLE WORD (no spaces, no punctuation, no articles). Example:

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
      can quote it back.

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
    role = (observation or {}).get("role", "")

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
            return ParseResult(raw_action="<art></art>", thoughts=thoughts)
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
