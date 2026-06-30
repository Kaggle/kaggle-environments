"""LLM harness for Word Art.

Implements the ``GameHarness`` protocol:

- ``get_legal_moves(observation)`` -- always returns ``None`` (both phases
  are free-form text: the artist submits ASCII art, the guesser submits a
  guessed word). The ``freeForm`` config flag on word_art is ``True`` by
  default.
- ``generate_prompt(observation, move_history, ...)`` -- dispatches on
  ``observation.role`` (``"artist"`` vs ``"guesser"``).
- ``parse_response(response, legal_action_strings, *, observation=None)``
  -- pulls ``"art"`` or ``"guess"`` from the last JSON object in the model
  response and returns it as a free-form ``submission``.

Word Art is 2v2: agents 0/1 are Team Blue, agents 2/3 are Team Yellow.
Each round, one teammate on each team draws ASCII art for a secret word
and passes it to their teammate, who has up to ``max_attempts`` guesses.
First-try correct scores 1 + ``first_try_bonus`` points; later-attempt
correct scores 1; failing all attempts scores 0. After ``num_rounds``
rounds the higher score wins. Roles within each team swap every round.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from kaggle_environments.core_harness import ParseResult, extract_last_json_object

# --- Helpers ----------------------------------------------------------------


def _format_history(history: Sequence[Mapping[str, Any]]) -> str:
    """Render a compact, human-readable view of completed rounds.

    Each entry summarises the word, both teams' art, every guess, and the
    points scored. We use a labelled prose block instead of dumping raw
    JSON because the ASCII art channel is multi-line and JSON-escaping
    obliterates it.

    Disqualified art is labelled explicitly: the env preserves the raw
    submission in history for replay transparency, but the guesser ONLY
    saw a placeholder at game time. Rendering the raw art without that
    annotation would mislead the model into thinking the teammate
    successfully communicated something.
    """
    if not history:
        return "No rounds completed yet."
    lines: list[str] = []
    for i, entry in enumerate(history):
        word = entry.get("word", "?")
        blue_art = entry.get("blue_art", "")
        yellow_art = entry.get("yellow_art", "")
        blue_disq = bool(entry.get("blue_art_disqualified"))
        yellow_disq = bool(entry.get("yellow_art_disqualified"))
        blue_guesses = entry.get("blue_guesses", []) or []
        yellow_guesses = entry.get("yellow_guesses", []) or []
        blue_points = entry.get("blue_points", 0)
        yellow_points = entry.get("yellow_points", 0)
        lines.append(f"Round {i + 1}: word was '{word}'.")
        lines.extend(_render_team_history_art("Blue", blue_art, blue_disq))
        lines.append(f"  Blue guesses: {blue_guesses!r} -> {blue_points} pt{'s' if blue_points != 1 else ''}")
        lines.extend(_render_team_history_art("Yellow", yellow_art, yellow_disq))
        lines.append(f"  Yellow guesses: {yellow_guesses!r} -> {yellow_points} pt{'s' if yellow_points != 1 else ''}")
    return "\n".join(lines)


def _render_team_history_art(team_label: str, art: str, disqualified: bool) -> list[str]:
    if disqualified:
        return [
            f"  {team_label} art: (DISQUALIFIED -- contained the target word; "
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
        "Both teams play in parallel; your score is independent of the "
        "other team's outcome for the round."
    )


def _round_status_block(observation: Mapping[str, Any]) -> str:
    rnd = observation.get("current_round", 0)
    n = observation.get("num_rounds", 0)
    blue_score = observation.get("blue_score", 0)
    yellow_score = observation.get("yellow_score", 0)
    return f"This is round {rnd + 1} of {n}. Current score: Blue {blue_score} - Yellow {yellow_score}."


# --- Rethink templates ------------------------------------------------------


# Both phases are free-form, so all parse failures fall into the
# "unparsable" bucket: the model didn't emit the JSON key we needed. The
# matched-but-illegal case (legal-action mismatch) doesn't exist here.
RETHINK_UNPARSABLE = """

Your previous response did not contain a valid JSON object with the required key.
Last 500 characters of your previous response:
{previous_response}

Re-read the output format above and respond again. The JSON must include the
required key and be parseable (no comments, no trailing commas, no surrounding
prose inside the JSON itself)."""


# --- Public functions (GameHarness protocol) --------------------------------


def get_legal_moves(observation: Mapping[str, Any]) -> dict[int, str] | None:
    """Always ``None`` -- both artist (ASCII art) and guesser (free-form word)
    submissions are open-ended text. The env config sets ``freeForm: true``."""
    return None


def generate_prompt(
    observation: Mapping[str, Any],
    move_history: list[str],  # noqa: ARG001 -- protocol arg; history shown via observation.history
    previous_response: str | None = None,
    previous_action: str | None = None,  # noqa: ARG001 -- never set in free-form path
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

    if previous_response is not None:
        prompt += RETHINK_UNPARSABLE.format(previous_response=previous_response[-500:])

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
  case-insensitive and whitespace-trimmed; only the exact word counts
  (no spelling variants).

{scoring}

Do not include ANY words in your art -- not the target, not a synonym,
not a label, not a NATO-alphabet or other phonetic spelling, not a
translation or a rhyme. The point of the game is to convey the WORD
through the IMAGE. Letters are fine as visual elements (an 'O' for an
eye, a 'V' for a beak); spelling things out is not.

CRITICAL (engine-enforced): the target word specifically is checked
mechanically. The check works by stripping every non-alphanumeric
character from your submission and lowercasing the result, then looking
for the target as a substring forwards OR reversed -- so 'cat', 'CAT',
'C A T', 'C-A-T', 'C.A.T', 'C\\nA\\nT', 'TAC', and any of these padded
with extra punctuation all trip it. If disqualified, your teammate sees a
placeholder instead of your drawing (no info, no first-try bonus,
almost certainly 0 points). The other "no words" rules above aren't
engine-enforced -- they're on your honor.

Art must be printable ASCII only, and is silently truncated at
{max_art_chars} characters -- keep it compact.

The secret word you must depict is: '{target_word}'.

Past rounds in this game so far:
{history_text}

Think step by step about how to depict the word visually, then return
your reasoning in 'thinking' and the ASCII art string in 'art' (escape
newlines as '\\n'). No text or markdown outside the JSON block. Example:

```json
{{"thinking": "I'll draw a cat face using basic ASCII characters...",
  "art": " /\\\\_/\\\\\\n( o.o )\\n > ^ <"}}
```"""


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
  and whitespace-trimmed; only the exact word counts (no plurals,
  synonyms, or partial matches).
- The opposing team plays in parallel and cannot see your art or guesses.
- If the artist included the target word in their drawing, the engine
  REPLACES it with a disqualification marker -- you'll see the marker
  text instead of a picture. Past rounds in the history below are
  likewise labelled "DISQUALIFIED" when this happened.

{scoring}

{attempt_pitch}
{prev_block}

Past rounds in this game so far:
{history_text}

Your teammate's drawing (be aware that monospace alignment matters):
{teammate_art if teammate_art else "(your teammate submitted nothing)"}

Think step by step about what the art depicts (letter shapes, spatial
layout, any annotations), then return your reasoning in 'thinking' and
a SINGLE WORD (no spaces, no punctuation, no articles) in 'guess'. No
text or markdown outside the JSON block. Example:

```json
{{"thinking": "Four-legged animal with a tail and pointy ears; the 'meow'-like whiskers suggest CAT.",
  "guess": "CAT"}}
```"""


def parse_response(
    response: str,
    legal_action_strings: Sequence[str] | None,
    *,
    observation: Mapping[str, Any] | None = None,
) -> ParseResult:
    """Extract the artist's art or the guesser's word from the LLM response.

    Both phases are free-form, so ``legal_action_strings`` is always
    ``None``. We look for the LAST JSON object in the response that
    contains either ``"art"`` or ``"guess"`` (the helper does the
    multi-block / fenced / bare-JSON disambiguation for us).

    Dispatch is role-strict: in production ``observation`` is always
    supplied, so we route on ``observation.role`` and require the
    matching key. If the model emits the wrong key (e.g. ``"guess"`` on
    an artist turn) we return ``ParseResult(raw_action=...)`` so the
    rethink loop can correct them rather than silently accepting an
    obviously-wrong submission. The trailing key-agnostic fallback only
    fires when ``observation`` isn't forwarded -- in practice only some
    unit tests hit it.
    """
    parsed = extract_last_json_object(response, required_keys=("art", "guess"))
    if parsed is None:
        # No JSON object found at all -- raw_action=None so core_harness
        # categorizes this as UNPARSABLE (not ILLEGAL) in telemetry. The
        # rethink loop still has the full response via `previous_response`.
        return ParseResult(raw_action=None)

    thinking = parsed.get("thinking")
    role = (observation or {}).get("role", "")

    if role == "artist":
        art = parsed.get("art")
        if art is None:
            # Model emitted JSON but used the wrong key. Surface what they
            # said so the rethink loop can correct them.
            return ParseResult(raw_action=json.dumps(parsed)[:500], thoughts=thinking)
        art_str = art if isinstance(art, str) else str(art)
        return ParseResult(
            submission=art_str,
            raw_action=art_str[:200],
            thoughts=thinking,
        )

    if role == "guesser":
        guess = parsed.get("guess")
        if guess is None:
            return ParseResult(raw_action=json.dumps(parsed)[:500], thoughts=thinking)
        guess_str = guess if isinstance(guess, str) else str(guess)
        return ParseResult(
            submission=guess_str,
            raw_action=guess_str[:200],
            thoughts=thinking,
        )

    # No role available (e.g. observation kwarg not forwarded). Accept
    # whichever key was present.
    for key in ("art", "guess"):
        value = parsed.get(key)
        if value is not None:
            value_str = value if isinstance(value, str) else str(value)
            return ParseResult(
                submission=value_str,
                raw_action=value_str[:200],
                thoughts=thinking,
            )
    return ParseResult(raw_action=json.dumps(parsed)[:500], thoughts=thinking)
