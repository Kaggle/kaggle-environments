"""Tests for the Word Art harness (core_harness integration).

Output-format contract exercised by these tests:

- Artist writes prose reasoning, then wraps the drawing in ``<art>...</art>``
  tags. Tags -- not JSON -- because ASCII art is full of newlines /
  backslashes / quotes that models routinely forget to escape.
- Guesser writes prose reasoning, then a ``{"guess": "..."}`` JSON object.
  Single-word answers don't have the escaping problem, and JSON keeps the
  guesser consistent with the rest of the repo's harnesses.
"""

from unittest.mock import patch

from absl.testing import absltest

from kaggle_environments import core_harness
from kaggle_environments.core_harness import ParseResult, create_agent_fn, set_telemetry_exporter
from kaggle_environments.envs.word_art.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


class _WordArtHarness:
    """Test-local GameHarness adapter; mirrors the prod wrapper shape."""

    def get_legal_moves(self, observation):
        return get_legal_moves(observation)

    def make_prompt(
        self,
        observation,
        move_history,
        previous_response=None,
        previous_action=None,
    ):
        return generate_prompt(observation, move_history, previous_response, previous_action)

    def parse_response(self, response, legal_action_strings, *, observation=None):
        # Module-level parser needs the observation to dispatch on role.
        return parse_response(response, legal_action_strings, observation=observation)


# --- Observation helpers ----------------------------------------------------


def _artist_obs(team="blue", target="ELEPHANT", **overrides):
    obs = {
        "num_rounds": 4,
        "max_attempts": 3,
        "first_try_bonus": 1,
        "max_art_chars": 4000,
        "current_round": 0,
        "phase": "art",
        "role": "artist",
        "team": team,
        "target_word": target,
        "teammate_art": "",
        "previous_guesses": [],
        "attempts_remaining": 0,
        "blue_score": 0,
        "yellow_score": 0,
        "blue_attempts_used": 0,
        "yellow_attempts_used": 0,
        "history": [],
    }
    obs.update(overrides)
    return obs


def _guesser_obs(team="blue", art=" _\n( o.o)", attempt=1, prev_guesses=(), **overrides):
    max_attempts = overrides.pop("max_attempts", 3)
    obs = {
        "num_rounds": 4,
        "max_attempts": max_attempts,
        "first_try_bonus": 1,
        "max_art_chars": 4000,
        "current_round": 0,
        "phase": "guess",
        "role": "guesser",
        "team": team,
        "target_word": "",
        "teammate_art": art,
        "previous_guesses": list(prev_guesses),
        "attempts_remaining": max_attempts - (attempt - 1),
        "blue_score": 0,
        "yellow_score": 0,
        "blue_attempts_used": (attempt - 1) if team == "blue" else 0,
        "yellow_attempts_used": (attempt - 1) if team == "yellow" else 0,
        "history": [],
    }
    obs.update(overrides)
    return obs


# --- LLM mock plumbing ------------------------------------------------------


class _StreamDelta:
    def __init__(self, content):
        self.content = content


class _StreamChoice:
    def __init__(self, content, finish_reason=None):
        self.delta = _StreamDelta(content)
        self.finish_reason = finish_reason


class _StreamChunk:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage


class _StreamUsage:
    prompt_tokens = 1
    completion_tokens = 1
    total_tokens = 2
    completion_tokens_details = None


def _fake_completion(content: str):
    return [
        _StreamChunk([_StreamChoice(content)]),
        _StreamChunk([_StreamChoice("", finish_reason="stop")]),
        _StreamChunk([], usage=_StreamUsage()),
    ]


_ENV = {
    "MODEL_NAME": "test-model",
    "MODEL_PROXY_KEY": "key",
    "MODEL_PROXY_URL": "dummy_url",
}


# --- get_legal_moves --------------------------------------------------------


class GetLegalMovesTest(absltest.TestCase):
    def test_artist_returns_none(self):
        self.assertIsNone(get_legal_moves(_artist_obs()))

    def test_guesser_returns_none(self):
        self.assertIsNone(get_legal_moves(_guesser_obs()))

    def test_empty_obs_returns_none(self):
        # Even the no-role probe is free-form; we never enumerate moves.
        self.assertIsNone(get_legal_moves({}))


# --- parse_response: artist (<art>...</art> tags) ---------------------------


class ParseResponseArtistTest(absltest.TestCase):
    def test_extracts_verbatim_art(self):
        """No JSON-escape gymnastics -- literal newlines / backslashes / quotes
        inside the tag pass through unchanged."""
        obs = _artist_obs()
        response = 'Reasoning: a cat.\n<art>\n /\\_/\\\n( o.o )\n > ^ <\n</art>'
        result = parse_response(response, None, observation=obs)
        self.assertEqual(result.submission, "\n /\\_/\\\n( o.o )\n > ^ <\n")

    def test_picks_last_of_multiple_art_blocks(self):
        """Model self-corrects: the earlier block is the rejected draft,
        the trailing block is the intent."""
        obs = _artist_obs()
        response = 'Draft:\n<art>DRAFT_DRAWING</art>\nActually, revised:\n<art>FINAL_DRAWING</art>'
        result = parse_response(response, None, observation=obs)
        self.assertEqual(result.submission, "FINAL_DRAWING")

    def test_tolerates_tag_case_and_whitespace_variants(self):
        for opener, closer in [
            ("<art>", "</art>"),
            ("<Art>", "</Art>"),
            ("<ART>", "</ART>"),
            ("< art >", "< / art >"),
            ("<art >", "</ art>"),
        ]:
            obs = _artist_obs()
            response = f"Prose.\n{opener}MEOW{closer}"
            result = parse_response(response, None, observation=obs)
            self.assertEqual(
                result.submission, "MEOW",
                msg=f"Failed on {opener!r}/{closer!r}",
            )

    def test_no_tag_returns_no_submission(self):
        obs = _artist_obs()
        response = "Here's a drawing of a cat: ^.^"
        result = parse_response(response, None, observation=obs)
        self.assertIsNone(result.submission)
        # No answer marker at all -> raw_action=None -> UNPARSABLE telemetry.
        self.assertIsNone(result.raw_action)

    def test_empty_tag_surfaces_raw_action(self):
        """Model wrote the tag but left it empty -- the rethink prompt should
        be able to quote it back."""
        obs = _artist_obs()
        for response in (
            "<art></art>",
            "<art>   </art>",
            "<art>\n\n</art>",
        ):
            result = parse_response(response, None, observation=obs)
            self.assertIsNone(result.submission, msg=repr(response))
            self.assertIsNotNone(result.raw_action, msg=repr(response))

    def test_wrong_role_marker_returns_no_submission(self):
        """Artist emitted a guesser-style JSON instead of an <art> tag -- no
        submission, raw_action=None (no <art> tag exists to quote)."""
        obs = _artist_obs()
        response = '{"guess": "ELEPHANT"}'
        result = parse_response(response, None, observation=obs)
        self.assertIsNone(result.submission)
        self.assertIsNone(result.raw_action)


# --- parse_response: guesser (JSON with "guess" key) ------------------------


class ParseResponseGuesserTest(absltest.TestCase):
    def test_extracts_guess_fenced_json(self):
        obs = _guesser_obs()
        response = 'Prose reasoning here.\n```json\n{"guess": "CAT"}\n```'
        result = parse_response(response, None, observation=obs)
        self.assertEqual(result.submission, "CAT")

    def test_extracts_guess_bare_json(self):
        obs = _guesser_obs()
        response = 'Reasoning: whiskers suggest cat. {"guess": "CAT"}'
        result = parse_response(response, None, observation=obs)
        self.assertEqual(result.submission, "CAT")

    def test_picks_last_json_block(self):
        obs = _guesser_obs()
        response = 'Maybe {"guess": "DOG"} but actually {"guess": "CAT"}'
        result = parse_response(response, None, observation=obs)
        self.assertEqual(result.submission, "CAT")

    def test_rejects_non_string_guess(self):
        # A number (or any non-string) in the guess slot is not a submission.
        # The parser deliberately does NOT coerce -- the model said something
        # structurally wrong and should get a rethink.
        obs = _guesser_obs()
        response = '{"guess": 42}'
        result = parse_response(response, None, observation=obs)
        self.assertIsNone(result.submission)
        self.assertIsNotNone(result.raw_action)
        self.assertIn("42", result.raw_action)

    def test_rejects_empty_guess(self):
        obs = _guesser_obs()
        response = '{"guess": ""}'
        result = parse_response(response, None, observation=obs)
        self.assertIsNone(result.submission)
        self.assertIsNotNone(result.raw_action)

    def test_rejects_whitespace_only_guess(self):
        obs = _guesser_obs()
        response = '{"guess": "   "}'
        result = parse_response(response, None, observation=obs)
        self.assertIsNone(result.submission)
        self.assertIsNotNone(result.raw_action)

    def test_missing_guess_key_returns_no_submission(self):
        obs = _guesser_obs()
        response = '{"note": "no idea", "other": "still"}'
        result = parse_response(response, None, observation=obs)
        self.assertIsNone(result.submission)
        # No JSON with "guess" key -> raw_action=None -> UNPARSABLE.
        self.assertIsNone(result.raw_action)

    def test_no_json_returns_no_submission(self):
        obs = _guesser_obs()
        response = "The art clearly shows a CAT. My answer is CAT."
        result = parse_response(response, None, observation=obs)
        self.assertIsNone(result.submission)
        self.assertIsNone(result.raw_action)


# --- parse_response: dispatch / no-role -------------------------------------


class ParseResponseDispatchTest(absltest.TestCase):
    def test_no_observation_refuses_to_submit(self):
        # Parser is role-strict. Without a role we can't tell which marker
        # format to look for, so we refuse. In production core_harness
        # always forwards `observation`; this only fires from ad-hoc test
        # callers.
        for response in ("<art>X</art>", '{"guess": "CAT"}'):
            result = parse_response(response, None)
            self.assertIsNone(result.submission)
            self.assertIsNone(result.raw_action)

    def test_prose_returns_no_submission(self):
        result = parse_response("Just some text", None, observation=_guesser_obs())
        self.assertIsNone(result.submission)
        self.assertIsNone(result.raw_action)


# --- Thoughts extraction ---------------------------------------------------


class ThoughtsExtractionTest(absltest.TestCase):
    """Prose reasoning that precedes the answer marker must be captured in
    ``ParseResult.thoughts`` so the replay records reasoning separately
    from the submitted answer. Without this, core_harness falls back to
    logging the full raw response and post-hoc analysis has to re-parse
    it to separate reasoning from action."""

    def test_artist_captures_prose_before_art_tag(self):
        obs = _artist_obs()
        response = (
            "I'll draw a cat face: pointy ears with slashes, round eyes.\n"
            "<art>\n /\\_/\\\n( o.o )\n</art>"
        )
        result = parse_response(response, None, observation=obs)
        self.assertIsNotNone(result.thoughts)
        self.assertIn("cat face", result.thoughts)
        self.assertIn("pointy ears", result.thoughts)
        # Answer content must NOT leak into thoughts.
        self.assertNotIn("<art>", result.thoughts)
        self.assertNotIn("/\\_/\\", result.thoughts)

    def test_artist_thoughts_stop_at_last_art_tag_on_rethink(self):
        """When the model self-corrects with a second <art> block, thoughts
        should include the earlier draft (it IS reasoning) but not the
        final block itself."""
        obs = _artist_obs()
        response = (
            "Draft:\n<art>DRAFT</art>\n"
            "Actually, revised approach:\n<art>FINAL</art>"
        )
        result = parse_response(response, None, observation=obs)
        self.assertEqual(result.submission, "FINAL")
        self.assertIsNotNone(result.thoughts)
        self.assertIn("Draft", result.thoughts)
        self.assertIn("<art>DRAFT</art>", result.thoughts)
        self.assertIn("revised", result.thoughts)
        # The winning block itself must NOT be in thoughts.
        self.assertNotIn("FINAL", result.thoughts)

    def test_artist_no_prose_leaves_thoughts_none(self):
        """Model wrote only the answer -- fallback to logging the full
        raw response (core_harness handles that when thoughts=None)."""
        obs = _artist_obs()
        response = "<art>MEOW</art>"
        result = parse_response(response, None, observation=obs)
        self.assertEqual(result.submission, "MEOW")
        self.assertIsNone(result.thoughts)

    def test_artist_whitespace_only_prose_leaves_thoughts_none(self):
        obs = _artist_obs()
        response = "   \n\n  <art>MEOW</art>"
        result = parse_response(response, None, observation=obs)
        self.assertIsNone(result.thoughts)

    def test_guesser_captures_prose_before_json(self):
        obs = _guesser_obs()
        response = (
            'Looks like a four-legged animal with whiskers. My guess: cat.\n'
            '{"guess": "CAT"}'
        )
        result = parse_response(response, None, observation=obs)
        self.assertIsNotNone(result.thoughts)
        self.assertIn("four-legged", result.thoughts)
        self.assertIn("whiskers", result.thoughts)
        # Answer JSON must NOT leak into thoughts.
        self.assertNotIn('"guess"', result.thoughts)

    def test_guesser_thoughts_stop_at_last_json_on_rethink(self):
        obs = _guesser_obs()
        response = (
            'Maybe {"guess": "DOG"}\n'
            'Wait -- the whiskers point to a cat instead.\n'
            '{"guess": "CAT"}'
        )
        result = parse_response(response, None, observation=obs)
        self.assertEqual(result.submission, "CAT")
        self.assertIsNotNone(result.thoughts)
        self.assertIn("DOG", result.thoughts)
        self.assertIn("Wait", result.thoughts)
        # The winning JSON block itself must NOT be in thoughts.
        self.assertNotIn('"CAT"', result.thoughts)

    def test_guesser_fenced_json_captures_prose(self):
        obs = _guesser_obs()
        response = (
            'Reasoning about the drawing.\n'
            '```json\n{"guess": "CAT"}\n```'
        )
        result = parse_response(response, None, observation=obs)
        self.assertIsNotNone(result.thoughts)
        self.assertIn("Reasoning", result.thoughts)
        self.assertNotIn("```", result.thoughts)

    def test_guesser_no_prose_leaves_thoughts_none(self):
        obs = _guesser_obs()
        response = '{"guess": "CAT"}'
        result = parse_response(response, None, observation=obs)
        self.assertEqual(result.submission, "CAT")
        self.assertIsNone(result.thoughts)

    def test_thoughts_preserved_when_answer_is_rejected(self):
        """Even on bad-value failures (empty tag, non-string guess), the
        prose reasoning should still make it into thoughts -- the model
        DID reason, we just couldn't extract a submission."""
        # Artist: empty tag.
        obs_a = _artist_obs()
        result = parse_response(
            "My drawing is coming up next.\n<art></art>",
            None,
            observation=obs_a,
        )
        self.assertIsNone(result.submission)
        self.assertIsNotNone(result.thoughts)
        self.assertIn("drawing", result.thoughts)

        # Guesser: non-string guess.
        obs_g = _guesser_obs()
        result = parse_response(
            'I think it might be a number.\n{"guess": 42}',
            None,
            observation=obs_g,
        )
        self.assertIsNone(result.submission)
        self.assertIsNotNone(result.thoughts)
        self.assertIn("number", result.thoughts)


# --- Parser regression: no ghost-substitution -------------------------------


class NoGhostFallbackTest(absltest.TestCase):
    """The free-form parser intentionally has NO prose fallback. If the
    model writes 'CAT' in prose but its JSON is missing/wrong, we MUST NOT
    silently submit CAT -- let the rethink loop handle it."""

    def test_guesser_prose_with_no_json_returns_nothing(self):
        obs = _guesser_obs()
        response = "The art clearly shows a CAT. My answer is CAT."
        result = parse_response(response, None, observation=obs)
        self.assertIsNone(result.submission)

    def test_guesser_prose_with_json_lacking_guess_returns_nothing(self):
        obs = _guesser_obs()
        response = 'I see a cat: CAT. {"note": "CAT"}'
        result = parse_response(response, None, observation=obs)
        self.assertIsNone(result.submission)

    def test_artist_prose_containing_ascii_art_but_no_tag(self):
        # Model drew something readable in prose but forgot the tag -- we
        # do NOT submit prose, forcing a rethink to comply with the format.
        obs = _artist_obs()
        response = "Here you go:\n /\\_/\\\n( o.o )"
        result = parse_response(response, None, observation=obs)
        self.assertIsNone(result.submission)


# --- ParseResult shape ------------------------------------------------------


class ParseResultShapeTest(absltest.TestCase):
    def test_artist_submission_is_str(self):
        result = parse_response(
            "<art>x</art>",
            None,
            observation=_artist_obs(),
        )
        self.assertIsInstance(result, ParseResult)
        self.assertIsInstance(result.submission, str)

    def test_guesser_submission_is_str(self):
        result = parse_response(
            '{"guess": "CAT"}',
            None,
            observation=_guesser_obs(),
        )
        self.assertIsInstance(result.submission, str)

    def test_success_raw_action_equals_submission_untruncated(self):
        """On a successful parse we set raw_action == submission verbatim
        (matches core_harness's parse_json_action convention). Prior
        versions capped this at [:200], which truncated most of an
        artist's drawing in telemetry for no clear benefit."""
        obs = _artist_obs()
        big_art = "X" * 3000  # well over the old 200 cap
        response = f"<art>{big_art}</art>"
        result = parse_response(response, None, observation=obs)
        self.assertEqual(result.submission, big_art)
        self.assertEqual(result.raw_action, big_art)


# --- generate_prompt --------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def test_artist_prompt_names_team_and_role(self):
        prompt = generate_prompt(_artist_obs(team="blue"), [])
        self.assertIn("ARTIST", prompt)
        self.assertIn("Blue", prompt)

    def test_artist_prompt_yellow_team(self):
        prompt = generate_prompt(_artist_obs(team="yellow"), [])
        self.assertIn("Yellow", prompt)

    def test_artist_prompt_contains_target_word(self):
        prompt = generate_prompt(_artist_obs(target="PIANO"), [])
        self.assertIn("PIANO", prompt)

    def test_artist_prompt_describes_scoring(self):
        prompt = generate_prompt(_artist_obs(), [])
        self.assertIn("first-try", prompt.lower())
        self.assertIn("Correct on attempt 1", prompt)
        self.assertIn("0 points", prompt)

    def test_artist_prompt_warns_about_writing_word_verbatim(self):
        prompt = generate_prompt(_artist_obs(target="ELEPHANT"), [])
        lower = prompt.lower()
        self.assertTrue("target-word" in lower or "target word" in lower)
        self.assertIn("engine-enforced", lower)
        self.assertIn("strips every non-alphanumeric", prompt)
        self.assertIn("reversed", prompt)
        self.assertIn("placeholder", prompt)

    def test_artist_prompt_broad_no_words_rule(self):
        prompt = generate_prompt(_artist_obs(), [])
        squished = " ".join(prompt.lower().split())
        self.assertIn("do not include any words", squished)
        self.assertIn("any-word check", squished)
        for kw in ("synonym", "label", "nato", "translation", "rhyme"):
            self.assertIn(kw, squished)
        self.assertIn("visual element", squished)

    def test_artist_prompt_requests_reasoning_before_art_tag(self):
        # Memory contract: every prompt asks for reasoning BEFORE the answer.
        # Now the answer is <art>...</art>; reasoning is prose above it.
        # The instruction must explicitly ask the model to WRITE the
        # reasoning out (not just "think" internally) so a reader can see
        # the chain of thought in the response.
        prompt = generate_prompt(_artist_obs(), [])
        lower = prompt.lower()
        squished = " ".join(lower.split())
        self.assertIn("think step by step", squished)
        # Parallel to the guesser prompt: "writing your reasoning as
        # ordinary prose" is what makes the CoT observable.
        self.assertIn("writing your reasoning as ordinary prose", squished)
        # And the reasoning instruction must precede the concrete example.
        instruct_pos = lower.find("<art>")
        self.assertGreater(instruct_pos, 0)
        self.assertLess(lower.find("think step by step"), instruct_pos)

    def test_artist_prompt_documents_art_tag_format(self):
        prompt = generate_prompt(_artist_obs(), [])
        # The instruction and the worked example must both be present so
        # the model knows the exact format expected.
        self.assertIn("<art>", prompt)
        self.assertIn("</art>", prompt)
        # And the prompt must call out that no escaping is needed (this is
        # the whole reason for using tags over JSON).
        self.assertIn("verbatim", prompt.lower())

    def test_guesser_prompt_shows_teammate_art(self):
        art = "  /\\_/\\\n ( o.o )\n  > ^ <"
        prompt = generate_prompt(_guesser_obs(art=art), [])
        self.assertIn(art, prompt)

    def test_guesser_prompt_first_attempt_advertises_bonus(self):
        prompt = generate_prompt(_guesser_obs(attempt=1), [])
        self.assertIn("first-try bonus", prompt)
        self.assertIn("attempt 1 of 3", prompt)

    def test_guesser_prompt_later_attempt_lists_previous_guesses(self):
        prompt = generate_prompt(
            _guesser_obs(attempt=2, prev_guesses=("DOG",)),
            [],
        )
        self.assertIn("'DOG'", prompt)
        self.assertIn("all wrong", prompt)
        self.assertIn("attempt 2 of 3", prompt)

    def test_guesser_prompt_requests_reasoning_before_json(self):
        prompt = generate_prompt(_guesser_obs(), [])
        lower = prompt.lower()
        self.assertIn("think step by step", lower)
        # Reasoning instruction must precede the JSON example.
        json_pos = prompt.find('{"guess"')
        self.assertGreater(json_pos, 0)
        self.assertLess(lower.find("think step by step"), json_pos)

    def test_guesser_prompt_example_shows_json_format(self):
        prompt = generate_prompt(_guesser_obs(), [])
        # Example must include a minimal valid JSON showing only the
        # required key -- reasoning is prose now, no "thinking" field.
        self.assertIn('{"guess": "CAT"}', prompt)

    def test_history_block_renders_completed_rounds(self):
        hist = [
            {
                "word": "CAT",
                "blue_art": "MEOW",
                "blue_guesses": ["CAT"],
                "blue_points": 2,
                "yellow_art": "GRR",
                "yellow_guesses": ["DOG", "BEAR", "LION"],
                "yellow_points": 0,
            }
        ]
        prompt = generate_prompt(_guesser_obs(history=hist, current_round=1), [])
        self.assertIn("Round 1", prompt)
        self.assertIn("'CAT'", prompt)
        self.assertIn("DOG", prompt)
        self.assertIn("2 points", prompt)

    def test_score_line_shows_current_round_and_scores(self):
        prompt = generate_prompt(
            _guesser_obs(current_round=2, blue_score=4, yellow_score=2),
            [],
        )
        self.assertIn("round 3 of 4", prompt)
        self.assertIn("Blue 4", prompt)
        self.assertIn("Yellow 2", prompt)

    def test_rethink_not_appended_on_first_attempt(self):
        prompt = generate_prompt(_artist_obs(), [])
        self.assertNotIn("Last 500 characters", prompt)
        self.assertNotIn("Your submitted", prompt)

    def test_artist_rethink_no_answer_branch(self):
        # No <art> tag found -> quote back the tail of the response.
        prompt = generate_prompt(
            _artist_obs(),
            [],
            previous_response="my last try was junk text",
        )
        self.assertIn("Last 500 characters", prompt)
        self.assertIn("my last try was junk text", prompt)
        self.assertIn("<art>", prompt)
        # Must NOT use the empty-tag branch's phrasing.
        self.assertNotIn("were empty", prompt)

    def test_artist_rethink_empty_tag_branch(self):
        # <art> tag present but empty -> quote it back verbatim.
        prompt = generate_prompt(
            _artist_obs(),
            [],
            previous_response="here you go: <art></art>",
            previous_action="<art></art>",
        )
        self.assertIn("empty or whitespace-only", prompt)
        self.assertIn("<art></art>", prompt)
        # Must NOT use the no-answer branch's phrasing.
        self.assertNotIn("Last 500 characters", prompt)

    def test_guesser_rethink_no_json_branch(self):
        prompt = generate_prompt(
            _guesser_obs(),
            [],
            previous_response="I think it's a cat but I'm not sure",
        )
        self.assertIn("Last 500 characters", prompt)
        self.assertIn("I think it's a cat", prompt)
        self.assertIn('"guess"', prompt)
        # Must NOT use the bad-value branch's phrasing.
        self.assertNotIn("Your submitted JSON was", prompt)

    def test_guesser_rethink_bad_value_branch(self):
        prompt = generate_prompt(
            _guesser_obs(),
            [],
            previous_response='reasoning ... {"guess": 42}',
            previous_action='{"guess": 42}',
        )
        squished = " ".join(prompt.split())
        self.assertIn("Your submitted JSON was", squished)
        self.assertIn('{"guess": 42}', prompt)
        # Must NOT use the no-json branch's phrasing.
        self.assertNotIn("Last 500 characters", prompt)

    def test_rethink_dispatch_respects_role(self):
        # An artist and a guesser with the same previous_response should
        # get role-specific rethink text (different format hints).
        artist = generate_prompt(_artist_obs(), [], previous_response="junk")
        guesser = generate_prompt(_guesser_obs(), [], previous_response="junk")
        self.assertIn("<art>", artist)
        self.assertNotIn('"guess"', artist.split("Rules:")[-1].split("Past rounds")[-1])
        # (Guesser rethink specifically mentions the JSON format.)
        self.assertIn('"guess"', guesser)

    def test_max_attempts_propagates_to_prompt(self):
        prompt = generate_prompt(_artist_obs(max_attempts=5), [])
        self.assertIn("attempt 2 through 5", prompt)
        self.assertIn("within 5 attempts", prompt)

    def test_first_try_bonus_propagates_from_observation(self):
        prompt = generate_prompt(_artist_obs(first_try_bonus=5), [])
        self.assertIn("Correct on attempt 1: 6 points", prompt)
        self.assertIn("1 base + 5 first-try bonus", prompt)

    def test_first_try_bonus_zero_renders_correctly(self):
        prompt = generate_prompt(_guesser_obs(first_try_bonus=0), [])
        self.assertIn("Correct on attempt 1: 1 points", prompt)
        self.assertIn("1 base + 0 first-try bonus", prompt)

    def test_artist_prompt_mentions_max_art_chars_truncation(self):
        prompt = generate_prompt(_artist_obs(max_art_chars=2500), [])
        self.assertIn("2500", prompt)
        self.assertIn("truncated", prompt.lower())

    def test_scoring_block_states_win_and_tie_conditions(self):
        for obs in (_artist_obs(), _guesser_obs()):
            prompt = generate_prompt(obs, [])
            lower = prompt.lower()
            self.assertIn("higher total wins", lower)
            self.assertIn("tie", lower)

    def test_scoring_block_states_teams_share_secret_word(self):
        for obs in (_artist_obs(), _guesser_obs()):
            prompt = generate_prompt(obs, [])
            self.assertIn("same secret word", prompt)

    def test_match_rule_wording_is_consistent_across_roles(self):
        artist = " ".join(generate_prompt(_artist_obs(), []).split())
        guesser = " ".join(generate_prompt(_guesser_obs(), []).split())
        phrase = "no plurals, synonyms, partial matches, or other spelling variants"
        self.assertIn(phrase, artist)
        self.assertIn(phrase, guesser)

    def test_guesser_prompt_explains_disqualification_marker(self):
        prompt = generate_prompt(_guesser_obs(), [])
        self.assertIn("disqualif", prompt.lower())
        self.assertIn("placeholder", prompt.lower())
        self.assertIn("target word", prompt.lower())

    def test_history_marks_disqualified_blue_entry(self):
        hist = [
            {
                "word": "CAT",
                "blue_art": "C A T",
                "blue_art_disqualified": True,
                "blue_art_disqualification_reason": "target_word",
                "blue_guesses": ["DOG", "BEAR", "LION"],
                "blue_points": 0,
                "yellow_art": "MEOW",
                "yellow_art_disqualified": False,
                "yellow_art_disqualification_reason": None,
                "yellow_guesses": ["CAT"],
                "yellow_points": 2,
            }
        ]
        prompt = generate_prompt(_guesser_obs(history=hist, current_round=1), [])
        self.assertIn("Blue art: (DISQUALIFIED", prompt)
        self.assertIn("contained the target word", prompt)
        self.assertIn("placeholder", prompt.lower())
        self.assertIn("Yellow art:", prompt)
        self.assertNotIn("Yellow art: (DISQUALIFIED", prompt)

    def test_history_marks_disqualified_yellow_entry(self):
        hist = [
            {
                "word": "DOG",
                "blue_art": "WOOF",
                "blue_art_disqualified": False,
                "blue_art_disqualification_reason": None,
                "blue_guesses": ["DOG"],
                "blue_points": 2,
                "yellow_art": "the dog runs",
                "yellow_art_disqualified": True,
                "yellow_art_disqualification_reason": "contains_words",
                "yellow_guesses": ["WOLF", "FOX"],
                "yellow_points": 0,
            }
        ]
        prompt = generate_prompt(_artist_obs(history=hist, current_round=1), [])
        self.assertIn("Yellow art: (DISQUALIFIED", prompt)
        self.assertIn("contained text", prompt)
        self.assertNotIn("Blue art: (DISQUALIFIED", prompt)


# --- AgentIntegrationTest ---------------------------------------------------


class AgentIntegrationTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.events: list[dict] = []
        set_telemetry_exporter(lambda module, **kw: self.events.append({"module": module, **kw}))

    def tearDown(self):
        set_telemetry_exporter(lambda module, **kwargs: None)
        super().tearDown()

    def test_artist_successful_submission(self):
        agent = create_agent_fn(_WordArtHarness())
        obs = _artist_obs(target="CAT")
        llm = "A cat face.\n<art>\n /\\_/\\\n( o.o )\n</art>"
        with (
            patch.dict("os.environ", _ENV, clear=False),
            patch.object(
                core_harness.litellm,
                "completion",
                return_value=_fake_completion(llm),
            ),
        ):
            result = agent(obs, {"freeForm": True})
        self.assertEqual(result["submission"], "\n /\\_/\\\n( o.o )\n")
        self.assertEqual(result["status"], "OK")

    def test_guesser_successful_submission(self):
        agent = create_agent_fn(_WordArtHarness())
        obs = _guesser_obs(art="MEOW")
        llm = 'Looks like a cat.\n{"guess": "CAT"}'
        with (
            patch.dict("os.environ", _ENV, clear=False),
            patch.object(
                core_harness.litellm,
                "completion",
                return_value=_fake_completion(llm),
            ),
        ):
            result = agent(obs, {"freeForm": True})
        self.assertEqual(result["submission"], "CAT")
        self.assertEqual(result["status"], "OK")

    def test_retry_after_unparseable_then_succeeds(self):
        agent = create_agent_fn(_WordArtHarness(), max_retries=3)
        obs = _guesser_obs()
        responses = [
            _fake_completion("I think it might be a cat"),  # no JSON
            _fake_completion('Looks like a cat.\n{"guess": "CAT"}'),
        ]
        with (
            patch.dict("os.environ", _ENV, clear=False),
            patch.object(
                core_harness.litellm,
                "completion",
                side_effect=responses,
            ) as mock_call,
        ):
            result = agent(obs, {"freeForm": True})
        self.assertEqual(result["submission"], "CAT")
        self.assertEqual(mock_call.call_count, 2)

    def test_artist_missing_tag_triggers_rethink(self):
        agent = create_agent_fn(_WordArtHarness(), max_retries=2)
        obs = _artist_obs()
        responses = [
            _fake_completion("Here's my drawing:\n /\\_/\\"),  # no <art> tag
            _fake_completion("Retry.\n<art>MEOW</art>"),
        ]
        with (
            patch.dict("os.environ", _ENV, clear=False),
            patch.object(
                core_harness.litellm,
                "completion",
                side_effect=responses,
            ) as mock_call,
        ):
            result = agent(obs, {"freeForm": True})
        self.assertEqual(result["submission"], "MEOW")
        self.assertEqual(mock_call.call_count, 2)

    def test_artist_no_valid_response_raises_after_retries(self):
        agent = create_agent_fn(_WordArtHarness(), max_retries=1)
        obs = _artist_obs()
        responses = [_fake_completion("Just prose, no tag.")]
        with (
            patch.dict("os.environ", _ENV, clear=False),
            patch.object(
                core_harness.litellm,
                "completion",
                side_effect=responses,
            ),
        ):
            with self.assertRaises(ValueError):
                agent(obs, {"freeForm": True})


if __name__ == "__main__":
    absltest.main()
