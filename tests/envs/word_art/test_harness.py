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
    max_attempts = overrides.get("max_attempts", 3)
    obs = {
        "num_rounds": 4,
        "max_attempts": max_attempts,
        "guess_points": [1] * max_attempts,
        "include_art_history": True,
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
        "guess_points": [1] * max_attempts,
        "include_art_history": True,
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
        response = 'Draft:\n<art>^_^</art>\nActually, revised:\n<art>>_<</art>'
        result = parse_response(response, None, observation=obs)
        self.assertEqual(result.submission, ">_<")

    def test_tolerates_tag_case_and_whitespace_variants(self):
        for opener, closer in [
            ("<art>", "</art>"),
            ("<Art>", "</Art>"),
            ("<ART>", "</ART>"),
            ("< art >", "< / art >"),
            ("<art >", "</ art>"),
        ]:
            obs = _artist_obs()
            response = f"Prose.\n{opener}( o.o ){closer}"
            result = parse_response(response, None, observation=obs)
            self.assertEqual(
                result.submission, "( o.o )",
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

    def test_stray_art_tag_in_prose_does_not_swallow_reasoning(self):
        """An unpaired `<art>` mentioned in the prose must not bind to the
        real drawing's closing tag -- that would submit the reasoning as
        part of the art."""
        obs = _artist_obs()
        response = (
            "I will use <art> tags for this. Drawing a cat face now.\n"
            "<art>\n( o.o )\n</art>"
        )
        result = parse_response(response, None, observation=obs)
        self.assertEqual(result.submission, "\n( o.o )\n")
        self.assertIn("tags for this", result.thoughts)

    def test_artist_thoughts_stop_at_last_art_tag_on_rethink(self):
        """When the model self-corrects with a second <art> block, thoughts
        should include the earlier draft (it IS reasoning) but not the
        final block itself."""
        obs = _artist_obs()
        response = (
            "Draft:\n<art>^_^</art>\n"
            "Actually, revised approach:\n<art>>_<</art>"
        )
        result = parse_response(response, None, observation=obs)
        self.assertEqual(result.submission, ">_<")
        self.assertIsNotNone(result.thoughts)
        self.assertIn("Draft", result.thoughts)
        self.assertIn("<art>^_^</art>", result.thoughts)
        self.assertIn("revised", result.thoughts)
        # The winning block itself must NOT be in thoughts.
        self.assertNotIn(">_<", result.thoughts)

    def test_artist_no_prose_leaves_thoughts_none(self):
        """Model wrote only the answer -- fallback to logging the full
        raw response (core_harness handles that when thoughts=None)."""
        obs = _artist_obs()
        response = "<art>( o.o )</art>"
        result = parse_response(response, None, observation=obs)
        self.assertEqual(result.submission, "( o.o )")
        self.assertIsNone(result.thoughts)

    def test_artist_whitespace_only_prose_leaves_thoughts_none(self):
        obs = _artist_obs()
        response = "   \n\n  <art>( o.o )</art>"
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
        self.assertIn("Scoring (per round, per team):", prompt)
        self.assertIn("Correct on attempt 1", prompt)
        self.assertIn("No correct guess within 3 attempts: 0 pts", prompt)

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

    def test_guesser_prompt_first_attempt_advertises_current_reward(self):
        # Custom guess_points so the attempt-1 reward differs from the
        # default (1 pt) — otherwise the pitch line looks identical to
        # every other attempt.
        prompt = generate_prompt(
            _guesser_obs(attempt=1, guess_points=[3, 2, 1]), [],
        )
        self.assertIn("attempt 1 of 3", prompt)
        self.assertIn("A correct guess NOW scores 3 pts", prompt)

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
        # Any of several equivalent prose-reasoning phrases is fine; the
        # invariant is that a WRITE-your-reasoning instruction PRECEDES
        # the JSON example. The verb must be explicit ("write", "think
        # step by step") so reasoning models produce observable
        # chain-of-thought in the response, not just internal thinking.
        reasoning_pos = min(
            (lower.find(p) for p in (
                "write your reasoning",
                "writing your reasoning",
                "think step by step",
            ) if lower.find(p) >= 0),
            default=-1,
        )
        self.assertGreater(reasoning_pos, 0, "guesser prompt must instruct the model to write reasoning before the JSON")
        json_pos = prompt.find('{"guess"')
        self.assertGreater(json_pos, 0)
        self.assertLess(reasoning_pos, json_pos)

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
        self.assertIn("2 pts", prompt)

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
        # max_attempts=5 → scoring block enumerates attempts 1..5 explicitly
        # and the "no correct guess" line names the same bound. Every attempt
        # gets a line in the table so the "attempt 2 through N" range shorthand
        # is gone.
        prompt = generate_prompt(_artist_obs(max_attempts=5), [])
        self.assertIn("Correct on attempt 2:", prompt)
        self.assertIn("Correct on attempt 5:", prompt)
        self.assertIn("within 5 attempts", prompt)

    def test_guess_points_propagate_from_observation(self):
        prompt = generate_prompt(_artist_obs(guess_points=[6, 2, 1]), [])
        self.assertIn("Correct on attempt 1: 6 pts", prompt)
        self.assertIn("Correct on attempt 2: 2 pts", prompt)
        self.assertIn("Correct on attempt 3: 1 pt", prompt)
        # Legacy "1 base + N first-try bonus" wording must be gone.
        self.assertNotIn("first-try bonus", prompt)
        self.assertNotIn("1 base +", prompt)

    def test_guess_points_fractional_render(self):
        prompt = generate_prompt(_guesser_obs(guess_points=[2, 1.5, 1]), [])
        self.assertIn("Correct on attempt 1: 2 pts", prompt)
        # Integer-valued floats render without a `.0` suffix.
        self.assertNotIn("2.0 pts", prompt)
        # Fractional values render naturally.
        self.assertIn("1.5 pts", prompt)
        self.assertIn("Correct on attempt 3: 1 pt", prompt)

    def test_guess_points_default_all_ones(self):
        prompt = generate_prompt(_artist_obs(), [])
        self.assertIn("Correct on attempt 1: 1 pt", prompt)
        self.assertIn("Correct on attempt 2: 1 pt", prompt)
        self.assertIn("Correct on attempt 3: 1 pt", prompt)
        # No leftover bonus wording under the default schedule.
        self.assertNotIn("first-try bonus", prompt)

    def test_artist_prompt_mentions_max_art_chars_truncation(self):
        prompt = generate_prompt(_artist_obs(max_art_chars=2500), [])
        self.assertIn("2500", prompt)
        self.assertIn("truncated", prompt.lower())

    def test_artist_prompt_warns_non_ascii_letters_are_dropped(self):
        """The sanitizer deletes them before the anti-text checks run, so a
        model reaching for Cyrillic homoglyphs loses the glyphs outright.
        The prompt has to say so or that failure is unattributable."""
        lower = generate_prompt(_artist_obs(), []).lower()
        self.assertIn("non-ascii", lower)
        self.assertIn("cyrillic", lower)

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
        # Both prompts must state the plural leniency AND the non-goals
        # (synonyms/tenses/spelling variants) so the model has consistent
        # expectations regardless of role.
        for prompt in (artist, guesser):
            self.assertIn("accepts singular/plural equivalents", prompt)
            self.assertIn("Synonyms, tenses, and other spelling variants don't count", prompt)

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
        self.assertIn("contained a text label", prompt)
        self.assertNotIn("Blue art: (DISQUALIFIED", prompt)

    def test_include_art_history_true_shows_art_body(self):
        """Default include_art_history=True: past-round ASCII art appears
        indented under the round header, matching the pre-toggle behaviour."""
        hist = [
            {
                "word": "SUN",
                "blue_art": "  \\|/\n --*--\n  /|\\",
                "blue_art_disqualified": False,
                "blue_art_disqualification_reason": None,
                "blue_guesses": ["SUN"],
                "blue_points": 2,
                "yellow_art": "((()))",
                "yellow_art_disqualified": False,
                "yellow_art_disqualification_reason": None,
                "yellow_guesses": ["SUN"],
                "yellow_points": 2,
            }
        ]
        prompt = generate_prompt(_artist_obs(history=hist, current_round=1), [])
        # Round header always renders.
        self.assertIn("Round 1: word was 'SUN'.", prompt)
        # Art body characters appear (the unique '--*--' spine and '((()))').
        self.assertIn("--*--", prompt)
        self.assertIn("((()))", prompt)
        # No suppression messaging appears.
        self.assertNotIn("art body omitted", prompt)
        self.assertNotIn("omitted for brevity", prompt)

    def test_include_art_history_false_omits_art_body(self):
        """include_art_history=False: no raw art body, but word, guesses,
        points, and disqualification annotations still render."""
        hist = [
            {
                "word": "MOON",
                "blue_art": "  ,--.\n /    \\\n \\    /\n  `--'",
                "blue_art_disqualified": False,
                "blue_art_disqualification_reason": None,
                "blue_guesses": ["MOON"],
                "blue_points": 2,
                # Yellow labelled their art — engine flagged contains_words.
                "yellow_art": "the M O O N is round",
                "yellow_art_disqualified": True,
                "yellow_art_disqualification_reason": "contains_words",
                "yellow_guesses": ["STAR", "PLANET", "MOON"],
                "yellow_points": 1,
            }
        ]
        prompt = generate_prompt(
            _artist_obs(history=hist, current_round=1, include_art_history=False), [],
        )
        # Round header, per-team guesses, points, and word all render.
        self.assertIn("Round 1: word was 'MOON'.", prompt)
        self.assertIn("Blue guesses: ['MOON'] -> 2 pts", prompt)
        self.assertIn("['STAR', 'PLANET', 'MOON'] -> 1 pt", prompt)
        # Disqualification annotation remains as a one-liner.
        self.assertIn("Yellow art: DISQUALIFIED", prompt)
        self.assertIn("contained a text label", prompt)
        self.assertIn("art body omitted", prompt)
        self.assertIn("Blue art: (omitted for brevity)", prompt)
        # Raw art body characters (from either team) must NOT appear.
        self.assertNotIn("`--'", prompt)
        self.assertNotIn(",--.", prompt)
        self.assertNotIn("the M O O N is round", prompt)

    def test_include_art_history_false_with_empty_history(self):
        """With no history yet, the toggle is inert — the placeholder line
        renders unchanged for both settings."""
        for flag in (True, False):
            prompt = generate_prompt(
                _artist_obs(history=[], include_art_history=flag), [],
            )
            self.assertIn("No rounds completed yet.", prompt)

    def test_running_score_line_matches_pts_formatting(self):
        """Fractional guess_points make blue_score / yellow_score floats.
        The running-score line must use the same formatting convention as
        history lines and the scoring table (`_format_points`), so a value
        of 3.0 renders as `3` (not `3.0`) and a fractional value renders
        naturally. Otherwise the same prompt shows the same quantity in
        two different formats."""
        prompt = generate_prompt(
            _artist_obs(blue_score=3.0, yellow_score=1.5, current_round=2), [],
        )
        self.assertIn("Current score: Blue 3 - Yellow 1.5.", prompt)
        self.assertNotIn("Blue 3.0", prompt)

    def test_opposing_team_visibility_rule_is_scoped_to_live_round(self):
        """Both role prompts must not claim the opposing team's drawing is
        permanently private — completed rounds are shared via the history
        block, so an unqualified "cannot see" claim contradicts what the
        model sees ~40 lines below in the same prompt. Both prompts must
        include the "during the live round" scoping; no unqualified
        negatives allowed."""
        for prompt in (
            generate_prompt(_artist_obs(), []),
            generate_prompt(_guesser_obs(), []),
        ):
            squished = " ".join(prompt.split())
            self.assertIn("during the live round", squished)
            # No unqualified negative that a past-art reader would find false.
            self.assertNotIn("cannot see your drawing (nor you theirs)", squished)
            self.assertNotIn("and cannot see your art or guesses.", squished)
        # The artist prompt names the history block explicitly (it's the
        # primary reader of past-round drawings). The guesser prompt doesn't
        # need to -- the history block below it has its own header.
        artist_prompt = " ".join(generate_prompt(_artist_obs(), []).split())
        self.assertIn("history block below", artist_prompt)

    def test_include_art_history_false_forfeit_and_omitted_differ(self):
        """Under include_art_history=False, an artist forfeit (empty art)
        must render distinctly from a real drawing that was merely omitted
        for brevity. Otherwise the model reasoning about a 0-point round
        would conflate "teammate drew something unreadable" with "teammate
        submitted nothing.\""""
        hist = [
            {
                "word": "MOON",
                "blue_art": "  ,--.\n /    \\\n \\    /\n  `--'",  # real drawing
                "blue_art_disqualified": False,
                "blue_art_disqualification_reason": None,
                "blue_guesses": ["MOON"],
                "blue_points": 2,
                "yellow_art": "",  # forfeit / timeout — no art submitted
                "yellow_art_disqualified": False,
                "yellow_art_disqualification_reason": None,
                "yellow_guesses": [],
                "yellow_points": 0,
            }
        ]
        prompt = generate_prompt(
            _artist_obs(history=hist, current_round=1, include_art_history=False), [],
        )
        self.assertIn("Blue art: (omitted for brevity)", prompt)
        self.assertIn("Yellow art: (nothing submitted)", prompt)
        # The two states must not collide onto the same line.
        self.assertNotIn("Yellow art: (omitted for brevity)", prompt)


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
            _fake_completion("Retry.\n<art>( o.o )</art>"),
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
        self.assertEqual(result["submission"], "( o.o )")
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


class ArtPreValidationTest(absltest.TestCase):
    """The engine disqualifies text-bearing art AFTER the turn is committed,
    which is worth 0 and unrecoverable. Pre-validating in the parser turns
    that into a retry the artist can actually act on."""

    def test_text_bearing_art_is_withheld_from_submission(self):
        obs = _artist_obs(target="CHEEKBONE")
        result = parse_response("Prose.\n<art>\n  O  <-- eye\n /|\\\n</art>", None, observation=obs)
        self.assertIsNone(result.submission)
        # raw_action carries the drawing so the rethink can quote it back
        # (and so telemetry categorizes this ILLEGAL, not UNPARSABLE).
        self.assertIn("eye", result.raw_action)

    def test_target_word_art_is_withheld_from_submission(self):
        obs = _artist_obs(target="CAT")
        result = parse_response("Prose.\n<art>C A T</art>", None, observation=obs)
        self.assertIsNone(result.submission)

    def test_clean_art_still_submits(self):
        obs = _artist_obs(target="SNOWFLAKE")
        result = parse_response("Prose.\n<art>\n  *\n \\|/\n--+--\n</art>", None, observation=obs)
        self.assertIsNotNone(result.submission)

    def test_rethink_names_the_offending_run(self):
        obs = _artist_obs(target="CHEEKBONE")
        rejected = "\n  O  <-- eye\n /|\\\n"
        prompt = generate_prompt(obs, [], previous_response="resp", previous_action=rejected)
        self.assertIn("REJECTED", prompt)
        self.assertIn("'eye'", prompt)
        self.assertIn("along a row", prompt)
        # The drawing is echoed so the model can edit rather than restart.
        self.assertIn("<-- eye", prompt)
        self.assertNotIn("empty or whitespace-only", prompt)

    def test_empty_block_still_gets_the_empty_rethink(self):
        """The empty-block sentinel is itself letter-bearing ('art'), so the
        rejection branch must not swallow it."""
        obs = _artist_obs()
        result = parse_response("<art>   </art>", None, observation=obs)
        prompt = generate_prompt(
            obs, [], previous_response="resp", previous_action=result.raw_action,
        )
        self.assertIn("empty or whitespace-only", prompt)
        self.assertNotIn("REJECTED", prompt)

    def test_artist_recovers_on_retry(self):
        agent = create_agent_fn(_WordArtHarness(), max_retries=2)
        obs = _artist_obs(target="CHEEKBONE")
        responses = [
            _fake_completion("Draft.\n<art>  O  <-- eye</art>"),
            _fake_completion("Removed the label.\n<art>  O\n /|\\</art>"),
        ]
        with (
            patch.dict("os.environ", _ENV, clear=False),
            patch.object(
                core_harness.litellm, "completion", side_effect=responses,
            ) as mock_call,
        ):
            result = agent(obs, {"freeForm": True})
        self.assertEqual(mock_call.call_count, 2)
        self.assertNotIn("eye", result["submission"])


if __name__ == "__main__":
    absltest.main()
