"""Tests for the Word Art harness (core_harness integration)."""

from unittest.mock import patch

from absl.testing import absltest

from kaggle_environments import core_harness
from kaggle_environments.core_harness import ParseResult, create_agent_fn, set_telemetry_exporter
from kaggle_environments.envs.word_art.harness.main import (
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


# --- parse_response ---------------------------------------------------------


class ParseResponseTest(absltest.TestCase):
    # --- Artist ---

    def test_artist_extracts_art_fenced_json(self):
        obs = _artist_obs()
        response = '```json\n{"thinking": "a cat", "art": " /\\\\_/\\\\\\n( o.o )"}\n```'
        result = parse_response(response, None, observation=obs)
        self.assertEqual(result.submission, " /\\_/\\\n( o.o )")
        self.assertEqual(result.thoughts, "a cat")

    def test_artist_extracts_art_bare_json(self):
        obs = _artist_obs()
        response = '{"thinking": "x", "art": "BANANA-banner"}'
        result = parse_response(response, None, observation=obs)
        self.assertEqual(result.submission, "BANANA-banner")

    def test_artist_picks_last_json_block(self):
        obs = _artist_obs()
        response = 'Draft: {"art": "DRAFT_DRAWING"}\nFinal: {"thinking": "revised", "art": "FINAL_DRAWING"}'
        result = parse_response(response, None, observation=obs)
        self.assertEqual(result.submission, "FINAL_DRAWING")

    def test_artist_missing_art_key_returns_no_submission(self):
        # JSON emitted but neither "art" nor "guess" is present →
        # extract_last_json_object returns None → telemetry UNPARSABLE.
        obs = _artist_obs()
        response = '{"thinking": "I forgot the art key"}'
        result = parse_response(response, None, observation=obs)
        self.assertIsNone(result.submission)
        self.assertIsNone(result.raw_action)

    def test_artist_wrong_role_key_returns_no_submission_but_surfaces_raw(self):
        # JSON with the OTHER role's key (a guess on an artist turn) -- the
        # model did emit structured output; it just answered the wrong
        # question. Surfaces raw_action so the rethink prompt can quote it
        # back and the telemetry categorizes as ILLEGAL.
        obs = _artist_obs()
        response = '{"guess": "ELEPHANT"}'
        result = parse_response(response, None, observation=obs)
        self.assertIsNone(result.submission)
        self.assertIsNotNone(result.raw_action)
        self.assertIn("guess", result.raw_action)

    def test_artist_prose_only_returns_no_submission(self):
        obs = _artist_obs()
        response = "Here is a drawing of a cat: ^.^"
        result = parse_response(response, None, observation=obs)
        self.assertIsNone(result.submission)
        # No JSON at all → raw_action=None → telemetry UNPARSABLE.
        self.assertIsNone(result.raw_action)

    # --- Guesser ---

    def test_guesser_extracts_guess(self):
        obs = _guesser_obs()
        response = '```json\n{"thinking": "looks like a cat", "guess": "CAT"}\n```'
        result = parse_response(response, None, observation=obs)
        self.assertEqual(result.submission, "CAT")
        self.assertEqual(result.thoughts, "looks like a cat")

    def test_guesser_coerces_non_string_guess(self):
        obs = _guesser_obs()
        response = '{"guess": 42}'
        result = parse_response(response, None, observation=obs)
        self.assertEqual(result.submission, "42")

    def test_guesser_missing_guess_key_returns_no_submission(self):
        obs = _guesser_obs()
        response = '{"thinking": "no idea", "art": "still"}'
        result = parse_response(response, None, observation=obs)
        self.assertIsNone(result.submission)

    def test_guesser_picks_last_json_block(self):
        obs = _guesser_obs()
        response = 'Maybe {"guess": "DOG"} but actually\n{"thinking": "wait, whiskers", "guess": "CAT"}'
        result = parse_response(response, None, observation=obs)
        self.assertEqual(result.submission, "CAT")

    # --- No-role fallback (e.g. obs not forwarded) ---

    def test_no_observation_accepts_either_key(self):
        response = '{"art": "X"}'
        self.assertEqual(parse_response(response, None).submission, "X")
        response = '{"guess": "CAT"}'
        self.assertEqual(parse_response(response, None).submission, "CAT")

    def test_prose_returns_no_submission(self):
        result = parse_response("Just some text", None)
        self.assertIsNone(result.submission)


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
        # Critical mechanic: the engine catches the target word verbatim,
        # obfuscated, or reversed. The prompt must spell out the rule, the
        # normalization that powers it, and the consequence.
        prompt = generate_prompt(_artist_obs(target="ELEPHANT"), [])
        self.assertIn("target word", prompt.lower())
        self.assertIn("engine-enforced", prompt.lower())
        # The prompt MUST describe the normalization so the model understands
        # what's actually caught (not just a vague "don't do it").
        self.assertIn("stripping every non-alphanumeric", prompt)
        self.assertIn("reversed", prompt)
        # Consequence (placeholder shown to teammate).
        self.assertIn("placeholder", prompt)

    def test_artist_prompt_broad_no_words_rule(self):
        # Beyond the engine-enforced target-word check, the prompt must
        # tell artists not to include ANY words (synonyms, labels, NATO,
        # translations, rhymes) -- soft rule, but it has to be stated.
        prompt = generate_prompt(_artist_obs(), [])
        lower = prompt.lower()
        self.assertIn("any words", lower)
        # Examples of what counts as "words" should appear, so the model
        # isn't left guessing what we mean by "no words":
        for kw in ("synonym", "label", "nato", "translation", "rhyme"):
            self.assertIn(kw, lower)
        # And the prompt must clarify that letters as visual elements are
        # fine -- otherwise the rule reads as "no letters at all".
        self.assertIn("visual element", lower)

    def test_artist_prompt_requests_thinking_before_json(self):
        # Memory contract: every prompt asks for reasoning BEFORE JSON.
        prompt = generate_prompt(_artist_obs(), [])
        lower = prompt.lower()
        self.assertIn("think step by step", lower)
        self.assertIn("thinking", lower)
        # Thinking instruction must precede the JSON example.
        json_block = prompt.find("```json")
        self.assertGreater(json_block, 0)
        self.assertLess(lower.find("think step by step"), json_block)

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

    def test_guesser_prompt_requests_thinking_before_json(self):
        prompt = generate_prompt(_guesser_obs(), [])
        lower = prompt.lower()
        self.assertIn("think step by step", lower)
        json_block = prompt.find("```json")
        self.assertGreater(json_block, 0)
        self.assertLess(lower.find("think step by step"), json_block)

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

    def test_rethink_suffix_appended_on_retry(self):
        prompt = generate_prompt(
            _artist_obs(),
            [],
            previous_response="my last try was junk text",
        )
        self.assertIn("Last 500 characters", prompt)
        self.assertIn("my last try was junk text", prompt)

    def test_rethink_not_appended_on_first_attempt(self):
        prompt = generate_prompt(_artist_obs(), [])
        self.assertNotIn("Last 500 characters", prompt)

    def test_max_attempts_propagates_to_prompt(self):
        # Env supports configurable max_attempts; prompt must reflect it.
        prompt = generate_prompt(_artist_obs(max_attempts=5), [])
        self.assertIn("attempt 2 through 5", prompt)
        self.assertIn("within 5 attempts", prompt)

    def test_first_try_bonus_propagates_from_observation(self):
        """The scoring text must reflect the env's first_try_bonus, not a
        hardcoded value. Previously the harness hardcoded 1, silently lying
        to the model whenever the env was configured differently."""
        prompt = generate_prompt(_artist_obs(first_try_bonus=5), [])
        # Base 1 + bonus 5 = 6 points on attempt 1.
        self.assertIn("Correct on attempt 1: 6 points", prompt)
        self.assertIn("1 base + 5 first-try bonus", prompt)

    def test_first_try_bonus_zero_renders_correctly(self):
        prompt = generate_prompt(_guesser_obs(first_try_bonus=0), [])
        self.assertIn("Correct on attempt 1: 1 points", prompt)
        self.assertIn("1 base + 0 first-try bonus", prompt)

    def test_artist_prompt_mentions_max_art_chars_truncation(self):
        """Artist must know the env will silently truncate over-long art."""
        prompt = generate_prompt(_artist_obs(max_art_chars=2500), [])
        self.assertIn("2500", prompt)
        self.assertIn("truncated", prompt.lower())

    def test_guesser_prompt_explains_disqualification_marker(self):
        """Guesser must be told what the placeholder string means when their
        teammate's art gets disqualified for containing the target word."""
        prompt = generate_prompt(_guesser_obs(), [])
        # The rule statement must call out the placeholder mechanic in
        # GAME RULES (so the model isn't surprised by the marker text).
        self.assertIn("disqualification", prompt.lower())
        # And the wording must mention WHY (target word in the art).
        self.assertIn("target word", prompt.lower())

    def test_history_marks_disqualified_blue_entry(self):
        """When a past round's blue art was disqualified, _format_history
        must label it so the model doesn't read the raw art as something
        the guesser actually saw."""
        hist = [
            {
                "word": "CAT",
                "blue_art": "C A T",  # was disqualified
                "blue_art_disqualified": True,
                "blue_guesses": ["DOG", "BEAR", "LION"],
                "blue_points": 0,
                "yellow_art": "MEOW",
                "yellow_art_disqualified": False,
                "yellow_guesses": ["CAT"],
                "yellow_points": 2,
            }
        ]
        prompt = generate_prompt(_guesser_obs(history=hist, current_round=1), [])
        # Blue art line must carry the disqualification label.
        self.assertIn("Blue art: (DISQUALIFIED", prompt)
        self.assertIn("placeholder", prompt.lower())
        # Yellow art line must NOT carry the label.
        self.assertIn("Yellow art:", prompt)
        self.assertNotIn("Yellow art: (DISQUALIFIED", prompt)

    def test_history_marks_disqualified_yellow_entry(self):
        hist = [
            {
                "word": "DOG",
                "blue_art": "WOOF",
                "blue_art_disqualified": False,
                "blue_guesses": ["DOG"],
                "blue_points": 2,
                "yellow_art": "D-O-G",
                "yellow_art_disqualified": True,
                "yellow_guesses": ["WOLF", "FOX"],
                "yellow_points": 0,
            }
        ]
        prompt = generate_prompt(_artist_obs(history=hist, current_round=1), [])
        self.assertIn("Yellow art: (DISQUALIFIED", prompt)
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
        llm = '{"thinking": "a cat face", "art": " /\\\\_/\\\\\\n( o.o )"}'
        with (
            patch.dict("os.environ", _ENV, clear=False),
            patch.object(
                core_harness.litellm,
                "completion",
                return_value=_fake_completion(llm),
            ),
        ):
            result = agent(obs, {"freeForm": True})
        self.assertEqual(result["submission"], " /\\_/\\\n( o.o )")
        self.assertEqual(result["status"], "OK")

    def test_guesser_successful_submission(self):
        agent = create_agent_fn(_WordArtHarness())
        obs = _guesser_obs(art="MEOW")
        llm = '{"thinking": "cat", "guess": "CAT"}'
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
            _fake_completion('{"thinking": "cat", "guess": "CAT"}'),
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

    def test_artist_wrong_key_triggers_rethink(self):
        """Artist who emits a 'guess' instead of 'art' should trigger the
        rethink loop. With a single retry budget the agent raises; with
        more, it gets a chance to fix and submits the art."""
        agent = create_agent_fn(_WordArtHarness(), max_retries=2)
        obs = _artist_obs()
        responses = [
            _fake_completion('{"guess": "CAT"}'),  # wrong key
            _fake_completion('{"thinking": "fix", "art": "MEOW"}'),
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
        """When the artist never produces a valid 'art' key, the agent
        raises ValueError after exhausting retries (core_harness behaviour)."""
        agent = create_agent_fn(_WordArtHarness(), max_retries=1)
        obs = _artist_obs()
        responses = [_fake_completion('{"guess": "CAT"}')]
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


# --- Parser regression: no ghost-substitution for guess responses -----------


class NoGhostFallbackTest(absltest.TestCase):
    """The free-form parser intentionally has NO prose fallback. If the
    model writes 'CAT' in prose but its JSON is missing/wrong, we MUST NOT
    silently submit CAT — let the rethink loop handle it."""

    def test_prose_with_no_json_returns_nothing(self):
        obs = _guesser_obs()
        response = "The art clearly shows a CAT. My answer is CAT."
        result = parse_response(response, None, observation=obs)
        self.assertIsNone(result.submission)

    def test_prose_with_json_lacking_guess_returns_nothing(self):
        obs = _guesser_obs()
        response = 'I see a cat: CAT. {"thinking": "CAT"}'
        result = parse_response(response, None, observation=obs)
        # No "guess" key → no submission, even though CAT is in the prose
        self.assertIsNone(result.submission)


# --- ParseResult shape (defensive: make sure we never return weird types) ---


class ParseResultShapeTest(absltest.TestCase):
    def test_artist_submission_is_str(self):
        result = parse_response(
            '{"art": "x"}',
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


if __name__ == "__main__":
    absltest.main()
