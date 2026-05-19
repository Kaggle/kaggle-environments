"""Tests for core_harness using a minimal in-memory game harness."""

import dataclasses
from unittest.mock import patch

from absl.testing import absltest

from kaggle_environments import core_harness
from kaggle_environments.core_harness import (
    ParseResult,
    create_agent_fn,
    set_telemetry_exporter,
)


class _SimpleHarness:
    """Pick-a-number harness: legal moves are 0..N-1, model answers a digit."""

    def __init__(self, num_actions: int = 3):
        self.num_actions = num_actions
        self.prompts: list[str] = []

    def get_legal_moves(self, observation):
        return {i: f"move_{i}" for i in range(self.num_actions)}

    def make_prompt(
        self,
        observation,
        move_history,
        previous_response=None,
        previous_action=None,
    ):
        prompt = f"history={move_history} prev={previous_action}"
        self.prompts.append(prompt)
        return prompt

    def parse_response(self, response, legal_action_strings):
        response = response.strip()
        if response in legal_action_strings:
            return ParseResult(legal_action=response, raw_action=response)
        return ParseResult(legal_action=None, raw_action=response or None)


class _Usage:
    prompt_tokens = 1
    completion_tokens = 1
    total_tokens = 2
    completion_tokens_details = None


class _Delta:
    def __init__(self, content):
        self.content = content


class _ChunkChoice:
    def __init__(self, content, finish_reason=None):
        self.delta = _Delta(content)
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage


def _fake_completion(
    content: str,
    *,
    pieces: int = 2,
    finish_reason: str | None = "stop",
):
    """Build a mock streaming litellm.completion iterator.

    Splits ``content`` into ``pieces`` delta chunks, followed by an empty
    chunk carrying ``finish_reason`` and a final usage-only chunk —
    mirroring litellm's stream when ``stream_options={"include_usage": True}``.

    Passing ``finish_reason=None`` suppresses the finish-reason chunk so tests
    can simulate a truncated stream.
    """
    if pieces < 1:
        raise ValueError("pieces must be >= 1")
    n = len(content)
    if n == 0 or pieces == 1:
        slices = [content]
    else:
        step = max(1, n // pieces)
        slices = [content[i:i + step] for i in range(0, n, step)]

    def _gen():
        for s in slices:
            yield _Chunk([_ChunkChoice(s)])
        if finish_reason is not None:
            yield _Chunk([_ChunkChoice("", finish_reason=finish_reason)])
        yield _Chunk([], usage=_Usage())

    return _gen()


_ENV = {
    "MODEL_NAME": "test-model",
    "MODEL_PROXY_KEY": "key",
    "MODEL_PROXY_URL": "dummy_url",
}


class CoreHarnessTest(absltest.TestCase):

    def setUp(self):
        super().setUp()
        # Reset telemetry between tests.
        self.events: list[dict] = []
        set_telemetry_exporter(
            lambda module, **kw: self.events.append({"module": module, **kw})
        )

    def tearDown(self):
        set_telemetry_exporter(lambda module, **kwargs: None)
        super().tearDown()

    def test_first_attempt_succeeds(self):
        harness = _SimpleHarness()
        agent = create_agent_fn(harness)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm,
            "completion",
            return_value=_fake_completion("move_1"),
        ) as mock_call:
            result = agent({}, {})

        self.assertEqual(result["submission"], 1)
        self.assertEqual(result["actionString"], "move_1")
        self.assertEqual(result["status"], "OK")
        self.assertEqual(mock_call.call_count, 1)
        self.assertEqual(len(harness.prompts), 1)

    def test_retry_then_succeeds(self):
        harness = _SimpleHarness()
        agent = create_agent_fn(harness, max_retries=3)
        responses = [_fake_completion("garbage"), _fake_completion("move_2")]
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion", side_effect=responses,
        ) as mock_call:
            result = agent({}, {})

        self.assertEqual(result["submission"], 2)
        self.assertEqual(result["actionString"], "move_2")
        self.assertEqual(mock_call.call_count, 2)
        # Second prompt should reflect rethink context.
        self.assertIn("prev=garbage", harness.prompts[1])

    def test_all_attempts_fail_forfeits_by_default(self):
        harness = _SimpleHarness()
        agent = create_agent_fn(harness, max_retries=2)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm,
            "completion",
            side_effect=lambda *a, **kw: _fake_completion("nope"),
        ):
            result = agent({}, {})

        self.assertEqual(result["submission"], -1)
        self.assertEqual(result["actionString"], "nope")
        self.assertIn("forfeiting", result["status"])
        kinds = [e.get("module") for e in self.events]
        self.assertTrue(
            any("illegal_move_forfeit" in e for e in self.events),
            f"expected illegal_move_forfeit telemetry, got {kinds}",
        )

    def test_all_attempts_fail_raises_when_forfeit_disabled(self):
        harness = _SimpleHarness()
        agent = create_agent_fn(harness, max_retries=2)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm,
            "completion",
            side_effect=lambda *a, **kw: _fake_completion("nope"),
        ):
            with self.assertRaisesRegex(ValueError, "Failed to parse"):
                agent({}, {"illegalMoveForfeit": False})

    def test_no_legal_moves_raises(self):
        harness = _SimpleHarness(num_actions=0)
        agent = create_agent_fn(harness)
        active_obs = {"playerId": 0, "currentPlayer": 0, "isTerminal": False}
        with patch.dict("os.environ", _ENV, clear=False):
            with self.assertRaisesRegex(ValueError, "No legal actions"):
                agent(active_obs, {})

    def test_terminal_obs_returns_inactive(self):
        harness = _SimpleHarness()
        agent = create_agent_fn(harness)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion",
        ) as mock_call:
            result = agent({"isTerminal": True, "playerId": 0, "currentPlayer": 0}, {})
        self.assertEqual(result, {"submission": None, "status": "INACTIVE"})
        mock_call.assert_not_called()

    def test_not_our_turn_returns_inactive(self):
        harness = _SimpleHarness()
        agent = create_agent_fn(harness)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion",
        ) as mock_call:
            result = agent({"isTerminal": False, "playerId": 0, "currentPlayer": 1}, {})
        self.assertEqual(result, {"submission": None, "status": "INACTIVE"})
        mock_call.assert_not_called()

    def test_empty_obs_returns_inactive(self):
        harness = _SimpleHarness(num_actions=0)
        agent = create_agent_fn(harness)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion",
        ) as mock_call:
            result = agent({"remainingOverageTime": 60, "step": 0}, {})
        self.assertEqual(result, {"submission": None, "status": "INACTIVE"})
        mock_call.assert_not_called()

    def test_missing_env_var_raises(self):
        harness = _SimpleHarness()
        agent = create_agent_fn(harness)
        env = {k: v for k, v in _ENV.items() if k != "MODEL_NAME"}
        with patch.dict("os.environ", env, clear=True):
            with self.assertRaisesRegex(ValueError, "MODEL_NAME"):
                agent({}, {})

    def test_telemetry_exporter_receives_events(self):
        harness = _SimpleHarness()
        agent = create_agent_fn(harness)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm,
            "completion",
            return_value=_fake_completion("move_0"),
        ):
            agent({}, {})

        kinds = {k for e in self.events for k in e if k != "module"}
        self.assertIn("setup_complete", kinds)
        self.assertIn("calling_llm", kinds)
        self.assertIn("action_is_legal", kinds)

    def test_save_prompt_in_call_details(self):
        harness = _SimpleHarness()
        agent = create_agent_fn(harness)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm,
            "completion",
            return_value=_fake_completion("move_1"),
        ):
            result = agent({}, {"savePrompt": True})

        self.assertEqual(result["submission"], 1)
        self.assertIn("prompt", result["call_details"][0])
        self.assertEqual(result["call_details"][0]["prompt"], harness.prompts[-1])

    def test_prompt_included_in_call_details_by_default(self):
        harness = _SimpleHarness()
        agent = create_agent_fn(harness)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm,
            "completion",
            return_value=_fake_completion("move_1"),
        ):
            result = agent({}, {})

        self.assertIn("prompt", result["call_details"][0])

    def test_prompt_omitted_when_save_prompt_false(self):
        harness = _SimpleHarness()
        agent = create_agent_fn(harness)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm,
            "completion",
            return_value=_fake_completion("move_1"),
        ):
            result = agent({}, {"savePrompt": False})

        self.assertNotIn("prompt", result["call_details"][0])

    def test_call_details_present_on_success(self):
        harness = _SimpleHarness()
        agent = create_agent_fn(harness)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm,
            "completion",
            return_value=_fake_completion("move_1"),
        ):
            result = agent({}, {})

        self.assertIn("call_details", result)
        self.assertLen(result["call_details"], 1)
        cd = result["call_details"][0]
        self.assertEqual(cd["generation_tokens"], 1)
        self.assertEqual(cd["prompt_tokens"], 1)
        self.assertEqual(cd["total_tokens"], 2)
        self.assertNotIn("reasoning_tokens", cd)
        self.assertEqual(cd["finish_reason"], "stop")
        self.assertEqual(cd["response"], "move_1")
        self.assertEqual(cd["model"], "test-model")
        self.assertIn("prompt", cd)

    def test_call_details_includes_prompt_when_save_prompt(self):
        harness = _SimpleHarness()
        agent = create_agent_fn(harness)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm,
            "completion",
            return_value=_fake_completion("move_1"),
        ):
            result = agent({}, {"savePrompt": True})

        cd = result["call_details"][0]
        self.assertIn("prompt", cd)
        self.assertEqual(cd["prompt"], harness.prompts[-1])

    def test_call_details_per_retry(self):
        harness = _SimpleHarness()
        agent = create_agent_fn(harness, max_retries=3)
        responses = [_fake_completion("garbage"), _fake_completion("move_2")]
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion", side_effect=responses,
        ):
            result = agent({}, {})

        self.assertLen(result["call_details"], 2)
        self.assertEqual(result["call_details"][0]["response"], "garbage")
        self.assertEqual(result["call_details"][1]["response"], "move_2")

    def test_include_generate_returns(self):
        harness = _SimpleHarness()
        agent = create_agent_fn(harness)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm,
            "completion",
            return_value=_fake_completion("move_0"),
        ):
            result = agent({}, {"includeGenerateReturns": True})

        import json
        self.assertIn("generate_returns", result)
        self.assertLen(result["generate_returns"], 1)
        gr = json.loads(result["generate_returns"][0])
        self.assertEqual(gr["request_for_logging"]["model"], "test-model")
        self.assertEqual(gr["generation_tokens"], 1)
        self.assertEqual(gr["prompt_tokens"], 1)
        self.assertEqual(gr["total_tokens"], 2)

    def test_generate_returns_omitted_by_default(self):
        harness = _SimpleHarness()
        agent = create_agent_fn(harness)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm,
            "completion",
            return_value=_fake_completion("move_0"),
        ):
            result = agent({}, {})

        self.assertNotIn("generate_returns", result)

    def test_streaming_assembles_chunks_and_passes_stream_kwargs(self):
        """Verify litellm is invoked with stream=True and chunks are concatenated."""
        harness = _SimpleHarness()
        agent = create_agent_fn(harness)
        # Use enough pieces that the move string is split across chunks.
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm,
            "completion",
            return_value=_fake_completion("move_1", pieces=3),
        ) as mock_call:
            result = agent({}, {})

        self.assertEqual(result["submission"], 1)
        self.assertEqual(result["actionString"], "move_1")
        # litellm was asked to stream and to include usage in the final chunk.
        _, kwargs = mock_call.call_args
        self.assertTrue(kwargs.get("stream"))
        self.assertEqual(kwargs.get("stream_options"), {"include_usage": True})
        # The streamed call still produces full usage details.
        cd = result["call_details"][0]
        self.assertEqual(cd["response"], "move_1")
        self.assertEqual(cd["finish_reason"], "stop")
        self.assertEqual(cd["prompt_tokens"], 1)
        self.assertEqual(cd["generation_tokens"], 1)
        # Streaming surfaces a first-token latency on each call_detail entry.
        self.assertIn("first_token_secs", cd)
        self.assertIsInstance(cd["first_token_secs"], float)
        self.assertGreaterEqual(cd["first_token_secs"], 0.0)
        self.assertLessEqual(cd["first_token_secs"], cd["duration_secs"])

    def test_empty_stream_raises(self):
        """A stream that delivers no content should raise, not silently succeed."""
        harness = _SimpleHarness()
        agent = create_agent_fn(harness)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm,
            "completion",
            return_value=_fake_completion("", pieces=1),
        ):
            with self.assertRaisesRegex(RuntimeError, "no content"):
                agent({}, {})

        # The exception path should record an llm_call_exception telemetry event.
        kinds = {k for e in self.events for k in e if k != "module"}
        self.assertIn("llm_call_exception", kinds)

    def test_missing_finish_reason_raises(self):
        """A stream that ends without a finish_reason should raise."""
        harness = _SimpleHarness()
        agent = create_agent_fn(harness)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm,
            "completion",
            return_value=_fake_completion("move_1", finish_reason=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "finish_reason"):
                agent({}, {})

        kinds = {k for e in self.events for k in e if k != "module"}
        self.assertIn("llm_call_exception", kinds)

    def test_retry_on_llm_exception_then_succeeds(self):
        """A raised exception during _call_llm should consume a retry, not abort."""
        harness = _SimpleHarness()
        agent = create_agent_fn(harness, max_retries=3)
        side_effects = [
            RuntimeError("transient network error"),
            _fake_completion("move_1"),
        ]
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion", side_effect=side_effects,
        ) as mock_call:
            result = agent({}, {})

        self.assertEqual(result["submission"], 1)
        self.assertEqual(mock_call.call_count, 2)
        # Only the successful call should appear in call_details.
        self.assertLen(result["call_details"], 1)
        self.assertEqual(result["call_details"][0]["response"], "move_1")

    def test_retry_on_parse_exception_then_succeeds(self):
        """An exception raised inside parse_response should consume a retry."""

        class _FlakyParseHarness(_SimpleHarness):
            def __init__(self):
                super().__init__()
                self.parse_calls = 0

            def parse_response(self, response, legal_action_strings):
                self.parse_calls += 1
                if self.parse_calls == 1:
                    raise RuntimeError("parser blew up")
                return super().parse_response(response, legal_action_strings)

        harness = _FlakyParseHarness()
        agent = create_agent_fn(harness, max_retries=3)
        responses = [_fake_completion("move_0"), _fake_completion("move_2")]
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion", side_effect=responses,
        ) as mock_call:
            result = agent({}, {})

        self.assertEqual(result["submission"], 2)
        self.assertEqual(mock_call.call_count, 2)
        self.assertEqual(harness.parse_calls, 2)

    def test_all_attempts_exception_reraises_last(self):
        """When every attempt raises, the final exception should be re-raised verbatim."""
        harness = _SimpleHarness()
        agent = create_agent_fn(harness, max_retries=2)
        side_effects = [
            RuntimeError("first failure"),
            RuntimeError("second failure"),
        ]
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion", side_effect=side_effects,
        ):
            with self.assertRaisesRegex(RuntimeError, "second failure"):
                agent({}, {})

    def test_exception_then_parse_failure_forfeits_by_default(self):
        """If the last attempt was a parse failure (not an exception), the
        forfeit path applies — last_exception should not leak through."""
        harness = _SimpleHarness()
        agent = create_agent_fn(harness, max_retries=2)
        side_effects = [
            RuntimeError("first attempt blew up"),
            _fake_completion("garbage"),  # parses to None → parse failure
        ]
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion", side_effect=side_effects,
        ):
            result = agent({}, {})

        self.assertEqual(result["submission"], -1)
        self.assertEqual(result["actionString"], "garbage")
        self.assertIn("forfeiting", result["status"])

    def test_exception_then_parse_failure_raises_when_forfeit_disabled(self):
        harness = _SimpleHarness()
        agent = create_agent_fn(harness, max_retries=2)
        side_effects = [
            RuntimeError("first attempt blew up"),
            _fake_completion("garbage"),
        ]
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion", side_effect=side_effects,
        ):
            with self.assertRaisesRegex(ValueError, "Failed to parse"):
                agent({}, {"illegalMoveForfeit": False})

    def test_move_history_accumulates_across_calls(self):
        harness = _SimpleHarness()
        agent = create_agent_fn(harness)
        responses = [_fake_completion("move_0"), _fake_completion("move_1")]
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion", side_effect=responses,
        ):
            agent({}, {})
            agent({}, {})

        # The second prompt should include the first move in its history.
        self.assertIn("move_0", harness.prompts[1])


class _FreeFormHarness:
    """Harness that returns None from get_legal_moves (free-form actions)."""

    def __init__(self):
        self.prompts: list[str] = []

    def get_legal_moves(self, observation):
        return None

    def make_prompt(self, observation, move_history, previous_response=None, previous_action=None):
        prompt = f"history={move_history} prev={previous_action}"
        self.prompts.append(prompt)
        return prompt

    def parse_response(self, response, legal_action_strings):
        assert legal_action_strings is None
        try:
            import json
            parsed = json.loads(response)
            if "clue" in parsed:
                return ParseResult(
                    submission={"clue": parsed["clue"], "number": parsed["number"]},
                    raw_action=response,
                )
        except Exception:
            pass
        return ParseResult(raw_action=response)


class _MixedHarness:
    """Harness that alternates between free-form and enumerable each call."""

    def __init__(self):
        self.call_count = 0
        self.prompts: list[str] = []

    def get_legal_moves(self, observation):
        self.call_count += 1
        if self.call_count % 2 == 1:
            return None  # odd calls: free-form
        return {0: "PASS", 1: "word_A", 2: "word_B"}  # even calls: enumerable

    def make_prompt(self, observation, move_history, previous_response=None, previous_action=None):
        prompt = f"history={move_history} prev={previous_action}"
        self.prompts.append(prompt)
        return prompt

    def parse_response(self, response, legal_action_strings):
        if legal_action_strings is None:
            try:
                import json
                parsed = json.loads(response)
                return ParseResult(submission=parsed, raw_action=response)
            except Exception:
                return ParseResult(raw_action=response)
        else:
            response = response.strip()
            if response in legal_action_strings:
                return ParseResult(legal_action=response, raw_action=response)
            return ParseResult(raw_action=response)


class FreeFormHarnessTest(absltest.TestCase):

    def setUp(self):
        super().setUp()
        self.events: list[dict] = []
        set_telemetry_exporter(
            lambda module, **kw: self.events.append({"module": module, **kw})
        )

    def tearDown(self):
        set_telemetry_exporter(lambda module, **kwargs: None)
        super().tearDown()

    def test_free_form_succeeds(self):
        harness = _FreeFormHarness()
        agent = create_agent_fn(harness)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm,
            "completion",
            return_value=_fake_completion('{"clue": "ANIMAL", "number": 2}'),
        ):
            result = agent({}, {"freeForm": True})

        self.assertEqual(result["submission"], {"clue": "ANIMAL", "number": 2})
        self.assertEqual(result["status"], "OK")

    def test_free_form_retry_then_succeeds(self):
        harness = _FreeFormHarness()
        agent = create_agent_fn(harness, max_retries=3)
        responses = [
            _fake_completion("not json"),
            _fake_completion('{"clue": "OCEAN", "number": 3}'),
        ]
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion", side_effect=responses,
        ) as mock_call:
            result = agent({}, {"freeForm": True})

        self.assertEqual(result["submission"], {"clue": "OCEAN", "number": 3})
        self.assertEqual(mock_call.call_count, 2)
        self.assertIn("prev=not json", harness.prompts[1])

    def test_free_form_all_attempts_fail_forfeits_by_default(self):
        harness = _FreeFormHarness()
        agent = create_agent_fn(harness, max_retries=2)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm,
            "completion",
            side_effect=lambda *a, **kw: _fake_completion("garbage"),
        ):
            result = agent({}, {"freeForm": True})

        self.assertEqual(result["submission"], -1)
        self.assertIn("forfeiting", result["status"])

    def test_free_form_all_attempts_fail_raises_when_forfeit_disabled(self):
        harness = _FreeFormHarness()
        agent = create_agent_fn(harness, max_retries=2)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm,
            "completion",
            side_effect=lambda *a, **kw: _fake_completion("garbage"),
        ):
            with self.assertRaisesRegex(ValueError, "Failed to parse"):
                agent({}, {"freeForm": True, "illegalMoveForfeit": False})

    def test_free_form_move_history_accumulates(self):
        harness = _FreeFormHarness()
        agent = create_agent_fn(harness)
        responses = [
            _fake_completion('{"clue": "ANIMAL", "number": 2}'),
            _fake_completion('{"clue": "OCEAN", "number": 1}'),
        ]
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion", side_effect=responses,
        ):
            agent({}, {"freeForm": True})
            agent({}, {"freeForm": True})

        self.assertIn("ANIMAL", harness.prompts[1])

    def test_mixed_harness_alternates_modes(self):
        """Free-form on first call, enumerable on second."""
        harness = _MixedHarness()
        agent = create_agent_fn(harness)
        responses = [
            _fake_completion('{"clue": "ANIMAL", "number": 2}'),
            _fake_completion("word_A"),
        ]
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion", side_effect=responses,
        ):
            r1 = agent({}, {"freeForm": True})
            r2 = agent({}, {"freeForm": True})

        # First call: free-form → submission is the dict
        self.assertEqual(r1["submission"], {"clue": "ANIMAL", "number": 2})
        # Second call: enumerable → submission is the action id (int)
        self.assertEqual(r2["submission"], 1)
        self.assertEqual(r2["actionString"], "word_A")

    def test_free_form_save_prompt(self):
        harness = _FreeFormHarness()
        agent = create_agent_fn(harness)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm,
            "completion",
            return_value=_fake_completion('{"clue": "FIRE", "number": 1}'),
        ):
            result = agent({}, {"freeForm": True, "savePrompt": True})

        self.assertIn("prompt", result["call_details"][0])
        self.assertEqual(result["call_details"][0]["prompt"], harness.prompts[-1])

    def test_none_legal_moves_without_free_form_config_returns_inactive(self):
        """get_legal_moves() returns None but freeForm not in config → empty-obs path."""
        harness = _FreeFormHarness()
        agent = create_agent_fn(harness)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion",
        ) as mock_call:
            # No playerId/currentPlayer → treated as empty obs
            result = agent({"remainingOverageTime": 60, "step": 0}, {})
        self.assertEqual(result, {"submission": None, "status": "INACTIVE"})
        mock_call.assert_not_called()

    def test_none_legal_moves_without_free_form_config_raises(self):
        """get_legal_moves() returns None but freeForm not in config → error when active."""
        harness = _FreeFormHarness()
        agent = create_agent_fn(harness)
        active_obs = {"playerId": 0, "currentPlayer": 0, "isTerminal": False}
        with patch.dict("os.environ", _ENV, clear=False):
            with self.assertRaisesRegex(ValueError, "No legal actions"):
                agent(active_obs, {})


class ParseResultTest(absltest.TestCase):

    def test_defaults(self):
        r = ParseResult()
        self.assertIsNone(r.legal_action)
        self.assertIsNone(r.raw_action)
        self.assertIsNone(r.submission)

    def test_frozen(self):
        r = ParseResult(legal_action="a", raw_action="a")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            r.legal_action = "b"  # type: ignore[misc]

    def test_submission_field(self):
        r = ParseResult(submission={"clue": "test", "number": 1})
        self.assertEqual(r.submission, {"clue": "test", "number": 1})


if __name__ == "__main__":
    absltest.main()
