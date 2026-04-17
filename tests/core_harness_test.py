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


def _fake_completion(content: str):
    """Build a mock litellm.completion return value with given content."""

    class _Msg:
        def __init__(self, c):
            self.content = c

    class _Choice:
        def __init__(self, c):
            self.message = _Msg(c)

    class _Usage:
        prompt_tokens = 1
        completion_tokens = 1

    class _Resp:
        def __init__(self, c):
            self.choices = [_Choice(c)]
            self.usage = _Usage()

    return _Resp(content)


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

    def test_all_attempts_fail_raises(self):
        harness = _SimpleHarness()
        agent = create_agent_fn(harness, max_retries=2)
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm,
            "completion",
            return_value=_fake_completion("nope"),
        ):
            with self.assertRaisesRegex(ValueError, "Failed to parse"):
                agent({}, {})

    def test_no_legal_moves_raises(self):
        harness = _SimpleHarness(num_actions=0)
        agent = create_agent_fn(harness)
        with patch.dict("os.environ", _ENV, clear=False):
            with self.assertRaisesRegex(ValueError, "No legal actions"):
                agent({}, {})

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


class ParseResultTest(absltest.TestCase):

    def test_defaults(self):
        r = ParseResult()
        self.assertIsNone(r.legal_action)
        self.assertIsNone(r.raw_action)

    def test_frozen(self):
        r = ParseResult(legal_action="a", raw_action="a")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            r.legal_action = "b"  # type: ignore[misc]


if __name__ == "__main__":
    absltest.main()
