"""Tests for ``kaggle_environments.local_harness_runner.run_llm_game``."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from absl.testing import absltest

from kaggle_environments import local_harness_runner


# Env vars the helper might consult; clear them per-test for determinism.
_HARNESS_ENV_KEYS = (
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "MODEL_NAME",
    "MODEL_PROXY_KEY",
    "MODEL_PROXY_URL",
)


def _make_fake_env() -> MagicMock:
    """A fake env whose ``run`` returns nothing and ``toJSON`` returns a dict."""
    env = MagicMock()
    step = MagicMock()
    step.status = "DONE"
    step.action = {"submission": 0}
    env.steps = [[step, step]]
    state = MagicMock()
    state.status = "DONE"
    state.reward = 0
    env.state = [state, state]
    env.toJSON.return_value = {"name": "fake", "steps": []}
    return env


def _clear_env() -> dict[str, str | None]:
    """Snapshot + clear the env vars the helper inspects. Returns the snapshot."""
    snapshot = {k: os.environ.pop(k, None) for k in _HARNESS_ENV_KEYS}
    return snapshot


def _restore_env(snapshot: dict[str, str | None]) -> None:
    for k, v in snapshot.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class RunLlmGameTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self._env_snapshot = _clear_env()
        self.addCleanup(_restore_env, self._env_snapshot)

    def test_missing_api_key_returns_early(self):
        with patch.object(local_harness_runner, "make") as mock_make:
            result = local_harness_runner.run_llm_game(
                "open_spiel_checkers", caller_file=__file__, cli=False,
            )
        self.assertIsNone(result)
        mock_make.assert_not_called()

    def test_env_vars_set_with_defaults(self):
        os.environ["GEMINI_API_KEY"] = "test-key"
        with patch.object(local_harness_runner, "make", return_value=_make_fake_env()):
            with tempfile.TemporaryDirectory() as tmp:
                local_harness_runner.run_llm_game(
                    "open_spiel_checkers",
                    caller_file=os.path.join(tmp, "test_llm_game.py"),
                    cli=False,
                )
        # Defaults are populated.
        self.assertEqual(os.environ.get("MODEL_NAME"), local_harness_runner._DEFAULT_MODEL)
        self.assertEqual(os.environ.get("MODEL_PROXY_KEY"), "test-key")
        self.assertEqual(os.environ.get("MODEL_PROXY_URL"), "dummy_url")

    def test_preserves_existing_env_vars(self):
        os.environ["GEMINI_API_KEY"] = "test-key"
        os.environ["MODEL_NAME"] = "user-model"
        os.environ["MODEL_PROXY_KEY"] = "user-proxy-key"
        os.environ["MODEL_PROXY_URL"] = "https://user-proxy.example"
        with patch.object(local_harness_runner, "make", return_value=_make_fake_env()):
            with tempfile.TemporaryDirectory() as tmp:
                local_harness_runner.run_llm_game(
                    "open_spiel_checkers",
                    caller_file=os.path.join(tmp, "test_llm_game.py"),
                    cli=False,
                )
        self.assertEqual(os.environ["MODEL_NAME"], "user-model")
        self.assertEqual(os.environ["MODEL_PROXY_KEY"], "user-proxy-key")
        self.assertEqual(os.environ["MODEL_PROXY_URL"], "https://user-proxy.example")

    def test_cli_model_override(self):
        os.environ["GEMINI_API_KEY"] = "test-key"
        with patch.object(local_harness_runner, "make", return_value=_make_fake_env()):
            with tempfile.TemporaryDirectory() as tmp:
                local_harness_runner.run_llm_game(
                    "open_spiel_checkers",
                    caller_file=os.path.join(tmp, "test_llm_game.py"),
                    argv=["--model", "claude-opus-4-7"],
                )
        self.assertEqual(os.environ["MODEL_NAME"], "claude-opus-4-7")

    def test_cli_replay_path_override(self):
        os.environ["GEMINI_API_KEY"] = "test-key"
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = os.path.join(tmp, "subdir", "custom-replay.json")
            with patch.object(local_harness_runner, "make", return_value=_make_fake_env()):
                local_harness_runner.run_llm_game(
                    "open_spiel_checkers",
                    caller_file=os.path.join(tmp, "test_llm_game.py"),
                    argv=["--replay-path", replay_path],
                )
            self.assertTrue(os.path.exists(replay_path))
            with open(replay_path) as f:
                self.assertEqual(json.load(f), {"name": "fake", "steps": []})

    def test_env_run_called_with_repeated_agent_path(self):
        os.environ["GEMINI_API_KEY"] = "test-key"
        fake_env = _make_fake_env()
        with patch.object(local_harness_runner, "make", return_value=fake_env):
            with tempfile.TemporaryDirectory() as tmp:
                caller_file = os.path.join(tmp, "test_llm_game.py")
                local_harness_runner.run_llm_game(
                    "word_association",
                    caller_file=caller_file,
                    agent_module="harness/main.py",
                    num_agents=4,
                    cli=False,
                )
        fake_env.run.assert_called_once()
        agents_arg = fake_env.run.call_args.args[0]
        expected = os.path.join(tmp, "harness", "main.py")
        self.assertEqual(agents_arg, [expected, expected, expected, expected])

    def test_replay_saved_to_default_location(self):
        os.environ["GEMINI_API_KEY"] = "test-key"
        with tempfile.TemporaryDirectory() as tmp:
            caller_file = os.path.join(tmp, "test_llm_game.py")
            with patch.object(local_harness_runner, "make", return_value=_make_fake_env()):
                local_harness_runner.run_llm_game(
                    "open_spiel_checkers", caller_file=caller_file, cli=False,
                )
            expected = os.path.join(
                tmp, "visualizer", "default", "replays", "test-replay.json",
            )
            self.assertTrue(os.path.exists(expected))
            with open(expected) as f:
                self.assertEqual(json.load(f), {"name": "fake", "steps": []})

    def test_configuration_forwarded_to_make(self):
        os.environ["GEMINI_API_KEY"] = "test-key"
        with patch.object(local_harness_runner, "make", return_value=_make_fake_env()) as mock_make:
            with tempfile.TemporaryDirectory() as tmp:
                local_harness_runner.run_llm_game(
                    "open_spiel_havannah",
                    caller_file=os.path.join(tmp, "test_llm_game.py"),
                    configuration={
                        "includeLegalActions": True,
                        "openSpielGameParameters": {"board_size": 5},
                    },
                    cli=False,
                )
        mock_make.assert_called_once_with(
            "open_spiel_havannah",
            configuration={
                "includeLegalActions": True,
                "openSpielGameParameters": {"board_size": 5},
            },
            debug=True,
        )

    def test_returns_env_for_post_processing(self):
        os.environ["GEMINI_API_KEY"] = "test-key"
        fake_env = _make_fake_env()
        with patch.object(local_harness_runner, "make", return_value=fake_env):
            with tempfile.TemporaryDirectory() as tmp:
                returned = local_harness_runner.run_llm_game(
                    "open_spiel_checkers",
                    caller_file=os.path.join(tmp, "test_llm_game.py"),
                    cli=False,
                )
        self.assertIs(returned, fake_env)


if __name__ == "__main__":
    absltest.main()
