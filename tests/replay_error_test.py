"""Tests for the per-step `error` field attached to failing agents.

Covers all three failure modes: ERROR (agent raises), TIMEOUT (agent exceeds
actTimeout, injected as a DeadlineExceeded action), and INVALID (action
rejected by the interpreter).
"""

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.errors import DeadlineExceeded


def raising_agent(obs, cfg):
    raise ValueError("intentional boom")


def invalid_agent(obs, cfg):
    return 999  # out-of-range column for connectx


class ReplayErrorTest(absltest.TestCase):

    def test_error_status_attaches_traceback(self):
        env = make("connectx")
        env.run([raising_agent, "random"])

        self.assertEqual(env.toJSON()["statuses"], ["ERROR", "DONE"])
        agent_state = env.steps[-1][0]
        self.assertEqual(agent_state["status"], "ERROR")
        error = agent_state["error"]
        self.assertEqual(error["type"], "ERROR")
        self.assertEqual(error["message"], "intentional boom")
        self.assertIn("ValueError: intentional boom", error["traceback"])

        log_error = env.logs[-1][0]["error"]
        self.assertEqual(log_error["type"], "ValueError")
        self.assertIn("ValueError: intentional boom", log_error["traceback"])

    def test_timeout_status_falls_back_to_generated_message(self):
        # Inject the DeadlineExceeded the runner would have produced, so the
        # timeout path is exercised deterministically instead of sleeping past
        # the banked overage. An empty message should fall back to a generated
        # "Exceeded actTimeout" string.
        env = make("connectx", configuration={"actTimeout": 1})
        env.reset(2)
        state = env.step([DeadlineExceeded(), 0])

        self.assertEqual(state[0]["status"], "TIMEOUT")
        error = state[0]["error"]
        self.assertEqual(error["type"], "TIMEOUT")
        self.assertEqual(error["message"], "Exceeded actTimeout (1s)")

    def test_timeout_status_preserves_custom_message(self):
        env = make("connectx", configuration={"actTimeout": 1})
        env.reset(2)
        state = env.step([DeadlineExceeded("agent ran too long"), 0])

        self.assertEqual(state[0]["status"], "TIMEOUT")
        error = state[0]["error"]
        self.assertEqual(error["type"], "TIMEOUT")
        self.assertEqual(error["message"], "agent ran too long")

    def test_invalid_status_preserves_interpreter_reason(self):
        env = make("connectx")
        env.run([invalid_agent, "random"])

        self.assertEqual(env.toJSON()["statuses"], ["INVALID", "DONE"])
        error = env.steps[-1][0]["error"]
        self.assertEqual(error["type"], "INVALID")
        self.assertIn("999", error["message"])


if __name__ == "__main__":
    absltest.main()
