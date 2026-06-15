"""Tests for the per-step `error` field attached to failing agents.

Covers all three failure modes: ERROR (agent raises), TIMEOUT (agent exceeds
actTimeout), and INVALID (action rejected by the interpreter).
"""

from absl.testing import absltest

from kaggle_environments import make


def raising_agent(obs, cfg):
    raise ValueError("intentional boom")


def slow_agent(obs, cfg):
    import time
    time.sleep(30)
    return 0


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

    def test_timeout_status_attaches_error(self):
        env = make("connectx", configuration={"actTimeout": 1})
        env.run([slow_agent, "random"])

        self.assertEqual(env.toJSON()["statuses"], ["TIMEOUT", "DONE"])
        error = env.steps[-1][0]["error"]
        self.assertEqual(error["type"], "TIMEOUT")
        self.assertTrue(error["message"], "TIMEOUT message should be non-empty")

    def test_invalid_status_preserves_interpreter_reason(self):
        env = make("connectx")
        env.run([invalid_agent, "random"])

        self.assertEqual(env.toJSON()["statuses"], ["INVALID", "DONE"])
        error = env.steps[-1][0]["error"]
        self.assertEqual(error["type"], "INVALID")
        self.assertIn("999", error["message"])


if __name__ == "__main__":
    absltest.main()
