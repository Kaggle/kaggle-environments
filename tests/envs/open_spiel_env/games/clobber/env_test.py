"""Env-level tests for open_spiel_clobber."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class ClobberEnvTest(absltest.TestCase):
    def test_clobber_agent_playthrough(self):
        env = make(
            "open_spiel_clobber",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_clobber")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))
        # Clobber is zero-sum with no draws.
        rewards = playthrough["rewards"]
        self.assertEqual(sorted(rewards), [-1.0, 1.0])
        final_obs = json.loads(playthrough["steps"][-1][0]["observation"]["observationString"])
        self.assertTrue(final_obs["is_terminal"])
        self.assertIn(final_obs["winner"], ("o", "x"))
        self.assertEqual(final_obs["rows"], 5)
        self.assertEqual(final_obs["columns"], 6)

    def test_clobber_manual_playthrough(self):
        # 2x2 game ends in three moves: a1b1, a2b2, b1b2 -> P0 wins.
        env = make(
            "open_spiel_clobber",
            configuration={"openSpielGameParameters": {"rows": 2, "columns": 2}},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        env.step([{"submission": 9}, {"submission": -1}])  # P0: a1b1
        env.step([{"submission": -1}, {"submission": 1}])  # P1: a2b2
        env.step([{"submission": 12}, {"submission": -1}])  # P0: b1b2
        self.assertTrue(env.done)
        self.assertEqual(env.toJSON()["rewards"], [1, -1])
        final_obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertTrue(final_obs["is_terminal"])
        self.assertEqual(final_obs["winner"], "o")
        self.assertEqual(final_obs["last_move"], "b1b2")
        # Only one white piece left at top-right; everything else empty.
        self.assertEqual(final_obs["board"], [[".", "o"], [".", "."]])

    def test_clobber_invalid_action(self):
        env = make("open_spiel_clobber", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        env.step([{"submission": 999}, {"submission": -1}])  # Invalid action.
        self.assertTrue(env.done)
        playthrough = env.toJSON()
        self.assertEqual(
            playthrough["rewards"],
            [
                open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
                -open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
            ],
        )


if __name__ == "__main__":
    absltest.main()
