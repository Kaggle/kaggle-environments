"""Env-level tests for open_spiel_dark_hex."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class DarkHexEnvTest(absltest.TestCase):
    def test_dark_hex_agent_playthrough(self):
        env = make(
            "open_spiel_dark_hex",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_dark_hex")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))
        rewards = playthrough["rewards"]
        self.assertEqual(sorted(rewards), [-1.0, 1.0])

    def test_dark_hex_manual_playthrough(self):
        env = make("open_spiel_dark_hex", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        env.step([{"submission": 1}, {"submission": -1}])  # p0: b1
        env.step([{"submission": -1}, {"submission": 0}])  # p1: a1
        env.step([{"submission": 4}, {"submission": -1}])  # p0: b2
        env.step([{"submission": -1}, {"submission": 3}])  # p1: a2
        env.step([{"submission": 7}, {"submission": -1}])  # p0: b3 (wins)
        self.assertTrue(env.done)
        self.assertEqual(env.toJSON()["rewards"], [1, -1])
        final_obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertTrue(final_obs["is_terminal"])
        self.assertEqual(final_obs["winner"], "x")

    def test_dark_hex_observation_hides_opponent(self):
        env = make("open_spiel_dark_hex", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        env.step([{"submission": 4}, {"submission": -1}])  # p0: b2 (center)
        # After p0's move, p1's view should not see the new x piece.
        obs_p1 = json.loads(env.state[1]["observation"]["observationString"])
        self.assertEqual(obs_p1["board"], [["."] * 3] * 3)
        # p0's view should see their own x piece.
        obs_p0 = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs_p0["board"][1][1], "x")

    def test_dark_hex_invalid_action(self):
        env = make("open_spiel_dark_hex", debug=True)
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
