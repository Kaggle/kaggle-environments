"""Env-level tests for open_spiel_y."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class YEnvTest(absltest.TestCase):
    def test_y_agent_playthrough(self):
        env = make(
            "open_spiel_y",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_y")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))
        rewards = playthrough["rewards"]
        self.assertEqual(sorted(rewards), [-1.0, 1.0])

    def test_y_manual_playthrough(self):
        env = make(
            "open_spiel_y",
            configuration={"openSpielGameParameters": {"board_size": 8}},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # P0 connects the left column (a1..a8), touching all three sides.
        # Action encoding: row * 8 + col, with rows/cols 0-indexed.
        moves = [0, 1, 8, 2, 16, 3, 24, 4, 32, 5, 40, 6, 48, 7, 56]
        for i, action in enumerate(moves):
            if i % 2 == 0:
                env.step([{"submission": action}, {"submission": -1}])
            else:
                env.step([{"submission": -1}, {"submission": action}])
        self.assertTrue(env.done)
        self.assertEqual(env.toJSON()["rewards"], [1, -1])
        final_obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertTrue(final_obs["is_terminal"])
        self.assertEqual(final_obs["winner"], "x")
        self.assertEqual(final_obs["board_size"], 8)
        self.assertEqual(final_obs["last_move"], "a8")
        self.assertEqual(final_obs["board"][0][0], "x")
        self.assertEqual(final_obs["board"][7][0], "x")

    def test_y_invalid_action(self):
        env = make("open_spiel_y", debug=True)
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
