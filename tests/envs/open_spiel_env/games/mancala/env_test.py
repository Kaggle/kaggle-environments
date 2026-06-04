"""Env-level tests for open_spiel_mancala."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class MancalaEnvTest(absltest.TestCase):
    def test_mancala_agent_playthrough(self):
        env = make(
            "open_spiel_mancala",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_mancala")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))
        final_obs = json.loads(playthrough["steps"][-1][0]["observation"]["observationString"])
        self.assertTrue(final_obs["is_terminal"])
        self.assertIn(final_obs["winner"], (0, 1, "draw"))
        self.assertEqual(len(final_obs["board"]), 14)
        self.assertEqual(final_obs["scores"], [final_obs["stores"]["0"], final_obs["stores"]["1"]])

    def test_mancala_initial_state(self):
        env = make("open_spiel_mancala", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        obs = json.loads(env.state[0]["observation"]["observationString"])
        # Mancala starts with 4 stones in each of the 12 pits, stores empty.
        self.assertEqual(obs["board"], [0, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4])
        self.assertEqual(obs["current_player"], 0)
        self.assertEqual(obs["scores"], [0, 0])
        self.assertFalse(obs["is_terminal"])
        self.assertIsNone(obs["winner"])
        self.assertIsNone(obs["last_action"])

    def test_mancala_invalid_action(self):
        env = make("open_spiel_mancala", debug=True)
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
