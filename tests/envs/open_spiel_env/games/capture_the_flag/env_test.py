"""Env-level tests for open_spiel_capture_the_flag."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class CaptureTheFlagEnvTest(absltest.TestCase):
    def test_capture_the_flag_agent_playthrough(self):
        env = make(
            "open_spiel_capture_the_flag",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_capture_the_flag")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))

    def test_capture_the_flag_observation_is_json(self):
        env = make(
            "open_spiel_capture_the_flag",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        obs = json.loads(env.state[0]["observation"]["observationString"])
        # Default grid is 5x7 with bases at (2,0) and (2,6).
        self.assertEqual(obs["num_rows"], 5)
        self.assertEqual(obs["num_cols"], 7)
        self.assertEqual(obs["a_base"], [2, 0])
        self.assertEqual(obs["b_base"], [2, 6])
        self.assertEqual(obs["a_pos"], [2, 0])
        self.assertEqual(obs["b_pos"], [2, 6])
        self.assertEqual(obs["flag_a_pos"], [2, 0])
        self.assertEqual(obs["flag_b_pos"], [2, 6])
        self.assertIsNone(obs["carrier_a"])
        self.assertIsNone(obs["carrier_b"])
        self.assertEqual(obs["score"], [0, 0])
        self.assertEqual(obs["move_number"], 0)
        self.assertEqual(obs["current_player"], "simultaneous")
        self.assertFalse(obs["is_terminal"])
        self.assertIsNone(obs["winner"])
        self.assertEqual(obs["action_names"], ["North", "East", "South", "West", "Stay"])
        self.assertEqual(len(obs["board"]), 5)
        self.assertTrue(all(len(row) == 7 for row in obs["board"]))

    def test_capture_the_flag_step_advances_positions(self):
        env = make(
            "open_spiel_capture_the_flag",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        # Both submit North (action 0). Grid rows go top-down, so North = row-1.
        env.step([{"submission": 0}, {"submission": 0}])
        obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs["a_pos"], [1, 0])
        self.assertEqual(obs["b_pos"], [1, 6])
        self.assertEqual(obs["move_number"], 1)

    def test_capture_the_flag_terminal_reveals_winner(self):
        env = make(
            "open_spiel_capture_the_flag",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        final = json.loads(env.state[0]["observation"]["observationString"])
        self.assertTrue(final["is_terminal"])
        self.assertIn(final["winner"], [0, 1, "draw"])

    def test_capture_the_flag_invalid_action(self):
        env = make("open_spiel_capture_the_flag", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        env.step([{"submission": 999}, {"submission": 0}])  # Invalid action.
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
