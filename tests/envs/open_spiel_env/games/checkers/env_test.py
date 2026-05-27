"""Env-level tests for open_spiel_checkers."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class CheckersEnvTest(absltest.TestCase):
    def test_checkers_agent_playthrough(self):
        env = make(
            "open_spiel_checkers",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_checkers")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))
        final_obs = json.loads(playthrough["steps"][-1][0]["observation"]["observationString"])
        self.assertTrue(final_obs["is_terminal"])
        self.assertIn(final_obs["winner"], ("o", "+", "draw"))

    def test_checkers_manual_playthrough(self):
        env = make("open_spiel_checkers", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        initial_obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertFalse(initial_obs["is_terminal"])
        self.assertIsNone(initial_obs["winner"])
        self.assertEqual(initial_obs["current_player"], "o")
        self.assertIsNone(initial_obs["last_move"])
        self.assertEqual(initial_obs["piece_counts"], {"o": 12, "+": 12, "O": 0, "*": 0})
        # board[0] is rank 1 (bottom): player 0 pieces on odd columns.
        self.assertEqual(initial_obs["board"][0], ["o", ".", "o", ".", "o", ".", "o", "."])
        self.assertEqual(initial_obs["board"][1], [".", "o", ".", "o", ".", "o", ".", "o"])
        # board[7] is rank 8 (top): player 1 pieces.
        self.assertEqual(initial_obs["board"][7], [".", "+", ".", "+", ".", "+", ".", "+"])
        # Action 322 = a3b4 (player 0 moves piece diagonally forward).
        env.step([{"submission": 322}, {"submission": -1}])
        after_obs = json.loads(env.state[1]["observation"]["observationString"])
        self.assertEqual(after_obs["current_player"], "+")
        self.assertEqual(after_obs["last_move"], "a3b4")
        self.assertEqual(after_obs["move_number"], 1)
        # a3 is now empty, b4 has player 0 piece.
        self.assertEqual(after_obs["board"][2][0], ".")  # a3 empty
        self.assertEqual(after_obs["board"][3][1], "o")  # b4 has piece

    def test_checkers_invalid_action(self):
        env = make("open_spiel_checkers", debug=True)
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
