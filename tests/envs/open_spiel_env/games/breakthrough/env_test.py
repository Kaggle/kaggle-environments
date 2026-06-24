"""Env-level tests for open_spiel_breakthrough."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class BreakthroughEnvTest(absltest.TestCase):
    def test_breakthrough_agent_playthrough(self):
        env = make(
            "open_spiel_breakthrough",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_breakthrough")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))
        # Breakthrough cannot draw under standard rules; one side must win.
        rewards = playthrough["rewards"]
        self.assertEqual(sorted(rewards), [-1.0, 1.0])

    def test_breakthrough_manual_playthrough(self):
        # Use a 4x4 board for a short forced win for Black.
        # Action encoding (mixed-base [rows, cols, 6, 2]):
        #   action = r1 * cols * 12 + c1 * 12 + dir * 2 + capture
        # Black dirs 0/1/2 = diag-down-left / down / diag-down-right.
        # White dirs 3/4/5 = diag-up-left / up / diag-up-right.
        env = make(
            "open_spiel_breakthrough",
            configuration={"openSpielGameParameters": {"rows": 4, "columns": 4}},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        moves = [
            (0, 2),  # Black a4a3
            (1, 188),  # White d1d2
            (0, 50),  # Black a3a2
            (1, 176),  # White c1c2
            (0, 101),  # Black a2b1* (diagonal capture into White's home row)
        ]
        for player, action in moves:
            if player == 0:
                env.step([{"submission": action}, {"submission": -1}])
            else:
                env.step([{"submission": -1}, {"submission": action}])
        self.assertTrue(env.done)
        self.assertEqual(env.toJSON()["rewards"], [1, -1])
        final_obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertTrue(final_obs["is_terminal"])
        self.assertEqual(final_obs["winner"], "b")
        self.assertEqual(final_obs["last_move"], "a2b1*")
        self.assertEqual(final_obs["rows"], 4)
        self.assertEqual(final_obs["columns"], 4)
        # Black piece reached White's home row (bottom row label "1").
        self.assertEqual(final_obs["board"][3], ["w", "b", ".", "."])
        # Pieces: Black still has 4, White lost one to the capture.
        self.assertEqual(final_obs["pieces"], {"b": 4, "w": 3})

    def test_breakthrough_invalid_action(self):
        env = make("open_spiel_breakthrough", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # 999 is outside the 8x8 action range (num_distinct_actions == 768).
        env.step([{"submission": 999}, {"submission": -1}])
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
