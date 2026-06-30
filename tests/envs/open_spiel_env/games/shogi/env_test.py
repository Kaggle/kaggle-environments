"""Env-level tests for open_spiel_shogi."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class ShogiEnvTest(absltest.TestCase):
    def test_shogi_agent_playthrough(self):
        env = make(
            "open_spiel_shogi",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_shogi")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))
        # Shogi is zero-sum: one side wins (±1) or it ends in a draw (0/0).
        rewards = playthrough["rewards"]
        self.assertIn(sorted(rewards), ([-1.0, 1.0], [0.0, 0.0]))

    def test_shogi_initial_state(self):
        env = make("open_spiel_shogi", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs["current_player"], "b")
        self.assertFalse(obs["is_terminal"])
        self.assertIsNone(obs["winner"])
        self.assertEqual(obs["move_number"], 1)
        self.assertIsNone(obs["last_move"])
        self.assertEqual(obs["move_history"], [])
        self.assertEqual(obs["captured"], {"b": {}, "w": {}})
        # Standard 9x9 initial shogi position with Sente (uppercase) on the
        # bottom three ranks and Gote (lowercase) on the top three.
        board = obs["board"]
        self.assertEqual(len(board), 9)
        self.assertTrue(all(len(row) == 9 for row in board))
        self.assertEqual(board[0], ["l", "n", "s", "g", "k", "g", "s", "n", "l"])
        self.assertEqual(board[2], ["p"] * 9)
        self.assertEqual(board[6], ["P"] * 9)
        self.assertEqual(board[8], ["L", "N", "S", "G", "K", "G", "S", "N", "L"])
        # Middle three ranks are empty.
        for rank in (3, 4, 5):
            self.assertEqual(board[rank], ["."] * 9)
        self.assertEqual(
            obs["sfen"],
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        )

    def test_shogi_manual_playthrough(self):
        env = make("open_spiel_shogi", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # Action 18 corresponds to "9i9h" -- Sente's lance shuffle.
        env.step([{"submission": 18}, {"submission": -1}])
        obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs["current_player"], "w")
        # SFEN move number is the full-move counter (chess convention):
        # it stays at 1 after Sente's first half-move and only ticks up
        # after Gote replies.
        self.assertEqual(obs["move_number"], 1)
        self.assertEqual(obs["last_move"], "9i9h")
        self.assertEqual(obs["move_history"], ["9i9h"])
        # Bottom-left corner emptied; lance now sits one square up.
        self.assertEqual(obs["board"][8][0], ".")
        self.assertEqual(obs["board"][7][0], "L")
        self.assertFalse(obs["is_terminal"])

    def test_shogi_invalid_action(self):
        env = make("open_spiel_shogi", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # Action 0 is not in shogi's initial legal action list.
        env.step([{"submission": 0}, {"submission": -1}])
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
