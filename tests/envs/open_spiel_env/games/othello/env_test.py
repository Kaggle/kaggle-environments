"""Env-level tests for open_spiel_othello."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class OthelloEnvTest(absltest.TestCase):
    def test_othello_agent_playthrough(self):
        env = make(
            "open_spiel_othello",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_othello")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))
        # Rewards must sum to zero (draw is +0/+0, else +1/-1).
        self.assertEqual(sum(playthrough["rewards"]), 0.0)

    def test_othello_initial_state_schema(self):
        env = make("open_spiel_othello", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])
        obs = json.loads(env.state[0]["observation"]["observationString"])

        # Board layout: 8x8, four disks in the standard starting square.
        self.assertEqual(len(obs["board"]), 8)
        self.assertTrue(all(len(row) == 8 for row in obs["board"]))
        self.assertEqual(obs["board"][3][3], "o")
        self.assertEqual(obs["board"][3][4], "x")
        self.assertEqual(obs["board"][4][3], "x")
        self.assertEqual(obs["board"][4][4], "o")
        # Every other cell empty.
        for r in range(8):
            for c in range(8):
                if (r, c) not in {(3, 3), (3, 4), (4, 3), (4, 4)}:
                    self.assertEqual(obs["board"][r][c], "")

        self.assertEqual(obs["rows"], 8)
        self.assertEqual(obs["columns"], 8)
        self.assertEqual(obs["current_player"], "x")
        self.assertFalse(obs["is_terminal"])
        self.assertIsNone(obs["winner"])
        self.assertEqual(obs["disks"], {"x": 2, "o": 2})
        self.assertIsNone(obs["last_move"])
        self.assertEqual(obs["move_history"], [])
        self.assertEqual(obs["move_number"], 0)
        self.assertFalse(obs["must_pass"])

    def test_othello_manual_playthrough(self):
        # Standard opening sequence: Black d3, White c3, Black b3.
        # Action encoding: action_id = row * 8 + col, with row 0 = rank 1
        # and col 0 = file 'a'. d3 = row 2, col 3 -> 19.
        env = make("open_spiel_othello", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        env.step([{"submission": 19}, {"submission": -1}])  # Black d3
        env.step([{"submission": -1}, {"submission": 18}])  # White c3
        env.step([{"submission": 17}, {"submission": -1}])  # Black b3

        obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs["move_history"], ["d3", "c3", "b3"])
        self.assertEqual(obs["last_move"], "b3")
        self.assertEqual(obs["move_number"], 3)
        self.assertEqual(obs["current_player"], "o")
        # After d3 + c3 + b3: Black placed at b3/c3/d3 (with c3 captured
        # back to White by the c3 reply), and the d4 disk was flipped
        # then restored; the c4 disk was flipped by Black's b3 capture.
        # Verify piece total is what OpenSpiel says rather than pinning
        # every square.
        black_count = sum(row.count("x") for row in obs["board"])
        white_count = sum(row.count("o") for row in obs["board"])
        self.assertEqual(obs["disks"], {"x": black_count, "o": white_count})
        self.assertEqual(black_count + white_count, 4 + 3)  # 4 starting + 3 placements
        self.assertFalse(obs["is_terminal"])

    def test_othello_invalid_action(self):
        env = make("open_spiel_othello", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        # 999 is well outside the 65-action range (64 cells + pass).
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
