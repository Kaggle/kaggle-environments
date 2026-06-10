"""Env-level tests for open_spiel_quoridor."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class QuoridorEnvTest(absltest.TestCase):
    def test_quoridor_agent_playthrough(self):
        env = make(
            "open_spiel_quoridor",
            configuration={
                "includeLegalActions": True,
                "openSpielGameParameters": {"board_size": 3, "wall_count": 1},
            },
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_quoridor")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))
        # A 3x3 board has 3 cells of distance and almost no wall placements,
        # so random play always terminates with a winner well within the step
        # budget. If this starts producing draws, it signals a real regression
        # (e.g. the env exhausting steps without terminating).
        self.assertEqual(sorted(playthrough["rewards"]), [-1.0, 1.0])

    def test_quoridor_manual_playthrough(self):
        """3x3 Quoridor: player 0 walks b3 -> b2 -> b1 to win."""
        env = make(
            "open_spiel_quoridor",
            configuration={"openSpielGameParameters": {"board_size": 3, "wall_count": 1}},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # Pawn-move actions are relative to the current player's location, so the
        # same action id ``2`` means "step up" for whichever player is to move.
        env.step([{"submission": 2}, {"submission": -1}])  # P0: b3 -> b2
        env.step([{"submission": -1}, {"submission": 10}])  # P1: b1 -> a1 (dodge)
        env.step([{"submission": 2}, {"submission": -1}])  # P0: b2 -> b1 (win)
        self.assertTrue(env.done)
        self.assertEqual(env.toJSON()["rewards"], [1.0, -1.0])
        final_obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertTrue(final_obs["is_terminal"])
        self.assertEqual(final_obs["winner"], "x")
        self.assertEqual(final_obs["board_size"], 3)
        self.assertEqual(final_obs["pawns"], {"x": "b1", "o": "a1"})
        # Neither player placed a wall.
        self.assertEqual(final_obs["walls_remaining"], {"x": 1, "o": 1})
        self.assertEqual(final_obs["vertical_walls"], [])
        self.assertEqual(final_obs["horizontal_walls"], [])

    def test_quoridor_wall_parsing(self):
        """Verify both vertical and horizontal walls are parsed from the ASCII."""
        env = make(
            "open_spiel_quoridor",
            configuration={"openSpielGameParameters": {"board_size": 9}},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        env.step([{"submission": 1}, {"submission": -1}])  # P0: "a1v"
        env.step([{"submission": -1}, {"submission": 21}])  # P1: "c1h"
        obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertIn("a1v", obs["vertical_walls"])
        self.assertIn("c1h", obs["horizontal_walls"])
        self.assertEqual(obs["walls_remaining"], {"x": 9, "o": 9})

    def test_quoridor_walls_remaining_after_pawn_moves(self):
        """Pawn moves must not be miscounted as walls.

        Regression test for the wall-counter: an earlier version classified
        history actions by raw id parity, which would silently treat any
        pawn-move id with an odd diameter coordinate as a wall placement.
        """
        env = make(
            "open_spiel_quoridor",
            configuration={"openSpielGameParameters": {"board_size": 9}},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup.
        # Both players move their pawn forward one square. No walls placed.
        env.step([{"submission": 2}, {"submission": -1}])  # P0: e9 -> e8
        env.step([{"submission": -1}, {"submission": 2}])  # P1: e1 -> e2
        obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs["walls_remaining"], {"x": 10, "o": 10})
        self.assertEqual(obs["vertical_walls"], [])
        self.assertEqual(obs["horizontal_walls"], [])

    def test_quoridor_invalid_action(self):
        env = make(
            "open_spiel_quoridor",
            configuration={"openSpielGameParameters": {"board_size": 3, "wall_count": 1}},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # On a 3x3 board the diameter is 5, so action ids range over 0..24.
        # Action 999 is well outside that range and is not legal.
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
