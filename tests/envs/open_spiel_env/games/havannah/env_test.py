"""Env-level tests for open_spiel_havannah."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class HavannahEnvTest(absltest.TestCase):
    def test_havannah_agent_playthrough(self):
        env = make(
            "open_spiel_havannah",
            configuration={
                "includeLegalActions": True,
                "openSpielGameParameters": {"board_size": 4},
            },
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_havannah")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))
        rewards = playthrough["rewards"]
        self.assertIn(sorted(rewards), ([-1.0, 1.0], [0.0, 0.0]))

    def test_havannah_manual_playthrough(self):
        env = make(
            "open_spiel_havannah",
            configuration={"openSpielGameParameters": {"board_size": 4}},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # P0 fills the top edge a1..d1 to form a bridge between corners a1 and d1.
        # Action encoding: x + y * diameter (diameter = 2*board_size - 1 = 7).
        # P1 plays harmless cells in row 2 (a2, b2, c2 = actions 7, 8, 9).
        moves = [(0, 0), (1, 7), (0, 1), (1, 8), (0, 2), (1, 9), (0, 3)]
        for player, action in moves:
            if player == 0:
                env.step([{"submission": action}, {"submission": -1}])
            else:
                env.step([{"submission": -1}, {"submission": action}])
        self.assertTrue(env.done)
        self.assertEqual(env.toJSON()["rewards"], [1, -1])
        final_obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertTrue(final_obs["is_terminal"])
        self.assertEqual(final_obs["winner"], "x")
        self.assertEqual(final_obs["board_size"], 4)
        self.assertEqual(final_obs["last_move"], "d1")
        # Top row (y=0) has 4 cells, all filled by x.
        self.assertEqual(final_obs["board"][0], ["x", "x", "x", "x"])
        # Row y=1 has 5 cells; first three are o, rest empty.
        self.assertEqual(final_obs["board"][1][:3], ["o", "o", "o"])

    def test_havannah_invalid_action(self):
        env = make(
            "open_spiel_havannah",
            configuration={"openSpielGameParameters": {"board_size": 4}},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # Action 6 falls inside a cut-off corner; it is not a legal cell.
        env.step([{"submission": 6}, {"submission": -1}])
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
