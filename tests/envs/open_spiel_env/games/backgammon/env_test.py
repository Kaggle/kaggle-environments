"""Env-level tests for open_spiel_backgammon."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class BackgammonEnvTest(absltest.TestCase):
    def test_backgammon_agent_playthrough(self):
        env = make(
            "open_spiel_backgammon",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_backgammon")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))
        final_obs = json.loads(playthrough["steps"][-1][0]["observation"]["observationString"])
        self.assertTrue(final_obs["is_terminal"])
        self.assertIn(final_obs["winner"], ("x", "o", "draw"))

    def test_backgammon_observation_is_json(self):
        env = make(
            "open_spiel_backgammon",
            configuration={"seed": 42, "includeLegalActions": True},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertFalse(obs["is_terminal"])
        self.assertIsNone(obs["winner"])
        self.assertIn(obs["current_player"], ("x", "o"))
        # Seed 42: O plays first with dice [3, 6].
        self.assertEqual(obs["current_player"], "o")
        self.assertEqual(obs["dice"], [{"value": 3, "used": False}, {"value": 6, "used": False}])
        # Standard backgammon starting position (OpenSpiel coords).
        self.assertEqual(len(obs["board"]), 24)
        self.assertEqual(obs["board"][0], {"player": "x", "count": 2})
        self.assertEqual(obs["board"][5], {"player": "o", "count": 5})
        self.assertEqual(obs["board"][7], {"player": "o", "count": 3})
        self.assertEqual(obs["board"][11], {"player": "x", "count": 5})
        self.assertEqual(obs["board"][12], {"player": "o", "count": 5})
        self.assertEqual(obs["board"][16], {"player": "x", "count": 3})
        self.assertEqual(obs["board"][18], {"player": "x", "count": 5})
        self.assertEqual(obs["board"][23], {"player": "o", "count": 2})
        self.assertEqual(obs["bar"], {"x": 0, "o": 0})
        self.assertEqual(obs["off"], {"x": 0, "o": 0})
        # 15 checkers per player on the board.
        x_count = sum(p["count"] for p in obs["board"] if p and p["player"] == "x")
        o_count = sum(p["count"] for p in obs["board"] if p and p["player"] == "o")
        self.assertEqual(x_count, 15)
        self.assertEqual(o_count, 15)

    def test_backgammon_invalid_action(self):
        # Seed 7 makes player X (0) move first so the invalid submission
        # from player 0 below is the one actually evaluated.
        env = make("open_spiel_backgammon", configuration={"seed": 7}, debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        env.step([{"submission": 9999}, {"submission": -1}])  # Invalid action.
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
