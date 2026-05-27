"""Env-level tests for open_spiel_gin_rummy."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class GinRummyEnvTest(absltest.TestCase):
    def test_gin_rummy_agent_playthrough(self):
        env = make(
            "open_spiel_gin_rummy",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_gin_rummy")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))

    def test_gin_rummy_observation_is_json(self):
        env = make(
            "open_spiel_gin_rummy",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # After dealing, it is player 0's turn (FirstUpcard phase).
        obs_p0 = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs_p0["phase"], "FirstUpcard")
        self.assertEqual(obs_p0["current_player"], 0)
        self.assertFalse(obs_p0["is_terminal"])
        self.assertEqual(obs_p0["knock_card"], 10)
        self.assertEqual(obs_p0["stock_size"], 31)
        self.assertIsNotNone(obs_p0["upcard"])
        # Player 0 sees their own 10-card hand; opponent's hand is hidden.
        self.assertEqual(len(obs_p0["hands"]["0"]), 10)
        self.assertEqual(obs_p0["hands"]["1"], [])
        self.assertIsNotNone(obs_p0["deadwood"]["0"])
        self.assertIsNone(obs_p0["deadwood"]["1"])

    def test_gin_rummy_observation_hides_opponent(self):
        env = make("open_spiel_gin_rummy", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # Each player should only see their own hand in their observation.
        obs_p0 = json.loads(env.state[0]["observation"]["observationString"])
        obs_p1 = json.loads(env.state[1]["observation"]["observationString"])
        self.assertEqual(len(obs_p0["hands"]["0"]), 10)
        self.assertEqual(obs_p0["hands"]["1"], [])
        self.assertEqual(obs_p1["hands"]["0"], [])
        self.assertEqual(len(obs_p1["hands"]["1"]), 10)

    def test_gin_rummy_invalid_action(self):
        env = make("open_spiel_gin_rummy", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # In FirstUpcard phase only actions 52 (Draw upcard) and 54 (Pass)
        # are legal, so 0 (the As card) is an invalid action.
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
