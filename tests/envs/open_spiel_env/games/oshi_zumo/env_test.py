"""Env-level tests for open_spiel_oshi_zumo."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class OshiZumoEnvTest(absltest.TestCase):
    def test_oshi_zumo_agent_playthrough(self):
        env = make(
            "open_spiel_oshi_zumo",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_oshi_zumo")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))
        rewards = playthrough["rewards"]
        self.assertEqual(len(rewards), 2)
        # Zero-sum game.
        self.assertAlmostEqual(rewards[0] + rewards[1], 0.0)
        # Verify the proxy returns JSON observations.
        obs_str = playthrough["steps"][1][0]["observation"]["observationString"]
        obs = json.loads(obs_str)
        self.assertIn("field", obs)
        self.assertIn("coins", obs)
        self.assertIn("wrestler_position", obs)

    def test_oshi_zumo_manual_playthrough(self):
        env = make(
            "open_spiel_oshi_zumo",
            {"openSpielGameParameters": {"coins": 4, "size": 1, "horizon": 100}},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        # Both players bid all 4 coins -> wrestler stays at center, coins exhausted.
        env.step([{"submission": 4}, {"submission": 4}])
        self.assertTrue(env.done)
        json_out = env.toJSON()
        self.assertTrue(all(status == "DONE" for status in json_out["statuses"]))
        # Wrestler still at center on a size=1 field -> draw.
        self.assertEqual(json_out["rewards"], [0.0, 0.0])
        obs = json.loads(json_out["steps"][-1][0]["observation"]["observationString"])
        self.assertEqual(obs["coins"], [0, 0])
        self.assertTrue(obs["is_terminal"])

    def test_oshi_zumo_invalid_action(self):
        env = make("open_spiel_oshi_zumo", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        # Bidding more coins than the player has is illegal.
        env.step([{"submission": 999}, {"submission": 0}])
        self.assertTrue(env.done)
        json_out = env.toJSON()
        self.assertEqual(json_out["rewards"][0], open_spiel_env.DEFAULT_INVALID_ACTION_REWARD)
        self.assertEqual(json_out["rewards"][1], -open_spiel_env.DEFAULT_INVALID_ACTION_REWARD)


if __name__ == "__main__":
    absltest.main()
