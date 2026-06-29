"""Env-level tests for open_spiel_markov_soccer."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env

# Cap the horizon so random agents reliably reach a terminal state in test time.
_SHORT_HORIZON_CONFIG = {"openSpielGameParameters": {"horizon": 50}}


class MarkovSoccerEnvTest(absltest.TestCase):
    def test_markov_soccer_agent_playthrough(self):
        env = make(
            "open_spiel_markov_soccer",
            configuration={"includeLegalActions": True, **_SHORT_HORIZON_CONFIG},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_markov_soccer")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))
        # Zero-sum: +1/-1 on a goal, 0/0 on a horizon draw.
        rewards = playthrough["rewards"]
        self.assertEqual(len(rewards), 2)
        self.assertAlmostEqual(rewards[0] + rewards[1], 0.0)

    def test_markov_soccer_observation_is_json(self):
        env = make(
            "open_spiel_markov_soccer",
            configuration={"includeLegalActions": True, **_SHORT_HORIZON_CONFIG},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        obs_p0 = json.loads(env.state[0]["observation"]["observationString"])
        obs_p1 = json.loads(env.state[1]["observation"]["observationString"])

        # Both players see the same (perfect-information) state.
        self.assertEqual(obs_p0, obs_p1)

        # Default Markov Soccer field is 4 rows x 5 cols.
        self.assertEqual(len(obs_p0["board"]), 4)
        self.assertTrue(all(len(row) == 5 for row in obs_p0["board"]))

        self.assertEqual(obs_p0["actions"], ["up", "down", "left", "right", "stand"])
        self.assertFalse(obs_p0["is_terminal"])
        self.assertIsNone(obs_p0["winner"])

        # Players are placed; ball has spawned (loose) after the initial chance node.
        self.assertIsNotNone(obs_p0["player_a_pos"])
        self.assertIsNotNone(obs_p0["player_b_pos"])
        self.assertIsNotNone(obs_p0["ball_pos"])
        self.assertIsNone(obs_p0["ball_owner"])

        # Simultaneous-move dispatch surfaces as PlayerId.SIMULTANEOUS (-2).
        self.assertEqual(env.state[0]["observation"]["currentPlayer"], -2)

    def test_markov_soccer_terminal_has_winner(self):
        env = make(
            "open_spiel_markov_soccer",
            configuration={"includeLegalActions": True, **_SHORT_HORIZON_CONFIG},
            debug=True,
        )
        env.run(["random", "random"])
        final_obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertTrue(final_obs["is_terminal"])
        self.assertIn(final_obs["winner"], ["A", "B", "draw"])

    def test_markov_soccer_invalid_action(self):
        env = make(
            "open_spiel_markov_soccer",
            configuration=_SHORT_HORIZON_CONFIG,
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        # 999 is not a legal Markov Soccer action (legal: 0..4).
        env.step([{"submission": 999}, {"submission": 0}])
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
