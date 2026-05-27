"""Env-level tests for open_spiel_goofspiel."""

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class GoofspielEnvTest(absltest.TestCase):
    def test_goofspiel_agent_playthrough(self):
        open_spiel_env._register_game_envs(["goofspiel(num_cards=4,points_order=descending,returns_type=total_points)"])
        env = make("open_spiel_goofspiel", debug=True)
        env.run(["random", "random"])
        json = env.toJSON()
        self.assertEqual(json["name"], "open_spiel_goofspiel")
        self.assertTrue(all([status == "DONE" for status in json["statuses"]]))
        # Both players should have rewards (total_points mode).
        self.assertEqual(len(json["rewards"]), 2)
        self.assertTrue(all(r is not None for r in json["rewards"]))

    def test_goofspiel_manual_playthrough(self):
        open_spiel_env._register_game_envs(["goofspiel(num_cards=4,points_order=descending,returns_type=total_points)"])
        env = make("open_spiel_goofspiel", debug=True)
        env.reset()
        # Initial setup step.
        env.step([{"submission": -1}, {"submission": -1}])
        # After setup, both players should be ACTIVE (simultaneous node).
        self.assertEqual(env.state[0]["status"], "ACTIVE")
        self.assertEqual(env.state[1]["status"], "ACTIVE")
        # Play all 4 rounds: both players submit actions each step.
        # With descending point order and 4 cards, there are 4 bidding rounds.
        # Legal actions are card indices (0-3 initially).
        for _ in range(4):
            if env.done:
                break
            env.step([{"submission": 0}, {"submission": 0}])
        self.assertTrue(env.done)
        json = env.toJSON()
        self.assertTrue(all([status == "DONE" for status in json["statuses"]]))


if __name__ == "__main__":
    absltest.main()
