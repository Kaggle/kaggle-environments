"""Env-level tests for open_spiel_matching_pennies_3p."""

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class MatchingPennies3pEnvTest(absltest.TestCase):
    def test_matching_pennies_manual(self):
        open_spiel_env._register_game_envs(["matching_pennies_3p"])
        env = make("open_spiel_matching_pennies_3p", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}, {"submission": -1}])  # Setup.
        # All 3 players should be ACTIVE.
        for pid in range(3):
            self.assertEqual(env.state[pid]["status"], "ACTIVE")
        # All play Heads (action 0).
        env.step([{"submission": 0}, {"submission": 0}, {"submission": 0}])
        self.assertTrue(env.done)
        json = env.toJSON()
        self.assertTrue(all([status == "DONE" for status in json["statuses"]]))
        self.assertEqual(len(json["rewards"]), 3)


if __name__ == "__main__":
    absltest.main()
