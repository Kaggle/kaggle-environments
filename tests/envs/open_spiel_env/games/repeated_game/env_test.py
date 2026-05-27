"""Env-level tests for open_spiel_repeated_game (Prisoner's Dilemma)."""

from absl.testing import absltest

from kaggle_environments import make


class RepeatedGameEnvTest(absltest.TestCase):
    def test_repeated_prisoners_dilemma_agent_playthrough(self):
        """Test repeated Prisoner's Dilemma with random agents for 10 rounds."""
        env = make(
            "open_spiel_repeated_game",
            {"openSpielGameParameters": {"num_repetitions": 10}},
            debug=True,
        )
        env.run(["random", "random"])
        json = env.toJSON()
        self.assertEqual(json["name"], "open_spiel_repeated_game")
        self.assertTrue(all(s == "DONE" for s in json["statuses"]))
        self.assertEqual(len(json["rewards"]), 2)
        self.assertTrue(all(r is not None for r in json["rewards"]))

    def test_repeated_prisoners_dilemma_mutual_cooperate(self):
        """Both players cooperate every round. Expected reward: 5 * 10 = 50 each."""
        env = make(
            "open_spiel_repeated_game",
            {"openSpielGameParameters": {"num_repetitions": 10}},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        for _ in range(10):
            self.assertFalse(env.done)
            self.assertEqual(env.state[0]["status"], "ACTIVE")
            self.assertEqual(env.state[1]["status"], "ACTIVE")
            env.step([{"submission": 0}, {"submission": 0}])  # Both cooperate.
        self.assertTrue(env.done)
        json = env.toJSON()
        self.assertEqual(json["rewards"], [50.0, 50.0])

    def test_repeated_prisoners_dilemma_mutual_defect(self):
        """Both players defect every round. Expected reward: 1 * 10 = 10 each."""
        env = make(
            "open_spiel_repeated_game",
            {"openSpielGameParameters": {"num_repetitions": 10}},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        for _ in range(10):
            env.step([{"submission": 1}, {"submission": 1}])  # Both defect.
        self.assertTrue(env.done)
        json = env.toJSON()
        self.assertEqual(json["rewards"], [10.0, 10.0])

    def test_repeated_prisoners_dilemma_asymmetric(self):
        """P0 always cooperates, P1 always defects. P0 gets 0, P1 gets 100."""
        env = make(
            "open_spiel_repeated_game",
            {"openSpielGameParameters": {"num_repetitions": 10}},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        for _ in range(10):
            env.step([{"submission": 0}, {"submission": 1}])  # P0 cooperate, P1 defect.
        self.assertTrue(env.done)
        json = env.toJSON()
        self.assertEqual(json["rewards"], [0.0, 100.0])


if __name__ == "__main__":
    absltest.main()
