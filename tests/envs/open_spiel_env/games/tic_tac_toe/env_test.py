"""Env-level tests for open_spiel_tic_tac_toe."""

from absl.testing import absltest

from kaggle_environments import make


class TicTacToeEnvTest(absltest.TestCase):
    def test_tic_tac_toe_agent_playthrough(self):
        env = make("open_spiel_tic_tac_toe", debug=True)
        env.run(["random", "random"])
        json = env.toJSON()
        self.assertEqual(json["name"], "open_spiel_tic_tac_toe")
        self.assertTrue(all([status == "DONE" for status in json["statuses"]]))

    def test_tic_tac_toe_manual_playthrough(self):
        env = make("open_spiel_tic_tac_toe", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        env.step([{"submission": 0}, {"submission": -1}])
        env.step([{"submission": -1}, {"submission": 1}])
        env.step([{"submission": 3}, {"submission": -1}])
        env.step([{"submission": -1}, {"submission": 4}])
        env.step([{"submission": 6}, {"submission": -1}])
        self.assertTrue(env.done)
        self.assertEqual(env.toJSON()["rewards"], [1, -1])


if __name__ == "__main__":
    absltest.main()
