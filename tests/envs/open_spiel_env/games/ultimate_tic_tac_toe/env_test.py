"""Env-level tests for open_spiel_ultimate_tic_tac_toe."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class UltimateTicTacToeEnvTest(absltest.TestCase):
    def test_ultimate_tic_tac_toe_agent_playthrough(self):
        env = make(
            "open_spiel_ultimate_tic_tac_toe",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_ultimate_tic_tac_toe")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))

    def test_ultimate_tic_tac_toe_manual_playthrough(self):
        env = make("open_spiel_ultimate_tic_tac_toe", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step

        # Sequential game: only the current player submits
        # Step 1: Player 0 selects sub-grid 6
        env.step([{"submission": 6}, {"submission": -1}])
        obs_p0 = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs_p0["active_subgrid"], 6)
        self.assertEqual(obs_p0["phase"], "choose_cell")
        self.assertEqual(obs_p0["current_player"], "x")

        # Step 2: Player 0 selects cell 1 of sub-grid 6
        env.step([{"submission": 1}, {"submission": -1}])
        obs_p1 = json.loads(env.state[1]["observation"]["observationString"])
        self.assertEqual(obs_p1["active_subgrid"], 1)
        self.assertEqual(obs_p1["phase"], "choose_cell")
        self.assertEqual(obs_p1["current_player"], "o")
        self.assertEqual(obs_p1["board"][6][1], "x")

        # Step 3: Player 1 selects cell 2 of sub-grid 1
        env.step([{"submission": -1}, {"submission": 2}])
        obs_p0 = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs_p0["active_subgrid"], 2)
        self.assertEqual(obs_p0["phase"], "choose_cell")
        self.assertEqual(obs_p0["current_player"], "x")
        self.assertEqual(obs_p0["board"][1][2], "o")

    def test_ultimate_tic_tac_toe_invalid_action(self):
        env = make("open_spiel_ultimate_tic_tac_toe", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step
        env.step([{"submission": 999}, {"submission": -1}])  # Invalid action
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
