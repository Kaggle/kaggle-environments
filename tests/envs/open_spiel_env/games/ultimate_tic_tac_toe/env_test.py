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

        # Setup step: no history yet, X's turn, opening free move.
        env.step([{"submission": -1}, {"submission": -1}])
        obs_setup = json.loads(env.state[0]["observation"]["observationString"])
        self.assertIsNone(obs_setup["active_subgrid"])
        self.assertEqual(obs_setup["phase"], "choose_subgrid")
        self.assertEqual(obs_setup["board_context"], "opening")

        # Step 1: X selects sub-grid 6. Now X plays a cell within a board
        # X themselves just chose -> board_context = "self_selected".
        env.step([{"submission": 6}, {"submission": -1}])
        obs_p0 = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs_p0["active_subgrid"], 6)
        self.assertEqual(obs_p0["phase"], "choose_cell")
        self.assertEqual(obs_p0["board_context"], "self_selected")
        self.assertEqual(obs_p0["current_player"], "x")

        # Step 2: X plays cell 1 of sub-grid 6 -> O is forced to board 1
        # because of X's cell choice: board_context = "opponent_directed".
        env.step([{"submission": 1}, {"submission": -1}])
        obs_p1 = json.loads(env.state[1]["observation"]["observationString"])
        self.assertEqual(obs_p1["active_subgrid"], 1)
        self.assertEqual(obs_p1["phase"], "choose_cell")
        self.assertEqual(obs_p1["board_context"], "opponent_directed")
        self.assertEqual(obs_p1["current_player"], "o")
        self.assertEqual(obs_p1["board"][6][1], "x")

        # Step 3: O plays cell 2 of sub-grid 1 -> X forced to board 2.
        env.step([{"submission": -1}, {"submission": 2}])
        obs_p0 = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs_p0["active_subgrid"], 2)
        self.assertEqual(obs_p0["phase"], "choose_cell")
        self.assertEqual(obs_p0["board_context"], "opponent_directed")
        self.assertEqual(obs_p0["current_player"], "x")
        self.assertEqual(obs_p0["board"][1][2], "o")

    def test_board_context_redirected_when_sent_to_completed_board(self):
        """When the opponent's cell maps to an inactive board, the current
        player gets a free move and board_context should be "redirected"
        (not "opening", which is reserved for the very first turn)."""
        import pyspiel

        game = pyspiel.load_game("ultimate_tic_tac_toe_proxy")
        state = game.new_initial_state()

        import random

        random.seed(1)  # Empirically produces a "redirected" state at step 32.
        found = False
        for _ in range(200):
            if state.is_terminal():
                break
            d = json.loads(state.observation_string(0))
            if d["board_context"] == "redirected":
                self.assertEqual(d["phase"], "choose_subgrid")
                self.assertIsNone(d["active_subgrid"])
                # The move history must be non-empty (otherwise it would be
                # "opening" — that's the whole distinction).
                self.assertGreater(state.move_number(), 0)
                found = True
                break
            state.apply_action(random.choice(state.legal_actions()))
        self.assertTrue(found, "Expected to observe a 'redirected' board_context in this rollout")

    def test_board_context_none_when_terminal(self):
        import random

        import pyspiel

        game = pyspiel.load_game("ultimate_tic_tac_toe_proxy")
        state = game.new_initial_state()
        random.seed(42)
        while not state.is_terminal():
            state.apply_action(random.choice(state.legal_actions()))
        d = json.loads(state.observation_string(0))
        self.assertTrue(d["is_terminal"])
        self.assertIsNone(d["board_context"])
        self.assertIsNone(d["active_subgrid"])
        self.assertIsNone(d["phase"])

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
