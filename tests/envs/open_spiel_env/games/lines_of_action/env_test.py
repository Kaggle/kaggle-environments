"""Env-level tests for open_spiel_lines_of_action."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class LinesOfActionEnvTest(absltest.TestCase):
    def test_lines_of_action_agent_playthrough(self):
        env = make(
            "open_spiel_lines_of_action",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_lines_of_action")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))

    def test_lines_of_action_manual_playthrough(self):
        env = make("open_spiel_lines_of_action", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # Initial board has black (x) on top/bottom rows except corners,
        # white (o) on left/right columns except corners. Black moves first.
        initial_obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertFalse(initial_obs["is_terminal"])
        self.assertIsNone(initial_obs["winner"])
        self.assertEqual(initial_obs["current_player"], "x")
        self.assertIsNone(initial_obs["last_move"])
        # board[0] is rank 1 (bottom). Bottom rank: ".xxxxxx."; rank 2: "o......o".
        self.assertEqual(initial_obs["board"][0], [".", "x", "x", "x", "x", "x", "x", "."])
        self.assertEqual(initial_obs["board"][1], ["o", ".", ".", ".", ".", ".", ".", "o"])
        # Action 142 is "b1-h1" (move b1 piece along the bottom row to h1).
        env.step([{"submission": 142}, {"submission": -1}])
        after_obs = json.loads(env.state[1]["observation"]["observationString"])
        self.assertEqual(after_obs["current_player"], "o")
        self.assertEqual(after_obs["last_move"], "b1-h1")
        self.assertEqual(after_obs["move_number"], 1)
        self.assertEqual(after_obs["board"][0], [".", ".", "x", "x", "x", "x", "x", "x"])

    def test_proxy_exposes_full_move_history(self):
        """The proxy must surface the full game's move history (both
        players, oldest first) so the harness can render it in the prompt
        -- the framework's per-agent ``move_history`` argument only
        contains the calling agent's own moves."""
        env = make(
            "open_spiel_lines_of_action",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup.
        # No moves yet -- move_history is the empty list.
        initial_obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(initial_obs["move_history"], [])
        # Black plays b1-h1 (action id 142).
        env.step([{"submission": 142}, {"submission": -1}])
        # White plays its first legal action (looked up from the env so the
        # test doesn't have to know which moves are legal post-b1-h1).
        white_legal = env.state[1]["observation"]["legalActions"]
        white_legal_strs = env.state[1]["observation"]["legalActionStrings"]
        white_action = white_legal[0]
        white_action_str = white_legal_strs[0]
        env.step([{"submission": -1}, {"submission": white_action}])
        # Both players see the full history in their observation.
        for player_idx in (0, 1):
            obs = json.loads(env.state[player_idx]["observation"]["observationString"])
            self.assertEqual(obs["move_history"], ["b1-h1", white_action_str])

    def test_lines_of_action_invalid_action(self):
        env = make("open_spiel_lines_of_action", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        env.step([{"submission": 999}, {"submission": -1}])  # Invalid action.
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
