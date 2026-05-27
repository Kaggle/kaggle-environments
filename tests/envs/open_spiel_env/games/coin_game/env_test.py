"""Env-level tests for open_spiel_coin_game."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class CoinGameEnvTest(absltest.TestCase):
    def test_coin_game_agent_playthrough(self):
        env = make(
            "open_spiel_coin_game",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_coin_game")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))

    def test_coin_game_observation_is_json(self):
        env = make(
            "open_spiel_coin_game",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        obs_p0 = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs_p0["phase"], "play")
        self.assertEqual(obs_p0["current_player"], 0)
        self.assertEqual(obs_p0["num_rows"], 8)
        self.assertEqual(obs_p0["num_columns"], 8)
        self.assertEqual(obs_p0["episode_length"], 20)
        self.assertEqual(obs_p0["coin_colors"], ["a", "b", "c"])
        self.assertEqual(len(obs_p0["board"]), 8)
        self.assertTrue(all(len(row) == 8 for row in obs_p0["board"]))
        # Player 0 sees their own preference but not opponent's.
        self.assertIn(obs_p0["your_preference"], ["a", "b", "c"])
        self.assertEqual(obs_p0["your_player_id"], 0)
        self.assertNotIn("preferences", obs_p0)
        # Both players are placed on the board.
        self.assertIsNotNone(obs_p0["player_positions"]["0"])
        self.assertIsNotNone(obs_p0["player_positions"]["1"])
        self.assertFalse(obs_p0["is_terminal"])
        self.assertEqual(obs_p0["move_number"], 0)

    def test_coin_game_observation_hides_opponent_preference(self):
        env = make("open_spiel_coin_game", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        obs_p0 = json.loads(env.state[0]["observation"]["observationString"])
        obs_p1 = json.loads(env.state[1]["observation"]["observationString"])
        self.assertEqual(obs_p0["your_player_id"], 0)
        self.assertEqual(obs_p1["your_player_id"], 1)
        # Each player only learns its own preference; never the opponent's.
        self.assertNotIn("preferences", obs_p0)
        self.assertNotIn("preferences", obs_p1)

    def test_coin_game_terminal_reveals_preferences_and_returns(self):
        env = make(
            "open_spiel_coin_game",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        final_obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertTrue(final_obs["is_terminal"])
        self.assertEqual(final_obs["move_number"], 20)
        self.assertEqual(final_obs["moves_remaining"], 0)
        self.assertIn(final_obs["winner"], [0, 1, "draw"])
        self.assertEqual(len(final_obs["returns"]), 2)
        # On terminal, both preferences are revealed for scoring inspection.
        self.assertIn("0", final_obs["preferences"])
        self.assertIn("1", final_obs["preferences"])

    def test_coin_game_invalid_action(self):
        env = make("open_spiel_coin_game", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
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
