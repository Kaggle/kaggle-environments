"""Env-level tests for open_spiel_snake."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class SnakeEnvTest(absltest.TestCase):
    def test_snake_agent_playthrough(self):
        env = make(
            "open_spiel_snake",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        json_out = env.toJSON()
        self.assertEqual(json_out["name"], "open_spiel_snake")
        self.assertTrue(all(status == "DONE" for status in json_out["statuses"]))

    def test_snake_observation_is_json(self):
        env = make(
            "open_spiel_snake",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs["num_rows"], 10)
        self.assertEqual(obs["num_columns"], 10)
        self.assertEqual(obs["num_players"], 2)
        self.assertEqual(len(obs["board"]), 10)
        self.assertTrue(all(len(row) == 10 for row in obs["board"]))
        self.assertEqual(obs["snakes"][0]["body"], [[1, 1]])
        self.assertEqual(obs["snakes"][1]["body"], [[8, 8]])
        self.assertTrue(obs["snakes"][0]["alive"])
        self.assertTrue(obs["snakes"][1]["alive"])
        self.assertEqual(obs["scores"], [0.0, 0.0])
        self.assertEqual(obs["current_player"], 0)
        self.assertFalse(obs["is_terminal"])
        self.assertIsNone(obs["winner"])
        self.assertEqual(obs["turn"], 0)
        # Food is placed on an empty square.
        self.assertIsNotNone(obs["food"])
        fr, fc = obs["food"]
        self.assertEqual(obs["board"][fr][fc], "*")

    def test_snake_manual_playthrough(self):
        # Sequential implementation of simultaneous play: each player submits
        # in turn, then the buffered moves are applied together. Drive both
        # snakes off the board on the first round so both die and the game
        # terminates immediately.
        env = make("open_spiel_snake", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        # P0 at (1,1): UP -> (0,1)? still on board. Send LEFT (2) twice to die.
        # First round: both submit a move; resolution moves snakes and may kill.
        # P0 LEFT from (1,1) -> (1,0) (alive). P1 RIGHT from (8,8) -> (8,9) (alive).
        # Second round: P0 LEFT from (1,0) -> (1,-1) (wall, dies).
        # P1 RIGHT from (8,9) -> (8,10) (wall, dies).
        env.step([{"submission": 2}, {"submission": -1}])  # P0 LEFT
        env.step([{"submission": -1}, {"submission": 3}])  # P1 RIGHT
        env.step([{"submission": 2}, {"submission": -1}])  # P0 LEFT (off board)
        env.step([{"submission": -1}, {"submission": 3}])  # P1 RIGHT (off board)
        self.assertTrue(env.done)
        obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertTrue(obs["is_terminal"])
        self.assertFalse(obs["snakes"][0]["alive"])
        self.assertFalse(obs["snakes"][1]["alive"])

    def test_snake_invalid_action(self):
        env = make("open_spiel_snake", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        env.step([{"submission": 999}, {"submission": -1}])  # Invalid action.
        self.assertTrue(env.done)
        json_out = env.toJSON()
        self.assertEqual(
            json_out["rewards"],
            [
                open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
                -open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
            ],
        )


if __name__ == "__main__":
    absltest.main()
