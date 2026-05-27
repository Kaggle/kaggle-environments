"""Env-level tests for open_spiel_python_ant_foraging."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class PythonAntForagingEnvTest(absltest.TestCase):
    def test_python_ant_foraging_agent_playthrough(self):
        env = make(
            "open_spiel_python_ant_foraging",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_python_ant_foraging")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))
        # Cooperative game: both ants share the same reward.
        rewards = playthrough["rewards"]
        self.assertEqual(rewards[0], rewards[1])
        final_obs = json.loads(playthrough["steps"][-1][0]["observation"]["observationString"])
        self.assertTrue(final_obs["is_terminal"])

    def test_python_ant_foraging_observation_is_json(self):
        env = make(
            "open_spiel_python_ant_foraging",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs["grid_size"], 8)
        self.assertEqual(obs["num_ants"], 2)
        self.assertEqual(obs["num_food"], 3)
        self.assertEqual(obs["max_turns"], 50)
        self.assertEqual(obs["turn"], 0)
        self.assertEqual(obs["food_collected"], 0)
        self.assertEqual(obs["score"], 0)
        self.assertEqual(obs["current_player"], 0)
        self.assertFalse(obs["is_terminal"])
        # Grid is grid_size x grid_size with the nest marked.
        self.assertEqual(len(obs["grid"]), 8)
        self.assertTrue(all(len(row) == 8 for row in obs["grid"]))
        nest_r, nest_c = obs["nest_position"]
        self.assertEqual(obs["grid"][nest_r][nest_c], "N")
        # Both ants start at the nest, carrying nothing.
        self.assertEqual(obs["ant_positions"], [[nest_r, nest_c], [nest_r, nest_c]])
        self.assertEqual(obs["carrying_food"], [False, False])
        # Food cells exist on the grid.
        self.assertEqual(len(obs["food_positions"]), 3)
        for fr, fc in obs["food_positions"]:
            self.assertEqual(obs["grid"][fr][fc], "F")
        # Pheromone grids are initially zero everywhere.
        for grid_name in ("pheromone_to_food", "pheromone_to_nest"):
            grid = obs[grid_name]
            self.assertEqual(len(grid), 8)
            self.assertTrue(all(v == 0.0 for row in grid for v in row))

    def test_python_ant_foraging_manual_playthrough(self):
        env = make(
            "open_spiel_python_ant_foraging",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        # Sequential game: ants alternate. Move ant 0 first, then ant 1.
        env.step([{"submission": 0}, {"submission": -1}])  # Ant 0 stays.
        obs = json.loads(env.state[1]["observation"]["observationString"])
        self.assertEqual(obs["current_player"], 1)
        self.assertEqual(obs["turn"], 0)  # Turn only advances after both ants move.
        env.step([{"submission": -1}, {"submission": 0}])  # Ant 1 stays.
        obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs["current_player"], 0)
        self.assertEqual(obs["turn"], 1)  # Full round completed.

    def test_python_ant_foraging_invalid_action(self):
        env = make("open_spiel_python_ant_foraging", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        env.step([{"submission": 999}, {"submission": -1}])  # Invalid action.
        self.assertTrue(env.done)
        playthrough = env.toJSON()
        self.assertEqual(
            playthrough["rewards"][0],
            open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
        )
        self.assertEqual(
            playthrough["rewards"][1],
            -open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
        )


if __name__ == "__main__":
    absltest.main()
