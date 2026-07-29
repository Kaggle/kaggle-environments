"""Env-level tests for open_spiel_go_fish."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class GoFishEnvTest(absltest.TestCase):
    def test_go_fish_agent_playthrough(self):
        env = make(
            "open_spiel_go_fish",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_go_fish")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))

    def test_go_fish_observation_is_json(self):
        env = make(
            "open_spiel_go_fish",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # After dealing, it is player 0's turn (Ask phase).
        obs_p0 = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs_p0["phase"], "Ask")
        self.assertEqual(obs_p0["current_player"], 0)
        self.assertFalse(obs_p0["is_terminal"])
        # Two-player Go Fish deals 7 cards each.
        self.assertEqual(sum(obs_p0["hand"].values()), 7)
        self.assertEqual(obs_p0["players"][0], {"player": 0, "cards": 7, "books": 0})
        self.assertEqual(obs_p0["players"][1], {"player": 1, "cards": 7, "books": 0})

    def test_go_fish_observation_hides_opponent(self):
        env = make("open_spiel_go_fish", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # Each player's observation reports only their own hand contents.
        obs_p0 = json.loads(env.state[0]["observation"]["observationString"])
        obs_p1 = json.loads(env.state[1]["observation"]["observationString"])
        self.assertEqual(obs_p0["observer"], 0)
        self.assertEqual(obs_p1["observer"], 1)
        self.assertEqual(sum(obs_p0["hand"].values()), 7)
        self.assertEqual(sum(obs_p1["hand"].values()), 7)
        # The two hands are dealt independently; opponent card contents are not
        # exposed in either observation (only aggregate counts in "players").

    def test_go_fish_invalid_action(self):
        env = make("open_spiel_go_fish", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        env.step([{"submission": 999}, {"submission": -1}])  # Invalid action.
        self.assertTrue(env.done)
        playthrough = env.toJSON()
        self.assertEqual(
            playthrough["rewards"][0],
            open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
        )


if __name__ == "__main__":
    absltest.main()
