"""Env-level tests for open_spiel_negotiation."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class NegotiationEnvTest(absltest.TestCase):
    def test_negotiation_agent_playthrough(self):
        env = make(
            "open_spiel_negotiation",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_negotiation")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))

    def test_negotiation_observation_is_json_and_hides_opponent_utilities(self):
        env = make("open_spiel_negotiation", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step (chance).
        obs_p0 = json.loads(env.state[0]["observation"]["observationString"])
        obs_p1 = json.loads(env.state[1]["observation"]["observationString"])
        self.assertEqual(obs_p0["current_player"], 0)
        self.assertEqual(obs_p0["turn_type"], "proposal")
        self.assertEqual(len(obs_p0["item_pool"]), 3)
        self.assertEqual(len(obs_p0["my_utilities"]), 3)
        self.assertEqual(obs_p0["viewing_player"], 0)
        self.assertEqual(obs_p1["viewing_player"], 1)
        # Each player only sees their own utility vector; the other's stays hidden.
        self.assertNotIn("opponent_utilities", obs_p0)
        self.assertEqual(obs_p0["params"]["accept_action"], 216)
        self.assertEqual(obs_p0["params"]["num_distinct_proposals"], 217)
        self.assertFalse(obs_p0["is_terminal"])
        self.assertEqual(obs_p0["proposals"], [])
        self.assertEqual(obs_p0["utterances"], [])

    def test_negotiation_acceptance_terminates_and_assigns_returns(self):
        env = make("open_spiel_negotiation", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step (chance).
        # Player 0 proposes [0, 0, 0] (action 0), then utters [0, 0, 0] (action 217).
        env.step([{"submission": 0}, {"submission": -1}])
        env.step([{"submission": 217}, {"submission": -1}])
        # Player 1 accepts the proposal (action = num_distinct_proposals - 1 = 216).
        env.step([{"submission": -1}, {"submission": 216}])
        self.assertTrue(env.done)
        final = json.loads(env.state[0]["observation"]["observationString"])
        self.assertTrue(final["agreement_reached"])
        self.assertTrue(final["is_terminal"])
        # The accepting player keeps every item in the pool, so the proposer
        # (who offered themselves nothing) ends with zero utility.
        rewards = env.toJSON()["rewards"]
        self.assertEqual(rewards[0], 0.0)
        self.assertGreater(rewards[1], 0.0)
        self.assertEqual(final["winner"], 1)
        # The accept action shows up in the proposals history with a flag.
        self.assertEqual(final["proposals"][-1], {"player": 1, "accept": True})

    def test_negotiation_invalid_action(self):
        env = make("open_spiel_negotiation", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        env.step([{"submission": 9999}, {"submission": -1}])  # Invalid action.
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
