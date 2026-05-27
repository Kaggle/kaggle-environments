"""Env-level tests for open_spiel_amazons."""

import json
import random

import pyspiel
from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class AmazonsEnvTest(absltest.TestCase):
    def test_amazons_agent_playthrough(self):
        env = make("open_spiel_amazons", debug=True)
        env.run(["random", "random"])
        json_data = env.toJSON()
        self.assertEqual(json_data["name"], "open_spiel_amazons")
        self.assertTrue(all(status == "DONE" for status in json_data["statuses"]))

    def test_amazons_manual_playthrough(self):
        # Walks player 0 (X) through one full Amazons turn (from -> to -> shoot)
        # by picking the first legal action at each sub-action, and verifies
        # the proxy advances `phase` through all three values and then resets
        # to "from" for player O. Picking from `legalActions` keeps this test
        # robust to pyspiel version differences in the starting layout.
        env = make("open_spiel_amazons", {"includeLegalActions": True}, debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        for expected_next_phase in ("to", "shoot", "from"):
            legal = env.state[0]["observation"]["legalActions"]
            env.step([{"submission": legal[0]}, {"submission": -1}])
            obs = json.loads(env.state[0]["observation"]["observationString"])
            if expected_next_phase == "from":
                # After three sub-actions it is player 1's (O) turn.
                obs = json.loads(env.state[1]["observation"]["observationString"])
                self.assertEqual(env.state[0]["status"], "INACTIVE")
                self.assertEqual(env.state[1]["status"], "ACTIVE")
                self.assertEqual(obs["current_player"], "o")
            self.assertEqual(obs["phase"], expected_next_phase)
            self.assertFalse(obs["is_terminal"])
        # An arrow (burned square) was fired somewhere on the board.
        self.assertTrue(any("#" in row for row in obs["board"]))

    def test_amazons_terminal_state(self):
        # Drive an Amazons game to natural termination using only legal random
        # actions, then verify the proxy reports a winner and clears the phase.
        rng = random.Random(0)
        game = pyspiel.load_game("amazons_proxy")
        state = game.new_initial_state()
        while not state.is_terminal():
            state.apply_action(rng.choice(state.legal_actions()))
        obs = json.loads(state.observation_string(0))
        self.assertTrue(obs["is_terminal"])
        self.assertIn(obs["winner"], ("x", "o"))
        self.assertIsNone(obs["phase"])

    def test_amazons_invalid_action(self):
        env = make("open_spiel_amazons", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        env.step([{"submission": 999}, {"submission": -1}])  # Invalid action.
        self.assertTrue(env.done)
        json_data = env.toJSON()
        self.assertEqual(json_data["rewards"][0], open_spiel_env.DEFAULT_INVALID_ACTION_REWARD)
        self.assertEqual(json_data["rewards"][1], -open_spiel_env.DEFAULT_INVALID_ACTION_REWARD)


if __name__ == "__main__":
    absltest.main()
