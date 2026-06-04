"""Env-level tests for open_spiel_dots_and_boxes."""

import json
import random

import pyspiel
from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class DotsAndBoxesEnvTest(absltest.TestCase):
    def test_dots_and_boxes_agent_playthrough(self):
        env = make("open_spiel_dots_and_boxes", debug=True)
        env.run(["random", "random"])
        json_data = env.toJSON()
        self.assertEqual(json_data["name"], "open_spiel_dots_and_boxes")
        self.assertTrue(all(status == "DONE" for status in json_data["statuses"]))

    def test_dots_and_boxes_manual_playthrough(self):
        # Walks a 2x2 board to a state where P1 closes both top-row boxes back
        # to back. Confirms the proxy honors the "extra turn" rule (P1 keeps
        # the move after closing a box), the boxes/scores grids match what we
        # would expect by hand, and the game eventually terminates with a
        # winner and all four boxes claimed.
        env = make(
            "open_spiel_dots_and_boxes",
            {
                "includeLegalActions": True,
                "openSpielGameParameters": {"num_rows": 2, "num_cols": 2},
            },
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.

        def submit(player: int, action: int) -> None:
            actions = [{"submission": -1}, {"submission": -1}]
            actions[player] = {"submission": action}
            env.step(actions)

        # Action layout for num_rows=num_cols=2: horizontal lines 0..5
        # (h(r,c) = r*2 + c) and vertical lines 6..11 (v(r,c) = 6 + r*3 + c).
        submit(0, 0)  # P1: h(0,0) — top of box (0,0)
        submit(1, 1)  # P2: h(0,1) — top of box (0,1)
        submit(0, 6)  # P1: v(0,0) — left of box (0,0)
        submit(1, 8)  # P2: v(0,2) — right of box (0,1)
        submit(0, 7)  # P1: v(0,1) — shared edge between (0,0) and (0,1)
        submit(1, 4)  # P2: h(2,0) — bottom row, no box closes
        submit(0, 2)  # P1: h(1,0) closes box (0,0) → score [1,0], P1 keeps turn
        submit(0, 3)  # P1: h(1,1) closes box (0,1) → score [2,0], P1 keeps turn

        obs0 = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs0["scores"], [2, 0])
        self.assertEqual(obs0["boxes"][0], [1, 1])
        self.assertEqual(obs0["current_player"], "1")
        self.assertFalse(obs0["is_terminal"])
        self.assertEqual(obs0["last_action"], {"orientation": "h", "row": 1, "col": 1, "player": "1"})

        # Drain the remaining lines with random legal play. The exact score
        # split depends on the RNG, but the game must finish with all four
        # boxes claimed.
        rng = random.Random(0)
        while not env.done:
            current = env.state[0]["observation"]["currentPlayer"]
            legal = env.state[current]["observation"]["legalActions"]
            if not legal:
                break
            submit(current, rng.choice(legal))
        obs_final = json.loads(env.state[0]["observation"]["observationString"])
        self.assertTrue(obs_final["is_terminal"])
        self.assertIn(obs_final["winner"], ("1", "2", "draw"))
        self.assertEqual(sum(obs_final["scores"]), 4)

    def test_dots_and_boxes_terminal_state(self):
        # Drive a fresh dots_and_boxes game to termination using only legal
        # random moves and verify the proxy reports a coherent final state.
        rng = random.Random(0)
        game = pyspiel.load_game("dots_and_boxes_proxy")
        state = game.new_initial_state()
        while not state.is_terminal():
            state.apply_action(rng.choice(state.legal_actions()))
        obs = json.loads(state.observation_string(0))
        self.assertTrue(obs["is_terminal"])
        self.assertEqual(obs["current_player"], "")
        self.assertIn(obs["winner"], ("1", "2", "draw"))
        self.assertEqual(sum(obs["scores"]), obs["num_rows"] * obs["num_cols"])

    def test_dots_and_boxes_invalid_action(self):
        env = make("open_spiel_dots_and_boxes", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        env.step([{"submission": 999}, {"submission": -1}])  # Invalid action.
        self.assertTrue(env.done)
        json_data = env.toJSON()
        self.assertEqual(json_data["rewards"][0], open_spiel_env.DEFAULT_INVALID_ACTION_REWARD)
        self.assertEqual(json_data["rewards"][1], -open_spiel_env.DEFAULT_INVALID_ACTION_REWARD)


if __name__ == "__main__":
    absltest.main()
