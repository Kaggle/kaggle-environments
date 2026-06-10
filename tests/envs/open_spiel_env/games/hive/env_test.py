"""Env-level tests for open_spiel_hive."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env
from kaggle_environments.envs.open_spiel_env.games.hive.hive_proxy import (
    _compute_tile_positions,
)


class HiveEnvTest(absltest.TestCase):
    def test_hive_agent_playthrough(self):
        env = make(
            "open_spiel_hive",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_hive")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))
        rewards = playthrough["rewards"]
        self.assertIn(sorted(rewards), ([-1.0, 1.0], [0.0, 0.0]))
        final_obs = json.loads(env.steps[-1][0]["observation"]["observationString"])
        self.assertTrue(final_obs["is_terminal"])
        self.assertIn(final_obs["winner"], ("white", "black", "draw"))

    def test_hive_manual_playthrough(self):
        env = make("open_spiel_hive", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # Action 209 places white's first Ant (wA1) at the origin.
        env.step([{"submission": 209}, {"submission": -1}])
        obs_after_w = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs_after_w["status"], "InProgress")
        self.assertEqual(obs_after_w["current_player"], "black")
        self.assertEqual(obs_after_w["move_number"], 1)
        self.assertEqual(obs_after_w["last_move"], "wA1")
        self.assertEqual(obs_after_w["pieces"], {"wA1": [0, 0, 0]})
        self.assertEqual(
            obs_after_w["expansions"],
            {"mosquito": True, "ladybug": True, "pillbug": True},
        )
        # Action 2947 encodes "bA1 wA1/" (Black's first Ant placed NE of wA1).
        env.step([{"submission": -1}, {"submission": 2947}])
        obs_after_b = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs_after_b["move_number"], 2)
        self.assertEqual(obs_after_b["last_move"], "bA1 wA1/")
        self.assertEqual(obs_after_b["current_player"], "white")
        self.assertEqual(obs_after_b["pieces"]["wA1"], [0, 0, 0])
        # "bA1 wA1/" -> bA1 is NE of wA1, offset (+1, -1).
        self.assertEqual(obs_after_b["pieces"]["bA1"], [1, -1, 0])

    def test_hive_invalid_action(self):
        env = make("open_spiel_hive", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # Action 0 is not a legal placement on the empty board (only the 13
        # placements of white's bug types are legal on move 1).
        env.step([{"submission": 0}, {"submission": -1}])
        self.assertTrue(env.done)
        playthrough = env.toJSON()
        self.assertEqual(
            playthrough["rewards"],
            [
                open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
                -open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
            ],
        )


class ComputeTilePositionsTest(absltest.TestCase):
    """Unit tests for the move-replay position tracker.

    This is the most fragile part of the proxy because it has to deal with
    beetles/mosquitoes/ladybugs stacking and then moving off, while keeping
    the underlying tiles' positions intact.
    """

    def test_first_move_lands_at_origin(self):
        self.assertEqual(_compute_tile_positions(["wA1"]), {"wA1": [0, 0, 0]})

    def test_six_cardinal_directions(self):
        # Place wA1 at origin, then add six adjacent tiles using each of the
        # six UHP direction encodings around wA1.
        positions = _compute_tile_positions(
            [
                "wA1",
                "bA1 wA1/",  # NE
                "bA2 wA1-",  # E
                "bA3 wA1\\",  # SE
                "bG1 /wA1",  # SW
                "bG2 -wA1",  # W
                "bG3 \\wA1",  # NW
            ]
        )
        self.assertEqual(positions["wA1"], [0, 0, 0])
        self.assertEqual(positions["bA1"], [1, -1, 0])
        self.assertEqual(positions["bA2"], [1, 0, 0])
        self.assertEqual(positions["bA3"], [0, 1, 0])
        self.assertEqual(positions["bG1"], [-1, 1, 0])
        self.assertEqual(positions["bG2"], [-1, 0, 0])
        self.assertEqual(positions["bG3"], [0, -1, 0])

    def test_beetle_climbs_onto_tile(self):
        # ``wB1 wA1`` (no direction symbol) means wB1 climbs on top of wA1.
        positions = _compute_tile_positions(["wA1", "wB1 wA1"])
        self.assertEqual(positions["wA1"], [0, 0, 0])
        self.assertEqual(positions["wB1"], [0, 0, 1])

    def test_beetle_climbs_onto_stack(self):
        # Two beetles stack on the same tile: wB1 climbs onto wA1 (h=1), then
        # wB2 climbs onto wB1 (which is itself at h=1) -> wB2 lands at h=2.
        positions = _compute_tile_positions(["wA1", "wB1 wA1", "wB2 wB1"])
        self.assertEqual(positions["wA1"], [0, 0, 0])
        self.assertEqual(positions["wB1"], [0, 0, 1])
        self.assertEqual(positions["wB2"], [0, 0, 2])

    def test_beetle_moves_off_stack_to_new_ground_hex(self):
        # wA1 at origin, bA1 east of it, wB1 climbs onto wA1, then wB1 moves
        # east of bA1 -- a brand new ground hex. The beetle's stack-height
        # should reset to 0, and wA1 should still be at its ground position.
        positions = _compute_tile_positions(
            [
                "wA1",  # (0, 0, 0)
                "bA1 wA1-",  # (1, 0, 0)
                "wB1 wA1",  # climb -> (0, 0, 1)
                "wB1 bA1-",  # move east of bA1 -> (2, 0, 0), ground level again
            ]
        )
        self.assertEqual(positions["wA1"], [0, 0, 0])
        self.assertEqual(positions["bA1"], [1, 0, 0])
        self.assertEqual(positions["wB1"], [2, 0, 0])

    def test_beetle_moves_off_stack_onto_another_stack(self):
        # Lateral climb: wB1 is on top of wA1 (h=1), bB1 sits on a new tile,
        # then wB1 moves directly onto bB1's hex (climbing it). The new h
        # should be (existing max h at that coord) + 1 = 1.
        positions = _compute_tile_positions(
            [
                "wA1",  # (0, 0, 0)
                "bA1 wA1-",  # (1, 0, 0)
                "wB1 wA1",  # climb wA1 -> (0, 0, 1)
                "bB1 bA1-",  # (2, 0, 0)
                "wB1 bB1",  # wB1 climbs onto bB1 -> (2, 0, 1)
            ]
        )
        self.assertEqual(positions["wB1"], [2, 0, 1])
        self.assertEqual(positions["bB1"], [2, 0, 0])
        # The previously-covered wA1 should still be at its original location.
        self.assertEqual(positions["wA1"], [0, 0, 0])

    def test_pass_is_skipped(self):
        # ``pass`` should not affect any tile positions.
        positions = _compute_tile_positions(["wA1", "pass", "bA1 wA1-"])
        self.assertEqual(positions, {"wA1": [0, 0, 0], "bA1": [1, 0, 0]})


if __name__ == "__main__":
    absltest.main()
