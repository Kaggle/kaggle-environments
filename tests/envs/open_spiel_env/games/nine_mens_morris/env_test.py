"""Env-level tests for open_spiel_nine_mens_morris."""

import json

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


# Movement/capture-action encoding: 24 + source*24 + dest.
def _move(src: int, dst: int) -> int:
    return 24 + src * 24 + dst


class NineMensMorrisEnvTest(absltest.TestCase):
    def test_nine_mens_morris_agent_playthrough(self):
        env = make(
            "open_spiel_nine_mens_morris",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_nine_mens_morris")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))
        # Nine Men's Morris is zero-sum: one side wins (+1/-1) or draws (0/0).
        rewards = playthrough["rewards"]
        self.assertIn(sorted(rewards), ([-1.0, 1.0], [0.0, 0.0]))

    def test_nine_mens_morris_manual_playthrough(self):
        """Force a mill on move 5 and verify the capture phase."""
        env = make(
            "open_spiel_nine_mens_morris",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.

        # Placement sequence: W builds the top-row mill (points 0, 1, 2);
        # B parks harmlessly on points 9 and 10.
        for player, action in [(0, 0), (1, 9), (0, 1), (1, 10), (0, 2)]:
            if player == 0:
                env.step([{"submission": action}, {"submission": -1}])
            else:
                env.step([{"submission": -1}, {"submission": action}])

        obs_after_mill = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs_after_mill["phase"], "capture")
        # Still W's turn: W picks which black piece to capture.
        self.assertEqual(obs_after_mill["current_player"], "W")
        self.assertEqual(obs_after_mill["board"][0], "W")
        self.assertEqual(obs_after_mill["board"][1], "W")
        self.assertEqual(obs_after_mill["board"][2], "W")
        # Capture labels must be disambiguated from placement labels.
        capture_labels = env.state[0]["observation"]["legalActionStrings"]
        self.assertTrue(all("Capture" in s for s in capture_labels))

        # W captures B's piece at point 9.
        env.step([{"submission": 9}, {"submission": -1}])
        obs_after_capture = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs_after_capture["phase"], "placement")
        self.assertEqual(obs_after_capture["num_men"], {"W": 9, "B": 8})
        self.assertEqual(obs_after_capture["current_player"], "B")
        self.assertEqual(obs_after_capture["board"][9], ".")

    def test_nine_mens_morris_invalid_action(self):
        env = make("open_spiel_nine_mens_morris", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Setup step.
        # A movement-encoded action is illegal during placement.
        env.step([{"submission": _move(0, 1)}, {"submission": -1}])
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
