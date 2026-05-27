"""Env-level tests for open_spiel_chess."""

import json
import pathlib

import pyspiel
from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class ChessEnvTest(absltest.TestCase):
    def test_chess_openings_manually_configured(self):
        open_spiel_env._register_game_envs(["chess"])
        openings_path = pathlib.Path(
            open_spiel_env.GAMES_DIR,
            "chess/openings.jsonl",
        )
        self.assertTrue(openings_path.is_file())
        with open(openings_path, "r", encoding="utf-8") as f:
            for line in f:
                opening = json.loads(line)
                config = {
                    "initialActions": opening.pop("initialActions"),
                    "metadata": opening,
                }
                env = make(
                    "open_spiel_chess",
                    config,
                    debug=True,
                )
                env.reset()
                # Setup step
                env.step(
                    [
                        {"submission": pyspiel.INVALID_ACTION},
                        {"submission": pyspiel.INVALID_ACTION},
                    ]
                )
                obs = env.state[0]["observation"]
                _, state = pyspiel.deserialize_game_and_state(obs["serializedGameAndState"])
                self.assertEqual(str(state), opening["fen"])
                self.assertEqual(str(state), env.toJSON()["configuration"]["metadata"]["fen"])

    def test_chess_openings_configured_with_seed(self):
        open_spiel_env._register_game_envs(["chess"])
        config = {
            "useImage": True,
            "seed": 1,
        }
        env = make(
            "open_spiel_chess",
            config,
            debug=True,
        )
        env.reset()
        # Image config is loaded during setup step.
        self.assertFalse("imageConfig" in env.configuration)
        # Setup step
        env.step(
            [
                {"submission": pyspiel.INVALID_ACTION},
                {"submission": pyspiel.INVALID_ACTION},
            ]
        )
        self.assertTrue("imageConfig" in env.configuration)
        self.assertEqual(env.configuration["imageConfig"]["color"], "blue")
        self.assertEqual(env.configuration["imageConfig"]["pieceSet"], "cardinal")
        self.assertTrue("imageConfig" in env.state[0]["observation"])


if __name__ == "__main__":
    absltest.main()
