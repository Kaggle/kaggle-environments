"""Tests for OpenSpiel environment."""

import json
import pathlib

from absl.testing import absltest
from kaggle_environments import make
import pyspiel
from . import open_spiel as open_spiel_env


# Expected that not all pyspiel registered games can be registered as Kaggle
# envs (e.g. does not yet support simultaneous move games), but should register
# at least this many
_REGISTERED_GAMES_THRESHOLD = 50


# These games may fail to register.
_GAME_BLOCKLIST = [
    "efg_game",
    "scotland_yard",
]


class OpenSpielEnvTest(absltest.TestCase):

  def test_envs_load(self):
    short_names = []
    for game_type in pyspiel.registered_games():
      if game_type.short_name in _GAME_BLOCKLIST:
        continue
      short_names.append(game_type.short_name)
    envs = open_spiel_env._register_game_envs(short_names)
    self.assertGreater(len(envs), _REGISTERED_GAMES_THRESHOLD)

  def test_tic_tac_toe_agent_playthrough(self):
    open_spiel_env._register_game_envs(["tic_tac_toe"])
    env = make("open_spiel_tic_tac_toe", debug=True)
    env.run(["random", "random"])
    json_playthrough = env.toJSON()
    self.assertEqual(json_playthrough["name"], "open_spiel_tic_tac_toe")
    self.assertTrue(
        all([status == "DONE" for status in json_playthrough["statuses"]])
    )

  def test_tic_tac_toe_manual_playthrough(self):
    open_spiel_env._register_game_envs(["tic_tac_toe"])
    env = make("open_spiel_tic_tac_toe", debug=True)
    env.reset()
    env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
    env.step([{"submission": 0}, {"submission": -1}])
    env.step([{"submission": -1}, {"submission": 1}])
    env.step([{"submission": 3}, {"submission": -1}])
    env.step([{"submission": -1}, {"submission": 4}])
    env.step([{"submission": 6}, {"submission": -1}])
    self.assertTrue(env.done)
    self.assertEqual(env.toJSON()["rewards"], [1, -1])

  def test_invalid_action(self):
    open_spiel_env._register_game_envs(["tic_tac_toe"])
    env = make("open_spiel_tic_tac_toe", debug=True)
    env.reset()
    for i in range(5):  # Try repeatedly applying an illegal action
      env.step([
          {"submission": pyspiel.INVALID_ACTION},
          {"submission": pyspiel.INVALID_ACTION},
      ])
      if env.done:
        break
    self.assertEqual(i, 1)  # Zeroth step is setup step, should fail next step.
    json_playthrough = env.toJSON()
    self.assertTrue(
        all([status == "DONE" for status in json_playthrough["statuses"]])
    )
    self.assertEqual(
        json_playthrough["rewards"],
        [
            open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
            -open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
        ]
    )

  def test_serialized_game_and_state(self):
    open_spiel_env._register_game_envs(["tic_tac_toe"])
    env = make("open_spiel_tic_tac_toe", debug=True)
    env.reset()
    env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
    kaggle_state = env.step([{"submission": 0}, {"submission": -1}])
    serialize_game_and_state = kaggle_state[1]["observation"][
        "serializedGameAndState"
    ]
    game, state = pyspiel.deserialize_game_and_state(serialize_game_and_state)
    self.assertEqual(game.get_type().short_name, "tic_tac_toe_proxy")
    self.assertEqual(state.history(), [0])

  def test_agent_error(self):
    open_spiel_env._register_game_envs(["tic_tac_toe"])
    env = make("open_spiel_tic_tac_toe", debug=True)
    env.reset()
    # Setup step
    env.step([
        {"submission": pyspiel.INVALID_ACTION},
        {"submission": pyspiel.INVALID_ACTION},
    ])
    env.step([
        {"submission": open_spiel_env.AGENT_ERROR_ACTION},
        {"submission": pyspiel.INVALID_ACTION},
    ])
    self.assertTrue(env.done)
    json_playthrough = env.toJSON()
    self.assertEqual(json_playthrough["rewards"], [None, None])
    self.assertEqual(json_playthrough["statuses"], ["ERROR", "ERROR"])

  def test_initial_actions(self):
    open_spiel_env._register_game_envs(["tic_tac_toe"])
    env = make(
        "open_spiel_tic_tac_toe",
        {"initialActions": [0, 1, 3, 4]},
        debug=True,
    )
    env.reset()
    # Setup step
    env.step([
        {"submission": pyspiel.INVALID_ACTION},
        {"submission": pyspiel.INVALID_ACTION},
    ])
    env.step([
        {"submission": 2},
        {"submission": pyspiel.INVALID_ACTION},
    ])
    env.step([
        {"submission": pyspiel.INVALID_ACTION},
        {"submission": 7},
    ])
    self.assertTrue(env.done)
    json_playthrough = env.toJSON()
    self.assertEqual(json_playthrough["rewards"], [-1, 1])

  def test_chess_openings(self):
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
        env.step([
            {"submission": pyspiel.INVALID_ACTION},
            {"submission": pyspiel.INVALID_ACTION},
        ])
        obs = env.state[0]["observation"]
        _, state = pyspiel.deserialize_game_and_state(
            obs["serializedGameAndState"]
        )
        self.assertEqual(str(state), opening["fen"])
        self.assertEqual(str(state),
                         env.toJSON()["configuration"]["metadata"]["fen"])


if __name__ == "__main__":
  absltest.main()
