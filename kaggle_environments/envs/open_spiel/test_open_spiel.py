from absl.testing import absltest
import sys
from kaggle_environments import make
import pyspiel
from . import open_spiel as open_spiel_env


# Expected that not all pyspiel registered games can be registered as Kaggle
# envs (e.g. does not yet support simultaneous move games), but should register
# at least this many
_REGISTERED_GAMES_THRESHOLD = 50


class OpenSpielEnvTest(absltest.TestCase):

  def test_envs_load(self):
    envs = open_spiel_env._register_game_envs(
        [game_type.short_name for game_type in pyspiel.registered_games()]
    )
    self.assertTrue(len(envs) > _REGISTERED_GAMES_THRESHOLD)

  def test_tic_tac_toe_agent_playthrough(self):
    envs = open_spiel_env._register_game_envs(["tic_tac_toe"])
    env = make("open_spiel_tic_tac_toe", debug=True)
    env.run(["random", "random"])
    json = env.toJSON()
    self.assertEqual(json["name"], "open_spiel_tic_tac_toe")
    self.assertTrue(all([status == "DONE" for status in json["statuses"]]))

  def test_tic_tac_toe_manual_playthrough(self):
    envs = open_spiel_env._register_game_envs(["tic_tac_toe"])
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
    envs = open_spiel_env._register_game_envs(["tic_tac_toe"])
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
    json = env.toJSON()
    self.assertTrue(all([status == "DONE" for status in json["statuses"]]))
    self.assertEqual(
        json["rewards"],
        [
            open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
            -open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
        ]
    )

  def test_agent_error(self):
    envs = open_spiel_env._register_game_envs(["tic_tac_toe"])
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
    json = env.toJSON()
    self.assertEqual(json["rewards"], [None, None])
    self.assertEqual(json["statuses"], ["ERROR", "ERROR"])


if __name__ == '__main__':
  absltest.main()