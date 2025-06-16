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

  def test_tic_tac_toe_playthrough(self):
    envs = open_spiel_env._register_game_envs(["tic_tac_toe"])
    print(envs)
    env = make("open_spiel_tic_tac_toe", debug=True)
    env.run(["random", "random"])
    json = env.toJSON()
    self.assertEqual(json["name"], "open_spiel_tic_tac_toe")
    self.assertTrue(all([status == "DONE" for status in json["statuses"]]))


if __name__ == '__main__':
  absltest.main()
