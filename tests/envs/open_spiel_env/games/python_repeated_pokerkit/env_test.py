"""Env-level tests for open_spiel_python_repeated_pokerkit."""

import pokerkit  # noqa: F401
from absl.testing import absltest
from open_spiel.python.games import pokerkit_wrapper  # noqa: F401

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env


class PythonRepeatedPokerkitEnvTest(absltest.TestCase):
    def test_repeated_pokerkit(self):
        pokerkit_game_str = (
            "python_repeated_pokerkit("
            "bet_size_schedule=,"
            "blind_schedule=,"
            "bring_in_schedule=,"
            "first_button_player=-1,"
            "max_num_hands=20,"
            "pokerkit_game_params=python_pokerkit_wrapper("
            "blinds=1 2,"
            "num_players=2,"
            "stack_sizes=200 200,"
            "variant=NoLimitTexasHoldem),"
            "reset_stacks=True,"
            "rotate_dealer=True)"
        )
        envs = open_spiel_env._register_game_envs([pokerkit_game_str])
        env = make(envs["open_spiel_python_repeated_pokerkit"])
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        for i in range(20):
            if i % 2 == 0:
                env.step([{"submission": -1}, {"submission": 0}])
            else:
                env.step([{"submission": 0}, {"submission": -1}])
        self.assertTrue(env.done)
        self.assertEqual(env.toJSON()["rewards"], [0.0, 0.0])

    def test_default_repeated_pokerkit_loads(self):
        env = make("open_spiel_python_repeated_pokerkit")
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        for i in range(100):
            if i % 2 == 0:
                env.step([{"submission": -1}, {"submission": 0}])
            else:
                env.step([{"submission": 0}, {"submission": -1}])
        self.assertTrue(env.done)
        self.assertEqual(env.toJSON()["rewards"], [0.0, 0.0])


if __name__ == "__main__":
    absltest.main()
