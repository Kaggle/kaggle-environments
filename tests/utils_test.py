"""Tests for kaggle_environments.utils."""

from types import SimpleNamespace

from absl.testing import absltest

from kaggle_environments.utils import resolve_episode_seed


def _make_env(*, info=None, configuration=None):
    env = SimpleNamespace()
    if info is not None:
        env.info = info
    env.configuration = configuration if configuration is not None else SimpleNamespace()
    return env


class ResolveEpisodeSeedTest(absltest.TestCase):

    def test_initializes_info_when_missing(self):
        env = SimpleNamespace(configuration=SimpleNamespace(seed=7))
        seed = resolve_episode_seed(env)
        self.assertEqual(seed, 7)
        self.assertEqual(env.info, {"seed": 7})

    def test_initializes_info_when_none(self):
        env = SimpleNamespace(info=None, configuration=SimpleNamespace(seed=11))
        seed = resolve_episode_seed(env)
        self.assertEqual(seed, 11)
        self.assertEqual(env.info, {"seed": 11})

    def test_env_info_takes_precedence_over_configuration(self):
        env = _make_env(info={"seed": 42}, configuration=SimpleNamespace(seed=99))
        seed = resolve_episode_seed(env)
        self.assertEqual(seed, 42)
        # configuration is still scrubbed even when info won.
        self.assertIsNone(env.configuration.seed)

    def test_configuration_used_when_info_empty(self):
        env = _make_env(info={}, configuration=SimpleNamespace(seed=123))
        seed = resolve_episode_seed(env)
        self.assertEqual(seed, 123)
        self.assertIsNone(env.configuration.seed)
        self.assertEqual(env.info["seed"], 123)

    def test_fallback_used_when_no_seed_anywhere(self):
        env = _make_env(info={}, configuration=SimpleNamespace())
        seed = resolve_episode_seed(env, fallback=lambda: 555)
        self.assertEqual(seed, 555)
        self.assertEqual(env.info["seed"], 555)

    def test_default_fallback_is_random_31bit_int(self):
        env = _make_env(info={}, configuration=SimpleNamespace())
        seed = resolve_episode_seed(env)
        self.assertIsInstance(seed, int)
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 2**31)

    def test_custom_config_key(self):
        env = _make_env(info={}, configuration=SimpleNamespace(randomSeed=314))
        seed = resolve_episode_seed(env, config_key="randomSeed")
        self.assertEqual(seed, 314)
        self.assertIsNone(env.configuration.randomSeed)

    def test_dict_configuration(self):
        env = _make_env(info={}, configuration={"seed": 17})
        seed = resolve_episode_seed(env)
        self.assertEqual(seed, 17)
        self.assertIsNone(env.configuration["seed"])
        self.assertEqual(env.info["seed"], 17)

    def test_dict_configuration_with_custom_key(self):
        env = _make_env(info={}, configuration={"randomSeed": 9})
        seed = resolve_episode_seed(env, config_key="randomSeed")
        self.assertEqual(seed, 9)
        self.assertIsNone(env.configuration["randomSeed"])

    def test_dict_configuration_falls_back_when_key_missing(self):
        env = _make_env(info={}, configuration={})
        seed = resolve_episode_seed(env, fallback=lambda: 1234)
        self.assertEqual(seed, 1234)
        self.assertIsNone(env.configuration["seed"])
        self.assertEqual(env.info["seed"], 1234)

    def test_seed_value_of_zero_is_respected(self):
        # zero is a valid seed; the resolver must not treat it as missing.
        env = _make_env(info={}, configuration=SimpleNamespace(seed=0))
        seed = resolve_episode_seed(env, fallback=lambda: 999)
        self.assertEqual(seed, 0)
        self.assertEqual(env.info["seed"], 0)

    def test_idempotent_across_calls(self):
        # Second call (e.g. after env.reset() runs twice) reuses env.info.
        env = _make_env(info={}, configuration=SimpleNamespace(seed=8))
        first = resolve_episode_seed(env)
        # Pretend something repopulated the config; the helper should still
        # return the original resolved seed.
        env.configuration.seed = 99
        second = resolve_episode_seed(env)
        self.assertEqual(first, 8)
        self.assertEqual(second, 8)


if __name__ == "__main__":
    absltest.main()
