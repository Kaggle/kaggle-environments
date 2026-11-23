"""Tests for exact strategy evaluation in Kuhn and Leduc poker."""

import numpy as np
from absl.testing import absltest

from kaggle_environments.envs.open_spiel_env.eval import sequence_form

_PASS, _BET = 0, 1


def _kuhn_equilibrium(game, alpha):
    """The Kuhn equilibrium family: player 0 bluffs the Jack with probability alpha in [0, 1/3]."""
    p0, p1 = game.players
    x0, x1 = np.zeros(p0.shape), np.zeros(p1.shape)

    def put(strategy, treeplex, key, bet_probability):
        row = treeplex.index[key]
        strategy[row, _BET] = bet_probability
        strategy[row, _PASS] = 1.0 - bet_probability

    put(x0, p0, (0, ""), alpha)
    put(x0, p0, (1, ""), 0.0)
    put(x0, p0, (2, ""), 3.0 * alpha)
    put(x0, p0, (0, "pb"), 0.0)
    put(x0, p0, (1, "pb"), alpha + 1.0 / 3.0)
    put(x0, p0, (2, "pb"), 1.0)
    put(x1, p1, (0, "p"), 1.0 / 3.0)
    put(x1, p1, (1, "p"), 0.0)
    put(x1, p1, (2, "p"), 1.0)
    put(x1, p1, (0, "b"), 0.0)
    put(x1, p1, (1, "b"), 1.0 / 3.0)
    put(x1, p1, (2, "b"), 1.0)
    return x0, x1


class KuhnPokerTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.game = sequence_form.kuhn_poker()

    def test_structure(self):
        for treeplex in self.game.players:
            self.assertEqual(treeplex.n_info, 6)
            self.assertEqual(treeplex.n_seq, 13)
            self.assertTrue(treeplex.mask.all())

    def test_uniform_strategy_has_nash_conv_eleven_twelfths(self):
        x0, x1 = self.game.uniform()
        self.assertAlmostEqual(self.game.nash_conv(x0, x1), 11.0 / 12.0, places=12)
        self.assertAlmostEqual(self.game.exploitability(x0, x1), 11.0 / 24.0, places=12)

    def test_known_equilibria_have_value_minus_one_eighteenth(self):
        for alpha in (0.0, 0.1, 1.0 / 3.0):
            x0, x1 = _kuhn_equilibrium(self.game, alpha)
            self.assertAlmostEqual(self.game.value(x0, x1), -1.0 / 18.0, places=12)
            self.assertAlmostEqual(self.game.nash_conv(x0, x1), 0.0, places=12)

    def test_always_pass_is_exploited_for_the_ante_by_always_betting(self):
        # A player who never bets and never calls loses the ante to any bet.
        always_pass = np.zeros(self.game.players[0].shape)
        always_pass[:, _PASS] = 1.0
        value, _ = self.game.best_response(1, always_pass)
        self.assertAlmostEqual(value, 1.0, places=12)

    def test_best_response_strategy_achieves_the_best_response_value(self):
        x0, x1 = self.game.uniform()
        value_0, response_0 = self.game.best_response(0, x1)
        self.assertAlmostEqual(self.game.value(response_0, x1), value_0, places=12)
        value_1, response_1 = self.game.best_response(1, x0)
        self.assertAlmostEqual(-self.game.value(x0, response_1), value_1, places=12)

    def test_solve_converges_to_the_game_value(self):
        x0, x1 = sequence_form.solve(self.game, 2000)
        self.assertLess(self.game.nash_conv(x0, x1), 1e-4)
        self.assertAlmostEqual(self.game.value(x0, x1), -1.0 / 18.0, places=4)


class LeducPokerTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.game = sequence_form.leduc_poker()
        cls.game_suits = sequence_form.leduc_poker(suits=True)

    def test_information_set_counts(self):
        self.assertEqual([tp.n_info for tp in self.game.players], [144, 144])
        self.assertEqual([tp.n_info for tp in self.game_suits.players], [468, 468])

    def test_action_sets(self):
        for treeplex in self.game.players:
            for key, actions in zip(treeplex.keys, treeplex.actions):
                betting = key[3]
                facing_raise = betting.endswith("r")
                self.assertEqual("f" in actions, facing_raise, msg=str(key))
                self.assertIn("c", actions)
                self.assertEqual("r" in actions, betting.count("r") < 2, msg=str(key))

    def test_uniform_strategy_nash_conv_matches_the_published_value(self):
        for game in (self.game, self.game_suits):
            x0, x1 = game.uniform()
            self.assertAlmostEqual(game.nash_conv(x0, x1), 4.747222222222222, places=10)

    def test_rank_and_card_keying_give_the_same_uniform_value(self):
        self.assertAlmostEqual(
            self.game.value(*self.game.uniform()),
            self.game_suits.value(*self.game_suits.uniform()),
            places=12,
        )

    def test_plan_round_trip(self):
        rng = np.random.default_rng(0)
        for treeplex in self.game.players:
            strategy = rng.random(treeplex.shape) * treeplex.mask
            strategy /= strategy.sum(axis=1, keepdims=True)
            plan = treeplex.plan(strategy)
            np.testing.assert_allclose(treeplex.plan(treeplex.behavioural(plan)), plan, atol=1e-12)

    def test_best_response_is_no_worse_than_any_random_strategy(self):
        rng = np.random.default_rng(1)
        x1 = self.game.players[1].uniform()
        best, response = self.game.best_response(0, x1)
        self.assertAlmostEqual(self.game.value(response, x1), best, places=10)
        treeplex = self.game.players[0]
        for _ in range(20):
            strategy = rng.random(treeplex.shape) * treeplex.mask
            strategy /= strategy.sum(axis=1, keepdims=True)
            self.assertLessEqual(self.game.value(strategy, x1), best + 1e-10)

    def test_solve_approaches_the_game_value(self):
        x0, x1 = sequence_form.solve(self.game, 3000)
        nash_conv = self.game.nash_conv(x0, x1)
        self.assertLess(nash_conv, 2e-3)
        # The value of any profile is within NashConv of the game value.
        self.assertAlmostEqual(self.game.value(x0, x1), -0.0856, delta=nash_conv + 1e-4)


if __name__ == "__main__":
    absltest.main()
