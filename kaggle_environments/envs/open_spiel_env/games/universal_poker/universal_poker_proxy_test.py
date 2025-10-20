"""Test for proxied Universal Poker game."""

import pyspiel
from absl.testing import absltest, parameterized

from . import universal_poker_proxy as universal_poker


class UniversalPokerTest(parameterized.TestCase):
    def test_game_is_registered(self):
        game = pyspiel.load_game("universal_poker_proxy")
        self.assertIsInstance(game, universal_poker.UniversalPokerGame)

    def test_game_parameters(self):
        game = pyspiel.load_game(pyspiel.hunl_game_string("fullgame"))
        game_params = game.get_parameters()
        self.assertEqual(
            game_params,
            {
                "betting": "nolimit",
                "bettingAbstraction": "fullgame",
                "blind": "100 50",
                "boardCards": "",
                "firstPlayer": "2 1 1 1",
                "handReaches": "",
                "maxRaises": "",
                "numBoardCards": "0 3 1 1",
                "numHoleCards": 2,
                "numPlayers": 2,
                "numRanks": 13,
                "numRounds": 4,
                "numSuits": 4,
                "potSize": 0,
                "stack": "20000 20000",
                "calcOddsNumSims": 0,
            },
        )

    def test_random_sim(self):
        game = universal_poker.UniversalPokerGame()
        pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=False)


if __name__ == "__main__":
    absltest.main()
