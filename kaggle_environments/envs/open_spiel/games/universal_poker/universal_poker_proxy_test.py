"""Test for proxied Universal Poker game."""

import json
import random

from absl.testing import absltest
from absl.testing import parameterized
import pyspiel
from . import universal_poker_proxy as universal_poker


class UniversalPokerTest(parameterized.TestCase):

  def test_game_is_registered(self):
    game = pyspiel.load_game('universal_poker_proxy')
    self.assertIsInstance(game, universal_poker.UniversalPokerGame)

  def test_game_parameters(self):
    game = pyspiel.load_game("universal_poker_proxy(betting=nolimit,numPlayers=2,stack=20000 20000,numRounds=4,blind=50 100,firstPlayer=2 1 1 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,bettingAbstraction=fullgame)")
    game_params = game.get_parameters()
    self.assertEqual(
        game_params,
        {
            'betting': 'nolimit',
            'bettingAbstraction': 'fullgame',
            'blind': '50 100',
            'boardCards': '',
            'firstPlayer': '2 1 1 1',
            'handReaches': '',
            'maxRaises': '',
            'numBoardCards': '0 3 1 1',
            'numHoleCards': 2,
            'numPlayers': 2,
            'numRanks': 13,
            'numRounds': 4,
            'numSuits': 4,
            'potSize': 0,
            'raiseSize': '100 100',
            'stack': '20000 20000',
        }
    )

  def test_random_sim(self):
    game = universal_poker.UniversalPokerGame()
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=False)


if __name__ == '__main__':
  absltest.main()
