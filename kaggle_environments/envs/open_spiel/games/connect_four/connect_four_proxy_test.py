"""Test for proxied Connect Four game."""

import json

from absl.testing import absltest
from absl.testing import parameterized
import pyspiel
from . import connect_four_proxy as connect_four


NUM_ROWS = 6
NUM_COLS = 7


class ConnectFourTest(parameterized.TestCase):

  def test_game_is_registered(self):
    game = pyspiel.load_game('connect_four_proxy')
    self.assertIsInstance(game, connect_four.ConnectFourGame)

  def test_random_sim(self):
    game = connect_four.ConnectFourGame()
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=False)

  def test_state_to_json(self):
    game = connect_four.ConnectFourGame()
    state = game.new_initial_state()
    json_state = json.loads(state.to_json())
    expected_board = [['.'] * NUM_COLS for _ in range(NUM_ROWS)]
    self.assertEqual(json_state['board'], expected_board)
    self.assertEqual(json_state['current_player'], 'x')
    state.apply_action(3)
    json_state = json.loads(state.to_json())
    expected_board[0][3] = 'x'
    self.assertEqual(json_state['board'], expected_board)
    self.assertEqual(json_state['current_player'], 'o')
    state.apply_action(2)
    json_state = json.loads(state.to_json())
    expected_board[0][2] = 'o'
    self.assertEqual(json_state['board'], expected_board)
    self.assertEqual(json_state['current_player'], 'x')
    state.apply_action(2)
    json_state = json.loads(state.to_json())
    expected_board[1][2] = 'x'
    self.assertEqual(json_state['board'], expected_board)
    self.assertEqual(json_state['current_player'], 'o')

  def test_action_to_json(self):
    game = connect_four.ConnectFourGame()
    state = game.new_initial_state()
    action = json.loads(state.action_to_json(3))
    self.assertEqual(json.loads(state.action_to_json(3)), action)
    self.assertEqual(action['col'], 3)


if __name__ == '__main__':
  absltest.main()
