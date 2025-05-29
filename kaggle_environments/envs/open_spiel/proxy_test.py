"""Proxy tests."""

from . import proxy
from absl.testing import absltest
from absl.testing import parameterized
import pyspiel


def make_game() -> proxy.Game:
  return proxy.Game(pyspiel.load_game('tic_tac_toe()'))


class TestState(proxy.State):

  def __str__(self) -> str:
    return 'TestState: ' + super().__str__()


class TestGame(proxy.Game):

  def new_initial_state(self, *args, **kwargs) -> TestState:
    return TestState(
        self.__wrapped__.new_initial_state(*args, **kwargs), game=self
    )


class ProxiesTest(parameterized.TestCase):

  def test_types(self):
    game = make_game()
    self.assertIsInstance(game, pyspiel.Game)
    state = game.new_initial_state()
    self.assertIsInstance(state, pyspiel.State)

  def test_get_game(self):
    game = make_game()
    state = game.new_initial_state()
    self.assertIsInstance(state.get_game(), proxy.Game)
    new_state = state.get_game().new_initial_state()
    self.assertIsInstance(new_state, proxy.State)
    self.assertIsNot(new_state, state)

  def test_clone(self):
    game = make_game()
    state = game.new_initial_state()
    state.apply_action(state.legal_actions()[0])
    clone = state.clone()
    self.assertIsInstance(clone, proxy.State)
    self.assertEqual(state.history(), clone.history())
    clone.apply_action(clone.legal_actions()[0])
    self.assertEqual(state.history(), clone.history()[:-1])

  def test_subclassing(self):
    game = TestGame(pyspiel.load_game('tic_tac_toe()'))
    state = game.new_initial_state()
    self.assertIsInstance(state, TestState)
    self.assertIsInstance(state.clone(), TestState)
    self.assertIsInstance(state.get_game(), TestGame)
    wrapped_state = state.__wrapped__  # type: ignore
    self.assertEqual(str(state), 'TestState: ' + str(wrapped_state))


if __name__ == '__main__':
  absltest.main()