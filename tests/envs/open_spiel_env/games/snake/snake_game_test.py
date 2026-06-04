"""Tests for Snake game."""

from absl.testing import absltest

from kaggle_environments.envs.open_spiel_env.games.snake import snake_game as snake

import pyspiel


class SnakeGameTest(absltest.TestCase):

  def test_game_creation(self):
    game = pyspiel.load_game("snake")
    self.assertIsNotNone(game)
    state = game.new_initial_state()
    self.assertIsNotNone(state)
    self.assertFalse(state.is_terminal())
    # Default 2 players
    self.assertEqual(state.num_players, 2)

  def test_movement(self):
    game = pyspiel.load_game(
        "snake", {"rows": 10, "columns": 10, "players": 1}
    )
    state = game.new_initial_state()
    # Initial snake is at center.
    head_pos = state.snakes[0][0]

    # Move UP (0)
    state.apply_action(0)
    new_head_pos = state.snakes[0][0]
    self.assertEqual(new_head_pos[0], head_pos[0] - 1)
    self.assertEqual(new_head_pos[1], head_pos[1])

  def test_wall_collision(self):
    game = pyspiel.load_game(
        "snake", {"rows": 3, "columns": 3, "players": 1}
    )
    state = game.new_initial_state()
    # Clear foods so the test only measures wall collision (random food may
    # otherwise land on the snake's path and grant a point).
    state.foods = []
    # Snake at 1, 1.
    # Move UP to 0, 1
    state.apply_action(0)
    # Move UP to -1, 1 (collision)
    state.apply_action(0)
    self.assertTrue(state.is_terminal())
    # Check returns (0 score)
    self.assertEqual(state.returns()[0], 0)

  def test_eating_food(self):
    game = pyspiel.load_game(
        "snake", {"rows": 10, "columns": 10, "players": 1}
    )
    state = game.new_initial_state()

    # Force food pair to known cells (one reachable, one far away).
    head_r, head_c = state.snakes[0][0]
    food_r, food_c = head_r - 1, head_c
    partner = (state.rows - 1 - food_r, state.cols - 1 - food_c)
    state.foods = [(food_r, food_c), partner]

    initial_len = len(state.snakes[0])
    # Move UP to eat
    state.apply_action(0)

    self.assertLen(state.snakes[0], initial_len + 1)
    self.assertEqual(state.returns()[0], 1.0)
    # Eaten food cell should be gone.
    self.assertNotIn((food_r, food_c), state.foods)

  def test_food_pair_symmetric(self):
    game = pyspiel.load_game(
        "snake", {"rows": 10, "columns": 10, "players": 2}
    )
    state = game.new_initial_state()

    self.assertLen(state.foods, 2)
    (r0, c0), (r1, c1) = state.foods
    # 180° rotation about board center.
    self.assertEqual(r0 + r1, state.rows - 1)
    self.assertEqual(c0 + c1, state.cols - 1)

    # Distance from each snake start to its nearest food is equal.
    def manhattan(a, b):
      return abs(a[0] - b[0]) + abs(a[1] - b[1])

    p0_start = state.snakes[0][0]
    p1_start = state.snakes[1][0]
    d0 = min(manhattan(p0_start, f) for f in state.foods)
    d1 = min(manhattan(p1_start, f) for f in state.foods)
    self.assertEqual(d0, d1)

  def test_food_respawn_timer(self):
    game = pyspiel.load_game(
        "snake",
        {"rows": 10, "columns": 10, "players": 2, "food_respawn_interval": 3},
    )
    state = game.new_initial_state()

    # Pin foods to cells far from either snake so neither can eat them.
    state.foods = [(4, 0), (5, 9)]

    # Step 3 turns of safe moves: P0 starts at (1,1) heads DOWN; P1 starts
    # at (8,8) heads UP — both stay well clear of the walls and each other.
    for _ in range(3):
      state.apply_action(1)  # P0 DOWN
      state.apply_action(0)  # P1 UP

    # Old pinned cells must be gone (replaced by a fresh symmetric pair).
    self.assertNotIn((4, 0), state.foods)
    self.assertNotIn((5, 9), state.foods)
    self.assertLen(state.foods, 2)
    (r0, c0), (r1, c1) = state.foods
    self.assertEqual(r0 + r1, state.rows - 1)
    self.assertEqual(c0 + c1, state.cols - 1)

  def test_uneaten_food_removed_on_respawn(self):
    game = pyspiel.load_game(
        "snake",
        {"rows": 10, "columns": 10, "players": 2, "food_respawn_interval": 2},
    )
    state = game.new_initial_state()
    state.foods = [(0, 0), (9, 9)]  # Out of reach in 2 turns.

    for _ in range(2):
      state.apply_action(1)  # P0 DOWN
      state.apply_action(0)  # P1 UP

    self.assertNotIn((0, 0), state.foods)
    self.assertNotIn((9, 9), state.foods)

  def test_simultaneous_movement(self):
    game = pyspiel.load_game(
        "snake", {"rows": 5, "columns": 5, "players": 2}
    )
    state = game.new_initial_state()

    # 2 players.
    p0_head = state.snakes[0][0]
    p1_head = state.snakes[1][0]

    # P0 moves DOWN (1)
    state.apply_action(1)

    # State shouldn't change yet (buffered)
    self.assertEqual(state.snakes[0][0], p0_head)

    # P1 moves UP (0)
    state.apply_action(0)

    # Now state update
    self.assertEqual(state.snakes[0][0], (p0_head[0] + 1, p0_head[1]))
    self.assertEqual(state.snakes[1][0], (p1_head[0] - 1, p1_head[1]))

  def test_collision_other_snake(self):
    # Set up a small board where they collide
    game = pyspiel.load_game(
        "snake", {"rows": 4, "columns": 4, "players": 2}
    )
    state = game.new_initial_state()

    # Force positions for collision test
    # P0 head at (1,1)
    # P1 head at (1,3)
    state.snakes[0] = [(1, 1)]
    state.snakes[1] = [(1, 3)]

    # P0 moves RIGHT (3) -> (1,2)
    # P1 moves LEFT (2) -> (1,2)
    # Head to Head collision! Both should die.

    state.apply_action(3)  # P0 RIGHT
    state.apply_action(2)  # P1 LEFT

    # Check survival
    self.assertFalse(state.is_alive[0])
    self.assertFalse(state.is_alive[1])
    self.assertTrue(state.is_terminal())

  def test_one_survivor_wins(self):
    game = pyspiel.load_game(
        "snake", {"rows": 4, "columns": 4, "players": 2}
    )
    state = game.new_initial_state()

    # Force positions:
    # P0 at (1,1), P1 at (3,3)
    state.snakes[0] = [(1, 1)]
    state.snakes[1] = [(3, 3)]

    # P0 hits wall: UP -> (0,1) -> UP -> (-1,1)
    state.apply_action(0)  # P0 UP
    state.apply_action(0)  # P1 UP (safe)

    # Next turn
    state.apply_action(0)  # P0 UP (crash)
    state.apply_action(0)  # P1 UP (safe)

    self.assertFalse(state.is_alive[0])
    self.assertTrue(state.is_alive[1])
    self.assertTrue(state.is_terminal())


if __name__ == "__main__":
  absltest.main()