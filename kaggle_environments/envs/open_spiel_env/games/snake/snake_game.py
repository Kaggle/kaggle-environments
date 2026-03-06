"""Snake game implemented in Python for OpenSpiel."""

import enum
import random
from typing import List, Tuple

import numpy as np

import pyspiel

_NUM_PLAYERS = 2  # Default to 2 players for head-to-head
_MIN_PLAYERS = 1
_MAX_PLAYERS = 4
_NUM_ROWS = 10
_NUM_COLS = 10
_GAME_TYPE = pyspiel.GameType(
    short_name="snake",
    long_name="Snake",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=_MAX_PLAYERS,
    min_num_players=_MIN_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={
        "rows": _NUM_ROWS,
        "columns": _NUM_COLS,
        "players": _NUM_PLAYERS,
    },
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=4,
    max_chance_outcomes=0,
    num_players=_NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=100.0,
    utility_sum=0.0,
    max_game_length=_NUM_ROWS * _NUM_COLS * 2,
)


class Action(enum.IntEnum):
  UP = 0
  DOWN = 1
  LEFT = 2
  RIGHT = 3


class SnakeGame(pyspiel.Game):
  """A Python version of the Snake game."""

  def __init__(self, params=None):
    self.rows = params.get("rows", _NUM_ROWS) if params else _NUM_ROWS
    self.cols = params.get("columns", _NUM_COLS) if params else _NUM_COLS
    self.num_players = (
        params.get("players", _NUM_PLAYERS) if params else _NUM_PLAYERS
    )

    # Update game info with dynamic player count
    game_info = pyspiel.GameInfo(
        num_distinct_actions=4,
        max_chance_outcomes=0,
        num_players=self.num_players,
        min_utility=-1.0,
        max_utility=100.0,
        utility_sum=0.0,
        max_game_length=self.rows * self.cols * 2,
    )
    super().__init__(_GAME_TYPE, game_info, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return SnakeState(self)

  def make_py_observer(self, params=None):
    """Returns an object used for observing game state."""
    return SnakeObserver(params, self.rows, self.cols)


class SnakeState(pyspiel.State):
  """A python version of the Snake state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._game = game
    self.rows = game.rows
    self.cols = game.cols
    self.num_players = game.num_players
    self._is_terminal = False
    self._game_over_reason = None

    # Initialize snakes
    # Starting positions: corners/edges depending on player count
    # 2 players: (1,1) and (R-2, C-2)
    # 3 players: + (1, C-2)
    # 4 players: + (R-2, 1)

    start_positions = [
        (1, 1),
        (self.rows - 2, self.cols - 2),
        (1, self.cols - 2),
        (self.rows - 2, 1),
    ]

    self.snakes: List[List[Tuple[int, int]]] = []  # List of lists of coords
    self.scores = [0.0] * self.num_players
    self.is_alive = [True] * self.num_players

    for i in range(self.num_players):
      start_r, start_c = start_positions[i % len(start_positions)]
      self.snakes.append([(start_r, start_c)])

    self.food = None
    self._place_food()
    self._steps = 0
    self._next_player = 0
    self._move_buffer = [None] * self.num_players

  def _place_food(self):
    """Places food in a random empty location."""
    available_positions = []
    for r in range(self.rows):
      for c in range(self.cols):
        occupied = False
        for snake in self.snakes:
          if (r, c) in snake:
            occupied = True
            break
        if not occupied:
          available_positions.append((r, c))

    if not available_positions:
      # If board is full, game over.
      self._is_terminal = True
      self._game_over_reason = "Won (Board Full)"
      return

    self.food = random.choice(available_positions)

  def current_player(self):
    """Returns id of the next player to move."""
    if self._is_terminal:
      return pyspiel.PlayerId.TERMINAL
    return self._next_player

  def _legal_actions(self):
    """Returns a list of legal actions."""
    if self._is_terminal:
      return []
    # If player is dead, their only action is pass/nothing,
    # but OpenSpiel doesn't have PASS for simultaneous easily wrapped in
    # sequential.
    # We'll just allow any action and ignore it if dead.
    return [a.value for a in Action]

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    # Store move
    self._move_buffer[self._next_player] = action

    # Move to next player
    self._next_player += 1

    # If all players have moved, process the turn
    if self._next_player >= self.num_players:
      self._process_simultaneous_turn()
      self._next_player = 0
      self._move_buffer = [None] * self.num_players

  def _process_simultaneous_turn(self):
    self._steps += 1

    # Calculate new heads
    new_heads = []
    for i in range(self.num_players):
      if not self.is_alive[i]:
        new_heads.append(None)
        continue

      action = self._move_buffer[i]
      if not self.snakes[i]:
        new_heads.append(None)
        continue

      head_r, head_c = self.snakes[i][0]

      if action == Action.UP:
        head_r -= 1
      elif action == Action.DOWN:
        head_r += 1
      elif action == Action.LEFT:
        head_c -= 1
      elif action == Action.RIGHT:
        head_c += 1

      new_heads.append((head_r, head_c))

    # Determine which snakes eat food (simultaneous arrival allowed)
    eating = [False] * self.num_players
    food_eaten = False

    for i, head in enumerate(new_heads):
      if head is None:
        continue
      if head == self.food:
        eating[i] = True
        food_eaten = True

    # Move snakes (tentatively)
    # If not eating, remove tail.
    # Add new head.

    # Conflict resolution:
    # 1. Wall collision -> die
    # 2. Self collision -> die
    # 3. Other snake collision -> die
    # 4. Head-to-Head collision -> both die

    next_snakes = []
    temp_alive = list(self.is_alive)

    for i in range(self.num_players):
      if not self.is_alive[i]:
        next_snakes.append(self.snakes[i])  # Dead snake body stays? Or removed?
        # Standard snake: dead snake is removed.
        # Let's say dead snake is cleared from board immediately.
        # So next_snakes[i] should be empty.
        # But currently self.snakes[i] might still have body.
        # Let's verify: if I died last turn, I am already empty?
        continue

      # Calculate new body
      current_snake = list(self.snakes[i])
      head = new_heads[i]

      # Check Wall
      if not (0 <= head[0] < self.rows and 0 <= head[1] < self.cols):
        temp_alive[i] = False
        next_snakes.append([])  # Removed
        continue

      # Update body
      current_snake.insert(0, head)
      if not eating[i]:
        current_snake.pop()
      next_snakes.append(current_snake)

    # Check collisions
    # We need to check collisions against ALL snakes'
    # bodies as they are *after* the move.
    # Note: Head-to-Head is a special case of hitting a body (another head).

    # Create a set of all body parts for fast lookup,
    # mapping pos -> list of player_ids
    # But wait, Head-to-Head means both die.
    # Head-to-Body means Head dies.

    # Let's iterate all alive players and check valid conditions.
    final_alive = list(temp_alive)

    for i in range(self.num_players):
      if not temp_alive[i]:
        continue

      head = next_snakes[i][0]

      # Check against all snakes
      for j in range(self.num_players):
        if not temp_alive[j]:
          continue  # Don't collide with already dead snakes (removed)

        snake_body = next_snakes[j]
        # collision with self: check if head in body[1:]
        # collision with other: check if head in body

        start_index = 1 if i == j else 0
        if head in snake_body[start_index:]:
          # Collision!
          # If head-to-head (i.e. head == snake_body[0] and i != j), both die?
          # The loop will catch j colliding with i later (or earlier).
          # So we just mark i as dead.
          final_alive[i] = False
          break

    # Update state
    self.is_alive = final_alive

    # Update scores and bodies
    for i in range(self.num_players):
      if self.is_alive[i]:
        self.snakes[i] = next_snakes[i]
        if eating[i]:
          self.scores[i] += 1.0
      else:
        self.snakes[i] = []  # Remove dead snakes

    # Handle food
    if food_eaten:
      self._place_food()

    # Check termination
    # Game ends if 0 or 1 player alive (if started with >1)
    # If single player mode, end if dead.
    num_alive = sum(self.is_alive)
    if self.num_players > 1:
      if num_alive <= 1:
        self._is_terminal = True
        # If 1 alive, they win. +Bonus?
        # Returns are just scores.
    else:
      if num_alive == 0:
        self._is_terminal = True

  def _action_to_string(self, action):
    """Action -> string."""
    return Action(action).name

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._is_terminal

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    return self.scores

  def __str__(self):
    """String representation of the state."""
    board = np.full((self.rows, self.cols), ".")

    if self.food:
      board[self.food] = "*"

    for i, snake in enumerate(self.snakes):
      if not self.is_alive[i]:
        continue

      # Symbol for player: 0, 1, 2, 3
      # Head is H0, H1? Or just id.
      # Let's use digit for body, Upper char for head?
      # 0: a, A
      # 1: b, B
      # ...

      chars = [("a", "A"), ("b", "B"), ("c", "C"), ("d", "D")]
      body_char, head_char = chars[i % 4]

      for r, c in snake:
        board[r, c] = body_char

      if snake:
        hr, hc = snake[0]
        board[hr, hc] = head_char

    return "\n".join("".join(row) for row in board)


class SnakeObserver:
  """Observer for Snake."""

  def __init__(self, rows, cols, num_players):
    self.rows = rows
    self.cols = cols
    self.num_players = num_players
    # Channels:
    # 0: Food
    # 1..N: Snake i Body
    # N+1..2N: Snake i Head
    shape = (1 + 2 * num_players, self.rows, self.cols)
    self.tensor = np.zeros(np.prod(shape), np.float32)
    self.dict = {"observation": np.reshape(self.tensor, shape)}

  def set_from(self, state):
    """Sets the observer's tensor representation from the game state.

    The observation tensor has the following channels:
    - Channel 0: Food position.
    - Channels 1 to N: Body of snake i.
    - Channels N+1 to 2N: Head of snake i.

    Args:
      state: The current game state (SnakeState).
    """
    obs = self.dict["observation"]
    obs.fill(0)

    # Food
    if state.food:
      fr, fc = state.food
      obs[0, fr, fc] = 1.0

    for i in range(self.num_players):
      # Body
      if state.is_alive[i]:
        for r, c in state.snakes[i]:
          obs[1 + i, r, c] = 1.0

        # Head
        if state.snakes[i]:
          hr, hc = state.snakes[i][0]
          obs[1 + self.num_players + i, hr, hc] = 1.0

  def string_from(self, state):
    return str(state)


# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, SnakeGame)