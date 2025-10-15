"""Change Tic Tac Toe state and action string representations.

Intended as an example of how to subclass proxy.Game and proxy.State.

Example:
  game = TicTacToeGame()
  state = game.new_initial_state()
  state.apply_action(state.legal_actions()[0])
  print(state)

Shows the board in a more readable format:
 x | . | .
-----------
 . | . | .
-----------
 . | . | .

Also remaps action strings to following format: <row,col>

Adds to_json() and action_to_json() methods as symbolic representation.
"""

import json
from typing import Any

import pyspiel

from ... import proxy

NUM_COLS = 3
NUM_ROWS = 3


class TicTacToeState(proxy.State):
    """Tic Tac Toe state proxy."""

    def observation_string(self, player: int) -> str:
        del player  # Unused.
        return str(self)  # Perfect information game, return state as observation.

    def to_json(self) -> str:
        board = "".join(str(self).split())
        assert len(board) == 9
        board = list(board)
        board = [None if cell == "." else cell for cell in board]
        if self.is_terminal():
            current_player = None
        elif board.count("x") > board.count("o"):
            current_player = "o"
        else:
            current_player = "x"
        return json.dumps(dict(board=board, current_player=current_player))

    def _action_to_string(self, player: int, action: int) -> str:
        del player  # Unused.
        row, col = divmod(action, NUM_COLS)
        return f"<{row},{col}>"

    def action_to_json(self, *args: int) -> str:
        match len(args):
            case 1:
                action = args[0]
                return self._action_to_json(self.current_player(), action)
            case 2:
                return self._action_to_json(args[0], args[1])
            case _:
                raise ValueError(f"Invalid args, expected (player) or (player, action), got: {args}")

    def _action_to_json(self, player: int, action: int) -> str:
        row, col = divmod(action, NUM_COLS)
        player_str = None
        if player == 0:
            player_str = "x"
        elif player == 1:
            player_str = "o"
        return json.dumps({"player": player_str, "row": int(row), "col": int(col)})

    def observation_json(self, player: int) -> str:
        del player  # Unused.
        return self.to_json()

    def observation_string(self, player: int) -> str:
        return self.observation_json(player)


class TicTacToeGame(proxy.Game):
    """Tic Tac Toe game proxy."""

    def __init__(self, params: Any | None = None):
        del params
        wrapped = pyspiel.load_game("tic_tac_toe()")
        super().__init__(wrapped, short_name="tic_tac_toe_proxy", long_name="Tic Tac Toe (proxy)")

    def new_initial_state(self, *args) -> TicTacToeState:
        return TicTacToeState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(TicTacToeGame().get_type(), TicTacToeGame)
