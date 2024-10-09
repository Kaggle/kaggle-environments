import chess
import random
import json
from os import path

ERROR = "ERROR"
DONE = "DONE"
INACTIVE = "INACTIVE"
ACTIVE = "ACTIVE"

def random_agent(obs):
  """
  Selects a random legal move from the board.
  Returns:
    A string representing the chosen move in UCI notation (e.g., "e2e4").
  """
  board = obs.board
  board_obj = chess.Board(board)
  moves = list(board_obj.legal_moves)
  return random.choice(moves).uci()

agents = {"random": random_agent}

def interpreter(state, env):
    if env.done:
        return state

    # Isolate the active and inactive agents.
    active = state[0] if state[0].status == ACTIVE else state[1]
    inactive = state[0] if state[0].status == INACTIVE else state[1]
    if active.status != ACTIVE or inactive.status != INACTIVE:
        active.status = DONE if active.status == ACTIVE else active.status
        inactive.status = DONE if inactive.status == INACTIVE else inactive.status
        return state

    # The board is shared, only update the first state.
    board = state[0].observation.board

    # Create a chess board object from the FEN string
    board_obj = chess.Board(board)

    # Get the action (move) from the agent
    action = active.action

    # Check if the move is legal
    try:
        move = chess.Move.from_uci(action)
        if move not in board_obj.legal_moves:
            raise ValueError("Illegal move") 
    except:
        active.status = ERROR
        active.reward = -1
        inactive.status = DONE
        return state

    # Make the move
    board_obj.push(move)

    # Update the board in the observation
    state[0].observation.board = board_obj.fen() 
    state[1].observation.board = board_obj.fen()

    # Check for game end conditions
    if board_obj.is_checkmate():
        active.reward = 1
        active.status = DONE
        inactive.reward = -1
        inactive.status = DONE
    elif board_obj.is_stalemate() or board_obj.is_insufficient_material() or board_obj.is_game_over():
        active.status = DONE
        inactive.status = DONE
    else:
        # Switch turns
        active.status = INACTIVE
        inactive.status = ACTIVE

    return state

def renderer(state, env):
  board_str = state[0].observation.board
  board_obj = chess.Board(board_str)

  # Unicode characters for chess pieces
  piece_symbols = {
      'P': '♙', 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔',
      'p': '♟', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚',
      '.': ' '  # Empty square
  }

  board_repr = ""
  for square in chess.SQUARES:
    piece = board_obj.piece_at(square)
    if piece:
      board_repr += piece_symbols[piece.symbol()]
    else:
      board_repr += piece_symbols['.']
    if chess.square_file(square) == 7:  # End of a rank
      board_repr += "\n"

  return board_repr

jsonpath = path.abspath(path.join(path.dirname(__file__), "chess.json"))
with open(jsonpath) as f:
    specification = json.load(f)

def html_renderer():
    jspath = path.abspath(path.join(path.dirname(__file__), "chess.js"))
    with open(jspath) as f:
        return f.read()
