from Chessnut import Game
import random
import json
from os import path
from collections import defaultdict

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
    board = Game(obs.board)
    moves = list(board.get_moves())
    return random.choice(moves)

def king_shuffle_agent(obs):
    """Moves the king pawn and then shuffles the king."""
    game = Game(obs.board)
    moves = list(game.get_moves())

    to_move = ["e7e5", "e2e4", "e8e7", "e7e8", "e1e2", "e2e1"]
    for move in to_move:
        if move in moves:
            return move

    # If no other moves are possible, pick a random legal move (or return None)
    return random.choice(moves) if moves else None


agents = {"random": random_agent, "king_shuffle": king_shuffle_agent}

def sufficient_material(pieces):
    """Checks if the given pieces are sufficient for checkmate."""
    if pieces['q'] > 0 or pieces['r'] > 0 or pieces['p'] > 0:
        return True
    if pieces['n'] + pieces['b'] >= 3:
        return True
    # TODO: they have to be opposite color bishops 
    if pieces['b'] >= 2:
        return True
    
    return False


def is_insufficient_material(board):
    white_pieces = defaultdict(int)
    black_pieces = defaultdict(int)

    for square in range(64):
        piece = board.get_piece(square)
        if piece:
            if piece.isupper():
                white_pieces[piece.lower()] += 1
            else:
                black_pieces[piece.lower()] += 1

    if not sufficient_material(white_pieces) and not sufficient_material(black_pieces):
        return True

    return False


def square_str_to_int(square_str):
    """Converts a square string (e.g., "a2") to an integer index (0-63)."""
    if len(square_str) != 2 or not ('a' <= square_str[0] <= 'h' and '1' <= square_str[1] <= '8'):
        raise ValueError("Invalid square string")

    col = ord(square_str[0]) - ord('a')  # Get column index (0-7)
    row = int(square_str[1]) - 1        # Get row index (0-7)
    return row * 8 + col

def is_pawn_move_or_capture(board, move):
    move = move.lower()
    if board.get_piece(square_str_to_int(move[2:4])).lower() == "p":
        return True
    if board.get_piece(square_str_to_int(move[:2])) != " ":
        return True
    return False



seen_positions = defaultdict(int)
pawn_or_capture_move_count = 0

def interpreter(state, env):
    global seen_positions
    global pawn_or_capture_move_count
    if env.done:
        seen_positions = defaultdict(int)
        pawn_or_capture_move_count = 0
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

   # Create a chessnut game object from the FEN string
    game = Game(board)

    # Get the action (move) from the agent
    action = active.action

    if action and is_pawn_move_or_capture(game.board, action):
        pawn_or_capture_move_count = 0
    else:
        pawn_or_capture_move_count += 1

    # Check if the move is legal
    try:
        game.apply_move(action)
    except:
        active.status = ERROR
        active.reward = -1
        inactive.status = DONE
        return state
    board_str = game.get_fen().split(" ")[0]
    seen_positions[board_str] += 1

    # Update the board in the observation
    state[0].observation.board = game.get_fen()
    state[1].observation.board = game.get_fen()

    # Check for game end conditions
    if pawn_or_capture_move_count == 100 or is_insufficient_material(game.board):
        active.status = DONE
        inactive.status = DONE
    elif seen_positions[board_str] >= 3 or game.status == Game.STALEMATE:
        active.status = DONE
        inactive.status = DONE
    elif game.status == Game.CHECKMATE:
        active.reward = 1
        active.status = DONE
        inactive.reward = -1
        inactive.status = DONE

    else:
        # Switch turns
        active.status = INACTIVE
        inactive.status = ACTIVE

    return state

def renderer(state, env):
  board_fen = state[0].observation.board
  game = Game(board_fen)
  return game.board
  
jsonpath = path.abspath(path.join(path.dirname(__file__), "chess.json"))
with open(jsonpath) as f:
    specification = json.load(f)

def html_renderer():
    jspath = path.abspath(path.join(path.dirname(__file__), "chess.js"))
    with open(jspath) as f:
        return f.read()
