"""
Chess prompt generation for LLM evaluation.

Simple API for generating chess prompts with variable substitution.
"""

import chess
import json


def fen_to_board_json(fen: str) -> dict:
    """
    Convert FEN string to structured JSON board representation.
    
    Args:
        fen: Standard FEN notation string
        
    Returns:
        Dictionary with board structure optimized for LLM comprehension
    """
    board = chess.Board(fen)
    
    # Create 8x8 board array with square-piece mapping
    board_array = []
    
    # Iterate through ranks (8 to 1)
    for rank in range(8):
        rank_array = []
        # Iterate through files (a to h)  
        for file in range(8):
            square = chess.square(file, 7 - rank)  # chess.square uses 0-based indexing
            piece = board.piece_at(square)
            
            # Get square name
            square_name = chess.square_name(square)
            
            # Get piece name or "empty"
            if piece is None:
                piece_name = "empty"
            else:
                color = "white" if piece.color else "black"
                piece_type = piece.piece_type
                type_names = {
                    chess.PAWN: "pawn",
                    chess.ROOK: "rook", 
                    chess.KNIGHT: "knight",
                    chess.BISHOP: "bishop",
                    chess.QUEEN: "queen",
                    chess.KING: "king"
                }
                piece_name = f"{color}_{type_names[piece_type]}"
            
            rank_array.append({square_name: piece_name})
        
        board_array.append(rank_array)
    
    # Extract game state information
    game_state = {
        "active_player": "white" if board.turn else "black",
        "move_number": board.fullmove_number,
        "castling_rights": {
            "white_kingside": board.has_kingside_castling_rights(chess.WHITE),
            "white_queenside": board.has_queenside_castling_rights(chess.WHITE),
            "black_kingside": board.has_kingside_castling_rights(chess.BLACK),
            "black_queenside": board.has_queenside_castling_rights(chess.BLACK)
        },
        "en_passant_target": chess.square_name(board.ep_square) if board.ep_square else None
    }
    
    # Calculate material balance (standard piece values)
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                   chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    
    white_material = sum(piece_values[piece.piece_type] 
                        for piece in board.piece_map().values() 
                        if piece.color == chess.WHITE)
    black_material = sum(piece_values[piece.piece_type]
                        for piece in board.piece_map().values()
                        if piece.color == chess.BLACK)
    
    # Find king positions
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)
    
    position_analysis = {
        "material_balance": {
            "white": white_material,
            "black": black_material
        },
        "king_safety": {
            "white_king_position": chess.square_name(white_king_square) if white_king_square else None,
            "black_king_position": chess.square_name(black_king_square) if black_king_square else None,
            "castled": {
                "white": not board.has_castling_rights(chess.WHITE) and white_king_square not in [chess.E1],
                "black": not board.has_castling_rights(chess.BLACK) and black_king_square not in [chess.E8]
            }
        }
    }
    
    return {
        "board": board_array,
        "game_state": game_state,
        "position_analysis": position_analysis,
        "fen": fen
    }


def generate_fen_prompt(fen_state: str, move_history: str, color: str) -> str:
    """
    Generate a chess prompt for the given position.
    
    Args:
        state_str: FEN string representing the current position
        move_history: String of moves played so far  
        color: Player color ("white" or "black")
    
    Returns:
        Formatted prompt string
    """
    return f"""Let's play chess. The current game state in FEN is:
{fen_state}
The moves played so far are:
{move_history}
You are playing as player {color}.
It is now your turn. Play your strongest move. The move MUST be legal. Reason step by step to come up with your move, then output your final answer in the format "Final Answer: X" where X is your chosen move in Algebraic Notation."""


def generate_board_json_prompt(fen_state: str, move_history: str, color: str) -> str:
    """
    Generate a chess prompt using structured JSON board representation.
    
    Args:
        fen_state: FEN string representing the current position
        move_history: String of moves played so far  
        color: Player color ("white" or "black")
    
    Returns:
        Formatted prompt string with JSON board representation
    """
    board_json = fen_to_board_json(fen_state)
    board_json_str = json.dumps(board_json, indent=2)
    
    return f"""Let's play chess. The current game state is:
{board_json_str}
The moves played so far are:
{move_history}
You are playing as player {color}.
It is now your turn. Play your strongest move. The move MUST be legal. Reason step by step to come up with your move, then output your final answer in the format "Final Answer: X" where X is your chosen move in Algebraic Notation."""


def generate_board_json_no_pgn_prompt(fen_state: str, move_history: str, color: str) -> str:
    """
    Generate a chess prompt using structured JSON board representation without PGN history.
    
    Args:
        fen_state: FEN string representing the current position
        move_history: String of moves played so far (ignored)
        color: Player color ("white" or "black")
    
    Returns:
        Formatted prompt string with JSON board representation, no move history
    """
    board_json = fen_to_board_json(fen_state)
    board_json_str = json.dumps(board_json, indent=2)
    
    return f"""Let's play chess. The current game state is:
{board_json_str}
You are playing as player {color}.
It is now your turn. Play your strongest move. The move MUST be legal. Reason step by step to come up with your move, then output your final answer in the format "Final Answer: X" where X is your chosen move in Algebraic Notation."""


