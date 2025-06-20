"""
Chess move evaluation for LLM prompt testing.

Validates move legality and scores moves using Stockfish analysis.
"""

import chess
import chess.engine
from dataclasses import dataclass
from typing import Optional


@dataclass
class MoveEvaluation:
    """Result of evaluating a chess move."""
    is_legal: bool
    expectation_change: Optional[float] = None  # WDL expectation change (-1 to +1)
    cp_change: Optional[float] = None  # centipawn change for reference
    error_message: Optional[str] = None


def evaluate_move(fen: str, move: str, stockfish_path: str = "stockfish") -> MoveEvaluation:
    """
    Evaluate a chess move for legality and quality using WDL expectations.
    
    Uses Win/Draw/Loss expectation changes as recommended by python-chess docs
    for meaningful position comparison. Most moves will show small negative 
    values since they rarely match the engine's top choice.
    
    Args:
        fen: FEN string representing the current position
        move: Move in algebraic notation or UCI format
        stockfish_path: Path to Stockfish binary
        
    Returns:
        MoveEvaluation with legality and expectation/centipawn changes
    """
    try:
        # Parse the board position
        board = chess.Board(fen)
    except ValueError as e:
        return MoveEvaluation(
            is_legal=False,
            error_message=f"Invalid FEN: {e}"
        )
    
    # Try to parse the move
    try:
        # First try UCI format (e.g., "e2e4")
        if len(move) >= 4 and move[0] in 'abcdefgh' and move[1] in '12345678':
            chess_move = chess.Move.from_uci(move)
        else:
            # Try algebraic notation (e.g., "Nf3", "e4")
            chess_move = board.parse_san(move)
    except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError) as e:
        return MoveEvaluation(
            is_legal=False,
            error_message=f"Could not parse move '{move}': {e}"
        )
    
    # Check if move is legal in this position
    if not board.is_legal(chess_move):
        return MoveEvaluation(
            is_legal=False,
            error_message=f"Move '{move}' is not legal in this position"
        )
    
    # Move is legal, now score it with Stockfish
    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            # Remember whose turn it is before the move
            player_is_white = board.turn
            
            # Analyze position before the move (1 second)
            info_before = engine.analyse(board, chess.engine.Limit(time=1.0))
            
            # Make the move and analyze after
            board.push(chess_move)
            info_after = engine.analyse(board, chess.engine.Limit(time=1.0))
            
            # Calculate changes using WDL expectations (recommended by python-chess docs)
            score_before = info_before["score"]
            score_after = info_after["score"]
            
            # Handle mate scores - need to access relative scores properly
            score_before_relative = score_before.white() if player_is_white else score_before.black()
            score_after_relative = score_after.white() if player_is_white else score_after.black()
            
            if score_before_relative.is_mate() or score_after_relative.is_mate():
                # Simplified mate handling
                if score_before_relative.is_mate() and score_after_relative.is_mate():
                    expectation_change = 0.0
                    cp_change = 0.0
                elif score_after_relative.is_mate():
                    mate_in = score_after_relative.mate()
                    if mate_in > 0:  # Mate for current player
                        expectation_change = 0.5
                        cp_change = 1000.0
                    else:  # Mate against current player
                        expectation_change = -0.5
                        cp_change = -1000.0
                else:
                    expectation_change = 0.3
                    cp_change = 500.0
            else:
                # Use WDL expectations for meaningful comparison
                if player_is_white:
                    wdl_before = score_before.white().wdl()
                    wdl_after = score_after.white().wdl()
                    cp_before = score_before.white().score(mate_score=1000)
                    cp_after = score_after.white().score(mate_score=1000)
                else:
                    wdl_before = score_before.black().wdl()
                    wdl_after = score_after.black().wdl()
                    cp_before = score_before.black().score(mate_score=1000)
                    cp_after = score_after.black().score(mate_score=1000)
                
                expectation_change = wdl_after.expectation() - wdl_before.expectation()
                cp_change = cp_after - cp_before
            
            return MoveEvaluation(
                is_legal=True,
                expectation_change=expectation_change,
                cp_change=cp_change
            )
            
    except Exception as e:
        return MoveEvaluation(
            is_legal=True,
            error_message=f"Stockfish analysis failed: {e}"
        )