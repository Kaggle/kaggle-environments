#!/usr/bin/env python3
"""
Test the move evaluator with sample positions.
"""

import json
from move_evaluator import evaluate_move, MoveEvaluation


def test_with_sample_positions():
    """Test move evaluator with positions from chess_positions.jsonl."""
    print("Testing Move Evaluator")
    print("=" * 40)
    
    # Load first few positions from the file
    positions = []
    try:
        with open("chess_positions.jsonl", "r") as f:
            for i, line in enumerate(f):
                if i >= 3:  # Just test first 3 positions
                    break
                positions.append(json.loads(line))
    except FileNotFoundError:
        print("chess_positions.jsonl not found")
        return
    
    # Test cases for each position
    test_moves = [
        # Common moves to test
        ["e4", "Nf3", "d4", "Bc4", "Qh5"],  # Opening moves
        ["Ke2", "Kf1", "Rxa1", "Bb5"],      # Mid-game moves
        ["invalid", "Ke9", "Zz4"]           # Invalid moves
    ]
    
    for i, pos in enumerate(positions):
        print(f"\nPosition {i+1}:")
        print(f"FEN: {pos['fen']}")
        print(f"Moves so far: {pos['pgn'][:50]}...")
        
        # Try a few test moves
        for move in test_moves[min(i, len(test_moves)-1)]:
            result = evaluate_move(pos['fen'], move)
            status = "✓ Legal" if result.is_legal else "✗ Illegal"
            
            if result.is_legal and result.expectation_change is not None:
                exp_pct = result.expectation_change * 100
                score_str = f" (WDL: {exp_pct:+.1f}%, CP: {result.cp_change:+.0f})"
            elif result.error_message:
                score_str = f" ({result.error_message})"
            else:
                score_str = ""
            
            print(f"  {move:8} → {status}{score_str}")


def test_basic_functionality():
    """Test basic functionality with known positions."""
    print("\nBasic Functionality Tests")
    print("=" * 40)
    
    # Starting position
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    test_cases = [
        ("e4", True, "Standard opening"),
        ("Nf3", True, "Knight development"),
        ("e5", False, "Wrong turn - pawn blocked"),
        ("a4", True, "Legal but unusual"),
        ("invalid", False, "Invalid move"),
    ]
    
    for move, should_be_legal, description in test_cases:
        result = evaluate_move(start_fen, move)
        status = "PASS" if result.is_legal == should_be_legal else "FAIL"
        
        score_info = ""
        if result.is_legal and result.expectation_change is not None:
            exp_pct = result.expectation_change * 100
            score_info = f" WDL: {exp_pct:+.1f}% CP: {result.cp_change:+.0f}"
        elif result.error_message:
            score_info = f" Error: {result.error_message}"
            
        print(f"{status:4} | {move:8} → {description}{score_info}")


if __name__ == "__main__":
    test_basic_functionality()
    test_with_sample_positions()