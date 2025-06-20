#!/usr/bin/env python3
"""
Single position evaluation tool for chess prompt testing.

Loads one example from chess_positions.jsonl, runs it against Gemini 2.5 pro,
parses the result, and grades it with the move evaluator.
"""

import json
import sys
import chess
from pathlib import Path

# Add the current directory to the path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from clients.gemini_client import GeminiClient
from prompts import generate_fen_prompt
from response_parser import parse_chess_response
from move_evaluator import evaluate_move
from logger import get_logger


def load_position(jsonl_path: str, index: int = 0) -> dict:
    """Load a specific position from the JSONL file."""
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line.strip())
    raise IndexError(f"Position {index} not found in {jsonl_path}")


def determine_turn_color(fen: str) -> str:
    """Determine whose turn it is from the FEN string."""
    board = chess.Board(fen)
    return "white" if board.turn else "black"


def main():
    """Run single position evaluation."""
    # Setup
    logger = get_logger()
    client = GeminiClient(logger)
    
    # Load position
    jsonl_path = current_dir / "chess_positions.jsonl"
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found")
        sys.exit(1)
    
    try:
        position = load_position(str(jsonl_path), 2)
    except IndexError as e:
        print(f"Error loading position: {e}")
        sys.exit(1)
    
    fen = position["fen"]
    pgn = position["pgn"]
    color = determine_turn_color(fen)
    
    print("=" * 80)
    print("CHESS POSITION EVALUATION")
    print("=" * 80)
    print(f"FEN: {fen}")
    print(f"Turn: {color}")
    print(f"Move history: {pgn}")
    print()
    
    # Generate prompt
    prompt = generate_fen_prompt(fen, pgn, color)
    print("PROMPT:")
    print("-" * 40)
    print(prompt)
    print()
    
    # Get LLM response
    print("SENDING TO GEMINI 2.5 PRO...")
    print("-" * 40)
    response = client.send_message(prompt, max_tokens=20000)
    
    if not response.is_success:
        print(f"❌ LLM ERROR: {response.error}")
        sys.exit(1)
    
    print("✅ LLM RESPONSE RECEIVED")
    print(f"Duration: {response.duration_ms}ms")
    print(f"Tokens - Prompt: {response.prompt_tokens}, Completion: {response.completion_tokens}")
    print()
    print("RAW RESPONSE:")
    print("-" * 40)
    print(response.response_text)
    print()
    
    # Parse the move
    print("PARSING MOVE...")
    print("-" * 40)
    parse_result = parse_chess_response(response.response_text)
    
    if not parse_result.success:
        print(f"❌ PARSE ERROR: {parse_result.error_message}")
        sys.exit(1)
    
    print(f"✅ PARSED MOVE: {parse_result.move}")
    print()
    
    # Evaluate the move
    print("EVALUATING MOVE...")
    print("-" * 40)
    evaluation = evaluate_move(fen, parse_result.move)
    
    if not evaluation.is_legal:
        print(f"❌ ILLEGAL MOVE: {evaluation.error_message}")
    else:
        print("✅ LEGAL MOVE")
        if evaluation.expectation_change is not None:
            print(f"WDL Expectation Change: {evaluation.expectation_change:.4f}")
            print(f"Centipawn Change: {evaluation.cp_change:.1f}")
        else:
            print(f"⚠️  EVALUATION ERROR: {evaluation.error_message}")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Position: {fen}")
    print(f"LLM Move: {parse_result.move}")
    print(f"Legal: {'✅' if evaluation.is_legal else '❌'}")
    if evaluation.is_legal and evaluation.expectation_change is not None:
        print(f"Quality (WDL): {evaluation.expectation_change:.4f}")
        print(f"Quality (CP): {evaluation.cp_change:.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()