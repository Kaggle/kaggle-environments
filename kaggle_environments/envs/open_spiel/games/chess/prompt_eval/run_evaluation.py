#!/usr/bin/env python3
"""
Chess prompt evaluation runner.

Run batch evaluation of chess positions using a specific model and prompt strategy.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from logger import get_logger
from clients import GeminiClient, AnthropicClient, OpenAIClient
from prompts import generate_fen_prompt, generate_board_json_prompt, generate_board_json_no_pgn_prompt
from response_parser import parse_chess_response, ParseResult
from move_evaluator import evaluate_move, MoveEvaluation
from result_writer import ResultWriter, EvaluationResult, generate_run_id, generate_output_filename


def create_client(model_id: str, logger):
    """Create LLM client based on model identifier."""
    if model_id.startswith('gemini'):
        return GeminiClient(logger)
    elif model_id.startswith('claude'):
        return AnthropicClient(logger)
    elif model_id.startswith('o3'):
        return OpenAIClient(logger)
    else:
        raise ValueError(f"Unknown model: {model_id}")


def load_positions(positions_file: str, max_positions: int = None): # Review: not sure if you've covered this elsewhere, but default to filtered_chess_positions.jsonl
    """Load chess positions from JSONL file."""
    positions = []
    with open(positions_file, 'r') as f:
        for i, line in enumerate(f):
            if max_positions and i >= max_positions:
                break
            positions.append(json.loads(line.strip()))
    return positions


def process_single_position(position, position_index, model_id, strategy, 
                           run_id, run_timestamp, logger):
    """Process a single chess position and return evaluation result."""
    start_time = datetime.now()
    
    # Create client (each thread gets its own)
    client = create_client(model_id, logger)
    
    # Generate prompt
    if strategy == 'fen_basic':
        # Extract player to move from FEN
        fen_parts = position['fen'].split()
        player = 'white' if fen_parts[1] == 'w' else 'black'
        prompt = generate_fen_prompt(position['fen'], position['pgn'], player)
    elif strategy == 'board_json':
        # Extract player to move from FEN
        fen_parts = position['fen'].split()
        player = 'white' if fen_parts[1] == 'w' else 'black'
        prompt = generate_board_json_prompt(position['fen'], position['pgn'], player)
    elif strategy == 'board_json_no_pgn':
        # Extract player to move from FEN
        fen_parts = position['fen'].split()
        player = 'white' if fen_parts[1] == 'w' else 'black'
        prompt = generate_board_json_no_pgn_prompt(position['fen'], position['pgn'], player)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Send to LLM
    response = client.send_message(prompt, max_tokens=32000)
    
    # Parse move (always attempt, even if LLM failed)
    if response.is_success:
        parse_result = parse_chess_response(response.response_text)
    else:
        parse_result = ParseResult(success=False, error_message="LLM request failed")
    
    # Evaluate move (always attempt, even if parsing failed)
    if parse_result.success:
        eval_result = evaluate_move(position['fen'], parse_result.move)
    else:
        eval_result = MoveEvaluation(is_legal=False, error_message="No move to evaluate")
    
    end_time = datetime.now()
    
    # Create evaluation result
    return EvaluationResult(
        position=position,
        position_index=position_index,
        prompt_text=prompt,
        strategy_name=strategy,
        llm_response=response,
        model_id=model_id,
        temperature=1.0,  # TODO: make configurable
        max_tokens=32000,
        parse_result=parse_result,
        move_evaluation=eval_result,
        start_time=start_time,
        end_time=end_time,
        run_id=run_id,
        run_timestamp=run_timestamp
    )


def run_evaluation(positions_file: str, model_id: str, strategy: str, 
                  output_dir: str, max_positions: int = None, workers: int = 4):
    """Run parallel evaluation of positions."""
    logger = get_logger()
    
    # Load positions
    positions = load_positions(positions_file, max_positions)
    logger.info(f"Loaded {len(positions)} positions")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate run metadata
    run_id = generate_run_id()
    run_timestamp = datetime.now()
    output_file = generate_output_filename(output_dir, run_id)
    
    logger.info(f"Starting evaluation run {run_id} with {workers} workers")
    logger.info(f"Output file: {output_file}")
    
    # Process positions in parallel and write results as they complete
    completed_count = 0
    successful_count = 0
    legal_moves = 0
    last_log_time = datetime.now()
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        with ResultWriter(output_file) as writer:
            # Submit all positions for processing
            future_to_position = {
                executor.submit(
                    process_single_position, 
                    position, i, model_id, strategy, 
                    run_id, run_timestamp, logger
                ): (position, i) 
                for i, position in enumerate(positions)
            }
            
            running_count = len(positions)
            logger.info(f"Submitted {running_count} positions to {workers} workers")
            
            # Process results as they complete (streaming writes)
            for future in as_completed(future_to_position):
                position, position_index = future_to_position[future]
                completed_count += 1
                running_count = len(positions) - completed_count
                
                try:
                    result = future.result()
                    writer.write_result(result)
                    
                    # Track success stats
                    if result.parse_result.success:
                        successful_count += 1
                        if result.move_evaluation.is_legal:
                            legal_moves += 1
                    
                    # Log progress summary every 10 completions or 30 seconds
                    now = datetime.now()
                    if completed_count % 10 == 0 or (now - last_log_time).seconds >= 30:
                        legal_rate = legal_moves / successful_count if successful_count > 0 else 0
                        logger.info(f"Progress: {completed_count}/{len(positions)} complete, "
                                  f"{running_count} running | "
                                  f"Success: {successful_count}/{completed_count} | "
                                  f"Legal rate: {legal_rate:.1%}")
                        last_log_time = now
                        
                except Exception as e:
                    logger.error(f"Position {position_index} failed: {e}")
            
            # Final summary
            final_legal_rate = legal_moves / successful_count if successful_count > 0 else 0
            logger.info(f"Final: {completed_count} completed, {successful_count} successful, "
                       f"Legal rate: {final_legal_rate:.1%}")
    
    logger.info("Evaluation complete")
    return f"Processed {len(positions)} positions, results written to {output_file}"


def main():
    parser = argparse.ArgumentParser(description='Run chess prompt evaluation')
    parser.add_argument('--positions', default='filtered_chess_positions.jsonl',
                       help='Path to positions JSONL file (default: filtered_chess_positions.jsonl)')
    parser.add_argument('--model', default='gemini',
                       help='Model provider (gemini, claude, or o3) - default: gemini')
    parser.add_argument('--strategy', default='fen_basic',
                       help='Prompt strategy: fen_basic, board_json, or board_json_no_pgn (default: fen_basic)')
    parser.add_argument('--output-dir', default='./results',
                       help='Output directory (default: ./results)')
    parser.add_argument('--max-positions', type=int,
                       help='Maximum positions to process (for testing)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.positions):
        print(f"Error: Positions file not found: {args.positions}")
        sys.exit(1)
    
    if args.strategy not in ['fen_basic', 'board_json', 'board_json_no_pgn']:
        print(f"Error: Unknown strategy: {args.strategy}")
        print("Available strategies: fen_basic, board_json, board_json_no_pgn")
        sys.exit(1)
    
    try:
        result = run_evaluation(
            positions_file=args.positions,
            model_id=args.model,
            strategy=args.strategy,
            output_dir=args.output_dir,
            max_positions=args.max_positions,
            workers=args.workers
        )
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()