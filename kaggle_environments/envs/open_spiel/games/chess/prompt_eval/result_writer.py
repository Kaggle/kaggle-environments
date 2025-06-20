"""
Result writer for chess prompt evaluation.

Handles writing evaluation results to JSONL format with proper schema.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass

from clients.llm_response import LlmResponse
from response_parser import ParseResult
from move_evaluator import MoveEvaluation


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single position."""
    # Input data
    position: Dict[str, Any]
    position_index: int
    prompt_text: str
    strategy_name: str
    
    # LLM response
    llm_response: LlmResponse
    model_id: str
    temperature: float
    max_tokens: int
    
    # Processing results
    parse_result: ParseResult
    move_evaluation: MoveEvaluation
    
    # Timing
    start_time: datetime
    end_time: datetime
    
    # Run metadata
    run_id: str
    run_timestamp: datetime


class ResultWriter:
    """Writes evaluation results to JSONL format."""
    
    def __init__(self, output_file: str):
        self.output_file = output_file
        self._file_handle = None
    
    def __enter__(self):
        self._file_handle = open(self.output_file, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file_handle:
            self._file_handle.close()
    
    def write_result(self, result: EvaluationResult):
        """Write a single evaluation result to JSONL file."""
        record = self._build_record(result)
        json_line = json.dumps(record, ensure_ascii=False)
        self._file_handle.write(json_line + '\n')
        self._file_handle.flush()
    
    def _build_record(self, result: EvaluationResult) -> Dict[str, Any]:
        """Build JSON record according to our schema."""
        # Extract player to move from FEN
        fen_parts = result.position['fen'].split()
        player_to_move = 'white' if fen_parts[1] == 'w' else 'black'
        
        return {
            "run_metadata": {
                "run_id": result.run_id,
                "timestamp": result.run_timestamp.isoformat(),
            },
            "position": {
                "fen": result.position['fen'],
                "pgn": result.position['pgn'],
                "position_index": result.position_index,
                "player_to_move": player_to_move,
            },
            "prompt": {
                "strategy_name": result.strategy_name,
                "prompt_text": result.prompt_text,
            },
            "llm": {
                "provider": self._get_provider(result.model_id),
                "model_id": result.model_id,
                "temperature": result.temperature,
                "max_tokens": result.max_tokens,
                "other_params": {}
            },
            "response": {
                "response_text": result.llm_response.response_text or "",
                "prompt_tokens": result.llm_response.prompt_tokens or 0,
                "completion_tokens": result.llm_response.completion_tokens or 0,
                "duration_ms": result.llm_response.duration_ms or 0,
                "stop_reason": result.llm_response.stop_reason or "",
                "error": result.llm_response.error,
            },
            "parsing": {
                "success": result.parse_result.success,
                "extracted_move": result.parse_result.move,
                "error_message": result.parse_result.error_message,
            },
            "evaluation": {
                "is_legal": result.move_evaluation.is_legal,
                "expectation_change": result.move_evaluation.expectation_change,
                "cp_change": result.move_evaluation.cp_change,
                "error_message": result.move_evaluation.error_message,
                "stockfish_version": "16.1"  # TODO: get actual version
            },
            "timing": {
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat(),
                "total_duration_ms": int((result.end_time - result.start_time).total_seconds() * 1000)
            }
        }
    
    def _get_provider(self, model_id: str) -> str:
        """Extract provider name from model ID."""
        if model_id.startswith('gemini'):
            return 'gemini'
        elif model_id.startswith('claude'):
            return 'anthropic'
        elif model_id.startswith('o3'):
            return 'openai'
        else:
            return 'unknown'


def generate_run_id() -> str:
    """Generate a unique run ID."""
    return str(uuid.uuid4())


def generate_output_filename(output_dir: str, run_id: str) -> str:
    """Generate output filename for evaluation results."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{output_dir}/eval_results_{run_id}_{timestamp}.jsonl"