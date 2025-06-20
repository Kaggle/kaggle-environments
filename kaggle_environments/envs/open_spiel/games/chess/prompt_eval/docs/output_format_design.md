# Chess Prompt Evaluation Output Format Design

## Overview
Design a comprehensive, analysis-ready output format for the chess prompt evaluation system that supports:
- Multithreaded writing during batch processing
- Prompt strategy evolution over time
- Statistical analysis and visualization
- Cost tracking and debugging
- Easy data filtering and aggregation

## Proposed Architecture

### 1. **Primary Output: Evaluation Records (JSONL)**
- **File naming**: `eval_results_{run_id}_{timestamp}.jsonl`
- **Format**: One JSON object per line (supports concurrent append)
- **Content**: Complete evaluation record for each position

### 2. **Run Metadata (JSON)**
- **File naming**: `run_metadata_{run_id}_{timestamp}.json`
- **Content**: Run configuration, summary statistics, and metadata

### 3. **Optional: Experiment Index (JSON)**
- **File naming**: `experiment_index.json`
- **Content**: Registry of all runs for easy lookup and comparison

## Detailed Schema Design

### Evaluation Record Schema (JSONL)
```json
{
  "run_metadata": {
    "run_id": "uuid",
    "timestamp": "ISO8601",
  },
  "position": {
    "fen": "string",
    "pgn": "string", 
    "position_index": "int",
    "player_to_move": "white|black",
  },
  "prompt": {
    "strategy_name": "string",
    "prompt_text": "string",
  },
  "llm": {
    "provider": "anthropic|openai|gemini",
    "model_id": "string",
    "temperature": "float",
    "max_tokens": "int",
    "other_params": {}
  },
  "response": {
    "response_text": "string",
    "prompt_tokens": "int",
    "completion_tokens": "int",  // Not sure if this includes thinking tokens, but let's make sure it does
    "duration_ms": "int",
    "stop_reason": "string",
    "error": "string|null",
  },
  "parsing": {
    "success": "bool",
    "extracted_move": "string|null",
    "error_message": "string|null",
  },
  "evaluation": {
    "is_legal": "bool",
    "expectation_change": "float|null",
    "cp_change": "float|null", 
    "error_message": "string|null",
    "stockfish_version": "string"
  },
  "timing": {
    "start_time": "ISO8601",
    "end_time": "ISO8601",
    "total_duration_ms": "int"
  }
}
```

### Run Metadata Schema
```json
{
  "run_id": "uuid",
  "timestamp": "ISO8601",
  "configuration": {
    "position_file": "string",
    "position_count": "int",
    "prompt_strategy": "string",
    "model": "string",
    "parallel_workers": "int",
  },
  "results_file": "string",
  "status": "running|completed|failed|cancelled",
  "summary": {
    "total_evaluations": "int",
    "successful_evaluations": "int", 
    "failed_evaluations": "int",
    "legal_move_rate": "float",
    "average_expectation_change": "float",
    "total_duration_ms": "int"
  },
  "error_summary": {
    "parsing_errors": "int",
    "llm_errors": "int", 
    "evaluation_errors": "int"
  }
}
```

## Benefits of This Design

1. **Multithreaded Support**: JSONL format allows concurrent appends
2. **Analysis Ready**: Structured data with proper typing for statistics
3. **Prompt Evolution**: Clear versioning and strategy identification
4. **Cost Tracking**: Detailed cost breakdown per evaluation
5. **Debugging**: Complete trace from position to final evaluation
6. **Filtering**: Easy to filter by model, strategy, position type, etc.
7. **Scalability**: Separate files per run prevent massive single files
8. **Human Readable**: JSON format for easy inspection and debugging

## Implementation Notes

- Use UUIDs for run_id to ensure uniqueness
- Include version numbers for schema evolution
- Store raw response hashes instead of full raw responses to save space
- Support incremental analysis during long-running evaluations

## File Organization Example

```
prompt_eval/
├── results/
│   ├── experiment_index.json
│   ├── eval_results_abc123_20240101_120000.jsonl
│   ├── run_metadata_abc123_20240101_120000.json
│   ├── eval_results_def456_20240101_140000.jsonl
│   └── run_metadata_def456_20240101_140000.json
└── ...
```