# Chess Prompt Evaluation Runner Implementation Plan

## Overview
Based on the simplified output format, implement a batch evaluation runner that processes multiple chess positions with a single model + prompt strategy combination.

## Key Architecture Decision
**One run = One model + One strategy + Multiple positions**
- To compare strategies: run multiple separate evaluations
- To compare models: run multiple separate evaluations
- Analysis happens by comparing results across runs

## Runner Parameters

### Command Line Interface
```bash
python run_evaluation.py \
  --positions chess_positions.jsonl \
  --model gemini-1.5-pro \
  --strategy fen_basic \
  --workers 4 \
  --max-positions 100 \
  --output-dir results/
```

### Core Parameters
- `--positions`: Path to input positions JSONL file
- `--model`: Model identifier (maps to client: gemini-1.5-pro, claude-3-5-sonnet, gpt-4o, etc.)
- `--strategy`: Prompt strategy name (fen_basic, json_board, etc.)
- `--workers`: Number of parallel worker threads (default: 4)
- `--max-positions`: Optional limit on positions to process (for testing)
- `--output-dir`: Directory for results files (default: ./results/)

### Optional Parameters
- `--temperature`: Model temperature (default: strategy-specific)
- `--max-tokens`: Max tokens per response (default: 2000)
- `--timeout`: Timeout per position in seconds (default: 120)
- `--resume`: Resume from partial run (using run_id)

## Implementation Steps

### Phase 1: Core Single-Threaded Runner ✅ Testable
**Files to create:**
- `run_evaluation.py` - Main runner script
- `evaluation_runner.py` - Core runner class
- `result_writer.py` - Thread-safe JSONL writing

**Functionality:**
1. Parse command line arguments
2. Load positions from input file
3. Initialize LLM client based on model parameter
4. Create run metadata and output files
5. Process positions sequentially:
   - Generate prompt using specified strategy
   - Send to LLM and get response
   - Parse move from response
   - Evaluate move with Stockfish
   - Write result to JSONL
6. Update run metadata with final summary

**Testing:** Run with 5-10 positions, single worker, test all error paths

### Phase 2: Thread-Safe Parallel Processing ✅ Testable
**Enhancements:**
- Add ThreadPoolExecutor for parallel position processing
- Implement thread-safe result writing (queue-based or file locking)
- Add progress tracking and logging
- Handle worker thread errors gracefully

**Testing:** Run with 20-50 positions, multiple workers, verify results consistency

### Phase 3: Robust Error Handling & Recovery ✅ Testable
**Enhancements:**
- Graceful shutdown on Ctrl+C
- Resume capability for interrupted runs
- Retry logic for transient API failures
- Rate limiting and backoff for API quotas
- Comprehensive error categorization

**Testing:** Test interruption/resume, API failures, malformed positions

### Phase 4: Progress Monitoring & Real-time Updates ✅ Testable
**Enhancements:**
- Real-time progress display (tqdm or custom)
- Live updating of run metadata during execution
- Periodic summary statistics logging
- ETA calculation based on current pace

**Testing:** Long-running evaluation with progress monitoring

## Detailed Implementation Design

### Class Structure
```python
class EvaluationRunner:
    def __init__(self, config: RunnerConfig)
    def run(self) -> str  # Returns run_id
    def _process_position(self, position: dict, position_index: int) -> dict
    def _write_result(self, result: dict)
    def _update_metadata(self, status: str, summary: dict = None)
    def _cleanup(self)

class ResultWriter:
    def __init__(self, jsonl_path: str)
    def write_result(self, result: dict)  # Thread-safe
    def close(self)

class RunnerConfig:
    # All configuration parameters as dataclass
```

### Key Components

#### 1. Position Processing Pipeline
```python
def _process_position(self, position: dict, position_index: int) -> dict:
    start_time = datetime.now()
    try:
        # 1. Generate prompt
        prompt = self.prompt_generator.generate(position['fen'], position['pgn'], 
                                               self.config.strategy)
        
        # 2. Send to LLM
        response = self.llm_client.send_message(prompt, 
                                               max_tokens=self.config.max_tokens,
                                               temperature=self.config.temperature)
        
        # 3. Parse move
        parse_result = self.parser.parse_chess_response(response.response_text)
        
        # 4. Evaluate move
        if parse_result.success:
            eval_result = evaluate_move(position['fen'], parse_result.move)
        else:
            eval_result = MoveEvaluation(is_legal=False, 
                                       error_message="Parse failed")
        
        # 5. Build result record
        return self._build_result_record(position, position_index, prompt, 
                                       response, parse_result, eval_result, 
                                       start_time)
    except Exception as e:
        return self._build_error_record(position, position_index, e, start_time)
```

#### 2. Thread-Safe Result Writing
```python
class ResultWriter:
    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        self.write_queue = queue.Queue()
        self.writer_thread = threading.Thread(target=self._writer_worker)
        self.writer_thread.start()
    
    def write_result(self, result: dict):
        self.write_queue.put(result)
    
    def _writer_worker(self):
        with open(self.jsonl_path, 'a') as f:
            while True:
                try:
                    result = self.write_queue.get(timeout=1)
                    if result is None:  # Shutdown signal
                        break
                    f.write(json.dumps(result) + '\n')
                    f.flush()
                    self.write_queue.task_done()
                except queue.Empty:
                    continue
```

#### 3. Progress Tracking
- Use separate thread to periodically update run metadata
- Calculate legal move rate, average expectation change in real-time  
- Log progress every N positions or M seconds
- Handle graceful shutdown with final stats

#### 4. Model Client Integration
```python
def _create_llm_client(model: str) -> BaseClient:
    if model.startswith('gemini'):
        return GeminiClient(logger, model)
    elif model.startswith('claude'):
        return AnthropicClient(logger, model)
    elif model.startswith('gpt') or model.startswith('o'):
        return OpenAIClient(logger, model)
    else:
        raise ValueError(f"Unknown model: {model}")
```

#### 5. Prompt Strategy Integration
- Extend `prompts.py` to support multiple strategies
- Strategy registry: `{'fen_basic': generate_fen_prompt, 'json_board': generate_json_prompt}`
- Each strategy function: `(fen, pgn, player) -> prompt_text`

## Testing Strategy

### Unit Tests
- Test result record building with all data combinations
- Test error handling for each pipeline stage
- Test thread-safe writing under concurrent load
- Test metadata calculation and updates

### Integration Tests  
- End-to-end run with small position set (5 positions)
- Multi-worker run with known positions and expected results
- API failure simulation and recovery testing
- Interruption and resume testing

### Performance Tests
- Measure throughput with different worker counts
- Memory usage monitoring during long runs
- File I/O performance with high-frequency writes

## File Organization
```
prompt_eval/
├── run_evaluation.py       # CLI entry point
├── evaluation_runner.py    # Core runner class
├── result_writer.py        # Thread-safe writing
├── runner_config.py        # Configuration dataclass
├── test_runner.py          # Integration tests
├── results/                # Output directory
│   ├── eval_results_*.jsonl
│   └── run_metadata_*.json
└── ...
```

## Success Criteria
1. **Correctness**: Results match single-threaded baseline
2. **Performance**: 4x speedup with 4 workers (accounting for API latency)
3. **Reliability**: Handle API failures, interruptions, malformed data
4. **Usability**: Clear progress indication, helpful error messages
5. **Testability**: Comprehensive test coverage for all components