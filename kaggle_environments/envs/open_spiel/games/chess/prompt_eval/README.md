# Chess Prompt Evaluation System

This system evaluates how well Large Language Models (LLMs) can play chess when given different prompt formats. The goal is to find optimal prompt strategies that maximize LLM chess performance, particularly focusing on **legal move generation rate** and **move quality**.

## Project Overview

### Core Hypothesis
Different ways of representing chess game state (FEN, PGN, JSON, etc.) and different prompting strategies will significantly affect LLM chess performance. This system tests these hypotheses empirically.

### Primary Metrics
1. **Legal Move Rate**: Percentage of moves generated that are actually legal in the given position
2. **Move Quality**: Stockfish evaluation of move strength using Win/Draw/Loss (WDL) expectation changes

### Evaluation Approach
- Test prompts on ~200 chess positions from partial games
- Focus primarily on Gemini models
- Compare different prompt formats rather than head-to-head model comparisons
- Measure both move legality and move strength

## Current Architecture

### Core Components

#### 1. LLM Clients (`clients/`)
- **`base_client.py`**: Abstract base class for all LLM interactions
- **`anthropic_client.py`**: Claude integration with thinking token support
- **`openai_client.py`**: GPT/o3 integration with reasoning model support  
- **`gemini_client.py`**: Google Gemini integration with robust retry logic
- **`llm_response.py`**: Standardized response format across all providers

#### 2. Chess Evaluation (`move_evaluator.py`)
- Move legality validation using python-chess
- Move quality scoring using Stockfish WDL expectations
- Handles both algebraic notation (Nf3, e4) and UCI format (g1f3, e2e4)
- Robust error handling for invalid positions/moves

#### 3. Response Parsing (`response_parser.py`)
- Extracts chess moves from free-form LLM responses
- Supports multiple formats: "Final Answer: Nf3", implicit moves, etc.
- Handles various edge cases and formatting inconsistencies
- Comprehensive test suite covering 20+ response patterns

#### 4. Prompt Generation (`prompts.py`)
- **FEN-based prompts**: Traditional chess notation with move history
- **JSON board representation**: Structured board state with detailed game information
- **No-PGN variants**: Testing impact of move history on performance
- Extensible framework for adding new prompt strategies

#### 5. Position Generation (`dump_chess_positions.py`)
- Generates diverse chess positions from Stockfish self-play
- Creates partial games (5-95% complete) for realistic evaluation scenarios
- Outputs JSONL format with FEN position and move history
- **Filtered corpus**: `filtered_chess_positions.jsonl` with balanced positions (~300cp evaluation)

#### 6. Evaluation Infrastructure
- **`run_evaluation.py`**: Main batch evaluation runner with parallel processing
- **`result_writer.py`**: Structured JSONL output with comprehensive evaluation data
- **`single_eval.py`**: Single position evaluation for testing
- **`results/`**: Directory containing completed evaluation runs

#### 7. Utilities
- **`config.py`**: API key management with .env support
- **`logger.py`**: Structured logging for evaluation runs
- **`test_*.py`**: Unit tests for core components

## Current Status

### ✅ What's Built
- [x] Multi-provider LLM client infrastructure (Anthropic, OpenAI, Gemini)
- [x] Chess move validation and quality evaluation using Stockfish
- [x] Robust response parsing with comprehensive test coverage (20+ patterns)
- [x] Position generation from Stockfish games (filtered to ~300cp balanced positions)
- [x] **Multiple prompt strategies**: FEN-based, JSON board representation, with/without PGN
- [x] **Batch processing system**: Parallel evaluation with `run_evaluation.py`
- [x] **Results storage format**: Structured JSONL output with full evaluation data
- [x] **End-to-end evaluation pipeline**: From positions to analyzed results
- [x] Unit tests for core components

### 📊 **System Ready for Production Use**
The evaluation system is now **fully operational** and has been used to run actual experiments. Recent evaluation runs demonstrate:
- **Legal move rates**: Tracking how often LLMs generate valid chess moves
- **Move quality analysis**: WDL expectation changes using Stockfish evaluation
- **Prompt strategy comparison**: Testing FEN vs JSON board representations
- **Performance data**: Response times, token usage, error handling

## Setup

### Install Dependencies

```bash
cd kaggle_environments/envs/open_spiel/games/chess/prompt_eval/
pip install -r requirements.txt
```

### Set API Keys

Create a `.env` file:
```bash
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  
GEMINI_API_KEY=your_key_here
```

### Generate Test Positions

```bash
# Generate 1000 positions for evaluation (creates balanced positions ~300cp)
python dump_chess_positions.py --num 1000 --out filtered_chess_positions.jsonl
```

## Usage

### Running Batch Evaluations

```bash
# Run evaluation on 50 positions using Gemini with JSON board format
python run_evaluation.py \
    --model gemini \
    --strategy board_json_no_pgn \
    --positions filtered_chess_positions.jsonl \
    --max-positions 50 \
    --max-workers 5

# Results saved to results/eval_results_[id]_[timestamp].jsonl
```

### Single Position Evaluation

```python
from kaggle_environments.envs.open_spiel.games.chess.prompt_eval.clients import GeminiClient
from kaggle_environments.envs.open_spiel.games.chess.prompt_eval.prompts import generate_fen_prompt
from kaggle_environments.envs.open_spiel.games.chess.prompt_eval.response_parser import parse_chess_response
from kaggle_environments.envs.open_spiel.games.chess.prompt_eval.move_evaluator import evaluate_move
from kaggle_environments.envs.open_spiel.games.chess.prompt_eval import get_logger

# Setup
logger = get_logger()
client = GeminiClient(logger)

# Generate prompt for a position
fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
prompt = generate_fen_prompt(fen, "1. e4", "black")

# Get LLM response
response = client.send_message(prompt, max_tokens=2000)

if response.is_success:
    # Parse the move from response
    parse_result = parse_chess_response(response.response_text)
    
    if parse_result.success:
        # Evaluate move quality
        evaluation = evaluate_move(fen, parse_result.move)
        
        print(f"Move: {parse_result.move}")
        print(f"Legal: {evaluation.is_legal}")
        if evaluation.is_legal:
            print(f"Quality (WDL change): {evaluation.expectation_change:.3f}")
    else:
        print(f"Could not parse move: {parse_result.error_message}")
```

### Available Prompt Strategies

- **`fen`**: Traditional FEN notation with PGN move history
- **`board_json`**: Structured JSON board representation with PGN history
- **`board_json_no_pgn`**: JSON board representation without move history

### Analyzing Results

```bash
# View evaluation results
head -n 5 results/eval_results_*.jsonl | jq .

# Check legal move rate and quality statistics
grep '"is_legal": true' results/eval_results_*.jsonl | wc -l
```

### Run Tests

```bash
# Test core components
python test_response_parser.py
python test_move_evaluator.py

# Test single evaluation end-to-end
python single_eval.py
```

## Key Design Decisions

### Why WDL Expectations?
Using Win/Draw/Loss expectation changes rather than centipawns for move evaluation, as recommended by python-chess documentation. This provides more meaningful position comparisons.

### Why Robust Response Parsing?
LLMs produce highly variable response formats. The parser handles 20+ different patterns to maximize successful move extraction.

### Why Multiple LLM Providers?
Different models have different strengths. The unified client interface allows easy comparison while handling provider-specific quirks (thinking tokens, rate limits, etc.).

### Why Stockfish for Ground Truth?
Stockfish provides consistent, high-quality move evaluation for benchmarking LLM move quality.

## Critical Implementation Notes

### Parallelization Required
- Models can take >1 minute per move with reasoning
- Batch processing system must handle concurrent requests
- Need to respect rate limits and manage costs

### Position Diversity
- `dump_chess_positions.py` creates realistic game positions
- Avoids opening book/endgame table positions where engines have perfect play
- 5-95% game completion ensures diverse tactical/strategic scenarios

### Error Handling
- Robust parsing handles malformed LLM responses
- Move evaluation gracefully handles invalid positions/moves  
- Client retry logic handles API failures and rate limits

## File Structure

```
prompt_eval/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                   # API key management
├── logger.py                   # Logging utilities
├── prompts.py                  # Multiple prompt strategies
├── response_parser.py          # LLM response → chess move
├── move_evaluator.py           # Move legality + quality evaluation
├── dump_chess_positions.py     # Position corpus generation
├── run_evaluation.py           # Main batch evaluation runner
├── result_writer.py            # Structured results output
├── single_eval.py              # Single position testing
├── chess_positions.jsonl       # Original position corpus
├── filtered_chess_positions.jsonl # Balanced positions (~300cp)
├── test_*.py                   # Unit tests
├── clients/                    # LLM provider integrations
│   ├── __init__.py
│   ├── base_client.py          # Abstract base class
│   ├── llm_response.py         # Response data structure
│   ├── anthropic_client.py     # Claude integration
│   ├── openai_client.py        # GPT/o3 integration
│   └── gemini_client.py        # Gemini integration
├── docs/                       # Planning and design documents
│   ├── TODO.md
│   ├── board_format.md
│   ├── output_format_design.md
│   └── runner_plan.md
└── results/                    # Completed evaluation runs
    └── eval_results_*.jsonl    # JSONL files with evaluation data
```
