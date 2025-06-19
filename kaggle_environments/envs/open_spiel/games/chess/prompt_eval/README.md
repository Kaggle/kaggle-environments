# Chess Prompt Evaluation Tools

This directory contains LLM client libraries for evaluating chess prompts across different AI providers.

## Setup

### Install Dependencies

Install the required packages for this directory only:

```bash
cd kaggle_environments/envs/open_spiel/games/chess/prompt_eval/
pip install -r requirements.txt
```

This installs the LLM dependencies locally without affecting the main kaggle-environments repo.

### Set API Keys

**Option 1: .env file (Recommended)**

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your actual API keys:
   ```bash
   # The .env file is already in .gitignore so it won't be committed
   ANTHROPIC_API_KEY=your_actual_anthropic_key
   OPENAI_API_KEY=your_actual_openai_key
   GEMINI_API_KEY=your_actual_gemini_key
   ```

**Option 2: Environment variables**

Set the environment variables in your shell:
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key" 
export GEMINI_API_KEY="your-gemini-key"
```

## Usage

### Basic Example

```python
from kaggle_environments.envs.open_spiel.games.chess.prompt_eval import get_logger
from kaggle_environments.envs.open_spiel.games.chess.prompt_eval.llm import AnthropicClient

# Create logger
logger = get_logger()

# Initialize client
client = AnthropicClient(logger)

# Send a chess prompt
response = client.send_message(
    prompt="Analyze this chess position: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    system_prompt="You are a chess grandmaster.",
    max_tokens=2000,  # Higher limits needed for reasoning models (o3, Gemini 2.5 Pro)
    temperature=0.7
)

if response.is_success:
    print(f"Response: {response.response_text}")
    print(f"Tokens used: {response.prompt_tokens + response.completion_tokens}")
else:
    print(f"Error: {response.error}")
```

### Important Notes for Reasoning Models

**OpenAI o3** and **Gemini 2.5 Pro** are reasoning models that use internal thinking tokens before generating responses. You may need higher `max_tokens` limits (2000-10000) to get meaningful output, as they use many tokens for internal reasoning that don't appear in the final response.

### Available Clients

- `AnthropicClient` - For Claude models
- `OpenAIClient` - For GPT models
- `GeminiClient` - For Google Gemini models

### Test the Setup

Run the example script to test your setup:

```bash
cd kaggle_environments/envs/open_spiel/games/chess/prompt_eval/
python3 example_usage.py
```

## Files

- `llm/` - LLM client implementations
- `logger.py` - Simple logging utilities
- `config.py` - Configuration and API key management
- `example_usage.py` - Example usage script
- `chess_positions.txt` - Chess positions for evaluation
- `dump_chess_positions.py` - Script to extract chess positions