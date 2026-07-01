"""Run a full Shogi game with LLM agents for local integration testing.

Usage:
    GEMINI_API_KEY=... uv run python -m \\
        kaggle_environments.envs.open_spiel_env.games.shogi.test_llm_game
"""

from kaggle_environments.local_harness_runner import run_llm_game

if __name__ == "__main__":
    run_llm_game("open_spiel_shogi", caller_file=__file__)
