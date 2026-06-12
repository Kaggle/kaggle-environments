"""Run a full Connect Four game with LLM agents for local integration testing.

For an interactive debug runner that logs every LLM prompt/response and lets
you cap the number of moves, see ``debug_match_runner.py`` in the same dir.

Usage:
    GEMINI_API_KEY=... uv run python -m \\
        kaggle_environments.envs.open_spiel_env.games.connect_four.test_llm_game
"""

from kaggle_environments.local_harness_runner import run_llm_game

if __name__ == "__main__":
    run_llm_game("open_spiel_connect_four", caller_file=__file__)
