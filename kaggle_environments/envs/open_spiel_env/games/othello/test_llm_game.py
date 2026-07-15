"""Run a full Reversi game with LLM agents for local integration testing."""

from kaggle_environments.local_harness_runner import run_llm_game

if __name__ == "__main__":
    run_llm_game("open_spiel_othello", caller_file=__file__)
