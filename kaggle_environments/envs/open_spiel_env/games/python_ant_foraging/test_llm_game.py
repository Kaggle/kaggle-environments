"""Run a full Python Ant Foraging game with LLM agents for local integration testing.

Usage:
    GEMINI_API_KEY=... uv run python -m \\
        kaggle_environments.envs.open_spiel_env.games.python_ant_foraging.test_llm_game
"""

from kaggle_environments.local_harness_runner import run_llm_game

if __name__ == "__main__":
    run_llm_game(
        "open_spiel_python_ant_foraging",
        caller_file=__file__,
        configuration={"includeLegalActions": True, "actTimeout": 120},
        replay_filename="llm-replay.json",
    )
