"""Run a full Havannah game with LLM agents for local integration testing.

Usage:
    GEMINI_API_KEY=... uv run python -m \\
        kaggle_environments.envs.open_spiel_env.games.havannah.test_llm_game
"""

from kaggle_environments.local_harness_runner import run_llm_game

if __name__ == "__main__":
    # Smaller board for a quicker local test.
    run_llm_game(
        "open_spiel_havannah",
        caller_file=__file__,
        configuration={
            "includeLegalActions": True,
            "openSpielGameParameters": {"board_size": 5},
        },
    )
