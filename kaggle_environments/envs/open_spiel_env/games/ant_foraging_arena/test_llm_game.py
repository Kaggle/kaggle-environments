"""Run a full Ant Foraging Arena game with LLM agents for local integration testing.

Usage:
    GEMINI_API_KEY=... uv run python -m \\
        kaggle_environments.envs.open_spiel_env.games.ant_foraging_arena.test_llm_game
"""

from kaggle_environments.local_harness_runner import run_llm_game

if __name__ == "__main__":
    run_llm_game(
        "open_spiel_ant_foraging_arena",
        caller_file=__file__,
        configuration={"includeLegalActions": True, "actTimeout": 120},
        replay_filename="llm-replay.json",
    )
