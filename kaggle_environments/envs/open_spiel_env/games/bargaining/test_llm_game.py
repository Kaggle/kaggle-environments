"""Run a full Bargaining game with LLM agents for local integration testing.

Usage:
    GEMINI_API_KEY=... uv run python -m \\
        kaggle_environments.envs.open_spiel_env.games.bargaining.test_llm_game
"""

import json

from kaggle_environments.local_harness_runner import run_llm_game

if __name__ == "__main__":
    env = run_llm_game(
        "open_spiel_bargaining",
        caller_file=__file__,
        configuration={"includeLegalActions": True},
    )
    if env is not None:
        try:
            parsed = json.loads(env.state[0].observation.observationString)
            returns = parsed.get("returns")
            print(f"  agreement_reached={parsed.get('agreement_reached')} returns={returns}")
        except (TypeError, ValueError):
            pass
