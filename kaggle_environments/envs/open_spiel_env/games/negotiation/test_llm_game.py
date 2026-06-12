"""Run a full Negotiation game with LLM agents for local integration testing.

Usage:
    GEMINI_API_KEY=... uv run python -m \\
        kaggle_environments.envs.open_spiel_env.games.negotiation.test_llm_game
"""

import json

from kaggle_environments.local_harness_runner import run_llm_game

if __name__ == "__main__":
    env = run_llm_game(
        "open_spiel_negotiation",
        caller_file=__file__,
        configuration={"includeLegalActions": True},
    )
    if env is not None:
        try:
            parsed = json.loads(env.state[0].observation.observationString)
            print(
                f"  agreement_reached={parsed.get('agreement_reached')} "
                f"winner={parsed.get('winner')}"
            )
        except (TypeError, ValueError):
            pass
