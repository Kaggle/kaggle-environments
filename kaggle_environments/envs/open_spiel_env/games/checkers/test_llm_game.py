"""Run a full Checkers game with LLM agents for local integration testing.

Usage:
    GEMINI_API_KEY=... uv run python -m \\
        kaggle_environments.envs.open_spiel_env.games.checkers.test_llm_game
"""

import json
import os

from kaggle_environments import make


def run_llm_game():
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("Set GEMINI_API_KEY or OPENAI_API_KEY to run this test.")
        print("Example: export GEMINI_API_KEY=your_key")
        return

    if "MODEL_NAME" not in os.environ:
        os.environ["MODEL_NAME"] = "gemini-2.5-flash"
    if "MODEL_PROXY_KEY" not in os.environ:
        os.environ["MODEL_PROXY_KEY"] = os.environ.get("GEMINI_API_KEY", os.environ.get("OPENAI_API_KEY", "dummy"))
    if "MODEL_PROXY_URL" not in os.environ:
        os.environ["MODEL_PROXY_URL"] = "dummy_url"

    env = make("open_spiel_checkers", debug=True)

    dir_path = os.path.dirname(os.path.abspath(__file__))
    agent_path = os.path.join(dir_path, "harness.py")

    print("Running Checkers game with LLM agents...")
    env.run([agent_path, agent_path])

    print("\n=== GAME STEPS ===")
    for idx, step in enumerate(env.steps):
        print(f"--- Step {idx} ---")
        for agent_idx, agent_state in enumerate(step):
            action = agent_state.action
            status = agent_state.status
            print(f"  Agent {agent_idx} ({status}): {action}")

    print("\n=== RESULTS ===")
    for i, state in enumerate(env.state):
        print(f"Agent {i}: status={state.status}, reward={state.reward}")

    replay_dir = os.path.join(dir_path, "visualizer", "default", "replays")
    os.makedirs(replay_dir, exist_ok=True)
    replay_path = os.path.join(replay_dir, "test-replay.json")
    with open(replay_path, "w") as f:
        json.dump(env.toJSON(), f)
    print(f"\nReplay saved to {replay_path}")


if __name__ == "__main__":
    run_llm_game()
