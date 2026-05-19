"""Run a full chess game with LLM agents for local integration testing.

Usage:
    GEMINI_API_KEY=... python3 -m \\
        kaggle_environments.envs.open_spiel_env.games.chess.test_llm_game

    # Or with a custom model and move limit:
    GEMINI_API_KEY=... python3 -m \\
        kaggle_environments.envs.open_spiel_env.games.chess.test_llm_game \\
        --model gemini-2.5-flash --max-moves 20
"""

import argparse
import json
import os

from kaggle_environments import make


def run_llm_game(model: str, max_moves: int) -> None:
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("Set GEMINI_API_KEY or OPENAI_API_KEY to run this test.")
        print("Example: export GEMINI_API_KEY=your_key")
        return

    if "MODEL_NAME" not in os.environ:
        os.environ["MODEL_NAME"] = model
    if "MODEL_PROXY_KEY" not in os.environ:
        os.environ["MODEL_PROXY_KEY"] = os.environ.get("GEMINI_API_KEY", os.environ.get("OPENAI_API_KEY", "dummy"))
    if "MODEL_PROXY_URL" not in os.environ:
        os.environ["MODEL_PROXY_URL"] = "dummy_url"

    env = make(
        "open_spiel_chess",
        {"episodeSteps": max_moves * 2 + 100},
        debug=True,
    )

    dir_path = os.path.dirname(os.path.abspath(__file__))
    agent_path = os.path.join(dir_path, "harness.py")

    print(f"Running chess game with LLM agents (model={os.environ['MODEL_NAME']}, max_moves={max_moves})...")
    env.run([agent_path, agent_path])

    print("\n=== GAME STEPS ===")
    for idx, step in enumerate(env.steps):
        for agent_idx, agent_state in enumerate(step):
            action = agent_state.action
            status = agent_state.status
            action_str = action.get("actionString", "") if isinstance(action, dict) else ""
            if status == "ACTIVE" and action_str:
                player = "White" if agent_idx == 1 else "Black"
                print(f"  Step {idx}: {player} plays {action_str}")

    print("\n=== RESULTS ===")
    for i, state in enumerate(env.state):
        player = "White" if i == 1 else "Black"
        print(f"  {player} (agent {i}): status={state.status}, reward={state.reward}")

    replay_dir = os.path.join(dir_path, "visualizer", "default", "replays")
    os.makedirs(replay_dir, exist_ok=True)
    replay_path = os.path.join(replay_dir, "test-replay.json")
    with open(replay_path, "w") as f:
        json.dump(env.toJSON(), f)
    print(f"\nReplay saved to {replay_path}")


def main():
    parser = argparse.ArgumentParser(description="Run a chess game with LLM agents")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME", "gemini-2.5-flash"),
        help="LLM model name (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=40,
        help="Maximum number of moves per side (default: 40)",
    )
    args = parser.parse_args()
    run_llm_game(args.model, args.max_moves)


if __name__ == "__main__":
    main()
