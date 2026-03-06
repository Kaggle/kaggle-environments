import os
import json
from kaggle_environments import make

def run_llm_game():
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("Please set GEMINI_API_KEY or OPENAI_API_KEY in your environment variables to run this test.")
        print("Example: export GEMINI_API_KEY=your_key")
        return
        
    # Inject dummy environment variables so local testing evaluates correctly against the required runner
    if "MODEL_NAME" not in os.environ:
        os.environ["MODEL_NAME"] = "gemini-2.5-flash"
    if "MODEL_PROXY_KEY" not in os.environ:
        # Pass the local api key in so that litellm can use it when it drops the dummy proxy url
        os.environ["MODEL_PROXY_KEY"] = os.environ.get("GEMINI_API_KEY", os.environ.get("OPENAI_API_KEY", "dummy"))
    if "MODEL_PROXY_URL" not in os.environ:
        os.environ["MODEL_PROXY_URL"] = "dummy_url"

    print("Initializing Codenames Game with LLM Agents...")
    env = make("codenames", debug=True)
    
    # We use our litellm harness for all 4 slots.
    # Kaggle environments requires an absolute or properly relative path
    # relative to where the script is run from. Using absolute path here is safest.
    dir_path = os.path.dirname(os.path.abspath(__file__))
    agent_path = os.path.join(dir_path, "harness", "main.py")
    
    # Start the simulation. This will use API calls and may take a moment.
    env.run([agent_path, agent_path, agent_path, agent_path])
    
    print("\n=== GAME STEPS ===")
    for idx, step in enumerate(env.steps):
        print(f"--- Step {idx} ---")
        for agent_idx, agent_state in enumerate(step):
            action = agent_state.action
            status = agent_state.status
            print(f"Agent {agent_idx} ({status}): {action}")
    print("==================\n")
    
    print("Game Finished!")
    
    for i, state in enumerate(env.state):
        print(f"Agent {i} Status: {state.status} | Reward: {state.reward}")
        
    rewards = [state.reward for state in env.state]
    if rewards[0] > 0:
        print("WINNER: Team Red 🟥")
    elif rewards[2] > 0:
        print("WINNER: Team Blue 🟦")
    else:
        print("Result: Tie or Error")

    # Save replay data for the visualizer
    replay_dir = os.path.join(dir_path, "visualizer", "default", "replays")
    os.makedirs(replay_dir, exist_ok=True)
    replay_path = os.path.join(replay_dir, "test-replay.json")
    
    with open(replay_path, "w") as f:
        json.dump(env.toJSON(), f)
    print(f"Saved replay to {replay_path}")

    # Also save HTML render if needed
    html = env.render(mode="html")
    with open(os.path.join(dir_path, "codenames_replay.html"), "w") as f:
        f.write(html)
    print("Saved HTML render to codenames_replay.html")

if __name__ == "__main__":
    run_llm_game()
