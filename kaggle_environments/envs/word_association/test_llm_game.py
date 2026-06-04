"""Run a full Word Association game with LLM agents for local integration testing.

Usage:
    GEMINI_API_KEY=... uv run python -m \\
        kaggle_environments.envs.word_association.test_llm_game
"""

import os

from kaggle_environments.local_harness_runner import run_llm_game

if __name__ == "__main__":
    env = run_llm_game(
        "word_association",
        caller_file=__file__,
        agent_module="harness/main.py",
        num_agents=4,
        configuration={"games_per_episode": 1},
    )
    if env is not None:
        rewards = [state.reward for state in env.state]
        if rewards[0] > 0:
            print("WINNER: Team Blue")
        elif rewards[2] > 0:
            print("WINNER: Team Yellow")
        else:
            print("Result: Tie or Error")

        # Also save an HTML render for visual inspection.
        dir_path = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(dir_path, "word_association_replay.html")
        with open(html_path, "w") as f:
            f.write(env.render(mode="html"))
        print(f"Saved HTML render to {html_path}")
