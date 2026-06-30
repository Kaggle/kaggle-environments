"""Run a full Word Art game with LLM agents for local integration testing.

Usage:
    GEMINI_API_KEY=... uv run python -m \\
        kaggle_environments.envs.word_art.test_llm_game

By default plays an 8-round game with 4 LLM agents (2v2). The saved replay
goes to visualizer/default/replays/test-replay.json so you can immediately
inspect it in the web visualizer.
"""

import os

from kaggle_environments.local_harness_runner import run_llm_game

if __name__ == "__main__":
    env = run_llm_game(
        "word_art",
        caller_file=__file__,
        agent_module="harness/main.py",
        num_agents=4,
        configuration={"num_rounds": 4},
    )
    if env is not None:
        rewards = [state.reward for state in env.state]
        blue, yellow = rewards[0], rewards[2]
        if blue > yellow:
            print(f"WINNER: Team Blue ({blue}-{yellow})")
        elif yellow > blue:
            print(f"WINNER: Team Yellow ({yellow}-{blue})")
        else:
            print(f"TIE: {blue}-{yellow}")

        # Also save an HTML render for visual inspection.
        dir_path = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(dir_path, "word_art_replay.html")
        with open(html_path, "w") as f:
            f.write(env.render(mode="html"))
        print(f"Saved HTML render to {html_path}")
