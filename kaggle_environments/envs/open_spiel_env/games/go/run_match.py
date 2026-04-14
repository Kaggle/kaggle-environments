"""Debug script: drive a short Go match through the LLM harness.

Calls `harness.agent_fn` on every step — including the initial Kaggle
setup step — to mirror what `env.run()` does, and logs each prompt and
response so illegal-move failures are easy to diagnose.

Usage:
    GEMINI_API_KEY=... .venv/bin/python -m \\
        kaggle_environments.envs.open_spiel_env.games.go.run_match \\
        --num-moves 5 --model gemini-2.5-flash
"""

import argparse
import os
import sys

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env
from kaggle_environments.envs.open_spiel_env.games.go import harness


def _wrap_litellm_with_logging() -> None:
    """Replace harness.litellm.completion with a version that prints I/O."""
    original = harness.litellm.completion
    call_idx = 0

    def logged_completion(*args, **kwargs):
        nonlocal call_idx
        call_idx += 1
        messages = kwargs.get("messages", [])
        prompt = messages[-1]["content"] if messages else ""
        print(f"\n----- LLM call #{call_idx}: prompt -----")
        print(prompt)
        resp = original(*args, **kwargs)
        print(f"----- LLM call #{call_idx}: response -----")
        print(resp.choices[0].message.content)
        print("-" * 40)
        return resp

    harness.litellm.completion = logged_completion


def _collect_submissions(env) -> list[dict]:
    return [
        harness.agent_fn(s["observation"], {})
        if s["status"] == "ACTIVE"
        else {"submission": -1}
        for s in env.state
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-moves", type=int, default=5)
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--komi", type=float, default=7.5)
    parser.add_argument(
        "--model", default=os.environ.get("MODEL_NAME", "gemini-2.5-flash")
    )
    args = parser.parse_args()

    if "GEMINI_API_KEY" not in os.environ:
        sys.exit("GEMINI_API_KEY must be set in the environment.")

    os.environ["MODEL_NAME"] = args.model
    os.environ.setdefault("MODEL_PROXY_KEY", "unused")
    os.environ.setdefault("MODEL_PROXY_URL", "dummy_url")

    open_spiel_env._register_game_envs(["go"])
    env = make(
        "open_spiel_go",
        {
            "openSpielGameParameters": {
                "board_size": args.board_size,
                "komi": args.komi,
            }
        },
        debug=True,
    )
    env.reset()
    _wrap_litellm_with_logging()

    moves_played = 0
    while moves_played < args.num_moves:
        is_setup_step = len(env.steps) == 1
        submissions = _collect_submissions(env)

        if is_setup_step:
            print("\n========== Setup step ==========")
            print(f">>> Submissions: {submissions}")
        else:
            moves_played += 1
            played = harness._MOVE_HISTORY[-1] if harness._MOVE_HISTORY else "?"
            print(f"\n========== Move {moves_played}: {played} ==========")

        env.step(submissions)
        if env.done:
            print("\nGame finished.")
            break

    print("\n========== Final board ==========")
    print(env.state[0]["observation"].get("observationString", "(no observation)"))


if __name__ == "__main__":
    main()
