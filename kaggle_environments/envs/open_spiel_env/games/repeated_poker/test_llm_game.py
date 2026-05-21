"""Debug script: drive a short repeated_poker session through the LLM harness.

Usage:
    GEMINI_API_KEY=... uv run python -m \\
        kaggle_environments.envs.open_spiel_env.games.repeated_poker.test_llm_game \\
        --num-hands 2 --model gemini-2.5-flash
"""

import argparse
import os
import sys

import litellm

from kaggle_environments import make
from kaggle_environments.core_harness import create_agent_fn
from kaggle_environments.envs.open_spiel_env import open_spiel_env
from kaggle_environments.envs.open_spiel_env.games.repeated_poker import harness


def _wrap_litellm_with_logging() -> None:
    """Replace litellm.completion with a version that prints I/O."""
    original = litellm.completion
    call_idx = 0

    def logged_completion(*args, **kwargs):
        nonlocal call_idx
        call_idx += 1
        messages = kwargs.get("messages", [])
        prompt = messages[-1]["content"] if messages else ""
        print(f"\n----- LLM call #{call_idx}: prompt -----")
        print(prompt)
        resp = original(*args, **kwargs)
        # When streaming, returned value is an iterator -- pass through.
        if hasattr(resp, "choices"):
            print(f"----- LLM call #{call_idx}: response -----")
            print(resp.choices[0].message.content)
            print("-" * 40)
        return resp

    litellm.completion = logged_completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME", "gemini-2.5-flash"),
    )
    args = parser.parse_args()

    if "GEMINI_API_KEY" not in os.environ and "OPENAI_API_KEY" not in os.environ:
        sys.exit("GEMINI_API_KEY or OPENAI_API_KEY must be set in the environment.")

    os.environ["MODEL_NAME"] = args.model
    os.environ.setdefault(
        "MODEL_PROXY_KEY",
        os.environ.get("GEMINI_API_KEY", os.environ.get("OPENAI_API_KEY", "unused")),
    )
    os.environ.setdefault("MODEL_PROXY_URL", "dummy_url")

    open_spiel_env._register_game_envs(["repeated_poker"])
    env = make(
        "open_spiel_repeated_poker",
        configuration={"setNumHands": args.num_hands},
        debug=True,
    )
    env.reset()
    _wrap_litellm_with_logging()

    agent_fn = create_agent_fn(harness._PokerHarness())

    move_count = 0
    while not env.done:
        submissions = []
        for s in env.state:
            if s["status"] == "ACTIVE":
                result = agent_fn(s["observation"], {})
                submissions.append(result)
            else:
                submissions.append({"submission": -1})

        if any(isinstance(sub, dict) and sub.get("actionString") for sub in submissions):
            move_count += 1
            played = next(
                sub["actionString"]
                for sub in submissions
                if isinstance(sub, dict) and sub.get("actionString")
            )
            print(f"\n========== Move {move_count}: {played} ==========")

        env.step(submissions)

    print("\n========== Final state ==========")
    print(env.state[0]["observation"].get("observationString", "(no observation)"))
    print("\n========== Results ==========")
    for i, s in enumerate(env.state):
        print(f"Agent {i}: status={s['status']}, reward={s['reward']}")


if __name__ == "__main__":
    main()
