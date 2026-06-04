"""Local-debug helper for running an LLM harness end-to-end against an env.

Each game's ``test_llm_game.py`` collapses to a three-line wrapper:

    \"\"\"Run a full Checkers game with LLM agents for local integration testing.\"\"\"
    from kaggle_environments.local_harness_runner import run_llm_game

    if __name__ == "__main__":
        run_llm_game("open_spiel_checkers", caller_file=__file__)

This module is for local development and integration testing only. It is NOT
loaded by Kaggle in production -- the production loader uses the per-call
``GameHarness`` protocol in ``core_harness.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from kaggle_environments import make


_DEFAULT_MODEL = "gemini-2.5-flash"
_REPLAY_SUBDIR = os.path.join("visualizer", "default", "replays")


def _check_api_keys() -> bool:
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return True
    print("Set GEMINI_API_KEY or OPENAI_API_KEY to run this test.")
    print("Example: export GEMINI_API_KEY=your_key")
    return False


def _set_env_defaults(model_override: str | None) -> None:
    if model_override is not None:
        os.environ["MODEL_NAME"] = model_override
    elif "MODEL_NAME" not in os.environ:
        os.environ["MODEL_NAME"] = _DEFAULT_MODEL
    if "MODEL_PROXY_KEY" not in os.environ:
        os.environ["MODEL_PROXY_KEY"] = os.environ.get(
            "GEMINI_API_KEY", os.environ.get("OPENAI_API_KEY", "dummy")
        )
    if "MODEL_PROXY_URL" not in os.environ:
        os.environ["MODEL_PROXY_URL"] = "dummy_url"


def _parse_cli(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an LLM harness end-to-end against a Kaggle environment.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override MODEL_NAME for this run.",
    )
    parser.add_argument(
        "--replay-path",
        default=None,
        help="Override the replay save path (default: <caller_dir>/visualizer/default/replays/<replay_filename>).",
    )
    return parser.parse_args(argv)


def _print_steps_and_results(env: Any) -> None:
    print("\n=== GAME STEPS ===")
    for idx, step in enumerate(env.steps):
        print(f"--- Step {idx} ---")
        for agent_idx, agent_state in enumerate(step):
            print(f"  Agent {agent_idx} ({agent_state.status}): {agent_state.action}")

    print("\n=== RESULTS ===")
    for i, state in enumerate(env.state):
        print(f"Agent {i}: status={state.status}, reward={state.reward}")


def _save_replay(env: Any, replay_path: str) -> None:
    os.makedirs(os.path.dirname(replay_path), exist_ok=True)
    with open(replay_path, "w") as f:
        json.dump(env.toJSON(), f)
    print(f"\nReplay saved to {replay_path}")


def run_llm_game(
    env_name: str,
    *,
    caller_file: str,
    agent_module: str = "harness.py",
    num_agents: int = 2,
    configuration: dict[str, Any] | None = None,
    replay_filename: str = "test-replay.json",
    cli: bool = True,
    argv: list[str] | None = None,
) -> Any:
    """Run a full game with LLM agents and save the replay.

    Args:
        env_name: Env to instantiate (e.g. ``"open_spiel_checkers"``).
        caller_file: Pass ``__file__`` from the caller. The helper resolves
            the agent path and the replay directory relative to this.
        agent_module: Path to the harness module relative to the caller's
            directory. Defaults to ``"harness.py"``; use ``"harness/main.py"``
            for packaged harnesses like word_association.
        num_agents: How many copies of the agent to pass to ``env.run``.
            Defaults to 2; word_association uses 4.
        configuration: Optional dict forwarded to ``make(configuration=...)``.
        replay_filename: Filename for the saved replay; written to
            ``<caller_dir>/visualizer/default/replays/<filename>`` by default.
        cli: When True (default), parse ``--model`` and ``--replay-path`` from
            ``sys.argv``. Set False to disable so the caller can bring its own
            argparse.
        argv: Optional override for ``sys.argv[1:]`` (test seam).

    Returns:
        The ``env`` after the run, so callers can do game-specific
        post-processing (e.g. word_association's winner banner).
        Returns ``None`` and prints a hint if no API key is present.
    """
    if not _check_api_keys():
        return None

    args = _parse_cli(argv if argv is not None else sys.argv[1:]) if cli else _parse_cli([])
    _set_env_defaults(args.model)

    env = make(env_name, configuration=configuration, debug=True)

    caller_dir = os.path.dirname(os.path.abspath(caller_file))
    agent_path = os.path.join(caller_dir, agent_module)
    replay_path = args.replay_path or os.path.join(
        caller_dir, _REPLAY_SUBDIR, replay_filename,
    )

    print(f"Running {env_name} with LLM agents (model={os.environ['MODEL_NAME']})...")
    env.run([agent_path] * num_agents)

    _print_steps_and_results(env)
    _save_replay(env, replay_path)

    return env
