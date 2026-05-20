"""Verify that the migrated connect_four harness produces the same prompts as the old one.

Walks an old-format Kaggle connect_four replay JSON, recreates the game move-by-move,
and at each LLM call regenerates the prompt with the new harness's
``generate_prompt`` -- including rethink retries -- then compares against the
prompt stored in the old replay's ``generate_returns``.

Usage:
    python3 -m kaggle_environments.envs.open_spiel_env.games.connect_four.scripts.verify_harness_prompts \\
        /path/to/old_replay.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from typing import Any

import pyspiel

from kaggle_environments.envs.open_spiel_env.games.connect_four.harness import (
    generate_prompt,
    parse_response,
)


def _extract_old_prompt(generate_return_json: str) -> str:
    """Pull the user prompt text out of an old-format generate_returns entry."""
    payload = json.loads(generate_return_json)
    messages = payload["request_for_logging"]["messages"]
    # The harness sent a single user message with one text content block.
    return messages[0]["content"][0]["text"]


def _extract_old_response(generate_return_json: str) -> str:
    """Pull the model response text out of an old-format generate_returns entry."""
    payload = json.loads(generate_return_json)
    return payload.get("main_response", "")


def _build_observation(state: pyspiel.State) -> dict[str, Any]:
    """Build a harness-style observation dict from a pyspiel connect_four state."""
    game = state.get_game()
    player_id = state.current_player()
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "currentPlayer": player_id,
        "serializedGameAndState": pyspiel.serialize_game_and_state(game, state),
    }


def _find_move_for_step(step: list[dict]) -> tuple[int, dict] | None:
    """Return (agent_idx, agent_record) for the agent that played this step.

    The acting agent has a non-empty ``generate_returns`` and a populated
    ``actionString``. Returns ``None`` if no such agent (e.g. setup step).
    """
    for j, agent in enumerate(step):
        action = agent.get("action")
        if not isinstance(action, dict):
            continue
        gr = action.get("generate_returns")
        astr = action.get("actionString")
        if gr and astr:
            return j, agent
    return None


def _diff(expected: str, actual: str) -> str:
    """Return a unified diff between two strings."""
    return "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile="old_prompt",
            tofile="new_prompt",
            n=2,
        )
    )


def verify(replay_path: str, verbose: bool = False) -> int:
    """Verify all prompts in the replay. Returns the number of mismatches."""
    with open(replay_path) as f:
        replay = json.load(f)

    if replay.get("name") != "open_spiel_connect_four":
        print(
            f"WARNING: replay name is {replay.get('name')!r}, "
            "expected 'open_spiel_connect_four'",
            file=sys.stderr,
        )

    # Honour non-default game params (e.g. Connect 5 with rows/columns/x_in_row).
    config_params = (
        replay.get("configuration", {}).get("openSpielGameParameters") or {}
    )
    game_params = {
        k: v for k, v in config_params.items()
        if k in {"rows", "columns", "x_in_row"}
    }
    game = pyspiel.load_game("connect_four", game_params)
    state = game.new_initial_state()

    total_prompts = 0
    mismatches = 0
    moves_verified = 0

    for step_idx, step in enumerate(replay["steps"]):
        found = _find_move_for_step(step)
        if found is None:
            continue
        agent_idx, agent = found
        action = agent["action"]
        action_str: str = action["actionString"]
        generate_returns: list[str] = action["generate_returns"]

        # Build the observation that the harness would have seen.
        observation = _build_observation(state)
        legal_actions = state.legal_actions()
        legal_action_strings = [state.action_to_string(a) for a in legal_actions]

        previous_response: str | None = None
        previous_action: str | None = None

        for attempt_idx, gr_entry in enumerate(generate_returns):
            total_prompts += 1
            try:
                old_prompt = _extract_old_prompt(gr_entry)
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                print(
                    f"step {step_idx} agent {agent_idx} attempt {attempt_idx}: "
                    f"could not extract old prompt: {e}",
                    file=sys.stderr,
                )
                mismatches += 1
                continue

            new_prompt = generate_prompt(
                observation,
                [],  # move_history list is unused by the connect_four harness
                previous_response=previous_response,
                previous_action=previous_action,
            )

            if new_prompt == old_prompt:
                if verbose:
                    print(
                        f"step {step_idx} agent {agent_idx} attempt {attempt_idx}: OK"
                    )
            else:
                mismatches += 1
                print(
                    f"\n=== MISMATCH at step {step_idx} agent {agent_idx} "
                    f"attempt {attempt_idx} (action={action_str!r}) ==="
                )
                print(_diff(old_prompt, new_prompt))

            # Set up retry context: simulate what core_harness would pass next.
            old_response = _extract_old_response(gr_entry)
            parse_result = parse_response(old_response, legal_action_strings)
            previous_response = old_response
            previous_action = parse_result.raw_action

        # Apply the final action and advance state. Connect Four action
        # strings are "<player><column>" (e.g. "x3"), so strip the leading
        # player char to recover the column index.
        try:
            column = int(action_str[1:])
            state.apply_action(column)
        except Exception as e:
            print(
                f"\nERROR: could not apply action {action_str!r} at step {step_idx}: {e}",
                file=sys.stderr,
            )
            print("Aborting state replay; further prompts may be wrong.", file=sys.stderr)
            return mismatches + 1
        moves_verified += 1

    print(
        f"\nVerified {moves_verified} moves / {total_prompts} prompts; "
        f"{mismatches} mismatch(es)."
    )
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify migrated connect_four harness prompts against an old replay."
    )
    parser.add_argument("replay", help="Path to old-format connect_four replay JSON")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print a line per matching prompt"
    )
    args = parser.parse_args()

    mismatches = verify(args.replay, verbose=args.verbose)
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
