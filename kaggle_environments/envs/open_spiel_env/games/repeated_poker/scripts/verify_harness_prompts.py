"""Verify that the migrated repeated_poker harness produces the same prompts as the old one.

Walks an old-format Kaggle repeated_poker replay JSON, replaying the game move
by move from ``new_initial_state`` (consuming preset chance actions from the
replay's ``presetHands`` config). At each player decision, regenerates the
prompt with the new harness's ``generate_prompt`` -- including rethink retries
-- and diffs it against the prompt stored in the old replay's
``generate_returns``.

Forward-replay is preferred over per-step ``pyspiel.deserialize_game_and_state``
because deserialization is O(history) and the cost compounds quadratically over
a long session.

Usage:
    python3 -m kaggle_environments.envs.open_spiel_env.games.repeated_poker.scripts.verify_harness_prompts \\
        /path/to/old_replay.json

    # Or fetch a sample replay first:
    curl -o /tmp/old.json https://www.kaggleusercontent.com/episodes/77159528.json
    python3 -m kaggle_environments.envs.open_spiel_env.games.repeated_poker.scripts.verify_harness_prompts \\
        /tmp/old.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys

import pyspiel

from kaggle_environments.envs.open_spiel_env.games.repeated_poker import harness


def _extract_old_prompt(generate_return_json: str) -> str:
    """Pull the user prompt text out of an old-format generate_returns entry."""
    payload = json.loads(generate_return_json)
    messages = payload["request_for_logging"]["messages"]
    return messages[0]["content"][0]["text"]


def _extract_old_response(generate_return_json: str) -> str:
    """Pull the model response text out of an old-format generate_returns entry."""
    payload = json.loads(generate_return_json)
    return payload.get("main_response", "")


def _find_move_for_step(step: list[dict]) -> tuple[int, dict] | None:
    """Return ``(agent_idx, agent_record)`` for the agent that acted this step."""
    for j, agent in enumerate(step):
        action = agent.get("action")
        if not isinstance(action, dict):
            continue
        gr = action.get("generate_returns")
        astr = action.get("actionString")
        if gr and astr:
            return j, agent
    return None


def _action_id_from_string(state: pyspiel.State, action_str: str) -> int:
    """Look up the action id for ``action_str`` in the state's legal actions."""
    player = state.current_player()
    for a in state.legal_actions():
        if state.action_to_string(player, a) == action_str:
            return a
    raise ValueError(f"Action {action_str!r} not in legal actions for player {player}")


def _drain_chance_actions(
    state: pyspiel.State,
    preset_hands: list[list[int]],
    next_index: list[int],
) -> None:
    """Advance through chance nodes, consuming the next preset card each time.

    Mirrors ``_get_preset_chance_action`` in ``open_spiel_env.py``.
    """
    while not state.is_terminal() and state.is_chance_node():
        hand_idx = len(state.acpc_hand_histories())
        if hand_idx >= len(preset_hands):
            raise ValueError(f"Ran out of preset hands at hand_idx={hand_idx}")
        pos = next_index[hand_idx]
        card = preset_hands[hand_idx][pos]
        next_index[hand_idx] = pos + 1
        state.apply_action(card)


def _diff(expected: str, actual: str) -> str:
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
    with open(replay_path) as f:
        replay = json.load(f)

    if replay.get("name") != "open_spiel_repeated_poker":
        print(
            f"WARNING: replay name is {replay.get('name')!r}, expected 'open_spiel_repeated_poker'",
            file=sys.stderr,
        )

    game = pyspiel.load_game(replay["configuration"]["openSpielGameString"])
    state = game.new_initial_state()
    preset_hands: list[list[int]] = replay["configuration"]["presetHands"]
    next_index = [0] * len(preset_hands)

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

        _drain_chance_actions(state, preset_hands, next_index)

        if state.is_terminal():
            print(
                f"step {step_idx}: state is terminal but replay still has actions to play; aborting",
                file=sys.stderr,
            )
            return mismatches + 1

        if state.current_player() != agent_idx:
            print(
                f"step {step_idx} agent {agent_idx}: state.current_player()"
                f" = {state.current_player()} disagrees with replay; aborting",
                file=sys.stderr,
            )
            return mismatches + 1

        previous_response: str | None = None

        for attempt_idx, gr_entry in enumerate(generate_returns):
            total_prompts += 1
            try:
                old_prompt = _extract_old_prompt(gr_entry)
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                print(
                    f"step {step_idx} agent {agent_idx} attempt {attempt_idx}: could not extract old prompt: {e}",
                    file=sys.stderr,
                )
                mismatches += 1
                continue

            new_prompt = harness.generate_prompt_from_state(state, previous_response=previous_response)

            if new_prompt == old_prompt:
                if verbose:
                    print(f"step {step_idx} agent {agent_idx} attempt {attempt_idx}: OK")
            else:
                mismatches += 1
                print(
                    f"\n=== MISMATCH at step {step_idx} agent {agent_idx} "
                    f"attempt {attempt_idx} (action={action_str!r}) ==="
                )
                print(_diff(old_prompt, new_prompt))

            previous_response = _extract_old_response(gr_entry)
            # parse_response_with_state is only needed if subsequent attempts
            # use raw_action -- the upstream RETHINK_REPEATED_POKER strategy
            # does not, so we can skip it entirely.

        try:
            action_id = _action_id_from_string(state, action_str)
        except ValueError as e:
            print(
                f"\nERROR: could not apply action {action_str!r} at step {step_idx}: {e}",
                file=sys.stderr,
            )
            return mismatches + 1
        state.apply_action(action_id)
        moves_verified += 1

    print(f"\nVerified {moves_verified} moves / {total_prompts} prompts; {mismatches} mismatch(es).")
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Verify migrated repeated_poker harness prompts against an old replay.")
    )
    parser.add_argument("replay", help="Path to old-format repeated_poker replay JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print a line per matching prompt")
    args = parser.parse_args()

    mismatches = verify(args.replay, verbose=args.verbose)
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
