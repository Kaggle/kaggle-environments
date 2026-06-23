"""Run a full Ultimate Tic-Tac-Toe game with LLM agents for local integration testing."""

import os
import sys

from kaggle_environments import make
from kaggle_environments.core_harness import create_agent_fn
from kaggle_environments.envs.open_spiel_env import open_spiel_env
from kaggle_environments.envs.open_spiel_env.games.ultimate_tic_tac_toe import harness


class _UltimateTicTacToeHarness:
    """Local GameHarness adapter for the debug runner."""

    def get_legal_moves(self, observation):
        return harness.get_legal_moves(observation)

    def make_prompt(
        self,
        observation,
        move_history,
        previous_response=None,
        previous_action=None,
    ):
        return harness.generate_prompt(observation, move_history, previous_response, previous_action)

    def parse_response(self, response, legal_action_strings, *, observation=None):
        return harness.parse_response(response, legal_action_strings)


def main() -> None:
    if "GEMINI_API_KEY" not in os.environ:
        sys.exit("GEMINI_API_KEY must be set in the environment.")

    os.environ.setdefault("MODEL_NAME", "gemini-2.5-flash")
    os.environ.setdefault("MODEL_PROXY_KEY", os.environ["GEMINI_API_KEY"])
    os.environ.setdefault("MODEL_PROXY_URL", "dummy_url")

    open_spiel_env._register_game_envs(["ultimate_tic_tac_toe"])
    env = make("open_spiel_ultimate_tic_tac_toe", debug=True)
    env.reset()

    agent_fn = create_agent_fn(_UltimateTicTacToeHarness())

    move_count = 0
    while not env.done:
        submissions = []
        for s in env.state:
            if s["status"] == "ACTIVE":
                result = agent_fn(s["observation"], {})
                submissions.append(result)
            else:
                submissions.append({"submission": -1})

        # Print step actions
        for i, sub in enumerate(submissions):
            if isinstance(sub, dict) and sub.get("actionString"):
                move_count += 1
                print(f"\n========== Move {move_count} (Agent {i}): {sub['actionString']} ==========")
                if sub.get("thoughts"):
                    print(f"Thoughts: {sub['thoughts']}")

        env.step(submissions)

    print("\n========== Final board ==========")
    print(env.state[0]["observation"].get("observationString", "(no observation)"))
    print("\n========== Results ==========")
    for i, s in enumerate(env.state):
        print(f"Agent {i}: status={s['status']}, reward={s['reward']}")


if __name__ == "__main__":
    main()
