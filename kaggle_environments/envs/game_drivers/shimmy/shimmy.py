"""Dynamic Shimmy/OpenSpiel environment for kaggle-environments.

This environment can run ANY OpenSpiel game by specifying the game name via the `info` parameter.

Usage:
    # Via Python API
    env = make("shimmy", info={"game_name": "connect_four"})
    env.run(["random", "random"])

    # Via CLI
    python -m kaggle_environments.main run --environment shimmy --info '{"game_name": "connect_four"}' --agents random random

Supported games include all OpenSpiel games available through Shimmy:
- tic_tac_toe
- connect_four
- chess
- go
- gin_rummy
- And many more...

Note: This environment uses a default renderer. For game-specific visualization,
use the dedicated game modules (e.g., shimmy_tic_tac_toe, shimmy_connect_four).
"""

import json
import os
from typing import Any

import kaggle_evaluation.core.relay as relay
from remote_game_drivers.core.base_classes import KaggleAgentId
from remote_game_drivers.shimmy_remote_driver.game_driver import OpenSpielGameDriver
from remote_game_drivers.shimmy_remote_driver.remote_agent import RandomAgent, RemoteAgent

# Per-environment state tracking (keyed by env id to avoid test pollution)
_env_state: dict[int, dict] = {}


def _random_agent(obs: dict, config: dict) -> int:
    """Placeholder for kaggle-environments agent registry."""
    return 0


agents = {"random": _random_agent}


class _KaggleCallableAgent(RemoteAgent):
    """Wraps a kaggle-environments callable agent as a RemoteAgent."""

    def __init__(self, callable_agent: Any, configuration: dict | None = None):
        self.callable_agent = callable_agent
        self.configuration = configuration or {}

    def process_turn(self, action_space: Any, observation: Any, info: dict | None = None) -> dict[str, Any]:
        """Entry point called by relay."""
        if hasattr(observation, "tolist"):
            obs_dict = {"observation": observation.tolist()}
        elif hasattr(observation, "__iter__"):
            obs_dict = {"observation": list(observation)}
        else:
            obs_dict = {"observation": observation}

        if info and "action_mask" in info:
            obs_dict["legalActions"] = [i for i, legal in enumerate(info["action_mask"]) if legal]

        action = self.callable_agent(obs_dict, self.configuration)
        return {"action": action}

    def choose_action(self, action_space: Any, observation: Any, info: dict | None = None) -> Any:
        result = self.process_turn(action_space, observation, info)
        return result["action"]


def _is_http_url(s: Any) -> bool:
    if not isinstance(s, str):
        return False
    return s.startswith("http://") or s.startswith("https://")


def _get_agent_instances(env, environment_name: str) -> list[RemoteAgent]:
    """Get RemoteAgent instances for the agents passed to env.run()."""
    from kaggle_environments.envs.game_drivers.protobuf_agent import ProtobufAgent

    agent_specs = env.info.get("_agent_specs", [])

    if not agent_specs:
        raise ValueError("No agents provided. Agents must be passed to env.run() for game driver environments.")

    agent_instances = []
    for i, spec in enumerate(agent_specs):
        if spec == "random":
            agent_instances.append(RandomAgent())
        elif _is_http_url(spec):
            protobuf_agent = ProtobufAgent(spec, environment_name=environment_name)
            agent_instances.append(_KaggleCallableAgent(protobuf_agent, dict(env.configuration)))
        elif callable(spec):
            agent_instances.append(_KaggleCallableAgent(spec, dict(env.configuration)))
        else:
            raise ValueError(
                f"Unsupported agent type for agent {i}: {type(spec).__name__}. "
                f"Expected 'random', HTTP URL string, or callable."
            )

    return agent_instances


def _run_game_with_driver(
    agent_instances: list,
    game_name: str,
    episode_steps: int | None = None,
) -> dict[KaggleAgentId, float | int]:
    """Run the game using OpenSpielGameDriver."""
    agent_ids = [KaggleAgentId(f"agent_{i}") for i in range(len(agent_instances))]

    relay.set_allowed_modules(["remote_game_drivers.gymnasium_remote_driver.serialization"])

    servers = []
    relay_clients: dict[KaggleAgentId, relay.Client] = {}

    for agent, agent_id in zip(agent_instances, agent_ids):
        server, port = relay.define_server(agent.process_turn)
        server.start()
        servers.append(server)
        relay_clients[agent_id] = relay.Client(port=port)

    try:
        driver_config = {
            "agent_ids": list(agent_ids),
            "game_config": {},
        }

        driver = OpenSpielGameDriver(driver_config=driver_config, relay_clients=relay_clients)
        results = driver.run_game(game_name=game_name)

        for client in relay_clients.values():
            client.close()
        return results
    finally:
        for server in servers:
            server.stop(0)
        relay.set_allowed_modules(None)


def interpreter(state, env):
    """
    Dynamic interpreter that reads game_name from env.info.

    The game name must be specified via info={"game_name": "tic_tac_toe"}.
    """
    global _env_state

    env_id = id(env)
    if env_id not in _env_state:
        _env_state[env_id] = {"complete": False, "results": {}}

    env_data = _env_state[env_id]

    # Detect new episode: if step is 0 and we were previously complete, reset
    current_step = state[0].observation.step if state else 0
    if current_step == 0 and env_data["complete"]:
        env_data["complete"] = False
        env_data["results"] = {}

    if env_data["complete"]:
        return state

    agent_specs = env.info.get("_agent_specs", [])
    if not agent_specs:
        return state

    # Get game name from info
    game_name = env.info.get("game_name")
    if not game_name:
        raise ValueError("No game_name specified in info. Use: make('shimmy', info={'game_name': 'tic_tac_toe'})")

    agent_instances = _get_agent_instances(env, "shimmy")
    episode_steps = getattr(env.configuration, "episodeSteps", None)

    try:
        results = _run_game_with_driver(agent_instances, game_name, episode_steps)
        env_data["complete"] = True
        env_data["results"] = results

        for agent_id, reward in results.items():
            idx = int(str(agent_id).split("_")[1])
            state[idx].reward = reward
            state[idx].status = "DONE"

        return state
    except Exception as e:
        for i in range(len(state)):
            state[i].status = f"ERROR: {e}"
            state[i].reward = 0
        env_data["complete"] = True
        return state


def renderer(state, env):
    """Default renderer - shows game state as string."""
    game_name = env.info.get("game_name", "unknown")
    return f"Shimmy game: {game_name}\nState: {state}"


def html_renderer():
    """Default HTML renderer."""
    return "<div>Shimmy Game</div>"


# Load specification
dirpath = os.path.dirname(__file__)
jsonpath = os.path.abspath(os.path.join(dirpath, "..", "shimmy_game.json"))
with open(jsonpath) as f:
    specification = json.load(f)

specification["name"] = "shimmy"
specification["title"] = "Shimmy (Dynamic)"
specification["description"] = "Dynamic OpenSpiel game environment - specify game via info={'game_name': '...'}"
