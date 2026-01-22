"""Base module for Shimmy/OpenSpiel game environments.

This provides all the common functionality for shimmy games.
Individual games only need to specify:
- GAME_NAME: The OpenSpiel game name (e.g., "tic_tac_toe")
- ENV_NAME: The kaggle-environments name (e.g., "shimmy_tic_tac_toe")
- Optionally override renderer() for custom visualization

Usage in a game module:
    from kaggle_environments.envs.game_drivers.shimmy_base import create_shimmy_environment

    GAME_NAME = "tic_tac_toe"
    ENV_NAME = "shimmy_tic_tac_toe"
    TITLE = "Shimmy Tic-Tac-Toe"
    DESCRIPTION = "Tic-Tac-Toe via OpenSpiel through Shimmy"

    # Get all exports
    env = create_shimmy_environment(GAME_NAME, ENV_NAME, TITLE, DESCRIPTION)
    agents = env["agents"]
    interpreter = env["interpreter"]
    renderer = env["renderer"]
    html_renderer = env["html_renderer"]
    specification = env["specification"]
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


class KaggleCallableAgent(RemoteAgent):
    """Wraps a kaggle-environments callable agent as a RemoteAgent.

    This allows standard kaggle-environments agents (functions with signature
    (observation, configuration) -> action) to work with the game driver.
    """

    def __init__(self, callable_agent: Any, configuration: dict | None = None):
        self.callable_agent = callable_agent
        self.configuration = configuration or {}

    def process_turn(self, action_space: Any, observation: Any, info: dict | None = None) -> dict[str, Any]:
        """Entry point called by relay. Adapts to kaggle-environments signature."""
        # Build observation dict - convert numpy array to list if needed
        if hasattr(observation, "tolist"):
            obs_dict = {"observation": observation.tolist()}
        elif hasattr(observation, "__iter__"):
            obs_dict = {"observation": list(observation)}
        else:
            obs_dict = {"observation": observation}

        # Add action mask as legalActions if available
        if info and "action_mask" in info:
            obs_dict["legalActions"] = [i for i, legal in enumerate(info["action_mask"]) if legal]

        # Call the kaggle-environments agent
        action = self.callable_agent(obs_dict, self.configuration)
        return {"action": action}

    def choose_action(self, action_space: Any, observation: Any, info: dict | None = None) -> Any:
        """Not used directly - process_turn handles everything."""
        result = self.process_turn(action_space, observation, info)
        return result["action"]


def _is_http_url(s: Any) -> bool:
    """Check if a string is an HTTP/HTTPS URL."""
    if not isinstance(s, str):
        return False
    return s.startswith("http://") or s.startswith("https://")


def _get_agent_instances(env, environment_name: str) -> list[RemoteAgent]:
    """Get RemoteAgent instances for the agents passed to env.run().

    Supports:
    - "random" -> RandomAgent()
    - HTTP URL string -> ProtobufAgent wrapped as KaggleCallableAgent
    - Callable -> KaggleCallableAgent wrapper

    Raises:
        ValueError: If no agents were provided or an unsupported agent type is encountered.
    """
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
            agent_instances.append(KaggleCallableAgent(protobuf_agent, dict(env.configuration)))
        elif callable(spec):
            agent_instances.append(KaggleCallableAgent(spec, dict(env.configuration)))
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
    """Run the game using OpenSpielGameDriver with relay-based agents."""
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


def _create_interpreter(game_name: str, environment_name: str):
    """Create an interpreter function for the given game."""

    def interpreter(state, env):
        """Interpreter that delegates to OpenSpielGameDriver."""
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

        agent_instances = _get_agent_instances(env, environment_name)
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

    return interpreter


def _default_renderer(state, env):
    """Default renderer - just shows game state as string."""
    return f"Game state: {state}"


def _default_html_renderer():
    """Default HTML renderer."""
    return "<div>Shimmy Game</div>"


def _load_specification(env_name: str, title: str, description: str) -> dict:
    """Load the generic shimmy game specification and customize it."""
    dirpath = os.path.dirname(__file__)
    jsonpath = os.path.abspath(os.path.join(dirpath, "shimmy_game.json"))
    with open(jsonpath) as f:
        spec = json.load(f)

    spec["name"] = env_name
    spec["title"] = title
    spec["description"] = description
    return spec


def create_shimmy_environment(
    game_name: str,
    env_name: str,
    title: str,
    description: str,
    renderer: Any = None,
    html_renderer: Any = None,
) -> dict[str, Any]:
    """Create all exports needed for a shimmy game environment.

    Args:
        game_name: OpenSpiel game name (e.g., "tic_tac_toe")
        env_name: kaggle-environments name (e.g., "shimmy_tic_tac_toe")
        title: Human-readable title
        description: Environment description
        renderer: Optional custom renderer function
        html_renderer: Optional custom HTML renderer function

    Returns:
        Dict with all exports: agents, interpreter, renderer, html_renderer, specification
    """
    return {
        "agents": {"random": _random_agent},
        "interpreter": _create_interpreter(game_name, env_name),
        "renderer": renderer or _default_renderer,
        "html_renderer": html_renderer or _default_html_renderer,
        "specification": _load_specification(env_name, title, description),
    }
