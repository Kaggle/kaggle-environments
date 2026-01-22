"""Shimmy Tic-Tac-Toe environment for kaggle-environments.

Uses OpenSpielGameDriver from remote_game_drivers to run games.
This delegates game logic to the driver rather than reimplementing it.

Architecture:
- The interpreter runs the ENTIRE game on first call using driver.run_game()
- Subsequent interpreter calls just return the final state
- Agents are wrapped as relay servers/clients for the driver
- Supports built-in agents ("random"), callable agents, and ProtobufAgent instances
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


def random_agent(obs: dict, config: dict) -> int:
    """Placeholder for kaggle-environments agent registry."""
    return 0


agents = {"random": random_agent}


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
        # Build observation dict in kaggle-environments format
        obs_dict = {
            "board": observation.tolist() if hasattr(observation, "tolist") else list(observation),
        }
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


def _get_agent_instances(env) -> list[RemoteAgent]:
    """Get RemoteAgent instances for the agents passed to env.run().

    Supports:
    - "random" -> RandomAgent()
    - HTTP URL string -> ProtobufAgent wrapped as KaggleCallableAgent
    - Callable -> KaggleCallableAgent wrapper

    Raises:
        ValueError: If no agents were provided or an unsupported agent type is encountered.
    """
    from kaggle_environments.envs.game_drivers.protobuf_agent import ProtobufAgent

    # Check if agent info was stored on env.info
    agent_specs = env.info.get("_agent_specs", [])

    if not agent_specs:
        raise ValueError("No agents provided. Agents must be passed to env.run() for game driver environments.")

    agent_instances = []
    for i, spec in enumerate(agent_specs):
        if spec == "random":
            agent_instances.append(RandomAgent())
        elif _is_http_url(spec):
            # HTTP URL string - create ProtobufAgent (uses protobuf, posts to /)
            protobuf_agent = ProtobufAgent(spec, environment_name="shimmy_tic_tac_toe")
            agent_instances.append(KaggleCallableAgent(protobuf_agent, dict(env.configuration)))
        elif callable(spec):
            # Wrap callable agent
            agent_instances.append(KaggleCallableAgent(spec, dict(env.configuration)))
        else:
            raise ValueError(
                f"Unsupported agent type for agent {i}: {type(spec).__name__}. "
                f"Expected 'random', HTTP URL string, or callable."
            )

    return agent_instances


def _run_game_with_driver(agent_instances: list, episode_steps: int | None = None) -> dict[KaggleAgentId, float | int]:
    """Run the game using OpenSpielGameDriver with relay-based agents.

    This follows the same pattern as test_utils.run_local_game() in Hearth.
    """
    agent_ids = [KaggleAgentId(f"agent_{i}") for i in range(len(agent_instances))]

    relay.set_allowed_modules(["remote_game_drivers.gymnasium_remote_driver.serialization"])

    # Create relay server/client for each agent
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
        # Note: episodeSteps is not currently used for OpenSpiel games
        # OpenSpiel games have their own termination conditions

        driver = OpenSpielGameDriver(driver_config=driver_config, relay_clients=relay_clients)
        results = driver.run_game(game_name="tic_tac_toe")

        for client in relay_clients.values():
            client.close()
        return results
    finally:
        for server in servers:
            server.stop(0)
        relay.set_allowed_modules(None)


def interpreter(state, env):
    """
    Interpreter that delegates to OpenSpielGameDriver.

    On first call, runs the entire game via the driver.
    Returns final results mapped to kaggle-environments state format.

    Supports custom agents passed to env.run() via env.info["_agent_specs"].
    """
    global _env_state

    # Use env id to track state per environment instance (avoids test pollution)
    env_id = id(env)
    if env_id not in _env_state:
        _env_state[env_id] = {"complete": False, "results": {}}

    env_data = _env_state[env_id]

    # If game already complete, just return final state
    if env_data["complete"]:
        return state

    # Check if agents have been set (happens in env.run(), not env.reset())
    # If not set yet, this is just the initial state setup - return unchanged
    agent_specs = env.info.get("_agent_specs", [])
    if not agent_specs:
        return state

    # Get agent instances based on what was passed to env.run()
    agent_instances = _get_agent_instances(env)

    # Get episode steps limit from configuration
    episode_steps = getattr(env.configuration, "episodeSteps", None)

    try:
        results = _run_game_with_driver(agent_instances, episode_steps)
        env_data["complete"] = True
        env_data["results"] = results

        # Map results to kaggle-environments state
        for agent_id, reward in results.items():
            # agent_id is like "agent_0", "agent_1"
            idx = int(str(agent_id).split("_")[1])
            state[idx].reward = reward
            state[idx].status = "DONE"

        return state
    except Exception as e:
        # Game failed - mark all agents as error
        for i in range(len(state)):
            state[i].status = f"ERROR: {e}"
            state[i].reward = 0
        env_data["complete"] = True
        return state


def renderer(state, env):
    """Render the current game state."""
    board = state[0].observation.board
    if not board or len(board) != 9:
        return "Game not started"

    def format_cell(c):
        # OpenSpiel format: 0=empty, 1=player 0, 2=player 1
        if c == 0:
            return " "
        elif c == 1:
            return "X"
        elif c == 2:
            return "O"
        return str(c)

    lines = []
    lines.append("  0 | 1 | 2")
    lines.append(" -----------")
    for row in range(3):
        cells = [format_cell(board[row * 3 + col]) for col in range(3)]
        lines.append(f"  {cells[0]} | {cells[1]} | {cells[2]}")
        if row < 2:
            lines.append(" -----------")

    return "\n".join(lines)


def html_renderer():
    """HTML renderer (placeholder for now)."""
    return "<div>Shimmy Tic-Tac-Toe</div>"


# Load generic shimmy game specification and customize for tic-tac-toe
dirpath = os.path.dirname(__file__)
generic_jsonpath = os.path.abspath(os.path.join(dirpath, "..", "shimmy_game.json"))
with open(generic_jsonpath) as f:
    specification = json.load(f)

# Override with tic-tac-toe specific values
specification["name"] = "shimmy_tic_tac_toe"
specification["title"] = "Shimmy Tic-Tac-Toe"
specification["description"] = "Tic-Tac-Toe via OpenSpiel through Shimmy remote game driver"
