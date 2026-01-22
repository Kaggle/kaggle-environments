"""Shimmy Tic-Tac-Toe environment for kaggle-environments.

Uses OpenSpielGameDriver from remote_game_drivers to run games.
This delegates game logic to the driver rather than reimplementing it.

Architecture:
- The interpreter runs the ENTIRE game on first call using driver.run_game()
- Subsequent interpreter calls just return the final state
- Agents are wrapped as relay servers/clients for the driver
"""

import json
import os

import kaggle_evaluation.core.relay as relay
from remote_game_drivers.core.base_classes import KaggleAgentId
from remote_game_drivers.shimmy_remote_driver.game_driver import OpenSpielGameDriver
from remote_game_drivers.shimmy_remote_driver.remote_agent import RandomAgent

# Global state
_game_complete: bool = False
_results: dict[KaggleAgentId, float | int] = {}


def random_agent(obs: dict, config: dict) -> int:
    """Placeholder for kaggle-environments agent registry."""
    return 0


agents = {"random": random_agent}


def _run_game_with_driver(agent_instances: list) -> dict[KaggleAgentId, float | int]:
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
        driver_config = {"agent_ids": list(agent_ids)}
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
    """
    global _game_complete, _results

    # If game already complete, just return final state
    if _game_complete:
        return state

    # Run the entire game using the driver
    # Use RandomAgent instances for both players
    agent_instances = [RandomAgent(), RandomAgent()]

    try:
        _results = _run_game_with_driver(agent_instances)
        _game_complete = True

        # Map results to kaggle-environments state
        for agent_id, reward in _results.items():
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
        _game_complete = True
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


# Load specification
dirpath = os.path.dirname(__file__)
jsonpath = os.path.abspath(os.path.join(dirpath, "shimmy_tic_tac_toe.json"))
with open(jsonpath) as f:
    specification = json.load(f)
