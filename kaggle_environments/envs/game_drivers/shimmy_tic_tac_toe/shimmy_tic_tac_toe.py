"""
Shimmy Tic-Tac-Toe environment for kaggle-environments.
Uses remote_game_drivers with HTTPClient for protobuf-based agent communication.
"""

import json
import os

from kaggle_evaluation.core.relay import HTTPClient
from remote_game_drivers.shimmy_remote_driver.game_driver import ShimmyGameDriver

# Global driver instance (initialized on first interpreter call)
_driver: ShimmyGameDriver | None = None
_agent_clients: dict[int, HTTPClient] = {}


def random_agent(obs: dict, config: dict) -> int:
    """Simple random agent for testing."""
    import random

    valid_actions = [i for i, cell in enumerate(obs["board"]) if cell == "."]
    return random.choice(valid_actions) if valid_actions else 0


agents = {"random": random_agent}


def interpreter(state, env):
    """
    Main game loop interpreter for Shimmy Tic-Tac-Toe.

    This bridges kaggle-environments state management with remote_game_drivers.
    Agent communication uses HTTPClient with protobuf serialization.
    """
    global _driver, _agent_clients

    # Initialize driver on first call
    if _driver is None:
        _driver = ShimmyGameDriver(agent_ids=[0, 1], game_name="openspiel.tic_tac_toe")
        _driver.start_new_game(game_name="openspiel.tic_tac_toe")

        # Initialize HTTP clients for each agent
        # Agent URLs should be passed via env.configuration.agent_urls
        # Format: ['http://agent0:8080', 'http://agent1:8080']
        agent_urls = getattr(env.configuration, "agent_urls", None)
        if agent_urls:
            for i, url in enumerate(agent_urls):
                _agent_clients[i] = HTTPClient(base_url=url)

    # Game is done
    if env.done:
        return state

    # Determine active/inactive agents
    active = state[0] if state[0].status == "ACTIVE" else state[1]
    inactive = state[0] if state[0].status == "INACTIVE" else state[1]

    if active.status != "ACTIVE" or inactive.status != "INACTIVE":
        active.status = "DONE" if active.status == "ACTIVE" else active.status
        inactive.status = "DONE" if inactive.status == "INACTIVE" else inactive.status
        return state

    # Get current player from driver
    current_player = _driver.get_current_player()

    # Get observation from driver
    obs_dict = _driver.get_observation(current_player)

    # Update kaggle-environments observation
    active.observation.board = obs_dict.get("observation", {}).get("board", [])
    active.observation.mark = current_player

    # Get action from agent
    action = active.action

    # Validate action
    if action is None or not isinstance(action, int) or action < 0 or action > 8:
        active.status = f"Invalid action: {action}"
        inactive.status = "DONE"
        inactive.reward = 1
        active.reward = -1
        return state

    # Step the driver with the action
    try:
        _driver.step({current_player: action})
    except Exception as e:
        active.status = f"Invalid move: {e}"
        inactive.status = "DONE"
        inactive.reward = 1
        active.reward = -1
        return state

    # Check if game is done
    if _driver.episode_complete:
        rewards = _driver.get_rewards()
        state[0].reward = rewards.get(0, 0)
        state[1].reward = rewards.get(1, 0)
        state[0].status = "DONE"
        state[1].status = "DONE"
        return state

    # Swap active/inactive
    active.status = "INACTIVE"
    inactive.status = "ACTIVE"

    return state


def renderer(state, env):
    """Render the current game state."""
    board = state[0].observation.board
    if not board or len(board) != 9:
        return "Game not started"

    def format_cell(c):
        if c == ".":
            return " "
        return c

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
