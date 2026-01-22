"""
Integration test for Shimmy Tic-Tac-Toe environment.

Tests the complete flow:
1. Environment initialization with remote game driver
2. Agent communication via HTTP with protobuf serialization
3. Game execution and state management
"""

import threading
import time
from pathlib import Path

import pytest
import requests

from kaggle_environments import make
from kaggle_environments.envs.game_drivers.protobuf_agent import ProtobufAgent
from kaggle_environments.main import action_http
from kaggle_environments.utils import structify

# Path to test agent file
TEST_AGENT_PATH = Path(__file__).parent / "test_agent.py"


def simple_agent(observation, configuration):
    """Simple test agent that picks a random legal move."""
    import random

    # Use legalActions if available (preferred - comes from action mask)
    legal_actions = observation.get("legalActions", [])
    if legal_actions:
        return random.choice(legal_actions)

    # Fallback: check board for empty cells
    board = observation.get("board", [])
    if board:
        empty_cells = [i for i, cell in enumerate(board) if cell == 0]
        if empty_cells:
            return random.choice(empty_cells)

    return 0


def start_http_server(port: int, max_wait: float = 5.0) -> threading.Thread:
    """Start main.py http-server in a thread and wait for it to be ready."""

    def run_server():
        args = structify(
            {
                "host": "127.0.0.1",
                "port": port,
                "debug": False,
                "log_path": None,
            }
        )
        action_http(args)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Wait for server to be ready by polling
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            requests.get(f"http://127.0.0.1:{port}", timeout=0.5)
            # Server is ready if we get any response
            break
        except requests.exceptions.ConnectionError:
            time.sleep(0.2)
        except requests.exceptions.Timeout:
            time.sleep(0.2)

    return thread


def load_agent_on_server(port: int, agent_path: str, environment: str = "shimmy_tic_tac_toe") -> None:
    """Load an agent on the http-server via JSON 'act' action."""
    url = f"http://127.0.0.1:{port}"
    data = {
        "action": "act",
        "environment": environment,
        "agents": [str(agent_path)],
        "state": {"observation": {"board": [0] * 9, "remainingOverageTime": 60}},
        "configuration": {"actTimeout": 5},
    }
    response = requests.post(url, json=data, timeout=10)
    response.raise_for_status()


def test_shimmy_tic_tac_toe_random_agents():
    """Test running a game with built-in agents."""
    env = make("shimmy_tic_tac_toe")
    env.run(["random", "random"])

    # Check that game completed
    assert env.done

    # Check that rewards were assigned
    final_state = env.steps[-1]
    rewards = [agent.reward for agent in final_state]

    # Rewards should be numeric (OpenSpiel may use different scales)
    assert all(isinstance(r, (int, float)) for r in rewards)


def test_shimmy_tic_tac_toe_callable_agents():
    """Test running a game with custom callable agents."""
    env = make("shimmy_tic_tac_toe")

    # Run with custom callable agents (simple_agent picks random empty cell)
    env.run([simple_agent, simple_agent])

    # Check that game completed
    assert env.done

    # Check that rewards were assigned
    final_state = env.steps[-1]
    rewards = [agent.reward for agent in final_state]

    # Rewards should be numeric
    assert all(isinstance(r, (int, float)) for r in rewards)


def test_shimmy_tic_tac_toe_protobuf_agents():
    """Test running a game with ProtobufAgent instances via main.py http-server.

    This uses main.py http-server which supports both JSON and protobuf protocols.
    """
    # Start two http-servers for two agents (daemon threads auto-cleanup)
    start_http_server(port=8091)
    start_http_server(port=8092)

    # Load agents on each server via JSON 'act' action
    load_agent_on_server(8091, TEST_AGENT_PATH)
    load_agent_on_server(8092, TEST_AGENT_PATH)

    # Create ProtobufAgent instances
    agent1 = ProtobufAgent("http://127.0.0.1:8091")
    agent2 = ProtobufAgent("http://127.0.0.1:8092")

    # Run game with ProtobufAgent instances
    env = make("shimmy_tic_tac_toe")
    env.run([agent1, agent2])

    # Check that game completed
    assert env.done

    # Check that rewards were assigned
    final_state = env.steps[-1]
    rewards = [agent.reward for agent in final_state]

    # Rewards should be numeric (OpenSpiel may use different scales)
    assert all(isinstance(r, (int, float)) for r in rewards)


def test_shimmy_tic_tac_toe_http_url_agents():
    """Test running a game with HTTP URL strings as agents (mimics C# backend flow).

    This is how the C# backend invokes kaggle-environments:
        python main.py run --environment shimmy_tic_tac_toe --agents http://agent1:8080 http://agent2:8081

    The interpreter must detect HTTP URLs and create ProtobufAgent instances automatically.
    """
    # Start two http-servers for two agents (daemon threads auto-cleanup)
    start_http_server(port=8093)
    start_http_server(port=8094)

    # Load agents on each server via JSON 'act' action
    load_agent_on_server(8093, TEST_AGENT_PATH)
    load_agent_on_server(8094, TEST_AGENT_PATH)

    # Run game with HTTP URL strings (like C# backend would pass)
    env = make("shimmy_tic_tac_toe")
    env.run(["http://127.0.0.1:8093", "http://127.0.0.1:8094"])

    # Check that game completed
    assert env.done

    # Check that rewards were assigned
    final_state = env.steps[-1]
    rewards = [agent.reward for agent in final_state]

    # Rewards should be numeric (OpenSpiel may use different scales)
    assert all(isinstance(r, (int, float)) for r in rewards)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
