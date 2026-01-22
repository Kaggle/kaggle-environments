"""
Integration test for Shimmy Tic-Tac-Toe environment.

Tests the complete flow:
1. Environment initialization with remote game driver
2. Agent communication via HTTP with protobuf serialization
3. Game execution and state management
"""

import threading
import time

import pytest

from kaggle_environments import make
from kaggle_environments.envs.game_drivers.agent_server import create_agent_server
from kaggle_environments.envs.game_drivers.protobuf_agent import ProtobufAgent


def simple_agent(observation, configuration):
    """Simple test agent that picks first available move."""
    import random

    board = observation.get("board", [])
    if not board:
        return 0

    # OpenSpiel format: 0=empty, 1=player 0's mark, 2=player 1's mark
    # Find all empty cells
    empty_cells = [i for i, cell in enumerate(board) if cell == 0]

    # Return random empty cell
    return random.choice(empty_cells) if empty_cells else 0


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


def test_agent_server_wrapper():
    """Test that agent_server.py can wrap a simple agent function."""
    app = create_agent_server(simple_agent, port=8081)
    assert app is not None

    # Test that we can create a test client
    client = app.test_client()
    assert client is not None


def test_shimmy_tic_tac_toe_http_agents():
    """Test running a game with HTTP agents via agent_server.py."""
    # Create agent servers for two agents
    app1 = create_agent_server(simple_agent, port=8091)
    app2 = create_agent_server(simple_agent, port=8092)

    # Run servers in background threads
    def run_server(app, port):
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

    thread1 = threading.Thread(target=run_server, args=(app1, 8091), daemon=True)
    thread2 = threading.Thread(target=run_server, args=(app2, 8092), daemon=True)

    thread1.start()
    thread2.start()

    # Give servers time to start
    time.sleep(0.5)

    # Create ProtobufAgent instances
    agent1 = ProtobufAgent("http://127.0.0.1:8091")
    agent2 = ProtobufAgent("http://127.0.0.1:8092")

    # Run game with HTTP agents
    env = make("shimmy_tic_tac_toe")
    env.run([agent1, agent2])

    # Check that game completed
    assert env.done

    # Check that rewards were assigned
    final_state = env.steps[-1]
    rewards = [agent.reward for agent in final_state]

    # Rewards should be numeric (OpenSpiel may use different scales)
    assert all(isinstance(r, (int, float)) for r in rewards)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
