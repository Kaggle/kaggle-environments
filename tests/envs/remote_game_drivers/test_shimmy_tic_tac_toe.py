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


def test_agent_server_wrapper():
    """Test that agent_server.py can wrap a simple agent function."""
    app = create_agent_server(simple_agent, port=8081)
    assert app is not None

    # Test that we can create a test client
    client = app.test_client()
    assert client is not None


def test_shimmy_tic_tac_toe_http_agents():
    """Test running a game with HTTP agents via agent_server.py (ProtobufAgent instances)."""
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


def test_shimmy_tic_tac_toe_http_url_agents():
    """Test running a game with HTTP URL strings as agents (mimics C# backend flow).

    This is how the C# backend invokes kaggle-environments:
        python main.py run --environment shimmy_tic_tac_toe --agents http://agent1:8080 http://agent2:8081

    The interpreter must detect HTTP URLs and create ProtobufAgent instances automatically.
    """
    # Create agent servers for two agents
    app1 = create_agent_server(simple_agent, port=8093)
    app2 = create_agent_server(simple_agent, port=8094)

    # Run servers in background threads
    def run_server(app, port):
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

    thread1 = threading.Thread(target=run_server, args=(app1, 8093), daemon=True)
    thread2 = threading.Thread(target=run_server, args=(app2, 8094), daemon=True)

    thread1.start()
    thread2.start()

    # Give servers time to start
    time.sleep(0.5)

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
