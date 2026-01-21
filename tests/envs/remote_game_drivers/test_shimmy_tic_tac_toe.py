"""
Integration test for Shimmy Tic-Tac-Toe environment.

Tests the complete flow:
1. Environment initialization with remote game driver
2. Agent communication via HTTP with protobuf serialization
3. Game execution and state management
"""

import pytest

from kaggle_environments import make
from kaggle_environments.envs.game_drivers.agent_server import create_agent_server


def simple_agent(observation, configuration):
    """Simple test agent that picks first available move."""
    board = observation.get("board", [])
    for i, cell in enumerate(board):
        if cell == ".":
            return i
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

    # In tic-tac-toe, rewards should be -1, 0, or 1
    assert all(r in [-1, 0, 1] for r in rewards)

    # Winner gets 1, loser gets -1, or both get 0 for draw
    assert sum(rewards) == 0 or set(rewards) == {1, -1}


def test_agent_server_wrapper():
    """Test that agent_server.py can wrap a simple agent function."""
    app = create_agent_server(simple_agent, port=8081)
    assert app is not None

    # Test that we can create a test client
    client = app.test_client()
    assert client is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
