"""Tests for all Shimmy/OpenSpiel game environments.

This file contains parametrized tests that run against all supported shimmy games.
Each game is tested with random agents to verify basic functionality.

To add a new game:
1. Add the environment name to SHIMMY_GAMES list
2. Ensure the game module exists in kaggle_environments/envs/game_drivers/
"""

import pytest

from kaggle_environments import make

# List of shimmy game environment names to test
# Add new games here as they are implemented
SHIMMY_GAMES = [
    "shimmy_tic_tac_toe",
    "shimmy_connect_four",
]


@pytest.mark.parametrize("game_name", SHIMMY_GAMES)
def test_shimmy_game_random_agents(game_name: str):
    """Test running a game with built-in random agents.

    This is the basic smoke test for each shimmy game:
    - Environment can be created
    - Game runs to completion with random agents
    - Rewards are assigned (numeric values)
    """
    env = make(game_name)
    env.run(["random", "random"])

    # Check that game completed
    assert env.done, f"{game_name}: Game did not complete"

    # Check that rewards were assigned
    final_state = env.steps[-1]
    rewards = [agent.reward for agent in final_state]

    # Rewards should be numeric (OpenSpiel may use different scales)
    assert all(isinstance(r, (int, float)) for r in rewards), f"{game_name}: Rewards should be numeric, got {rewards}"


@pytest.mark.parametrize("game_name", SHIMMY_GAMES)
def test_shimmy_game_specification(game_name: str):
    """Test that each game has a valid specification."""
    env = make(game_name)

    # Check required specification fields
    assert env.name == game_name
    assert env.specification is not None
    assert "name" in env.specification
    assert "title" in env.specification
    assert "description" in env.specification
    assert env.specification["name"] == game_name


@pytest.mark.parametrize("game_name", SHIMMY_GAMES)
def test_shimmy_game_configuration(game_name: str):
    """Test that each game has expected configuration options."""
    env = make(game_name)

    # All shimmy games should have these configuration options
    assert hasattr(env.configuration, "episodeSteps")
    assert hasattr(env.configuration, "actTimeout")
    assert hasattr(env.configuration, "runTimeout")
