"""Tests for all Shimmy/OpenSpiel game environments.

This file contains parametrized tests that run against all supported shimmy games.
Each game is tested with random agents to verify basic functionality.

To add a new game:
1. For dedicated modules: Add the environment name to SHIMMY_GAMES list
2. For dynamic games: Add the OpenSpiel game name to DYNAMIC_SHIMMY_GAMES list
"""

import pytest

from kaggle_environments import make

# List of shimmy game environment names with dedicated modules
# Add new games here as they are implemented
SHIMMY_GAMES = [
    "shimmy_tic_tac_toe",
    "shimmy_connect_four",
]

# List of OpenSpiel game names that can be run via the dynamic "shimmy" environment
# These games don't need dedicated modules - just specify game_name in info
DYNAMIC_SHIMMY_GAMES = [
    "tic_tac_toe",
    "connect_four",
    "chess",
    "go",
    "gin_rummy",
    "backgammon",
    "checkers",
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


# Tests for the dynamic "shimmy" environment that uses info parameter
@pytest.mark.parametrize("game_name", DYNAMIC_SHIMMY_GAMES)
def test_dynamic_shimmy_random_agents(game_name: str):
    """Test running games via the dynamic shimmy environment.

    This tests the ability to run any OpenSpiel game by specifying
    the game name via info={'game_name': '...'}.
    """
    env = make("shimmy", info={"game_name": game_name})
    env.run(["random", "random"])

    assert env.done, f"Dynamic shimmy {game_name}: Game did not complete"

    final_state = env.steps[-1]
    rewards = [agent.reward for agent in final_state]

    assert all(isinstance(r, (int, float)) for r in rewards), (
        f"Dynamic shimmy {game_name}: Rewards should be numeric, got {rewards}"
    )


def test_dynamic_shimmy_missing_game_name():
    """Test that dynamic shimmy raises error when game_name is missing."""
    env = make("shimmy")

    with pytest.raises(ValueError, match="No game_name specified"):
        env.run(["random", "random"])
