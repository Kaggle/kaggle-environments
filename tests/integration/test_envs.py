"""
Integration tests for kaggle-environments.

Most tests run a single episode of the registered environment using random agents.
"""

import json
import pathlib
import subprocess
from typing import Any, Dict, List

import pytest

import kaggle_environments
from kaggle_environments import evaluate, make


class TestPackageConfig:
    """Tests for package configuration and environment."""

    def test_using_local_package(self):
        """Verify that we are using the local package, not the one installed via PyPi.
        The local package should take import precedence by default based on Python's import system,
        but it's worth verifying as the tests aren't meaningful if the wrong version is being used.
        """

        location = kaggle_environments.__file__
        test_script = pathlib.Path(__file__).resolve()
        import_path = pathlib.Path(location).resolve()

        test_project_base_dir = test_script.parent.parent.parent
        import_project_base_dir = import_path.parent.parent

        assert test_project_base_dir == import_project_base_dir, (
            f"Import path should share the same project root with test script.\n"
            f"Test: {test_project_base_dir}\n"
            f"Import: {import_project_base_dir}"
        )


# Environment configurations: env_name -> (num_agents, agent_name, config_overrides, skip_reason)
# Some environments need special handling or may be skipped
ENV_CONFIGS: Dict[str, tuple] = {
    # Standard 2-player games
    "rps": (2, "rock", {"episodeSteps": 10}, None),
    "connectx": (2, "random", {}, None),
    "mab": (2, "random", {"episodeSteps": 10}, None),
    "cabt": (2, "random", {}, None),
    # Multi-player games (use minimum agent count)
    "halite": (2, "random", {"episodeSteps": 20}, None),
    "hungry_geese": (2, "random", {"episodeSteps": 20}, None),
    "kore_fleets": (2, "random", {"episodeSteps": 20}, None),
    # Games with special requirements
    "lux_ai_s3": (2, "random_agent", {}, "Requires luxai_s3 package"),
    "lux_ai_2021": (2, None, {}, "Requires external dependencies"),
    "football": (2, None, {}, "Requires gfootball package"),
    # Werewolf requires specific agent configuration
    "werewolf": (6, "random", {}, "Complex setup required"),
}


def get_available_environments() -> List[str]:
    """Get list of all registered environments."""
    return list(kaggle_environments.environments.keys())


def get_env_agent_count(env_name: str) -> List[int]:
    """Get the valid agent counts for an environment."""
    try:
        env = make(env_name)
        return env.specification.agents
    except Exception:
        return []


def get_env_default_agents(env_name: str) -> Dict[str, Any]:
    """Get the default agents registered for an environment."""
    try:
        env = make(env_name)
        return dict(env.agents) if env.agents else {}
    except Exception:
        return {}


class TestEnvironmentDiscovery:
    """Tests for environment discovery and registration."""

    def test_environments_registered(self):
        """Verify that environments are registered."""
        envs = get_available_environments()
        assert len(envs) > 0, "No environments registered"

    def test_core_environments_present(self):
        """Verify core environments are available."""
        envs = get_available_environments()
        core_envs = ["rps", "connectx", "halite", "hungry_geese", "kore_fleets", "mab"]
        for env_name in core_envs:
            assert env_name in envs, f"Core environment '{env_name}' not registered"


class TestEnvironmentCreation:
    """Tests for environment creation."""

    @pytest.mark.parametrize("env_name", ["rps", "connectx", "halite", "hungry_geese", "kore_fleets", "mab", "cabt"])
    def test_make_environment(self, env_name: str):
        """Test that environments can be created."""
        env = make(env_name)
        assert env is not None
        assert env.name == env_name

    @pytest.mark.parametrize("env_name", ["rps", "connectx", "halite", "hungry_geese", "kore_fleets", "mab", "cabt"])
    def test_environment_has_agents(self, env_name: str):
        """Test that environments have registered agents."""
        _ = make(env_name)  # Verify env can be created
        agents = get_env_default_agents(env_name)
        assert len(agents) > 0, f"No agents registered for {env_name}"


class TestEpisodeExecution:
    """Integration tests that run full episodes."""

    def test_rps_episode(self):
        """Run a Rock-Paper-Scissors episode."""
        env = make("rps", configuration={"episodeSteps": 10})
        agents = ["rock", "paper"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        assert len(steps) > 1, "Should have multiple steps"

        # Check final state
        final_state = steps[-1]
        assert all(s.status == "DONE" for s in final_state), "All agents should be DONE"
        rewards = [s.reward for s in final_state]
        assert all(isinstance(r, (int, float)) for r in rewards), "All rewards should be numeric"
        assert all(r is not None for r in rewards), "All rewards should be non-None"

    def test_connectx_episode(self):
        """Run a ConnectX episode."""
        env = make("connectx")
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        assert len(steps) > 1, "Should have multiple steps"
        final_state = steps[-1]
        rewards = [s.reward for s in final_state]
        assert all(isinstance(r, (int, float)) for r in rewards), "All rewards should be numeric"
        # One player wins or it's a draw
        assert rewards[0] in [-1, 0, 1] and rewards[1] in [-1, 0, 1]

    def test_mab_episode(self):
        """Run a Multi-Armed Bandit episode."""
        env = make("mab", configuration={"episodeSteps": 10})
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        assert len(steps) > 1, "Should have multiple steps"
        final_state = steps[-1]
        rewards = [s.reward for s in final_state]
        assert all(isinstance(r, (int, float)) for r in rewards), "All rewards should be numeric"
        assert all(r is not None for r in rewards), "All rewards should be non-None"

    def test_halite_episode(self):
        """Run a Halite episode with 2 players."""
        env = make("halite", configuration={"episodeSteps": 20})
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        assert len(steps) > 1, "Should have multiple steps"
        final_state = steps[-1]
        rewards = [s.reward for s in final_state]
        assert all(isinstance(r, (int, float)) for r in rewards), "All rewards should be numeric"
        assert all(r is not None for r in rewards), "All rewards should be non-None"

    def test_hungry_geese_episode(self):
        """Run a Hungry Geese episode with 2 players."""
        env = make("hungry_geese", configuration={"episodeSteps": 20})
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        assert len(steps) > 1, "Should have multiple steps"
        final_state = steps[-1]
        rewards = [s.reward for s in final_state]
        assert all(isinstance(r, (int, float)) for r in rewards), "All rewards should be numeric"
        assert all(r is not None for r in rewards), "All rewards should be non-None"

    def test_kore_fleets_episode(self):
        """Run a Kore Fleets episode with 2 players."""
        env = make("kore_fleets", configuration={"episodeSteps": 20})
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        assert len(steps) > 1, "Should have multiple steps"
        final_state = steps[-1]
        rewards = [s.reward for s in final_state]
        assert all(isinstance(r, (int, float)) for r in rewards), "All rewards should be numeric"
        assert all(r is not None for r in rewards), "All rewards should be non-None"

    def test_cabt_episode(self):
        """Run a Card Battle episode."""
        env = make("cabt")
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        assert len(steps) > 1, "Should have multiple steps"
        final_state = steps[-1]
        rewards = [s.reward for s in final_state]
        assert all(isinstance(r, (int, float)) for r in rewards), "All rewards should be numeric"
        assert all(r is not None for r in rewards), "All rewards should be non-None"


class TestEvaluateFunction:
    """Tests using the evaluate() convenience function."""

    def test_evaluate_rps(self):
        """Test evaluate function with RPS."""
        rewards = evaluate("rps", ["rock", "paper"], configuration={"episodeSteps": 10})
        assert len(rewards) == 1  # One episode
        assert len(rewards[0]) == 2  # Two agents

    def test_evaluate_connectx(self):
        """Test evaluate function with ConnectX."""
        rewards = evaluate("connectx", ["random", "random"])
        assert len(rewards) == 1
        assert len(rewards[0]) == 2

    def test_evaluate_multiple_episodes(self):
        """Test running multiple episodes."""
        rewards = evaluate("rps", ["rock", "paper"], configuration={"episodeSteps": 10}, num_episodes=3)
        assert len(rewards) == 3


class TestOpenSpielEnvironments:
    """Tests for OpenSpiel-wrapped environments."""

    @pytest.mark.parametrize(
        "env_name",
        [
            pytest.param(
                "open_spiel_tic_tac_toe",
                marks=pytest.mark.skipif(
                    "open_spiel_tic_tac_toe" not in get_available_environments(), reason="OpenSpiel not available"
                ),
            ),
            pytest.param(
                "open_spiel_connect_four",
                marks=pytest.mark.skipif(
                    "open_spiel_connect_four" not in get_available_environments(), reason="OpenSpiel not available"
                ),
            ),
        ],
    )
    def test_open_spiel_episode(self, env_name: str):
        """Run an OpenSpiel environment episode."""
        env = make(env_name)
        agents = ["random", "random"]
        env.run(agents)
        assert env.done, "Episode should be done"


class TestEnvironmentRendering:
    """Tests for environment rendering."""

    def test_rps_render_ansi(self):
        """Test ANSI rendering."""
        env = make("rps", configuration={"episodeSteps": 5})
        env.run(["rock", "paper"])
        output = env.render(mode="ansi")
        assert output is not None
        assert len(output) > 0

    def test_connectx_render_ansi(self):
        """Test ConnectX ANSI rendering."""
        env = make("connectx")
        env.run(["random", "random"])
        output = env.render(mode="ansi")
        assert output is not None

    def test_render_json(self):
        """Test JSON rendering."""
        env = make("rps", configuration={"episodeSteps": 5})
        env.run(["rock", "paper"])
        output = env.render(mode="json")
        assert output is not None
        # Should be valid JSON
        data = json.loads(output)
        assert "steps" in data
        assert "configuration" in data


class TestAgentTypes:
    """Tests for different agent types."""

    def test_callable_agent(self):
        """Test using a callable as an agent."""

        def my_agent(obs, config):
            return 0  # Always rock

        env = make("rps", configuration={"episodeSteps": 5})
        env.run([my_agent, "paper"])
        assert env.done

    def test_lambda_agent(self):
        """Test using a lambda as an agent."""
        env = make("rps", configuration={"episodeSteps": 5})
        env.run([lambda obs, config: 0, lambda obs, config: 1])
        assert env.done

    def test_none_agent_in_training(self):
        """Test training mode with None agent."""
        env = make("rps", configuration={"episodeSteps": 5})
        trainer = env.train([None, "rock"])

        obs = trainer.reset()
        assert obs is not None

        done = False
        steps = 0
        while not done and steps < 10:
            obs, reward, done, info = trainer.step(0)  # Always rock
            steps += 1


class TestCLI:
    """Tests for the command-line interface."""

    def test_cli_evaluate(self):
        """Test CLI evaluate command."""
        result = subprocess.run(
            [
                "python",
                "-m",
                "kaggle_environments.main",
                "evaluate",
                "--environment",
                "connectx",
                "--agents",
                "random",
                "random",
                "--episodes",
                "3",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0, f"CLI command failed with: {result.stderr}"
        assert len(result.stdout) > 0, "CLI should produce output"

        # Output should contain episode results
        assert "[[" in result.stdout, "Output should contain reward arrays"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
