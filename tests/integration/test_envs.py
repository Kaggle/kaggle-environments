"""
Integration tests for kaggle-environments.

Runs a single episode of each registered environment using random agents.
These tests are designed to be run inside the Docker container built with docker/build_cpu.sh.
"""

import json
import pathlib
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


def get_deprecated_env_names() -> set:
    """Dynamically discover deprecated environment names from the deprecated_envs directory.

    This handles both:
    - Top-level environments (e.g., deprecated_envs/chess -> "chess")
    - Nested game definitions (e.g., deprecated_envs/open_speil_env/games/tic_tac_toe -> "open_spiel_tic_tac_toe")
    """
    deprecated_names = set()
    test_script = pathlib.Path(__file__).resolve()
    project_root = test_script.parent.parent.parent
    deprecated_dir = project_root / "deprecated_envs"

    if not deprecated_dir.exists():
        return deprecated_names

    # Walk through deprecated_envs directory
    for item in deprecated_dir.iterdir():
        if item.is_dir() and item.name != "__pycache__":
            # Top-level environment
            deprecated_names.add(item.name)

            # Check for nested games directory (e.g., open_speil_env/games/*)
            games_dir = item / "games"
            if games_dir.exists() and games_dir.is_dir():
                for game_item in games_dir.iterdir():
                    if game_item.is_dir() and game_item.name != "__pycache__":
                        # Construct environment name: open_speil_env + game_name
                        # Note: The actual env name uses "open_spiel_" prefix
                        env_prefix = "open_spiel_" if item.name == "open_speil_env" else item.name + "_"
                        deprecated_names.add(env_prefix + game_item.name)

    return deprecated_names


class TestEnvironmentDiscovery:
    """Tests for environment discovery and registration."""

    def test_environments_registered(self):
        """Verify that environments are registered."""
        envs = get_available_environments()
        assert len(envs) > 0, "No environments registered"
        print(f"\nRegistered environments: {envs}")

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
        print(f"\n{env_name}: agents={env.specification.agents}, version={env.version}")

    @pytest.mark.parametrize("env_name", ["rps", "connectx", "halite", "hungry_geese", "kore_fleets", "mab", "cabt"])
    def test_environment_has_agents(self, env_name: str):
        """Test that environments have registered agents."""
        _ = make(env_name)  # Verify env can be created
        agents = get_env_default_agents(env_name)
        assert len(agents) > 0, f"No agents registered for {env_name}"
        print(f"\n{env_name} agents: {list(agents.keys())}")


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
        print(f"\nRPS: {len(steps)} steps, rewards: {[s.reward for s in final_state]}")

    def test_connectx_episode(self):
        """Run a ConnectX episode."""
        env = make("connectx")
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        final_state = steps[-1]
        rewards = [s.reward for s in final_state]
        print(f"\nConnectX: {len(steps)} steps, rewards: {rewards}")

        # One player wins or it's a draw
        assert rewards[0] in [-1, 0, 1] and rewards[1] in [-1, 0, 1]

    def test_mab_episode(self):
        """Run a Multi-Armed Bandit episode."""
        env = make("mab", configuration={"episodeSteps": 10})
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        final_state = steps[-1]
        print(f"\nMAB: {len(steps)} steps, rewards: {[s.reward for s in final_state]}")

    def test_halite_episode(self):
        """Run a Halite episode with 2 players."""
        env = make("halite", configuration={"episodeSteps": 20})
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        final_state = steps[-1]
        print(f"\nHalite: {len(steps)} steps, rewards: {[s.reward for s in final_state]}")

    def test_hungry_geese_episode(self):
        """Run a Hungry Geese episode with 2 players."""
        env = make("hungry_geese", configuration={"episodeSteps": 20})
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        final_state = steps[-1]
        print(f"\nHungry Geese: {len(steps)} steps, rewards: {[s.reward for s in final_state]}")

    def test_kore_fleets_episode(self):
        """Run a Kore Fleets episode with 2 players."""
        env = make("kore_fleets", configuration={"episodeSteps": 20})
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        final_state = steps[-1]
        print(f"\nKore Fleets: {len(steps)} steps, rewards: {[s.reward for s in final_state]}")

    def test_cabt_episode(self):
        """Run a Card Battle episode."""
        env = make("cabt")
        agents = ["random", "random"]
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        final_state = steps[-1]
        rewards = [s.reward for s in final_state]
        print(f"\nCABT: {len(steps)} steps, rewards: {rewards}")

    def test_unconfigured_environments(self):
        """Catchall fallback test any environments not explicitly configured in ENV_CONFIGS and not deprecated.
        This should catch any new environments added to the codebase.
        """
        deprecated_envs = get_deprecated_env_names()

        all_envs = get_available_environments()
        configured_envs = set(ENV_CONFIGS.keys())

        # Find environments that are neither configured nor deprecated
        unconfigured_envs = [env for env in all_envs if env not in configured_envs and env not in deprecated_envs]

        # Environments that don't need to pass the scores check
        scores_allowlist = {"open_spiel_chess"}

        # Accumulate errors to report all failures
        errors = []

        # Test each unconfigured environment
        for env_name in unconfigured_envs:
            try:
                env = make(env_name)

                # Try to determine number of agents needed
                agent_counts = env.specification.agents
                if not agent_counts:
                    print(f"  Skipping {env_name}: no agent count specification")
                    continue

                # Use minimum agent count
                num_agents = min(agent_counts) if isinstance(agent_counts, list) else agent_counts

                # Try to find a random agent
                agents_dict = get_env_default_agents(env_name)
                if "random" in agents_dict:
                    agent = "random"
                elif agents_dict:
                    agent = list(agents_dict.keys())[0]
                else:
                    raise ValueError(f"No default agents available for {env_name}")

                # Run episode with short configuration
                agents = [agent] * num_agents
                config = {"episodeSteps": 10} if "episodeSteps" in env.configuration else {}
                env = make(env_name, configuration=config)
                steps = env.run(agents)

                # Verify episode completed and scores were received
                assert env.done, f"{env_name}: Episode should be done"
                final_state = steps[-1]
                rewards = [s.reward for s in final_state]

                # Check that rewards are not None (scores were received)
                # Skip this check for allowlisted environments
                if env_name not in scores_allowlist:
                    assert all(r is not None for r in rewards), f"{env_name}: All agents should receive scores"

                print(f"  {env_name}: SUCCESS - {len(steps)} steps, rewards: {rewards}")

            except Exception as e:
                error_msg = f"{env_name}: {str(e)}"
                print(f"  FAILED - {error_msg}")
                errors.append(error_msg)

        # Report all errors at the end
        if errors:
            raise AssertionError("The following environments failed:\n" + "\n".join(f"  - {err}" for err in errors))


class TestEvaluateFunction:
    """Tests using the evaluate() convenience function."""

    def test_evaluate_rps(self):
        """Test evaluate function with RPS."""
        rewards = evaluate("rps", ["rock", "paper"], configuration={"episodeSteps": 10})
        assert len(rewards) == 1  # One episode
        assert len(rewards[0]) == 2  # Two agents
        print(f"\nRPS evaluate rewards: {rewards}")

    def test_evaluate_connectx(self):
        """Test evaluate function with ConnectX."""
        rewards = evaluate("connectx", ["random", "random"])
        assert len(rewards) == 1
        assert len(rewards[0]) == 2
        print(f"\nConnectX evaluate rewards: {rewards}")

    def test_evaluate_multiple_episodes(self):
        """Test running multiple episodes."""
        rewards = evaluate("rps", ["rock", "paper"], configuration={"episodeSteps": 10}, num_episodes=3)
        assert len(rewards) == 3
        print(f"\nRPS 3 episodes rewards: {rewards}")


class TestOpenSpielEnvironments:
    """Tests for OpenSpiel-wrapped environments."""

    def _get_open_spiel_envs(self) -> List[str]:
        """Get list of registered OpenSpiel environments."""
        return [name for name in get_available_environments() if name.startswith("open_spiel_")]

    def test_open_spiel_envs_registered(self):
        """Check that OpenSpiel environments are registered."""
        os_envs = self._get_open_spiel_envs()
        print(f"\nOpenSpiel environments: {os_envs}")
        # May be empty if OpenSpiel not installed

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
        steps = env.run(agents)

        assert env.done, "Episode should be done"
        final_state = steps[-1]
        print(f"\n{env_name}: {len(steps)} steps, rewards: {[s.reward for s in final_state]}")


class TestEnvironmentRendering:
    """Tests for environment rendering."""

    def test_rps_render_ansi(self):
        """Test ANSI rendering."""
        env = make("rps", configuration={"episodeSteps": 5})
        env.run(["rock", "paper"])
        output = env.render(mode="ansi")
        assert output is not None
        assert len(output) > 0
        print(f"\nRPS ANSI render:\n{output[:500]}...")

    def test_connectx_render_ansi(self):
        """Test ConnectX ANSI rendering."""
        env = make("connectx")
        env.run(["random", "random"])
        output = env.render(mode="ansi")
        assert output is not None
        print(f"\nConnectX ANSI render:\n{output}")

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
