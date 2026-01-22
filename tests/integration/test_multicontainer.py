"""
Multi-container integration tests for kaggle-environments. See docker-compose.yml for setup.

These tests demonstrate running the orchestrator and agents in separate containers,
communicating via HTTP. This is the ideal setup for production-like testing.

For simpler single-container tests, see test_envs.py.
"""

import json
import os
import time

import pytest
import requests

# Configuration
ORCHESTRATOR_HOST = os.environ.get("ORCHESTRATOR_HOST", "localhost")
ORCHESTRATOR_PORT = int(os.environ.get("ORCHESTRATOR_PORT", "8080"))
ORCHESTRATOR_URL = f"http://{ORCHESTRATOR_HOST}:{ORCHESTRATOR_PORT}"


def is_orchestrator_available() -> bool:
    """Check if the orchestrator is running and accessible."""
    try:
        # The root endpoint defaults to "list" action, which returns a list of environments
        response = requests.get(f"{ORCHESTRATOR_URL}/", params={"action": "list"}, timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def wait_for_orchestrator(timeout: int = 30) -> bool:
    """Wait for the orchestrator to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_orchestrator_available():
            return True
        time.sleep(1)
    return False


def load_agent_on_server(host: str, port: int, agent_path: str, environment: str) -> bool:
    """Load an agent on a remote http-server via JSON 'act' action.

    This must be called before using ProtobufAgent to communicate with the server.
    """
    url = f"http://{host}:{port}"
    data = {
        "action": "act",
        "environment": environment,
        "agents": [agent_path],
        "state": {"observation": {"board": [0] * 9, "remainingOverageTime": 60}},
        "configuration": {"actTimeout": 5},
    }
    try:
        response = requests.post(url, json=data, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


# ... (RemoteAgent class remains the same) ...


@pytest.mark.skipif(
    not os.environ.get("MULTICONTAINER_TEST"), reason="Multi-container tests require MULTICONTAINER_TEST=1"
)
class TestMultiContainerSetup:
    """Tests that verify multi-container setup works correctly."""

    def test_orchestrator_health(self):
        """Test that the orchestrator health endpoint responds."""
        assert wait_for_orchestrator(timeout=10), "Orchestrator not available"
        response = requests.get(f"{ORCHESTRATOR_URL}/", params={"action": "list"})
        assert response.status_code == 200

    def test_orchestrator_environments(self):
        """Test that the orchestrator lists available environments."""
        assert wait_for_orchestrator(timeout=10), "Orchestrator not available"
        response = requests.get(f"{ORCHESTRATOR_URL}/", params={"action": "list"})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "rps" in data


# Game configurations for multicontainer tests
# Each tuple: (environment_name, agent_path_or_none, num_episodes)
# agent_path is needed for game drivers that require pre-loading agents
MULTICONTAINER_GAMES = [
    ("rps", None, 1),
    ("shimmy_tic_tac_toe", "tests/envs/remote_game_drivers/test_agent.py", 1),
    ("shimmy_tic_tac_toe", "tests/envs/remote_game_drivers/test_agent.py", 3),
]


@pytest.mark.skipif(
    not os.environ.get("MULTICONTAINER_TEST"), reason="Multi-container tests require MULTICONTAINER_TEST=1"
)
class TestMultiContainerEpisodes:
    """Tests that run episodes across multiple containers."""

    @pytest.mark.parametrize("environment,agent_path,num_episodes", MULTICONTAINER_GAMES)
    def test_multicontainer_episode(self, environment: str, agent_path: str | None, num_episodes: int):
        """Run game episodes with agents in separate containers.

        Tests both JSON-based (RPS) and protobuf-based (shimmy) agent communication.
        With num_episodes > 1, also tests that environments correctly reset state.
        """
        assert wait_for_orchestrator(timeout=10), "Orchestrator not available"

        # Load agents if required (game drivers need pre-loaded agents)
        if agent_path:
            for host in ["agent-1", "agent-2"]:
                assert load_agent_on_server(host, 8081, agent_path, environment), f"Failed to load agent on {host}"

        request_data = {
            "action": "evaluate",
            "environment": environment,
            "agents": ["http://agent-1:8081", "http://agent-2:8081"],
            "episodes": num_episodes,
        }

        response = requests.post(
            f"{ORCHESTRATOR_URL}/",
            json=request_data,
            timeout=60 * num_episodes,
        )

        assert response.status_code == 200

        try:
            data = response.json()
        except json.JSONDecodeError:
            pytest.fail(f"Failed to decode JSON response: {response.text}")

        # evaluate returns [[r1, r2], [r1, r2], ...]
        assert isinstance(data, list), f"Expected list of episode rewards, got {type(data)}"
        assert len(data) == num_episodes, f"Expected {num_episodes} episodes, got {len(data)}"

        for i, episode_rewards in enumerate(data):
            assert len(episode_rewards) == 2, f"Episode {i}: Expected 2 agent rewards"
            assert all(isinstance(r, (int, float, type(None))) for r in episode_rewards), (
                f"Episode {i}: Rewards should be numeric, got {episode_rewards}"
            )


# Standalone agent server for multi-container mode
def run_agent_server():
    """
    Run a standalone agent server using main.py http-server.

    In production, run this in a separate Docker container:
        python -m kaggle_environments.main http-server --port=8081 --host=0.0.0.0

    The orchestrator will then load agents via JSON 'act' requests before
    running games. Both JSON (UrlAgent) and protobuf (ProtobufAgent) protocols
    are supported.

    This function provides a simple wrapper for local testing:
        AGENT_PORT=8081 python test_multicontainer.py --serve
    """
    from kaggle_environments.main import action_http
    from kaggle_environments.utils import structify

    agent_port = int(os.environ.get("AGENT_PORT", "8081"))

    print(f"Starting agent server on port {agent_port}")
    print("Note: Agent must be loaded via 'act' request before use")

    args = structify(
        {
            "host": "0.0.0.0",
            "port": agent_port,
            "debug": False,
            "log_path": None,
        }
    )
    action_http(args)


if __name__ == "__main__":
    import sys

    if "--serve" in sys.argv:
        run_agent_server()
    else:
        pytest.main([__file__, "-v", "-s"])
