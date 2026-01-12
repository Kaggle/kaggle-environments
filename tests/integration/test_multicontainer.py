"""
Multi-container integration tests for kaggle-environments.

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


@pytest.mark.skipif(
    not os.environ.get("MULTICONTAINER_TEST"), reason="Multi-container tests require MULTICONTAINER_TEST=1"
)
class TestMultiContainerEpisodes:
    """Tests that run episodes across multiple containers."""

    def test_rps_multicontainer(self):
        """Run RPS with agents in separate containers."""
        assert wait_for_orchestrator(timeout=10), "Orchestrator not available"

        # Request to start an episode
        request_data = {
            "action": "run",
            "environment": "rps",
            "agents": [
                "http://agent-1:8081",
                "http://agent-2:8081",
            ],
            "configuration": {"episodeSteps": 10},
        }

        response = requests.post(
            f"{ORCHESTRATOR_URL}/",
            json=request_data,
            timeout=60,
        )

        assert response.status_code == 200
        # The response is the HTML/JSON output depending on render mode.
        # Default is JSON in main.py but main.py might return rendered JSON string.
        # Let's check if we can parse it.

        try:
            data = response.json()
        except json.JSONDecodeError:
            pytest.fail(f"Failed to decode JSON response: {response.text}")

        # Check for rewards in the last step
        assert "steps" in data
        assert len(data["steps"]) > 0
        last_step = data["steps"][-1]
        assert len(last_step) == 2
        assert "reward" in last_step[0]
        assert "reward" in last_step[1]


# Standalone agent server for multi-container mode
def run_agent_server():
    """
    Run a standalone agent server.

    Usage:
        AGENT_TYPE=random AGENT_PORT=8081 python test_multicontainer.py --serve
    """
    import random

    agent_type = os.environ.get("AGENT_TYPE", "random")
    agent_port = int(os.environ.get("AGENT_PORT", "8081"))
    agent_id = os.environ.get("AGENT_ID", "1")

    def random_agent(obs, config):
        """Simple random agent for RPS."""
        return random.randint(0, 2)

    def rock_agent(obs, config):
        """Always plays rock."""
        return 0

    agents = {
        "random": random_agent,
        "rock": rock_agent,
    }

    agent_fn = agents.get(agent_type, random_agent)
    agent = RemoteAgent(agent_fn, agent_id)  # noqa
    agent.serve(port=agent_port)


if __name__ == "__main__":
    import sys

    if "--serve" in sys.argv:
        run_agent_server()
    else:
        pytest.main([__file__, "-v", "-s"])
