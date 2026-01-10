"""
Multi-container integration tests for kaggle-environments.

These tests demonstrate running the orchestrator and agents in separate containers,
communicating via HTTP. This is the ideal setup for production-like testing.

For simpler single-container tests, see test_envs.py.
"""

import json
import os
import time
from typing import Any, Dict

import pytest
import requests

# Configuration
ORCHESTRATOR_HOST = os.environ.get("ORCHESTRATOR_HOST", "localhost")
ORCHESTRATOR_PORT = int(os.environ.get("ORCHESTRATOR_PORT", "8080"))
ORCHESTRATOR_URL = f"http://{ORCHESTRATOR_HOST}:{ORCHESTRATOR_PORT}"


def is_orchestrator_available() -> bool:
    """Check if the orchestrator is running and accessible."""
    try:
        response = requests.get(f"{ORCHESTRATOR_URL}/health", timeout=2)
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


class RemoteAgent:
    """
    An agent that runs in a separate container and communicates with the orchestrator via HTTP.

    This class wraps the agent logic to be called by the orchestrator when running
    in multi-container mode.
    """

    def __init__(self, agent_fn, agent_id: str):
        self.agent_fn = agent_fn
        self.agent_id = agent_id

    def __call__(self, observation: Dict[str, Any], configuration: Dict[str, Any]) -> Any:
        """Execute the agent and return the action."""
        return self.agent_fn(observation, configuration)

    def serve(self, host: str = "0.0.0.0", port: int = 8081):
        """
        Start an HTTP server to receive observations and return actions.

        This is used when the agent runs in its own container.
        """
        from http.server import BaseHTTPRequestHandler, HTTPServer

        agent = self

        class AgentHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)
                request = json.loads(post_data.decode("utf-8"))

                observation = request.get("observation", {})
                configuration = request.get("configuration", {})

                action = agent(observation, configuration)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"action": action}).encode("utf-8"))

            def log_message(self, format, *args):
                pass  # Suppress logging

        server = HTTPServer((host, port), AgentHandler)
        print(f"Agent {self.agent_id} serving on {host}:{port}")
        server.serve_forever()


@pytest.mark.skipif(
    not os.environ.get("MULTICONTAINER_TEST"), reason="Multi-container tests require MULTICONTAINER_TEST=1"
)
class TestMultiContainerSetup:
    """Tests that verify multi-container setup works correctly."""

    def test_orchestrator_health(self):
        """Test that the orchestrator health endpoint responds."""
        assert wait_for_orchestrator(timeout=10), "Orchestrator not available"
        response = requests.get(f"{ORCHESTRATOR_URL}/health")
        assert response.status_code == 200

    def test_orchestrator_environments(self):
        """Test that the orchestrator lists available environments."""
        assert wait_for_orchestrator(timeout=10), "Orchestrator not available"
        response = requests.get(f"{ORCHESTRATOR_URL}/environments")
        assert response.status_code == 200
        data = response.json()
        assert "environments" in data
        assert len(data["environments"]) > 0


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
            "environment": "rps",
            "agents": [
                {"url": "http://agent-1:8081"},
                {"url": "http://agent-2:8081"},
            ],
            "configuration": {"episodeSteps": 10},
        }

        response = requests.post(
            f"{ORCHESTRATOR_URL}/episode",
            json=request_data,
            timeout=60,
        )

        assert response.status_code == 200
        data = response.json()
        assert "rewards" in data
        assert len(data["rewards"]) == 2


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
    agent = RemoteAgent(agent_fn, agent_id)
    agent.serve(port=agent_port)


if __name__ == "__main__":
    import sys

    if "--serve" in sys.argv:
        run_agent_server()
    else:
        pytest.main([__file__, "-v", "-s"])
