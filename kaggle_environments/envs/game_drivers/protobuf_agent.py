"""
Protobuf-based HTTP agent for kaggle-environments.

This agent uses relay's HTTPClient for protobuf serialization,
compatible with agent_server.py.
"""

from typing import Any

from kaggle_evaluation.core.relay import HTTPClient


class ProtobufAgent:
    """Agent that communicates via HTTP with protobuf serialization."""

    def __init__(self, base_url: str):
        """
        Args:
            base_url: Base URL of the agent server (e.g., 'http://localhost:8080')
        """
        self.base_url = base_url
        self.client = HTTPClient(base_url=base_url)

    def __call__(self, observation: Any, configuration: Any) -> Any:
        """
        Call the remote agent via HTTP with protobuf serialization.

        Args:
            observation: Current observation from the environment
            configuration: Environment configuration

        Returns:
            Action from the agent
        """
        try:
            action = self.client.send("process_turn", observation, configuration)
            return action
        except Exception as e:
            print(f"ProtobufAgent error: {e}")
            return None
