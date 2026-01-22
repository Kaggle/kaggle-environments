"""
Protobuf-based HTTP agent for kaggle-environments.

This agent uses relay's HTTPClient with use_root_endpoint=True to POST
protobuf to / for compatibility with main.py http-server.
It implements the RemoteAgentProtocol interface, making it a drop-in
replacement for UrlAgent when protobuf serialization is needed.

Usage:
    # As a callable (like UrlAgent)
    agent = ProtobufAgent("http://localhost:8080")
    action = agent(observation, configuration)

    # The agent communicates with main.py http-server
"""

from typing import Any

from kaggle_evaluation.core.relay import HTTPClient

from kaggle_environments.errors import DeadlineExceeded


class ProtobufAgent:
    """Agent that communicates via HTTP with protobuf serialization.

    Implements RemoteAgentProtocol to be compatible with UrlAgent.
    Uses HTTPClient with use_root_endpoint=True to POST to /.
    """

    def __init__(self, base_url: str, environment_name: str | None = None, timeout: int = 60):
        """
        Args:
            base_url: Base URL of the agent server (e.g., 'http://localhost:8080')
            environment_name: Name of the environment (for compatibility with UrlAgent)
            timeout: Request timeout in seconds
        """
        self.raw = base_url  # Required by RemoteAgentProtocol
        self.base_url = base_url
        self.environment_name = environment_name
        self.client = HTTPClient(base_url=base_url, timeout_seconds=timeout, use_root_endpoint=True)

    def __call__(self, observation: Any, configuration: Any) -> Any:
        """
        Call the remote agent via HTTP with protobuf serialization.

        Args:
            observation: Current observation from the environment
            configuration: Environment configuration

        Returns:
            Action from the agent, or DeadlineExceeded/None on error
        """
        try:
            # Send observation and configuration to the agent
            # The endpoint name "process_turn" is included in the protobuf message
            result = self.client.send("process_turn", observation, configuration)

            # Handle dict response with 'action' key
            if isinstance(result, dict) and "action" in result:
                return result["action"]
            return result
        except TimeoutError:
            print(f"ProtobufAgent request timed out for {self.base_url}")
            return DeadlineExceeded()
        except Exception as e:
            print(f"ProtobufAgent error: {e}")
            return None

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()
