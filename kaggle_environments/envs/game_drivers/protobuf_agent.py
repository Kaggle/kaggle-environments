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

from kaggle_environments.agent import calculate_request_timeout
from kaggle_environments.errors import DeadlineExceeded


class ProtobufAgent:
    """Agent that communicates via HTTP with protobuf serialization.

    Implements RemoteAgentProtocol to be compatible with UrlAgent.
    Uses HTTPClient with use_root_endpoint=True to POST to /.
    """

    def __init__(self, base_url: str, environment_name: str | None = None):
        """
        Args:
            base_url: Base URL of the agent server (e.g., 'http://localhost:8080')
            environment_name: Name of the environment (for compatibility with UrlAgent)
        """
        self.raw = base_url  # Required by RemoteAgentProtocol
        self.base_url = base_url
        self.environment_name = environment_name
        self.client = HTTPClient(base_url=base_url, use_root_endpoint=True)

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
            # Update timeout dynamically based on remaining time
            timeout = calculate_request_timeout(observation, configuration)
            self.client.set_timeout(timeout)

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


def is_proto_agent_spec(raw: Any) -> bool:
    """Check if the raw agent specification is for a proto agent.
    Args:
        raw: Agent specification (could be dict, string, callable, etc.)
    Returns:
        bool: True if this is a proto agent specification
    """
    if isinstance(raw, dict):
        return raw.get("type") == "proto" or "proto_config" in raw
    return False


def build_proto_agent(raw: Any, environment_name: str) -> tuple[ProtobufAgent, bool]:
    """Build a ProtoAgent from a specification.
    Args:
        raw: Agent specification dict with proto configuration.
            Expected format:
            {
                'type': 'proto',
                'url': 'http://localhost:8080/agent'  # Required: HTTP endpoint URL
            }
            OR
            {
                'type': 'proto',
                'proto_config': {
                    'url': 'http://localhost:8080/agent'  # Required: HTTP endpoint URL
                }
            }
        environment_name: Name of the environment
    Returns:
        tuple: (ProtoAgent instance, is_parallelizable=True)
    Raises:
        ValueError: If the specification is invalid or missing URL
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Proto agent specification must be a dict, got {type(raw).__name__}")

    # Support both direct 'url' key and nested 'proto_config.url'
    url = raw.get("url")
    if url is None:
        proto_config = raw.get("proto_config", {})
        url = proto_config.get("url")

    if url is None:
        raise ValueError("Proto agent specification must include 'url' or 'proto_config.url'")

    agent = ProtobufAgent(url=url, environment_name=environment_name)

    # Proto agents can be parallelized since they communicate over HTTP
    return agent, True
