# Copyright 2020 Kaggle Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Proto-based agent implementation using kaggle_evaluation relay.py for networking.
This bypasses the traditional kaggle_environments request-based networking system.

The kaggle_evaluation package must be importable. You can either:
1. Install it via pip: pip install kaggle_evaluation
2. Set KAGGLE_EVALUATION_PATH environment variable to the hearth directory
3. Add the hearth directory to PYTHONPATH
"""

import os
import sys

from .errors import DeadlineExceeded

# Allow configuring the path via environment variable
_KAGGLE_EVALUATION_PATH = os.environ.get("KAGGLE_EVALUATION_PATH")
if _KAGGLE_EVALUATION_PATH and _KAGGLE_EVALUATION_PATH not in sys.path:
    sys.path.insert(0, _KAGGLE_EVALUATION_PATH)

try:
    import kaggle_evaluation.core.relay as relay

    RELAY_AVAILABLE = True
except ImportError:
    relay = None
    RELAY_AVAILABLE = False


class ProtoAgentError(Exception):
    """Exception raised for proto agent communication errors."""

    pass


class ProtoAgent:
    """Agent that communicates via protobuf using kaggle_evaluation relay.

    This agent uses the relay.Client for proto-based serialization/deserialization
    instead of the traditional HTTP requests library used by UrlAgent.

    Supports context manager protocol for automatic cleanup:
        with ProtoAgent(port=50051) as agent:
            action = agent(observation, configuration)
    """

    def __init__(
        self,
        channel_address="localhost",
        port=None,
        ports=None,
        environment_name=None,
        timeout_seconds=None,
        endpoint_name="process_turn",
    ):
        """Initialize a proto-based agent.

        Args:
            channel_address: Address of the server (default: 'localhost')
            port: Optional specific port to connect to
            ports: Optional list of ports to try
            environment_name: Name of the environment (for compatibility)
            timeout_seconds: Optional timeout for requests (uses relay default if None)
            endpoint_name: Name of the endpoint to call (default: 'process_turn')
        """
        if not RELAY_AVAILABLE:
            raise ImportError(
                "kaggle_evaluation.core.relay is not available. "
                "Install kaggle_evaluation or set KAGGLE_EVALUATION_PATH environment variable."
            )

        self.channel_address = channel_address
        self.port = port
        self.ports = ports
        self.environment_name = environment_name
        self.timeout_seconds = timeout_seconds
        self.endpoint_name = endpoint_name
        self._client = None

    @property
    def client(self):
        """Lazy initialization of the relay client."""
        if self._client is None:
            self._client = relay.Client(
                channel_address=self.channel_address,
                port=self.port,
                ports=self.ports,
            )
            if self.timeout_seconds is not None:
                self._client.endpoint_deadline_seconds = self.timeout_seconds
        return self._client

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False

    def __call__(self, observation, configuration):
        """Execute an action given observation and configuration.

        Args:
            observation: Current observation from the environment
            configuration: Environment configuration

        Returns:
            Action to take in the environment, or DeadlineExceeded/None on error
        """
        game_data = {
            "action_space": getattr(observation, "action_space", None),
            "observation": observation,
            "configuration": configuration,
        }

        try:
            result = self.client.send(self.endpoint_name, game_data)

            if result is None:
                return None

            if isinstance(result, dict) and "action" in result:
                return result["action"]

            return result

        except relay.GRPCDeadlineError:
            return DeadlineExceeded()
        except relay.ServerDiedError as e:
            print(f"Proto agent server died: {e}")
            return DeadlineExceeded()
        except Exception as e:
            print(f"Proto agent error ({type(e).__name__}): {e}")
            return None

    def close(self):
        """Close the client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None


def is_proto_agent_spec(raw):
    """Check if the raw agent specification is for a proto agent.

    Args:
        raw: Agent specification (could be dict, string, callable, etc.)

    Returns:
        bool: True if this is a proto agent specification
    """
    if isinstance(raw, dict):
        return raw.get("type") == "proto" or "proto_config" in raw
    return False


def build_proto_agent(raw, environment_name):
    """Build a ProtoAgent from a specification.

    Args:
        raw: Agent specification dict with proto configuration
        environment_name: Name of the environment

    Returns:
        tuple: (ProtoAgent instance, is_parallelizable=True)

    Raises:
        ValueError: If the specification is invalid
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Proto agent specification must be a dict, got {type(raw).__name__}")

    proto_config = raw.get("proto_config", {})

    agent = ProtoAgent(
        channel_address=proto_config.get("channel_address", "localhost"),
        port=proto_config.get("port"),
        ports=proto_config.get("ports"),
        environment_name=environment_name,
        timeout_seconds=proto_config.get("timeout_seconds"),
        endpoint_name=proto_config.get("endpoint_name", "process_turn"),
    )

    # Proto agents can be parallelized since they communicate over gRPC
    return agent, True
