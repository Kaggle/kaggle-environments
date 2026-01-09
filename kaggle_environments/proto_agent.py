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
Proto-based agent implementation using HTTP requests with proto byte serialization.

This agent uses the existing HTTP request system (like UrlAgent) but serializes
data using kaggle_evaluation.core.relay proto serialization instead of JSON.
This provides richer type support (DataFrames, numpy arrays, etc.) while
maintaining compatibility with the existing networking infrastructure.

The kaggle_evaluation package must be importable. You can either:
1. Install it via pip: pip install kaggle_evaluation
2. Set KAGGLE_EVALUATION_PATH environment variable to the hearth directory
3. Add the hearth directory to PYTHONPATH
"""

import os
import sys

import requests
from requests.exceptions import Timeout

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
    """Agent that communicates via HTTP using proto byte serialization.

    This agent uses the existing HTTP request infrastructure (like UrlAgent)
    but serializes/deserializes data using kaggle_evaluation.core.relay
    proto serialization instead of JSON. This provides support for rich
    Python types like DataFrames, numpy arrays, Pydantic models, etc.

    The server endpoint should accept POST requests with:
    - Content-Type: application/x-protobuf
    - Body: serialized proto bytes from relay._serialize()

    And return:
    - Content-Type: application/x-protobuf
    - Body: serialized proto bytes that deserialize to {'action': ...}
    """

    def __init__(self, url, environment_name=None):
        """Initialize a proto-based agent.

        Args:
            url: HTTP URL of the inference server endpoint
            environment_name: Name of the environment (for compatibility)
        """
        if not RELAY_AVAILABLE:
            raise ImportError(
                "kaggle_evaluation.core.relay is not available. "
                "Install kaggle_evaluation or set KAGGLE_EVALUATION_PATH environment variable."
            )

        self.url = url
        self.environment_name = environment_name

    def __call__(self, observation, configuration):
        """Execute an action given observation and configuration.

        Args:
            observation: Current observation from the environment
            configuration: Environment configuration

        Returns:
            Action to take in the environment, or DeadlineExceeded/None on error
        """
        # Prepare the game data - same structure as UrlAgent but will be proto-serialized
        game_data = {
            "action": "act",
            "configuration": configuration,
            "environment": self.environment_name,
            "state": {
                "observation": observation,
            },
        }

        # Serialize to proto bytes using relay
        proto_bytes = relay._serialize(game_data).SerializeToString()

        # Calculate timeout like UrlAgent does
        timeout = float(observation.remainingOverageTime) + float(configuration.actTimeout) + 1

        try:
            response = requests.post(
                url=self.url,
                data=proto_bytes,
                headers={"Content-Type": "application/x-protobuf"},
                timeout=timeout,
            )
            response.raise_for_status()

            # Deserialize the proto response
            from kaggle_evaluation.core import kaggle_evaluation_pb2

            response_payload = kaggle_evaluation_pb2.Payload()
            response_payload.ParseFromString(response.content)
            result = relay._deserialize(response_payload)

            if result is None:
                return None

            # Handle action extraction like UrlAgent
            if isinstance(result, dict) and "action" in result:
                action = result["action"]
                if action == "DeadlineExceeded":
                    return DeadlineExceeded()
                elif isinstance(action, str) and action.startswith("BaseException::"):
                    parts = action.split("::", 1)
                    return BaseException(parts[1])
                return action

            return result

        except Timeout:
            print(f"Proto agent request timed out after {timeout} seconds")
            return DeadlineExceeded()
        except requests.exceptions.RequestException as e:
            print(f"Proto agent request error: {e}")
            return DeadlineExceeded()
        except Exception as e:
            print(f"Proto agent error ({type(e).__name__}): {e}")
            return None


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

    agent = ProtoAgent(url=url, environment_name=environment_name)

    # Proto agents can be parallelized since they communicate over HTTP
    return agent, True
