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
Utilities for proto-based relay communication in kaggle_environments.
This module provides helper functions to work with kaggle_evaluation relay.

The kaggle_evaluation package must be importable. You can either:
1. Install it via pip: pip install kaggle_evaluation
2. Set KAGGLE_EVALUATION_PATH environment variable to the hearth directory
3. Add the hearth directory to PYTHONPATH
"""

import os
import sys
from typing import Any, Callable, Optional

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


def _require_relay():
    """Raise ImportError if relay is not available."""
    if not RELAY_AVAILABLE:
        raise ImportError(
            "kaggle_evaluation.core.relay is not available. "
            "Install kaggle_evaluation or set KAGGLE_EVALUATION_PATH environment variable."
        )


def create_proto_server(
    *endpoint_listeners: Callable,
    port: Optional[int] = None,
    ports: Optional[list[int]] = None,
) -> tuple:
    """Create a gRPC server for proto-based communication.

    Args:
        *endpoint_listeners: Functions that handle specific endpoints.
            Each function's __name__ becomes the endpoint name.
        port: Optional specific port to bind to
        ports: Optional list of ports to try

    Returns:
        tuple: (server, port) where server is the gRPC server and port is the bound port

    Example:
        def process_turn(game_data):
            return {'action': 0}

        server, port = create_proto_server(process_turn, port=50051)
        server.start()
    """
    _require_relay()
    return relay.define_server(*endpoint_listeners, port=port, ports=ports)


def create_proto_client(
    channel_address: str = "localhost",
    port: Optional[int] = None,
    ports: Optional[list[int]] = None,
    timeout_seconds: Optional[float] = None,
) -> "relay.Client":
    """Create a gRPC client for proto-based communication.

    Args:
        channel_address: Address of the server (default: 'localhost')
        port: Optional specific port to connect to
        ports: Optional list of ports to try
        timeout_seconds: Optional timeout for requests

    Returns:
        relay.Client: Client instance for making proto-based requests

    Example:
        client = create_proto_client(port=50051)
        result = client.send('process_turn', {'observation': obs})
        client.close()
    """
    _require_relay()
    client = relay.Client(channel_address=channel_address, port=port, ports=ports)
    if timeout_seconds is not None:
        client.endpoint_deadline_seconds = timeout_seconds
    return client


def serialize_payload(data: Any) -> Any:
    """Serialize data to proto Payload format.

    Supported types:
    - Primitives: str, int, float, bool, None
    - Collections: list, tuple, dict (with str keys)
    - Scientific: pandas.DataFrame, polars.DataFrame, numpy.ndarray
    - Data Models: Pydantic BaseModel, dataclasses, Enums
    - Binary: io.BytesIO

    Args:
        data: Python data to serialize

    Returns:
        Payload: Proto message
    """
    _require_relay()
    return relay._serialize(data)


def deserialize_payload(payload: Any) -> Any:
    """Deserialize proto Payload to Python data.

    Args:
        payload: Proto Payload message

    Returns:
        Python data (primitives, dicts, lists, dataframes, etc.)
    """
    _require_relay()
    return relay._deserialize(payload)


def set_allowed_modules(modules: Optional[list[str]]) -> None:
    """Set the allowlist of modules for data model deserialization.

    This is a security feature to prevent arbitrary code execution during
    deserialization of Pydantic models, dataclasses, and Enums.

    Args:
        modules: List of module names/prefixes allowed (e.g., ['myapp.models']),
                 or '*' to allow all modules (use with caution)

    Example:
        set_allowed_modules(['remote_game_drivers.gymnasium_remote_driver.serialization'])
    """
    _require_relay()
    relay.set_allowed_modules(modules)


def get_default_ports() -> list[int]:
    """Get the default list of gRPC ports used by relay.

    Returns:
        list[int]: Default ports that relay tries to bind to
    """
    _require_relay()
    return relay.GRPC_PORTS


def get_available_port(ports: Optional[list[int]] = None) -> int:
    """Find an available port from the given list or defaults.

    Args:
        ports: Optional list of ports to check. Uses defaults if None.

    Returns:
        int: First available port

    Raises:
        ValueError: If no ports are available
    """
    _require_relay()
    return relay._get_available_port(ports)
