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
Utilities for proto-based serialization in kaggle_environments.

This module provides helper functions to serialize/deserialize data using
kaggle_evaluation.core.relay proto format. The serialized bytes can be
sent over HTTP (or any transport) and deserialized on the other end.

Supported types:
- Primitives: str, int, float, bool, None
- Collections: list, tuple, dict (with str keys)
- Scientific: pandas.DataFrame, polars.DataFrame, numpy.ndarray
- Data Models: Pydantic BaseModel, dataclasses, Enums
- Binary: io.BytesIO

The kaggle_evaluation package must be importable. You can either:
1. Install it via pip: pip install kaggle_evaluation
2. Set KAGGLE_EVALUATION_PATH environment variable to the hearth directory
3. Add the hearth directory to PYTHONPATH
"""

import os
import sys
from typing import Any, Optional

# Allow configuring the path via environment variable
_KAGGLE_EVALUATION_PATH = os.environ.get("KAGGLE_EVALUATION_PATH")
if _KAGGLE_EVALUATION_PATH and _KAGGLE_EVALUATION_PATH not in sys.path:
    sys.path.insert(0, _KAGGLE_EVALUATION_PATH)

try:
    import kaggle_evaluation.core.relay as relay
    from kaggle_evaluation.core import kaggle_evaluation_pb2

    RELAY_AVAILABLE = True
except ImportError:
    relay = None
    kaggle_evaluation_pb2 = None
    RELAY_AVAILABLE = False


def _require_relay():
    """Raise ImportError if relay is not available."""
    if not RELAY_AVAILABLE:
        raise ImportError(
            "kaggle_evaluation.core.relay is not available. "
            "Install kaggle_evaluation or set KAGGLE_EVALUATION_PATH environment variable."
        )


def serialize_to_bytes(data: Any) -> bytes:
    """Serialize Python data to proto bytes for HTTP transport.

    This is the primary function for serializing data to send over HTTP.
    The bytes can be sent as the body of an HTTP POST request with
    Content-Type: application/x-protobuf.

    Args:
        data: Python data to serialize (dict, list, DataFrame, etc.)

    Returns:
        bytes: Serialized proto bytes ready for HTTP transport

    Example:
        game_data = {'observation': obs, 'configuration': config}
        proto_bytes = serialize_to_bytes(game_data)
        response = requests.post(url, data=proto_bytes,
                                 headers={'Content-Type': 'application/x-protobuf'})
    """
    _require_relay()
    payload = relay._serialize(data)
    return payload.SerializeToString()


def deserialize_from_bytes(proto_bytes: bytes) -> Any:
    """Deserialize proto bytes back to Python data.

    This is the primary function for deserializing data received over HTTP.
    Use this to parse the body of an HTTP response with proto content.

    Args:
        proto_bytes: Serialized proto bytes (from HTTP response body)

    Returns:
        Python data (dict, list, DataFrame, etc.)

    Example:
        response = requests.post(url, data=request_bytes, ...)
        result = deserialize_from_bytes(response.content)
        action = result.get('action')
    """
    _require_relay()
    payload = kaggle_evaluation_pb2.Payload()
    payload.ParseFromString(proto_bytes)
    return relay._deserialize(payload)


def serialize_payload(data: Any) -> Any:
    """Serialize data to proto Payload message object.

    Lower-level function that returns the Payload proto message
    rather than bytes. Use serialize_to_bytes() for HTTP transport.

    Args:
        data: Python data to serialize

    Returns:
        Payload: Proto message object
    """
    _require_relay()
    return relay._serialize(data)


def deserialize_payload(payload: Any) -> Any:
    """Deserialize proto Payload message object to Python data.

    Lower-level function that takes a Payload proto message.
    Use deserialize_from_bytes() for HTTP transport.

    Args:
        payload: Proto Payload message object

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
