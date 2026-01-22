"""Utilities for loading protobuf modules with runtime recompilation fallback.

In simulation contexts, all containers use the same Docker image, so we can
recompile protos at runtime if the pre-compiled versions are incompatible
with the installed protobuf version.

This module provides:
- get_proto_module(): Returns the kaggle_evaluation_pb2 module, recompiling if needed
- get_relay_functions(): Returns the _serialize and _deserialize functions from relay.py
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    pass


_proto_module: ModuleType | None = None
_relay_serialize: Callable[[Any], Any] | None = None
_relay_deserialize: Callable[[Any], Any] | None = None
_initialized: bool = False


def _find_proto_file() -> Path | None:
    """Find the kaggle_evaluation.proto file.

    Searches in order:
    1. kaggle_evaluation package (from wheel)
    2. Local game_drivers directory (fallback)
    """
    # Try to find proto file in kaggle_evaluation package
    try:
        import kaggle_evaluation

        pkg_dir = Path(kaggle_evaluation.__file__).parent
        proto_path = pkg_dir / "core" / "kaggle_evaluation.proto"
        if proto_path.exists():
            return proto_path
    except ImportError:
        pass

    # Fallback: check local game_drivers directory
    local_proto = Path(__file__).parent / "kaggle_evaluation.proto"
    if local_proto.exists():
        return local_proto

    return None


def _compile_proto_to_temp(proto_path: Path) -> Path:
    """Compile proto file to a temporary directory.

    Args:
        proto_path: Path to the .proto file

    Returns:
        Path to the temporary directory containing compiled files

    Raises:
        RuntimeError: If compilation fails
    """
    temp_dir = tempfile.mkdtemp(prefix="kaggle_proto_")
    temp_path = Path(temp_dir)

    # Create output directory structure
    output_dir = temp_path / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compile using grpc_tools.protoc
    try:
        from grpc_tools import protoc

        result = protoc.main(
            [
                "grpc_tools.protoc",
                f"--python_out={output_dir}",
                f"--grpc_python_out={output_dir}",
                f"-I{proto_path.parent}",
                str(proto_path),
            ]
        )

        if result != 0:
            raise RuntimeError(f"protoc returned non-zero exit code: {result}")

    except ImportError:
        # Fall back to subprocess if grpc_tools not available as module
        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            f"-I{proto_path.parent}",
            str(proto_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Proto compilation failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    # Create __init__.py
    (output_dir / "__init__.py").touch()

    return output_dir


def _load_module_from_path(module_name: str, file_path: Path) -> ModuleType:
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _try_import_precompiled() -> tuple[ModuleType | None, Callable | None, Callable | None]:
    """Try to import pre-compiled proto module and relay functions.

    Returns:
        Tuple of (proto_module, serialize_func, deserialize_func) or (None, None, None) if import fails
    """
    try:
        import kaggle_evaluation.core.generated.kaggle_evaluation_pb2 as proto_module
        from kaggle_evaluation.core.relay import _deserialize, _serialize

        # Verify the module works by accessing a known attribute
        _ = proto_module.KaggleEvaluationRequest

        return proto_module, _serialize, _deserialize
    except (ImportError, AttributeError, TypeError) as e:
        # Import failed or proto version mismatch
        print(f"[proto_utils] Pre-compiled proto import failed: {e}")
        return None, None, None


def _compile_and_load() -> tuple[ModuleType, Callable, Callable]:
    """Compile proto at runtime and load the resulting module.

    Returns:
        Tuple of (proto_module, serialize_func, deserialize_func)

    Raises:
        RuntimeError: If proto file not found or compilation fails
    """
    proto_path = _find_proto_file()
    if proto_path is None:
        raise RuntimeError(
            "Could not find kaggle_evaluation.proto file. "
            "Ensure kaggle_evaluation package is installed with proto files included."
        )

    print(f"[proto_utils] Compiling proto from {proto_path}")
    output_dir = _compile_proto_to_temp(proto_path)

    # Load the compiled module
    pb2_path = output_dir / "kaggle_evaluation_pb2.py"
    if not pb2_path.exists():
        raise RuntimeError(f"Compiled proto file not found at {pb2_path}")

    proto_module = _load_module_from_path(
        "kaggle_evaluation_runtime_pb2",
        pb2_path,
    )

    # Import relay functions - these should work with the runtime-compiled proto
    # because they use the proto module dynamically
    from kaggle_evaluation.core.relay import _deserialize, _serialize

    return proto_module, _serialize, _deserialize


def _initialize() -> None:
    """Initialize proto module and relay functions, with runtime compilation fallback."""
    global _proto_module, _relay_serialize, _relay_deserialize, _initialized

    if _initialized:
        return

    # First try pre-compiled
    _proto_module, _relay_serialize, _relay_deserialize = _try_import_precompiled()

    if _proto_module is None:
        # Fall back to runtime compilation
        try:
            _proto_module, _relay_serialize, _relay_deserialize = _compile_and_load()
            print("[proto_utils] Successfully compiled proto at runtime")
        except Exception as e:
            raise RuntimeError(f"Failed to load or compile proto: {e}") from e

    _initialized = True


def get_proto_module() -> ModuleType:
    """Get the kaggle_evaluation_pb2 module, recompiling if needed.

    Returns:
        The proto module with KaggleEvaluationRequest, KaggleEvaluationResponse, etc.

    Raises:
        RuntimeError: If proto cannot be loaded or compiled
    """
    _initialize()
    if _proto_module is None:
        raise RuntimeError("Proto module not initialized")
    return _proto_module


def get_relay_functions() -> tuple[Callable[[Any], Any], Callable[[Any], Any]]:
    """Get the _serialize and _deserialize functions from relay.py.

    Returns:
        Tuple of (serialize_func, deserialize_func)

    Raises:
        RuntimeError: If relay functions cannot be loaded
    """
    _initialize()
    if _relay_serialize is None or _relay_deserialize is None:
        raise RuntimeError("Relay functions not initialized")
    return _relay_serialize, _relay_deserialize
