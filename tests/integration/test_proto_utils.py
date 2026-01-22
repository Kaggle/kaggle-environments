"""Tests for proto_utils runtime compilation.

These tests verify that proto_utils can:
1. Load pre-compiled protos when compatible
2. Recompile protos at runtime when there's a version mismatch

The runtime compilation test requires Docker to simulate a protobuf version mismatch.
"""

import shutil

import pytest


class TestProtoUtilsBasic:
    """Basic tests that don't require Docker."""

    def test_find_proto_file(self):
        """Test that we can find the proto file."""
        from kaggle_environments.envs.game_drivers.proto_utils import _find_proto_file

        proto_path = _find_proto_file()
        # Should find either from kaggle_evaluation package or local fallback
        # May be None if kaggle_evaluation not installed without proto file
        if proto_path is not None:
            assert proto_path.exists()
            assert proto_path.name == "kaggle_evaluation.proto"

    def test_get_proto_module(self):
        """Test that we can get the proto module."""
        pytest.importorskip("kaggle_evaluation")

        from kaggle_environments.envs.game_drivers.proto_utils import get_proto_module

        proto_module = get_proto_module()
        assert proto_module is not None

        # Verify expected attributes exist
        assert hasattr(proto_module, "KaggleEvaluationRequest")
        assert hasattr(proto_module, "KaggleEvaluationResponse")
        assert hasattr(proto_module, "Payload")

    def test_get_relay_functions(self):
        """Test that we can get the relay functions."""
        pytest.importorskip("kaggle_evaluation")

        from kaggle_environments.envs.game_drivers.proto_utils import get_relay_functions

        serialize, deserialize = get_relay_functions()
        assert callable(serialize)
        assert callable(deserialize)

    def test_proto_roundtrip(self):
        """Test that we can serialize and deserialize data."""
        pytest.importorskip("kaggle_evaluation")

        from kaggle_environments.envs.game_drivers.proto_utils import (
            get_proto_module,
            get_relay_functions,
        )

        proto_module = get_proto_module()
        serialize, deserialize = get_relay_functions()

        # Test basic data roundtrip
        test_data = {"action": 42, "message": "hello"}
        payload = serialize(test_data)
        result = deserialize(payload)

        assert result == test_data

        # Test creating a request/response
        request = proto_module.KaggleEvaluationRequest(
            name="test_endpoint",
            args=[serialize("arg1"), serialize(123)],
        )
        request_bytes = request.SerializeToString()

        # Parse it back
        parsed = proto_module.KaggleEvaluationRequest()
        parsed.ParseFromString(request_bytes)

        assert parsed.name == "test_endpoint"
        assert deserialize(parsed.args[0]) == "arg1"
        assert deserialize(parsed.args[1]) == 123


class TestProtoUtilsRuntimeCompilation:
    """Tests for runtime proto compilation.

    These tests verify that proto_utils can recompile protos at runtime
    when there's a version mismatch.
    """

    def test_runtime_compilation(self):
        """Test that protos can be recompiled at runtime.

        This test verifies that _compile_proto_to_temp works correctly.
        """
        pytest.importorskip("kaggle_evaluation")
        pytest.importorskip("grpc_tools")

        # Reset the proto_utils module state
        import kaggle_environments.envs.game_drivers.proto_utils as proto_utils

        proto_utils._initialized = False
        proto_utils._proto_module = None
        proto_utils._relay_serialize = None
        proto_utils._relay_deserialize = None

        # Find the proto file first
        proto_path = proto_utils._find_proto_file()
        assert proto_path is not None, "Proto file not found"

        # Now test runtime compilation directly
        output_dir = proto_utils._compile_proto_to_temp(proto_path)
        assert output_dir.exists()

        pb2_path = output_dir / "kaggle_evaluation_pb2.py"
        assert pb2_path.exists(), f"Compiled proto not found at {pb2_path}"

        # Load the compiled module
        module = proto_utils._load_module_from_path("test_runtime_pb2", pb2_path)
        assert hasattr(module, "KaggleEvaluationRequest")
        assert hasattr(module, "KaggleEvaluationResponse")

        # Clean up
        shutil.rmtree(output_dir.parent)

    def test_full_initialization_flow(self):
        """Test the full initialization flow.

        This verifies that get_proto_module and get_relay_functions work correctly.
        """
        pytest.importorskip("kaggle_evaluation")
        pytest.importorskip("grpc_tools")

        import kaggle_environments.envs.game_drivers.proto_utils as proto_utils

        # Reset state
        proto_utils._initialized = False
        proto_utils._proto_module = None
        proto_utils._relay_serialize = None
        proto_utils._relay_deserialize = None

        # The module should initialize successfully (either pre-compiled or runtime)
        proto_module = proto_utils.get_proto_module()
        serialize, deserialize = proto_utils.get_relay_functions()

        # Verify it works
        test_data = {"test": "data", "number": 42}
        payload = serialize(test_data)
        result = deserialize(payload)
        assert result == test_data

        # Verify proto message creation works
        request = proto_module.KaggleEvaluationRequest(name="test")
        assert request.name == "test"
