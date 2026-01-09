"""
End-to-end integration test for proto-based remote game drivers.

This test demonstrates the complete flow:
1. Start inference servers for agents
2. Create proto agent specifications
3. Run a game using kaggle_environments

Required environment variables (if packages not installed via pip):
- KAGGLE_EVALUATION_PATH: Path to hearth directory containing kaggle_evaluation
- REMOTE_GAME_DRIVERS_PATH: Path to remote_game_drivers src directory
"""

import os
import sys
import time

# Allow configuring paths via environment variables
_KAGGLE_EVALUATION_PATH = os.environ.get("KAGGLE_EVALUATION_PATH")
_REMOTE_DRIVERS_PATH = os.environ.get("REMOTE_GAME_DRIVERS_PATH")

for path in [_KAGGLE_EVALUATION_PATH, _REMOTE_DRIVERS_PATH]:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def test_proto_agent_basic():
    """Test basic ProtoAgent functionality."""
    print("=== Testing ProtoAgent Basic Functionality ===")

    try:
        from kaggle_environments.proto_agent import ProtoAgent, is_proto_agent_spec

        # Test is_proto_agent_spec
        assert is_proto_agent_spec({"type": "proto", "proto_config": {}})
        assert is_proto_agent_spec({"proto_config": {"port": 50051}})
        assert not is_proto_agent_spec({"type": "url"})
        assert not is_proto_agent_spec("http://example.com")

        print("✓ is_proto_agent_spec works correctly")

        # Test ProtoAgent creation (without actually connecting)
        agent = ProtoAgent(channel_address="localhost", port=50051)
        assert agent.client is not None
        agent.close()

        print("✓ ProtoAgent can be created")
        print("✓ Basic functionality test passed\n")
        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}\n")
        return False


def test_proto_relay_utilities():
    """Test proto_relay utility functions."""
    print("=== Testing Proto Relay Utilities ===")

    try:
        from kaggle_environments.proto_relay import (
            RELAY_AVAILABLE,
            deserialize_payload,
            serialize_payload,
        )

        if not RELAY_AVAILABLE:
            print("⚠ Relay not available, skipping serialization tests")
            return True

        # Test serialization/deserialization
        test_data = {
            "string": "hello",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": [2, 3]},
        }

        payload = serialize_payload(test_data)
        result = deserialize_payload(payload)

        assert result == test_data
        print("✓ Serialization/deserialization works correctly")
        print("✓ Proto relay utilities test passed\n")
        return True

    except Exception as e:
        print(f"✗ Proto relay utilities test failed: {e}\n")
        return False


def test_inference_server_creation():
    """Test inference server creation and basic operation."""
    print("=== Testing Inference Server Creation ===")

    try:
        import kaggle_evaluation.core.relay as relay

        # Define a simple endpoint
        def test_endpoint(data):
            return {"result": f"Received: {data}"}

        # Create server
        server, port = relay.define_server(test_endpoint, port=None)
        assert server is not None
        assert port is not None

        print(f"✓ Server created successfully on port {port}")

        # Start server in background
        server.start()
        time.sleep(0.5)  # Give server time to start

        # Create client and test communication
        client = relay.Client(channel_address="localhost", port=port)
        response = client.send("test_endpoint", "hello")

        assert response == {"result": "Received: hello"}
        print("✓ Client-server communication works")

        # Cleanup
        client.close()
        server.stop(grace=1)

        print("✓ Inference server test passed\n")
        return True

    except Exception as e:
        print(f"✗ Inference server test failed: {e}\n")
        return False


def test_proto_agent_with_server():
    """Test ProtoAgent communicating with an inference server."""
    print("=== Testing ProtoAgent with Inference Server ===")

    try:
        import kaggle_evaluation.core.relay as relay

        from kaggle_environments.proto_agent import ProtoAgent

        # Define process_turn endpoint
        def process_turn(game_data):
            action_space = game_data.get("action_space")
            observation = game_data.get("observation")

            # Simple logic: return action 0
            return {"action": 0, "reasoning": f"Received observation: {observation}"}

        # Start server
        server, port = relay.define_server(process_turn, port=None)
        server.start()
        time.sleep(0.5)

        print(f"✓ Server started on port {port}")

        # Create ProtoAgent
        agent = ProtoAgent(channel_address="localhost", port=port)

        # Simulate a game turn
        observation = {"step": 0, "board": [0, 0, 0]}
        configuration = {"episodeSteps": 100, "actTimeout": 10}

        action = agent(observation, configuration)

        assert action == 0
        print("✓ ProtoAgent successfully communicated with server")

        # Cleanup
        agent.close()
        server.stop(grace=1)

        print("✓ ProtoAgent with server test passed\n")
        return True

    except Exception as e:
        print(f"✗ ProtoAgent with server test failed: {e}\n")
        return False


def test_build_agent_integration():
    """Test that build_agent correctly handles proto agent specs."""
    print("=== Testing build_agent Integration ===")

    try:
        from kaggle_environments.agent import build_agent

        # Test proto agent spec
        proto_spec = {
            "type": "proto",
            "proto_config": {
                "channel_address": "localhost",
                "port": 50051,
            },
        }

        agent, is_parallelizable = build_agent(proto_spec, {}, "test_env")

        assert agent is not None
        assert is_parallelizable is True
        print("✓ build_agent correctly creates ProtoAgent")

        # Cleanup
        if hasattr(agent, "close"):
            agent.close()

        print("✓ build_agent integration test passed\n")
        return True

    except Exception as e:
        print(f"✗ build_agent integration test failed: {e}\n")
        return False


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("PROTO-BASED REMOTE GAME DRIVERS INTEGRATION TESTS")
    print("=" * 60 + "\n")

    results = []

    # Run tests
    results.append(("Basic ProtoAgent", test_proto_agent_basic()))
    results.append(("Proto Relay Utilities", test_proto_relay_utilities()))
    results.append(("Inference Server Creation", test_inference_server_creation()))
    results.append(("ProtoAgent with Server", test_proto_agent_with_server()))
    results.append(("build_agent Integration", test_build_agent_integration()))

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:.<40} {status}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60 + "\n")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
