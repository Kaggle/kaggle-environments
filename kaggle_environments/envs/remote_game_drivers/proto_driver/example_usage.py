"""
Example usage of proto-based remote game drivers with kaggle_environments.

This example demonstrates how to:
1. Create a proto-based agent specification
2. Use the ProtoAgent class for communication
3. Integrate with remote_game_drivers

Required environment variables (if packages not installed via pip):
- KAGGLE_EVALUATION_PATH: Path to hearth directory containing kaggle_evaluation
- REMOTE_GAME_DRIVERS_PATH: Path to remote_game_drivers src directory
"""

import os
import sys

# Allow configuring paths via environment variables
_KAGGLE_EVALUATION_PATH = os.environ.get("KAGGLE_EVALUATION_PATH")
_REMOTE_DRIVERS_PATH = os.environ.get("REMOTE_GAME_DRIVERS_PATH")

for path in [_KAGGLE_EVALUATION_PATH, _REMOTE_DRIVERS_PATH]:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def example_proto_agent_spec():
    """Example of how to specify a proto-based agent."""

    # Proto agent specification using dict format
    proto_agent = {
        "type": "proto",
        "proto_config": {
            "channel_address": "localhost",
            "port": 50051,  # Optional: specific port
            # 'ports': [50051, 50052, 50053],  # Alternative: list of ports to try
        },
    }

    return proto_agent


def example_gymnasium_integration():
    """
    Example of integrating gymnasium games via remote_game_drivers.

    This demonstrates the pattern used in gym_gateway.py and gym_inference_server.py
    but adapted for kaggle_environments.
    """

    try:
        from remote_game_drivers.gymnasium_remote_driver.game_driver import GameDriver

        import kaggle_environments

        # Create proto agent specifications for both players
        agent_1 = {
            "type": "proto",
            "proto_config": {
                "channel_address": "localhost",
                "port": 50051,
            },
        }

        agent_2 = {
            "type": "proto",
            "proto_config": {
                "channel_address": "localhost",
                "port": 50052,
            },
        }

        # Create environment (would need to be registered first)
        # env = kaggle_environments.make('proto_driver_gymnasium')
        # env.run([agent_1, agent_2])

        print("Example setup complete. Agents would communicate via proto.")

    except ImportError as e:
        print(f"Dependencies not available: {e}")


def example_inference_server():
    """
    Example of how to set up an inference server that works with proto agents.

    This is analogous to gym_inference_server.py but for kaggle_environments.
    """

    try:
        import kaggle_evaluation.core.relay as relay
        from remote_game_drivers.gymnasium_remote_driver.remote_agent import RemoteAgent

        # Example agent class
        class ExampleAgent(RemoteAgent):
            def choose_action(self, action_space, observation, info=None):
                # Simple random action selection
                import random

                if hasattr(action_space, "n"):
                    return random.randint(0, action_space.n - 1)
                return 0

        # Define the endpoint that will be called by the proto agent
        def process_turn(game_data):
            """
            Endpoint called by ProtoAgent via relay.Client.send('process_turn', game_data)

            Args:
                game_data: Dict with 'action_space', 'observation', 'configuration'

            Returns:
                Dict with 'action' and optionally 'reasoning'
            """
            agent = ExampleAgent()
            action_space = game_data.get("action_space")
            observation = game_data.get("observation")

            result = agent.deliver_agent_info(action_space, observation)
            return result

        # Create and start the server
        server, port = relay.define_server(process_turn, port=50051)
        print(f"Proto inference server started on port {port}")

        # In production, you would:
        # server.start()
        # server.wait_for_termination()

        return server, port

    except ImportError as e:
        print(f"Dependencies not available: {e}")
        return None, None


def example_client_usage():
    """
    Example of using the ProtoAgent directly from kaggle_environments.
    """

    try:
        from kaggle_environments.proto_agent import ProtoAgent

        # Create a proto agent
        agent = ProtoAgent(channel_address="localhost", port=50051, environment_name="example_game")

        # Simulate an observation and configuration
        observation = {
            "step": 0,
            "board": [[0, 0], [0, 0]],
            "remainingOverageTime": 60,
        }

        configuration = {
            "episodeSteps": 100,
            "actTimeout": 10,
        }

        # Get action from the agent
        # action = agent(observation, configuration)
        # print(f"Agent returned action: {action}")

        # Clean up
        agent.close()

        print("Proto agent example complete")

    except ImportError as e:
        print(f"Dependencies not available: {e}")


if __name__ == "__main__":
    print("=== Proto Agent Specification Example ===")
    spec = example_proto_agent_spec()
    print(f"Agent spec: {spec}\n")

    print("=== Gymnasium Integration Example ===")
    example_gymnasium_integration()
    print()

    print("=== Client Usage Example ===")
    example_client_usage()
    print()

    print("=== Inference Server Example ===")
    print("(Server creation example - not actually starting)")
    example_inference_server()
