"""
Template for creating a proto-based inference server for kaggle_environments.

This template shows how to create an inference server that:
1. Receives game data via proto serialization
2. Processes it with your agent logic
3. Returns actions via proto serialization

Based on the pattern from hearth/game_drivers/gymnasium/gym_inference_server.py

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

import kaggle_evaluation.core.relay as relay


class ProtoInferenceServer:
    """
    Base class for proto-based inference servers.

    Subclass this and implement the process_turn method to create your agent.
    """

    def __init__(self, port=None, ports=None):
        """
        Initialize the inference server.

        Args:
            port: Optional specific port to bind to
            ports: Optional list of ports to try
        """
        self.port = port
        self.ports = ports
        self.server = None
        self.bound_port = None

    def process_turn(self, game_data):
        """
        Process a turn and return an action.

        This method should be overridden by subclasses.

        Args:
            game_data: Dict containing game state information.
                       Typical keys: 'action_space', 'observation', 'configuration'

        Returns:
            Dict with 'action' and optionally 'reasoning' keys
        """
        raise NotImplementedError("Subclasses must implement process_turn")

    def start(self):
        """Start the inference server."""
        # Create the server with the process_turn endpoint
        self.server, self.bound_port = relay.define_server(self.process_turn, port=self.port, ports=self.ports)

        print(f"Proto inference server started on port {self.bound_port}")
        self.server.start()
        return self.bound_port

    def wait_for_termination(self):
        """Block until the server is terminated."""
        if self.server:
            self.server.wait_for_termination()

    def stop(self):
        """Stop the inference server."""
        if self.server:
            self.server.stop(grace=5)


# Example 1: Simple random agent
class RandomAgentServer(ProtoInferenceServer):
    """Example: Random action selection agent."""

    def process_turn(self, game_data):
        import random

        action_space = game_data.get("action_space")

        # Simple random action based on action space
        if action_space is not None and hasattr(action_space, "n"):
            action = random.randint(0, action_space.n - 1)
        else:
            action = 0

        return {"action": action, "reasoning": "Random action selection"}


# Example 2: Agent using remote_game_drivers base classes
class RemoteGameDriverAgentServer(ProtoInferenceServer):
    """
    Example: Agent using remote_game_drivers RemoteAgent base class.

    This pattern matches gym_inference_server.py
    """

    def __init__(self, agent_class, port=None, ports=None):
        """
        Args:
            agent_class: Class that inherits from BaseRemoteAgent
            port: Optional specific port
            ports: Optional list of ports
        """
        super().__init__(port=port, ports=ports)
        self.agent_class = agent_class
        self.agent = None

    def process_turn(self, game_data):
        """
        Process turn using RemoteAgent pattern.

        Args:
            game_data: Dict with 'action_space', 'observation', and optional kwargs

        Returns:
            Dict with 'action' and optionally 'reasoning'
        """
        # Lazy initialization of agent
        if self.agent is None:
            self.agent = self.agent_class()

        # Extract standard fields
        action_space = game_data.get("action_space")
        observation = game_data.get("observation")

        # Extract any additional kwargs (e.g., 'info' for OpenSpiel)
        kwargs = {k: v for k, v in game_data.items() if k not in ["action_space", "observation"]}

        # Use the agent's deliver_agent_info method
        # This handles deserialization and calls choose_action
        return self.agent.deliver_agent_info(action_space, observation, **kwargs)


# Example 3: Custom agent with state
class StatefulAgentServer(ProtoInferenceServer):
    """Example: Agent that maintains state across turns."""

    def __init__(self, port=None, ports=None):
        super().__init__(port=port, ports=ports)
        self.turn_count = 0
        self.action_history = []

    def process_turn(self, game_data):
        self.turn_count += 1

        observation = game_data.get("observation")
        action_space = game_data.get("action_space")

        # Example: Alternate between actions
        if action_space is not None and hasattr(action_space, "n"):
            action = self.turn_count % action_space.n
        else:
            action = 0

        self.action_history.append(action)

        return {"action": action, "reasoning": f"Turn {self.turn_count}, history length: {len(self.action_history)}"}


# Main execution example
def main():
    """
    Example of how to run an inference server.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Proto-based inference server")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument(
        "--agent-type", type=str, default="random", choices=["random", "stateful"], help="Type of agent to run"
    )
    args = parser.parse_args()

    # Create the appropriate server
    if args.agent_type == "random":
        server = RandomAgentServer(port=args.port)
    elif args.agent_type == "stateful":
        server = StatefulAgentServer(port=args.port)
    else:
        raise ValueError(f"Unknown agent type: {args.agent_type}")

    # Start and run
    try:
        server.start()
        print("Server running. Press Ctrl+C to stop.")
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop()


if __name__ == "__main__":
    main()
