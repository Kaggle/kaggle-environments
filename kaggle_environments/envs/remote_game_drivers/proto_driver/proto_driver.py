"""
Remote game driver environment using proto-based networking.
This environment bypasses the traditional kaggle_environments schema system
and uses proto serialization via kaggle_evaluation relay.

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

try:
    import kaggle_evaluation.core.relay as relay

    RELAY_AVAILABLE = True
except ImportError:
    relay = None
    RELAY_AVAILABLE = False

try:
    from remote_game_drivers.core.base_classes import BaseGameDriver, KaggleAgentId

    GAME_DRIVERS_AVAILABLE = True
except ImportError:
    BaseGameDriver = None
    KaggleAgentId = None
    GAME_DRIVERS_AVAILABLE = False

DEPENDENCIES_AVAILABLE = RELAY_AVAILABLE and GAME_DRIVERS_AVAILABLE


def interpreter(state, environment):
    """
    Minimal interpreter for proto-based remote game drivers.

    The actual game logic is handled by the remote game driver,
    so this interpreter just passes through the state.
    """
    return state


def renderer(state, environment):
    """
    Minimal renderer for proto-based remote game drivers.

    Rendering is typically handled by the game driver itself.
    """
    return "Remote Game Driver Environment (Proto-based)"


def html_renderer(environment):
    """
    HTML renderer for proto-based remote game drivers.
    """
    return """
    <div>
        <h3>Remote Game Driver Environment</h3>
        <p>This environment uses proto-based networking via kaggle_evaluation relay.</p>
        <p>Rendering is handled by the game driver.</p>
    </div>
    """


class ProtoGameDriverWrapper:
    """
    Wrapper to integrate remote_game_drivers with kaggle_environments using proto networking.

    This class bridges the gap between kaggle_environments and remote_game_drivers,
    using proto-based serialization instead of the traditional schema system.
    """

    def __init__(self, game_driver_class, game_name, driver_config=None):
        """
        Args:
            game_driver_class: Class that inherits from BaseGameDriver
            game_name: Name of the game to run
            driver_config: Optional configuration for the game driver
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "Required dependencies not available. Ensure kaggle_evaluation and remote_game_drivers are installed."
            )

        self.game_driver_class = game_driver_class
        self.game_name = game_name
        self.driver_config = driver_config or {}
        self.game_driver = None

    def initialize_driver(self, num_agents=2):
        """Initialize the game driver with the specified number of agents."""
        if "agent_ids" not in self.driver_config:
            self.driver_config["agent_ids"] = [f"agent_{i}" for i in range(num_agents)]

        self.game_driver = self.game_driver_class(driver_config=self.driver_config)

    def run_with_proto_agents(self, proto_agent_configs):
        """
        Run a game with proto-based agents.

        Args:
            proto_agent_configs: List of dicts with proto agent configurations
                Each dict should have: {'channel_address': str, 'port': int, ...}

        Returns:
            dict: Results mapping agent_id to score
        """
        if self.game_driver is None:
            self.initialize_driver(len(proto_agent_configs))

        # Start the game
        self.game_driver.start_new_game(self.game_name)

        # Run the game loop
        results = self.game_driver.run_game(self.game_name)

        return results


def create_proto_driver_environment(game_driver_class, game_name, driver_config=None):
    """
    Create a kaggle_environments Environment that uses proto-based networking.

    Args:
        game_driver_class: Class that inherits from BaseGameDriver
        game_name: Name of the game to run
        driver_config: Optional configuration for the game driver

    Returns:
        dict: Environment specification for kaggle_environments
    """
    wrapper = ProtoGameDriverWrapper(game_driver_class, game_name, driver_config)

    return {
        "name": f"proto_driver_{game_name}",
        "title": f"Proto Driver: {game_name}",
        "description": f"Remote game driver environment for {game_name} using proto-based networking",
        "version": "1.0.0",
        "interpreter": interpreter,
        "renderer": renderer,
        "html_renderer": html_renderer,
        "agents": {},
        "specification": {
            "agents": [2, 4],  # Support 2-4 agents by default
            "configuration": {
                "episodeSteps": {"type": "integer", "default": 1000},
                "actTimeout": {"type": "number", "default": 60},
                "runTimeout": {"type": "number", "default": 3600},
                "remainingOverageTime": {"type": "number", "default": 60},
            },
            "observation": {"type": "object"},
            "action": {"type": "object"},
            "reward": {"type": "number"},
        },
        "proto_wrapper": wrapper,
    }
