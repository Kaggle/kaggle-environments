"""
Game drivers module for kaggle-environments.

This module contains integrations with remote game drivers from Hearth,
enabling support for Shimmy (OpenSpiel), PettingZoo, and Gymnasium games.
"""

from importlib import import_module
from os import listdir
from pathlib import Path

# Registry of game driver environments
GAME_DRIVER_ENVS = {}

# Auto-discover and register game driver environments
_module_dir = Path(__file__).parent

for name in listdir(_module_dir):
    env_dir = _module_dir / name
    if not env_dir.is_dir() or name.startswith("_") or name.startswith("."):
        continue

    # Import the environment module - will raise ImportError if dependencies missing
    env_module = import_module(f".{name}.{name}", __package__)

    # Register the environment
    GAME_DRIVER_ENVS[name] = {
        "agents": getattr(env_module, "agents", {}),
        "html_renderer": getattr(env_module, "html_renderer", None),
        "interpreter": getattr(env_module, "interpreter"),
        "renderer": getattr(env_module, "renderer"),
        "specification": getattr(env_module, "specification"),
    }

__all__ = ["GAME_DRIVER_ENVS", "agent_server"]
