"""
Game drivers module for kaggle-environments.

This module contains integrations with remote game drivers from Hearth,
enabling support for Shimmy (OpenSpiel), PettingZoo, and Gymnasium games.
"""

import subprocess
from importlib import import_module
from importlib.util import find_spec
from os import listdir
from pathlib import Path


def _install_wheel_if_needed():
    """Install kaggle_evaluation wheel at runtime if not already installed."""
    if find_spec("kaggle_evaluation") is not None:
        return  # Already installed

    # Find wheel file in remote_game_driver_wheels subdirectory
    wheel_dir = Path(__file__).parent / "remote_game_driver_wheels"
    if not wheel_dir.exists():
        raise ImportError(
            f"kaggle_evaluation not installed and wheel directory not found: {wheel_dir}\n"
            f"Expected wheel at: {wheel_dir}/kaggle_evaluation-*.whl"
        )

    wheel_files = list(wheel_dir.glob("kaggle_evaluation-*.whl"))
    if not wheel_files:
        raise ImportError(
            f"kaggle_evaluation not installed and no wheel found in: {wheel_dir}\n"
            f"Expected wheel file: kaggle_evaluation-*.whl"
        )

    wheel_path = wheel_files[0]

    # Install the wheel using uv pip, falling back to pip if uv is not available
    try:
        subprocess.check_call(
            ["uv", "pip", "install", "--quiet", str(wheel_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fall back to regular pip
        subprocess.check_call(
            ["pip", "install", "--quiet", str(wheel_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


# Install wheel at module import time if needed
_install_wheel_if_needed()


class _LazyGameDriverRegistry(dict):
    """Lazy-loading registry that discovers game driver environments on first access."""

    def __init__(self):
        super().__init__()
        self._loaded = False

    def _load_environments(self):
        """Discover and register game driver environments."""
        if self._loaded:
            return

        self._loaded = True
        module_dir = Path(__file__).parent

        for name in listdir(module_dir):
            env_dir = module_dir / name
            if not env_dir.is_dir() or name.startswith("_") or name.startswith("."):
                continue

            # Skip the wheel directory
            if name == "remote_game_driver_wheels":
                continue

            # Import the environment module - will raise ImportError if dependencies missing
            env_module = import_module(f".{name}.{name}", __package__)

            # Register the environment
            self[name] = {
                "agents": getattr(env_module, "agents", {}),
                "html_renderer": getattr(env_module, "html_renderer", None),
                "interpreter": getattr(env_module, "interpreter"),
                "renderer": getattr(env_module, "renderer"),
                "specification": getattr(env_module, "specification"),
            }

    def __getitem__(self, key):
        self._load_environments()
        return super().__getitem__(key)

    def __iter__(self):
        self._load_environments()
        return super().__iter__()

    def __len__(self):
        self._load_environments()
        return super().__len__()

    def keys(self):
        self._load_environments()
        return super().keys()

    def values(self):
        self._load_environments()
        return super().values()

    def items(self):
        self._load_environments()
        return super().items()


# Registry of game driver environments (lazy-loaded)
GAME_DRIVER_ENVS = _LazyGameDriverRegistry()

__all__ = ["GAME_DRIVER_ENVS", "agent_server"]
