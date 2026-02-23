"""
Vendored Reinforce Tactics game engine for Kaggle Environments.

This is a self-contained copy of the core game engine, free of
UI/pygame dependencies, intended for use inside the kaggle-environments
repository.
"""

from .constants import UNIT_DATA
from .core.game_state import GameState

__all__ = ["GameState", "UNIT_DATA"]
