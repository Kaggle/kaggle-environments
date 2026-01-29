"""
Vendored Reinforce Tactics game engine for Kaggle Environments.

This is a self-contained copy of the core game engine, free of
UI/pygame dependencies, intended for use inside the kaggle-environments
repository.
"""
from .core.game_state import GameState
from .constants import UNIT_DATA

__all__ = ['GameState', 'UNIT_DATA']
