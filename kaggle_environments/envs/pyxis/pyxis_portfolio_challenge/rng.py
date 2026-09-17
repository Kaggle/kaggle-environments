import contextvars
import random

_game_rng: contextvars.ContextVar[random.Random] = contextvars.ContextVar("game_rng")
_game_seed: contextvars.ContextVar[int] = contextvars.ContextVar("game_seed")


def init_game_rng(seed: int) -> random.Random:
    """Initialize the game-wide RNG. Call once per episode."""
    rng = random.Random(seed)
    _game_rng.set(rng)
    _game_seed.set(seed)
    return rng


def get_game_rng() -> random.Random:
    """Get the current game-wide RNG. Raises if not initialized."""
    try:
        return _game_rng.get()
    except LookupError:
        raise RuntimeError("Game RNG not initialized. Call init_game_rng(seed) first.")


def get_game_seed() -> int:
    """
    Get the seed used to initialize the current game-wide RNG.

    Raises if not initialized.
    """
    try:
        return _game_seed.get()
    except LookupError:
        raise RuntimeError("Game RNG not initialized. Call init_game_rng(seed) first.")
