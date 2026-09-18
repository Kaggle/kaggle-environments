"""Environment package public API."""

from pyxis_portfolio_challenge.environment.env_factory import (
    make_multi_agent_train_env as make_multi_agent_train_env,
)


def __getattr__(name):
    """Resolve `SelfPlayWrapper` on first use.

    `self_play` imports torch and stable-baselines3; importing it eagerly here
    would drag them into every consumer of the game engine.
    """
    if name == "SelfPlayWrapper":
        from pyxis_portfolio_challenge.environment.self_play import SelfPlayWrapper

        return SelfPlayWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# NOTE: `evaluate` cannot be re-exported here because the `evaluate.py`
# submodule shadows function-level imports. Import directly:
#   from pyxis_portfolio_challenge.environment.competition import evaluate

__all__ = [
    "SelfPlayWrapper",
    "make_multi_agent_train_env",
]
