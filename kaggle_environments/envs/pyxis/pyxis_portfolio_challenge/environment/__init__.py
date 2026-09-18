"""Environment package public API."""

from pyxis_portfolio_challenge.environment.env_factory import (
    make_multi_agent_train_env as make_multi_agent_train_env,
)
from pyxis_portfolio_challenge.environment.self_play import (
    SelfPlayWrapper as SelfPlayWrapper,
)

# NOTE: `evaluate` cannot be re-exported here because the `evaluate.py`
# submodule shadows function-level imports. Import directly:
#   from pyxis_portfolio_challenge.environment.competition import evaluate

__all__ = [
    "SelfPlayWrapper",
    "make_multi_agent_train_env",
]
