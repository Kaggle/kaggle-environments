import os

import upath

# has to be defined before any relative imports
PROJECT_ROOT = upath.UPath(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
)

from pyxis_portfolio_challenge.environment.env_factory import (  # noqa: E402
    make_multi_agent_train_env,
    make_train_env,
)
from pyxis_portfolio_challenge.environment.evaluate import (  # noqa: E402
    evaluate,
    parallel_evaluate,
)

__all__ = [
    "evaluate",
    "parallel_evaluate",
    "make_train_env",
    "make_multi_agent_train_env",
    "PROJECT_ROOT",
]
