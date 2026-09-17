# List of trial phases in chronological order
from enum import Enum

TRIAL_PHASES = ["Phase 1", "Phase 2", "Phase 3", "Approval"]


class InvestmentLevel(int, Enum):
    """
    Investment intensity levels for drug development.

    Each level affects cost, speed, success probability, capacity usage,
    and experience gain. Configuration is loaded from config.yaml.
    """

    NONE = 0  # Not investing (for idle assets)
    MINIMAL = 1  # Slow and cheap, more learning
    STANDARD = 2  # Normal development
    ACCELERATED = 3  # Fast and expensive, less learning
    STOP = 4  # Stop development early (for in-development assets only)
    DROP = 5  # Drop asset from portfolio regardless of state (agent decision)

    @classmethod
    def from_int(cls, value: int) -> "InvestmentLevel":
        """Create an InvestmentLevel from an integer."""
        for level in cls:
            if level.value == value:
                return level
        raise ValueError(f"{cls.__name__} has no value matching {value}")


MAX_NUM_ASSETS = 25

LEVELS = [
    {
        "num_assets": 5,
        "max_num_assets": 15,
        "equilibrium_num_assets": 5,
        "asset_arrival_sensitivity_below": 1.5,
        "asset_arrival_sensitivity_above": 3.0,
        "horizon": 15,
        "starting_cash": 10_000_000.0,
        "global_seed": 116739,
    },
    {
        "num_assets": 10,
        "max_num_assets": 20,
        "equilibrium_num_assets": 10,
        "asset_arrival_sensitivity_below": 1.5,
        "asset_arrival_sensitivity_above": 3.0,
        "horizon": 15,
        "starting_cash": 10_000_000.0,
        "global_seed": 256787,
    },
    {
        "num_assets": 15,
        "max_num_assets": 25,
        "equilibrium_num_assets": 15,
        "asset_arrival_sensitivity_below": 1.5,
        "asset_arrival_sensitivity_above": 3.0,
        "horizon": 15,
        "starting_cash": 10_000_000.0,
        "global_seed": 776646,
    },
]


DISCOUNT_RATE = 7.5 / 100
