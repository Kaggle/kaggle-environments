# List of trial phases in chronological order
from enum import Enum, auto

TRIAL_PHASES = ["Phase 1", "Phase 2", "Phase 3", "Approval"]


class InvestmentAction(Enum):
    """
    Discrete per-asset investor action applied in a single step.

    The member values are opaque identifiers — no code relies on them.
    """

    NONE = auto()  # Do not invest (for idle assets)
    INVEST = auto()  # Invest in an idle asset (start development)
    STOP = auto()  # Stop an in-development trial early
    DROP = auto()  # Voluntarily drop an asset from the portfolio


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
