import math
import uuid
from typing import Callable, Literal

import numpy as np

from pyxis_portfolio_challenge.game.asset import AssetState, DrugAsset
from pyxis_portfolio_challenge.game.constants import DISCOUNT_RATE
from pyxis_portfolio_challenge.game.game_state import GameState


def delta_npv(asset: DrugAsset) -> float:
    """
    Compute the value of a drug asset based on its NPV now compared to in one step.

    If the asset already has negative NPV, return a value of 0.

    Parameters
    ----------
    asset : DrugAsset
        The drug asset to evaluate.

    Returns
    -------
    float
        The change in NPV if the asset is delayed by one step, or zero if the NPV is
        negative.

    """
    current_npv = asset.enpv
    # Asset can only decrease NPV by delaying - so if current NPV is already
    # non-positive, we set its value to 0 to avoid investing in it
    if current_npv <= 0:
        return 0
    else:
        delayed_asset = asset.evolve()
        # This is the present value of investing in the asset now versus waiting 1 step
        return current_npv - DISCOUNT_RATE * delayed_asset.enpv


class KnapsackAgent:
    """
    Knapsack based agent that uses a knapsack solver to make decisions.

    This agent uses the 0/1 knapsack solver to optimise selection of assets at the next
    time step. Caution: it only ensures the budget constraint at the next time step,
    not the entire horizon.

    Parameters
    ----------
    units : float
        The units (e.g. millions, billions) of the costs and NPVs as used by the solver.
    include_ongoing_costs : bool
        Whether to subtract costs of assets in an ongoing trial from the budget.

    """

    def __init__(
        self,
        units: float = 1e6,
        value_function: Callable[[DrugAsset], float] = delta_npv,
        include_ongoing_costs: bool = True,
        cost_attribute: str = "remaining_trial_cost",
    ):
        """Initialize the KnapsackAgent."""
        super().__init__()
        self.units = units
        self.value_function = value_function
        self.include_ongoing_costs = include_ongoing_costs
        self.env = None
        self.cost_attribute = cost_attribute

    def set_env(self, env):
        """Set the environment for the agent."""
        self.env = env

    # Reference: https://www.w3schools.com/dsa/dsa_ref_knapsack.php
    def knapsack_01_solver(self, items, capacity):
        """
        Solves the 0-1 knapsack problem using dynamic programming.

        Parameters
        ----------
        items : list of tuple
            Each tuple contains (value, weight, id) for an item.
        capacity : int
            Maximum weight capacity of the knapsack.

        Returns
        -------
        selected_items : list of tuple
            List of selected items (tuples of (value, weight, id)).

        """
        n = len(items)

        # Create DP table where dp[i][w] represents max value with first i items and
        # weight limit w
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]

        # Fill the DP table
        for i in range(1, n + 1):
            value, weight, _ = items[i - 1]
            for w in range(1, capacity + 1):
                # Don't include current item
                dp[i][w] = dp[i - 1][w]

                # Include current item if it fits and improves the solution
                if weight <= w:
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weight] + value)

        # Backtrack to find which items were selected
        selected_items = []
        w = capacity
        for i in range(n, 0, -1):
            # If value differs from previous row, this item was included
            if dp[i][w] != dp[i - 1][w]:
                selected_items.append(items[i - 1])
                w -= items[i - 1][1]  # Reduce remaining capacity by item's weight

        return selected_items

    def make_investment_decisions(
        self, game_state: GameState
    ) -> dict[uuid.UUID, Literal["invest"]]:
        """
        Make investment decisions based on the current game state.

        Here the budget and costs are divided by the units parameter and rounded up or
        down to ensure that they are integers. The values are left as floats since these
        are handled by the knapsack solver.

        Parameters
        ----------
        game_state : GameState
            The current state of the game.

        Returns
        -------
        dict[uuid.UUID, Literal["invest"]]
           A dictionary containing the asset IDs and the corresponding actions.
           "invest" for Idle assets to start development.

        """
        budget = game_state.cash
        idle_assets = []
        investment_decisions = {}

        for asset in game_state.assets.values():
            if asset.state == AssetState.InDevelopment:
                if self.include_ongoing_costs:
                    budget -= getattr(asset, self.cost_attribute)

            elif asset.state == AssetState.Idle:
                value = self.value_function(asset)
                weight = math.ceil(getattr(asset, self.cost_attribute) / self.units)
                idle_assets.append((value, weight, asset.id))

        # CHECK: If budget is negative, we cannot invest in any assets
        if budget <= 0:
            return investment_decisions

        budget_rescaled = math.floor(budget / self.units)

        if idle_assets:
            knapsack_result = self.knapsack_01_solver(idle_assets, budget_rescaled)
            for _, _, asset_id in knapsack_result:
                investment_decisions[asset_id] = "invest"

        return investment_decisions

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Wrapper of the old logic.

        Wraps the old logic into an interface that works with the `evaluate` function.
        Selected Idle assets get the invest action (1); all others stay at 0.
        """
        game_state = self.env.unwrapped.game_state
        investment_decisions = self.make_investment_decisions(game_state)

        # reconstruct the action array
        asset_id_order = self.env.unwrapped._asset_id_order

        actions = np.zeros(len(asset_id_order))

        for i, asset_id in enumerate(asset_id_order):
            if investment_decisions.get(asset_id) == "invest":
                actions[i] = 1

        return actions

