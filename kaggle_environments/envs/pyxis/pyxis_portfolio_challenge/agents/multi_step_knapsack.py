"""
Multi-step knapsack agent for investment decisions.

This agent improves on the single-step KnapsackAgent by considering future cash
constraints and investment timing over a configurable lookahead horizon.
"""

import math
import uuid
from typing import Literal, Optional

import numpy as np

from pyxis_portfolio_challenge.game.asset import AssetState, DrugAsset
from pyxis_portfolio_challenge.game.constants import DISCOUNT_RATE
from pyxis_portfolio_challenge.game.game_state import GameState


def n_step_value(
    asset: DrugAsset, lookahead: int, discount_rate: float = DISCOUNT_RATE
) -> float:
    """
    Compute the N-step value of a drug asset.

    This measures the cost of delaying investment by `lookahead` steps,
    accounting for the time value of money.

    Parameters
    ----------
    asset : DrugAsset
        The drug asset to evaluate.
    lookahead : int
        Number of steps to look ahead.
    discount_rate : float
        The discount rate per time step.

    Returns
    -------
    float
        The value of investing now versus waiting `lookahead` steps.
        Zero if current eNPV is non-positive.

    """
    current_enpv = asset.enpv

    # If current eNPV is non-positive, no value in investing
    if current_enpv <= 0:
        return 0.0

    # Evolve the asset N steps to see what happens if we delay
    delayed_asset = asset
    for _ in range(lookahead):
        delayed_asset = delayed_asset.evolve()
        # If asset expires or fails during delay, future value is 0
        if delayed_asset.state in (AssetState.Expired, AssetState.Failed):
            # Full value of investing now since waiting means losing the asset
            return current_enpv

    # Calculate discounted future eNPV
    discount_factor = (1 + discount_rate) ** lookahead
    future_enpv_discounted = delayed_asset.enpv / discount_factor

    # Value is the difference: how much we gain by investing now vs later
    return current_enpv - future_enpv_discounted


def project_cash_flows(
    game_state: GameState,
    lookahead: int,
    reinvestment_percentage: Optional[float] = None,
) -> list[dict]:
    """
    Project cash flows for the next N steps assuming no new investments.

    This provides a deterministic (expected) projection of cash flows from
    assets currently in development or on market, without making new investments.

    Parameters
    ----------
    game_state : GameState
        The current game state.
    lookahead : int
        Number of steps to project ahead.
    reinvestment_percentage : float, optional
        Fraction of revenue available for reinvestment. Defaults to game_state value.

    Returns
    -------
    list[dict]
        List of projections for each future step, containing:
        - 'step': The future step number (1, 2, ...)
        - 'ongoing_costs': Expected costs from in-development assets
        - 'expected_revenue': Expected revenue from on-market assets
        - 'net_cash_flow': Net expected cash flow
        - 'projected_cash': Projected cash at end of step

    """
    if reinvestment_percentage is None:
        reinvestment_percentage = game_state.reinvestment_percentage

    projections = []
    projected_cash = game_state.cash

    # Track assets and their states for projection
    # We use expected values, not stochastic evolution
    projected_assets = {}
    for asset_id, asset in game_state.assets.items():
        projected_assets[asset_id] = {
            "asset": asset,
            "state": asset.state,
            "time_remaining": asset.trial.time_remaining if asset.trial else 0,
            "cost_this_step": asset.cost_this_step,
            "revenue_this_step": asset.revenue_this_step,
            "ptrs": asset.trial.ptrs if asset.trial else 1.0,
            "accumulated_prob": 1.0,  # Probability of reaching this state
        }

    for step in range(1, lookahead + 1):
        ongoing_costs = 0.0
        expected_revenue = 0.0

        for asset_id, proj in list(projected_assets.items()):
            if proj["state"] == AssetState.InDevelopment:
                # Pay costs (weighted by probability of being in this state)
                ongoing_costs += proj["cost_this_step"] * proj["accumulated_prob"]

                # Simulate trial time decrement
                proj["time_remaining"] -= 1
                if proj["time_remaining"] <= 0:
                    # Trial phase completes
                    # Asset has ptrs probability of continuing to next phase or market
                    proj["accumulated_prob"] *= proj["ptrs"]

                    # Check if there's a next trial
                    asset = proj["asset"]
                    if asset.trial and asset.trial.next_trial_on_success:
                        next_trial = asset.trial.next_trial_on_success
                        proj["time_remaining"] = next_trial.time_remaining
                        proj["cost_this_step"] = next_trial.cost_this_step
                        proj["ptrs"] = next_trial.ptrs
                    else:
                        # Asset reaches market
                        proj["state"] = AssetState.OnMarket
                        proj["time_on_market"] = 0
                        proj["cost_this_step"] = 0.0

            elif proj["state"] == AssetState.OnMarket:
                # Collect revenue (weighted by probability)
                expected_revenue += proj["revenue_this_step"] * proj["accumulated_prob"]

                # Update time on market for next iteration's revenue calculation
                if "time_on_market" not in proj:
                    proj["time_on_market"] = proj["asset"].time_on_market
                proj["time_on_market"] += 1

                # Update revenue for next step
                asset = proj["asset"]
                time_ratio = proj["time_on_market"] / (asset.time_until_max_revenue + 1)
                multiplier = min(time_ratio, 1)
                proj["revenue_this_step"] = multiplier * asset.max_revenue

        # Calculate net cash flow and projected cash
        cash_from_revenue = expected_revenue * reinvestment_percentage
        net_cash_flow = cash_from_revenue - ongoing_costs
        projected_cash += net_cash_flow

        projections.append({
            "step": step,
            "ongoing_costs": ongoing_costs,
            "expected_revenue": expected_revenue,
            "net_cash_flow": net_cash_flow,
            "projected_cash": projected_cash,
        })

    return projections


def compute_future_impact_score(
    asset: DrugAsset,
    projections: list[dict],
    safety_margin: float,
) -> float:
    """
    Compute how much an asset's ongoing costs would impact future cash.

    Returns a score from 0.0 to 1.0:
    - 1.0: No negative impact on projected cash (always above safety margin)
    - 0.0: Would cause bankruptcy in at least one projected step
    - 0.0-1.0: Scaled by severity of cash constraint violations

    Parameters
    ----------
    asset : DrugAsset
        The idle asset being considered for investment.
    projections : list[dict]
        Cash flow projections from project_cash_flows().
    safety_margin : float
        Minimum cash buffer to maintain.

    Returns
    -------
    float
        Impact score between 0.0 and 1.0.

    """
    if not projections:
        return 1.0

    # Calculate the total cost burden this asset would add over the trial period
    trial = asset.trial
    if trial is None:
        return 1.0

    # Calculate step-by-step costs for this asset
    asset_costs_per_step = []
    current_trial = trial
    while current_trial is not None:
        cost_per_step = current_trial.cost_this_step
        for _ in range(current_trial.time_remaining):
            asset_costs_per_step.append(cost_per_step)
        current_trial = current_trial.next_trial_on_success

    # Check each projected step to see if adding this asset would violate constraints
    violations = 0
    total_severity = 0.0

    for i, proj in enumerate(projections):
        if i < len(asset_costs_per_step):
            additional_cost = asset_costs_per_step[i]
        else:
            additional_cost = 0.0

        adjusted_cash = proj["projected_cash"] - additional_cost

        if adjusted_cash < 0:
            # Would cause bankruptcy
            return 0.0
        elif adjusted_cash < safety_margin:
            violations += 1
            # Severity: how far below safety margin
            severity = (safety_margin - adjusted_cash) / safety_margin
            total_severity += severity

    if violations == 0:
        return 1.0

    # Scale score based on number and severity of violations
    avg_severity = total_severity / len(projections)
    return max(0.0, 1.0 - avg_severity)


class MultiStepKnapsackAgent:
    """
    Multi-step knapsack agent that considers future cash constraints.

    This agent improves on the single-step KnapsackAgent by:
    1. Projecting cash flows N steps ahead
    2. Using a conservative budget based on minimum projected cash
    3. Computing N-step value that accounts for time value of money
    4. Scaling asset values by their impact on future cash constraints

    Parameters
    ----------
    lookahead : int
        Number of steps to look ahead for cash projection and value computation.
    safety_margin_multiplier : float
        Cash buffer as multiple of ongoing costs.
    units : float
        The units (e.g. millions) of costs for the knapsack solver.
    include_ongoing_costs : bool
        Whether to subtract ongoing costs from budget.
    cost_attribute : str
        Which cost attribute to use for knapsack weights.

    """

    def __init__(
        self,
        lookahead: int = 3,
        safety_margin_multiplier: float = 0.0,
        units: float = 1e6,
        include_ongoing_costs: bool = True,
        cost_attribute: str = "remaining_trial_cost",
    ):
        """Initialize the MultiStepKnapsackAgent."""
        super().__init__()
        self.lookahead = lookahead
        self.safety_margin_multiplier = safety_margin_multiplier
        self.units = units
        self.include_ongoing_costs = include_ongoing_costs
        self.cost_attribute = cost_attribute
        self.env = None

    def set_env(self, env):
        """Set the environment for the agent."""
        self.env = env

    def knapsack_01_solver(self, items, capacity):
        """
        Solve the 0-1 knapsack problem using dynamic programming.

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

        # Create DP table
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]

        # Fill the DP table
        for i in range(1, n + 1):
            value, weight, _ = items[i - 1]
            for w in range(1, capacity + 1):
                dp[i][w] = dp[i - 1][w]
                if weight <= w:
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weight] + value)

        # Backtrack to find selected items
        selected_items = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                selected_items.append(items[i - 1])
                w -= items[i - 1][1]

        return selected_items

    def make_investment_decisions(
        self, game_state: GameState
    ) -> dict[uuid.UUID, Optional[Literal["invest"]]]:
        """
        Make investment decisions considering future cash constraints.

        Parameters
        ----------
        game_state : GameState
            The current state of the game.

        Returns
        -------
        dict[uuid.UUID, Optional[Literal["invest"]]]
            A dictionary containing asset IDs and corresponding actions.

        """
        # 1. Calculate ongoing costs from in-development assets
        ongoing_costs = 0.0
        for asset in game_state.assets.values():
            if asset.state == AssetState.InDevelopment and self.include_ongoing_costs:
                ongoing_costs += asset.cost_this_step

        # 2. Project cash flows for N steps
        projections = project_cash_flows(game_state, self.lookahead)

        # 3. Compute conservative budget
        safety_margin = ongoing_costs * self.safety_margin_multiplier

        # Budget is constrained by both current cash and minimum projected future cash
        current_available = game_state.cash - ongoing_costs - safety_margin

        if projections:
            min_future_cash = min(p["projected_cash"] for p in projections)
            future_available = min_future_cash - safety_margin
        else:
            future_available = current_available

        budget = min(current_available, future_available)

        # If budget is non-positive, don't invest
        if budget <= 0:
            return {}

        # 4. Build knapsack items with N-step values
        idle_assets = []
        for asset in game_state.assets.values():
            if asset.state == AssetState.Idle:
                # Compute N-step value
                value = n_step_value(asset, self.lookahead)

                # Skip assets with zero or negative value
                if value <= 0:
                    continue

                # Scale value by future impact score
                impact_score = compute_future_impact_score(
                    asset, projections, safety_margin
                )
                adjusted_value = value * impact_score

                # Skip assets that would cause bankruptcy
                if adjusted_value <= 0:
                    continue

                weight = math.ceil(getattr(asset, self.cost_attribute) / self.units)
                idle_assets.append((adjusted_value, weight, asset.id))

        # 5. Solve knapsack
        budget_rescaled = math.floor(budget / self.units)

        if budget_rescaled <= 0 or not idle_assets:
            return {}

        investment_decisions = {}
        knapsack_result = self.knapsack_01_solver(idle_assets, budget_rescaled)
        for _, _, asset_id in knapsack_result:
            investment_decisions[asset_id] = "invest"

        return investment_decisions

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Wrapper for use with the evaluate function.

        Takes an observation and returns actions based on the current game state.
        """
        investment_decisions = self.make_investment_decisions(
            self.env.unwrapped.game_state
        )

        # Reconstruct the action array
        asset_id_order = self.env.unwrapped._asset_id_order

        actions = np.zeros(len(asset_id_order))
        for i, asset_id in enumerate(asset_id_order):
            if asset_id in investment_decisions:
                if investment_decisions[asset_id] == "invest":
                    actions[i] = 1

        return actions
