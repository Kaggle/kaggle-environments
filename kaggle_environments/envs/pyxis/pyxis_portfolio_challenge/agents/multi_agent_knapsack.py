"""Knapsack-based heuristic agent for multi-agent competitive environment."""

import math

import numpy as np

from pyxis_portfolio_challenge.agents.knapsack import delta_npv
from pyxis_portfolio_challenge.game.asset import AssetState


class MultiAgentKnapsackAgent:
    """
    Knapsack-based heuristic agent for the multi-agent environment.

    Each step it solves a 0-1 knapsack over its own idle, investable assets to
    pick the highest-value set of new investments that fits the current cash
    budget, then keeps only its highest-value picks up to a fixed concurrency
    cap. It uses only information a real competitor has: its own portfolio and
    the action masks. It does not inspect opponents' portfolios, the shared
    market, or any other env internals, and it does not bid in the BD auction.
    """

    # Maximum number of concurrent in-development trials. Fixed at 4 so the
    # agent never exceeds the clinical-sites operational-site limit: it advances
    # only its highest-value assets within that cap rather than over-submitting
    # and relying on the environment's site arbitration to pick survivors.
    MAX_CONCURRENT_INVESTMENTS = 4

    def __init__(
        self,
        agent_name: str,
        *,
        env=None,
        units: float = 1e6,
    ):
        """
        Initialise multi-agent knapsack agent.

        Parameters
        ----------
        agent_name : str
            The agent identifier in the multi-agent environment
            (e.g. ``"pharma_0"``).
        env : MultiAgentInvestmentGameEnv | None
            Environment reference. Can be ``None`` if ``set_env`` is
            called before the first ``__call__``.
        units : float
            Rescaling factor for the knapsack solver (default 1e6).

        """
        self.agent_name = agent_name
        self.env = env
        self.units = units

    def set_env(self, env):
        """Set or update the environment reference."""
        self.env = env

    def knapsack_01_solver(self, items, capacity):
        """
        Solve 0-1 knapsack problem using dynamic programming.

        Parameters
        ----------
        items : list[tuple[float, int, Any]]
            Each tuple contains ``(value, weight, identifier)``.
        capacity : int
            Maximum weight capacity of the knapsack.

        Returns
        -------
        list[tuple[float, int, Any]]
            Selected items.

        """
        n = len(items)
        if capacity <= 0 or n == 0:
            return []

        dp = [[0] * (capacity + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            value, weight, _ = items[i - 1]
            for w in range(1, capacity + 1):
                dp[i][w] = dp[i - 1][w]
                if weight <= w:
                    dp[i][w] = max(
                        dp[i - 1][w],
                        dp[i - 1][w - weight] + value,
                    )

        selected_items = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                selected_items.append(items[i - 1])
                w -= items[i - 1][1]

        return selected_items

    def __call__(self, observation) -> dict:
        """Return action based on knapsack optimisation of new investments."""
        portfolio = self.env.agent_portfolios[self.agent_name]
        masks = self.env.action_masks(self.agent_name)

        budget = portfolio.cash

        investments = np.zeros(self.env.max_num_assets, dtype=np.int64)

        if budget <= 0:
            return {**self.env.noop_action(), "investments": investments}

        # Concurrency cap: do not start more trials than keep total in-development
        # assets at or below the fixed limit.
        current_in_dev = sum(
            1 for a in portfolio.assets.values()
            if a.state == AssetState.InDevelopment
        )
        remaining_capacity = max(0, self.MAX_CONCURRENT_INVESTMENTS - current_in_dev)
        if remaining_capacity <= 0:
            return {**self.env.noop_action(), "investments": investments}

        # Build knapsack items from this agent's own idle, investable assets.
        items = []
        asset_order = self.env._asset_id_orders[self.agent_name]

        for i, asset_id in enumerate(asset_order):
            if asset_id is None or asset_id not in portfolio.assets:
                continue
            inv_mask = masks["investments"][i]
            # Support both binary mask (0/1) and MultiDiscrete mask (list of bools)
            if isinstance(inv_mask, list):
                can_invest = len(inv_mask) > 1 and inv_mask[1]
            else:
                can_invest = inv_mask == 1
            if not can_invest:
                continue

            asset = portfolio.assets[asset_id]
            value = delta_npv(asset)
            if value <= 0:
                continue

            weight = math.ceil(asset.remaining_trial_cost / self.units)
            if weight > 0:
                items.append((value, weight, i))

        # Solve knapsack for optimal mix (budget constraint).
        budget_rescaled = math.floor(budget / self.units)

        if items and budget_rescaled > 0:
            selected = self.knapsack_01_solver(items, budget_rescaled)
            # Sort by value descending so the capacity limit keeps the best items.
            selected.sort(key=lambda x: x[0], reverse=True)
            capacity_left = int(remaining_capacity)
            for _, _, idx in selected:
                if capacity_left <= 0:
                    continue
                investments[idx] = 1
                capacity_left -= 1

        return {**self.env.noop_action(), "investments": investments}
