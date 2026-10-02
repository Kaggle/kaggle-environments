"""Knapsack-based heuristic agent for multi-agent competitive environment."""

import math

import numpy as np

from pyxis_portfolio_challenge.agents.knapsack import delta_npv
from pyxis_portfolio_challenge.game.asset import AssetState, DrugAsset
from pyxis_portfolio_challenge.game.shared_market_state import indication_key

# Sentinel to distinguish BD items from regular investment items
_BD_ITEM = "bd"
_INV_ITEM = "inv"

# Fraction of an asset's cash-adjusted eNPV the heuristic bids in the
# continuous BD auction. 3/7 reproduces the old discrete level-3 (of
# break-even level 7) economics.
_BD_BID_FRACTION = 3.0 / 7.0


class MultiAgentKnapsackAgent:
    """
    Knapsack-based heuristic agent for multi-agent environment.

    Uses 0-1 knapsack optimization to select the best combination of
    pipeline investments and BD acquisitions within the available budget,
    subject to a concurrent investment capacity limit.

    This is the benchmark agent described in the strategic depth analysis.
    With ``capacity=None`` and ``enable_bd_bidding=True`` it represents the
    strongest viable heuristic baseline. When the clinical-sites feature is
    active it caps its own concurrent trials at the operational-site count
    and advances only its highest-value assets, rather than over-submitting
    and relying on the environment's index arbitration to pick survivors.
    """

    CAPACITY_PER_INVESTMENT = 1

    def __init__(
        self,
        agent_name: str,
        *,
        env=None,
        units: float = 1e6,
        capacity: int | None = None,
        enable_bd_bidding: bool = False,
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
        capacity : int | None
            Maximum number of concurrent investments. ``None`` means
            no explicit cap (falls back to env config or unlimited).
        enable_bd_bidding : bool
            Whether to include BD auction assets in the knapsack
            optimisation.

        """
        self.agent_name = agent_name
        self.env = env
        self.units = units
        self.capacity = capacity
        self.enable_bd_bidding = enable_bd_bidding

    def set_env(self, env):
        """Set or update the environment reference."""
        self.env = env

    def _expected_market_share(self, asset: DrugAsset) -> float:
        """
        Estimate per-drug market share a new drug would get in its indication.

        Uses a quality-weighted estimate consistent with the per-drug share
        formula in market_mechanics. Returns a fraction in (0, 1].
        If the indication is under rival exclusivity, returns 0.
        """
        shared = self.env.multi_agent_game.shared_market
        current_time = self.env.multi_agent_game.time
        portfolios = self.env.agent_portfolios

        if shared.indications_per_ta > 0:
            key = indication_key(asset.therapeutic_area, asset.indication)
            ind_market = shared.indication_markets.get(key)
            if ind_market is not None:
                if ind_market.is_in_exclusivity(current_time):
                    if ind_market.first_mover_agent != self.agent_name:
                        return 0.0
                    return 1.0
                new_quality = asset.max_revenue
                total_quality = new_quality
                for portfolio in portfolios.values():
                    for a in portfolio.assets.values():
                        if (
                            a.therapeutic_area == asset.therapeutic_area
                            and a.indication == asset.indication
                            and a.state == AssetState.OnMarket
                        ):
                            tenure_bonus = 1.0 + a.time_on_market * 0.05
                            total_quality += a.max_revenue * tenure_bonus
                if total_quality <= new_quality:
                    return 1.0
                return new_quality / total_quality
        else:
            ta_market = shared.ta_markets.get(asset.therapeutic_area)
            if ta_market is not None:
                if ta_market.is_in_exclusivity(current_time):
                    if ta_market.first_mover_agent != self.agent_name:
                        return 0.0
                    return 1.0
                new_quality = asset.max_revenue
                total_quality = new_quality
                for portfolio in portfolios.values():
                    for a in portfolio.assets.values():
                        if (
                            a.therapeutic_area == asset.therapeutic_area
                            and a.state == AssetState.OnMarket
                        ):
                            tenure_bonus = 1.0 + a.time_on_market * 0.05
                            total_quality += a.max_revenue * tenure_bonus
                if total_quality <= new_quality:
                    return 1.0
                return new_quality / total_quality
        return 1.0

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

    def _get_capacity(self, portfolio) -> int:
        """Max simultaneous in-development assets. Override for dynamic strategies."""
        if self.capacity is not None:
            return self.capacity
        if (
            getattr(self.env, "rd_capacity_config", None) is not None
            and self.env.rd_capacity_config.enabled
        ):
            return portfolio.capacity_base
        # Respect the clinical-sites concurrency limit: an agent can host at most
        # one in-development asset per operational site, so cap concurrent trials
        # at the operational-site count. Combined with the value-sorted truncation
        # below, this makes the agent advance only its highest-value assets rather
        # than over-submitting and letting index arbitration pick the survivors.
        if getattr(portfolio, "clinical_sites_enabled", False):
            return portfolio.operational_sites
        return 999

    def _effective_budget(self, portfolio) -> float:
        """Budget available for new investments. Override to inject projected cash."""
        return portfolio.cash

    def __call__(self, observation) -> dict:
        """Return action based on knapsack optimisation with capacity awareness."""
        portfolio = self.env.agent_portfolios[self.agent_name]
        masks = self.env.action_masks(self.agent_name)

        budget = self._effective_budget(portfolio)

        investments = np.zeros(self.env.max_num_assets, dtype=np.int64)
        # Continuous BD bids: cash amount per slot in GBP millions (0 = pass).
        bd_bids = np.zeros(self.env.bd_max_slots, dtype=np.float32)

        if budget <= 0:
            return {
                **self.env.noop_action(),
                "investments": investments,
                "bd_bids": bd_bids,
            }

        # Capacity constraint
        cap = self._get_capacity(portfolio)
        current_in_dev = sum(
            1 for a in portfolio.assets.values()
            if a.state == AssetState.InDevelopment
        )
        remaining_capacity = max(0, cap - current_in_dev)
        if remaining_capacity <= 0:
            return {
                **self.env.noop_action(),
                "investments": investments,
                "bd_bids": bd_bids,
            }

        # Build knapsack items from regular idle assets
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
                items.append((value, weight, (_INV_ITEM, i)))

        # Add BD assets as additional knapsack items (one per slot)
        if self.enable_bd_bidding and self.env.bd_enabled:
            shared = self.env.multi_agent_game.shared_market
            bd_assets = shared.current_bd_assets
            n_current = len(portfolio.assets)
            slots_free = portfolio.max_num_assets - n_current

            for slot_idx, bd_asset in enumerate(bd_assets):
                if slots_free <= 0:
                    break
                enpv = bd_asset.enpv
                if enpv <= 0:
                    continue
                value = delta_npv(bd_asset)
                if value <= 0:
                    continue
                ms = self._expected_market_share(bd_asset)
                if ms <= 0:
                    continue
                value *= ms

                # Continuous-bid heuristic: bid a fixed fraction of the asset's
                # cash-adjusted eNPV. The 3/7 fraction reproduces the economics
                # of the old discrete level-3 (of break-even level 7) bid.
                reinv_pct = portfolio.reinvestment_percentage
                cash_enpv = bd_asset.cash_enpv(reinv_pct)
                bid_price = max(0.0, _BD_BID_FRACTION * cash_enpv)
                # Clamp to the action-space cap (expressed in GBP millions).
                bid_millions = min(bid_price / 1e6, float(self.env.bd_max_bid))
                bid_price = bid_millions * 1e6
                total_cost = bid_price + bd_asset.remaining_trial_cost
                weight = math.ceil(total_cost / self.units)
                if weight > 0:
                    items.append(
                        (value, weight, (_BD_ITEM, (slot_idx, bid_millions)))
                    )

        # Solve knapsack for optimal mix (budget constraint)
        budget_rescaled = math.floor(budget / self.units)

        if items and budget_rescaled > 0:
            selected = self.knapsack_01_solver(items, budget_rescaled)
            # Sort by value descending so capacity limit keeps the best items
            selected.sort(key=lambda x: x[0], reverse=True)
            capacity_left = int(remaining_capacity)
            for _, _, (item_type, idx) in selected:
                if item_type == _INV_ITEM:
                    if capacity_left <= 0:
                        continue
                    investments[idx] = 1
                    capacity_left -= self.CAPACITY_PER_INVESTMENT
                elif item_type == _BD_ITEM:
                    if capacity_left <= 0:
                        continue
                    slot_idx, bid_millions = idx
                    bd_bids[slot_idx] = bid_millions
                    capacity_left -= self.CAPACITY_PER_INVESTMENT

        return {
            **self.env.noop_action(),
            "investments": investments,
            "bd_bids": bd_bids,
        }
