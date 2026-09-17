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


def compute_stop_score(asset: DrugAsset) -> float:
    """
    Compute a score for how much we want to stop this asset.

    Higher score = worse asset = more likely to stop.
    Score based on how much the interim signal deviates below PTRS.

    Parameters
    ----------
    asset : DrugAsset
        The drug asset to evaluate (should be InDevelopment).

    Returns
    -------
    float
        Score where higher = worse asset. Negative = performing above expectations.

    """
    if asset.state != AssetState.InDevelopment:
        return float("-inf")  # Can't stop non-InDevelopment assets

    # Negative eNPV = definitely stop
    if asset.enpv <= 0:
        return float("inf")

    interim_signal = asset.interim_signal
    original_ptrs = asset.trial.ptrs

    if original_ptrs <= 0:
        return 0.0

    # Score: how much worse is the signal than expected
    # score = (ptrs - signal) / ptrs
    # Positive score = underperforming, negative = outperforming
    return (original_ptrs - interim_signal) / original_ptrs


def compute_continuation_ev(asset: DrugAsset) -> float:
    """
    Compute expected value of continuing an InDevelopment asset.

    Uses interim signal to update the success probability estimate,
    then computes adjusted eNPV.

    EV_continue = (interim_signal / ptrs) * eNPV

    Parameters
    ----------
    asset : DrugAsset
        The drug asset to evaluate (should be InDevelopment).

    Returns
    -------
    float
        Expected value of continuing. Negative or low = should stop.

    """
    if asset.state != AssetState.InDevelopment:
        return 0.0

    if asset.enpv <= 0:
        return asset.enpv  # Already negative

    interim_signal = asset.interim_signal
    original_ptrs = asset.trial.ptrs

    if original_ptrs <= 0:
        return 0.0

    # Adjust eNPV based on updated probability estimate from interim signal
    # This gives us the expected value given what we've learned so far
    adjusted_enpv = asset.enpv * (interim_signal / original_ptrs)

    return adjusted_enpv


class KnapsackAgent:
    """
    Knapsack based agent that uses a knapsack solver to make decisions.

    This agent uses the 0/1 knapsack solver to optimise selection of assets at the next
    time step. Caution: it only ensures the budget constraint at the next time step,
    not the entire horizon.

    When enable_stop=True, the agent re-ranks ALL assets (Idle and InDevelopment) each
    step. For InDevelopment assets, the default action is STOP - selecting them in the
    knapsack means continuing development. This allows early termination of unpromising
    trials based on interim signals.

    Parameters
    ----------
    units : float
        The units (e.g. millions, billions) of the costs and NPVs as used by the solver.
    include_ongoing_costs : bool
        Whether to subtract costs of assets in an ongoing trial from the budget.
    enable_stop : bool
        Whether to enable STOP action for InDevelopment assets. When True, all
        assets are re-ranked each step and InDevelopment assets default to STOP
        unless selected.
    continuation_value_function : Callable[[DrugAsset], float]
        Value function for InDevelopment assets. Default uses interim
        signal-adjusted eNPV.

    """

    def __init__(
        self,
        units: float = 1e6,
        value_function: Callable[[DrugAsset], float] = delta_npv,
        include_ongoing_costs: bool = True,
        cost_attribute: str = "remaining_trial_cost",
        enable_stop: bool = False,
        target_capacity: int = 4,
    ):
        """Initialize the KnapsackAgent."""
        super().__init__()
        self.units = units
        self.value_function = value_function
        self.include_ongoing_costs = include_ongoing_costs
        self.env = None
        self.cost_attribute = cost_attribute
        self.enable_stop = enable_stop
        self.target_capacity = target_capacity

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
    ) -> dict[uuid.UUID, Literal["invest", "continue", "stop"]]:
        """
        Make investment decisions based on the current game state.

        Here the budget and costs are divided by the units parameter and rounded up or
        down to ensure that they are integers. The values are left as floats since these
        are handled by the knapsack solver.

        When enable_stop=True, uses capacity-aware stopping: if over target capacity,
        stop the worst-performing assets (by interim signal) until at or below capacity.

        Parameters
        ----------
        game_state : GameState
            The current state of the game.

        Returns
        -------
        dict[uuid.UUID, Literal["invest", "continue", "stop"]]
           A dictionary containing the asset IDs and the corresponding actions.
           "invest" for Idle assets to start development.
           "continue" for InDevelopment assets to keep developing.
           "stop" for InDevelopment assets to terminate early.

        """
        budget = game_state.cash
        idle_assets = []
        investment_decisions = {}
        in_dev_assets = []

        for asset in game_state.assets.values():
            if asset.state == AssetState.InDevelopment:
                if self.include_ongoing_costs:
                    budget -= getattr(asset, self.cost_attribute)
                in_dev_assets.append(asset)

            elif asset.state == AssetState.Idle:
                value = self.value_function(asset)
                weight = math.ceil(getattr(asset, self.cost_attribute) / self.units)
                idle_assets.append((value, weight, asset.id))

        # Capacity-aware stopping: only stop if over capacity
        if self.enable_stop and in_dev_assets:
            # Get capacity cost per asset (assume STANDARD = 2)
            capacity_per_asset = 2
            current_capacity = len(in_dev_assets) * capacity_per_asset

            if current_capacity > self.target_capacity:
                # Need to stop some assets - rank by stop score
                scored_assets = [
                    (compute_stop_score(asset), asset) for asset in in_dev_assets
                ]
                scored_assets.sort(key=lambda x: x[0], reverse=True)  # Worst first

                # Stop worst assets until at or below capacity
                capacity_to_free = current_capacity - self.target_capacity
                freed = 0
                for score, asset in scored_assets:
                    if freed >= capacity_to_free:
                        investment_decisions[asset.id] = "continue"
                    elif score > 0:  # Only stop if underperforming
                        investment_decisions[asset.id] = "stop"
                        freed += capacity_per_asset
                    else:
                        investment_decisions[asset.id] = "continue"
            else:
                # Under capacity - continue all
                for asset in in_dev_assets:
                    investment_decisions[asset.id] = "continue"

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

        When enable_stop=True:
        - InDevelopment assets with poor interim signals get STOP (action=4)
        - Other InDevelopment assets get STANDARD (action=2) to continue
        - Idle assets work as before (selected get invest action)
        """
        game_state = self.env.unwrapped.game_state
        investment_decisions = self.make_investment_decisions(game_state)

        # reconstruct the action array
        asset_id_order = self.env.unwrapped._asset_id_order

        # Check if investment levels are enabled (MultiDiscrete action space)
        # If so, use action value 2 (STANDARD) instead of 1 (MINIMAL)
        investment_levels_enabled = (
            hasattr(self.env.unwrapped, "investment_levels_config")
            and self.env.unwrapped.investment_levels_config is not None
            and self.env.unwrapped.investment_levels_config.enabled
        )
        invest_action = 2 if investment_levels_enabled else 1
        stop_action = 4  # STOP action for InDevelopment assets

        actions = np.zeros(len(asset_id_order))

        for i, asset_id in enumerate(asset_id_order):
            if asset_id in investment_decisions:
                decision = investment_decisions[asset_id]
                if decision == "invest":
                    actions[i] = invest_action
                elif decision == "continue":
                    # Continue development at STANDARD level
                    actions[i] = invest_action
                elif decision == "stop":
                    actions[i] = stop_action

        return actions


class KnapsackWithStopAgent(KnapsackAgent):
    """
    Knapsack agent with STOP action enabled.

    Uses capacity-aware stopping: when over target capacity, stops the
    worst-performing InDevelopment assets (by interim signal) until at
    or below capacity. Then runs knapsack on Idle assets.
    """

    def __init__(
        self,
        units: float = 1e6,
        value_function: Callable[[DrugAsset], float] = delta_npv,
        cost_attribute: str = "remaining_trial_cost",
        target_capacity: int = 4,
    ):
        """Initialize KnapsackWithStopAgent with enable_stop=True."""
        super().__init__(
            units=units,
            value_function=value_function,
            include_ongoing_costs=True,
            cost_attribute=cost_attribute,
            enable_stop=True,
            target_capacity=target_capacity,
        )


class MultipleChoiceKnapsackAgent:
    """
    Multiple-Choice Knapsack agent that optimizes (asset, level) pairs.

    For each asset, considers all valid investment levels and computes:
    - Value = eNPV * success_mod * speed_mod (expected return with level effects)
    - Cost = remaining_trial_cost * cost_mod (budget impact)
    - Capacity = capacity_cost (capacity impact)

    Solves the dual-constraint problem (budget + capacity) using efficiency-based
    greedy selection: efficiency = value / (cost + capacity_penalty).

    For InDevelopment assets, can STOP based on interim signal quality.
    """

    # Investment level definitions matching config
    LEVELS = {
        0: {
            "name": "NONE",
            "cost_mod": 0.0,
            "speed_mod": 0.0,
            "success_mod": 1.0,
            "capacity": 0,
        },
        1: {
            "name": "MINIMAL",
            "cost_mod": 0.5,
            "speed_mod": 0.5,
            "success_mod": 0.85,
            "capacity": 1,
        },
        2: {
            "name": "STANDARD",
            "cost_mod": 1.0,
            "speed_mod": 1.0,
            "success_mod": 1.0,
            "capacity": 2,
        },
        3: {
            "name": "ACCELERATED",
            "cost_mod": 2.0,
            "speed_mod": 2.0,
            "success_mod": 1.10,
            "capacity": 4,
        },
        4: {
            "name": "STOP",
            "cost_mod": 0.0,
            "speed_mod": 0.0,
            "success_mod": 0.0,
            "capacity": 0,
        },
    }

    def __init__(
        self,
        target_capacity: int = 4,
        capacity_penalty_weight: float = 50.0,
        stop_threshold: float = 0.5,
        min_ev_to_continue: float = 0.5e9,  # 0.5B optimal from benchmarking
    ):
        """
        Initialize MultipleChoiceKnapsackAgent.

        Parameters
        ----------
        target_capacity : int
            Maximum capacity to use (default: 4, matching base_capacity).
        capacity_penalty_weight : float
            Weight for capacity in efficiency calculation (in millions).
            Higher = more conservative with capacity.
        stop_threshold : float
            Stop InDevelopment assets if (ptrs - interim) / ptrs > threshold.
        min_ev_to_continue : float
            Minimum adjusted eNPV (= eNPV * interim/ptrs) to continue.
            Set > 0 to stop low-value assets even if signal is good.

        """
        self.target_capacity = target_capacity
        self.capacity_penalty_weight = capacity_penalty_weight
        self.stop_threshold = stop_threshold
        self.min_ev_to_continue = min_ev_to_continue
        self.env = None

    def set_env(self, env):
        """Set the environment for the agent."""
        self.env = env

    def _compute_option_value(
        self, asset: DrugAsset, level: int, is_in_development: bool = False
    ) -> tuple[float, float, int, float]:
        """
        Compute value metrics for an (asset, level) option.

        Returns (value, cost, capacity, efficiency).

        Value formula uses delta_npv (cost of delay) as base:
        - delta_npv captures: value of investing NOW vs waiting
        - Adjusted by success_mod (probability effect)
        - NOT multiplied by speed_mod (avoid overvaluing ACCELERATED)

        Efficiency = value / capacity (simple capacity efficiency)
        """
        level_info = self.LEVELS[level]

        if level == 0:  # NONE - don't invest
            return (0.0, 0.0, 0, 0.0)

        if level == 4:  # STOP
            return (0.0, 0.0, 0, 0.0)

        # Base value from delta_npv (cost of delay)
        base_value = delta_npv(asset)
        if base_value <= 0:
            return (0.0, 0.0, 0, float("-inf"))

        # Adjust for interim signal if InDevelopment
        if is_in_development:
            signal_ratio = asset.interim_signal / asset.trial.ptrs
            base_value = base_value * signal_ratio

        # Value = delta_npv * success_mod
        # Don't multiply by speed_mod - it overvalues ACCELERATED
        value = base_value * level_info["success_mod"]

        # Cost depends on state
        if is_in_development:
            cost = asset.cost_this_step * level_info["cost_mod"]
        else:
            cost = asset.remaining_trial_cost * level_info["cost_mod"]

        capacity = level_info["capacity"]

        # Efficiency = value per capacity unit (prioritize capacity-efficient options)
        if capacity > 0:
            efficiency = value / capacity
        else:
            efficiency = 0.0

        return (value, cost, capacity, efficiency)

    def _generate_options(
        self, game_state: GameState
    ) -> list[tuple[float, float, int, uuid.UUID, int, bool]]:
        """
        Generate all (asset, level) options with their metrics.

        Returns list of (efficiency, cost, capacity, asset_id, level, is_in_dev).
        """
        options = []

        for asset in game_state.assets.values():
            if asset.state == AssetState.Idle:
                # Only consider STANDARD and ACCELERATED for new investments
                # MINIMAL is too slow and has lower success rate
                for level in [2, 3]:
                    value, cost, cap, eff = self._compute_option_value(
                        asset, level, is_in_development=False
                    )
                    if value > 0:
                        options.append((eff, cost, cap, asset.id, level, False))

            elif asset.state == AssetState.InDevelopment:
                # For InDevelopment: can only STOP or CONTINUE at STANDARD
                # (cannot change level mid-trial)
                stop_score = compute_stop_score(asset)
                continuation_ev = compute_continuation_ev(asset)

                # STOP if:
                # 1. Signal is poor (stop_score > threshold), OR
                # 2. Adjusted eNPV is below minimum (even if signal looks OK)
                should_stop = (
                    stop_score > self.stop_threshold
                    or continuation_ev <= self.min_ev_to_continue
                )

                if should_stop:
                    # STOP - frees resources
                    options.append((float("inf"), 0, 0, asset.id, 4, True))
                else:
                    # Continue at STANDARD (level 2)
                    cost = asset.cost_this_step
                    capacity = 2  # STANDARD capacity
                    # Use continuation EV as value
                    eff = continuation_ev / capacity if capacity > 0 else 0
                    options.append((eff, cost, capacity, asset.id, 2, True))

        return options

    def _solve_mckp(
        self, options: list, budget: float, capacity_limit: int
    ) -> dict[uuid.UUID, int]:
        """
        Solve Multiple-Choice Knapsack with dual constraints.

        Uses efficiency-based greedy selection:
        1. Sort options by efficiency (value per resource unit)
        2. For each asset, select best feasible option
        3. Respect both budget and capacity constraints

        Parameters
        ----------
        options : list
            List of (efficiency, cost, capacity, asset_id, level, is_in_dev).
        budget : float
            Available cash.
        capacity_limit : int
            Maximum capacity to use.

        Returns
        -------
        dict[uuid.UUID, int]
            Mapping of asset_id -> selected level.

        """
        # Sort by efficiency (highest first)
        sorted_options = sorted(options, key=lambda x: x[0], reverse=True)

        selected = {}  # asset_id -> level
        used_budget = 0.0
        used_capacity = 0

        # First pass: handle STOP actions (they free resources)
        for eff, cost, cap, asset_id, level, is_in_dev in sorted_options:
            if level == 4:  # STOP
                selected[asset_id] = 4

        # Second pass: greedily select best options for remaining assets
        for eff, cost, cap, asset_id, level, is_in_dev in sorted_options:
            if asset_id in selected:
                continue  # Already decided

            # Check constraints
            if used_budget + cost > budget:
                continue  # Can't afford
            if used_capacity + cap > capacity_limit:
                continue  # No capacity

            # Select this option
            selected[asset_id] = level
            used_budget += cost
            used_capacity += cap

        return selected

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Select actions using Multiple-Choice Knapsack optimization."""
        game_state = self.env.unwrapped.game_state
        asset_id_order = self.env.unwrapped._asset_id_order

        # Generate all (asset, level) options
        options = self._generate_options(game_state)

        # Solve MCKP with dual constraints
        budget = game_state.cash
        decisions = self._solve_mckp(options, budget, self.target_capacity)

        # Build action array
        actions = np.zeros(len(asset_id_order), dtype=np.int32)
        for i, asset_id in enumerate(asset_id_order):
            if asset_id in decisions:
                actions[i] = decisions[asset_id]
            # else: default is 0 (NONE)

        return actions


def compute_distributional_stop_score(asset: DrugAsset) -> float:
    """
    Compute stop score using distributional PTRS information.

    Uses pessimistic PTRS (ptrs_range_low) and factors in confidence.
    Higher score = worse asset = more likely to stop.

    Key insight: With distributional PTRS, we should be MORE willing to stop
    when confidence is low (wide uncertainty range).
    """
    if asset.state != AssetState.InDevelopment:
        return float("-inf")

    if asset.enpv <= 0:
        return float("inf")

    interim_signal = asset.interim_signal
    trial = asset.trial

    # Use distributional PTRS if available
    if hasattr(trial, "_ptrs_prior_alpha") and trial._ptrs_prior_alpha is not None:
        ptrs_expected = trial.ptrs_expected
        ptrs_confidence = trial.ptrs_confidence
        ptrs_range_low = trial.ptrs_range_low

        if ptrs_expected <= 0:
            return 0.0

        # Base stop score: how much signal is below expected
        base_score = (ptrs_expected - interim_signal) / ptrs_expected

        # Confidence penalty: be more willing to stop when uncertain
        # Low confidence = add penalty, making asset more likely to stop
        confidence_penalty = (1.0 - ptrs_confidence) * 0.3

        # Also penalize if signal is below the pessimistic range
        if interim_signal < ptrs_range_low:
            confidence_penalty += 0.2

        return base_score + confidence_penalty
    else:
        # Fallback to original logic
        return compute_stop_score(asset)


def compute_distributional_continuation_ev(asset: DrugAsset) -> float:
    """
    Compute continuation EV using distributional PTRS (risk-adjusted).

    Key adaptations for distributional PTRS:
    1. Use pessimistic PTRS (ptrs_range_low) instead of point estimate
    2. Apply confidence discount: lower confidence = lower EV
    3. Compare interim signal against the full distribution

    This makes the agent more conservative when facing uncertainty.
    """
    if asset.state != AssetState.InDevelopment:
        return 0.0

    if asset.enpv <= 0:
        return asset.enpv

    interim_signal = asset.interim_signal
    trial = asset.trial

    # Use distributional PTRS if available
    if hasattr(trial, "_ptrs_prior_alpha") and trial._ptrs_prior_alpha is not None:
        ptrs_expected = trial.ptrs_expected
        ptrs_confidence = trial.ptrs_confidence
        ptrs_range_low = trial.ptrs_range_low

        if ptrs_expected <= 0:
            return 0.0

        # Use pessimistic PTRS for conservative valuation
        # Blend between expected and pessimistic based on confidence
        # High confidence: use expected; Low confidence: use pessimistic
        effective_ptrs = (
            ptrs_confidence * ptrs_expected + (1 - ptrs_confidence) * ptrs_range_low
        )

        # Compute signal ratio against the effective (pessimistic) PTRS
        if effective_ptrs > 0:
            signal_ratio = interim_signal / effective_ptrs
        else:
            signal_ratio = 0.0

        # Apply confidence discount to the EV
        # Low confidence = discount the expected value
        confidence_factor = 0.7 + 0.3 * ptrs_confidence  # Range: 0.7 to 1.0

        adjusted_enpv = asset.enpv * signal_ratio * confidence_factor

        return adjusted_enpv
    else:
        # Fallback to original logic
        return compute_continuation_ev(asset)


class DistributionalMCKAgent:
    """
    Multiple-Choice Knapsack agent adapted for distributional PTRS.

    Key adaptations from standard MCK:
    1. Uses pessimistic PTRS (ptrs_range_low) for continuation decisions
    2. Factors in confidence: low confidence = more conservative
    3. Higher stop threshold when facing uncertainty
    4. Applies variance penalty to high-uncertainty assets

    This agent is designed to handle the compound uncertainty introduced
    by the distributional PTRS feature.
    """

    LEVELS = {
        0: {
            "name": "NONE",
            "cost_mod": 0.0,
            "speed_mod": 0.0,
            "success_mod": 1.0,
            "capacity": 0,
        },
        1: {
            "name": "MINIMAL",
            "cost_mod": 0.5,
            "speed_mod": 0.5,
            "success_mod": 0.85,
            "capacity": 1,
        },
        2: {
            "name": "STANDARD",
            "cost_mod": 1.0,
            "speed_mod": 1.0,
            "success_mod": 1.0,
            "capacity": 2,
        },
        3: {
            "name": "ACCELERATED",
            "cost_mod": 2.0,
            "speed_mod": 2.0,
            "success_mod": 1.10,
            "capacity": 4,
        },
        4: {
            "name": "STOP",
            "cost_mod": 0.0,
            "speed_mod": 0.0,
            "success_mod": 0.0,
            "capacity": 0,
        },
    }

    def __init__(
        self,
        target_capacity: int = 4,
        stop_threshold: float = 0.3,  # Lower threshold = more aggressive stopping
        min_ev_to_continue: float = 0.3e9,  # Lower threshold for distributional
        confidence_weight: float = 0.5,  # How much to weight confidence in decisions
    ):
        """
        Initialize DistributionalMCKAgent.

        Parameters
        ----------
        target_capacity : int
            Maximum capacity to use (default: 4).
        stop_threshold : float
            Stop if distributional stop score > threshold.
            Lower than standard MCK (0.3 vs 0.5) to be more conservative.
        min_ev_to_continue : float
            Minimum risk-adjusted eNPV to continue.
            Lower than standard MCK because we're already being conservative.
        confidence_weight : float
            Weight for confidence in decisions (0 to 1).
            Higher = more sensitive to confidence levels.

        """
        self.target_capacity = target_capacity
        self.stop_threshold = stop_threshold
        self.min_ev_to_continue = min_ev_to_continue
        self.confidence_weight = confidence_weight
        self.env = None

    def set_env(self, env):
        """Set the environment for the agent."""
        self.env = env

    def _compute_option_value(
        self, asset: DrugAsset, level: int, is_in_development: bool = False
    ) -> tuple[float, float, int, float]:
        """
        Compute value metrics for an (asset, level) option.

        For distributional PTRS, applies confidence-weighted valuation.
        """
        level_info = self.LEVELS[level]

        if level == 0 or level == 4:
            return (0.0, 0.0, 0, 0.0)

        base_value = delta_npv(asset)
        if base_value <= 0:
            return (0.0, 0.0, 0, float("-inf"))

        trial = asset.trial

        # Apply distributional PTRS adjustments
        if hasattr(trial, "_ptrs_prior_alpha") and trial._ptrs_prior_alpha is not None:
            confidence = trial.ptrs_confidence
            ptrs_range = trial.ptrs_range_high - trial.ptrs_range_low

            # Variance penalty: wide range = uncertain = penalize
            variance_penalty = 1.0 - (ptrs_range * self.confidence_weight)
            variance_penalty = max(0.5, variance_penalty)  # Floor at 0.5

            # Confidence boost: high confidence = trust the value more
            confidence_factor = 0.8 + 0.2 * confidence

            base_value = base_value * variance_penalty * confidence_factor

        # Adjust for interim signal if InDevelopment
        if is_in_development:
            if (
                hasattr(trial, "_ptrs_prior_alpha")
                and trial._ptrs_prior_alpha is not None
            ):
                # Use distributional continuation EV
                continuation_ev = compute_distributional_continuation_ev(asset)
                if asset.enpv > 0:
                    signal_ratio = continuation_ev / asset.enpv
                else:
                    signal_ratio = 0
                base_value = base_value * max(0, signal_ratio)
            else:
                signal_ratio = (
                    asset.interim_signal / trial.ptrs if trial.ptrs > 0 else 0
                )
                base_value = base_value * signal_ratio

        value = base_value * level_info["success_mod"]

        if is_in_development:
            cost = asset.cost_this_step * level_info["cost_mod"]
        else:
            cost = asset.remaining_trial_cost * level_info["cost_mod"]

        capacity = level_info["capacity"]
        efficiency = value / capacity if capacity > 0 else 0.0

        return (value, cost, capacity, efficiency)

    def _generate_options(
        self, game_state: GameState
    ) -> list[tuple[float, float, int, uuid.UUID, int, bool]]:
        """Generate all (asset, level) options with distributional adjustments."""
        options = []

        for asset in game_state.assets.values():
            if asset.state == AssetState.Idle:
                for level in [2, 3]:
                    value, cost, cap, eff = self._compute_option_value(
                        asset, level, is_in_development=False
                    )
                    if value > 0:
                        options.append((eff, cost, cap, asset.id, level, False))

            elif asset.state == AssetState.InDevelopment:
                # Use distributional stop/continue logic
                stop_score = compute_distributional_stop_score(asset)
                continuation_ev = compute_distributional_continuation_ev(asset)

                should_stop = (
                    stop_score > self.stop_threshold
                    or continuation_ev <= self.min_ev_to_continue
                )

                if should_stop:
                    options.append((float("inf"), 0, 0, asset.id, 4, True))
                else:
                    cost = asset.cost_this_step
                    capacity = 2
                    eff = continuation_ev / capacity if capacity > 0 else 0
                    options.append((eff, cost, capacity, asset.id, 2, True))

        return options

    def _solve_mckp(
        self, options: list, budget: float, capacity_limit: int
    ) -> dict[uuid.UUID, int]:
        """Solve MCKP with dual constraints (same as standard MCK)."""
        sorted_options = sorted(options, key=lambda x: x[0], reverse=True)

        selected = {}
        used_budget = 0.0
        used_capacity = 0

        # First pass: handle STOP actions
        for eff, cost, cap, asset_id, level, is_in_dev in sorted_options:
            if level == 4:
                selected[asset_id] = 4

        # Second pass: greedily select best options
        for eff, cost, cap, asset_id, level, is_in_dev in sorted_options:
            if asset_id in selected:
                continue
            if used_budget + cost > budget:
                continue
            if used_capacity + cap > capacity_limit:
                continue

            selected[asset_id] = level
            used_budget += cost
            used_capacity += cap

        return selected

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Select actions using distributional MCK optimization."""
        game_state = self.env.unwrapped.game_state
        asset_id_order = self.env.unwrapped._asset_id_order

        options = self._generate_options(game_state)
        budget = game_state.cash
        decisions = self._solve_mckp(options, budget, self.target_capacity)

        actions = np.zeros(len(asset_id_order), dtype=np.int32)
        for i, asset_id in enumerate(asset_id_order):
            if asset_id in decisions:
                actions[i] = decisions[asset_id]

        return actions
