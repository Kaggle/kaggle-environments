import math
import uuid
from typing import Literal, Optional, Protocol

from pyxis_portfolio_challenge.environment.metrics import legacy_static_npv
from pyxis_portfolio_challenge.game.asset import AssetState
from pyxis_portfolio_challenge.game.game_state import GameEndReason, GameState

# Type alias for investment decisions to improve readability
InvestmentDecisions = Optional[dict[uuid.UUID, Optional[Literal["invest"]]]]


class Reward(Protocol):
    """Base class for individual reward components."""

    def compute(
        self,
        pre_step_game_state: GameState,
        post_step_game_state: GameState,
        investment_decisions: InvestmentDecisions = None,
    ) -> float:
        """Compute reward based on stored information."""
        pass


class CompositeReward(Reward):
    """Combines multiple reward components by summing."""

    def __init__(self, components: list[Reward]):
        """Initialize composite reward with list of reward components."""
        self.components = components

    def compute(
        self,
        pre_step_game_state: GameState,
        post_step_game_state: GameState,
        investment_decisions: InvestmentDecisions = None,
    ) -> float:
        """Compute total reward by summing all component rewards."""
        return sum(
            component.compute(
                pre_step_game_state=pre_step_game_state,
                post_step_game_state=post_step_game_state,
                investment_decisions=investment_decisions,
            )
            for component in self.components
        )


class LegacyStaticNPVReward(Reward):
    """
    Reward uses legacy_static_npv for change in NPV each step.

    Reflects the 'cost of delay' for all assets in the portfolio.
    """

    def compute(
        self,
        pre_step_game_state: GameState,
        post_step_game_state: GameState,
        investment_decisions: InvestmentDecisions = None,
    ) -> float:
        """Compute the reward based on the change in NPV."""
        return legacy_static_npv(post_step_game_state) - legacy_static_npv(
            pre_step_game_state
        )


class ENPVReward(Reward):
    """Reward is absolute NPV of portfolio."""

    def compute(
        self,
        pre_step_game_state: GameState,
        post_step_game_state: GameState,
        investment_decisions: InvestmentDecisions = None,
    ) -> float:
        """Compute the reward based on the eNPV."""
        return post_step_game_state.enpv()


class ENPVExcludeOnMarketReward(Reward):
    """Reward is absolute NPV of portfolio minus assets on market."""

    def compute(
        self,
        pre_step_game_state: GameState,
        post_step_game_state: GameState,
        investment_decisions: InvestmentDecisions = None,
    ) -> float:
        """Compute the reward based on the NPV after minus assets on market."""
        npv = post_step_game_state.enpv()
        for asset in post_step_game_state.assets.values():
            if asset.state == AssetState.OnMarket:
                npv -= asset.enpv
        return npv


class HorizonReachedBonus(Reward):
    """Applies large positive reward for reaching horizon."""

    def __init__(self, bonus_amount: float):
        """Initialize HorizonReachedBonus."""
        self.bonus_amount = bonus_amount

    def compute(
        self,
        pre_step_game_state: GameState,
        post_step_game_state: GameState,
        investment_decisions: InvestmentDecisions = None,
    ) -> float:
        """Return bonus if horizon is reached, otherwise 0."""
        if (
            post_step_game_state.game_ended
            and post_step_game_state.ended_reason == GameEndReason.HORIZON_REACHED
        ):
            return self.bonus_amount
        return 0.0


class SubtractCash(Reward):
    """Subtract cash from reward to penalize high cash holdings."""

    def compute(
        self,
        pre_step_game_state: GameState,
        post_step_game_state: GameState,
        investment_decisions: InvestmentDecisions = None,
    ) -> float:
        """Return negative cash as penalty."""
        return -post_step_game_state.cash


class NegativeCashPenalty(Reward):
    """Applies negative reward for negative cash after evolution."""

    def __init__(self, penalty_amount: float):
        """Initialize NegativeCashPenalty."""
        self.penalty_amount = penalty_amount

    def compute(
        self,
        pre_step_game_state: GameState,
        post_step_game_state: GameState,
        investment_decisions: InvestmentDecisions = None,
    ) -> float:
        """Return negative cash as penalty."""
        return self.penalty_amount if post_step_game_state.cash < 0.0 else 0.0


class NetCashFlowReward(Reward):
    """Reward is net cash flow during the step."""

    def __init__(self, weight: float = 1.0):
        """Initialize NetCashFlowReward."""
        self.weight = weight

    def compute(
        self,
        pre_step_game_state: GameState,
        post_step_game_state: GameState,
        investment_decisions: InvestmentDecisions = None,
    ) -> float:
        """Compute the reward based on net cash flow."""
        return self.weight * (post_step_game_state.cash - pre_step_game_state.cash)


class SymLogNetCashFlowReward(Reward):
    """
    Net cash flow reward with symmetric log compression.

    Raw cash flow rewards have huge range and high variance (e.g. 20B +/- 10B).
    This applies sign(x) * log1p(|x|) to compress the scale while preserving
    sign, mapping billions-scale values to a ~20-25 range.
    """

    def __init__(self, weight: float = 1.0):
        """Initialize SymLogNetCashFlowReward."""
        self.weight = weight

    def compute(
        self,
        pre_step_game_state: GameState,
        post_step_game_state: GameState,
        investment_decisions: InvestmentDecisions = None,
    ) -> float:
        """Compute the symlog-compressed net cash flow reward."""
        raw = post_step_game_state.cash - pre_step_game_state.cash
        return self.weight * math.copysign(math.log1p(abs(raw)), raw)


class NetCashFlowRewardNegativeScale(Reward):
    """
    Reward is net cash flow with negative components scaled by reinvestment_percentage.

    This addresses the asymmetry where costs hit cash at 100% but revenue
    only returns reinvestment_percentage (e.g. 15%). By scaling the negative
    (cost) component of the reward by the same factor, the reward signal
    reflects the true profitability of investments rather than being dominated
    by unscaled costs.
    """

    def __init__(self, scale: float = 0.15):
        """Initialize NetCashFlowRewardNegativeScale."""
        self.scale = scale

    def compute(
        self,
        pre_step_game_state: GameState,
        post_step_game_state: GameState,
        investment_decisions: InvestmentDecisions = None,
    ) -> float:
        """Compute the reward based on net cash flow with scaled negatives."""
        delta = post_step_game_state.cash - pre_step_game_state.cash
        if delta < 0:
            return delta * self.scale
        return delta


class ROIReward(Reward):
    """
    Reward based on cumulative return on investment.

    Measures cumulative_revenue / cumulative_cost at each step. The agent is
    rewarded for the *change* in ROI, so it gets positive signal when a new
    investment improves the portfolio's overall return efficiency and negative
    signal when it worsens it.

    This teaches selectivity: investing in a winner improves ROI, investing in
    a loser dilutes it. Unlike NCF, the agent can't game this by simply
    investing more — volume without quality is punished.
    """

    def __init__(self, scale: float = 1e9):
        """
        Initialize ROIReward.

        Parameters
        ----------
        scale : float
            Multiplier to bring ROI deltas into a range suitable for RL.
            ROI is a ratio (e.g. 0.02), so scale brings it to ~millions.

        """
        self.scale = scale

    def compute(
        self,
        pre_step_game_state: GameState,
        post_step_game_state: GameState,
        investment_decisions: InvestmentDecisions = None,
    ) -> float:
        """Compute reward as change in cumulative ROI."""
        pre_cost = sum(pre_step_game_state.realised_costs)
        post_cost = sum(post_step_game_state.realised_costs)

        pre_rev = sum(pre_step_game_state.realised_revenues)
        post_rev = sum(post_step_game_state.realised_revenues)

        pre_roi = (pre_rev / pre_cost) if pre_cost > 0 else 0.0
        post_roi = (post_rev / post_cost) if post_cost > 0 else 0.0

        return (post_roi - pre_roi) * self.scale


class DeltaENPVReward(Reward):
    """Reward is change in eNPV during the step."""

    def __init__(self, scale: float = 1.0):
        """
        Initialize DeltaENPVReward.

        Parameters
        ----------
        scale : float
            Multiplier applied to the eNPV delta.

        """
        self.scale = scale

    def compute(
        self,
        pre_step_game_state: GameState,
        post_step_game_state: GameState,
        investment_decisions: InvestmentDecisions = None,
    ) -> float:
        """Compute the reward based on change in eNPV."""
        return (post_step_game_state.enpv() - pre_step_game_state.enpv()) * self.scale


class PassTrialPhaseBonus(Reward):
    """Applies bonus for asset passing trial phase."""

    def __init__(self, bonus_amount: float):
        """Initialize PassTrialPhaseBonus."""
        self.bonus_amount = bonus_amount

    def compute(
        self,
        pre_step_game_state: GameState,
        post_step_game_state: GameState,
        investment_decisions: InvestmentDecisions = None,
    ) -> float:
        """Return bonus if trial phase is passed, otherwise 0."""
        # loop through assets and compare their states
        total_bonus = 0.0

        for asset_id, pre_asset in pre_step_game_state.assets.items():
            if asset_id not in post_step_game_state.assets:
                continue
            post_asset = post_step_game_state.assets[asset_id]

            # Check for transition from Development -> Idle/Market (Success)
            if pre_asset.state == AssetState.InDevelopment and (
                post_asset.state == AssetState.Idle
                or post_asset.state == AssetState.OnMarket
            ):
                total_bonus += self.bonus_amount

        return total_bonus


class DeltaEnpvActionBasedReward(Reward):
    """
    Reward is change in eNPV only for assets that received investment actions.

    This reward provides a clearer signal about the direct impact of investment
    decisions by only considering the eNPV delta for assets that were invested in.
    Assets that were not acted upon do not contribute to the reward.
    """

    def compute(
        self,
        pre_step_game_state: GameState,
        post_step_game_state: GameState,
        investment_decisions: InvestmentDecisions = None,
    ) -> float:
        """
        Compute the reward based on change in eNPV for invested assets only.

        Parameters
        ----------
        pre_step_game_state : GameState
            Game state before the action was taken.
        post_step_game_state : GameState
            Game state after the action was taken.
        investment_decisions : dict[uuid.UUID, Optional[Literal["invest"]]], optional
            Dictionary mapping asset IDs to investment decisions. Only assets
            with "invest" actions contribute to the reward.

        Returns
        -------
        float
            Sum of eNPV changes for assets that received investment actions.

        """
        if investment_decisions is None:
            # If no investment decisions provided, return 0
            return 0.0

        total_delta_enpv = 0.0

        for asset_id, decision in investment_decisions.items():
            if decision == "invest":
                # Get the asset from both states
                pre_asset = pre_step_game_state.assets.get(asset_id)
                post_asset = post_step_game_state.assets.get(asset_id)

                if pre_asset is not None and post_asset is not None:
                    # Add the change in eNPV for this asset
                    total_delta_enpv += post_asset.enpv - pre_asset.enpv

        return total_delta_enpv
