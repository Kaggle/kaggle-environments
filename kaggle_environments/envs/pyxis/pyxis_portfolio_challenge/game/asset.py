from __future__ import annotations

import logging
import uuid
from enum import Enum
from functools import cached_property
from typing import Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
)

from pyxis_portfolio_challenge.game.constants import (
    DISCOUNT_RATE,
    InvestmentLevel,
)
from pyxis_portfolio_challenge.game.trial import Trial, TrialPhase, TrialState

logger = logging.getLogger(__name__)


class AssetState(str, Enum):
    """Enum to represent the state of the asset."""

    def __new__(cls, value, integer):
        """Override the __new__ method."""
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.integer = integer
        return obj

    Idle = ("Idle", 0)
    InDevelopment = ("In Development", 1)
    OnMarket = ("On Market", 2)
    Failed = ("Failed", 3)
    Expired = ("Expired", 4)
    Dropped = ("Dropped", 5)

    @classmethod
    def from_int(cls, input_int):
        """Create an asset state an integer."""
        for state in cls:
            if state.integer == input_int:
                return state
        raise ValueError(f"{cls.__name__} has no value matching {input_int}")


def revenue_formula(
    time_on_market: int, max_revenue: float, time_until_max_revenue: int
) -> float:
    """
    Compute the revenue after a given amount of time on the market.

    Usage: suppose max_revenue = P and time_until_max_revenue = T. Then, assuming no
    patent expiry, the asset accrues the following revenue amounts for the first T
    steps after reaching On Market:
        P * 1/(T+1), P * 2/(T+1), ... , P * T/(T+1);
    and P for every step thereafter.

    The behaviour of this function is linear growth until game time is equal to
    time_until_max_revenue, and flat thereafter.

    Example:
        Drug has just reached it to market with max_revenue = 1000 and
        time_until_max_revenue = 4. The revenue over the next 6 time steps will be:
            Step 0: 200  # At launch
            Step 1: 400
            Step 2: 600
            Step 3: 800
            Step 4: 1000  # Max revenue reached
            Step 5: 1000  # Max revenue maintained

    """
    multiplier = min(time_on_market / (time_until_max_revenue + 1), 1)
    return multiplier * max_revenue


class DrugAsset(BaseModel):
    """
    A class representing a drug asset in the simulation.

    Parameters
    ----------
    id : uuid.UUID
        Unique identifier for the asset.
    name : str
        Name of the asset.
    therapeutic_area : str
        Therapeutic area of the asset (e.g., oncology, respiratory and immunology).
    type : str
        Asset type ("internal" or "BD").
    description : str
        Description of the asset.
    max_revenue : float
        Maximum revenue made by the asset (AKA Peak Year Sales).
    time_until_max_revenue : int
        Number of time steps it takes to reach maximum revenue after launch.
    time_until_patent_expiry : int
        Time steps until patent expiry (patent maturity).
    trial : Trial
        Current trial associated with the asset.
    state : AssetState
        Current state of the asset.
    time_on_market: int = 0
        Number of time steps the asset has been on the market.
    The game-wide RNG is accessed via `get_game_rng()` from the `rng` module.

    """

    # FUTURE: Have validation that checks if total time left in development exceeds
    # time left on patent. If so, delete asset or have it greyed out.

    # Runs validation if fields are updated
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: uuid.UUID
    name: str
    therapeutic_area: Literal[
        "oncology", "respiratory and immunology", "vaccines and infectious disease"
    ]
    indication: int = 0
    type: Literal["internal", "BD"]
    description: str
    max_revenue: float  # M (scaled by dynamic scope at arrival)
    raw_max_revenue: float  # M before dynamic scope scaling; used for BE floor norm
    time_until_max_revenue: int  # H
    time_until_patent_expiry: int  # T
    trial: Trial
    state: AssetState
    time_on_market: int
    current_investment_level: InvestmentLevel = InvestmentLevel.NONE

    def __eq__(self, other: "DrugAsset") -> bool:
        """Check equality of two DrugAsset objects."""
        if not isinstance(other, DrugAsset):
            return NotImplemented
        # don't compare id
        attrs_to_ignore = {"id"}
        return self.model_dump(exclude=attrs_to_ignore) == other.model_dump(
            exclude=attrs_to_ignore
        )

    @field_validator("max_revenue", mode="after")
    @classmethod
    def validate_max_revenue(cls, v) -> float:
        """Validate that max_revenue is non-negative."""
        if v < 0:
            raise ValueError(f"max_revenue must be non-negative, received: {v}")
        return v

    @field_validator("time_until_max_revenue", mode="after")
    @classmethod
    def validate_time_until_max_revenue(cls, v) -> int:
        """Validate that time_until_max_revenue is positive."""
        if v <= 0:
            raise ValueError(f"time_until_max_revenue must be positive, received: {v}")
        return v

    @field_validator("time_until_patent_expiry", mode="after")
    @classmethod
    def validate_time_until_patent_expiry(cls, v) -> int:
        """Validate that time_until_patent_expiry is non-negative."""
        if v < 0:
            raise ValueError(
                f"time_until_patent_expiry must be non-negative, received: {v}"
            )
        return v

    @field_validator("time_on_market", mode="after")
    @classmethod
    def validate_time_on_market(cls, v) -> int:
        """Validate that time_on_market is non-negative."""
        if v < 0:
            raise ValueError(f"time_on_market must be non-negative, received: {v}")
        return v

    def _get_future_events(self) -> iter[tuple[float, float]]:
        """Yields (cash_flow, probability) for each future time step."""
        if self.state in [AssetState.Failed, AssetState.Expired]:
            return

        accumulated_prob = 1.0
        time_step = 0
        _trial = self.trial

        # Trial period
        while _trial is not None and time_step < self.time_until_patent_expiry:
            for _ in range(_trial.time_remaining):
                yield -_trial.cost_this_step, accumulated_prob
                time_step += 1
                if time_step >= self.time_until_patent_expiry:
                    break
            accumulated_prob *= _trial.ptrs
            _trial = _trial.next_trial_on_success

        # On-market period
        projected_time_on_market = self.time_on_market
        while time_step < self.time_until_patent_expiry:
            projected_time_on_market += 1
            revenue = revenue_formula(
                projected_time_on_market,
                max_revenue=self.max_revenue,
                time_until_max_revenue=self.time_until_max_revenue,
            )
            yield revenue, accumulated_prob
            time_step += 1

    @cached_property
    def _projected_cash_flows(self) -> list[float]:
        return [event[0] for event in self._get_future_events()]

    @cached_property
    def _projected_probs(self) -> list[float]:
        return [event[1] for event in self._get_future_events()]

    @property
    def expected_costs_and_revenues(self) -> tuple[list[float], list[float]]:
        """Get lists of expected costs and expected revenues for the asset."""
        cash_flows = np.array(self._projected_cash_flows)
        probabilities = np.array(self._projected_probs)
        expected = cash_flows * probabilities
        expected_costs = list(
            -np.where(expected < 0, expected, 0)
        )  # Note the sign change
        expected_revenues = list(np.where(expected > 0, expected, 0))
        return expected_costs, expected_revenues

    @property
    def enpv(self) -> float:
        """Calculate the expected Net Present Value of the asset using cached values."""
        cash_flows = np.array(self._projected_cash_flows)
        probabilities = np.array(self._projected_probs)
        time_steps = np.arange(len(cash_flows))
        discount_factors = 1 / ((1 + DISCOUNT_RATE) ** time_steps)

        # Multiply cash_flows by probabilities by discount_factors
        enpv = float(np.sum(cash_flows * probabilities * discount_factors))
        return enpv

    def cash_enpv(self, reinvestment_percentage: float) -> float:
        """Expected NPV under cash accounting: revenues scaled, costs at full."""
        cash_flows = np.array(self._projected_cash_flows)
        probabilities = np.array(self._projected_probs)
        time_steps = np.arange(len(cash_flows))
        discount_factors = 1 / ((1 + DISCOUNT_RATE) ** time_steps)
        scaled = np.where(
            cash_flows > 0, cash_flows * reinvestment_percentage, cash_flows
        )
        return float(np.sum(scaled * probabilities * discount_factors))

    @property
    def eroi(self) -> float:
        """Calculate expected Return on Investment of the asset using cached values."""
        cash_flows = np.array(self._projected_cash_flows)
        probabilities = np.array(self._projected_probs)
        expected = cash_flows * probabilities
        expected_costs = list(
            -np.where(expected < 0, expected, 0)
        )  # Note the sign change
        expected_revenues = list(np.where(expected > 0, expected, 0))
        total_expected_cost = sum(expected_costs)
        if total_expected_cost == 0:
            return 0.0
        total_expected_revenue = sum(expected_revenues)
        return (total_expected_revenue - total_expected_cost) / total_expected_cost

    def to_develop(self, enable_interim_observations: bool = False) -> "DrugAsset":
        """
        Set the asset to InDevelopment state if possible.

        Parameters
        ----------
        enable_interim_observations : bool
            If True, initialize latent quality for interim trial observations.

        """
        if self.state != AssetState.Idle:
            raise ValueError(
                f"Cannot develop asset {self.name} as it is not Idle "
                f"(current state: {self.state})."
                f" Use env.action_masks to prevent this."
            )
        new_asset = self.model_copy(
            update={
                "state": AssetState.InDevelopment,
                "trial": self.trial.start_trial(
                    enable_interim_observations=enable_interim_observations
                ),
            }
        )
        return new_asset

    @model_validator(mode="after")
    def check_trial_in_progress_for_in_development(
        self,
    ) -> "DrugAsset":
        """Validate that trial is in progress if state is InDevelopment."""
        if self.state == AssetState.InDevelopment:
            if self.trial.state != TrialState.IN_PROGRESS:
                raise ValueError(
                    f"Trial must be in progress if asset is InDevelopment, "
                    f"received trial state: {self.trial.state}"
                )
        return self

    @model_validator(mode="after")
    def check_trial_phase_success_for_on_market(
        self,
    ) -> "DrugAsset":
        """Validate that trial is phase success if state is OnMarket."""
        if self.state == AssetState.OnMarket:
            if self.trial.state != TrialState.PHASE_SUCCESS:
                raise ValueError(
                    f"Trial must be state {TrialState.PHASE_SUCCESS} if asset"
                    f" is OnMarket, received trial state: {self.trial.state}"
                )
        return self

    @model_validator(mode="after")
    def check_time_on_market_zero_for_non_on_market(
        self,
    ) -> "DrugAsset":
        """Validate that time_on_market is zero if state is not OnMarket or Expired."""
        if self.state not in [AssetState.OnMarket, AssetState.Expired]:
            if self.time_on_market != 0:
                raise ValueError(
                    f"time_on_market must be 0 if asset is not OnMarket, "
                    f"received time_on_market: {self.time_on_market}"
                )
        return self

    @property
    def cost_this_step(self) -> float:
        """Get the cost incurred this step."""
        if self.state == AssetState.InDevelopment:
            return self.trial.cost_this_step
        return 0.0

    @property
    def cost_to_invest_this_step(self) -> float:
        """
        If the asset is invested in this step, get the cost incurred this step.

        This is a forward-looking property used in the Knapsack agent.
        """
        if self.state == AssetState.InDevelopment:
            raise ValueError(
                "Asset is already InDevelopment, cannot invest again this step."
            )
        return self.trial.cost_this_step

    @property
    def remaining_trial_cost(self) -> float:
        """Get the total cost of the next trial associated with the asset."""
        return self.trial.cost_remaining

    @property
    def revenue_this_step(self) -> float:
        """Get the revenue generated this step."""
        if self.state == AssetState.OnMarket:
            return revenue_formula(
                self.time_on_market,
                max_revenue=self.max_revenue,
                time_until_max_revenue=self.time_until_max_revenue,
            )
        return 0.0

    @property
    def trial_progress(self) -> float:
        """
        Get the current trial progress as a fraction from 0.0 to 1.0.

        Returns 0.0 if asset is not in development or interim observations
        are not enabled.
        """
        if self.state != AssetState.InDevelopment:
            return 0.0
        return self.trial.progress

    @property
    def interim_signal(self) -> float:
        """
        Get a noisy observation of the trial's latent quality.

        The signal becomes clearer as the trial progresses. Returns the
        trial's PTRS if interim observations are not enabled.

        Returns
        -------
        float
            A value in [0, 1] indicating the estimated success probability.

        """
        if self.state != AssetState.InDevelopment:
            return self.trial.ptrs
        return self.trial.get_interim_signal()

    @property
    def interim_observations_enabled(self) -> bool:
        """Check if interim observations are enabled for this asset's trial."""
        return self.trial._interim_observations_enabled

    @property
    def pending_trial_chain(self) -> list[Trial]:
        """Ordered list of non-approval, non-terminal trials from current position."""
        chain = []
        t = self.trial
        while t is not None:
            terminal = (TrialState.PHASE_FAILED, TrialState.PHASE_SUCCESS)
            if t.phase != TrialPhase.APPROVAL and t.state not in terminal:
                chain.append(t)
            t = t.next_trial_on_success
        return chain

    def apply_pending_readings(
        self,
        n: int,
        sigma_base: float,
        noise_multipliers: list[float],
        rng,
    ) -> None:
        """Draw n readings for every pending trial, scaling noise by phase distance."""
        chain = self.pending_trial_chain
        if len(chain) > len(noise_multipliers):
            raise ValueError(
                f"pending_trial_chain length {len(chain)} exceeds "
                f"noise_multipliers length {len(noise_multipliers)}"
            )
        for i, trial in enumerate(chain):
            trial.draw_and_accumulate(sigma_base * noise_multipliers[i], n, rng)

    def initialise_ptrs_readings(
        self,
        sigma_base: float,
        noise_multipliers: list[float],
        rng,
    ) -> None:
        """Seed each pending trial with one free reading at asset creation."""
        chain = self.pending_trial_chain
        if len(chain) > len(noise_multipliers):
            raise ValueError(
                f"pending_trial_chain length {len(chain)} exceeds "
                f"noise_multipliers length {len(noise_multipliers)}"
            )
        for i, trial in enumerate(chain):
            trial.draw_and_accumulate(sigma_base * noise_multipliers[i], 1, rng)

    def cost_this_step_with_modifier(self, cost_modifier: float) -> float:
        """Get the cost incurred this step with investment level modifier."""
        if self.state == AssetState.InDevelopment:
            return self.trial.cost_this_step * cost_modifier
        return 0.0

    def cost_to_invest_with_modifier(self, cost_modifier: float) -> float:
        """Get cost to invest this step with modifier (for idle assets)."""
        if self.state == AssetState.InDevelopment:
            raise ValueError(
                "Asset is already InDevelopment, cannot invest again this step."
            )
        return self.trial.cost_this_step * cost_modifier

    def to_develop_with_level(
        self, level: InvestmentLevel, enable_interim_observations: bool = False
    ) -> "DrugAsset":
        """
        Set the asset to InDevelopment state with specified investment level.

        Parameters
        ----------
        level : InvestmentLevel
            The investment level to use.
        enable_interim_observations : bool
            If True, initialize latent quality for interim trial observations.

        """
        if self.state != AssetState.Idle:
            raise ValueError(
                f"Cannot develop asset {self.name} as it is not Idle "
                f"(current state: {self.state})."
            )
        new_asset = self.model_copy(
            update={
                "state": AssetState.InDevelopment,
                "trial": self.trial.start_trial(
                    enable_interim_observations=enable_interim_observations
                ),
                "current_investment_level": level,
            }
        )
        return new_asset

    def set_investment_level(self, level: InvestmentLevel) -> "DrugAsset":
        """Change the investment level for an in-development asset."""
        if self.state != AssetState.InDevelopment:
            raise ValueError(
                f"Cannot set investment level for asset {self.name} "
                f"as it is not InDevelopment (current state: {self.state})."
            )
        new_asset = self.model_copy(update={"current_investment_level": level})
        return new_asset

    def stop_development(self) -> "DrugAsset":
        """
        Stop development of the asset early (agent decides to abandon).

        This allows the agent to cut losses on trials that appear to be
        failing based on interim observations.

        Returns
        -------
        DrugAsset
            A new asset with Failed state.

        Raises
        ------
        ValueError
            If the asset is not currently in development.

        """
        if self.state != AssetState.InDevelopment:
            raise ValueError(
                f"Cannot stop development of asset {self.name} "
                f"as it is not InDevelopment (current state: {self.state})."
            )
        logger.debug(f"Stopping development of asset: {self.name}")
        new_asset = DrugAsset(
            id=self.id,
            name=self.name,
            therapeutic_area=self.therapeutic_area,
            indication=self.indication,
            type=self.type,
            description=self.description,
            max_revenue=self.max_revenue,
            raw_max_revenue=self.raw_max_revenue,
            time_until_max_revenue=self.time_until_max_revenue,
            time_until_patent_expiry=self.time_until_patent_expiry,
            state=AssetState.Failed,
            time_on_market=0,
            trial=self.trial.stop_trial(),
            current_investment_level=InvestmentLevel.NONE,
        )
        return new_asset

    def drop(self) -> "DrugAsset":
        """
        Drop this asset from the portfolio (agent decision, any state).

        Distinct from Failed (trial outcome) — Dropped means the agent
        deliberately removed the asset regardless of its development status.
        Valid for Idle, InDevelopment, and OnMarket assets.
        """
        if self.state in (AssetState.Failed, AssetState.Expired, AssetState.Dropped):
            raise ValueError(
                f"Cannot drop asset {self.name} in terminal state {self.state}."
            )
        logger.debug(f"Dropping asset: {self.name} (was {self.state})")
        return DrugAsset(
            id=self.id,
            name=self.name,
            therapeutic_area=self.therapeutic_area,
            indication=self.indication,
            type=self.type,
            description=self.description,
            max_revenue=self.max_revenue,
            raw_max_revenue=self.raw_max_revenue,
            time_until_max_revenue=self.time_until_max_revenue,
            time_until_patent_expiry=self.time_until_patent_expiry,
            state=AssetState.Dropped,
            time_on_market=0,
            trial=(
                self.trial.stop_trial()
                if self.state == AssetState.InDevelopment
                else self.trial
            ),
            current_investment_level=InvestmentLevel.NONE,
        )

    def evolve(self) -> "DrugAsset":
        """Evolve the asset by one time step."""
        new_time_on_market = 0

        new_trial = self.trial
        new_state = self.state  # initialise new state as current state

        if new_state == AssetState.OnMarket:
            new_time_on_market = self.time_on_market + 1

        if new_state == AssetState.InDevelopment:
            logger.debug(f"Asset InDevelopment: {self.name}")
            logger.debug(f"Evolving Trial: {self.trial}")
            new_trial: Trial = self.trial.evolve()
            if new_trial.state == TrialState.PENDING:
                logger.debug(
                    f"Trial returned to pending asset set to Idle: {self.name}"
                )
                new_state = AssetState.Idle
            elif new_trial.state == TrialState.PHASE_FAILED:
                logger.debug(f"Trial failed asset set to Failed: {self.name}")
                new_state = AssetState.Failed
            elif new_trial.state == TrialState.PHASE_SUCCESS:
                # only here if final trial phase succeeded
                # intermediate trials will return next_trial_on_success
                # in PENDING state
                logger.debug(
                    f"Final trial succeeded asset set to OnMarket: {self.name}"
                )
                new_state = AssetState.OnMarket
                # Start on the market at time_on_market = 1 so the drug's first
                # on-market step earns revenue. Revenue is collected (game_state
                # Step B) before evolve (Step C) each step, so this value is what
                # the first collection reads; leaving it at 0 makes
                # revenue_formula(0) = 0 and wastes the first market step. This
                # matches the eNPV projection in _get_future_events, which
                # increments time_on_market before computing revenue.
                new_time_on_market = 1
            elif new_trial.state == TrialState.IN_PROGRESS:
                logger.debug(
                    f"Trial still in progress asset stays InDevelopment: {self.name}"
                )
            else:
                raise ValueError(f"Unknown Trial State: {new_trial.state}")

        new_time_until_patent_expiry = self.time_until_patent_expiry - 1
        if new_time_until_patent_expiry < 1:
            logger.debug(f"Asset Expired: {self.name}")
            new_state = AssetState.Expired

        new_asset = DrugAsset(
            id=self.id,
            name=self.name,
            therapeutic_area=self.therapeutic_area,
            indication=self.indication,
            type=self.type,
            description=self.description,
            max_revenue=self.max_revenue,
            raw_max_revenue=self.raw_max_revenue,
            time_until_max_revenue=self.time_until_max_revenue,
            time_until_patent_expiry=new_time_until_patent_expiry,
            state=new_state,
            time_on_market=new_time_on_market,
            trial=new_trial,
            current_investment_level=InvestmentLevel.NONE,
        )
        return new_asset

    def evolve_with_level(
        self,
        speed_modifier: float,
        success_modifier: float,
        global_success_modifier: float = 1.0,
    ) -> "DrugAsset":
        """
        Evolve the asset by one time step with investment level modifiers.

        Parameters
        ----------
        speed_modifier : float
            Multiplier for trial progress speed.
        success_modifier : float
            Multiplier for success probability from investment level.
        global_success_modifier : float
            Additional success modifier from capacity overage.

        Returns
        -------
        DrugAsset
            The evolved asset.

        """
        new_time_on_market = 0
        new_trial = self.trial
        new_state = self.state
        new_investment_level = self.current_investment_level

        if new_state == AssetState.OnMarket:
            new_time_on_market = self.time_on_market + 1
            new_investment_level = InvestmentLevel.NONE

        if new_state == AssetState.InDevelopment:
            logger.debug(
                f"Asset InDevelopment: {self.name}, level={new_investment_level}"
            )
            new_trial = self.trial.evolve_with_level(
                speed_modifier=speed_modifier,
                success_modifier=success_modifier,
                global_success_modifier=global_success_modifier,
            )

            if new_trial.state == TrialState.PENDING:
                logger.debug(
                    f"Trial returned to pending, asset set to Idle: {self.name}"
                )
                new_state = AssetState.Idle
                new_investment_level = InvestmentLevel.NONE
            elif new_trial.state == TrialState.PHASE_FAILED:
                logger.debug(f"Trial failed, asset set to Failed: {self.name}")
                new_state = AssetState.Failed
                new_investment_level = InvestmentLevel.NONE
            elif new_trial.state == TrialState.PHASE_SUCCESS:
                logger.debug(
                    f"Final trial succeeded, asset set to OnMarket: {self.name}"
                )
                new_state = AssetState.OnMarket
                new_investment_level = InvestmentLevel.NONE
                # See evolve(): start at time_on_market = 1 so the first
                # on-market step earns revenue instead of revenue_formula(0) = 0.
                new_time_on_market = 1
            elif new_trial.state == TrialState.IN_PROGRESS:
                logger.debug(
                    f"Trial in progress, asset stays InDevelopment: {self.name}"
                )

        new_time_until_patent_expiry = self.time_until_patent_expiry - 1
        if new_time_until_patent_expiry < 1:
            logger.debug(f"Asset Expired: {self.name}")
            new_state = AssetState.Expired
            new_investment_level = InvestmentLevel.NONE

        new_asset = DrugAsset(
            id=self.id,
            name=self.name,
            therapeutic_area=self.therapeutic_area,
            indication=self.indication,
            type=self.type,
            description=self.description,
            max_revenue=self.max_revenue,
            raw_max_revenue=self.raw_max_revenue,
            time_until_max_revenue=self.time_until_max_revenue,
            time_until_patent_expiry=new_time_until_patent_expiry,
            state=new_state,
            time_on_market=new_time_on_market,
            trial=new_trial,
            current_investment_level=new_investment_level,
        )
        return new_asset
