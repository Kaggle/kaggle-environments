from __future__ import annotations

import logging
import random
import uuid
from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    field_validator,
)
from scipy.special import expit
from scipy.special import logit as logit_fn

from pyxis_portfolio_challenge.rng import get_game_rng

logger = logging.getLogger(__name__)


class TrialState(Enum):
    """Enum to represent the state of the trial."""

    PENDING = 0
    IN_PROGRESS = 1
    PHASE_SUCCESS = 2
    PHASE_FAILED = 3


class TrialPhase(str, Enum):
    """Enum to represent the phase of the trial."""

    def __new__(cls, value, integer):
        """Override the __new__ method."""
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.integer = integer
        return obj

    PHASE_1 = ("Phase 1", 0)
    PHASE_2 = ("Phase 2", 1)
    PHASE_3 = ("Phase 3", 2)
    APPROVAL = ("Approval", 3)

    @classmethod
    def from_int(cls, input_int):
        """Create an asset state from an integer."""
        for phase in cls:
            if phase.integer == input_int:
                return phase
        raise ValueError(f"{cls.__name__} has no value matching {input_int}")


class Trial(BaseModel):
    """
    A class representing a trial in the simulation.

    Parameters
    ----------
    cost_remaining : float
        The remaining cost for the trial.
    time_remaining : int
        The number of time steps remaining for the trial.
    ptrs: float
        The probability of success for the trial.
        Stands for Probability of Technical and Regulatory Success.
    phase: TrialPhase
        The phase of the trial.
    state: TrialState
        The current state of the trial.
    next_trial_on_success: Trial | None
        The next trial to proceed to upon success of the current trial.

    """

    # Runs validation if fields are updated - might be useful in FUTURE
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    cost_remaining: float  # C
    time_remaining: int  # L
    ptrs: float  # P - This is the OBSERVED PTRS (what agents see, may be noisy)
    phase: TrialPhase
    state: TrialState
    next_trial_on_success: Trial | None

    # PTRS readings feature: precision-weighted accumulator (weight = 1/σ²)
    _ptrs_sample_count: int = PrivateAttr(default=0)
    _ptrs_weighted_sum: float = PrivateAttr(default=0.0)
    _ptrs_total_precision: float = PrivateAttr(default=0.0)

    # PTRS readings feature: hidden true PTRS that readings are drawn around and
    # that trial outcomes are rolled against (episode-level memorization bound).
    _true_ptrs: float | None = PrivateAttr(default=None)

    def __eq__(self, other: "Trial") -> bool:
        """Check equality of two Trial objects (public fields only)."""
        if not isinstance(other, Trial):
            return NotImplemented
        return self.model_dump() == other.model_dump()

    @field_validator("ptrs", mode="after")
    @classmethod
    def validate_success_prob(cls, v) -> float:
        """Validate that the input is a valid probability."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"ptrs must be between 0 and 1, received {v}")
        return v

    @field_validator("cost_remaining", mode="after")
    @classmethod
    def validate_cost_remaining(cls, v) -> float:
        """Validate that cost remaining is non-negative."""
        if v < 0.0:
            raise ValueError(f"cost_remaining must be non-negative, received {v}")
        return v

    @field_validator("time_remaining", mode="after")
    @classmethod
    def validate_time_remaining(cls, v) -> int:
        """Validate that time remaining is non-negative."""
        if v < 0.0:
            raise ValueError(f"time_remaining must be non-negative, received {v}")
        return v

    @property
    def cost_this_step(self) -> float:
        """Calculate the cost of the trial for this time step."""
        if self.time_remaining > 0:
            return self.cost_remaining / self.time_remaining
        return 0

    @property
    def ptrs_sample_mean(self) -> float | None:
        """Precision-weighted mean of the PTRS readings, or None if none taken."""
        if self._ptrs_total_precision == 0.0:
            return None
        return self._ptrs_weighted_sum / self._ptrs_total_precision

    @property
    def ptrs_sample_count(self) -> int:
        """Number of PTRS readings accumulated for this trial."""
        return self._ptrs_sample_count

    @property
    def ptrs_total_precision(self) -> float:
        """Sum of 1/σ² weights across all accumulated PTRS readings."""
        return self._ptrs_total_precision

    def draw_and_accumulate(self, sigma: float, n: int, rng: random.Random) -> None:
        """
        Draw n logit-normal samples and update the precision-weighted running mean.

        Each sample is weighted by 1/σ² so readings taken at closer phase distances
        (lower σ) contribute proportionally more to the estimate.
        """
        true_p = self._true_ptrs if self._true_ptrs is not None else self.ptrs
        eff_sigma = sigma if sigma > 0.0 else 1e-15
        precision = 1.0 / (eff_sigma * eff_sigma)
        for _ in range(n):
            eps = rng.gauss(0.0, sigma)
            sample = float(expit(logit_fn(true_p) + eps))
            self._ptrs_weighted_sum += sample * precision
            self._ptrs_total_precision += precision
            self._ptrs_sample_count += 1

    def start_trial(self) -> "Trial":
        """Start the trial by setting its state to IN_PROGRESS."""
        logger.debug(f"Starting Trial: {self.phase}")
        new_trial = Trial(
            cost_remaining=self.cost_remaining,
            time_remaining=self.time_remaining,
            ptrs=self.ptrs,
            phase=self.phase,
            state=TrialState.IN_PROGRESS,
            next_trial_on_success=self.next_trial_on_success,
        )
        self._copy_private_attrs(new_trial)
        return new_trial

    def stop_trial(self) -> "Trial":
        """
        Stop the trial early (agent decides to abandon).

        Returns a failed trial without rolling for success.

        Returns
        -------
        Trial
            A new Trial with PHASE_FAILED state.

        """
        logger.debug(f"Stopping Trial early: {self.phase}")
        return Trial(
            cost_remaining=0.0,
            time_remaining=0,
            ptrs=0.0,
            phase=self.phase,
            state=TrialState.PHASE_FAILED,
            next_trial_on_success=None,
        )

    def success(self) -> bool:
        """
        Determine if the trial has successfully completed.

        Rolls against the hidden true PTRS when the ptrs_readings feature has
        seeded one (episode-level memorization bound); otherwise against the
        observed PTRS. Readings are purely informational — they let the agent
        estimate _true_ptrs but do not change the outcome roll.
        """
        rng = get_game_rng()
        true_p = self._true_ptrs if self._true_ptrs is not None else self.ptrs
        return rng.random() < true_p

    def _copy_private_attrs(self, new_trial: "Trial") -> None:
        """Copy all private attributes to a new trial instance."""
        # Copy ptrs_readings hidden true PTRS
        new_trial._true_ptrs = self._true_ptrs
        # Copy ptrs_readings precision-weighted accumulator
        new_trial._ptrs_weighted_sum = self._ptrs_weighted_sum
        new_trial._ptrs_total_precision = self._ptrs_total_precision
        new_trial._ptrs_sample_count = self._ptrs_sample_count

    def evolve(self) -> "Trial":
        """
        Evolve the trial.

        Returns:
            Trial: The new trial.

        """
        logger.debug(f"Evolving Trial: {self.phase}")
        if self.time_remaining > 1:
            logger.debug(f"Trial time_remaining: {self.time_remaining}, stepping.")
            # still need to go through time steps for current trial completion
            new_trial = Trial(
                cost_remaining=self.cost_remaining - self.cost_this_step,
                time_remaining=self.time_remaining - 1,
                ptrs=self.ptrs,
                phase=self.phase,
                state=TrialState.IN_PROGRESS,
                next_trial_on_success=self.next_trial_on_success,
            )
            self._copy_private_attrs(new_trial)
            return new_trial

        # now time_remaining must be 1
        logger.debug(
            f"Trial time_remaining: {self.time_remaining}, drawing against PTRS."
        )
        if self.success():
            logger.debug("Trial phase successful")
            if self.next_trial_on_success is not None:
                logger.debug("Returning next trial on success.")
                return self.next_trial_on_success
            else:
                logger.debug(
                    "All phases successful, returning trial with PHASE_SUCCESS."
                )
                new_cost_remaining = self.cost_remaining - self.cost_this_step
                new_time_remaining = self.time_remaining - 1
                assert new_cost_remaining == 0.0, (
                    f"Trial successfully completed, but got new_cost_remaining"
                    f" `{new_cost_remaining}`, expected 0."
                )
                assert new_time_remaining == 0, (
                    f"Trial successfully completed, but got new_time_remaining"
                    f" `{new_time_remaining}`, expected 0"
                )
                return Trial(
                    cost_remaining=0.0,
                    time_remaining=0,
                    ptrs=1.0,
                    phase=self.phase,
                    state=TrialState.PHASE_SUCCESS,
                    next_trial_on_success=None,
                )  # successfully completed all trials

        logger.debug("Phase failed, returning trial with PHASE_FAILED.")
        return Trial(
            cost_remaining=0.0,
            time_remaining=0,
            ptrs=0.0,
            phase=self.phase,
            state=TrialState.PHASE_FAILED,
            next_trial_on_success=None,
        )  # failed trial


def trials_json_to_trials_sequence(
    json: dict,
    asset_id: uuid.UUID,
    pending_trial_phase: str,
    approval_phase_config,
    trial_cost_multiplier: float,
) -> "Trial":
    """
    Convert a JSON schema for trials into a chained Trial object for a DrugAsset.

    Returns the Trial corresponding to `pending_trial_phase`, not always Phase 1.

    Parameters
    ----------
    json : dict
        JSON dictionary containing trial phase definitions (phase_1, phase_2, phase_3).
    seed : int
        Random seed for reproducible trial outcome generation.
    asset_id : uuid.UUID
        Unique identifier of the drug asset these trials belong to.
    pending_trial_phase : str
        The trial phase to start from (e.g. "Phase 1", "Phase 2").
    approval_phase_config : ApprovalPhaseConfig | None
        If provided and enabled, an Approval trial is injected after Phase 3.
    trial_cost_multiplier : float
        Multiplier applied to trial costs from the JSON data.

    """
    ordered_phase_keys = ["phase_1", "phase_2", "phase_3"]

    json_phase_to_trial_phase = {
        "phase_1": "Phase 1",
        "phase_2": "Phase 2",
        "phase_3": "Phase 3",
        "approval": "Approval",
    }

    trial_phase_to_json_phase = {
        "Phase 1": "phase_1",
        "Phase 2": "phase_2",
        "Phase 3": "phase_3",
        "Approval": "approval",
    }

    # Validate schema
    for key in ordered_phase_keys:
        if key not in json:
            raise ValueError(f"Missing trial phase '{key}' in schema")

    # Validate pending phase
    if pending_trial_phase not in trial_phase_to_json_phase:
        raise ValueError(f"Invalid pending_trial_phase '{pending_trial_phase}'")

    # Build the chain backwards, starting from the end
    next_trial: Trial | None = None
    trials_by_phase: dict[str, Trial] = {}

    # If approval phase is enabled, create it first (it's the last in the chain)
    if approval_phase_config is not None and approval_phase_config.enabled:
        rng = get_game_rng()
        duration = rng.randint(
            approval_phase_config.duration_min,
            approval_phase_config.duration_max,
        )
        success_rate = rng.uniform(
            approval_phase_config.success_rate_min,
            approval_phase_config.success_rate_max,
        )
        approval_trial = Trial(
            cost_remaining=approval_phase_config.cost * trial_cost_multiplier,
            time_remaining=duration,
            ptrs=success_rate,
            phase=TrialPhase.APPROVAL,
            state=TrialState.PENDING,
            next_trial_on_success=None,
        )
        trials_by_phase["Approval"] = approval_trial
        next_trial = approval_trial

    for key in reversed(ordered_phase_keys):
        trial_data = dict(json[key])
        trial_data["cost_remaining"] = (
            trial_data["cost_remaining"] * trial_cost_multiplier
        )

        current_trial = Trial(
            **trial_data,
            state=TrialState.PENDING,
            next_trial_on_success=next_trial,
            phase=TrialPhase(json_phase_to_trial_phase[key]),
        )

        trials_by_phase[json_phase_to_trial_phase[key]] = current_trial
        next_trial = current_trial

    # Return the trial corresponding to the pending phase
    head_trial = trials_by_phase[pending_trial_phase]

    logger.debug(f"Created Trial chain starting at {head_trial.phase}")

    return head_trial
