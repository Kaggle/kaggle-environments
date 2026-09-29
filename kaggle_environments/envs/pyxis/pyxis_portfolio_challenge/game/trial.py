from __future__ import annotations

import logging
import random
import uuid
from enum import Enum

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    field_validator,
)
from scipy.special import expit
from scipy.special import logit as logit_fn
from scipy.stats import beta as beta_dist

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
    progress_accumulated: float
        Accumulated progress toward completing the trial phase.
        Used with investment levels to track fractional progress.

    """

    # Runs validation if fields are updated - might be useful in FUTURE
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    cost_remaining: float  # C
    time_remaining: int  # L
    ptrs: float  # P - This is the OBSERVED PTRS (what agents see, may be noisy)
    phase: TrialPhase
    state: TrialState
    next_trial_on_success: Trial | None
    progress_accumulated: float = 0.0  # For investment level speed tracking

    # PTRS readings feature: precision-weighted accumulator (weight = 1/σ²)
    _ptrs_sample_count: int = PrivateAttr(default=0)
    _ptrs_weighted_sum: float = PrivateAttr(default=0.0)
    _ptrs_total_precision: float = PrivateAttr(default=0.0)

    # Uncertain PTRS attributes (used when uncertain_ptrs feature is enabled)
    _true_ptrs: float | None = PrivateAttr(default=None)  # Hidden true PTRS
    _initial_observed_ptrs: float | None = PrivateAttr(
        default=None
    )  # For interpolation
    _effective_true_ptrs: float | None = PrivateAttr(
        default=None
    )  # true + expertise boost

    # Interim trial observations (Option 2: Latent quality variable)
    _latent_quality: float | None = PrivateAttr(default=None)
    _initial_time_remaining: int | None = PrivateAttr(default=None)
    _interim_observations_enabled: bool = PrivateAttr(default=False)

    # Distributional PTRS feature: Beta distribution parameters
    # Prior parameters: what the agent observes (centered on base_ptrs)
    _ptrs_prior_alpha: float | None = PrivateAttr(default=None)
    _ptrs_prior_beta: float | None = PrivateAttr(default=None)
    # True parameters: hidden distribution (centered on base_ptrs + ta_modifier + noise)
    # Used for sampling actual outcomes - the agent never sees these
    _ptrs_true_alpha: float | None = PrivateAttr(default=None)
    _ptrs_true_beta: float | None = PrivateAttr(default=None)
    # Cached percentiles (computed once when distribution is set)
    _ptrs_range_low_cached: float | None = PrivateAttr(default=None)
    _ptrs_range_high_cached: float | None = PrivateAttr(default=None)
    # Cached experience fraction to avoid redundant updates
    _last_exp_fraction: float | None = PrivateAttr(default=None)

    def __eq__(self, other: "Trial") -> bool:
        """Check equality of two Trial objects."""
        if not isinstance(other, Trial):
            return NotImplemented
        attrs_to_ignore = {
            "_true_ptrs",
            "_initial_observed_ptrs",
            "_effective_true_ptrs",
            "_ptrs_prior_alpha",
            "_ptrs_prior_beta",
            "_ptrs_true_alpha",
            "_ptrs_true_beta",
        }
        return self.model_dump(exclude=attrs_to_ignore) == other.model_dump(
            exclude=attrs_to_ignore
        )

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
    def progress(self) -> float:
        """
        Calculate trial progress as a fraction from 0.0 to 1.0.

        Returns 0.0 if initial_time_remaining is not set.
        """
        if self._initial_time_remaining is None or self._initial_time_remaining == 0:
            return 0.0
        elapsed = self._initial_time_remaining - self.time_remaining
        return min(1.0, elapsed / self._initial_time_remaining)

    @property
    def ptrs_expected(self) -> float:
        """
        Get the expected PTRS from the Beta prior distribution.

        Returns the observed PTRS if distributional PTRS is not enabled.
        """
        if self._ptrs_prior_alpha is None or self._ptrs_prior_beta is None:
            return self.ptrs
        return self._ptrs_prior_alpha / (self._ptrs_prior_alpha + self._ptrs_prior_beta)

    @property
    def ptrs_confidence(self) -> float:
        """
        Get confidence in PTRS estimate (0=very uncertain, 1=certain).

        For distributional PTRS: based on Beta concentration.
        For uncertain PTRS: based on convergence toward true PTRS.
        Otherwise: full confidence (no uncertainty feature active).
        """
        # Distributional PTRS: concentration-based confidence
        if self._ptrs_prior_alpha is not None and self._ptrs_prior_beta is not None:
            concentration = self._ptrs_prior_alpha + self._ptrs_prior_beta
            # Map concentration to [0,1]: 2 -> 0, 50+ -> ~1
            return min(1.0, (concentration - 2) / 48)

        # Uncertain PTRS: convergence-based confidence
        if self._true_ptrs is not None and self._initial_observed_ptrs is not None:
            initial_error = abs(self._initial_observed_ptrs - self._true_ptrs)
            if initial_error < 1e-6:
                return 1.0  # No noise was applied
            current_error = abs(self.ptrs - self._true_ptrs)
            # Confidence = how much of the initial error has been resolved
            return min(1.0, max(0.0, 1.0 - current_error / initial_error))

        return 1.0  # No uncertainty feature active

    @property
    def ptrs_range_low(self) -> float:
        """
        Get the 10th percentile / pessimistic PTRS estimate.

        For distributional PTRS: 10th percentile of Beta prior.
        For uncertain PTRS: observed PTRS minus remaining uncertainty.
        Otherwise: observed PTRS (no range).
        """
        # Distributional PTRS
        if self._ptrs_prior_alpha is not None and self._ptrs_prior_beta is not None:
            if self._ptrs_range_low_cached is not None:
                return self._ptrs_range_low_cached
            return float(
                beta_dist.ppf(0.10, self._ptrs_prior_alpha, self._ptrs_prior_beta)
            )

        # Uncertain PTRS: range based on remaining error
        if self._true_ptrs is not None and self._initial_observed_ptrs is not None:
            initial_error = abs(self._initial_observed_ptrs - self._true_ptrs)
            current_error = abs(self.ptrs - self._true_ptrs)
            remaining_uncertainty = max(initial_error, current_error)
            return max(0.0, self.ptrs - remaining_uncertainty)

        return self.ptrs

    @property
    def ptrs_range_high(self) -> float:
        """
        Get the 90th percentile / optimistic PTRS estimate.

        For distributional PTRS: 90th percentile of Beta prior.
        For uncertain PTRS: observed PTRS plus remaining uncertainty.
        Otherwise: observed PTRS (no range).
        """
        # Distributional PTRS
        if self._ptrs_prior_alpha is not None and self._ptrs_prior_beta is not None:
            if self._ptrs_range_high_cached is not None:
                return self._ptrs_range_high_cached
            return float(
                beta_dist.ppf(0.90, self._ptrs_prior_alpha, self._ptrs_prior_beta)
            )

        # Uncertain PTRS: range based on remaining error
        if self._true_ptrs is not None and self._initial_observed_ptrs is not None:
            initial_error = abs(self._initial_observed_ptrs - self._true_ptrs)
            current_error = abs(self.ptrs - self._true_ptrs)
            remaining_uncertainty = max(initial_error, current_error)
            return min(1.0, self.ptrs + remaining_uncertainty)

        return self.ptrs

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

    def initialize_latent_quality(self, rng: random.Random | None = None) -> None:
        """
        Initialize the latent quality for interim observations.

        Samples from Beta distribution centered on the TRUE PTRS (if available).
        This ensures interim signals properly reflect hidden TA quality modifiers
        when distributional PTRS is enabled.

        Should be called when trial starts (state -> IN_PROGRESS).

        Parameters
        ----------
        rng : random.Random | None
            Random number generator. Uses get_game_rng() if not provided.

        """
        if rng is None:
            rng = get_game_rng()

        # Use effective true PTRS if available (distributional/uncertain PTRS feature)
        # This ensures latent quality reflects hidden TA quality modifiers
        # Otherwise fall back to observed PTRS (backwards compatible)
        base_prob = (
            self._effective_true_ptrs
            if self._effective_true_ptrs is not None
            else self.ptrs
        )

        # Sample latent quality from Beta distribution
        # Alpha/beta chosen so mean = base_prob, with reasonable variance
        concentration = 10.0  # Higher = less variance
        alpha = base_prob * concentration
        beta = (1 - base_prob) * concentration

        # Handle edge cases
        if alpha <= 0:
            alpha = 0.1
        if beta <= 0:
            beta = 0.1

        # Use numpy for Beta sampling (convert RNG state)
        np_rng = np.random.RandomState(rng.randint(0, 2**31))
        self._latent_quality = float(np_rng.beta(alpha, beta))
        self._initial_time_remaining = self.time_remaining
        self._interim_observations_enabled = True

        logger.debug(
            f"Initialized latent quality: {self._latent_quality:.3f} "
            f"(base_prob: {base_prob:.3f}, observed_ptrs: {self.ptrs:.3f})"
        )

    def get_interim_signal(self) -> float:
        """
        Get a noisy observation of the latent quality.

        Signal becomes clearer (less noisy) as trial progresses.

        Returns
        -------
        float
            Noisy signal in [0, 1] indicating likely trial outcome.
            Returns PTRS if interim observations not enabled.

        """
        if not self._interim_observations_enabled or self._latent_quality is None:
            return self.ptrs

        # Noise scale decreases with progress
        # At 0% progress: noise_scale = 0.3 (very noisy)
        # At 100% progress: noise_scale = 0.0 (exact)
        noise_scale = 0.3 * (1 - self.progress)

        # Generate noise using the trial's RNG
        noise = get_game_rng().gauss(0, noise_scale)

        # Return noisy signal, clipped to [0, 1]
        signal = self._latent_quality + noise
        return max(0.0, min(1.0, signal))

    def success_from_latent_quality(self) -> bool:
        """
        Determine trial success using latent quality.

        If latent quality is not set, falls back to standard success().

        Returns
        -------
        bool
            True if trial succeeds.

        """
        if self._latent_quality is None:
            return self.success()

        # Success is probabilistic based on latent quality
        return get_game_rng().random() < self._latent_quality

    def start_trial(self, enable_interim_observations: bool = False) -> "Trial":
        """
        Start the trial by setting its state to IN_PROGRESS.

        Parameters
        ----------
        enable_interim_observations : bool
            If True, initialize latent quality for interim observations.

        """
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

        # Initialize interim observations if enabled
        if enable_interim_observations:
            new_trial._initial_time_remaining = self.time_remaining
            new_trial.initialize_latent_quality()

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

        Priority:
        1. Distributional PTRS: Sample from true Beta, compare to random
        2. Otherwise: the point value from ``_outcome_ptrs``
        """
        rng = get_game_rng()

        # Distributional PTRS: sample from true Beta distribution
        if self._ptrs_true_alpha is not None and self._ptrs_true_beta is not None:
            # Sample PTRS from the true (hidden) distribution
            np_rng = np.random.RandomState(rng.randint(0, 2**31))
            sampled_ptrs = float(
                np_rng.beta(self._ptrs_true_alpha, self._ptrs_true_beta)
            )
            # Compare to random draw
            return rng.random() < sampled_ptrs

        return rng.random() < self._outcome_ptrs

    @property
    def _outcome_ptrs(self) -> float:
        """
        The point PTRS a trial's outcome is drawn against.

        The experience-boosted true value when TA experience sets one, else the
        hidden true value, else the observed PTRS. With PTRS readings on,
        ``ptrs`` is the agent's noisy estimate of ``_true_ptrs``; drawing
        against it would let a lucky reading raise the real odds.
        """
        if self._effective_true_ptrs is not None:
            return self._effective_true_ptrs
        if self._true_ptrs is not None:
            return self._true_ptrs
        return self.ptrs

    def _copy_private_attrs(self, new_trial: "Trial") -> None:
        """Copy all private attributes to a new trial instance."""
        # Copy uncertain PTRS attributes
        new_trial._true_ptrs = self._true_ptrs
        new_trial._initial_observed_ptrs = self._initial_observed_ptrs
        new_trial._effective_true_ptrs = self._effective_true_ptrs
        # Copy interim observation attributes
        new_trial._latent_quality = self._latent_quality
        new_trial._initial_time_remaining = self._initial_time_remaining
        new_trial._interim_observations_enabled = self._interim_observations_enabled
        # Copy distributional PTRS attributes (both prior and true distributions)
        new_trial._ptrs_prior_alpha = self._ptrs_prior_alpha
        new_trial._ptrs_prior_beta = self._ptrs_prior_beta
        new_trial._ptrs_true_alpha = self._ptrs_true_alpha
        new_trial._ptrs_true_beta = self._ptrs_true_beta
        # Copy cached percentiles and experience fraction
        new_trial._ptrs_range_low_cached = self._ptrs_range_low_cached
        new_trial._ptrs_range_high_cached = self._ptrs_range_high_cached
        new_trial._last_exp_fraction = self._last_exp_fraction
        # Copy ptrs_readings precision-weighted accumulator
        new_trial._ptrs_weighted_sum = self._ptrs_weighted_sum
        new_trial._ptrs_total_precision = self._ptrs_total_precision
        new_trial._ptrs_sample_count = self._ptrs_sample_count

    def _determine_success(self) -> bool:
        """
        Determine trial success using appropriate method.

        Uses latent quality if enabled, otherwise falls back to standard success().
        """
        if self._interim_observations_enabled and self._latent_quality is not None:
            return self.success_from_latent_quality()
        return self.success()

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
        if self._determine_success():
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

    def evolve_with_level(
        self,
        speed_modifier: float,
        success_modifier: float,
        global_success_modifier: float = 1.0,
    ) -> "Trial":
        """
        Evolve the trial with investment level modifiers.

        Parameters
        ----------
        speed_modifier : float
            Multiplier for progress speed (>1 = faster, <1 = slower).
        success_modifier : float
            Multiplier for success probability from investment level.
        global_success_modifier : float
            Additional success modifier from capacity overage (default 1.0).

        Returns
        -------
        Trial
            The evolved trial.

        """
        logger.debug(
            f"Evolving Trial with level: {self.phase}, "
            f"speed={speed_modifier}, success={success_modifier}"
        )

        # Calculate new progress
        new_progress = self.progress_accumulated + speed_modifier

        # Check if we've completed enough progress for one time unit
        if new_progress < 1.0:
            # Not enough progress to complete a time unit
            # Calculate proportional cost based on progress made this step
            cost_spent = self.cost_this_step * speed_modifier
            new_cost = max(0.0, self.cost_remaining - cost_spent)

            new_trial = Trial(
                cost_remaining=new_cost,
                time_remaining=self.time_remaining,
                ptrs=self.ptrs,
                phase=self.phase,
                state=TrialState.IN_PROGRESS,
                next_trial_on_success=self.next_trial_on_success,
                progress_accumulated=new_progress,
            )
            self._copy_private_attrs(new_trial)
            return new_trial

        # Progress >= 1.0, we complete a time unit
        # Reset progress, keeping any overflow
        remaining_progress = new_progress - 1.0

        if self.time_remaining > 1:
            # Still more time remaining after this unit
            cost_spent = self.cost_this_step
            new_cost = max(0.0, self.cost_remaining - cost_spent)

            new_trial = Trial(
                cost_remaining=new_cost,
                time_remaining=self.time_remaining - 1,
                ptrs=self.ptrs,
                phase=self.phase,
                state=TrialState.IN_PROGRESS,
                next_trial_on_success=self.next_trial_on_success,
                progress_accumulated=remaining_progress,
            )
            self._copy_private_attrs(new_trial)
            return new_trial

        # time_remaining == 1 and progress completed, roll for success
        # Apply both investment level and global (capacity) success modifiers
        combined_success_modifier = success_modifier * global_success_modifier

        if self._success_with_modifier(combined_success_modifier):
            logger.debug("Trial phase successful (with level modifiers)")
            if self.next_trial_on_success is not None:
                logger.debug("Returning next trial on success.")
                return self.next_trial_on_success
            else:
                logger.debug("All phases successful.")
                return Trial(
                    cost_remaining=0.0,
                    time_remaining=0,
                    ptrs=1.0,
                    phase=self.phase,
                    state=TrialState.PHASE_SUCCESS,
                    next_trial_on_success=None,
                    progress_accumulated=0.0,
                )

        logger.debug("Phase failed (with level modifiers).")
        return Trial(
            cost_remaining=0.0,
            time_remaining=0,
            ptrs=0.0,
            phase=self.phase,
            state=TrialState.PHASE_FAILED,
            next_trial_on_success=None,
            progress_accumulated=0.0,
        )

    def _success_with_modifier(self, success_modifier: float) -> bool:
        """
        Determine if trial succeeds with a success probability modifier.

        Parameters
        ----------
        success_modifier : float
            Multiplier applied to success probability.

        Returns
        -------
        bool
            True if trial succeeds, False otherwise.

        """
        rng = get_game_rng()

        # Distributional PTRS: sample from true Beta distribution, apply modifier
        if self._ptrs_true_alpha is not None and self._ptrs_true_beta is not None:
            np_rng = np.random.RandomState(rng.randint(0, 2**31))
            sampled_ptrs = float(
                np_rng.beta(self._ptrs_true_alpha, self._ptrs_true_beta)
            )
            modified_prob = min(1.0, sampled_ptrs * success_modifier)
            return rng.random() < modified_prob

        # Point estimate: true value when hidden, else observed
        base_prob = self._outcome_ptrs

        # Apply modifier but cap at 1.0
        modified_prob = min(1.0, base_prob * success_modifier)

        return rng.random() < modified_prob


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
