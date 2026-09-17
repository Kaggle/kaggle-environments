import copy
import logging
import random
import uuid
import warnings
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Literal, Optional

import numpy as np
import upath
from scipy.stats import beta as beta_dist

from pyxis_portfolio_challenge.config import (
    ApprovalPhaseConfig,
    DistributionalPtrsConfig,
    PtrsReadingsConfig,
    TAExperienceConfig,
    UncertainPtrsConfig,
)
from pyxis_portfolio_challenge.file_io import list_files, load_json_bulk
from pyxis_portfolio_challenge.game.asset import AssetState, DrugAsset
from pyxis_portfolio_challenge.game.trial import (
    Trial,
    TrialPhase,
    TrialState,
    trials_json_to_trials_sequence,
)
from pyxis_portfolio_challenge.rng import get_game_rng

logger = logging.getLogger(__name__)

# This is a namespace object required to generate reproducible UUIDs for each asset
ASSET_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "investment_game_assets")


def get_current_observed_ptrs(
    trial: Trial,
    ta_experience: float,
    experience_to_full_knowledge: float,
) -> float:
    """
    Get current observed PTRS based on TA experience.

    Uses interpolation: observed = (1-α) × initial_observed + α × true
    where α increases with experience.

    Args:
        trial: The trial to get observed PTRS for
        ta_experience: Current TA experience
        experience_to_full_knowledge: Experience needed for α=1

    Returns:
        Current observed PTRS (interpolated toward true PTRS)

    """
    if trial._true_ptrs is None or trial._initial_observed_ptrs is None:
        # Uncertain PTRS not enabled, return current ptrs
        return trial.ptrs

    # α = 0 at no experience, approaches 1 at high experience
    alpha = min(1.0, ta_experience / experience_to_full_knowledge)

    return (1 - alpha) * trial._initial_observed_ptrs + alpha * trial._true_ptrs


def get_effective_true_ptrs(
    trial: Trial,
    ta_experience: float,
    max_expertise_boost: float,
    experience_to_max_boost: float,
) -> float:
    """
    Get effective true PTRS with expertise boost.

    effective_true_ptrs = true_ptrs + boost

    Args:
        trial: The trial to get effective PTRS for
        ta_experience: Current TA experience
        max_expertise_boost: Maximum additive boost
        experience_to_max_boost: Experience needed for max boost

    Returns:
        Effective true PTRS (with expertise boost)

    """
    if trial._true_ptrs is None:
        # Uncertain PTRS not enabled, return current ptrs
        return trial.ptrs

    boost = max_expertise_boost * min(1.0, ta_experience / experience_to_max_boost)
    return min(1.0, trial._true_ptrs + boost)


def update_trial_chain_ptrs_for_experience(
    head_trial: Trial,
    therapeutic_area: str,
    ta_experience: dict[str, float],
    ta_experience_config: "TAExperienceConfig",
) -> None:
    """
    Update trial chain PTRS values based on current TA experience.

    For each trial in the chain:
    - Updates ptrs (observed) based on experience (interpolation toward true)
    - Sets _effective_true_ptrs for trial outcome evaluation

    Args:
        head_trial: The first trial in the chain
        therapeutic_area: The asset's therapeutic area
        ta_experience: Current TA experience dict
        ta_experience_config: Configuration for TA experience system

    """
    experience = ta_experience.get(therapeutic_area, 0.0)
    alpha = min(1.0, experience / ta_experience_config.experience_to_full_knowledge)

    current_trial = head_trial
    trials_updated = 0
    total_boost = 0.0
    total_convergence = 0.0

    while current_trial is not None:
        if current_trial._true_ptrs is not None:
            old_observed = current_trial.ptrs

            # Update observed PTRS (what agent sees)
            current_trial.ptrs = get_current_observed_ptrs(
                trial=current_trial,
                ta_experience=experience,
                experience_to_full_knowledge=ta_experience_config.experience_to_full_knowledge,
            )

            # Set effective true PTRS (for trial outcome)
            current_trial._effective_true_ptrs = get_effective_true_ptrs(
                trial=current_trial,
                ta_experience=experience,
                max_expertise_boost=ta_experience_config.max_expertise_boost,
                experience_to_max_boost=ta_experience_config.experience_to_max_boost,
            )

            trials_updated += 1
            boost = current_trial._effective_true_ptrs - current_trial._true_ptrs
            total_boost += boost
            convergence = abs(current_trial.ptrs - current_trial._true_ptrs)
            total_convergence += convergence

            logger.debug(
                f"[UncertainPTRS] Updated {current_trial.phase.value}: "
                f"observed={old_observed:.3f}->{current_trial.ptrs:.3f}, "
                f"effective_true={current_trial._effective_true_ptrs:.3f} "
                f"(boost={boost:.4f})"
            )

        current_trial = current_trial.next_trial_on_success

    if trials_updated > 0:
        avg_boost = total_boost / trials_updated
        avg_error = total_convergence / trials_updated
        logger.debug(
            f"[UncertainPTRS] PTRS update for TA={therapeutic_area}: "
            f"exp={experience:.1f}, α={alpha:.2f}, "
            f"avg_boost={avg_boost:.4f}, avg_error={avg_error:.4f}"
        )


def apply_uncertain_ptrs_to_trial_chain(
    head_trial: Trial,
    therapeutic_area: str,
    uncertain_ptrs_config: UncertainPtrsConfig,
    rng: random.Random,
) -> None:
    """
    Apply uncertain PTRS noise to a trial chain.

    For each trial in the chain:
    - Store original PTRS as _true_ptrs
    - Add TA-specific and phase-specific noise to create observed PTRS
    - Store initial observed PTRS for interpolation

    This modifies the trial objects in place.

    Args:
        head_trial: The first trial in the chain
        therapeutic_area: The asset's therapeutic area
        uncertain_ptrs_config: Configuration for uncertain PTRS
        rng: Random number generator for reproducibility

    """
    # Get TA-specific base noise (will raise KeyError if TA not configured)
    ta_noise = uncertain_ptrs_config.ta_noise_config[therapeutic_area]

    # Phase noise multipliers
    phase_key_map = {
        TrialPhase.PHASE_1: "phase_1",
        TrialPhase.PHASE_2: "phase_2",
        TrialPhase.PHASE_3: "phase_3",
    }
    if hasattr(TrialPhase, "APPROVAL"):
        phase_key_map[TrialPhase.APPROVAL] = "approval"

    logger.info(
        f"[UncertainPTRS] Applying noise to trial chain: TA={therapeutic_area}, "
        f"base_noise={ta_noise:.3f}"
    )

    trials_modified = 0
    total_noise_applied = 0.0

    current_trial = head_trial
    while current_trial is not None:
        # Skip if trial is already completed (PHASE_SUCCESS state)
        if current_trial.state == TrialState.PHASE_SUCCESS:
            current_trial = current_trial.next_trial_on_success
            continue

        # Get phase multiplier
        phase_key = phase_key_map.get(current_trial.phase)
        if phase_key is None:
            current_trial = current_trial.next_trial_on_success
            continue
        phase_mult = uncertain_ptrs_config.phase_noise_multipliers[phase_key]

        # Calculate noise std
        noise_std = ta_noise * phase_mult

        # Store original PTRS as true PTRS
        true_ptrs = current_trial.ptrs

        # Add noise to create observed PTRS (using rng for reproducibility)
        noise = rng.gauss(0, noise_std)
        observed_ptrs = np.clip(true_ptrs + noise, 0.05, 0.95)

        # Set the private attributes
        current_trial._true_ptrs = true_ptrs
        current_trial._initial_observed_ptrs = float(observed_ptrs)
        current_trial.ptrs = float(observed_ptrs)  # Agent sees this

        trials_modified += 1
        total_noise_applied += abs(float(observed_ptrs) - true_ptrs)

        logger.debug(
            f"[UncertainPTRS]   {current_trial.phase.value}: "
            f"true={true_ptrs:.3f} -> observed={observed_ptrs:.3f} "
            f"(noise_std={noise_std:.3f}, actual_noise={noise:+.3f})"
        )

        current_trial = current_trial.next_trial_on_success

    if trials_modified > 0:
        avg_noise = total_noise_applied / trials_modified
        logger.info(
            f"[UncertainPTRS] Applied noise to {trials_modified} trials, "
            f"avg |noise|={avg_noise:.4f}"
        )


def apply_ptrs_readings_to_trial_chain(
    asset: "DrugAsset",
    ptrs_readings_config: PtrsReadingsConfig,
    rng: random.Random,
) -> None:
    """
    Initialise the ptrs_readings feature for a newly arrived asset.

    Sets _true_ptrs on each non-approval pending trial, draws one initial
    logit-normal sample per trial (using phase-distance noise), and sets
    trial.ptrs = ptrs_sample_mean so the agent's first observation is noisy.
    """
    chain = asset.pending_trial_chain  # excludes APPROVAL
    cfg = ptrs_readings_config

    # Store true PTRS on every pending trial in the chain
    for trial in chain:
        trial._true_ptrs = trial.ptrs

    # Draw initial reading (count=1) for each trial at the appropriate noise level
    asset.initialise_ptrs_readings(
        cfg.sigma_logit_base, list(cfg.noise_multipliers), rng
    )

    # Update the observed PTRS to the noisy first reading
    for trial in chain:
        if trial.ptrs_sample_mean is not None:
            trial.ptrs = trial.ptrs_sample_mean


def sample_ta_by_experience(
    ta_experience: dict[str, float],
    temperature: float,
    rng: random.Random,
) -> str | None:
    """
    Sample a therapeutic area biased by experience using softmax.

    Args:
        ta_experience: Dict mapping TA names to experience values
        temperature: Controls bias strength (0=uniform, higher=more biased)
        rng: Random number generator for reproducibility

    Returns:
        Sampled TA name, or None if temperature is 0 (uniform random)

    """
    if temperature <= 0:
        logger.debug("[UncertainPTRS] TA bias disabled (temperature=0)")
        return None  # No bias, caller should use uniform random

    tas = list(ta_experience.keys())
    if not tas:
        return None

    # Softmax with temperature
    exp_values = [np.exp(temperature * ta_experience.get(ta, 0.0)) for ta in tas]
    total = sum(exp_values)
    probs = [v / total for v in exp_values]

    # Calculate cumulative probability intervals for each TA
    cumsum = 0.0
    intervals = []
    for ta, prob in zip(tas, probs):
        interval_start = cumsum
        cumsum += prob
        interval_end = cumsum
        intervals.append((ta, prob, interval_start, interval_end))

    # Log the probability intervals for each TA
    logger.info("[TA Sampling] Probability intervals for new asset arrival:")
    for ta, prob, start, end in intervals:
        exp_val = ta_experience.get(ta, 0.0)
        logger.info(
            f"  {ta}: prob={prob:.4f}, interval=[{start:.4f}, {end:.4f}), "
            f"experience={exp_val:.2f}"
        )

    # Sample using the RNG
    r = rng.random()
    logger.info(f"[TA Sampling] Random draw: {r:.6f}")

    # Determine which TA was selected based on the random draw
    selected_ta = tas[-1]  # Default to last TA
    for ta, prob, start, end in intervals:
        if r < end:
            selected_ta = ta
            logger.info(
                f"[TA Sampling] Selected TA: {ta} "
                f"(draw {r:.6f} fell in interval [{start:.4f}, {end:.4f}))"
            )
            break

    return selected_ta


def apply_distributional_ptrs_to_trial_chain(
    head_trial: Trial,
    therapeutic_area: str,
    ta_quality_modifier: float,
    distributional_ptrs_config: DistributionalPtrsConfig,
    rng: random.Random,
) -> None:
    """
    Apply distributional PTRS to a trial chain.

    In distributional PTRS, the PTRS itself is a distribution.
    We maintain two Beta distributions:
    - Prior (observed by agent): centered on base_ptrs from asset data
    - True (for outcomes): centered on base_ptrs + ta_modifier + asset_noise

    Trial outcomes sample from the TRUE distribution. The agent only sees the prior.

    This modifies the trial objects in place.

    Args:
        head_trial: The first trial in the chain
        therapeutic_area: The asset's therapeutic area
        ta_quality_modifier: Hidden TA quality modifier (sampled at episode start)
        distributional_ptrs_config: Configuration for distributional PTRS
        rng: Random number generator for reproducibility

    """
    asset_noise_std = distributional_ptrs_config.asset_noise_std
    prior_concentration = distributional_ptrs_config.prior_concentration

    logger.info(
        f"[DistributionalPTRS] Applying to trial chain: TA={therapeutic_area}, "
        f"ta_modifier={ta_quality_modifier:.4f}, asset_noise_std={asset_noise_std:.3f}"
    )

    trials_modified = 0

    current_trial = head_trial
    while current_trial is not None:
        # Skip if trial is already completed
        if current_trial.state == TrialState.PHASE_SUCCESS:
            current_trial = current_trial.next_trial_on_success
            continue

        # Get base PTRS from asset data
        base_ptrs = current_trial.ptrs

        # Sample per-asset noise
        asset_noise = rng.gauss(0, asset_noise_std)

        # Compute shifted PTRS mean for TRUE distribution (hidden from agent)
        true_ptrs_mean = float(
            np.clip(base_ptrs + ta_quality_modifier + asset_noise, 0.05, 0.95)
        )

        # Compute Beta PRIOR parameters (what agent observes)
        # Prior is centered on base_ptrs - agent doesn't know TA modifier
        prior_alpha = base_ptrs * prior_concentration
        prior_beta = (1 - base_ptrs) * prior_concentration

        # Compute Beta TRUE parameters (for sampling outcomes)
        # True distribution is centered on shifted mean
        true_alpha = true_ptrs_mean * prior_concentration
        true_beta = (1 - true_ptrs_mean) * prior_concentration

        # Ensure valid Beta parameters
        prior_alpha = max(0.1, prior_alpha)
        prior_beta = max(0.1, prior_beta)
        true_alpha = max(0.1, true_alpha)
        true_beta = max(0.1, true_beta)

        # Store both distributions
        current_trial._ptrs_prior_alpha = prior_alpha
        current_trial._ptrs_prior_beta = prior_beta
        current_trial._ptrs_true_alpha = true_alpha
        current_trial._ptrs_true_beta = true_beta

        # Cache percentiles (ppf is expensive, called on every step for obs)
        current_trial._ptrs_range_low_cached = float(
            beta_dist.ppf(0.10, prior_alpha, prior_beta)
        )
        current_trial._ptrs_range_high_cached = float(
            beta_dist.ppf(0.90, prior_alpha, prior_beta)
        )

        # The observed PTRS is the prior mean (same as base_ptrs)
        current_trial.ptrs = base_ptrs

        trials_modified += 1

        logger.debug(
            f"[DistributionalPTRS]   {current_trial.phase.value}: "
            f"base={base_ptrs:.3f}, true_mean={true_ptrs_mean:.3f}, "
            f"prior=Beta({prior_alpha:.2f}, {prior_beta:.2f}), "
            f"true=Beta({true_alpha:.2f}, {true_beta:.2f})"
        )

        current_trial = current_trial.next_trial_on_success

    if trials_modified > 0:
        logger.info(f"[DistributionalPTRS] Applied to {trials_modified} trials")


def update_distributional_ptrs_for_experience(
    head_trial: Trial,
    therapeutic_area: str,
    ta_experience: dict[str, float],
    base_concentration: float,
    experience_to_full_knowledge: float,
    max_concentration_multiplier: float = 4.0,
    exp_fraction_threshold: float = 0.01,
) -> None:
    """
    Update distributional PTRS based on TA experience.

    As experience increases:
    1. The prior's CENTER shifts toward the true distribution (revealing TA modifier)
    2. The prior's CONCENTRATION increases (tighter bounds, more confidence)

    At full knowledge, the agent's prior matches the true distribution.

    Args:
        head_trial: The first trial in the chain
        therapeutic_area: The asset's therapeutic area
        ta_experience: Dict mapping TA names to experience values
        base_concentration: Base concentration from config
        experience_to_full_knowledge: Experience needed for full knowledge
        max_concentration_multiplier: Maximum multiplier at full knowledge
        exp_fraction_threshold: Minimum exp_fraction change to trigger update

    """
    experience = ta_experience.get(therapeutic_area, 0.0)

    # Calculate experience fraction (0 to 1)
    # At 0 experience: alpha = 0
    # At full knowledge: alpha = 1
    exp_fraction = min(1.0, experience / experience_to_full_knowledge)

    # Calculate concentration multiplier
    concentration_multiplier = 1.0 + (max_concentration_multiplier - 1.0) * exp_fraction
    effective_concentration = base_concentration * concentration_multiplier

    current_trial = head_trial
    trials_updated = 0

    while current_trial is not None:
        # Skip completed trials
        if current_trial.state == TrialState.PHASE_SUCCESS:
            current_trial = current_trial.next_trial_on_success
            continue

        # Only update if distributional PTRS was applied
        if (
            current_trial._ptrs_prior_alpha is None
            or current_trial._ptrs_true_alpha is None
        ):
            current_trial = current_trial.next_trial_on_success
            continue

        # Skip update if experience fraction hasn't changed enough
        if (
            current_trial._last_exp_fraction is not None
            and abs(exp_fraction - current_trial._last_exp_fraction)
            < exp_fraction_threshold
        ):
            current_trial = current_trial.next_trial_on_success
            continue

        # Get base PTRS (initial prior mean, before any experience)
        base_ptrs = current_trial._initial_observed_ptrs
        if base_ptrs is None:
            base_ptrs = current_trial.ptrs

        # Get true PTRS mean (includes hidden TA modifier)
        true_mean = current_trial._ptrs_true_alpha / (
            current_trial._ptrs_true_alpha + current_trial._ptrs_true_beta
        )

        # Interpolate prior mean from base toward true based on experience
        # At 0 experience: prior_mean = base_ptrs
        # At full knowledge: prior_mean = true_mean
        effective_mean = base_ptrs + exp_fraction * (true_mean - base_ptrs)

        # Update alpha and beta with shifted mean and increased concentration
        old_alpha = current_trial._ptrs_prior_alpha
        old_beta = current_trial._ptrs_prior_beta
        old_mean = (
            old_alpha / (old_alpha + old_beta) if (old_alpha + old_beta) > 0 else 0
        )

        new_alpha = max(0.1, effective_mean * effective_concentration)
        new_beta = max(0.1, (1 - effective_mean) * effective_concentration)

        current_trial._ptrs_prior_alpha = new_alpha
        current_trial._ptrs_prior_beta = new_beta
        current_trial._last_exp_fraction = exp_fraction

        # Update cached percentiles
        current_trial._ptrs_range_low_cached = float(
            beta_dist.ppf(0.10, new_alpha, new_beta)
        )
        current_trial._ptrs_range_high_cached = float(
            beta_dist.ppf(0.90, new_alpha, new_beta)
        )

        trials_updated += 1

        logger.debug(
            f"[DistributionalPTRS] Updated {current_trial.phase.value}: "
            f"exp_frac={exp_fraction:.2f}, mean {old_mean:.3f}->{effective_mean:.3f} "
            f"(true={true_mean:.3f}), conc={effective_concentration:.1f}"
        )

        current_trial = current_trial.next_trial_on_success

    if trials_updated > 0:
        logger.debug(
            f"[DistributionalPTRS] Updated {trials_updated} trials "
            f"for TA={therapeutic_area}, exp={experience:.2f}, "
            f"conc_mult={concentration_multiplier:.2f}"
        )


DUMMY_LIST_DATA = [
    {
        "name": "Asset 1",
        "therapeutic_area": "oncology",
        "type": "internal",
        "description": "Description for Asset 1",
        "max_revenue": 1000000,
        "time_until_max_revenue": 5,
        "time_until_patent_expiry": 10,
        "trials": {
            # "Pre-clinical": {"cost_remaining": 0, "ptrs": 0.8, "time_remaining": 0},
            "phase_1": {"cost_remaining": 200000, "ptrs": 0.7, "time_remaining": 4},
            "phase_2": {"cost_remaining": 300000, "ptrs": 0.6, "time_remaining": 3},
            "phase_3": {"cost_remaining": 400000, "ptrs": 0.5, "time_remaining": 2},
            # "Registration": {
            #     "cost_remaining": 500000,
            #     "ptrs": 0.4,
            #     "time_remaining": 1,
            # },
        },
        "state": AssetState.Idle,
        "pending_trial_phase": "Phase 1",
        "time_on_market": 0,
    },
    {
        "name": "Asset 2",
        "therapeutic_area": "respiratory and immunology",
        "type": "BD",
        "description": "Description for Asset 2",
        "max_revenue": 2000000,
        "time_until_max_revenue": 6,
        "time_until_patent_expiry": 11,
        "trials": {
            # "Pre-clinical": {
            #     "cost_remaining": 150000,
            #     "ptrs": 0.75,
            #     "time_remaining": 5,
            # },
            "phase_1": {"cost_remaining": 250000, "ptrs": 0.65, "time_remaining": 4},
            "phase_2": {"cost_remaining": 350000, "ptrs": 0.55, "time_remaining": 3},
            "phase_3": {"cost_remaining": 450000, "ptrs": 0.45, "time_remaining": 2},
            # "Registration": {
            #     "cost_remaining": 550000,
            #     "ptrs": 0.35,
            #     "time_remaining": 1,
            # },
        },
        "state": AssetState.Idle,
        "pending_trial_phase": "Phase 1",
        "time_on_market": 0,
    },
    {
        "name": "Asset 3",
        "therapeutic_area": "vaccines and infectious disease",
        "type": "internal",
        "description": "Description for Asset 3",
        "max_revenue": 1000000,
        "time_until_max_revenue": 5,
        "time_until_patent_expiry": 10,
        "trials": {
            # "Pre-clinical": {"cost_remaining": 0, "ptrs": 0.8, "time_remaining": 0},
            "phase_1": {"cost_remaining": 0, "ptrs": 0.0, "time_remaining": 0},
            "phase_2": {"cost_remaining": 0, "ptrs": 0.0, "time_remaining": 0},
            "phase_3": {"cost_remaining": 0, "ptrs": 0.0, "time_remaining": 0},
            # "Registration": {
            #     "cost_remaining": 0,
            #     "ptrs": 0.4,
            #     "time_remaining": 0,
            # },
        },
        "state": AssetState.OnMarket,
        "pending_trial_phase": None,
        "time_on_market": 3,
    },
]


def generate_asset_id(counter: int, generator_id: int = 0) -> uuid.UUID:
    """Generate a deterministic, episode-unique UUID from game seed and counter."""
    from pyxis_portfolio_challenge.rng import get_game_seed

    return uuid.uuid5(ASSET_NAMESPACE, f"{get_game_seed()}_{generator_id}_{counter}")


class AssetGeneratorBase(ABC):
    """Abstract base class for drug asset generators."""

    def __init__(self, asset_count: int = 0, generator_index: int = 0):
        """
        Initialise the asset generator with an asset count and generator index.

        Parameters
        ----------
        asset_count : int, optional
            Number of assets generated, used to generate unique IDs. Defaults to 0.
        generator_index : int, optional
            Stable integer identifier for this generator instance. Combined with
            asset_count to produce unique, reproducible asset UUIDs across all
            generators in a game. Defaults to 0.

        """
        super().__init__()

        if not isinstance(asset_count, int):
            raise TypeError(
                f"asset_count must be an integer, received {type(asset_count).__name__}"
            )

        self.asset_count = asset_count
        self.generator_index = generator_index

    @abstractmethod
    def __call__(
        self,
        num_assets: int,
        stage: Literal["initial", "new"],  # FUTURE: add "new_bd"
    ) -> dict[uuid.UUID, DrugAsset]:
        """
        Generate a dictionary of drug assets.

        If `stage` is "initial", it generates assets for the initial game setup.
        If `stage` is "new", it generates assets at the start of their develppment (e.g.
        Pre-clinical).

        Parameters
        ----------
        num_assets : int
            Number of assets to generate.
        stage : Literal["initial", "new"]
            Stage of asset generation, either "initial" or "new".

        Returns
        -------
        dict[uuid.UUID, DrugAsset]
            A dictionary mapping asset IDs to DrugAsset objects.

        """
        if not isinstance(num_assets, int):
            raise TypeError(
                f"num_assets must be an integer, received {type(num_assets).__name__}"
            )
        if num_assets < 1:
            raise ValueError(f"num_assets must be at least 1, received {num_assets}")
        if stage not in ["initial", "new"]:
            raise ValueError(
                f"stage must be one of ['initial', 'new'], received {stage}"
            )

        logger.debug(f"Generating {num_assets} `{stage}` assets")

        # Each DrugAsset should be instantiated with an RNG
        # rng = random.Random(f"{global_seed}_{asset_id}"})
        pass

    # TODO: Maybe add general method that iterates through the asset dict to add
    # the RNGs based on the asset ID and asset_count


@lru_cache(maxsize=4)
def _load_all_assets_cached(assets_dir: upath.UPath, stage: str) -> tuple:
    """Cached loader for assets. Returns tuple (immutable) for caching."""
    stage_path = assets_dir / stage
    asset_files = list_files(stage_path)
    asset_files = [upath.UPath(f) for f in asset_files]
    return tuple(load_json_bulk(asset_files))


class JSONAssetGenerator(AssetGeneratorBase):
    """Generate drug assets from pre-generated JSON files."""

    def __init__(
        self,
        assets_dir: upath.UPath,
        indication_spread: float,
        indication_drift_speed: float,
        trial_cost_multiplier: float,
        uncertain_ptrs_config: Optional[UncertainPtrsConfig] = None,
        distributional_ptrs_config: Optional[DistributionalPtrsConfig] = None,
        ta_quality_modifiers: Optional[dict[str, float]] = None,
        ta_experience_config: Optional[TAExperienceConfig] = None,
        indications_per_ta: Optional[dict[str, int]] = None,
        approval_phase_config: Optional[ApprovalPhaseConfig] = None,
        ptrs_readings_config: Optional[PtrsReadingsConfig] = None,
        generator_index: int = 0,
    ):
        """
        Initialise the JSON asset generator with an assets directory.

        Inherits the `asset_count` attribute from AssetGeneratorBase, which tracks the
        number of assets generated and is used to generate unique IDs for each asset.

        Parameters
        ----------
        assets_dir : str
            Directory containing pre-generated assets in JSON format, structured by
            stage.
            Expected structure:
            ```
            assets_dir/
                ├── initial/
                │   ├── asset_0.json
                │   ├── asset_1.json
                │   └── ...
                └── new/
                    ├── asset_0.json
                    ├── asset_1.json
                    └── ...
            ```
        indication_drift_speed : float
            Speed at which indication quality modifiers drift over time.
        trial_cost_multiplier : float
            Multiplier applied to trial costs when constructing drug assets.
        uncertain_ptrs_config : Optional[UncertainPtrsConfig]
            Configuration for uncertain PTRS feature. If None, feature disabled.
        distributional_ptrs_config : Optional[DistributionalPtrsConfig]
            Configuration for distributional PTRS feature. If None, feature disabled.
        ta_quality_modifiers : Optional[dict[str, float]]
            Hidden TA quality modifiers sampled at episode start. Required if
            distributional_ptrs_config is enabled.
        ta_experience_config : Optional[TAExperienceConfig]
            Configuration for TA experience system. If None, feature disabled.
        indications_per_ta : Optional[dict[str, int]]
            Number of indications per TA for random assignment. If None, all
            assets get indication=0.
        indication_spread : float
            Absolute Gaussian sigma in indication-index units. Controls the
            width of the hot zone around the current drift centre.
        approval_phase_config : Optional[ApprovalPhaseConfig]
            Configuration for approval phase. If enabled, an Approval trial is
            injected after Phase 3 in the trial chain.
        ptrs_readings_config : Optional[PtrsReadingsConfig]
            Configuration for the PTRS readings feature. If None, feature disabled.
        generator_index : int
            Index used to disambiguate UUIDs when multiple generators run in parallel.

        """
        super().__init__(generator_index=generator_index)

        if not isinstance(assets_dir, upath.UPath):
            raise TypeError(
                f"assets_dir must be a upath.UPath, received "
                f"{type(assets_dir).__name__}"
            )

        self.assets_dir = assets_dir
        self.uncertain_ptrs_config = uncertain_ptrs_config
        self.distributional_ptrs_config = distributional_ptrs_config
        self.ta_quality_modifiers = ta_quality_modifiers or {}
        self.ta_experience_config = ta_experience_config
        self.indications_per_ta = indications_per_ta
        self.indication_spread = indication_spread
        self.indication_drift_speed = indication_drift_speed
        self.trial_cost_multiplier = trial_cost_multiplier
        self.approval_phase_config = approval_phase_config
        self.ptrs_readings_config = ptrs_readings_config
        # Random permutation per TA: maps drift-order → observed index.
        # Set via set_indication_permutation(); None = identity mapping.
        self._indication_permutation: Optional[dict[str, list[int]]] = None

        self._all_assets = {}
        self._all_assets["initial"] = list(
            _load_all_assets_cached(assets_dir=assets_dir, stage="initial")
        )
        self._all_assets["new"] = list(
            _load_all_assets_cached(assets_dir=assets_dir, stage="new")
        )
        self.available_assets = {}
        self._reset_available_assets("initial")
        self._reset_available_assets("new")

    def _reset_available_assets(self, stage: Literal["initial", "new"]):
        """Reset the available assets for a given stage."""
        reset_assets = [asset.copy() for asset in self._all_assets[stage]]
        get_game_rng().shuffle(reset_assets)
        self.available_assets[stage] = reset_assets

    def _generate_single_asset(
        self,
        stage: Literal["initial", "new"],
        target_ta: Optional[str] = None,
    ) -> DrugAsset:
        """
        Function that returns one DrugAsset at a time for a given stage.

        Args:
            stage: "initial" or "new"
            target_ta: Optional TA to bias selection toward. If provided, will
                       prefer assets from this TA (with fallback to any asset).

        """
        if stage not in ["initial", "new"]:
            raise ValueError(
                f"stage must be one of ['initial', 'new'], received {stage}"
            )

        if not self.available_assets[stage]:
            self._reset_available_assets(stage=stage)

        asset_data = None

        # If target_ta specified, try to find an asset from that TA
        if target_ta is not None:
            # Look for matching TA in available assets
            for i, candidate in enumerate(self.available_assets[stage]):
                if candidate.get("therapeutic_area") == target_ta:
                    asset_data = self.available_assets[stage].pop(i)
                    logger.debug(
                        f"TA bias: selected asset from target TA '{target_ta}'"
                    )
                    break

            # If no match found, fall back to any asset
            if asset_data is None:
                logger.debug(
                    f"TA bias: no asset found for target TA '{target_ta}', "
                    f"using any available asset"
                )

        # Default: pop any asset
        if asset_data is None:
            asset_data = self.available_assets[stage].pop()

        self.asset_count += 1
        return self._asset_from_asset_data(asset_data)

    def set_indication_permutation(self, permutation: dict[str, list[int]]) -> None:
        """
        Set random permutation mapping drift-order → observed index.

        Called once per episode to prevent the RL agent from learning
        that specific indication integers correlate with episode timing.
        """
        self._indication_permutation = permutation

    def _sample_indication(
        self, num_indications: int, therapeutic_area: str = ""
    ) -> int:
        """
        Sample an indication index, biased by episode progress.

        The internal drift favours a centre that moves over the
        episode. A per-episode random permutation maps the drift
        order to observed indication integers so the agent cannot
        learn a fixed temporal pattern.
        """
        progress = getattr(self, "_current_episode_progress", None)
        if progress is None or num_indications < 2:
            raw = get_game_rng().randint(0, num_indications - 1)
        else:
            # Centre drifts through indications over the episode.
            # drift_speed controls how many full sweeps per episode.
            # Wraps around via modulo so values > 1.0 revisit indications.
            scaled_progress = (progress * self.indication_drift_speed) % 1.0
            center = scaled_progress * (num_indications - 1)

            # Gaussian weights (linear, no wrap-around)
            spread = self.indication_spread
            weights = []
            for i in range(num_indications):
                dist = abs(i - center)
                weights.append(np.exp(-0.5 * (dist / spread) ** 2))

            # Sample from weighted distribution
            total = sum(weights)
            r = get_game_rng().random() * total
            cumulative = 0.0
            raw = num_indications - 1
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    raw = i
                    break

        # Apply per-episode permutation
        if self._indication_permutation and therapeutic_area:
            perm = self._indication_permutation.get(therapeutic_area)
            if perm and raw < len(perm):
                return perm[raw]
        return raw

    def _asset_from_asset_data(self, asset_data: dict) -> DrugAsset:
        """Convert asset data dictionary to DrugAsset object with unique ID and RNG."""
        # Convert trials data from JSON schema to dict of Trial objects
        logger.debug(f"asset_data: {asset_data}")
        asset_id = generate_asset_id(self.asset_count, self.generator_index)
        therapeutic_area = asset_data["therapeutic_area"]

        if AssetState(asset_data["state"]) == AssetState.OnMarket:
            # Use APPROVAL as final phase if approval is enabled, else PHASE_3
            final_phase = (
                TrialPhase.APPROVAL
                if self.approval_phase_config is not None
                and self.approval_phase_config.enabled
                else TrialPhase.PHASE_3
            )
            trial = Trial(
                phase=final_phase,
                state=TrialState.PHASE_SUCCESS,
                cost_remaining=0.0,
                time_remaining=0,
                ptrs=1.0,
                next_trial_on_success=None,
            )
        else:
            trial = trials_json_to_trials_sequence(
                asset_data["trials"],
                asset_id=asset_id,
                pending_trial_phase=asset_data["pending_trial_phase"],
                approval_phase_config=self.approval_phase_config,
                trial_cost_multiplier=self.trial_cost_multiplier,
            )

            # Apply uncertain PTRS noise if enabled (legacy feature)
            if (
                self.uncertain_ptrs_config is not None
                and self.uncertain_ptrs_config.enabled
            ):
                apply_uncertain_ptrs_to_trial_chain(
                    head_trial=trial,
                    therapeutic_area=therapeutic_area,
                    uncertain_ptrs_config=self.uncertain_ptrs_config,
                    rng=get_game_rng(),
                )

            # Apply distributional PTRS if enabled (new feature)
            if (
                self.distributional_ptrs_config is not None
                and self.distributional_ptrs_config.enabled
            ):
                # Get TA quality modifier (sampled at episode start)
                ta_modifier = self.ta_quality_modifiers[therapeutic_area]
                apply_distributional_ptrs_to_trial_chain(
                    head_trial=trial,
                    therapeutic_area=therapeutic_area,
                    ta_quality_modifier=ta_modifier,
                    distributional_ptrs_config=self.distributional_ptrs_config,
                    rng=get_game_rng(),
                )

        # Assign indication based on TA with time-dependent drift
        indication = 0
        if self.indications_per_ta and therapeutic_area in self.indications_per_ta:
            num_ind = self.indications_per_ta[therapeutic_area]
            if num_ind > 1:
                indication = self._sample_indication(num_ind, therapeutic_area)

        asset = DrugAsset(
            id=asset_id,
            name=asset_data["name"],
            therapeutic_area=therapeutic_area,
            indication=indication,
            type=asset_data["type"],
            description=asset_data["description"],
            max_revenue=asset_data["max_revenue"],
            raw_max_revenue=asset_data["max_revenue"],
            time_until_max_revenue=asset_data["time_until_max_revenue"],
            time_until_patent_expiry=asset_data["time_until_patent_expiry"],
            state=AssetState(asset_data["state"]),
            time_on_market=asset_data["time_on_market"],
            trial=trial,
        )

        if (
            self.ptrs_readings_config is not None
            and self.ptrs_readings_config.enabled
            and isinstance(self.ptrs_readings_config, PtrsReadingsConfig)
        ):
            apply_ptrs_readings_to_trial_chain(
                asset=asset,
                ptrs_readings_config=self.ptrs_readings_config,
                rng=get_game_rng(),
            )

        logger.debug(f"Created asset: {asset}")
        return asset

    def __call__(
        self,
        num_assets: int,
        stage: Literal["initial", "new"],
        ta_experience: Optional[dict[str, float]] = None,
        episode_progress: Optional[float] = None,
    ) -> dict[uuid.UUID, DrugAsset]:
        """
        Generate a dictionary of drug assets from pre-generated JSONs.

        Parameters
        ----------
        num_assets : int
            Number of assets to generate.
        stage : Literal["initial", "new"]
            Stage of asset generation, either "initial" or "new".
        ta_experience : Optional[dict[str, float]]
            Optional TA experience dict for biasing asset arrival.
            Only used when stage="new" and uncertain_ptrs_config is set.
        episode_progress : Optional[float]
            Fraction of episode elapsed (0.0 to 1.0). Used for
            time-dependent indication assignment.

        Returns
        -------
        dict[uuid.UUID, DrugAsset]
            A dictionary mapping asset IDs to DrugAsset objects.

        """
        super().__call__(num_assets, stage)
        self._current_episode_progress = episode_progress

        if num_assets > len(self._all_assets[stage]):
            raise ValueError(
                f"num_assets ({num_assets}) cannot be greater than "
                f"max number ({len(self._all_assets[stage])}) for stage '{stage}'"
            )

        if num_assets > len(self._all_assets[stage]):
            warnings.warn(
                f"num_assets {num_assets} is greater than number of"
                f" available assets. Assets will be reused."
            )

        assets = {}

        for _ in range(num_assets):
            target_ta = None

            # Apply TA bias for new asset arrivals if TA experience config is enabled
            if (
                stage == "new"
                and ta_experience is not None
                and self.ta_experience_config is not None
                and self.ta_experience_config.enabled
                and self.ta_experience_config.asset_arrival_temperature > 0
            ):
                target_ta = sample_ta_by_experience(
                    ta_experience=ta_experience,
                    temperature=self.ta_experience_config.asset_arrival_temperature,
                    rng=get_game_rng(),
                )
                if target_ta:
                    logger.debug(f"TA bias: sampled target TA '{target_ta}'")

            asset = self._generate_single_asset(stage=stage, target_ta=target_ta)
            assets[asset.id] = asset

        return assets

    def generate_bd_asset(
        self,
        therapeutic_area: str,
        indication: int,
        target_phase: int,
    ) -> DrugAsset:
        """
        Generate a BD asset for a specific TA, indication, and target phase.

        Picks a random asset template from the "initial" pool matching the TA,
        overrides the indication, and adjusts the trial chain so the asset
        is pending the target phase.

        Parameters
        ----------
        therapeutic_area : str
            Target therapeutic area.
        indication : int
            Target indication index.
        target_phase : int
            Target pending phase (0=Phase 1, 1=Phase 2, 2=Phase 3).

        """
        phase_map = {0: "Phase 1", 1: "Phase 2", 2: "Phase 3"}
        pending_phase = phase_map[target_phase]

        # Use "new" assets — they always have all phases populated
        candidates = [
            a
            for a in self._all_assets["new"]
            if a.get("therapeutic_area") == therapeutic_area
        ]
        if not candidates:
            candidates = self._all_assets["new"]

        asset_data = copy.deepcopy(get_game_rng().choice(candidates))
        asset_data["pending_trial_phase"] = pending_phase
        asset_data["state"] = "Idle"

        self.asset_count += 1
        asset_id = generate_asset_id(self.asset_count, self.generator_index)

        trial = trials_json_to_trials_sequence(
            asset_data["trials"],
            asset_id=asset_id,
            pending_trial_phase=pending_phase,
            approval_phase_config=self.approval_phase_config,
            trial_cost_multiplier=self.trial_cost_multiplier,
        )

        asset = DrugAsset(
            id=asset_id,
            name=f"BD-{asset_data['name']}",
            therapeutic_area=therapeutic_area,
            indication=indication,
            type="BD",
            description=asset_data["description"],
            max_revenue=asset_data["max_revenue"],
            raw_max_revenue=asset_data["max_revenue"],
            time_until_max_revenue=asset_data["time_until_max_revenue"],
            time_until_patent_expiry=asset_data["time_until_patent_expiry"],
            state=AssetState.Idle,
            time_on_market=0,
            trial=trial,
        )

        if self.ptrs_readings_config is not None and self.ptrs_readings_config.enabled:
            apply_ptrs_readings_to_trial_chain(
                asset=asset,
                ptrs_readings_config=self.ptrs_readings_config,
                rng=get_game_rng(),
            )

        return asset


class FixedListAssetGenerator(AssetGeneratorBase):
    """Generate drug assets from a fixed list of asset data dictionaries."""

    def __init__(
        self,
        assets_data_list: Optional[list[dict]] = DUMMY_LIST_DATA,
        uncertain_ptrs_config: Optional[UncertainPtrsConfig] = None,
        distributional_ptrs_config: Optional[DistributionalPtrsConfig] = None,
        ta_quality_modifiers: Optional[dict[str, float]] = None,
        ta_experience_config: Optional[TAExperienceConfig] = None,
        indications_per_ta: Optional[dict[str, int]] = None,
        approval_phase_config: Optional[ApprovalPhaseConfig] = None,
        trial_cost_multiplier: float = 1.0,
        ptrs_readings_config: Optional[PtrsReadingsConfig] = None,
        generator_index: int = 0,
    ):
        """
        Initialise the fixed list asset generator.

        Inherits the `asset_count` attribute from AssetGeneratorBase, which tracks the
        number of assets generated and is used to generate unique IDs for each asset.

        Parameters
        ----------
        assets_data_list : list[dict]
            A list of dictionaries containing asset data.
            Each dictionary should contain the fields required to create a DrugAsset.
        uncertain_ptrs_config : UncertainPtrsConfig, optional
            Configuration for uncertain PTRS feature.
        distributional_ptrs_config : DistributionalPtrsConfig, optional
            Configuration for distributional PTRS feature.
        ta_quality_modifiers : dict[str, float], optional
            Hidden TA quality modifiers sampled at episode start.
        ta_experience_config : TAExperienceConfig, optional
            Configuration for TA experience system.
        indications_per_ta : dict[str, int], optional
            Number of indications per TA for random assignment.
        approval_phase_config : ApprovalPhaseConfig, optional
            Configuration for approval phase.
        trial_cost_multiplier : float
            Multiplier for trial phase costs.
        ptrs_readings_config : PtrsReadingsConfig, optional
            Configuration for the PTRS readings feature.
        generator_index : int
            Index used to disambiguate UUIDs when multiple generators run in parallel.

        """
        super().__init__(generator_index=generator_index)

        if not isinstance(assets_data_list, list):
            raise TypeError(
                f"assets_data_list must be a list, "
                f"received {type(assets_data_list).__name__}"
            )
        if not all(isinstance(item, dict) for item in assets_data_list):
            raise TypeError("All items in assets_data_list must be dictionaries")

        self.assets_data_list = assets_data_list
        self.uncertain_ptrs_config = uncertain_ptrs_config
        self.distributional_ptrs_config = distributional_ptrs_config
        self.ta_quality_modifiers = ta_quality_modifiers or {}
        self.ta_experience_config = ta_experience_config
        self.indications_per_ta = indications_per_ta
        self.approval_phase_config = approval_phase_config
        self.trial_cost_multiplier = trial_cost_multiplier
        self.ptrs_readings_config = ptrs_readings_config
        self._indication_permutation: Optional[dict[str, list[int]]] = None

    def set_indication_permutation(self, permutation: dict[str, list[int]]) -> None:
        """Set random permutation mapping drift-order → observed index."""
        self._indication_permutation = permutation

    def __call__(
        self,
        num_assets: int,
        stage: Literal["initial", "new"],
        ta_experience: Optional[dict[str, float]] = None,
        episode_progress: Optional[float] = None,
    ) -> dict[uuid.UUID, DrugAsset]:
        """
        Generate a dictionary of drug assets from a fixed list.

        Parameters
        ----------
        num_assets : int
            Number of assets to generate.
        stage : Literal["initial", "new"]
            Stage of asset generation, either "initial" or "new".
            NOTE: CURRENTLY UNUSED.
        ta_experience : Optional[dict[str, float]]
            Optional TA experience dict for biasing asset arrival.
            Only used when stage="new" and uncertain_ptrs_config is set.
        episode_progress : Optional[float]
            Unused. Accepted for interface compatibility.

        Returns
        -------
        dict[uuid.UUID, DrugAsset]
            A dictionary mapping asset IDs to DrugAsset objects.

        """
        super().__call__(num_assets, stage)
        assets = {}
        for _ in range(num_assets):
            target_ta = None

            # Apply TA bias for new asset arrivals if TA experience config is enabled
            if (
                stage == "new"
                and ta_experience is not None
                and self.ta_experience_config is not None
                and self.ta_experience_config.enabled
                and self.ta_experience_config.asset_arrival_temperature > 0
            ):
                target_ta = sample_ta_by_experience(
                    ta_experience=ta_experience,
                    temperature=self.ta_experience_config.asset_arrival_temperature,
                    rng=get_game_rng(),
                )

            # Pick a random asset from assets_data_list and copy it
            # If target_ta specified, try to find a matching asset
            if target_ta is not None:
                matching_assets = [
                    a
                    for a in self.assets_data_list
                    if a.get("therapeutic_area") == target_ta
                ]
                if matching_assets:
                    asset_data = copy.deepcopy(get_game_rng().choice(matching_assets))
                else:
                    asset_data = copy.deepcopy(
                        get_game_rng().choice(self.assets_data_list)
                    )
            else:
                asset_data = copy.deepcopy(get_game_rng().choice(self.assets_data_list))
            # Add id and rng fields to asset_data
            self.asset_count += 1
            asset_id = generate_asset_id(self.asset_count, self.generator_index)
            if AssetState(asset_data["state"]) == AssetState.OnMarket:
                final_phase = (
                    TrialPhase.APPROVAL
                    if self.approval_phase_config is not None
                    and self.approval_phase_config.enabled
                    else TrialPhase.PHASE_3
                )
                trial = Trial(
                    phase=final_phase,
                    state=TrialState.PHASE_SUCCESS,
                    cost_remaining=0.0,
                    time_remaining=0,
                    ptrs=1.0,
                    next_trial_on_success=None,
                )
            else:
                therapeutic_area = asset_data["therapeutic_area"]
                trial = trials_json_to_trials_sequence(
                    asset_data["trials"],
                    asset_id=asset_id,
                    pending_trial_phase=asset_data["pending_trial_phase"],
                    approval_phase_config=self.approval_phase_config,
                    trial_cost_multiplier=self.trial_cost_multiplier,
                )

                # Apply uncertain PTRS noise if enabled (legacy feature)
                if (
                    self.uncertain_ptrs_config is not None
                    and self.uncertain_ptrs_config.enabled
                ):
                    apply_uncertain_ptrs_to_trial_chain(
                        head_trial=trial,
                        therapeutic_area=therapeutic_area,
                        uncertain_ptrs_config=self.uncertain_ptrs_config,
                        rng=get_game_rng(),
                    )

                # Apply distributional PTRS if enabled (new feature)
                if (
                    self.distributional_ptrs_config is not None
                    and self.distributional_ptrs_config.enabled
                ):
                    ta_modifier = self.ta_quality_modifiers[therapeutic_area]
                    apply_distributional_ptrs_to_trial_chain(
                        head_trial=trial,
                        therapeutic_area=therapeutic_area,
                        ta_quality_modifier=ta_modifier,
                        distributional_ptrs_config=self.distributional_ptrs_config,
                        rng=get_game_rng(),
                    )

            # Assign indication based on TA
            ta = asset_data["therapeutic_area"]
            indication = 0
            if self.indications_per_ta and ta in self.indications_per_ta:
                num_ind = self.indications_per_ta[ta]
                if num_ind > 1:
                    raw = get_game_rng().randint(0, num_ind - 1)
                    # Apply per-episode permutation
                    if self._indication_permutation:
                        perm = self._indication_permutation.get(ta)
                        if perm and raw < len(perm):
                            raw = perm[raw]
                    indication = raw

            asset = DrugAsset(
                id=asset_id,
                name=asset_data["name"],
                therapeutic_area=ta,
                indication=indication,
                type=asset_data["type"],
                description=asset_data["description"],
                max_revenue=asset_data["max_revenue"],
                raw_max_revenue=asset_data["max_revenue"],
                time_until_max_revenue=asset_data["time_until_max_revenue"],
                time_until_patent_expiry=asset_data["time_until_patent_expiry"],
                state=AssetState(asset_data["state"]),
                time_on_market=asset_data["time_on_market"],
                trial=trial,
            )

            if (
                self.ptrs_readings_config is not None
                and self.ptrs_readings_config.enabled
                and isinstance(self.ptrs_readings_config, PtrsReadingsConfig)
            ):
                apply_ptrs_readings_to_trial_chain(
                    asset=asset,
                    ptrs_readings_config=self.ptrs_readings_config,
                    rng=get_game_rng(),
                )

            assets[asset.id] = asset
        return assets
