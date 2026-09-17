from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import uuid
from enum import Enum
from typing import Literal, Optional, Self

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    field_validator,
    model_validator,
)
from scipy.stats import norm

from pyxis_portfolio_challenge.config import (
    ApprovalPhaseConfig,
    CapacityConfig,
    ClinicalSitesConfig,
    DistributionalPtrsConfig,
    DropActionConfig,
    InterimTrialObservationsConfig,
    InvestmentLevelsConfig,
    MarketingConfig,
    PtrsReadingsConfig,
    TAExperienceConfig,
    UncertainPtrsConfig,
    fibonacci_cost,
)
from pyxis_portfolio_challenge.game.asset import AssetState, DrugAsset
from pyxis_portfolio_challenge.game.asset_generators import (
    AssetGeneratorBase,
    update_distributional_ptrs_for_experience,
    update_trial_chain_ptrs_for_experience,
)
from pyxis_portfolio_challenge.game.clinical_sites import resolve_site_grants
from pyxis_portfolio_challenge.game.constants import MAX_NUM_ASSETS, InvestmentLevel
from pyxis_portfolio_challenge.game.trial import TrialPhase
from pyxis_portfolio_challenge.rng import get_game_rng, init_game_rng

logger = logging.getLogger(__name__)


def _compute_package_code_hash() -> str:
    """Hash all files in the package directory for version fingerprinting."""
    package_dir = pathlib.Path(__file__).parent.parent  # pyxis_portfolio_challenge/
    hasher = hashlib.sha256()
    for f in sorted(package_dir.rglob("*")):
        if f.is_file():
            hasher.update(f.read_bytes())
    return hasher.hexdigest()


_PACKAGE_CODE_HASH: str = _compute_package_code_hash()


class GameEndReason(str, Enum):
    """Enum containing reasons for game ending."""

    ONGOING_INVESTMENTS = "Ran out of cash due to ongoing investments."
    NEW_INVESTMENTS = "Ran out of cash due to new investments."
    RESEARCH_COSTS = "Ran out of cash due to research costs."
    PTRS_READINGS_COSTS = "Ran out of cash due to ptrs readings costs."
    HORIZON_REACHED = "Game ended as horizon was reached."


class GameState(BaseModel):
    """
    A class representing the state of the game at a specific point in time.

    Parameters
    ----------
    id : uuid.UUID
        Unique identifier for the game state.
    cash : float
        The amount of cash available to the player.
    time : int
        The current time step in the game.
    horizon : int
        The time horizon for the game.
    equilibrium_num_assets: int
        The equilibrium point for the mean-reverting random walk. Portfolio
        size fluctuates around this value.
    max_num_assets: int
        The maximum number of assets allowed in the game.
    asset_arrival_sensitivity_below: float
        Controls mean reversion speed when below equilibrium (lower = faster recovery).
    asset_arrival_sensitivity_above: float
        Controls fluctuation width at/above equilibrium (higher = wider).
    assets : dict[uuid.UUID, DrugAsset]
        A dictionary of DrugAsset objects representing the assets available in the game.
    failed_assets : dict[uuid.UUID, DrugAsset]
        A dictionary of DrugAsset objects that failed during clinical trials.
    expired_assets : dict[uuid.UUID, DrugAsset]
        A dictionary of expired DrugAsset objects that are no longer in the game.
    realised_costs : list[float]
        A list of costs already realised in previous time steps of the game.
    realised_revenues : list[float]
        A list of revenues already realised in previous time steps of the game.
    _asset_generator : AssetGeneratorBase
        The asset generator used to create new assets.
    The game-wide RNG is accessed via `get_game_rng()` from the `rng` module.

    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: uuid.UUID
    cash: float
    time: int
    horizon: int
    equilibrium_num_assets: int
    max_num_assets: int = MAX_NUM_ASSETS
    asset_arrival_sensitivity_below: float
    asset_arrival_sensitivity_above: float
    reinvestment_percentage: float
    initial_cash: float
    assets: dict[uuid.UUID, DrugAsset]
    failed_assets: dict[uuid.UUID, DrugAsset]
    expired_assets: dict[uuid.UUID, DrugAsset]
    dropped_assets: dict[uuid.UUID, DrugAsset]
    realised_costs: list[float]
    realised_revenues: list[float]
    running_enpv: list[float]
    running_eroi: list[float]
    game_ended: bool
    ended_reason: Optional[str]

    # Uncertain PTRS feature: TA experience tracking
    ta_experience: dict[str, float] = {}  # {therapeutic_area: weighted_experience}

    # Investment levels feature: capacity tracking
    capacity_used: float = 0.0  # Current capacity usage
    capacity_base: float = 80.0  # Base capacity (from config)

    # Clinical sites feature: per-agent concurrency capacity (Model B).
    # operational_sites host trials now; sites_in_development are build-delay
    # timers (steps remaining) for purchased-but-not-yet-usable sites. Occupied
    # sites are derived from the live InDevelopment asset count, so free sites =
    # operational_sites - (# InDevelopment assets).
    operational_sites: int = 0
    sites_in_development: list[int] = []

    # Distributional PTRS feature: TA quality estimates (observable posteriors)
    # These represent the agent's belief about TA quality based on trial outcomes
    ta_quality_estimates: dict[str, float] = {}  # Posterior mean estimate
    ta_quality_confidences: dict[str, float] = {}  # Posterior confidence (0-1)

    _asset_generator: AssetGeneratorBase = PrivateAttr()
    _ta_experience_config: Optional[object] = PrivateAttr(default=None)
    _uncertain_ptrs_config: Optional[object] = PrivateAttr(default=None)
    _investment_levels_config: Optional[object] = PrivateAttr(default=None)
    _interim_trial_observations_config: Optional[object] = PrivateAttr(default=None)
    _distributional_ptrs_config: Optional[object] = PrivateAttr(default=None)
    _rd_capacity_config: Optional[object] = PrivateAttr(default=None)
    _drop_action_config: Optional[object] = PrivateAttr(default=None)
    _ptrs_readings_config: Optional[object] = PrivateAttr(default=None)
    _clinical_sites_config: Optional[object] = PrivateAttr(default=None)
    # Hidden TA quality modifiers (sampled at episode start, not observable)
    _ta_quality_modifiers: dict[str, float] = PrivateAttr(default_factory=dict)
    # Marketing config (set once at game start)
    _marketing_config: Optional[object] = PrivateAttr(default=None)
    # Per-drug brand equity scores (asset_id -> score); accumulates with spend,
    # decays toward floor
    _brand_scores: dict[uuid.UUID, float] = PrivateAttr(default_factory=dict)
    # Per-drug permanent floor scores based on drug quality; decay never goes below this
    _brand_score_floors: dict[uuid.UUID, float] = PrivateAttr(default_factory=dict)
    # Per-agent deep-copies of shared BD assets, keyed by str(asset.id).
    # Each clone accumulates that agent's private PTRS readings.
    _bd_asset_clones: dict[str, object] = PrivateAttr(default_factory=dict)

    def _post_init_update_enpv_eroi(self) -> Self:
        """Safely updates running totals after the instance is fully created."""
        self.running_enpv.append(self.enpv())
        self.running_eroi.append(self.eroi())
        return self

    def rebase_time_to_zero(self) -> None:
        """
        Reset this game state's clock to 0 (warmup pre-roll boundary).

        ``time`` is the only absolute-time-stamped field on a GameState —
        assets carry relative durations only (``time_on_market``,
        ``time_until_*``), so nothing else needs shifting. Keeping this next
        to the field means any future absolute-time state is rebased here.
        """
        self.time = 0

    def _update_ptrs_for_experience(self) -> None:
        """
        Update PTRS values for all assets based on current TA experience.

        For uncertain_ptrs:
        - Updates observed PTRS via interpolation toward true PTRS
        - Sets effective true PTRS for trial outcome evaluation

        For distributional_ptrs:
        - Increases Beta concentration (tighter bounds) based on experience
        - Higher experience = higher confidence = smaller uncertainty range

        Called internally by step().
        """
        # Check if TA experience is enabled
        if self._ta_experience_config is None or not self._ta_experience_config.enabled:
            return

        # Log TA experience at start of step
        exp_str = ", ".join([
            f"{ta}={exp:.2f}" for ta, exp in sorted(self.ta_experience.items())
        ])

        # Handle uncertain_ptrs (point-based with noise convergence)
        if (
            self._uncertain_ptrs_config is not None
            and self._uncertain_ptrs_config.enabled
        ):
            logger.info(f"[UncertainPTRS] Step {self.time} TA Experience: {exp_str}")

            for asset in self.assets.values():
                update_trial_chain_ptrs_for_experience(
                    head_trial=asset.trial,
                    therapeutic_area=asset.therapeutic_area,
                    ta_experience=self.ta_experience,
                    ta_experience_config=self._ta_experience_config,
                )

                # Log per-asset PTRS details
                trial = asset.trial
                if trial is not None and trial._true_ptrs is not None:
                    ta_exp = self.ta_experience.get(asset.therapeutic_area, 0.0)
                    alpha = min(
                        1.0,
                        ta_exp
                        / self._ta_experience_config.experience_to_full_knowledge,
                    )
                    logger.info(
                        f"[UncertainPTRS] Asset '{asset.name}' "
                        f"({asset.therapeutic_area}): "
                        f"observed={trial.ptrs:.3f}, "
                        f"true={trial._true_ptrs:.3f}, "
                        f"effective={trial._effective_true_ptrs:.3f}, "
                        f"error={abs(trial.ptrs - trial._true_ptrs):.3f}, "
                        f"α={alpha:.2f}"
                    )

        # Handle distributional_ptrs (Beta concentration increases with experience)
        elif (
            self._distributional_ptrs_config is not None
            and self._distributional_ptrs_config.enabled
        ):
            logger.info(
                f"[DistributionalPTRS] Step {self.time} TA Experience: {exp_str}"
            )

            for asset in self.assets.values():
                update_distributional_ptrs_for_experience(
                    head_trial=asset.trial,
                    therapeutic_area=asset.therapeutic_area,
                    ta_experience=self.ta_experience,
                    base_concentration=self._distributional_ptrs_config.prior_concentration,
                    experience_to_full_knowledge=(
                        self._ta_experience_config.experience_to_full_knowledge
                    ),
                )

    def _calculate_capacity_usage(
        self,
        assets: dict[uuid.UUID, DrugAsset],
    ) -> float:
        """
        Calculate total capacity usage from all in-development assets.

        Parameters
        ----------
        assets : dict[uuid.UUID, DrugAsset]
            Dictionary of assets to calculate capacity for.

        Returns
        -------
        float
            Total capacity usage.

        """
        if self._rd_capacity_config is None or not self._rd_capacity_config.enabled:
            return 0.0

        use_levels = (
            self._investment_levels_config is not None
            and self._investment_levels_config.enabled
        )

        total_capacity = 0.0
        if use_levels:
            level_name_map = {
                InvestmentLevel.NONE: "none",
                InvestmentLevel.MINIMAL: "minimal",
                InvestmentLevel.STANDARD: "standard",
                InvestmentLevel.ACCELERATED: "accelerated",
            }
            for asset in assets.values():
                if asset.state == AssetState.InDevelopment:
                    level_name = level_name_map[asset.current_investment_level]
                    level_config = self._investment_levels_config.get_level_params(
                        level_name
                    )
                    total_capacity += level_config.capacity_cost
        else:
            # Binary action space: each InDevelopment asset = 1 capacity unit
            for asset in assets.values():
                if asset.state == AssetState.InDevelopment:
                    total_capacity += 1.0

        return total_capacity

    def _get_global_success_modifier(self, capacity_used: float) -> float:
        """
        Calculate global success modifier based on capacity overage.

        Parameters
        ----------
        capacity_used : float
            Current total capacity usage.

        Returns
        -------
        float
            Success modifier (1.0 if under capacity, <1.0 if over).

        """
        if self._rd_capacity_config is None:
            return 1.0

        return self._rd_capacity_config.calculate_success_modifier(capacity_used)

    def _get_global_cost_modifier(self, capacity_used: float) -> float:
        """
        Calculate global cost modifier based on capacity overage.

        Parameters
        ----------
        capacity_used : float
            Current total capacity usage.

        Returns
        -------
        float
            Cost modifier (1.0 if under capacity, >1.0 if over).

        """
        if self._rd_capacity_config is None:
            return 1.0

        return self._rd_capacity_config.calculate_cost_modifier(capacity_used)

    @property
    def drop_action_enabled(self) -> bool:
        """Whether the voluntary drop action feature is active."""
        return (
            self._drop_action_config is not None
            and self._drop_action_config.enabled
        )

    # ------------------------------------------------------------------
    # Clinical sites accounting (Model B)
    # ------------------------------------------------------------------
    @property
    def clinical_sites_enabled(self) -> bool:
        """Whether the clinical sites capacity feature is active."""
        return (
            self._clinical_sites_config is not None
            and self._clinical_sites_config.enabled
        )

    @property
    def sites_occupied(self) -> int:
        """
        Number of sites currently hosting a trial.

        Derived from the live InDevelopment asset count (Model B): a site is
        occupied for exactly as long as its asset is InDevelopment, and freed
        the moment the asset leaves that state (including the gap between phases).
        """
        return sum(
            1
            for asset in self.assets.values()
            if asset.state == AssetState.InDevelopment
        )

    @property
    def free_sites(self) -> int:
        """Operational sites not currently hosting a trial (never negative)."""
        if not self.clinical_sites_enabled:
            return 0
        return max(0, self.operational_sites - self.sites_occupied)

    @property
    def total_sites_owned(self) -> int:
        """
        Sites the agent owns.

        Operational + in-development (auction wins are operational, so already
        counted). Drives the Fibonacci purchase price.
        """
        return self.operational_sites + len(self.sites_in_development)

    def next_site_purchase_cost(self) -> float:
        """Cash cost to buy the next site at the current ownership level."""
        if not self.clinical_sites_enabled:
            return 0.0
        return self._clinical_sites_config.purchase_cost(self.total_sites_owned)

    def can_afford_site_purchase(self) -> bool:
        """Whether the agent can currently afford another site purchase."""
        if not self.clinical_sites_enabled:
            return False
        return self.cash >= self.next_site_purchase_cost()

    def advance_site_timers(self) -> int:
        """
        Decrement in-development site build timers; promote any that finish.

        Called once at the start of each step. Returns the number of sites that
        became operational this step.
        """
        if not self.clinical_sites_enabled or not self.sites_in_development:
            return 0
        remaining: list[int] = []
        promoted = 0
        for timer in self.sites_in_development:
            new_timer = timer - 1
            if new_timer <= 0:
                promoted += 1
            else:
                remaining.append(new_timer)
        if promoted:
            self.operational_sites += promoted
        self.sites_in_development = remaining
        return promoted

    def add_operational_site(self) -> None:
        """Add one immediately-usable site (e.g. an auction win)."""
        if not self.clinical_sites_enabled:
            return
        self.operational_sites += 1

    def start_site_build(self) -> None:
        """
        Begin building one purchased site (adds a build-delay timer).

        Cash is charged by the caller; this only mutates site state.
        """
        if not self.clinical_sites_enabled:
            return
        self.sites_in_development = self.sites_in_development + [
            self._clinical_sites_config.site_development_steps
        ]

    def with_auction_site_win(self, price: float) -> "GameState":
        """
        Return a copy that has won an immediately-operational site at auction.

        The won site is usable at once (no build delay), so ``operational_sites``
        is incremented directly. ``price`` is charged to cash and booked as a
        realised cost this step; bankruptcy is allowed (mirrors the BD auction,
        which has no affordability mask). Private config attributes and the
        original state are left untouched (the mutable site list is copied).
        """
        updated_costs = list(self.realised_costs)
        if updated_costs:
            updated_costs[-1] += price
        else:
            updated_costs.append(price)
        new_cash = self.cash - price
        return self.model_copy(
            update={
                "cash": new_cash,
                "operational_sites": self.operational_sites + 1,
                "sites_in_development": list(self.sites_in_development),
                "realised_costs": updated_costs,
                "game_ended": new_cash < 0 or self.time >= self.horizon,
                "ended_reason": (
                    self.ended_reason
                    if self.ended_reason
                    else (
                        "horizon_reached"
                        if self.time >= self.horizon
                        else ("bankrupt" if new_cash < 0 else None)
                    )
                ),
            }
        )

    @property
    def capacity_ratio(self) -> float:
        """Get capacity usage ratio (used / base)."""
        if self.capacity_base <= 0:
            return 0.0
        return self.capacity_used / self.capacity_base

    @property
    def capacity_headroom(self) -> float:
        """Get capacity headroom ((base - used) / base). Negative if over capacity."""
        if self.capacity_base <= 0:
            return 0.0
        return (self.capacity_base - self.capacity_used) / self.capacity_base

    @property
    def success_modifier(self) -> float:
        """Get current global success modifier based on capacity usage."""
        return self._get_global_success_modifier(self.capacity_used)

    @property
    def cost_modifier(self) -> float:
        """Get current global cost modifier based on capacity usage."""
        return self._get_global_cost_modifier(self.capacity_used)

    @model_validator(mode="after")
    def post_init_check_game_ended_horizon(self) -> Self:
        """Raises if game_ended is False but time has reached horizon."""
        if self.time >= self.horizon and not self.game_ended:
            raise RuntimeError(
                "Game has reached horizon but game_ended is False. Shouldn't happen."
            )
        return self

    @model_validator(mode="after")
    def post_init_check_game_ended_cash_negative(self) -> Self:
        """Raises if game_ended is False but cash is negative."""
        if self.cash < 0.0 and not self.game_ended:
            raise RuntimeError(
                "Game cash < 0. but game_ended is False. Shouldn't happen."
            )
        return self

    @field_validator("assets", mode="after")
    @classmethod
    def validate_no_expired_or_failed_assets_in_assets(cls, v) -> int:
        """Validate no expired or failed assets in assets dict."""
        for asset in v.values():
            if asset.state == AssetState.Expired:
                raise ValueError(
                    f"assets dict contains expired asset {asset.id},"
                    " which should be in expired_assets dict."
                )
            if asset.state == AssetState.Failed:
                raise ValueError(
                    f"assets dict contains failed asset {asset.id},"
                    " which should be in failed_assets dict."
                )
            if asset.state == AssetState.Dropped:
                raise ValueError(
                    f"assets dict contains dropped asset {asset.id},"
                    " which should be in dropped_assets dict."
                )

        return v

    @field_validator("time", mode="after")
    @classmethod
    def validate_time(cls, v) -> int:
        """Validate that time is non-negative."""
        if v < 0:
            raise ValueError(f"time must be non-negative, received {v}")
        return v

    @field_validator("horizon", mode="after")
    @classmethod
    def validate_horizon(cls, v) -> int:
        """Validate that horizon is positive."""
        if v <= 0:
            raise ValueError(f"horizon must be positive, received {v}")
        return v

    @model_validator(mode="after")
    def validate_time_and_horizon(self) -> GameState:
        """Validate that time is not greater than horizon."""
        if self.time > self.horizon:
            raise ValueError(
                f"time must be less than or equal to horizon, "
                f"received time: {self.time}, horizon: {self.horizon}"
            )
        return self

    @classmethod
    def initialise_new_game(
        cls,
        asset_generator_cls: type[AssetGeneratorBase],
        num_assets: int,
        cash: float,
        horizon: int,
        max_num_assets: int,
        asset_arrival_sensitivity_below: float,
        asset_arrival_sensitivity_above: float,
        reinvestment_percentage: float,
        seed: int | None,
        *,
        investment_levels_config: InvestmentLevelsConfig,
        interim_trial_observations_config: InterimTrialObservationsConfig,
        distributional_ptrs_config: DistributionalPtrsConfig,
        rd_capacity_config: CapacityConfig,
        drop_action_config: DropActionConfig,
        marketing_config: MarketingConfig,
        clinical_sites_config: ClinicalSitesConfig,
        ptrs_readings_config: PtrsReadingsConfig,
        ta_experience_config: TAExperienceConfig,
        uncertain_ptrs_config: UncertainPtrsConfig,
        approval_phase_config: ApprovalPhaseConfig,
        **asset_generator_kwargs,
    ) -> GameState:
        """
        Initialise the game state from the start of the simulation.

        If seed is provided, the game-wide RNG is re-initialized with that seed before
        generating assets. If seed is None, the existing RNG (already initialized via
        init_game_rng) is used — allowing multiple calls to share a single RNG stream
        (e.g. multi-agent initialization).

        Parameters
        ----------
        asset_generator_cls : type[AssetGeneratorBase]
            The class of the asset generator to use.
        num_assets : int
            The number of assets to generate for this simulation.
        cash : float
            The initial amount of cash available to the player.
        horizon : int
            The time horizon for the simulation.
        max_num_assets : int
            The maximum number of assets allowed in the game.
        asset_arrival_sensitivity_below : float
            Controls mean reversion speed when below equilibrium.
        asset_arrival_sensitivity_above : float
            Controls fluctuation width at/above equilibrium.
        reinvestment_percentage : float
            Fraction of revenues available as cash for reinvestment (0.0-1.0).
        seed : int | None
            Random seed. If not None, re-initializes the game-wide RNG with this seed.
            Pass None to reuse the existing RNG (for multi-agent initialization where
            all agents share a single RNG stream).
        investment_levels_config : InvestmentLevelsConfig
            Configuration for the investment levels feature.
        interim_trial_observations_config : InterimTrialObservationsConfig
            Configuration for the interim trial observations feature.
        distributional_ptrs_config : DistributionalPtrsConfig
            Configuration for the distributional PTRS feature.
        rd_capacity_config : CapacityConfig
            Configuration for the R&D capacity constraint feature.
        drop_action_config : DropActionConfig
            Configuration for the standalone drop action feature.
        marketing_config : MarketingConfig
            Configuration for the marketing feature.
        clinical_sites_config : ClinicalSitesConfig
            Configuration for the clinical sites feature.
        ptrs_readings_config : PtrsReadingsConfig
            Configuration for the PTRS readings feature.
        ta_experience_config : TAExperienceConfig
            Configuration for the TA experience feature.
        uncertain_ptrs_config : UncertainPtrsConfig
            Configuration for the uncertain PTRS feature.
        approval_phase_config : ApprovalPhaseConfig
            Configuration for the approval phase feature.
        **asset_generator_kwargs : dict
            Additional keyword arguments for the asset generator.

        Returns
        -------
        GameState
            A GameState object at time 0.

        """
        if seed is not None:
            init_game_rng(seed)

        logger.debug("Initialising new game state...")

        # Sample TA quality modifiers if distributional PTRS is enabled
        ta_quality_modifiers = {}
        ta_quality_estimates = {}
        ta_quality_confidences = {}
        ta_rng = get_game_rng()

        if (
            distributional_ptrs_config.enabled
        ):
            for ta, variance in distributional_ptrs_config.ta_quality_variance.items():
                # Sample hidden TA quality modifier from Normal(0, sqrt(variance))
                std = variance**0.5
                modifier = ta_rng.gauss(0, std)
                ta_quality_modifiers[ta] = modifier
                # Initialize observable estimates to zero (no information yet)
                ta_quality_estimates[ta] = 0.0
                # Initial confidence is low (no trial outcomes yet)
                ta_quality_confidences[ta] = 0.0
                logger.info(
                    f"[DistributionalPTRS] Sampled TA quality modifier for {ta}: "
                    f"{modifier:.4f} (variance={variance})"
                )
        else:
            # Initialize empty dicts when feature is disabled
            for ta in [
                "oncology",
                "respiratory and immunology",
                "vaccines and infectious disease",
            ]:
                ta_quality_estimates[ta] = 0.0
                ta_quality_confidences[ta] = 1.0  # Full confidence when no uncertainty

        # Pass the asset-generator-relevant configs plus the sampled TA quality
        # modifiers. Remaining generator args (assets_dir / assets_data_list,
        # indication_spread, indication_drift_speed, trial_cost_multiplier,
        # indications_per_ta, generator_index) flow through asset_generator_kwargs.
        asset_generator = asset_generator_cls(
            uncertain_ptrs_config=uncertain_ptrs_config,
            distributional_ptrs_config=distributional_ptrs_config,
            ta_quality_modifiers=ta_quality_modifiers,
            ta_experience_config=ta_experience_config,
            approval_phase_config=approval_phase_config,
            ptrs_readings_config=ptrs_readings_config,
            **asset_generator_kwargs,
        )
        assets = asset_generator(num_assets, "initial")

        # log the args for debugging
        logger.debug(
            f"GameState initialisation parameters: num_assets={num_assets}, "
            f"cash={cash}, horizon={horizon}, max_num_assets={max_num_assets}, "
            f"asset_generator_kwargs={asset_generator_kwargs}"
        )
        # Initialize TA experience to zero for all TAs
        initial_ta_experience = {
            "oncology": 0.0,
            "respiratory and immunology": 0.0,
            "vaccines and infectious disease": 0.0,
        }

        # Get capacity base from rd_capacity config
        capacity_base = rd_capacity_config.base_capacity

        # Clinical sites: seed the starting operational-site endowment
        initial_operational_sites = (
            clinical_sites_config.starting_sites
            if clinical_sites_config.enabled
            else 0
        )

        game_state = cls(
            id=uuid.uuid4(),
            cash=cash,
            time=0,
            horizon=horizon,
            equilibrium_num_assets=num_assets,
            max_num_assets=max_num_assets,
            asset_arrival_sensitivity_below=asset_arrival_sensitivity_below,
            asset_arrival_sensitivity_above=asset_arrival_sensitivity_above,
            reinvestment_percentage=reinvestment_percentage,
            initial_cash=cash,
            assets=assets,
            failed_assets={},
            expired_assets={},
            dropped_assets={},
            realised_costs=[],
            realised_revenues=[],
            running_enpv=[],
            running_eroi=[],
            game_ended=False,
            ended_reason=None,
            ta_experience=initial_ta_experience,
            capacity_used=0.0,
            capacity_base=capacity_base,
            ta_quality_estimates=ta_quality_estimates,
            ta_quality_confidences=ta_quality_confidences,
            operational_sites=initial_operational_sites,
            sites_in_development=[],
        )
        # Initialise asset generator after since it is private attribute
        game_state._asset_generator = asset_generator
        # Store TA experience config if provided
        game_state._ta_experience_config = ta_experience_config
        # Store uncertain PTRS config
        game_state._uncertain_ptrs_config = uncertain_ptrs_config
        # Store investment levels config if provided
        game_state._investment_levels_config = investment_levels_config
        # Store interim trial observations config if provided
        game_state._interim_trial_observations_config = (
            interim_trial_observations_config
        )
        # Store distributional PTRS config and TA quality modifiers
        game_state._distributional_ptrs_config = distributional_ptrs_config
        # Store R&D capacity config if provided
        game_state._rd_capacity_config = rd_capacity_config
        game_state._drop_action_config = drop_action_config
        game_state._ptrs_readings_config = ptrs_readings_config
        game_state._marketing_config = marketing_config
        game_state._brand_scores = {}
        game_state._brand_score_floors = {}
        game_state._clinical_sites_config = clinical_sites_config
        game_state._bd_asset_clones = {}
        game_state._ta_quality_modifiers = ta_quality_modifiers
        logger.debug("Initialised new game state...")
        return game_state._post_init_update_enpv_eroi()

    def content_fingerprint(self, seed: int | None = None) -> str:
        """
        Compute a deterministic fingerprint for this initial game state.

        Incorporates the serialized state (excluding the random id), the seed,
        and a hash of the package source code.
        """
        state_data = self.model_dump(mode="json")
        state_data.pop("id")  # exclude the random UUID4
        fingerprint_input = {
            "state": state_data,
            "seed": seed,
            "code_hash": _PACKAGE_CODE_HASH,
        }
        serialized = json.dumps(fingerprint_input, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def _add_new_assets_mean_reverting(
        self, assets_not_expired: dict[uuid.UUID, DrugAsset]
    ) -> dict[uuid.UUID, DrugAsset]:
        """
        Add new assets using dual Gaussian CDF mechanism.

        Two modes:
        1. Below equilibrium: Mean-reverting with sigma_below
        2. At/above equilibrium: Stochastic expansion with sigma_above

        Repeatedly samples to add assets until:
        - Failed to add (random draw fails)
        - Hit max_num_assets hard cap

        Parameters
        ----------
        assets_not_expired : dict[uuid.UUID, DrugAsset]
            Dictionary of assets that have not expired

        Returns
        -------
        dict[uuid.UUID, DrugAsset]
            Updated assets_not_expired dict

        """
        initial_num_assets = len(assets_not_expired)

        while True:
            current_num_assets = len(assets_not_expired)
            deviation = current_num_assets - self.equilibrium_num_assets

            # Stop if at max capacity
            if current_num_assets >= self.max_num_assets:
                logger.debug(
                    f"Asset arrival blocked: at max_num_assets={self.max_num_assets}"
                )
                break

            # Calculate probability based on position relative to equilibrium
            if deviation < 0:
                # BELOW EQUILIBRIUM: Mean-reverting CDF
                z_score = abs(deviation) / self.asset_arrival_sensitivity_below
                probability = 2 * norm.cdf(z_score) - 1
                mode = "below-equilibrium"
            else:
                # AT OR ABOVE EQUILIBRIUM: Complement of CDF (tail probability)
                offset = deviation  # 0, 1, 2, ...
                z_score = (offset + 1) / self.asset_arrival_sensitivity_above
                probability = 2 * (1 - norm.cdf(z_score))
                mode = "above-equilibrium"

            # Draw random number and decide whether to add asset
            random_draw = get_game_rng().random()
            if random_draw < probability:
                # Pass ta_experience for TA bias in asset arrival
                new_asset = self._asset_generator(
                    1,
                    "new",
                    ta_experience=self.ta_experience,
                    episode_progress=self.time / self.horizon,
                )
                asset_id = list(new_asset.keys())[0]
                logger.debug(
                    f"Added new asset ({mode}, deviation={deviation}, "
                    f"p={probability:.3f}, draw={random_draw:.3f}): {asset_id}"
                )
                assets_not_expired.update(new_asset)
            else:
                # Failed to add, stop for this step
                logger.debug(
                    f"Asset arrival rejected ({mode}, deviation={deviation}, "
                    f"p={probability:.3f}, draw={random_draw:.3f})"
                )
                break

        # Log final summary
        final_num_assets = len(assets_not_expired)
        assets_added = final_num_assets - initial_num_assets

        logger.debug(f"Asset arrival: +{assets_added} assets")

        return assets_not_expired

    @property
    def bankrupt(self) -> bool:
        """Whether the game is bankrupt."""
        return self.cash < 0.0

    @property
    def capital_over_time(self) -> list[float]:
        """
        Get the capital for all time steps of a game.

        If you call this property without having reached the horizon, the array is
        padded with zeros.
        """
        initial = np.array([self.initial_cash] + [0.0] * self.horizon)
        revenue = np.array(
            self.realised_revenues
            + [0.0] * (self.horizon - len(self.realised_revenues) + 1)
        )
        cost = np.array(
            self.realised_costs + [0.0] * (self.horizon - len(self.realised_costs) + 1)
        )
        return list(np.cumsum(initial + revenue * self.reinvestment_percentage - cost))

    @property
    def enpv_over_time(self) -> list[float]:
        """
        Get the enpv for all time steps of a game.

        If you call this property without having reached the horizon, the array is
        padded with zeros.
        """
        return self.running_enpv + [0.0] * (self.horizon - len(self.running_enpv))

    @property
    def eroi_over_time(self) -> list[float]:
        """
        Get the eroi for all time steps of a game.

        If you call this property without having reached the horizon, the array is
        padded with zeros.
        """
        return self.running_eroi + [0.0] * (self.horizon - len(self.running_eroi))

    def enpv(self) -> float:
        """
        Calculate the expected Net Present Value (NPV) of the game state.

        The intended use of this function is to compute the eNPV at the end of the game.
        It does not take into account budget constraints.

        Returns
        -------
        float
            The calculated NPV of the game state.

        """
        total_enpv = self.cash
        for asset_id, asset in list(self.assets.items()):
            if asset.state == AssetState.Idle:
                continue  # Skip if Idle
            total_enpv += asset.enpv
        return total_enpv

    def eroi(self) -> float:
        """
        Calculate the expected Return On Investment (ROI) of the game state.

        The intended use of this function is to compute the eROI at the end of the game.
        It does not take into account budget constraints.

        Returns
        -------
        float
            The calculated ROI of the game state.

        """

        def compute_asset_totals(asset: DrugAsset) -> tuple[float, float]:
            cash_flows = np.array(asset._projected_cash_flows)
            probabilities = np.array(asset._projected_probs)
            expected = cash_flows * probabilities
            expected_costs = list(
                -np.where(expected < 0, expected, 0)
            )  # Note the sign change
            expected_revenues = list(np.where(expected > 0, expected, 0))
            return sum(expected_costs), sum(expected_revenues)

        total_expected_revenue = 0.0
        total_expected_cost = 0.0
        for asset_id, asset in list(self.assets.items()):
            if asset.state == AssetState.Idle:
                continue  # Skip if Idle
            asset_total_cost, asset_total_revenue = compute_asset_totals(asset)
            total_expected_cost += asset_total_cost
            total_expected_revenue += asset_total_revenue
        if total_expected_cost == 0:
            return 0.0
        return (total_expected_revenue - total_expected_cost) / total_expected_cost

    def realised_roi(self) -> float:
        """
        Calculate the realised ROI of the game state.

        The intended use of this function is to compute the realised ROI at the end of
        the game.

        Returns
        -------
        float
            The calculated ROI of the game state.

        """
        total_realised_revenue = sum(self.realised_revenues)
        total_realised_cost = sum(self.realised_costs)
        if total_realised_cost == 0:
            return 0.0
        return (total_realised_revenue - total_realised_cost) / total_realised_cost

    def in_development_assets(
        self,
    ) -> dict[uuid.UUID, DrugAsset]:
        """Get list of in development assets."""
        return {
            asset_id: asset
            for asset_id, asset in self.assets.items()
            if asset.state == AssetState.InDevelopment
        }

    def _create_ended_state(
        self,
        cash: float,
        reason: GameEndReason,
        assets: dict[uuid.UUID, DrugAsset],
    ) -> "GameState":
        """Create a game-ended state with given parameters."""
        # Filter out failed/expired assets that may be in the dict
        # (e.g. from stop_development() during Phase A)
        active_assets = {
            aid: a
            for aid, a in assets.items()
            if a.state
            not in (AssetState.Failed, AssetState.Expired, AssetState.Dropped)
        }
        new_failed = {
            aid: a for aid, a in assets.items() if a.state == AssetState.Failed
        }
        new_expired = {
            aid: a for aid, a in assets.items() if a.state == AssetState.Expired
        }
        new_dropped = {
            aid: a for aid, a in assets.items() if a.state == AssetState.Dropped
        }
        ended_state = GameState(
            id=self.id,
            cash=cash,
            time=self.time,
            horizon=self.horizon,
            equilibrium_num_assets=self.equilibrium_num_assets,
            max_num_assets=self.max_num_assets,
            asset_arrival_sensitivity_below=self.asset_arrival_sensitivity_below,
            asset_arrival_sensitivity_above=self.asset_arrival_sensitivity_above,
            initial_cash=self.initial_cash,
            reinvestment_percentage=self.reinvestment_percentage,
            assets=active_assets,
            failed_assets={**self.failed_assets, **new_failed},
            expired_assets={**self.expired_assets, **new_expired},
            dropped_assets={**self.dropped_assets, **new_dropped},
            realised_costs=self.realised_costs,
            realised_revenues=self.realised_revenues,
            running_enpv=self.running_enpv,
            running_eroi=self.running_eroi,
            game_ended=True,
            ended_reason=reason,
            ta_experience=self.ta_experience.copy(),
            capacity_used=self.capacity_used,
            capacity_base=self.capacity_base,
            ta_quality_estimates=self.ta_quality_estimates.copy(),
            ta_quality_confidences=self.ta_quality_confidences.copy(),
            operational_sites=self.operational_sites,
            sites_in_development=list(self.sites_in_development),
        )
        ended_state._investment_levels_config = self._investment_levels_config
        ended_state._uncertain_ptrs_config = self._uncertain_ptrs_config
        ended_state._interim_trial_observations_config = (
            self._interim_trial_observations_config
        )
        ended_state._distributional_ptrs_config = self._distributional_ptrs_config
        ended_state._rd_capacity_config = self._rd_capacity_config
        ended_state._drop_action_config = self._drop_action_config
        ended_state._ptrs_readings_config = self._ptrs_readings_config
        ended_state._marketing_config = self._marketing_config
        ended_state._brand_scores = dict(self._brand_scores)
        ended_state._brand_score_floors = dict(self._brand_score_floors)
        ended_state._clinical_sites_config = self._clinical_sites_config
        ended_state._ta_quality_modifiers = self._ta_quality_modifiers.copy()
        ended_state._bd_asset_clones = {}
        return ended_state

    def _get_level_config(self, level: InvestmentLevel) -> dict:
        """Get configuration for an investment level."""
        level_name_map = {
            InvestmentLevel.NONE: "none",
            InvestmentLevel.MINIMAL: "minimal",
            InvestmentLevel.STANDARD: "standard",
            InvestmentLevel.ACCELERATED: "accelerated",
        }
        if self._investment_levels_config is None:
            # Default configs if feature disabled
            defaults = {
                "none": {
                    "cost_modifier": 0.0,
                    "speed_modifier": 0.0,
                    "success_modifier": 1.0,
                    "capacity_cost": 0,
                    "experience_modifier": 0.0,
                },
                "minimal": {
                    "cost_modifier": 0.7,
                    "speed_modifier": 0.67,
                    "success_modifier": 0.90,
                    "capacity_cost": 1,
                    "experience_modifier": 1.5,
                },
                "standard": {
                    "cost_modifier": 1.0,
                    "speed_modifier": 1.0,
                    "success_modifier": 1.0,
                    "capacity_cost": 2,
                    "experience_modifier": 1.0,
                },
                "accelerated": {
                    "cost_modifier": 1.5,
                    "speed_modifier": 1.33,
                    "success_modifier": 1.05,
                    "capacity_cost": 3,
                    "experience_modifier": 0.5,
                },
            }
            return defaults[level_name_map[level]]
        # Return dict for consistent access pattern
        params = self._investment_levels_config.get_level_params(level_name_map[level])
        return {
            "cost_modifier": params.cost_modifier,
            "speed_modifier": params.speed_modifier,
            "success_modifier": params.success_modifier,
            "capacity_cost": params.capacity_cost,
            "experience_modifier": params.experience_modifier,
        }

    def step(
        self,
        investor_actions: dict[uuid.UUID, InvestmentLevel | Literal["invest"] | None],
        market_shares: dict[uuid.UUID, float] | None = None,
        pricing_multipliers: dict[uuid.UUID, float] | None = None,
        demand_multipliers: dict[uuid.UUID, float] | None = None,
        brand_equity_actions: dict[uuid.UUID, int] | None = None,
        demand_creation_actions: dict[str, int] | None = None,
        demand_creation_cost_base: float = 0.0,
        research_actions: dict[uuid.UUID, int] | None = None,
        bd_current_assets: list | None = None,
        site_priorities: dict[uuid.UUID, float] | None = None,
        buy_site: bool = False,
    ) -> "GameState":
        """
        Advance the game state by one time step.

        Args:
            investor_actions: Mapping of asset IDs to investment levels.
             A dictionary mapping asset IDs to investment levels.
             - InvestmentLevel.NONE: Do not invest (for idle assets)
             - InvestmentLevel.MINIMAL/STANDARD/ACCELERATED: Invest at this level
             - "invest": Backward compatible, treated as InvestmentLevel.STANDARD
             For in-development assets, this changes their investment level.
            market_shares : dict[uuid.UUID, float] | None
             Optional per-drug market share multipliers (0.0-1.0).
             Used by multi-agent environment for revenue competition.
             If None, full revenue is collected (single-agent default).
            pricing_multipliers : dict[uuid.UUID, float] | None
             Optional per-drug pricing multipliers for on-market drugs.
             Applied to revenue before market share and reinvestment_percentage.
             If None, all drugs use 1.0x pricing (default).
            demand_multipliers : dict[uuid.UUID, float] | None
             Optional per-drug shared demand multipliers applied to revenue.
             If None, no demand-creation boost is applied.
            brand_equity_actions : dict[uuid.UUID, int] | None
             Optional per-drug brand-equity spend decisions for this step.
             If None, no brand-equity spend occurs.
            demand_creation_actions : dict[str, int] | None
             Optional per-indication demand-creation spend decisions for this step.
             If None, no demand-creation spend occurs.
            demand_creation_cost_base : float
             Static per-action demand-creation cost (indication-wide market
             potential); charged once per active demand-creation action.
            research_actions : dict[uuid.UUID, int] | None
             Optional per-asset PTRS reading counts to purchase this step.
             If None, no readings are taken.
            bd_current_assets : list | None
             Optional list of BD assets currently on offer, so readings can be
             taken on assets the agent does not yet own.
            site_priorities : dict[uuid.UUID, float] | None
             Optional per-asset priority scores for clinical-site arbitration
             (agent_priority mode). When the agent requests more new trials than
             it has free sites, sites go to the highest-priority assets; the rest
             are costless no-ops. When None, arbitration falls back to ascending
             asset order. Ignored unless the clinical sites feature is enabled.
            buy_site : bool
             Clinical sites "upgrade" action: when True, buy at most one new site
             this step. Pays the current Fibonacci purchase cost and enters the
             site into ``sites_in_development`` with the build-delay timer. Only
             honoured when the feature is enabled and the agent can afford it;
             otherwise a costless no-op.

        Returns:
            GameState: New game state.

        """
        logger.debug(f"STARTING STEP: {self.time + 1} out of {self.horizon}")

        # Clinical sites: promote any sites whose build delay elapsed this step,
        # before the new-trial gate runs (so a just-finished site can host a
        # trial this step). Mutates self; the promoted counts are copied forward
        # into the next state's constructors below.
        self.advance_site_timers()

        # Convert string actions to InvestmentLevel
        normalized_actions: dict[uuid.UUID, InvestmentLevel] = {}
        for asset_id, action in investor_actions.items():
            if action == "invest":
                normalized_actions[asset_id] = InvestmentLevel.STANDARD
            elif action == "drop":
                normalized_actions[asset_id] = InvestmentLevel.DROP
            elif isinstance(action, InvestmentLevel):
                normalized_actions[asset_id] = action
            # None or missing = no action

        # Log TA experience at start of step (only when enabled)
        if (
            self._ta_experience_config is not None
            and self._ta_experience_config.enabled
        ):
            logger.info("TA Experience (post-decay from previous step):")
            for ta, exp in sorted(self.ta_experience.items()):
                logger.info(f"  {ta}: {exp:.4f}")

        logger.debug("Current game state before step:")
        logger.debug(
            f"  Cash: {self.cash}, Capacity: {self.capacity_used}/{self.capacity_base}"
        )
        logger.debug(f"  Time: {self.time}, Assets: {len(self.assets)}")

        current_cash = self.cash
        current_realised_cost = 0.0
        current_realised_revenue = 0.0

        # Check if investment levels feature is enabled
        use_investment_levels = (
            self._investment_levels_config is not None
            and self._investment_levels_config.enabled
        )

        # Calculate global cost modifier based on current capacity usage
        # (penalty from previous step's overcommitment)
        global_cost_modifier = self._get_global_cost_modifier(self.capacity_used)
        if use_investment_levels and global_cost_modifier > 1.0:
            logger.info(
                f"[Capacity Cost Penalty] capacity_used={self.capacity_used:.1f}, "
                f"cost_modifier={global_cost_modifier:.3f}"
            )

        # A (Pt.1): pay for ongoing investments
        logger.debug("Step A (Pt.1): paying for ongoing investments")
        in_dev_assets = self.in_development_assets()
        for in_dev_asset in in_dev_assets.values():
            # Check if there's a level change for this asset
            new_level = normalized_actions.get(in_dev_asset.id)

            # If STOP or DROP is requested, don't pay costs
            # (asset will be stopped/dropped in Pt.2)
            if new_level in (InvestmentLevel.STOP, InvestmentLevel.DROP):
                logger.debug(
                    f"Skipping cost for {in_dev_asset.name} (will be stopped/dropped)"
                )
                continue

            if new_level is not None and new_level != InvestmentLevel.NONE:
                level = new_level
            else:
                level = in_dev_asset.current_investment_level

            if use_investment_levels:
                level_config = self._get_level_config(level)
                base_cost = in_dev_asset.cost_this_step_with_modifier(
                    level_config["cost_modifier"]
                )
                # Apply global cost modifier for capacity overage
                cost = base_cost * global_cost_modifier
            else:
                # Apply global cost modifier for capacity overage
                cost = in_dev_asset.cost_this_step * global_cost_modifier

            logger.debug(
                f"Paying for ongoing: {in_dev_asset.name}, "
                f"level={level.name}, cost={cost:.2f}"
            )
            current_cash -= cost
            current_realised_cost += cost

        if current_cash <= 0.0:
            logger.debug(f"GAME ENDED: {GameEndReason.ONGOING_INVESTMENTS.value}")
            return self._create_ended_state(
                current_cash, GameEndReason.ONGOING_INVESTMENTS, self.assets
            )

        assets_for_step = self.assets.copy()

        # Clinical sites upgrade: buy at most one new site this step. The site
        # enters the build pipeline (2-year delay), so it does not change
        # free_sites for the gate below; only the cash is spent now. Ignored if
        # unaffordable (costless no-op), mirroring the env-level affordability
        # mask on the upgrade action.
        if buy_site and self.clinical_sites_enabled:
            purchase_cost = self.next_site_purchase_cost()
            if current_cash >= purchase_cost:
                current_cash -= purchase_cost
                current_realised_cost += purchase_cost
                self.start_site_build()
                logger.debug(
                    f"Clinical sites: bought a site for {purchase_cost:.2f} "
                    f"(now {self.total_sites_owned} owned, "
                    f"{len(self.sites_in_development)} building)"
                )
            else:
                logger.debug(
                    f"Clinical sites: upgrade skipped, cannot afford "
                    f"{purchase_cost:.2f} (cash={current_cash:.2f})"
                )

        # Clinical sites gate: if the agent requests more new trials than it has
        # free sites, arbitrate which requests are granted (agent_priority order
        # or ascending asset index). Denied requests become costless no-ops.
        denied_site_requests: set[uuid.UUID] = set()
        if self.clinical_sites_enabled:
            requested = [
                asset_id
                for asset_id in self.assets  # ascending asset (arrival) order
                if normalized_actions.get(asset_id)
                not in (None, InvestmentLevel.NONE, InvestmentLevel.DROP,
                        InvestmentLevel.STOP)
                and self.assets[asset_id].state == AssetState.Idle
            ]
            granted = resolve_site_grants(
                requested, self.free_sites, site_priorities
            )
            denied_site_requests = set(requested) - granted
            if denied_site_requests:
                logger.debug(
                    f"Clinical sites: {len(denied_site_requests)} new-trial "
                    f"request(s) denied (free_sites={self.free_sites})"
                )

        # A (Pt.2): pay for new investments and update investment levels
        logger.debug("Step A (Pt.2): paying for new investments")
        for asset_id, level in normalized_actions.items():
            if level == InvestmentLevel.NONE:
                continue

            asset = assets_for_step[asset_id]

            if level == InvestmentLevel.DROP:
                if self._drop_action_config is not None and asset.trial is not None:
                    current_cash -= self._drop_action_config.calculate_drop_fee(
                        asset.trial.cost_remaining
                    )
                assets_for_step[asset_id] = asset.drop()
                logger.debug(f"Dropped asset: {asset.name} (was {asset.state})")
                continue

            if asset.state == AssetState.Idle:
                # Clinical sites: this new trial lost the site arbitration this
                # step — costless no-op (asset stays Idle, no cost charged).
                if asset_id in denied_site_requests:
                    logger.debug(
                        f"Clinical sites: skipping {asset.name} (no free site)"
                    )
                    continue

                # New investment
                # Check if interim trial observations are enabled
                enable_interim = (
                    self._interim_trial_observations_config is not None
                    and self._interim_trial_observations_config.enabled
                )
                if use_investment_levels:
                    invested_asset = asset.to_develop_with_level(
                        level, enable_interim_observations=enable_interim
                    )
                    level_config = self._get_level_config(level)
                    base_cost = invested_asset.cost_this_step_with_modifier(
                        level_config["cost_modifier"]
                    )
                    # Apply global cost modifier for capacity overage
                    cost = base_cost * global_cost_modifier
                else:
                    invested_asset = asset.to_develop(
                        enable_interim_observations=enable_interim
                    )
                    cost = invested_asset.cost_this_step * global_cost_modifier

                logger.debug(
                    f"New investment: {asset.name}, level={level.name}, cost={cost:.2f}"
                )
                current_cash -= cost
                current_realised_cost += cost
                assets_for_step[asset_id] = invested_asset

            elif asset.state == AssetState.InDevelopment:
                # Handle in-development assets
                if level == InvestmentLevel.STOP:
                    # Stop development early (agent decides to abandon)
                    assets_for_step[asset_id] = asset.stop_development()
                    logger.debug(f"Stopped development: {asset.name}")
                elif use_investment_levels:
                    # Level change for existing development
                    if level != asset.current_investment_level:
                        assets_for_step[asset_id] = asset.set_investment_level(level)
                        logger.debug(f"Level change: {asset.name} -> {level.name}")
                else:
                    # Legacy mode: can't re-invest in something already in development
                    raise ValueError(
                        f"Cannot invest in asset {asset_id} - already in development. "
                        f"Asset state: {asset.state}"
                    )

        if current_cash < 0.0:
            logger.debug(f"GAME ENDED: {GameEndReason.NEW_INVESTMENTS.value}")
            return self._create_ended_state(
                current_cash, GameEndReason.NEW_INVESTMENTS, assets_for_step
            )

        # A (Pt.4): pay for and immediately apply ptrs research readings (before
        # evolution so positional sigma assignment matches the chain at the time
        # of the request)
        if (
            self._ptrs_readings_config is not None
            and self._ptrs_readings_config.enabled
            and research_actions
        ):
            logger.debug("Step A (Pt.4): paying for and applying ptrs readings")
            cfg = self._ptrs_readings_config
            rng = get_game_rng()

            def _reading_base_cost(target) -> float:
                """Base cost of one reading: a fraction of the cost remaining."""
                cost_rem = (
                    target.trial.cost_remaining if target.trial is not None else 0.0
                )
                # Shared with the response layer so the displayed reading cost
                # can never drift from what is charged here.
                return cfg.reading_base_cost(cost_rem)

            for asset_id, n in research_actions.items():
                if n <= 0:
                    continue
                asset = assets_for_step.get(asset_id)
                if asset is None:
                    continue
                cost = fibonacci_cost(n, _reading_base_cost(asset))
                current_cash -= cost
                current_realised_cost += cost
                if asset.pending_trial_chain:
                    asset.apply_pending_readings(
                        n,
                        cfg.sigma_logit_base,
                        list(cfg.noise_multipliers),
                        rng,
                    )
                    for trial in asset.pending_trial_chain:
                        if trial.ptrs_sample_mean is not None:
                            trial.ptrs = trial.ptrs_sample_mean
                    # Invalidate enpv/eroi caches so the obs builder sees new ptrs
                    asset.__dict__.pop("_projected_probs", None)
                    asset.__dict__.pop("_projected_cash_flows", None)
            # BD asset readings — apply to per-agent clone, not the shared market asset
            if bd_current_assets:
                bd_lookup = {str(a.id): a for a in bd_current_assets}
                for asset_id, n in research_actions.items():
                    if n <= 0:
                        continue
                    shared = bd_lookup.get(str(asset_id))
                    if shared is None:
                        continue
                    if asset_id in assets_for_step:
                        # Agent won it this step — handled as a portfolio asset above.
                        continue
                    clone = self._bd_asset_clones.get(str(asset_id))
                    if clone is None:
                        clone = shared.model_copy(deep=True)
                        self._bd_asset_clones[str(asset_id)] = clone
                    cost = fibonacci_cost(n, _reading_base_cost(shared))
                    current_cash -= cost
                    current_realised_cost += cost
                    if clone.pending_trial_chain:
                        clone.apply_pending_readings(
                            n,
                            cfg.sigma_logit_base,
                            list(cfg.noise_multipliers),
                            rng,
                        )
                        for trial in clone.pending_trial_chain:
                            if trial.ptrs_sample_mean is not None:
                                trial.ptrs = trial.ptrs_sample_mean
                        # model_copy(deep=True) copies __dict__, so the clone
                        # inherits any cached_property values already computed on
                        # the shared asset (the BD action mask calls cash_enpv on
                        # it every step). Invalidate so enpv/eroi reflect the
                        # agent's private readings.
                        clone.__dict__.pop("_projected_probs", None)
                        clone.__dict__.pop("_projected_cash_flows", None)
            if current_cash < 0.0:
                logger.debug(f"GAME ENDED: {GameEndReason.PTRS_READINGS_COSTS.value}")
                return self._create_ended_state(
                    current_cash, GameEndReason.PTRS_READINGS_COSTS, assets_for_step
                )

        logger.debug("Imaginary investment time period passing.")
        #######################################################
        # This is where the imaginary time period happens.    #
        # Imagine a quarter passing, drug trials are ongoing. #
        #######################################################

        # B collect any revenues
        logger.debug("Step B: collecting revenues")
        logger.debug(f"Current cash before collecting revenues: {current_cash}")
        for asset_id, asset in assets_for_step.items():
            if asset.state == AssetState.OnMarket:
                # Apply pricing multiplier (per-drug price level)
                price_mult = 1.0
                if pricing_multipliers is not None:
                    price_mult = pricing_multipliers.get(asset_id, 1.0)
                priced_revenue = asset.revenue_this_step * price_mult

                # Apply market share and demand creation multiplier
                if market_shares is not None:
                    share = market_shares.get(asset_id, 0.0)
                else:
                    share = 1.0
                demand_mult = (
                    demand_multipliers.get(asset_id, 1.0)
                    if demand_multipliers is not None
                    else 1.0
                )
                effective_revenue = priced_revenue * share * demand_mult
                cash_collected = effective_revenue * self.reinvestment_percentage
                logger.debug(
                    f"Collecting revenue: {asset.name}, "
                    f"revenue={asset.revenue_this_step}, price_mult={price_mult:.2f}, "
                    f"share={share:.2f}, demand_mult={demand_mult:.3f}, "
                    f"collected={cash_collected}"
                )
                current_cash += cash_collected
                # Track price/share-adjusted revenue for metrics
                current_realised_revenue += effective_revenue
        logger.debug(f"Current cash after collecting revenues: {current_cash}")

        # B.5: pay marketing costs (demand creation + brand equity) and update scores
        marketing_cfg = self._marketing_config
        new_brand_scores: dict[uuid.UUID, float] = dict(self._brand_scores)
        if marketing_cfg is not None and marketing_cfg.enabled:
            # Demand creation: deduct an indication-wide cost for each indication
            # the agent sizes. DC boosts the *whole* indication's shared demand
            # multiplier, so the cost is anchored to the pool-wide market size
            # (the static peak max_revenue passed in as demand_creation_cost_base)
            # rather than to whichever drug the agent happens to hold there. This
            # makes the spend unconditional -- an agent can size a market even
            # before it has a drug on-market -- so DC is never free.
            if demand_creation_actions is not None:
                cost = marketing_cfg.dc_cost(demand_creation_cost_base)
                for action in demand_creation_actions.values():
                    if action == 1:
                        current_cash -= cost
                        current_realised_cost += cost
            # Brand equity: update scores and deduct costs
            if brand_equity_actions is not None:
                for asset_id, asset in assets_for_step.items():
                    if brand_equity_actions.get(asset_id, 0) == 1:
                        cost = marketing_cfg.be_cost(asset.max_revenue)
                        current_cash -= cost
                        current_realised_cost += cost
                        new_brand_scores[asset_id] = (
                            new_brand_scores.get(asset_id, 0.0) + marketing_cfg.be_boost
                        )
            if current_cash < 0.0:
                logger.debug("GAME ENDED: marketing costs exceeded cash")
                return self._create_ended_state(
                    current_cash, GameEndReason.ONGOING_INVESTMENTS, assets_for_step
                )
            # Decay all brand scores each step toward their floor (not toward zero)
            new_brand_scores = {
                aid: max(
                    self._brand_score_floors.get(aid, 0.0),
                    score * (1.0 - marketing_cfg.be_decay_rate),
                )
                for aid, score in new_brand_scores.items()
            }

        # C evolve assets
        logger.debug("Step C: evolving assets")

        # Calculate capacity usage and global success modifier
        new_capacity_used = self._calculate_capacity_usage(assets_for_step)
        global_success_modifier = self._get_global_success_modifier(new_capacity_used)

        if use_investment_levels:
            logger.info(
                f"[Capacity] Usage: {new_capacity_used:.1f}/{self.capacity_base:.1f} "
                f"(ratio: {new_capacity_used / self.capacity_base:.2f}), "
                f"success_modifier: {global_success_modifier:.3f}"
            )

        # Track trial completions for TA experience (uncertain PTRS feature)
        # Also track investment levels for experience modifier
        pre_evolve_trials = {
            asset_id: (
                asset.therapeutic_area,
                asset.trial.phase,
                asset.trial.time_remaining,
                asset.current_investment_level,
            )
            for asset_id, asset in assets_for_step.items()
            if asset.state == AssetState.InDevelopment
        }

        # Evolve assets with investment level modifiers
        evolved_assets = {}
        for asset_id, asset in assets_for_step.items():
            if asset.state == AssetState.Dropped:
                evolved_assets[asset_id] = asset
                continue
            if asset.state == AssetState.InDevelopment and use_investment_levels:
                level_config = self._get_level_config(asset.current_investment_level)
                evolved_assets[asset_id] = asset.evolve_with_level(
                    speed_modifier=level_config["speed_modifier"],
                    success_modifier=level_config["success_modifier"],
                    global_success_modifier=global_success_modifier,
                )
            elif (
                asset.state == AssetState.InDevelopment
                and global_success_modifier < 1.0
            ):
                # Apply capacity overage success penalty even without investment levels
                evolved_assets[asset_id] = asset.evolve_with_level(
                    speed_modifier=1.0,
                    success_modifier=1.0,
                    global_success_modifier=global_success_modifier,
                )
            else:
                evolved_assets[asset_id] = asset.evolve()

        # Detect trial completions by comparing pre/post evolve states
        # A trial completes when time_remaining was 1 (now 0) and phase resolved
        trial_completions = []
        for asset_id, (
            ta,
            phase,
            time_remaining,
            inv_level,
        ) in pre_evolve_trials.items():
            if time_remaining == 1:
                # Trial phase was about to complete
                evolved_asset = evolved_assets[asset_id]
                # Check if trial completed (phase changed or asset state changed)
                if evolved_asset.state != AssetState.InDevelopment or (
                    evolved_asset.trial.phase != phase
                ):
                    trial_completions.append((ta, phase, inv_level))
                    logger.debug(f"Trial completion detected: {ta} {phase.value}")

        # Record trial completions for TA experience (experience gained at end of trial,
        # regardless of whether trial passes or fails)
        updated_ta_experience = self.ta_experience.copy()

        ta_experience_enabled = (
            self._ta_experience_config is not None
            and self._ta_experience_config.enabled
        )

        if ta_experience_enabled:
            # Apply decay to existing experience BEFORE adding new experience.
            # This ensures new experience gained this step isn't immediately decayed.
            # Decay happens every step to existing experience.
            decay_rate = self._ta_experience_config.experience_decay_rate
            if decay_rate < 1.0:
                logger.info(f"[TAExperience] Applying decay_rate={decay_rate:.4f}")
                for ta in sorted(updated_ta_experience.keys()):
                    old_exp = updated_ta_experience[ta]
                    updated_ta_experience[ta] *= decay_rate
                    new_exp = updated_ta_experience[ta]
                    logger.info(
                        f"  {ta}: {old_exp:.4f} -> {new_exp:.4f} "
                        f"(decay: -{old_exp - new_exp:.4f})"
                    )

            # Add new experience from trial completions (not decayed this step)
            for ta, phase, inv_level in trial_completions:
                phase_key_map = {
                    TrialPhase.PHASE_1: "phase_1",
                    TrialPhase.PHASE_2: "phase_2",
                    TrialPhase.PHASE_3: "phase_3",
                    TrialPhase.APPROVAL: "approval",
                }
                phase_key = phase_key_map[phase]

                weight = self._ta_experience_config.phase_experience_weights[phase_key]

                # Apply experience_modifier from investment level
                # Accelerated = less learning (0.5x), Minimal = more learning (1.5x)
                if use_investment_levels:
                    level_config = self._get_level_config(inv_level)
                    exp_modifier = level_config["experience_modifier"]
                    weight *= exp_modifier
                    logger.debug(
                        f"Experience modifier from {inv_level.name}: {exp_modifier:.2f}"
                    )

                # Check total experience cap before adding new experience
                current_total = sum(updated_ta_experience.values())
                max_total = self._ta_experience_config.max_total_experience

                if max_total is not None and current_total >= max_total:
                    logger.info(
                        f"[TAExperience] Trial completed: {ta} {phase.value} "
                        f"+0.0 exp (CAPPED at {max_total:.1f}, "
                        f"total: {current_total:.1f})"
                    )
                    continue

                # Calculate how much experience we can actually add (may be capped)
                actual_weight = weight
                if max_total is not None:
                    available = max_total - current_total
                    if weight > available:
                        actual_weight = available
                        logger.info(
                            f"[TAExperience] Experience capped: wanted +{weight:.1f}, "
                            f"adding +{actual_weight:.1f} (cap: {max_total:.1f})"
                        )

                current = updated_ta_experience.get(ta, 0.0)
                updated_ta_experience[ta] = current + actual_weight
                logger.info(
                    f"[TAExperience] Trial completed: {ta} {phase.value} "
                    f"+{actual_weight:.1f} exp (total: {updated_ta_experience[ta]:.1f})"
                )

        newly_failed_assets = {
            asset_id: asset
            for asset_id, asset in evolved_assets.items()
            if asset.state == AssetState.Failed
        }
        newly_expired_assets = {
            asset_id: asset
            for asset_id, asset in evolved_assets.items()
            if asset.state == AssetState.Expired
        }
        newly_dropped_assets = {
            asset_id: asset
            for asset_id, asset in evolved_assets.items()
            if asset.state == AssetState.Dropped
        }
        active_assets = {
            asset_id: asset
            for asset_id, asset in evolved_assets.items()
            if asset.state
            not in (AssetState.Expired, AssetState.Failed, AssetState.Dropped)
        }

        # Log failed/expired/dropped assets
        for asset_id in newly_failed_assets:
            logger.debug(f"Asset failed: {asset_id}.")
        for asset_id in newly_expired_assets:
            logger.debug(f"Asset expired: {asset_id}.")
        for asset_id in newly_dropped_assets:
            logger.debug(f"Asset dropped: {asset_id}.")

        # Mean-reverting random walk for asset arrivals (Gaussian CDF-based)
        active_assets = self._add_new_assets_mean_reverting(active_assets)

        # progress time
        current_time = self.time + 1

        # check horizon
        if current_time >= self.horizon:
            # Game ended return game state with status ended and reason
            logger.debug(f"GAME ENDED: {GameEndReason.HORIZON_REACHED.value}")
            final_state = GameState(
                id=self.id,
                cash=current_cash,
                time=current_time,
                horizon=self.horizon,
                equilibrium_num_assets=self.equilibrium_num_assets,
                max_num_assets=self.max_num_assets,
                asset_arrival_sensitivity_below=self.asset_arrival_sensitivity_below,
                asset_arrival_sensitivity_above=self.asset_arrival_sensitivity_above,
                reinvestment_percentage=self.reinvestment_percentage,
                initial_cash=self.initial_cash,
                assets=active_assets,
                failed_assets={
                    **self.failed_assets,
                    **newly_failed_assets,
                },
                expired_assets={
                    **self.expired_assets,
                    **newly_expired_assets,
                },
                dropped_assets={
                    **self.dropped_assets,
                    **newly_dropped_assets,
                },
                realised_costs=self.realised_costs + [current_realised_cost],
                realised_revenues=self.realised_revenues + [current_realised_revenue],
                running_enpv=self.running_enpv,
                running_eroi=self.running_eroi,
                game_ended=True,
                ended_reason=GameEndReason.HORIZON_REACHED,
                ta_experience=updated_ta_experience,
                capacity_used=new_capacity_used,
                capacity_base=self.capacity_base,
                ta_quality_estimates=self.ta_quality_estimates.copy(),
                ta_quality_confidences=self.ta_quality_confidences.copy(),
                operational_sites=self.operational_sites,
                sites_in_development=list(self.sites_in_development),
            )
            final_state._uncertain_ptrs_config = self._uncertain_ptrs_config
            final_state._investment_levels_config = self._investment_levels_config
            final_state._interim_trial_observations_config = (
                self._interim_trial_observations_config
            )
            final_state._distributional_ptrs_config = self._distributional_ptrs_config
            final_state._rd_capacity_config = self._rd_capacity_config
            final_state._drop_action_config = self._drop_action_config
            final_state._ptrs_readings_config = self._ptrs_readings_config
            final_state._marketing_config = self._marketing_config
            final_state._brand_scores = new_brand_scores
            final_state._brand_score_floors = dict(self._brand_score_floors)
            final_state._clinical_sites_config = self._clinical_sites_config
            final_state._ta_quality_modifiers = self._ta_quality_modifiers.copy()
            final_state._bd_asset_clones = {}
            # Note: decay is applied at trial completion time (already done above)
            final_state._update_ptrs_for_experience()
            return final_state._post_init_update_enpv_eroi()

        # otherwise return new game state
        logger.debug(f"COMPLETED STEP: {current_time} out of {self.horizon}")
        new_game_state = GameState(
            id=self.id,
            cash=current_cash,
            time=current_time,
            horizon=self.horizon,
            equilibrium_num_assets=self.equilibrium_num_assets,
            max_num_assets=self.max_num_assets,
            asset_arrival_sensitivity_below=self.asset_arrival_sensitivity_below,
            asset_arrival_sensitivity_above=self.asset_arrival_sensitivity_above,
            reinvestment_percentage=self.reinvestment_percentage,
            initial_cash=self.initial_cash,
            assets=active_assets,
            failed_assets={
                **self.failed_assets,
                **newly_failed_assets,
            },
            expired_assets={
                **self.expired_assets,
                **newly_expired_assets,
            },
            dropped_assets={
                **self.dropped_assets,
                **newly_dropped_assets,
            },
            realised_costs=self.realised_costs + [current_realised_cost],
            realised_revenues=self.realised_revenues + [current_realised_revenue],
            running_enpv=self.running_enpv,
            running_eroi=self.running_eroi,
            game_ended=False,
            ended_reason=None,
            ta_experience=updated_ta_experience,
            capacity_used=new_capacity_used,
            capacity_base=self.capacity_base,
            ta_quality_estimates=self.ta_quality_estimates.copy(),
            ta_quality_confidences=self.ta_quality_confidences.copy(),
            operational_sites=self.operational_sites,
            sites_in_development=list(self.sites_in_development),
        )
        new_game_state._asset_generator = self._asset_generator
        new_game_state._ta_experience_config = self._ta_experience_config
        new_game_state._uncertain_ptrs_config = self._uncertain_ptrs_config
        new_game_state._investment_levels_config = self._investment_levels_config
        new_game_state._interim_trial_observations_config = (
            self._interim_trial_observations_config
        )
        new_game_state._distributional_ptrs_config = self._distributional_ptrs_config
        new_game_state._rd_capacity_config = self._rd_capacity_config
        new_game_state._drop_action_config = self._drop_action_config
        new_game_state._ptrs_readings_config = self._ptrs_readings_config
        new_game_state._marketing_config = self._marketing_config
        new_game_state._brand_scores = new_brand_scores
        new_game_state._brand_score_floors = dict(self._brand_score_floors)
        new_game_state._clinical_sites_config = self._clinical_sites_config
        new_game_state._ta_quality_modifiers = self._ta_quality_modifiers.copy()
        new_game_state._bd_asset_clones = dict(self._bd_asset_clones)

        # Apply uncertain PTRS mechanics: update PTRS values based on experience
        # Note: decay is now applied at trial completion time, not every step
        new_game_state._update_ptrs_for_experience()

        return new_game_state._post_init_update_enpv_eroi()
