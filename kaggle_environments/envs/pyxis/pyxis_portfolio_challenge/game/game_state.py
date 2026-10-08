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
    ClinicalSitesConfig,
    DropActionConfig,
    MarketingConfig,
    PtrsReadingsConfig,
    fibonacci_cost,
)
from pyxis_portfolio_challenge.game.asset import AssetState, DrugAsset
from pyxis_portfolio_challenge.game.asset_generators import (
    AssetGeneratorBase,
)
from pyxis_portfolio_challenge.game.clinical_sites import resolve_site_grants
from pyxis_portfolio_challenge.game.constants import MAX_NUM_ASSETS, InvestmentAction
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

    # Clinical sites feature: per-agent concurrency capacity (Model B).
    # operational_sites host trials now; sites_in_development are build-delay
    # timers (steps remaining) for purchased-but-not-yet-usable sites. Occupied
    # sites are derived from the live InDevelopment asset count, so free sites =
    # operational_sites - (# InDevelopment assets).
    operational_sites: int = 0
    sites_in_development: list[int] = []

    _asset_generator: AssetGeneratorBase = PrivateAttr()
    _drop_action_config: Optional[object] = PrivateAttr(default=None)
    _ptrs_readings_config: Optional[object] = PrivateAttr(default=None)
    _clinical_sites_config: Optional[object] = PrivateAttr(default=None)
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
        drop_action_config: DropActionConfig,
        marketing_config: MarketingConfig,
        clinical_sites_config: ClinicalSitesConfig,
        ptrs_readings_config: PtrsReadingsConfig,
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
        drop_action_config : DropActionConfig
            Configuration for the standalone drop action feature.
        marketing_config : MarketingConfig
            Configuration for the marketing feature.
        clinical_sites_config : ClinicalSitesConfig
            Configuration for the clinical sites feature.
        ptrs_readings_config : PtrsReadingsConfig
            Configuration for the PTRS readings feature.
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

        # Pass the asset-generator-relevant configs. Remaining generator args
        # (assets_dir / assets_data_list, indication_spread, indication_drift_speed,
        # trial_cost_multiplier, indications_per_ta, generator_index) flow through
        # asset_generator_kwargs.
        asset_generator = asset_generator_cls(
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
            operational_sites=initial_operational_sites,
            sites_in_development=[],
        )
        # Initialise asset generator after since it is private attribute
        game_state._asset_generator = asset_generator
        game_state._drop_action_config = drop_action_config
        game_state._ptrs_readings_config = ptrs_readings_config
        game_state._marketing_config = marketing_config
        game_state._brand_scores = {}
        game_state._brand_score_floors = {}
        game_state._clinical_sites_config = clinical_sites_config
        game_state._bd_asset_clones = {}
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
                new_asset = self._asset_generator(
                    1,
                    "new",
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
            operational_sites=self.operational_sites,
            sites_in_development=list(self.sites_in_development),
        )
        ended_state._drop_action_config = self._drop_action_config
        ended_state._ptrs_readings_config = self._ptrs_readings_config
        ended_state._marketing_config = self._marketing_config
        ended_state._brand_scores = dict(self._brand_scores)
        ended_state._brand_score_floors = dict(self._brand_score_floors)
        ended_state._clinical_sites_config = self._clinical_sites_config
        ended_state._bd_asset_clones = {}
        return ended_state

    def step(
        self,
        investor_actions: dict[uuid.UUID, InvestmentAction | Literal["invest"] | None],
        market_shares: dict[uuid.UUID, float] | None = None,
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
            investor_actions: Mapping of asset IDs to investment actions.
             - InvestmentAction.NONE: Do not invest (for idle assets)
             - InvestmentAction.INVEST / "invest": Invest in an idle asset
             - InvestmentAction.STOP: Stop an in-development trial
             - InvestmentAction.DROP / "drop": Voluntarily drop an asset
            market_shares : dict[uuid.UUID, float] | None
             Optional per-drug market share multipliers (0.0-1.0).
             Used by multi-agent environment for revenue competition.
             If None, full revenue is collected (single-agent default).
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

        # Convert string actions to InvestmentAction
        normalized_actions: dict[uuid.UUID, InvestmentAction] = {}
        for asset_id, action in investor_actions.items():
            if action == "invest":
                normalized_actions[asset_id] = InvestmentAction.INVEST
            elif action == "drop":
                normalized_actions[asset_id] = InvestmentAction.DROP
            elif isinstance(action, InvestmentAction):
                normalized_actions[asset_id] = action
            # None or missing = no action

        logger.debug("Current game state before step:")
        logger.debug(f"  Cash: {self.cash}")
        logger.debug(f"  Time: {self.time}, Assets: {len(self.assets)}")

        current_cash = self.cash
        current_realised_cost = 0.0
        current_realised_revenue = 0.0

        # A (Pt.1): pay for ongoing investments
        logger.debug("Step A (Pt.1): paying for ongoing investments")
        in_dev_assets = self.in_development_assets()
        for in_dev_asset in in_dev_assets.values():
            # Check if there's an action for this asset
            new_action = normalized_actions.get(in_dev_asset.id)

            # If STOP or DROP is requested, don't pay costs
            # (asset will be stopped/dropped in Pt.2)
            if new_action in (InvestmentAction.STOP, InvestmentAction.DROP):
                logger.debug(
                    f"Skipping cost for {in_dev_asset.name} (will be stopped/dropped)"
                )
                continue

            cost = in_dev_asset.cost_this_step

            logger.debug(
                f"Paying for ongoing: {in_dev_asset.name}, cost={cost:.2f}"
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
                not in (None, InvestmentAction.NONE, InvestmentAction.DROP,
                        InvestmentAction.STOP)
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

        # A (Pt.2): pay for new investments and apply stop/drop actions
        logger.debug("Step A (Pt.2): paying for new investments")
        for asset_id, action in normalized_actions.items():
            if action == InvestmentAction.NONE:
                continue

            asset = assets_for_step[asset_id]

            if action == InvestmentAction.DROP:
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
                invested_asset = asset.to_develop()
                cost = invested_asset.cost_this_step

                logger.debug(
                    f"New investment: {asset.name}, cost={cost:.2f}"
                )
                current_cash -= cost
                current_realised_cost += cost
                assets_for_step[asset_id] = invested_asset

            elif asset.state == AssetState.InDevelopment:
                # Handle in-development assets
                if action == InvestmentAction.STOP:
                    # Stop development early (agent decides to abandon)
                    assets_for_step[asset_id] = asset.stop_development()
                    logger.debug(f"Stopped development: {asset.name}")
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
                effective_revenue = asset.revenue_this_step * share * demand_mult
                cash_collected = effective_revenue * self.reinvestment_percentage
                logger.debug(
                    f"Collecting revenue: {asset.name}, "
                    f"revenue={asset.revenue_this_step}, "
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
            # Brand equity: update scores and deduct costs. On-market drugs only:
            # a pre-launch score is reset to the floor at launch, so the spend
            # would buy nothing. The mask forbids it; this keeps it free if sent.
            if brand_equity_actions is not None:
                for asset_id, asset in assets_for_step.items():
                    if asset.state != AssetState.OnMarket:
                        continue
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

        evolved_assets = {}
        for asset_id, asset in assets_for_step.items():
            if asset.state == AssetState.Dropped:
                evolved_assets[asset_id] = asset
                continue
            evolved_assets[asset_id] = asset.evolve()

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
                operational_sites=self.operational_sites,
                sites_in_development=list(self.sites_in_development),
            )
            final_state._drop_action_config = self._drop_action_config
            final_state._ptrs_readings_config = self._ptrs_readings_config
            final_state._marketing_config = self._marketing_config
            final_state._brand_scores = new_brand_scores
            final_state._brand_score_floors = dict(self._brand_score_floors)
            final_state._clinical_sites_config = self._clinical_sites_config
            final_state._bd_asset_clones = {}
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
            operational_sites=self.operational_sites,
            sites_in_development=list(self.sites_in_development),
        )
        new_game_state._asset_generator = self._asset_generator
        new_game_state._drop_action_config = self._drop_action_config
        new_game_state._ptrs_readings_config = self._ptrs_readings_config
        new_game_state._marketing_config = self._marketing_config
        new_game_state._brand_scores = new_brand_scores
        new_game_state._brand_score_floors = dict(self._brand_score_floors)
        new_game_state._clinical_sites_config = self._clinical_sites_config
        new_game_state._bd_asset_clones = dict(self._bd_asset_clones)

        return new_game_state._post_init_update_enpv_eroi()
