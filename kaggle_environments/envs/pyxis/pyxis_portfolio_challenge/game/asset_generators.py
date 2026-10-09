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
from scipy.special import expit
from scipy.special import logit as logit_fn

from pyxis_portfolio_challenge.config import (
    ApprovalPhaseConfig,
    PtrsReadingsConfig,
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


def apply_ptrs_readings_to_trial_chain(
    asset: "DrugAsset",
    ptrs_readings_config: PtrsReadingsConfig,
    rng: random.Random,
) -> None:
    """
    Initialise the ptrs_readings feature for a newly arrived asset.

    Draws each trial's hidden per-episode true PTRS as one logit-normal sample
    centred on the file PTRS (spread = sigma_ep x the phase-distance multiplier),
    stores it on _true_ptrs, then draws one initial reading per trial (using the
    same phase-distance noise) centred on that truth, and sets trial.ptrs =
    ptrs_sample_mean so the agent's first observation is a noisy estimate. The
    file PTRS is only a prior mean; the realised _true_ptrs is what readings
    estimate and what drives trial outcomes.

    The episode spread scales by the same noise_multipliers as readings so the
    memorization bound holds at every phase distance: at distance d both the
    file-value prior and one reading carry 1/(sigma_ep*m_d)^2 precision, so the
    file lookup is worth exactly one reading regardless of how far out the phase.
    """
    chain = asset.pending_trial_chain  # excludes APPROVAL
    cfg = ptrs_readings_config
    mult = list(cfg.noise_multipliers)
    if len(chain) > len(mult):
        raise ValueError(
            f"pending_trial_chain length {len(chain)} exceeds "
            f"noise_multipliers length {len(mult)}"
        )

    # Draw the per-episode realised truth on every pending trial: one logit-normal
    # sample around the file PTRS, spread by sigma_ep x phase-distance multiplier.
    for i, trial in enumerate(chain):
        file_ptrs = trial.ptrs
        sigma_i = cfg.sigma_ep * mult[i]
        trial._true_ptrs = float(
            expit(logit_fn(file_ptrs) + rng.gauss(0.0, sigma_i))
        )

    # Draw initial reading (count=1) for each trial at the appropriate noise level
    asset.initialise_ptrs_readings(
        cfg.sigma_logit_base, list(cfg.noise_multipliers), rng
    )

    # Update the observed PTRS to the noisy first reading
    for trial in chain:
        if trial.ptrs_sample_mean is not None:
            trial.ptrs = trial.ptrs_sample_mean


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
    ) -> DrugAsset:
        """
        Function that returns one DrugAsset at a time for a given stage.

        Args:
            stage: "initial" or "new"

        """
        if stage not in ["initial", "new"]:
            raise ValueError(
                f"stage must be one of ['initial', 'new'], received {stage}"
            )

        if not self.available_assets[stage]:
            self._reset_available_assets(stage=stage)

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
            asset = self._generate_single_asset(stage=stage)
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
            # Pick a random asset from assets_data_list and copy it
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
                trial = trials_json_to_trials_sequence(
                    asset_data["trials"],
                    asset_id=asset_id,
                    pending_trial_phase=asset_data["pending_trial_phase"],
                    approval_phase_config=self.approval_phase_config,
                    trial_cost_multiplier=self.trial_cost_multiplier,
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
