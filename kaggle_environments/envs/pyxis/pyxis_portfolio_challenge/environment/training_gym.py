from __future__ import annotations

import logging
import uuid
from typing import Literal, Optional, Union

import gymnasium as gym
import numpy as np
import upath

from pyxis_portfolio_challenge.config import (
    ApprovalPhaseConfig,
    ClinicalSitesConfig,
    DistributionalPtrsConfig,
    InterimTrialObservationsConfig,
    InvestmentLevelsConfig,
    MarketingConfig,
    PtrsReadingsConfig,
    TAExperienceConfig,
    UncertainPtrsConfig,
)
from pyxis_portfolio_challenge.environment.metrics import (
    EvaluationMetric,
    MetricsContext,
    collect_metrics,
)
from pyxis_portfolio_challenge.environment.obs_layout import (
    NUM_TRIAL_PHASES,
    TA_INDEX,
    TA_ORDER,
    ObsLayout,
)
from pyxis_portfolio_challenge.environment.reward import (
    Reward,
)
from pyxis_portfolio_challenge.game.asset import AssetState
from pyxis_portfolio_challenge.game.asset_generators import JSONAssetGenerator
from pyxis_portfolio_challenge.game.constants import InvestmentLevel
from pyxis_portfolio_challenge.game.game_state import GameState
from pyxis_portfolio_challenge.game.trial import TrialPhase, TrialState

logger = logging.getLogger(__name__)


class InvestmentGameEnv(gym.Env):
    """Custom Gym environment for the investment game."""

    def __init__(
        self,
        assets_dir: upath.UPath,
        equilibrium_num_assets: int,
        reinvestment_percentage: float,
        starting_cash: float,
        max_num_assets: int,
        asset_arrival_sensitivity_below: float,
        asset_arrival_sensitivity_above: float,
        horizon: int,
        reward_fn: Reward,
        shuffle_order: bool,
        mask_first_order_assets: bool,
        mask_negative_enpv_assets: bool,
        flatten_obs: bool,
        distributional_ptrs_config: Optional[DistributionalPtrsConfig],
        ta_experience_config: Optional[TAExperienceConfig],
        uncertain_ptrs_config: Optional[UncertainPtrsConfig],
        investment_levels_config: Optional[InvestmentLevelsConfig],
        interim_trial_observations_config: Optional[InterimTrialObservationsConfig],
        rd_capacity_config,
        drop_action_config,
        marketing_config: MarketingConfig,
        clinical_sites_config: ClinicalSitesConfig,
        ptrs_readings_config: PtrsReadingsConfig,
        approval_phase_config: ApprovalPhaseConfig,
        metrics: list[EvaluationMetric],
        initial_game_state: Optional[GameState],
    ):
        """
        Initialise Gym environment for the investment game.

        Parameters
        ----------
        assets_dir:  upath.UPath
            The directory containing asset data.
        equilibrium_num_assets: int
            The target number of assets (equilibrium point for random walk).
        reinvestment_percentage: float
            A scaling factor for asset max_revenue.
        starting_cash: float
            The initial cash available to the agent.
        horizon: int
            The maximum number of steps in the environment.
        max_num_assets: int
            The maximum number of assets in the environment.
        asset_arrival_sensitivity_below: float
            Controls mean reversion speed when below target (lower = faster recovery).
        asset_arrival_sensitivity_above: float
            Controls fluctuation width at/above target (higher = wider).
        reward_fn: Reward
            The reward function to use.
        shuffle_order: bool
            Whether to shuffle the order of assets.
        mask_first_order_assets:
            Whether to mask the asset investments that would take cash below zero to the
            first order.
        mask_negative_enpv_assets:
            Whether to mask assets with negative expected NPV from investment actions.
            This prevents the agent from investing in value-destroying assets.
        metrics: list[EvaluationMetric]
            List of evaluation metrics to collect.
        flatten_obs: bool
            Whether to return flattened numpy array observations for performance.
        initial_game_state: Optional[GameState]
            An optional GameState to use as the initial state. If provided, this will be
            used instead of creating a new game state. This is useful for creating
            environments from existing game states (e.g., for agent hints API).
        max_concurrent_investments: Optional[int]
            Maximum number of assets that can be in development simultaneously.
            If None (default), no limit is enforced (unlimited concurrent investments).
        cash_risk_reserve_multiplier: Optional[float]
            Multiplier for cash risk reserve based on ongoing development costs.
            Reserve = sum(ongoing_costs) * multiplier. If None (default), no
            reserve is enforced (all cash available for investment).
        distributional_ptrs_config: Optional[DistributionalPtrsConfig]
            Configuration for distributional PTRS feature. If None, feature disabled.
        ta_experience_config: Optional[TAExperienceConfig]
            Configuration for TA experience feature.
        uncertain_ptrs_config: Optional[UncertainPtrsConfig]
            Configuration for uncertain PTRS feature.
        investment_levels_config: Optional[InvestmentLevelsConfig]
            Configuration for investment levels feature.
        interim_trial_observations_config:
            Configuration for interim trial observations.
        rd_capacity_config:
            Configuration for R&D capacity constraint feature.
        drop_action_config:
            Configuration for the standalone drop action feature.
        marketing_config: MarketingConfig
            Configuration for the marketing feature. Multi-agent-only; must be
            passed disabled to the single-agent env (enabling it raises).
        clinical_sites_config: ClinicalSitesConfig
            Configuration for the clinical sites feature. Multi-agent-only; must
            be passed disabled to the single-agent env (enabling it raises).
        ptrs_readings_config: PtrsReadingsConfig
            Configuration for the PTRS readings feature. Multi-agent-only; must
            be passed disabled to the single-agent env (enabling it raises).
        approval_phase_config: ApprovalPhaseConfig
            Configuration for the approval phase feature. Multi-agent-only; must
            be passed disabled to the single-agent env (enabling it raises).

        """
        self.equilibrium_num_assets = equilibrium_num_assets
        self.starting_cash = starting_cash
        self.horizon = horizon
        self.asset_arrival_sensitivity_below = asset_arrival_sensitivity_below
        self.asset_arrival_sensitivity_above = asset_arrival_sensitivity_above
        self.reward_fn = reward_fn
        self.assets_dir = assets_dir
        self.reinvestment_percentage = reinvestment_percentage
        self.shuffle_order = shuffle_order
        self.max_num_assets = max_num_assets
        self.mask_first_order_assets = mask_first_order_assets
        self.mask_negative_enpv_assets = mask_negative_enpv_assets
        self.metrics = metrics or []
        self.flatten_obs = flatten_obs
        self.initial_game_state = initial_game_state
        self.ta_experience_config = ta_experience_config
        self.uncertain_ptrs_config = uncertain_ptrs_config
        self.investment_levels_config = investment_levels_config
        self.interim_trial_observations_config = interim_trial_observations_config
        self.rd_capacity_config = rd_capacity_config
        self.drop_action_config = drop_action_config
        self.distributional_ptrs_config = distributional_ptrs_config
        self.marketing_config = marketing_config
        self.clinical_sites_config = clinical_sites_config
        self.ptrs_readings_config = ptrs_readings_config
        self.approval_phase_config = approval_phase_config

        # The single-agent env does not implement these multi-agent-only features.
        # They must be passed (as disabled configs) but never enabled here — a
        # disabled config keeps behaviour off; an enabled one is a config error.
        unsupported_enabled = [
            name
            for name, cfg in (
                ("marketing", self.marketing_config),
                ("clinical_sites", self.clinical_sites_config),
                ("ptrs_readings", self.ptrs_readings_config),
                ("approval_phase", self.approval_phase_config),
            )
            if cfg.enabled
        ]
        if unsupported_enabled:
            raise ValueError(
                "The single-agent InvestmentGameEnv does not support the "
                f"multi-agent-only feature(s): {', '.join(unsupported_enabled)}. "
                "Disable them in the config to use the single-agent env."
            )

        if (
            self.investment_levels_config.enabled
            and self.drop_action_config.enabled
        ):
            raise ValueError(
                "investment_levels and drop_action are mutually exclusive "
                "— enable at most one."
            )

        # Initialise now but will be overwritten in reset()
        if initial_game_state is not None:
            self.game_state = initial_game_state
        else:
            self.game_state = GameState.initialise_new_game(
                asset_generator_cls=JSONAssetGenerator,
                num_assets=equilibrium_num_assets,
                max_num_assets=max_num_assets,
                cash=starting_cash,
                horizon=horizon,
                asset_arrival_sensitivity_below=asset_arrival_sensitivity_below,
                asset_arrival_sensitivity_above=asset_arrival_sensitivity_above,
                reinvestment_percentage=reinvestment_percentage,
                seed=42,
                assets_dir=assets_dir,
                ta_experience_config=ta_experience_config,
                uncertain_ptrs_config=uncertain_ptrs_config,
                investment_levels_config=investment_levels_config,
                interim_trial_observations_config=interim_trial_observations_config,  # noqa: E501
                distributional_ptrs_config=distributional_ptrs_config,
                rd_capacity_config=rd_capacity_config,
                drop_action_config=drop_action_config,
                marketing_config=marketing_config,
                clinical_sites_config=clinical_sites_config,
                ptrs_readings_config=ptrs_readings_config,
                approval_phase_config=approval_phase_config,
                indication_spread=4.0,
                indication_drift_speed=1.0,
                trial_cost_multiplier=1.0,
            )

        self.actual_asset_ids = [0] * equilibrium_num_assets  # Will be set in reset()
        self._asset_id_order = [0] * max_num_assets  # Will be set in reset()

        self._layout = ObsLayout.from_config(
            ta_experience_config=ta_experience_config,
            rd_capacity_config=rd_capacity_config,
            distributional_ptrs_config=distributional_ptrs_config,
            uncertain_ptrs_config=uncertain_ptrs_config,
            interim_trial_observations_config=interim_trial_observations_config,
        )

        self._setup_gym_spaces()

        # Pre-allocate buffer for flattened observations
        if self.flatten_obs:
            L = self._layout
            self._obs_buffer = np.zeros(
                L.global_features + self.max_num_assets * L.asset_total_features,
                dtype=np.float32,
            )

    def _setup_gym_spaces(self):
        """
        Setup the action and observation spaces for the Gym environment.

        This also creates the helper dictionary _phase_to_observation that converts a
        phase name to its corresponding observation index.

        The action space is MultiDiscrete when investment_levels is enabled (4 choices
        per asset: NONE=0, MINIMAL=1, STANDARD=2, ACCELERATED=3), otherwise MultiBinary
        for backward compatibility.
        For the observation space, Dict spaces are used for the assets and for the
        trials of a given asset (unless flatten_obs is True).
        """
        logger.debug("Setting up gym observation spaces. .")

        # Action space: MultiDiscrete(6) for investment levels, MultiDiscrete(3)
        # for drop action, or MultiBinary for the default binary case.
        if (
            self.investment_levels_config.enabled
        ):
            self.action_space = gym.spaces.MultiDiscrete(
                [len(InvestmentLevel)] * self.max_num_assets
            )
        elif (
            self.drop_action_config.enabled
        ):
            # Ternary: 0=do nothing, 1=invest, 2=drop
            self.action_space = gym.spaces.MultiDiscrete([3] * self.max_num_assets)
        else:
            self.action_space = gym.spaces.MultiBinary(self.max_num_assets)

        L = self._layout

        if self.flatten_obs:
            total_size = (
                L.global_features + self.max_num_assets * L.asset_total_features
            )
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total_size,),
                dtype=np.float32,
            )
        else:
            trial_fields = {
                "cost_remaining": gym.spaces.Box(
                    low=0, high=float("inf"), shape=(), dtype=float
                ),
                "time_remaining": gym.spaces.Box(
                    low=0, high=int(1e9), shape=(), dtype=int
                ),
                "ptrs": gym.spaces.Box(low=0, high=1, shape=(), dtype=float),
            }
            if L.distributional_ptrs_enabled:
                trial_fields["ptrs_expected"] = gym.spaces.Box(
                    low=0, high=1, shape=(), dtype=float
                )
                trial_fields["ptrs_confidence"] = gym.spaces.Box(
                    low=0, high=1, shape=(), dtype=float
                )
                trial_fields["ptrs_range_low"] = gym.spaces.Box(
                    low=0, high=1, shape=(), dtype=float
                )
                trial_fields["ptrs_range_high"] = gym.spaces.Box(
                    low=0, high=1, shape=(), dtype=float
                )
            trial_space = gym.spaces.Dict(trial_fields)

            asset_fields = {
                "max_revenue": gym.spaces.Box(
                    low=0, high=float("inf"), shape=(), dtype=float
                ),
                "time_until_max_revenue": gym.spaces.Box(
                    low=0, high=int(1e9), shape=(), dtype=int
                ),
                "time_until_patent_expiry": gym.spaces.Box(
                    low=0, high=int(1e9), shape=(), dtype=int
                ),
                "pending_trial_phase": gym.spaces.Discrete(len(TrialPhase) + 1),
                "time_on_market": gym.spaces.Box(
                    low=0, high=int(1e9), shape=(), dtype=int
                ),
                "cost_this_step": gym.spaces.Box(
                    low=0, high=float("inf"), shape=(), dtype=float
                ),
                "revenue_this_step": gym.spaces.Box(
                    low=0, high=float("inf"), shape=(), dtype=float
                ),
                "enpv": gym.spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(), dtype=float
                ),
                "eroi": gym.spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(), dtype=float
                ),
                "trials": gym.spaces.Tuple([trial_space] * len(TrialPhase)),
                "state": gym.spaces.Discrete(len(AssetState)),
                "ta_index": gym.spaces.Discrete(3),
            }
            if L.interim_obs_enabled:
                asset_fields["interim_signal"] = gym.spaces.Box(
                    low=0, high=1, shape=(), dtype=float
                )
                asset_fields["trial_progress"] = gym.spaces.Box(
                    low=0, high=1, shape=(), dtype=float
                )
            asset_space = gym.spaces.Dict(asset_fields)

            obs_fields = {
                "cash": gym.spaces.Box(
                    low=float("-inf"), high=float("inf"), shape=(1,), dtype=np.float32
                ),
                "assets": gym.spaces.Tuple([asset_space] * self.max_num_assets),
            }
            if L.ta_experience_enabled:
                obs_fields["ta_experience"] = gym.spaces.Dict({
                    ta: gym.spaces.Box(low=0, high=float("inf"), shape=(), dtype=float)
                    for ta in TA_ORDER
                })
            if L.capacity_enabled:
                obs_fields["capacity"] = gym.spaces.Dict({
                    "capacity_ratio": gym.spaces.Box(
                        low=0, high=float("inf"), shape=(), dtype=float
                    ),
                    "capacity_headroom": gym.spaces.Box(
                        low=-float("inf"), high=float("inf"), shape=(), dtype=float
                    ),
                    "success_modifier": gym.spaces.Box(
                        low=0, high=1, shape=(), dtype=float
                    ),
                })
            if L.ta_quality_enabled:
                obs_fields["ta_quality"] = gym.spaces.Dict({
                    ta: gym.spaces.Dict({
                        "estimate": gym.spaces.Box(
                            low=-1, high=1, shape=(), dtype=float
                        ),
                        "confidence": gym.spaces.Box(
                            low=0, high=1, shape=(), dtype=float
                        ),
                    })
                    for ta in TA_ORDER
                })
            self.observation_space = gym.spaces.Dict(obs_fields)

        self._phase_to_observation = {
            phase.value: phase.integer + 1 for phase in TrialPhase
        }

    @property
    def _padding_asset_obs(self):
        """Generate padding asset observation with all zero/empty values."""
        L = self._layout
        trial_pad = {"cost_remaining": 0.0, "time_remaining": 0, "ptrs": 0.0}
        if L.distributional_ptrs_enabled:
            trial_pad["ptrs_expected"] = 0.0
            trial_pad["ptrs_confidence"] = 1.0
            trial_pad["ptrs_range_low"] = 0.0
            trial_pad["ptrs_range_high"] = 0.0

        obs = {
            "max_revenue": 0.0,
            "time_until_max_revenue": 0,
            "time_until_patent_expiry": 0,
            "pending_trial_phase": 0,
            "time_on_market": 0,
            "cost_this_step": 0.0,
            "revenue_this_step": 0.0,
            "enpv": 0.0,
            "eroi": 0.0,
            "trials": tuple([dict(trial_pad) for _ in range(len(TrialPhase))]),
            "state": AssetState.Expired.integer,
            "ta_index": 0,
        }
        if L.interim_obs_enabled:
            obs["interim_signal"] = 0.0
            obs["trial_progress"] = 0.0
        return obs

    def _get_obs(self) -> Union[dict[str, Union[np.ndarray, tuple]], np.ndarray]:
        """
        Get the current observation from the environment.

        This parses the GameState object into the correct format for the defined
        observation space.

        Note assets are sorted according to self._asset_id_order, allowing actions to be
        correctly matched with corresponding assets.

        Returns
        -------
        dict[str, Union[np.ndarray, tuple]] or np.ndarray
            The current observation from the environment.  Returns flattened array
            if flatten_obs is True.

        """
        if self.flatten_obs:
            return self._get_obs_flattened()
        return self._get_obs_dict()

    def _get_obs_flattened(self) -> np.ndarray:
        """
        Get flattened observation as a single numpy array.

        Layout (conditional on enabled features):
        [cash, [ta_exp*3], [capacity*3], [ta_quality*6], asset_0..., asset_1..., ...]

        Each asset: [10 base scalars, [interim_signal, trial_progress],
                     ta_index, trial_phase_0..., trial_phase_1..., ...]

        Each trial phase: [cost_remaining, time_remaining, ptrs,
                          [ptrs_expected, ptrs_confidence,
                           ptrs_range_low, ptrs_range_high]]

        Returns
        -------
        np.ndarray
            Flattened observation array.

        """
        obs = self._obs_buffer
        L = self._layout

        # Cache layout fields as locals for hot-path performance
        asset_total = L.asset_total_features
        asset_scalar = L.asset_scalar_features
        trial_feat = L.trial_features
        dist_on = L.distributional_ptrs_enabled
        interim_on = L.interim_obs_enabled
        off_interim = L.offset_interim_signal
        off_progress = L.offset_trial_progress
        off_ta_idx = L.offset_ta_index

        # Global features
        pos = 0
        obs[pos] = self.game_state.cash
        pos += 1

        if L.ta_experience_enabled:
            for i, ta in enumerate(TA_ORDER):
                obs[pos + i] = self.game_state.ta_experience.get(ta, 0.0)
            pos += L.num_ta_exp_features

        if L.capacity_enabled:
            obs[pos] = self.game_state.capacity_ratio
            obs[pos + 1] = self.game_state.capacity_headroom
            obs[pos + 2] = self.game_state.success_modifier
            pos += L.num_capacity_features

        if L.ta_quality_enabled:
            for i, ta in enumerate(TA_ORDER):
                obs[pos + 2 * i] = self.game_state.ta_quality_estimates.get(ta, 0.0)
                obs[pos + 2 * i + 1] = self.game_state.ta_quality_confidences.get(
                    ta, 1.0
                )
            pos += L.num_ta_quality_features

        # Asset features
        offset = L.global_features
        assets = self.game_state.assets
        phase_to_obs = self._phase_to_observation
        expired_state = AssetState.Expired.integer

        for asset_id in self._asset_id_order:
            if asset_id == 0:
                obs[offset : offset + asset_total] = 0.0
                obs[offset + 9] = expired_state
                if dist_on:
                    trial_offset = offset + asset_scalar
                    for _ in TrialPhase:
                        obs[trial_offset + 4] = 1.0  # ptrs_confidence
                        trial_offset += trial_feat
            else:
                asset = assets[asset_id]
                obs[offset] = asset.max_revenue
                obs[offset + 1] = asset.time_until_max_revenue
                obs[offset + 2] = asset.time_until_patent_expiry

                if asset.state == AssetState.OnMarket:
                    obs[offset + 3] = 0
                elif asset.trial.state == TrialState.PHASE_FAILED:
                    obs[offset + 3] = 0
                else:
                    obs[offset + 3] = phase_to_obs[asset.trial.phase]

                obs[offset + 4] = asset.time_on_market
                obs[offset + 5] = asset.cost_this_step
                obs[offset + 6] = asset.revenue_this_step
                obs[offset + 7] = asset.enpv
                obs[offset + 8] = asset.eroi
                obs[offset + 9] = asset.state.integer

                if interim_on:
                    obs[offset + off_interim] = asset.interim_signal
                    obs[offset + off_progress] = asset.trial_progress

                obs[offset + off_ta_idx] = TA_INDEX.get(asset.therapeutic_area, 0)

                # Trial features
                trial_offset = offset + asset_scalar
                _trial = asset.trial
                failure_detected = False

                for phase in TrialPhase:
                    if failure_detected:
                        obs[trial_offset] = 0.0
                        obs[trial_offset + 1] = 0
                        obs[trial_offset + 2] = 0.0
                        if dist_on:
                            obs[trial_offset + 3] = 0.0
                            obs[trial_offset + 4] = 1.0
                            obs[trial_offset + 5] = 0.0
                            obs[trial_offset + 6] = 0.0
                    elif _trial and _trial.phase == phase:
                        if _trial.state == TrialState.PHASE_FAILED:
                            failure_detected = True
                        obs[trial_offset] = _trial.cost_remaining
                        obs[trial_offset + 1] = _trial.time_remaining
                        obs[trial_offset + 2] = _trial.ptrs
                        if dist_on:
                            obs[trial_offset + 3] = _trial.ptrs_expected
                            obs[trial_offset + 4] = _trial.ptrs_confidence
                            obs[trial_offset + 5] = _trial.ptrs_range_low
                            obs[trial_offset + 6] = _trial.ptrs_range_high
                        _trial = _trial.next_trial_on_success
                    else:
                        obs[trial_offset] = 0.0
                        obs[trial_offset + 1] = 0
                        obs[trial_offset + 2] = 1.0
                        if dist_on:
                            obs[trial_offset + 3] = 1.0
                            obs[trial_offset + 4] = 1.0
                            obs[trial_offset + 5] = 1.0
                            obs[trial_offset + 6] = 1.0
                    trial_offset += trial_feat

            offset += asset_total

        return obs

    def _get_obs_dict(self) -> dict[str, Union[np.ndarray, tuple]]:
        """
        Get observation as nested dictionary (original implementation).

        Returns
        -------
        dict[str, Union[np.ndarray, tuple]]
            The current observation from the environment.

        """
        L = self._layout
        dist_on = L.distributional_ptrs_enabled
        interim_on = L.interim_obs_enabled

        def _make_trial_obs(
            cost, time_rem, ptrs, ptrs_exp, ptrs_conf, ptrs_lo, ptrs_hi
        ):
            t = {"cost_remaining": cost, "time_remaining": time_rem, "ptrs": ptrs}
            if dist_on:
                t["ptrs_expected"] = ptrs_exp
                t["ptrs_confidence"] = ptrs_conf
                t["ptrs_range_low"] = ptrs_lo
                t["ptrs_range_high"] = ptrs_hi
            return t

        def _get_asset_obs(asset):
            _trial = asset.trial
            trials_obs_list = []

            failure_detected = False
            for phase in TrialPhase:
                if failure_detected:
                    trials_obs_list.append(
                        _make_trial_obs(0.0, 0, 0.0, 0.0, 1.0, 0.0, 0.0)
                    )
                    continue

                if _trial and _trial.phase == phase:
                    if _trial.state == TrialState.PHASE_FAILED:
                        failure_detected = True
                    trials_obs_list.append(
                        _make_trial_obs(
                            _trial.cost_remaining,
                            _trial.time_remaining,
                            _trial.ptrs,
                            _trial.ptrs_expected,
                            _trial.ptrs_confidence,
                            _trial.ptrs_range_low,
                            _trial.ptrs_range_high,
                        )
                    )
                    _trial = _trial.next_trial_on_success
                else:
                    trials_obs_list.append(
                        _make_trial_obs(0.0, 0, 1.0, 1.0, 1.0, 1.0, 1.0)
                    )

            if asset.state == AssetState.OnMarket:
                pending_trial_phase = 0
            elif asset.trial.state == TrialState.PHASE_FAILED:
                pending_trial_phase = 0
            else:
                pending_trial_phase = self._phase_to_observation[asset.trial.phase]

            obs = {
                "max_revenue": asset.max_revenue,
                "time_until_max_revenue": asset.time_until_max_revenue,
                "time_until_patent_expiry": asset.time_until_patent_expiry,
                "pending_trial_phase": pending_trial_phase,
                "time_on_market": asset.time_on_market,
                "cost_this_step": asset.cost_this_step,
                "revenue_this_step": asset.revenue_this_step,
                "enpv": asset.enpv,
                "eroi": asset.eroi,
                "trials": tuple(trials_obs_list),
                "state": asset.state.integer,
                "ta_index": TA_INDEX.get(asset.therapeutic_area, 0),
            }
            if interim_on:
                obs["interim_signal"] = asset.interim_signal
                obs["trial_progress"] = asset.trial_progress
            return obs

        asset_obs = []
        for asset_id in self._asset_id_order:
            if asset_id == 0:
                asset_obs.append(self._padding_asset_obs)
            else:
                asset = self.game_state.assets[asset_id]
                asset_obs.append(_get_asset_obs(asset))

        result = {
            "cash": np.array([self.game_state.cash], dtype=np.float32),
            "assets": tuple(asset_obs),
        }

        if L.ta_experience_enabled:
            result["ta_experience"] = {
                ta: self.game_state.ta_experience.get(ta, 0.0) for ta in TA_ORDER
            }
        if L.capacity_enabled:
            result["capacity"] = {
                "capacity_ratio": self.game_state.capacity_ratio,
                "capacity_headroom": self.game_state.capacity_headroom,
                "success_modifier": self.game_state.success_modifier,
            }
        if L.ta_quality_enabled:
            result["ta_quality"] = {
                ta: {
                    "estimate": self.game_state.ta_quality_estimates.get(ta, 0.0),
                    "confidence": self.game_state.ta_quality_confidences.get(ta, 1.0),
                }
                for ta in TA_ORDER
            }

        return result

    def _get_info(self):
        return {}

    def action_masks(self) -> list[list[bool]]:
        """
        Return a list of action masks for the current observation.

        This method is called by MaskablePPO, which expects a mask for each possible
        action in the action space, flattened into a list. The mask is True if
        the action is valid, False otherwise.
        (See https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html for more
        info.)

        For MultiBinary (legacy): each asset has [can_not_invest, can_invest]
        For MultiDiscrete (investment levels): each asset has
            [can_NONE, can_MINIMAL, can_STANDARD, can_ACCELERATED, can_STOP]
        For MultiDiscrete (drop action): each asset has
            [can_do_nothing, can_invest, can_drop]

        As in _get_obs, assets are sorted according to self._asset_id_order.

        Returns
        -------
        list[list[bool]]
            A list of action masks for the current observation.

        """
        use_investment_levels = (
            self.investment_levels_config.enabled
        )
        use_drop_action = (
            self.drop_action_config.enabled
        )

        if use_investment_levels:
            return self._action_masks_investment_levels()
        elif use_drop_action:
            return self._action_masks_ternary()
        return self._action_masks_binary()

    def _action_masks_binary(self) -> list[list[bool]]:
        """Return action masks for MultiBinary action space (legacy)."""
        action_mask = []
        available_cash = self.game_state.cash
        for asset_id in self._asset_id_order:
            if asset_id == 0:
                action_mask.append([True, False])
            else:
                asset = self.game_state.assets[asset_id]
                if asset.state == AssetState.Idle:
                    can_invest = True

                    # Check cash affordability (existing logic)
                    if self.mask_first_order_assets:
                        if available_cash - asset.cost_to_invest_this_step < 0:
                            can_invest = False

                    # Check eNPV threshold
                    # (prevent investing in negative expected value)
                    if self.mask_negative_enpv_assets:
                        if asset.enpv < 0:
                            can_invest = False

                    action_mask.append([True, True] if can_invest else [True, False])
                else:
                    action_mask.append([True, False])

        return action_mask

    def _action_masks_ternary(self) -> list[list[bool]]:
        """
        Return action masks for MultiDiscrete([3]*N) drop-action space.

        Per asset: [can_do_nothing, can_invest, can_drop].
        Padding slots block invest and drop. When mask_first_order_assets is set,
        both invest and drop are masked out when their cost cannot be covered by
        the cash on hand — dropping charges a fee, so it is not always free.
        """
        action_mask = []
        available_cash = self.game_state.cash
        for asset_id in self._asset_id_order:
            if asset_id == 0:
                action_mask.append([True, False, False])
                continue

            asset = self.game_state.assets[asset_id]
            can_drop = True
            if self.mask_first_order_assets:
                if available_cash - self._drop_fee(asset) < 0:
                    can_drop = False

            if asset.state == AssetState.Idle:
                can_invest = True
                if self.mask_first_order_assets:
                    if available_cash - asset.cost_to_invest_this_step < 0:
                        can_invest = False
                if self.mask_negative_enpv_assets and asset.enpv < 0:
                    can_invest = False
                action_mask.append([True, can_invest, can_drop])
            else:
                action_mask.append([True, False, can_drop])
        return action_mask

    def _drop_fee(self, asset) -> float:
        """Fee charged for dropping this asset, or 0.0 if drops are free."""
        if self.drop_action_config is None or asset.trial is None:
            return 0.0
        return self.drop_action_config.calculate_drop_fee(asset.trial.cost_remaining)

    def _action_masks_investment_levels(self) -> list[list[bool]]:
        """
        Return action masks for MultiDiscrete action space (investment levels).

        Action indices:
            0: NONE - don't invest / keep current level
            1: MINIMAL - slow and cheap development
            2: STANDARD - normal development
            3: ACCELERATED - fast and expensive development
            4: STOP - stop development early (in-development assets only)
        """
        action_mask = []
        for asset_id in self._asset_id_order:
            if asset_id == 0:
                # Padding: only NONE valid
                action_mask.append([True, False, False, False, False])
            else:
                asset = self.game_state.assets[asset_id]
                if asset.state == AssetState.Idle:
                    can_invest = True

                    # Check eNPV threshold
                    if self.mask_negative_enpv_assets and asset.enpv < 0:
                        can_invest = False

                    if can_invest:
                        # All levels valid for idle assets, but not STOP
                        action_mask.append([True, True, True, True, False])
                    else:
                        # Only NONE valid
                        action_mask.append([True, False, False, False, False])
                elif asset.state == AssetState.InDevelopment:
                    # Can change investment level (all levels valid)
                    # Can also STOP to abandon development early
                    action_mask.append([True, True, True, True, True])
                else:
                    # OnMarket, Failed, Expired: only NONE valid
                    action_mask.append([True, False, False, False, False])

        return action_mask

    def _create_shuffled_asset_order(self):
        """Create a shuffled order of asset IDs."""
        self.actual_asset_ids = list(self.game_state.assets.keys())
        padded_asset_ids = self.actual_asset_ids + [0] * (
            self.max_num_assets - len(self.actual_asset_ids)
        )
        if self.shuffle_order:
            self.np_random.shuffle(padded_asset_ids)
        self._asset_id_order = padded_asset_ids

    def _maintain_unshuffled_asset_order(self):
        """
        Maintain an unshuffled order of asset IDs.

        Assets that still exist keep their original positions.
        Expired assets are removed from the order.
        New assets are appended to the end.

        This works with the mean-reverting asset arrival mechanism where
        the number of new assets can be 0, 1, or multiple, and doesn't
        necessarily match the number of expired assets.
        """
        prev_ids = set(self.actual_asset_ids)
        curr_ids = set(self.game_state.assets.keys())

        expired = list(prev_ids - curr_ids)
        created = list(curr_ids - prev_ids)

        # Remove expired assets from the order
        for expired_id in expired:
            if expired_id in self.actual_asset_ids:
                self.actual_asset_ids.remove(expired_id)

        # Append all new assets to the end
        self.actual_asset_ids.extend(created)

        # Pad with zeros to reach max_num_assets
        padded_asset_ids = self.actual_asset_ids + [0] * (
            self.max_num_assets - len(self.actual_asset_ids)
        )
        self._asset_id_order = padded_asset_ids

    def extend_horizon_for_warmup(self, extra_steps: int) -> None:
        """
        Temporarily extend the game-state horizon by ``extra_steps``.

        Lets a warmup pre-roll run without tripping ``time >= horizon``
        termination, however long the warmup. Paired with
        :meth:`rebase_clock_after_warmup`, which rebases the clock to 0 and
        restores the configured horizon so warmup and horizon are additive.
        """
        self.game_state.horizon = self.game_state.horizon + extra_steps

    def rebase_clock_after_warmup(self) -> None:
        """
        Rebase the clock to 0 and restore the configured horizon.

        Treats the warmup steps as a pre-roll: the agent then plays a full
        ``horizon`` starting at time 0 (see :meth:`extend_horizon_for_warmup`).
        """
        self.game_state.rebase_time_to_zero()
        self.game_state.horizon = self.horizon

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[Union[dict[str, Union[np.ndarray, tuple]], np.ndarray], dict]:
        """
        Reset the environment to an initial state, randomly.

        If "preserve_game_state" is set to True in options, the game state will not be
        reset. This is primarily used by GymAgent in agent. py to allow accessing the
        flattened observation of the GameState instance it is acting on.

        If an initial_game_state was provided during initialization, it will be used
        unless a seed is provided or preserve_game_state is True.

        Parameters
        ----------
        seed:  Optional[int]
            Random seed for initialization.
        options: Optional[dict]
            Additional options for resetting the environment, including
            "preserve_game_state":  Whether to preserve the current game state.

        Returns
        -------
        observation: dict[str, Union[np.ndarray, tuple]] or np.ndarray
            The initial observation from the environment.
        info: dict
            Additional information from the environment.

        """
        super().reset(seed=seed)

        preserve_state = options and options.get("preserve_game_state", False)

        if not preserve_state:
            # If an initial_game_state was provided and no seed is given, use it
            if self.initial_game_state is not None and seed is None:
                self.game_state = self.initial_game_state
            else:
                self.game_state = GameState.initialise_new_game(
                    asset_generator_cls=JSONAssetGenerator,
                    num_assets=self.equilibrium_num_assets,
                    max_num_assets=self.max_num_assets,
                    cash=self.starting_cash,
                    horizon=self.horizon,
                    asset_arrival_sensitivity_below=self.asset_arrival_sensitivity_below,
                    asset_arrival_sensitivity_above=self.asset_arrival_sensitivity_above,
                    reinvestment_percentage=self.reinvestment_percentage,
                    seed=seed,
                    **{
                        "assets_dir": self.assets_dir,
                        "ta_experience_config": self.ta_experience_config,
                        "uncertain_ptrs_config": self.uncertain_ptrs_config,
                        "investment_levels_config": self.investment_levels_config,
                        "interim_trial_observations_config": (
                            self.interim_trial_observations_config
                        ),
                        "distributional_ptrs_config": self.distributional_ptrs_config,
                        "rd_capacity_config": self.rd_capacity_config,
                        "drop_action_config": self.drop_action_config,
                        "marketing_config": self.marketing_config,
                        "clinical_sites_config": self.clinical_sites_config,
                        "ptrs_readings_config": self.ptrs_readings_config,
                        "approval_phase_config": self.approval_phase_config,
                        "indication_spread": 4.0,
                        "indication_drift_speed": 1.0,
                        "trial_cost_multiplier": 1.0,
                    },
                )

        self._create_shuffled_asset_order()

        observation = self._get_obs()
        info = self._get_info()

        self._episode_fingerprint = self.game_state.content_fingerprint(seed)
        ctx = MetricsContext(
            self.game_state, reward=0.0, episode_id=self._episode_fingerprint
        )
        collect_metrics(
            collection_fn="on_episode_begin", context=ctx, metrics=self.metrics
        )

        return observation, info

    def _action_to_investment_decision(
        self, action
    ) -> dict[uuid.UUID, Union[InvestmentLevel, Literal["invest"], None]]:
        """
        Convert the action vector to investment decisions.

        For MultiBinary (legacy):
            If the action is 1, use "invest" (backward compatible).
            If the action is 0, exclude from dictionary.

        For MultiDiscrete (investment levels):
            Action 0 = NONE (exclude from dictionary)
            Action 1 = MINIMAL
            Action 2 = STANDARD
            Action 3 = ACCELERATED

        For MultiDiscrete (drop action):
            Action 0 = do nothing (exclude from dictionary)
            Action 1 = "invest" (standard investment)
            Action 2 = "drop" (remove asset from portfolio)

        As in _get_obs, assets are sorted according to self._asset_id_order.

        Parameters
        ----------
        action: np.ndarray
            The action vector from the agent.

        Returns
        -------
        dict
            A dictionary mapping asset_ids to investment decisions.

        """
        use_investment_levels = (
            self.investment_levels_config.enabled
        )
        use_drop_action = (
            self.drop_action_config.enabled
        )

        investment_decisions = {}
        for asset_id, act in zip(self._asset_id_order, action):
            if asset_id == 0:
                continue

            if use_investment_levels:
                level = InvestmentLevel.from_int(int(act))
                if level != InvestmentLevel.NONE:
                    investment_decisions[asset_id] = level
            elif use_drop_action:
                if act == 1:
                    investment_decisions[asset_id] = "invest"
                elif act == 2:
                    investment_decisions[asset_id] = "drop"
            else:
                if act == 1:
                    investment_decisions[asset_id] = "invest"

        return investment_decisions

    def action_masks_binary(self) -> np.ndarray:
        """
        Return a flattened binary action mask for the current observation.

        This method flattens the list of action masks into a single binary array.
        This is useful for certain RL algorithms that require a flat mask.
        """
        both_valid = np.array(self.action_masks()).all(axis=1)
        return both_valid.astype(int)

    def validate_action(self, action: np.ndarray) -> None:
        """
        Validate the action against the current action masks.

        Parameters
        ----------
        action: np.ndarray
            The action vector from the agent.

        """
        if action.shape[0] != self.max_num_assets:
            raise ValueError(
                f"Action must have length equal to input size:  {self.max_num_assets}"
            )

        valid = True
        action_masks = self.action_masks()  # List of [bool, bool, ...] per asset

        for act, mask in zip(action, action_masks):
            act_idx = int(act)
            if act_idx < 0 or act_idx >= len(mask):
                valid = False
                break
            if not mask[act_idx]:
                valid = False
                break

        if not valid:
            raise ValueError("Action invalid according to action masks.")

    def step(
        self, action
    ) -> tuple[
        Union[dict[str, Union[np.ndarray, tuple]], np.ndarray], float, bool, bool, dict
    ]:
        """
        Take a step in the InvestmentGameEnv Gym environment.

        This method converts the action vector returned by the agent to an investment
        decisions dictionary, and uses this to step the GameState instance.
        The reward is the change in net present value (NPV) of the game state.
        FUTURE:  more sophisticated reward functions may be implemented.

        The return objects follow the standard Gym API.

        Parameters
        ----------
        action: np.ndarray
            The action vector from the agent.

        Returns
        -------
        observation: dict[str, Union[np.ndarray, tuple]] or np.ndarray
            The observation from the environment after taking the step.
        reward: float
            The reward from the environment after taking the step.
        terminated: bool
            Whether the episode has terminated, True if game has ended, otherwise False.
        truncated: bool
            Currently always False and never used.
        info: dict
            Additional information from the environment.

        """
        if action.shape[0] != self.max_num_assets:
            raise ValueError(
                f"Action must have length equal to input size:  {self.max_num_assets}"
            )

        self.validate_action(action)

        pre_step_game_state = self.game_state
        investment_decisions = self._action_to_investment_decision(action)

        episode_id = getattr(self, "_episode_fingerprint", None)
        metrics_ctx = MetricsContext(
            pre_step_game_state, reward=0.0, investment_decisions=investment_decisions,
            episode_id=episode_id,
        )
        collect_metrics(
            collection_fn="on_step_begin", context=metrics_ctx, metrics=self.metrics
        )

        self.game_state = self.game_state.step(investment_decisions)

        terminated = self.game_state.game_ended

        truncated = False

        if self.shuffle_order:
            self._create_shuffled_asset_order()
        else:
            self._maintain_unshuffled_asset_order()

        observation = self._get_obs()
        info = self._get_info()

        reward = self.reward_fn.compute(
            pre_step_game_state=pre_step_game_state,
            post_step_game_state=self.game_state,
            investment_decisions=investment_decisions,
        )

        metrics_ctx = MetricsContext(
            self.game_state, reward=reward, investment_decisions=investment_decisions,
            episode_id=episode_id,
        )
        collect_metrics(
            collection_fn="on_step_end", context=metrics_ctx, metrics=self.metrics
        )

        if terminated:
            collect_metrics(
                collection_fn="on_episode_end",
                context=metrics_ctx,
                metrics=self.metrics,
            )

        return observation, reward, terminated, truncated, info

    def flatten_dict_obs(self, dict_obs: dict) -> np.ndarray:
        """
        Convert a dictionary observation to flattened array format.

        Parameters
        ----------
        dict_obs: dict
            Dictionary observation from _get_obs_dict().

        Returns
        -------
        np.ndarray
            Flattened observation array.

        """
        L = self._layout
        num_assets = len(dict_obs["assets"])
        total_size = L.global_features + num_assets * L.asset_total_features
        obs = np.zeros(total_size, dtype=np.float32)

        pos = 0
        obs[pos] = dict_obs["cash"][0]
        pos += 1

        if L.ta_experience_enabled:
            ta_experience = dict_obs.get("ta_experience", {})
            for i, ta in enumerate(TA_ORDER):
                obs[pos + i] = ta_experience.get(ta, 0.0)
            pos += L.num_ta_exp_features

        if L.capacity_enabled:
            capacity = dict_obs.get("capacity", {})
            obs[pos] = capacity.get("capacity_ratio", 0.0)
            obs[pos + 1] = capacity.get("capacity_headroom", 1.0)
            obs[pos + 2] = capacity.get("success_modifier", 1.0)
            pos += L.num_capacity_features

        if L.ta_quality_enabled:
            ta_quality = dict_obs.get("ta_quality", {})
            for i, ta in enumerate(TA_ORDER):
                ta_data = ta_quality.get(ta, {"estimate": 0.0, "confidence": 1.0})
                obs[pos + 2 * i] = ta_data.get("estimate", 0.0)
                obs[pos + 2 * i + 1] = ta_data.get("confidence", 1.0)
            pos += L.num_ta_quality_features

        offset = L.global_features
        for asset_obs_d in dict_obs["assets"]:
            obs[offset] = asset_obs_d["max_revenue"]
            obs[offset + 1] = asset_obs_d["time_until_max_revenue"]
            obs[offset + 2] = asset_obs_d["time_until_patent_expiry"]
            obs[offset + 3] = asset_obs_d["pending_trial_phase"]
            obs[offset + 4] = asset_obs_d["time_on_market"]
            obs[offset + 5] = asset_obs_d["cost_this_step"]
            obs[offset + 6] = asset_obs_d["revenue_this_step"]
            obs[offset + 7] = asset_obs_d["enpv"]
            obs[offset + 8] = asset_obs_d["eroi"]
            obs[offset + 9] = asset_obs_d["state"]

            if L.interim_obs_enabled:
                obs[offset + L.offset_interim_signal] = asset_obs_d.get(
                    "interim_signal", 0.0
                )
                obs[offset + L.offset_trial_progress] = asset_obs_d.get(
                    "trial_progress", 0.0
                )

            obs[offset + L.offset_ta_index] = asset_obs_d.get("ta_index", 0)

            trial_offset = offset + L.asset_scalar_features
            for trial in asset_obs_d["trials"]:
                obs[trial_offset] = trial["cost_remaining"]
                obs[trial_offset + 1] = trial["time_remaining"]
                obs[trial_offset + 2] = trial["ptrs"]
                if L.distributional_ptrs_enabled:
                    obs[trial_offset + 3] = trial.get("ptrs_expected", trial["ptrs"])
                    obs[trial_offset + 4] = trial.get("ptrs_confidence", 1.0)
                    obs[trial_offset + 5] = trial.get("ptrs_range_low", trial["ptrs"])
                    obs[trial_offset + 6] = trial.get("ptrs_range_high", trial["ptrs"])
                trial_offset += L.trial_features

            offset += L.asset_total_features

        return obs

    def unflatten_to_dict_obs(
        self, flat_obs: np.ndarray, num_assets: int
    ) -> dict[str, Union[np.ndarray, tuple]]:
        """
        Convert a flattened observation array to dictionary format.

        Parameters
        ----------
        flat_obs: np.ndarray
            Flattened observation array.
        num_assets: int
            Number of assets in the observation.

        Returns
        -------
        dict
            Dictionary observation.

        """
        L = self._layout
        pos = 0

        result = {"cash": np.array([flat_obs[pos]], dtype=np.float32), "assets": []}
        pos += 1

        if L.ta_experience_enabled:
            result["ta_experience"] = {
                ta: float(flat_obs[pos + i]) for i, ta in enumerate(TA_ORDER)
            }
            pos += L.num_ta_exp_features

        if L.capacity_enabled:
            result["capacity"] = {
                "capacity_ratio": float(flat_obs[pos]),
                "capacity_headroom": float(flat_obs[pos + 1]),
                "success_modifier": float(flat_obs[pos + 2]),
            }
            pos += L.num_capacity_features

        offset = L.global_features
        for _ in range(num_assets):
            trials = []
            trial_offset = offset + L.asset_scalar_features
            for _ in range(NUM_TRIAL_PHASES):
                trials.append({
                    "cost_remaining": float(flat_obs[trial_offset]),
                    "time_remaining": int(flat_obs[trial_offset + 1]),
                    "ptrs": float(flat_obs[trial_offset + 2]),
                })
                trial_offset += L.trial_features

            asset_obs_d = {
                "max_revenue": float(flat_obs[offset]),
                "time_until_max_revenue": int(flat_obs[offset + 1]),
                "time_until_patent_expiry": int(flat_obs[offset + 2]),
                "pending_trial_phase": int(flat_obs[offset + 3]),
                "time_on_market": int(flat_obs[offset + 4]),
                "cost_this_step": float(flat_obs[offset + 5]),
                "revenue_this_step": float(flat_obs[offset + 6]),
                "enpv": float(flat_obs[offset + 7]),
                "eroi": float(flat_obs[offset + 8]),
                "state": int(flat_obs[offset + 9]),
                "ta_index": int(flat_obs[offset + L.offset_ta_index]),
                "trials": tuple(trials),
            }
            if L.interim_obs_enabled:
                asset_obs_d["interim_signal"] = float(
                    flat_obs[offset + L.offset_interim_signal]
                )
                asset_obs_d["trial_progress"] = float(
                    flat_obs[offset + L.offset_trial_progress]
                )
            result["assets"].append(asset_obs_d)
            offset += L.asset_total_features

        result["assets"] = tuple(result["assets"])
        return result
