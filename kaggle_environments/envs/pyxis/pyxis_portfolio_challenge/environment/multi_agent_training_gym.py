"""Multi-agent competitive investment game environment using PettingZoo."""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import upath
from pettingzoo import ParallelEnv

from pyxis_portfolio_challenge.config import (
    ApprovalPhaseConfig,
    ClinicalSitesConfig,
    DistributionalPtrsConfig,
    InterimTrialObservationsConfig,
    InvestmentLevelsConfig,
    MarketingConfig,
    PricingConfig,
    PtrsReadingsConfig,
    TAExperienceConfig,
    UncertainPtrsConfig,
)
from pyxis_portfolio_challenge.environment.obs_layout import TA_INDEX, ObsLayout
from pyxis_portfolio_challenge.environment.reward import Reward
from pyxis_portfolio_challenge.game.asset import AssetState
from pyxis_portfolio_challenge.game.constants import InvestmentLevel
from pyxis_portfolio_challenge.game.game_state import GameState
from pyxis_portfolio_challenge.game.multi_agent_game import MultiAgentGame
from pyxis_portfolio_challenge.game.shared_market_state import (
    THERAPEUTIC_AREAS,
    AlertType,
    indication_key,
)
from pyxis_portfolio_challenge.game.trial import TrialPhase, TrialState

logger = logging.getLogger(__name__)

# Cash value of one BD bid unit: the continuous BD action is expressed in GBP
# millions, so a raw action value of 1.0 corresponds to a £1M cash bid.
_BD_BID_UNIT = 1_000_000.0

# Cash value of one clinical-site auction bid unit. Like BD, the continuous
# site_bid action is expressed in GBP millions.
_SITE_BID_UNIT = 1_000_000.0

# BD observation features: available + 9 asset details + eroi = 11 per slot
_BD_OBS_FEATURES_PER_SLOT = 11
# Non-approval trial phases tracked in BD ptrs readings obs (Ph1, Ph2, Ph3)
_BD_NUM_NON_APPROVAL_PHASES = 3
# Public ask-price anchor, appended only when ptrs_readings is enabled. The BD
# auction prices every level off the *shared* asset's cash eNPV, which stays
# frozen at the public first reading — but once an agent commissions readings
# its obs eNPV comes from its private clone, hiding the anchor. This feature
# keeps it observable so the agent can compute price(level).
_BD_ASK_CASH_ENPV_IDX = _BD_OBS_FEATURES_PER_SLOT
# Per-phase (ptrs_mean, equiv_n_norm) pairs start after the ask anchor
_BD_PHASE_FEATURES_IDX = _BD_OBS_FEATURES_PER_SLOT + 1

# Indication market features (per indication slot)
_INDICATION_FEATURES = 5  # exclusivity, share, first_mover, my_drugs, competitor_drugs

# Alert features
# event_type one-hot (6: drug_release, bd_deal, pipeline_leak, clinical_site_deal,
# be_spend, dc_spend) + agent_idx + ta_idx + indication + age + phase + deal_price
# + be_count (number of leaked BE spends in the indication; 0 for other types)
_ALERT_FEATURES = 13


class MultiAgentInvestmentGameEnv(ParallelEnv):
    """
    Multi-agent competitive investment game environment.

    Uses MultiAgentGame to orchestrate N GameState instances.
    All single-player dynamics come from GameState.step().
    Cross-agent interactions (BD deals, market share, alerts)
    are handled by MultiAgentGame.
    """

    metadata = {"render_modes": ["human"], "name": "multi_agent_investment_game"}

    def __init__(
        self,
        assets_dir: upath.UPath,
        num_agents: int,
        starting_cash: float,
        max_num_assets: int,
        horizon: int,
        equilibrium_num_assets: int,
        asset_arrival_sensitivity_below: float,
        asset_arrival_sensitivity_above: float,
        reinvestment_percentage: float,
        # BD market parameters (Poisson-distributed, event-driven)
        bd_enabled: bool,
        bd_assets_dir: upath.UPath,
        bd_base_lambda: float,
        bd_leak_lambda_boost: float,
        bd_min_step: int,
        bd_max_bid: float,
        bd_max_slots: int,
        bd_phase_weights: list[float],
        bd_indication_activity_bias: float,
        # Competition parameters
        exclusivity_period: int,
        first_mover_bonus: float,
        disable_market_share_competition: bool,
        # Intelligence parameters (event-driven leaks)
        alert_history_length: int,
        leak_phase_probabilities: list[float],
        be_leak_probability: float,
        dc_leak_probability: float,
        alerts_per_agent: int,
        # Reward parameters
        reward_fn: Reward,
        # Masking and ordering
        shuffle_order: bool,
        mask_first_order_assets: bool,
        mask_negative_enpv_assets: bool,
        # Feature configs
        flatten_obs: bool,
        distributional_ptrs_config: DistributionalPtrsConfig,
        ta_experience_config: TAExperienceConfig,
        uncertain_ptrs_config: UncertainPtrsConfig,
        investment_levels_config: InvestmentLevelsConfig,
        interim_trial_observations_config: InterimTrialObservationsConfig,
        rd_capacity_config,
        drop_action_config,
        approval_phase_config: ApprovalPhaseConfig,
        # Pricing configuration
        pricing_config: PricingConfig,
        # Multi-agent reward type
        reward_type: str,
        reward_scale: float,
        # Indication-based market segmentation
        max_indications_per_ta: int,
        target_drugs_per_indication: float,
        on_market_fraction: float,
        indication_spread: float,
        indication_drift_speed: float,
        trial_cost_multiplier: float,
        # Congestion penalty
        congestion_exponent: float,
        congestion_ramp_steps: int,
        congestion_incumbent_penalty: float,
        render_mode: Optional[str],
        marketing_config: MarketingConfig,
        ptrs_readings_config: PtrsReadingsConfig,
        clinical_sites_config: ClinicalSitesConfig,
        bd_persist_steps: int,
        dc_leak_min_agents: int,
    ):
        """Initialize multi-agent investment game environment."""
        super().__init__()

        # Store configuration for reset
        self._num_agents = num_agents
        self.assets_dir = assets_dir
        self.starting_cash = starting_cash
        self.max_num_assets = max_num_assets
        self.horizon = horizon
        self.equilibrium_num_assets = equilibrium_num_assets
        self.asset_arrival_sensitivity_below = asset_arrival_sensitivity_below
        self.asset_arrival_sensitivity_above = asset_arrival_sensitivity_above
        self.reinvestment_percentage = reinvestment_percentage

        # BD configuration
        self.bd_enabled = bd_enabled
        self.bd_assets_dir = bd_assets_dir
        self.bd_base_lambda = bd_base_lambda
        self.bd_leak_lambda_boost = bd_leak_lambda_boost
        self.bd_min_step = bd_min_step
        self.bd_max_bid = bd_max_bid
        self.bd_max_slots = bd_max_slots
        self.bd_persist_steps = bd_persist_steps
        self.bd_phase_weights = bd_phase_weights
        self.bd_indication_activity_bias = bd_indication_activity_bias

        # Competition
        self.exclusivity_period = exclusivity_period
        self.first_mover_bonus = first_mover_bonus
        self.disable_market_share_competition = disable_market_share_competition

        # Intelligence
        self.alert_history_length = alert_history_length
        self.leak_phase_probabilities = leak_phase_probabilities
        self.be_leak_probability = be_leak_probability
        self.dc_leak_probability = dc_leak_probability
        self.dc_leak_min_agents = dc_leak_min_agents
        self.alerts_per_agent = alerts_per_agent
        self.max_alerts = alerts_per_agent * max(num_agents - 1, 1)

        self.shuffle_order = shuffle_order
        self.mask_first_order_assets = mask_first_order_assets
        self.mask_negative_enpv_assets = mask_negative_enpv_assets
        self.flatten_obs = flatten_obs
        self.render_mode = render_mode

        # Feature configs
        self.distributional_ptrs_config = distributional_ptrs_config
        self.ta_experience_config = ta_experience_config
        self.uncertain_ptrs_config = uncertain_ptrs_config
        self.investment_levels_config = investment_levels_config
        self.interim_trial_observations_config = interim_trial_observations_config
        self.rd_capacity_config = rd_capacity_config
        self.drop_action_config = drop_action_config
        self.marketing_config = marketing_config
        self.ptrs_readings_config = ptrs_readings_config
        self.clinical_sites_config = clinical_sites_config
        self.approval_phase_config = approval_phase_config
        self._indication_features = (
            6 if (marketing_config.enabled) else 5
        )

        if (
            self.investment_levels_config.enabled
            and self.drop_action_config.enabled
        ):
            raise ValueError(
                "investment_levels and drop_action are mutually exclusive "
                "— enable at most one."
            )
        self.pricing_config = pricing_config

        # Reward
        self._reward_fn = reward_fn
        self._reward_type = reward_type
        self._reward_scale = reward_scale

        # Indication-based market segmentation
        self.max_indications_per_ta = max_indications_per_ta
        self.target_drugs_per_indication = target_drugs_per_indication
        self.on_market_fraction = on_market_fraction
        self.indication_spread = indication_spread
        self.indication_drift_speed = indication_drift_speed
        self.trial_cost_multiplier = trial_cost_multiplier
        self.congestion_exponent = congestion_exponent
        self.congestion_ramp_steps = congestion_ramp_steps
        self.congestion_incumbent_penalty = congestion_incumbent_penalty

        # Persistent RNG for generating per-episode seeds.
        # When reset(seed=N) is called, this is re-seeded.
        # When reset() is called without a seed, a new game seed is drawn
        # from this RNG so each episode is different.
        self._episode_rng = np.random.default_rng(42)

        # Agent names
        self.possible_agents = [f"pharma_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents.copy()

        # Will be initialized on reset
        self.multi_agent_game: Optional[MultiAgentGame] = None
        self._asset_id_orders: dict[str, list] = {}
        self._indications_per_ta: int = 0  # Set on reset
        # Per-agent pricing multipliers (asset_id -> multiplier), updated each step
        self._current_pricing: dict[str, dict[str, float]] = {}

        # Build observation layout from feature configs
        self._layout = ObsLayout.from_config(
            ta_experience_config=ta_experience_config,
            rd_capacity_config=rd_capacity_config,
            distributional_ptrs_config=distributional_ptrs_config,
            uncertain_ptrs_config=uncertain_ptrs_config,
            interim_trial_observations_config=interim_trial_observations_config,
            pricing_config=pricing_config,
            marketing_config=marketing_config,
            ptrs_readings_config=ptrs_readings_config,
            clinical_sites_config=clinical_sites_config,
            has_time_feature=True,
            has_indication_feature=True,
        )

        # BD obs: 11 base features + public ask-price anchor
        # + (ptrs_mean + equiv_n_norm) × 3 phases when ptrs_readings enabled
        bd_ptrs_readings_extras = (
            1 + _BD_NUM_NON_APPROVAL_PHASES * 2
            if ptrs_readings_config.enabled
            else 0
        )
        self._bd_obs_size = _BD_OBS_FEATURES_PER_SLOT + bd_ptrs_readings_extras

        # Calculate observation size
        L = self._layout
        self._obs_size = (
            L.global_features
            + self.max_num_assets * L.asset_total_features
            + self.bd_max_slots * self._bd_obs_size
            + len(THERAPEUTIC_AREAS)
            * self.max_indications_per_ta
            * self._indication_features
            + self.max_alerts * _ALERT_FEATURES
        )

    @property
    def agent_portfolios(self) -> dict[str, GameState]:
        """Access agent GameState instances (compatibility property)."""
        if self.multi_agent_game is None:
            return {}
        return self.multi_agent_game.agent_states

    @property
    def time(self) -> int:
        """Current game time step."""
        if self.multi_agent_game is None:
            return 0
        return self.multi_agent_game.time

    @property
    def _sites_on(self) -> bool:
        """True when the clinical-sites feature is enabled."""
        return (
            self.clinical_sites_config.enabled
        )

    @property
    def _site_auction_on(self) -> bool:
        """True when the clinical-site PvP auction is enabled."""
        return self._sites_on and self.clinical_sites_config.auction_enabled

    @property
    def _site_priority_on(self) -> bool:
        """True when per-asset site priority arbitration is enabled."""
        return self._sites_on and self.clinical_sites_config.agent_priority

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> gym.Space:
        """Return observation space for an agent."""
        if self.flatten_obs:
            return gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._obs_size,),
                dtype=np.float32,
            )

        L = self._layout

        # Trial space
        trial_fields = {
            "cost_remaining": gym.spaces.Box(
                low=0, high=float("inf"), shape=(), dtype=float
            ),
            "time_remaining": gym.spaces.Box(low=0, high=int(1e9), shape=(), dtype=int),
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

        # Asset space
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
            "time_on_market": gym.spaces.Box(low=0, high=int(1e9), shape=(), dtype=int),
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
            "indication": gym.spaces.Discrete(self.max_indications_per_ta),
        }
        if L.interim_obs_enabled:
            asset_fields["interim_signal"] = gym.spaces.Box(
                low=0, high=1, shape=(), dtype=float
            )
            asset_fields["trial_progress"] = gym.spaces.Box(
                low=0, high=1, shape=(), dtype=float
            )
        if L.pricing_enabled:
            asset_fields["price_multiplier"] = gym.spaces.Box(
                low=0, high=float("inf"), shape=(), dtype=float
            )
        asset_space = gym.spaces.Dict(asset_fields)

        # BD slot space
        bd_fields = {
            "available": gym.spaces.Discrete(2),
            "max_revenue": gym.spaces.Box(
                low=0, high=float("inf"), shape=(), dtype=float
            ),
            "time_until_max_revenue": gym.spaces.Box(
                low=0, high=int(1e9), shape=(), dtype=int
            ),
            "time_until_patent_expiry": gym.spaces.Box(
                low=0, high=int(1e9), shape=(), dtype=int
            ),
            "ta_index": gym.spaces.Discrete(3),
            "indication": gym.spaces.Discrete(self.max_indications_per_ta),
            "enpv": gym.spaces.Box(
                low=-float("inf"), high=float("inf"), shape=(), dtype=float
            ),
            "trial_phase": gym.spaces.Discrete(len(TrialPhase) + 1),
            "ptrs": gym.spaces.Box(low=0, high=1, shape=(), dtype=float),
            "steps_remaining": gym.spaces.Box(low=0, high=1, shape=(), dtype=float),
            "eroi": gym.spaces.Box(
                low=-float("inf"), high=float("inf"), shape=(), dtype=float
            ),
        }
        if self.ptrs_readings_config.enabled:
            bd_fields["ask_cash_enpv"] = gym.spaces.Box(
                low=-float("inf"), high=float("inf"), shape=(), dtype=float
            )
            for _ph in range(_BD_NUM_NON_APPROVAL_PHASES):
                bd_fields[f"ph{_ph}_ptrs"] = gym.spaces.Box(
                    low=0, high=1, shape=(), dtype=float
                )
                bd_fields[f"ph{_ph}_equiv_n_norm"] = gym.spaces.Box(
                    low=0, high=1, shape=(), dtype=float
                )
        bd_space = gym.spaces.Dict(bd_fields)

        # Indication market space
        indication_fields = {
            "exclusivity_remaining": gym.spaces.Box(
                low=0, high=float("inf"), shape=(), dtype=float
            ),
            "my_avg_share": gym.spaces.Box(low=0, high=1, shape=(), dtype=float),
            "first_mover": gym.spaces.Discrete(2),
            "my_drugs": gym.spaces.Box(low=0, high=float("inf"), shape=(), dtype=int),
            "competitor_drugs": gym.spaces.Box(
                low=0, high=float("inf"), shape=(), dtype=int
            ),
        }
        indication_space = gym.spaces.Dict(indication_fields)

        # Alert space
        alert_fields = {
            "event_type": gym.spaces.Discrete(6),
            "agent_index": gym.spaces.Box(
                low=0, high=self._num_agents, shape=(), dtype=int
            ),
            "ta_index": gym.spaces.Discrete(3),
            "indication": gym.spaces.Discrete(self.max_indications_per_ta),
            "age": gym.spaces.Box(low=0, high=float("inf"), shape=(), dtype=int),
            "phase": gym.spaces.Box(low=0, high=1, shape=(), dtype=float),
            # Price the opponent paid to win a BD asset or clinical site;
            # 0.0 for non-deal alerts.
            "bd_price": gym.spaces.Box(low=0, high=float("inf"), shape=(), dtype=float),
            # Number of leaked brand-equity spends in this indication (dominance
            # signal); 0 for non-BE alerts.
            "be_count": gym.spaces.Box(
                low=0, high=float("inf"), shape=(), dtype=int
            ),
        }
        alert_space = gym.spaces.Dict(alert_fields)

        # Top-level space
        obs_fields = {
            "cash": gym.spaces.Box(
                low=float("-inf"),
                high=float("inf"),
                shape=(1,),
                dtype=np.float32,
            ),
            "time": gym.spaces.Box(low=0, high=int(1e9), shape=(1,), dtype=np.float32),
            "assets": gym.spaces.Tuple([asset_space] * self.max_num_assets),
            "bd_market": gym.spaces.Tuple([bd_space] * self.bd_max_slots),
            "indication_markets": gym.spaces.Dict({
                ta: gym.spaces.Tuple([indication_space] * self.max_indications_per_ta)
                for ta in THERAPEUTIC_AREAS
            }),
            "alerts": gym.spaces.Tuple([alert_space] * self.max_alerts),
        }
        if L.ta_experience_enabled:
            obs_fields["ta_experience"] = gym.spaces.Dict({
                ta: gym.spaces.Box(low=0, high=float("inf"), shape=(), dtype=float)
                for ta in THERAPEUTIC_AREAS
            })

        if L.clinical_sites_enabled:
            obs_fields["clinical_sites"] = gym.spaces.Dict({
                "operational_sites": gym.spaces.Box(
                    low=0, high=float("inf"), shape=(), dtype=float
                ),
                "free_sites": gym.spaces.Box(
                    low=0, high=float("inf"), shape=(), dtype=float
                ),
                "sites_in_development": gym.spaces.Box(
                    low=0, high=float("inf"), shape=(), dtype=float
                ),
                "site_auction_active": gym.spaces.Discrete(2),
            })

        return gym.spaces.Dict(obs_fields)

    def enabled_action_heads(self) -> list[str]:
        """
        Action heads an agent must supply every step (matches action_space).

        The set is config-dependent: only heads for enabled features appear.
        Every listed head is *required* in each step's action dict -- there are
        no silent defaults, so an agent must emit the head's no-op value (see
        :meth:`noop_action`) for any feature it does not want to use.
        """
        heads = ["investments", "bd_bids"]
        if self.ptrs_readings_config.enabled:
            heads.append("ptrs_research")
        if self._sites_on:
            heads.append("upgrade")
        if self._site_auction_on:
            heads.append("site_bid")
        if self._site_priority_on:
            heads.append("site_priority")
        if self.pricing_config.enabled:
            heads.append("pricing")
        if self.marketing_config.enabled:
            heads.append("demand_creation")
            heads.append("brand_equity")
        return heads

    def noop_action(self) -> dict[str, Any]:
        """
        A full do-nothing action containing every enabled head explicitly.

        Agents can build their action from this and overwrite only the heads
        they wish to use; this guarantees no required head is silently omitted
        and keeps the config-conditional head set in one place.
        """
        m = self.max_num_assets
        nbd = self.bd_max_slots
        nind = self.max_indications_per_ta * len(THERAPEUTIC_AREAS)
        nptrs = m + nbd
        default_level = (
            self.pricing_config.default_level if self.pricing_config.enabled else 0
        )
        builders = {
            "investments": lambda: np.zeros(m, dtype=np.int64),
            "bd_bids": lambda: np.zeros(nbd, dtype=np.float32),
            "ptrs_research": lambda: np.zeros(nptrs, dtype=np.int64),
            "upgrade": lambda: 0,
            "site_bid": lambda: np.array([0.0], dtype=np.float32),
            "site_priority": lambda: np.zeros(m, dtype=np.float32),
            "pricing": lambda: np.full(m, default_level, dtype=np.int64),
            "demand_creation": lambda: np.zeros(nind, dtype=np.int64),
            "brand_equity": lambda: np.zeros(m, dtype=np.int64),
        }
        return {head: builders[head]() for head in self.enabled_action_heads()}

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gym.Space:
        """
        Return action space for an agent.

        When investment_levels is enabled, investments use MultiDiscrete with
        6 choices per asset (NONE/MINIMAL/STANDARD/ACCELERATED/STOP/DROP).
        When drop_action is enabled, investments use MultiDiscrete([3]*N)
        (do nothing / invest / drop).
        Otherwise, investments use MultiBinary (invest or not).

        BD bids use a continuous Box, one entry per slot: a raw cash bid in
        GBP millions in [0, bd_max_bid]. A bid of 0 (rounded) means pass.
        The highest bid wins and pays its own bid; there is no affordability
        mask, so an overbid can bankrupt the winner.
        """
        use_levels = (
            self.investment_levels_config.enabled
        )
        use_drop = (
            self.drop_action_config.enabled
        )
        if use_levels:
            inv_space = gym.spaces.MultiDiscrete(
                [len(InvestmentLevel)] * self.max_num_assets
            )
        elif use_drop:
            inv_space = gym.spaces.MultiDiscrete([3] * self.max_num_assets)
        else:
            inv_space = gym.spaces.MultiBinary(self.max_num_assets)

        spaces = {
            "investments": inv_space,
            "bd_bids": gym.spaces.Box(
                low=0.0,
                high=float(self.bd_max_bid),
                shape=(self.bd_max_slots,),
                dtype=np.float32,
            ),
        }
        if self.ptrs_readings_config.enabled:
            n_slots = self.max_num_assets + self.bd_max_slots
            spaces["ptrs_research"] = gym.spaces.MultiDiscrete(
                [self.ptrs_readings_config.action_space_max_readings + 1] * n_slots
            )
        # Clinical-site actions. ``upgrade`` is a masked categorical (buy at most
        # one site this step); ``site_bid`` is a continuous cash bid in GBP
        # millions for the PvP auction (unmasked — off-cadence bids are ignored,
        # an overbid can bankrupt the winner); ``site_priority`` is a continuous
        # per-asset arbitration score used only when ``agent_priority`` is on.
        if self._sites_on:
            spaces["upgrade"] = gym.spaces.Discrete(2)
        if self._site_auction_on:
            spaces["site_bid"] = gym.spaces.Box(
                low=0.0,
                high=float(self.clinical_sites_config.site_max_bid),
                shape=(1,),
                dtype=np.float32,
            )
        if self._site_priority_on:
            spaces["site_priority"] = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.max_num_assets,),
                dtype=np.float32,
            )
        if self.pricing_config.enabled:
            num_price_levels = len(self.pricing_config.levels)
            spaces["pricing"] = gym.spaces.MultiDiscrete(
                [num_price_levels] * self.max_num_assets
            )
        if self.marketing_config.enabled:
            num_ind_slots = self.max_indications_per_ta * len(THERAPEUTIC_AREAS)
            spaces["demand_creation"] = gym.spaces.MultiDiscrete([2] * num_ind_slots)
            spaces["brand_equity"] = gym.spaces.MultiDiscrete([2] * self.max_num_assets)
        return gym.spaces.Dict(spaces)

    def _drop_fee(self, asset) -> float:
        """Fee charged for dropping this asset, or 0.0 if drops are free."""
        if not self.drop_action_config.enabled or asset.trial is None:
            return 0.0
        return self.drop_action_config.calculate_drop_fee(asset.trial.cost_remaining)

    def action_masks(self, agent: str) -> dict[str, np.ndarray]:
        """
        Return action masks for an agent.

        When investment_levels is enabled, returns per-asset masks with shape
        (max_num_assets, num_levels) matching MultiDiscrete action space:
            0: NONE - always valid
            1: MINIMAL - valid for investable Idle assets
            2: STANDARD - valid for investable Idle assets
            3: ACCELERATED - valid for investable Idle assets
            4: STOP - valid for InDevelopment assets only

        When investment_levels is disabled, returns binary mask (max_num_assets,)
        matching MultiBinary action space.

        BD bids are a continuous Box and are NOT masked (an overbid is allowed
        and can bankrupt the winner), so no "bd_bids" entry is returned.
        """
        game_state = self.multi_agent_game.agent_states[agent]
        asset_order = self._asset_id_orders[agent]
        use_levels = (
            self.investment_levels_config.enabled
        )

        if use_levels:
            num_levels = len(InvestmentLevel)
            investment_mask = []
            for i in range(self.max_num_assets):
                asset_id = asset_order[i] if i < len(asset_order) else None
                if asset_id is None or asset_id not in game_state.assets:
                    # Padding slot: only NONE valid
                    mask = [True] + [False] * (num_levels - 1)
                else:
                    asset = game_state.assets[asset_id]
                    if asset.state == AssetState.Idle:
                        can_invest = True
                        if self.mask_first_order_assets:
                            if game_state.cash - asset.cost_to_invest_this_step < 0:
                                can_invest = False
                        if self.mask_negative_enpv_assets:
                            if asset.enpv < 0:
                                can_invest = False
                        if can_invest:
                            # NONE + all investment levels, no STOP
                            mask = [True, True, True, True, False]
                        else:
                            mask = [True] + [False] * (num_levels - 1)
                    elif asset.state == AssetState.InDevelopment:
                        # Can change level or STOP
                        mask = [True, True, True, True, True]
                    else:
                        # OnMarket, Failed, Expired: only NONE
                        mask = [True] + [False] * (num_levels - 1)
                investment_mask.append(mask)
        else:
            use_drop = (
                self.drop_action_config.enabled
            )
            if use_drop:
                investment_mask = []
                for i in range(self.max_num_assets):
                    asset_id = asset_order[i] if i < len(asset_order) else None
                    if asset_id is None or asset_id not in game_state.assets:
                        investment_mask.append([True, False, False])
                        continue

                    asset = game_state.assets[asset_id]
                    can_drop = True
                    if self.mask_first_order_assets:
                        if game_state.cash - self._drop_fee(asset) < 0:
                            can_drop = False

                    if asset.state == AssetState.Idle:
                        can_invest = True
                        if self.mask_first_order_assets:
                            if game_state.cash - asset.cost_to_invest_this_step < 0:
                                can_invest = False
                        if self.mask_negative_enpv_assets and asset.enpv < 0:
                            can_invest = False
                        investment_mask.append([True, can_invest, can_drop])
                    else:
                        investment_mask.append([True, False, can_drop])
            else:
                investment_mask = np.zeros(self.max_num_assets, dtype=np.int8)
                for i, asset_id in enumerate(asset_order):
                    if asset_id is not None and asset_id in game_state.assets:
                        asset = game_state.assets[asset_id]
                        if asset.state == AssetState.Idle:
                            can_invest = True
                            if self.mask_first_order_assets:
                                if game_state.cash - asset.cost_to_invest_this_step < 0:
                                    can_invest = False
                            if self.mask_negative_enpv_assets:
                                if asset.enpv < 0:
                                    can_invest = False
                            if can_invest:
                                investment_mask[i] = 1

        # BD bids are continuous (Box) and unmasked — no per-slot mask.
        result = {
            "investments": investment_mask,
        }

        # Clinical-site upgrade: a masked categorical [no-op, buy]. Buying is
        # only offered when the agent can afford the next Fibonacci-priced site,
        # so a deterministic policy can never bankrupt itself on an upgrade.
        # site_bid / site_priority are continuous and unmasked.
        if self._sites_on:
            can_buy = game_state.can_afford_site_purchase()
            result["upgrade"] = [True, bool(can_buy)]

        # PTRS reading masks: per asset slot (portfolio + BD), per count
        if self.ptrs_readings_config.enabled:
            from pyxis_portfolio_challenge.config import fibonacci_cost

            cfg = self.ptrs_readings_config
            ptrs_research_mask = []

            def _reading_base_cost(a) -> float:
                has_trial = a is not None and a.trial is not None
                cost_rem = a.trial.cost_remaining if has_trial else 0.0
                raw = cfg.cost_fraction * cost_rem
                r = cfg.cost_rounding
                return round(raw / r) * r if r > 1 else raw

            # Portfolio slots
            for i in range(self.max_num_assets):
                asset_id = asset_order[i] if i < len(asset_order) else None
                asset = game_state.assets.get(asset_id) if asset_id else None
                has_pending = bool(asset and asset.pending_trial_chain)
                base = _reading_base_cost(asset)
                slot_mask = [True]  # 0 readings always valid
                for n in range(1, cfg.action_space_max_readings + 1):
                    slot_mask.append(
                        has_pending and game_state.cash >= fibonacci_cost(n, base)
                    )
                ptrs_research_mask.append(slot_mask)

            # BD slots
            bd_assets = (
                self.multi_agent_game.shared_market.current_bd_assets
                if self.multi_agent_game is not None
                else []
            )
            for slot in range(self.bd_max_slots):
                bd_asset = bd_assets[slot] if slot < len(bd_assets) else None
                has_pending = bool(bd_asset and bd_asset.pending_trial_chain)
                base = _reading_base_cost(bd_asset)
                slot_mask = [True]
                for n in range(1, cfg.action_space_max_readings + 1):
                    slot_mask.append(
                        has_pending and game_state.cash >= fibonacci_cost(n, base)
                    )
                ptrs_research_mask.append(slot_mask)
            result["ptrs_research"] = ptrs_research_mask

        # Pricing masks: only on-market assets can have non-default pricing
        if self.pricing_config.enabled:
            num_price_levels = len(self.pricing_config.levels)
            default_level = self.pricing_config.default_level
            pricing_mask = []
            for i in range(self.max_num_assets):
                asset_id = asset_order[i] if i < len(asset_order) else None
                if asset_id is not None and asset_id in game_state.assets:
                    asset = game_state.assets[asset_id]
                    if asset.state == AssetState.OnMarket:
                        # All price levels valid for on-market drugs
                        pricing_mask.append([True] * num_price_levels)
                        continue
                # Not on market or empty slot: only default level valid
                mask = [False] * num_price_levels
                mask[default_level] = True
                pricing_mask.append(mask)
            result["pricing"] = pricing_mask

        # Marketing masks
        if self.marketing_config.enabled:
            num_ind_slots = self.max_indications_per_ta * len(THERAPEUTIC_AREAS)
            # Demand creation sizes a shared indication market, so pre-market
            # spend is allowed. Brand equity is per drug and its score is reset
            # to the floor at launch, so only on-market drugs can take it.
            result["demand_creation"] = [[True, True]] * num_ind_slots
            brand_mask = []
            for i in range(self.max_num_assets):
                asset_id = asset_order[i] if i < len(asset_order) else None
                asset = game_state.assets.get(asset_id) if asset_id else None
                on_market = asset is not None and asset.state == AssetState.OnMarket
                brand_mask.append([True, on_market])
            result["brand_equity"] = brand_mask

        return result

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment."""
        self.agents = self.possible_agents.copy()

        if seed is not None:
            self._episode_rng = np.random.default_rng(seed)
        actual_seed = int(self._episode_rng.integers(0, 2**31))

        # Compute indications per TA based on game parameters
        num_tas = 3
        raw = (
            self._num_agents
            * (self.equilibrium_num_assets / num_tas)
            * self.on_market_fraction
            / self.target_drugs_per_indication
        )
        self._indications_per_ta = min(max(2, round(raw)), self.max_indications_per_ta)

        self.multi_agent_game = MultiAgentGame.initialise(
            num_agents=self._num_agents,
            seed=actual_seed,
            starting_cash=self.starting_cash,
            horizon=self.horizon,
            equilibrium_num_assets=self.equilibrium_num_assets,
            max_num_assets=self.max_num_assets,
            asset_arrival_sensitivity_below=self.asset_arrival_sensitivity_below,
            asset_arrival_sensitivity_above=self.asset_arrival_sensitivity_above,
            reinvestment_percentage=self.reinvestment_percentage,
            assets_dir=self.assets_dir,
            exclusivity_period=self.exclusivity_period,
            first_mover_bonus=self.first_mover_bonus,
            disable_market_share_competition=self.disable_market_share_competition,
            alert_history_length=self.alert_history_length,
            reward_fn_config={},
            distributional_ptrs_config=self.distributional_ptrs_config,
            ta_experience_config=self.ta_experience_config,
            uncertain_ptrs_config=self.uncertain_ptrs_config,
            investment_levels_config=self.investment_levels_config,
            interim_trial_observations_config=self.interim_trial_observations_config,
            rd_capacity_config=self.rd_capacity_config,
            drop_action_config=self.drop_action_config,
            ptrs_readings_config=self.ptrs_readings_config,
            clinical_sites_config=self.clinical_sites_config,
            marketing_config=self.marketing_config,
            indications_per_ta=self._indications_per_ta,
            indication_spread=self.indication_spread,
            indication_drift_speed=self.indication_drift_speed,
            trial_cost_multiplier=self.trial_cost_multiplier,
            approval_phase_config=self.approval_phase_config,
            # BD configuration
            bd_enabled=self.bd_enabled,
            bd_assets_dir=self.bd_assets_dir,
            bd_base_lambda=self.bd_base_lambda,
            bd_leak_lambda_boost=self.bd_leak_lambda_boost,
            bd_min_step=self.bd_min_step,
            bd_max_bid=self.bd_max_bid,
            bd_phase_weights=self.bd_phase_weights,
            bd_indication_activity_bias=self.bd_indication_activity_bias,
            bd_max_slots=self.bd_max_slots,
            bd_persist_steps=self.bd_persist_steps,
            # Leak configuration
            leak_phase_probabilities=self.leak_phase_probabilities,
            be_leak_probability=self.be_leak_probability,
            dc_leak_probability=self.dc_leak_probability,
            dc_leak_min_agents=self.dc_leak_min_agents,
            # Congestion penalty
            congestion_exponent=self.congestion_exponent,
            congestion_ramp_steps=self.congestion_ramp_steps,
            congestion_incumbent_penalty=self.congestion_incumbent_penalty,
            # Pricing elasticity
            pricing_elasticity=self.pricing_config.elasticity,
        )

        # Initialize asset ordering for observations
        self._asset_id_orders = {}
        for agent in self.possible_agents:
            game_state = self.multi_agent_game.agent_states[agent]
            asset_ids = list(game_state.assets.keys())
            asset_order = asset_ids + [None] * (self.max_num_assets - len(asset_ids))
            if self.shuffle_order:
                self.np_random.shuffle(asset_order)
            self._asset_id_orders[agent] = asset_order

        # Build initial observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        return observations, infos

    def step(
        self,
        actions: dict[str, Any],
    ) -> tuple[
        dict[str, Any],  # observations
        dict[str, float],  # rewards
        dict[str, bool],  # terminations
        dict[str, bool],  # truncations
        dict[str, Any],  # infos
    ]:
        """Execute one step in the environment."""
        # Store pre-step game for reward calculation
        pre_step_game = self.multi_agent_game

        # Parse actions
        parsed = self._parse_actions(actions)
        investments = parsed["investments"]
        bd_bids = parsed["bd_bids"]
        pricing_levels = parsed["pricing"]
        upgrade_actions = parsed["upgrade"]
        site_bid_cash = parsed["site_bid"]
        site_priority_arrays = parsed["site_priority"]

        # BD bids are continuous and unmasked: a positive bid on a slot with no
        # asset is legal policy output and is simply ignored by the game (the
        # auction only iterates over live BD slots), so no validation is needed.
        bd_avail_now = (
            pre_step_game.shared_market.current_bd_assets if self.bd_enabled else []
        )
        n_bd_live = len(bd_avail_now)
        if self.ptrs_readings_config.enabled:
            ptrs_raw = parsed.get("ptrs_research", {})
            for agent, arr in ptrs_raw.items():
                for slot in range(self.bd_max_slots):
                    n = int(arr[self.max_num_assets + slot])
                    if n > 0 and slot >= n_bd_live:
                        raise ValueError(
                            f"Agent '{agent}' commissioned {n} PTRS reading(s) "
                            f"on BD slot {slot} which has no asset (only "
                            f"{n_bd_live} BD asset(s) available). This violates "
                            f"the action mask."
                        )

        # Convert investment arrays to GameState-compatible action dicts
        use_levels = (
            self.investment_levels_config.enabled
        )
        use_drop = (
            self.drop_action_config.enabled
        )
        investor_actions = {}
        for agent in self.agents:
            agent_actions = {}
            asset_order = self._asset_id_orders[agent]
            agent_investments = investments[agent]

            for i, invest in enumerate(agent_investments):
                if i < len(asset_order) and asset_order[i] is not None:
                    asset_id = asset_order[i]
                    if use_levels:
                        level = InvestmentLevel.from_int(int(invest))
                        if level != InvestmentLevel.NONE:
                            agent_actions[asset_id] = level
                    elif use_drop:
                        if invest == 1:
                            agent_actions[asset_id] = "invest"
                        elif invest == 2:
                            agent_actions[asset_id] = "drop"
                    else:
                        if invest:
                            agent_actions[asset_id] = InvestmentLevel.STANDARD
            investor_actions[agent] = agent_actions

        # Convert pricing level indices to per-drug multipliers
        pricing_actions: dict[str, dict] | None = None
        if self.pricing_config.enabled:
            levels_list = self.pricing_config.levels
            pricing_actions = {}
            for agent in self.agents:
                agent_pricing: dict = {}
                asset_order = self._asset_id_orders[agent]
                agent_pricing_levels = pricing_levels[agent]
                for i, level_idx in enumerate(agent_pricing_levels):
                    if i < len(asset_order) and asset_order[i] is not None:
                        asset_id = asset_order[i]
                        mult = levels_list[int(level_idx)]
                        if mult != 1.0:
                            agent_pricing[asset_id] = mult
                pricing_actions[agent] = agent_pricing

        # Store pricing for observation building
        if pricing_actions is not None:
            self._current_pricing = pricing_actions
        else:
            self._current_pricing = {}

        # Decode marketing actions into per-agent dicts
        marketing_actions: dict[str, dict] | None = None
        if self.marketing_config.enabled:
            marketing_actions = {}
            for agent in self.agents:
                dc_arr = parsed["demand_creation"][agent]
                be_arr = parsed["brand_equity"][agent]
                asset_order = self._asset_id_orders[agent]
                dc_dict: dict[str, int] = {}
                for ta_idx, ta in enumerate(THERAPEUTIC_AREAS):
                    for ind_idx in range(self._indications_per_ta):
                        bucket = ta_idx * self.max_indications_per_ta + ind_idx
                        dc_dict[indication_key(ta, ind_idx)] = int(dc_arr[bucket])
                be_dict: dict = {}
                for i, asset_id in enumerate(asset_order):
                    if asset_id is not None and i < len(be_arr):
                        be_dict[asset_id] = int(be_arr[i])
                marketing_actions[agent] = {
                    "demand_creation": dc_dict,
                    "brand_equity": be_dict,
                }

        # Decode research actions: per-slot counts → {asset_uuid: n} per agent
        research_actions: dict[str, dict] | None = None
        if self.ptrs_readings_config.enabled:
            research_raw = parsed.get("ptrs_research", {})
            research_actions = {}
            for agent in self.agents:
                arr = research_raw.get(
                    agent,
                    np.zeros(self.max_num_assets + self.bd_max_slots, dtype=np.int64),
                )
                asset_order = self._asset_id_orders[agent]
                game_state = pre_step_game.agent_states[agent]
                counts: dict = {}
                for i in range(self.max_num_assets):
                    n = int(arr[i])
                    if n > 0 and i < len(asset_order) and asset_order[i] is not None:
                        asset_id = asset_order[i]
                        if asset_id in game_state.assets:
                            counts[asset_id] = n
                bd_avail = pre_step_game.shared_market.current_bd_assets
                for slot in range(self.bd_max_slots):
                    n = int(arr[self.max_num_assets + slot])
                    if n > 0 and slot < len(bd_avail):
                        counts[bd_avail[slot].id] = n
                research_actions[agent] = counts

        # Decode clinical-site actions into GameState-compatible structures.
        buy_site_actions: dict[str, bool] | None = None
        site_bids: dict[str, float] | None = None
        site_priorities: dict[str, dict] | None = None
        if self._sites_on:
            buy_site_actions = {
                agent: bool(int(upgrade_actions[agent]) == 1)
                for agent in self.agents
            }
        if self._site_auction_on:
            site_bids = {
                agent: float(site_bid_cash[agent]) for agent in self.agents
            }
        if self._site_priority_on:
            site_priorities = {}
            for agent in self.agents:
                asset_order = self._asset_id_orders[agent]
                arr = site_priority_arrays[agent]
                per_asset: dict = {}
                for i in range(self.max_num_assets):
                    if i < len(asset_order) and asset_order[i] is not None:
                        per_asset[asset_order[i]] = float(arr[i])
                site_priorities[agent] = per_asset

        # Step the multi-agent game
        has_bids = any(
            any(bid > 0 for bid in bids) for bids in bd_bids.values()
        )
        has_site_bids = site_bids is not None and any(
            bid > 0 for bid in site_bids.values()
        )
        self.multi_agent_game = pre_step_game.step(
            investor_actions=investor_actions,
            bd_bids=bd_bids if has_bids else None,
            pricing_actions=pricing_actions,
            research_actions=research_actions,
            marketing_actions=marketing_actions,
            site_bids=site_bids if has_site_bids else None,
            buy_site_actions=buy_site_actions,
            site_priorities=site_priorities,
        )

        # Update asset orderings for new/removed assets
        self._update_asset_orderings()

        # Calculate rewards
        base_rewards = {}
        for agent in self.agents:
            pre_state = pre_step_game.agent_states[agent]
            post_state = self.multi_agent_game.agent_states[agent]
            base_rewards[agent] = self._reward_fn.compute(
                pre_step_game_state=pre_state,
                post_step_game_state=post_state,
            )

        rewards = self._apply_reward_type(base_rewards)

        # Check terminations
        # Per-agent: game_ended (bankruptcy or horizon)
        # Episode terminates when ALL agents' games have ended
        terminations = {
            agent: self.multi_agent_game.agent_states[agent].game_ended
            for agent in self.agents
        }
        truncations = {agent: False for agent in self.agents}

        # Build observations and infos
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def _apply_reward_type(self, base_rewards: dict[str, float]) -> dict[str, float]:
        """Apply multi-agent reward type wrapper to base rewards."""
        if self._reward_type == "absolute":
            return {aid: r / self._reward_scale for aid, r in base_rewards.items()}

        if self._reward_type == "relative_rank":
            ranked = sorted(
                base_rewards.keys(),
                key=lambda x: base_rewards[x],
                reverse=True,
            )
            n = len(ranked)
            rewards = {}
            for rank, aid in enumerate(ranked):
                state = self.multi_agent_game.agent_states[aid]
                if state.game_ended and state.cash < 0:
                    # Bankrupt: strong negative
                    rewards[aid] = -1.0
                else:
                    # Zero-sum ranking: winner +1.0, loser -1.0
                    # For 2 agents: rank 0 → +1.0, rank 1 → -1.0
                    # For N agents: linearly spaced from +1.0 to -1.0
                    rewards[aid] = 1.0 - 2.0 * rank / max(n - 1, 1)
            return rewards

        if self._reward_type == "zero_sum":
            active = [
                aid
                for aid in base_rewards
                if not (
                    self.multi_agent_game.agent_states[aid].game_ended
                    and self.multi_agent_game.agent_states[aid].cash < 0
                )
            ]
            if not active:
                return {aid: 0.0 for aid in base_rewards}
            total = sum(base_rewards[aid] for aid in active)
            rewards = {}
            for aid in base_rewards:
                state = self.multi_agent_game.agent_states[aid]
                if state.game_ended and state.cash < 0:
                    rewards[aid] = -1.0
                else:
                    others_total = total - base_rewards[aid]
                    others_count = len(active) - 1
                    if others_count > 0:
                        others_mean = others_total / others_count
                        rewards[aid] = (
                            base_rewards[aid] - others_mean
                        ) / self._reward_scale
                    else:
                        rewards[aid] = base_rewards[aid] / self._reward_scale
            return rewards

        raise ValueError(f"Unknown reward_type: {self._reward_type}")

    def _parse_actions(self, actions: dict[str, Any]) -> dict[str, dict]:
        """
        Parse actions from agents.

        Every agent must submit a dict containing *every* enabled action head
        (see :meth:`enabled_action_heads`) each step -- there are no silent
        defaults. An agent that does not want to use a feature must emit that
        head's explicit no-op value, exactly as it emits "don't invest" for the
        investment head. Missing heads, unknown/disabled heads, and non-dict
        actions all raise :class:`ValueError` rather than being papered over,
        so a feature can never silently become a no-op by omission.

        BD bids are cash amounts in GBP millions (0 = pass).
        Pricing is integer level indices into pricing_config.levels.
        """
        investments = {}
        bd_bids = {}
        pricing = {}
        demand_creation = {}
        brand_equity = {}
        ptrs_research = {}
        upgrade = {}
        site_bid = {}
        site_priority = {}
        default_level = (
            self.pricing_config.default_level if self.pricing_config.enabled else 0
        )
        num_ind_slots = self.max_indications_per_ta * len(THERAPEUTIC_AREAS)
        num_ptrs_research_slots = self.max_num_assets + self.bd_max_slots
        required = self.enabled_action_heads()
        required_set = set(required)

        for agent in self.agents:
            action = actions.get(agent)
            if not isinstance(action, dict):
                raise ValueError(
                    f"Agent {agent!r} must submit a dict action containing every "
                    f"enabled head {required}, got {type(action).__name__}. Build "
                    f"from env.noop_action() and overwrite the heads you want to "
                    f"use."
                )
            missing = [h for h in required if h not in action]
            if missing:
                raise ValueError(
                    f"Agent {agent!r} action is missing required head(s) "
                    f"{missing}. Every enabled head must be supplied each step; "
                    f"emit its no-op value for features you do not want to use. "
                    f"Enabled heads: {required}."
                )
            extra = [h for h in action if h not in required_set]
            if extra:
                raise ValueError(
                    f"Agent {agent!r} action contains unknown or disabled "
                    f"head(s) {extra}; supplying them would silently do nothing. "
                    f"Enabled heads: {required}."
                )

            investments[agent] = np.asarray(action["investments"], dtype=np.int64)
            bd_bids[agent] = self._decode_bd_bids(action["bd_bids"])
            if self.pricing_config.enabled:
                pricing[agent] = np.asarray(action["pricing"], dtype=np.int64)
            else:
                pricing[agent] = np.full(
                    self.max_num_assets, default_level, dtype=np.int64
                )
            if "demand_creation" in required_set:
                demand_creation[agent] = np.asarray(
                    action["demand_creation"], dtype=np.int64
                )
            else:
                demand_creation[agent] = np.zeros(num_ind_slots, dtype=np.int64)
            if "brand_equity" in required_set:
                brand_equity[agent] = np.asarray(
                    action["brand_equity"], dtype=np.int64
                )
            else:
                brand_equity[agent] = np.zeros(self.max_num_assets, dtype=np.int64)
            if self.ptrs_readings_config.enabled:
                ptrs_research[agent] = np.asarray(
                    action["ptrs_research"], dtype=np.int64
                )
            else:
                ptrs_research[agent] = np.zeros(
                    num_ptrs_research_slots, dtype=np.int64
                )
            upgrade[agent] = (
                int(np.rint(action["upgrade"])) if self._sites_on else 0
            )
            site_bid[agent] = (
                self._decode_site_bid(action["site_bid"])
                if self._site_auction_on
                else 0.0
            )
            site_priority[agent] = (
                np.asarray(action["site_priority"], dtype=np.float32)
                if self._site_priority_on
                else np.zeros(self.max_num_assets, dtype=np.float32)
            )

        return {
            "investments": investments,
            "bd_bids": bd_bids,
            "pricing": pricing,
            "demand_creation": demand_creation,
            "brand_equity": brand_equity,
            "ptrs_research": ptrs_research,
            "upgrade": upgrade,
            "site_bid": site_bid,
            "site_priority": site_priority,
        }

    def _decode_bd_bids(self, raw_bids) -> list[float]:
        """
        Convert a raw BD action (cash bid in GBP millions) to GBP cash.

        The action is clamped to ``[0, bd_max_bid]`` and rounded to the nearest
        integer GBP million, then scaled to GBP. A value of 0 (or negative) is a
        pass. There is no affordability clamp — an overbid can bankrupt the
        winner (see ``resolve_bd_bid``).
        """
        arr = np.asarray(raw_bids, dtype=np.float64)
        millions = np.clip(np.rint(arr), 0.0, float(self.bd_max_bid))
        return [float(m) * _BD_BID_UNIT for m in millions]

    def _decode_site_bid(self, raw_bid) -> float:
        """
        Convert a raw clinical-site auction action (£M) to a GBP cash bid.

        Mirrors :meth:`_decode_bd_bids`: the scalar action is clamped to
        ``[0, site_max_bid]`` and rounded to the nearest integer GBP million,
        then scaled to GBP. A value of 0 (or negative) is a pass. There is no
        affordability clamp — an overbid can bankrupt the winner.
        """
        max_bid = float(self.clinical_sites_config.site_max_bid)
        val = float(np.asarray(raw_bid, dtype=np.float64).reshape(-1)[0])
        millions = float(np.clip(np.rint(val), 0.0, max_bid))
        return millions * _SITE_BID_UNIT

    def _update_asset_orderings(self) -> None:
        """Update asset orderings for new assets acquired this step."""
        for agent in self.agents:
            game_state = self.multi_agent_game.agent_states[agent]
            asset_order = self._asset_id_orders[agent]

            # Remove stale entries first to free up slots
            for i in range(len(asset_order)):
                if (
                    asset_order[i] is not None
                    and asset_order[i] not in game_state.assets
                ):
                    asset_order[i] = None

            # Then add new assets to available slots
            ordered_ids = set(aid for aid in asset_order if aid is not None)
            for asset_id in game_state.assets:
                if asset_id not in ordered_ids:
                    for i in range(len(asset_order)):
                        if asset_order[i] is None:
                            asset_order[i] = asset_id
                            break

    def _get_observation(self, agent: str) -> Union[np.ndarray, dict]:
        """Build observation for an agent (flat or dict)."""
        if not self.flatten_obs:
            return self._get_observation_dict(agent)
        return self._get_observation_flat(agent)

    def _get_observation_flat(self, agent: str) -> np.ndarray:
        """Build flattened observation for an agent."""
        obs = np.zeros(self._obs_size, dtype=np.float32)
        game_state = self.multi_agent_game.agent_states[agent]
        shared_market = self.multi_agent_game.shared_market

        L = self._layout
        asset_total = L.asset_total_features
        asset_scalar = L.asset_scalar_features
        trial_feat = L.trial_features
        dist_on = L.distributional_ptrs_enabled
        interim_on = L.interim_obs_enabled
        off_interim = L.offset_interim_signal
        off_progress = L.offset_trial_progress
        off_ta_idx = L.offset_ta_index
        off_indication = L.offset_indication
        off_pricing = L.offset_pricing

        # Global features
        pos = 0
        obs[pos] = game_state.cash
        obs[pos + 1] = self.multi_agent_game.time
        pos = 2

        if L.ta_experience_enabled:
            for i, ta in enumerate(THERAPEUTIC_AREAS):
                obs[pos + i] = game_state.ta_experience.get(ta, 0.0)
            pos += L.num_ta_exp_features

        # Clinical-site globals (appended after all other global blocks).
        if L.clinical_sites_enabled:
            site_off = L.offset_clinical_sites
            obs[site_off] = game_state.operational_sites
            obs[site_off + 1] = game_state.free_sites
            obs[site_off + 2] = len(game_state.sites_in_development)
            obs[site_off + 3] = float(shared_market.site_auction_available())

        offset = L.global_features

        # Per-asset features
        asset_order = self._asset_id_orders[agent]
        ptrs_readings_cfg = self.ptrs_readings_config
        for i in range(self.max_num_assets):
            asset_offset = offset + i * asset_total
            asset_id = asset_order[i] if i < len(asset_order) else None

            if asset_id is None or asset_id not in game_state.assets:
                obs[asset_offset + 9] = AssetState.Expired.integer
                if dist_on:
                    trial_off = asset_offset + asset_scalar
                    for _ in TrialPhase:
                        obs[trial_off + 4] = 1.0  # ptrs_confidence
                        trial_off += trial_feat
                continue

            asset = game_state.assets[asset_id]
            obs[asset_offset] = asset.max_revenue
            obs[asset_offset + 1] = asset.time_until_max_revenue
            obs[asset_offset + 2] = asset.time_until_patent_expiry

            if asset.state == AssetState.OnMarket:
                obs[asset_offset + 3] = 0
            elif asset.trial and asset.trial.state == TrialState.PHASE_FAILED:
                obs[asset_offset + 3] = 0
            elif asset.trial:
                obs[asset_offset + 3] = asset.trial.phase.integer + 1
            else:
                obs[asset_offset + 3] = 0

            obs[asset_offset + 4] = asset.time_on_market
            obs[asset_offset + 5] = asset.cost_this_step
            obs[asset_offset + 6] = asset.revenue_this_step
            # Own-asset value is exposed as cash-adjusted eNPV (revenues scaled
            # by reinvestment_percentage) so the observation aligns with the NCF
            # reward the agent actually optimises. Full eNPV would overstate the
            # liquid value that flows to cash. (Field kept named "enpv" in the
            # dict obs / converters for layout stability.)
            obs[asset_offset + 7] = asset.cash_enpv(self.reinvestment_percentage)
            obs[asset_offset + 8] = asset.eroi
            obs[asset_offset + 9] = asset.state.integer

            if interim_on:
                obs[asset_offset + off_interim] = asset.interim_signal
                obs[asset_offset + off_progress] = asset.trial_progress

            obs[asset_offset + off_ta_idx] = TA_INDEX.get(asset.therapeutic_area, 0)
            obs[asset_offset + off_indication] = asset.indication

            if L.pricing_enabled:
                agent_pricing = self._current_pricing.get(agent, {})
                obs[asset_offset + off_pricing] = agent_pricing.get(asset_id, 1.0)

            if L.marketing_enabled:
                obs[asset_offset + L.offset_brand_score] = game_state._brand_scores.get(
                    asset_id, 0.0
                )

            # Trial phase features
            trial_off = asset_offset + asset_scalar
            ptrs_readings_on = (
                ptrs_readings_cfg is not None and ptrs_readings_cfg.enabled
            )
            off_ptrs_count = L.offset_ptrs_count
            trial = asset.trial
            for phase in TrialPhase:
                if trial and trial.phase == phase:
                    obs[trial_off] = trial.cost_remaining
                    obs[trial_off + 1] = trial.time_remaining
                    if ptrs_readings_on and trial.ptrs_sample_mean is not None:
                        obs[trial_off + 2] = trial.ptrs_sample_mean
                    else:
                        obs[trial_off + 2] = trial.ptrs
                    if dist_on:
                        obs[trial_off + 3] = trial.ptrs_expected
                        obs[trial_off + 4] = trial.ptrs_confidence
                        obs[trial_off + 5] = trial.ptrs_range_low
                        obs[trial_off + 6] = trial.ptrs_range_high
                    if ptrs_readings_on and off_ptrs_count >= 0:
                        equiv_n = (
                            trial.ptrs_total_precision
                            * ptrs_readings_cfg.sigma_logit_base**2
                        )
                        obs[trial_off + off_ptrs_count] = (
                            min(equiv_n, ptrs_readings_cfg.max_sample_obs)
                            / ptrs_readings_cfg.max_sample_obs
                        )
                    trial = trial.next_trial_on_success
                else:
                    if dist_on:
                        obs[trial_off + 4] = 1.0  # ptrs_confidence default
                trial_off += trial_feat

        offset += self.max_num_assets * asset_total

        # BD observation
        bd_obs_list = shared_market.get_bd_observations()
        bd_assets = shared_market.current_bd_assets
        bd_slot_size = self._bd_obs_size
        ptrs_readings_on = ptrs_readings_cfg is not None and ptrs_readings_cfg.enabled
        bd_clones = game_state._bd_asset_clones
        for slot in range(self.bd_max_slots):
            bd_offset = offset + slot * bd_slot_size
            if slot < len(bd_obs_list):
                bd_obs_d = bd_obs_list[slot]
                bd_shared = bd_assets[slot] if slot < len(bd_assets) else None
                clone = bd_clones.get(str(bd_shared.id)) if bd_shared else None
                asset_for_obs = clone if clone is not None else bd_shared
                chain = asset_for_obs.pending_trial_chain if asset_for_obs else []
                obs[bd_offset]      = 1.0
                obs[bd_offset + 1]  = bd_obs_d["max_revenue"]
                obs[bd_offset + 2]  = bd_obs_d["time_until_max_revenue"]
                obs[bd_offset + 3]  = bd_obs_d["time_until_patent_expiry"]
                obs[bd_offset + 4]  = TA_INDEX.get(bd_obs_d["therapeutic_area"], 0)
                obs[bd_offset + 5]  = bd_obs_d["indication"]
                # Cash-adjusted eNPV (revenues scaled by reinvestment_percentage)
                # so the BD value matches the own-asset value slot and the NCF
                # reward. Reads the agent's private clone when readings exist.
                obs[bd_offset + 6]  = (
                    asset_for_obs.cash_enpv(self.reinvestment_percentage)
                    if asset_for_obs
                    else bd_obs_d["enpv"]
                )
                obs[bd_offset + 7] = bd_obs_d["trial_phase"]
                obs[bd_offset + 8] = chain[0].ptrs if chain else bd_obs_d["ptrs"]
                obs[bd_offset + 9] = bd_obs_d.get("steps_remaining", 1.0)
                obs[bd_offset + 10] = asset_for_obs.eroi if asset_for_obs else 0.0
                if ptrs_readings_on:
                    obs[bd_offset + _BD_ASK_CASH_ENPV_IDX] = (
                        bd_shared.cash_enpv(self.reinvestment_percentage)
                        if bd_shared
                        else 0.0
                    )
                    for ph in range(_BD_NUM_NON_APPROVAL_PHASES):
                        base = bd_offset + _BD_PHASE_FEATURES_IDX + ph * 2
                        if ph < len(chain):
                            t = chain[ph]
                            equiv_n = (
                                t.ptrs_total_precision
                                * ptrs_readings_cfg.sigma_logit_base**2
                            )
                            obs[base] = t.ptrs
                            obs[base + 1] = (
                                min(equiv_n, ptrs_readings_cfg.max_sample_obs)
                                / ptrs_readings_cfg.max_sample_obs
                            )
                        # else: leave as 0.0 (array pre-zeroed)
        offset += self.bd_max_slots * bd_slot_size

        # Indication market features
        per_drug_shares = self.multi_agent_game._cached_market_shares.get(agent, {})

        for ta_idx, ta in enumerate(THERAPEUTIC_AREAS):
            for ind_idx in range(self.max_indications_per_ta):
                slot = ta_idx * self.max_indications_per_ta + ind_idx
                ind_offset = offset + slot * self._indication_features

                key = indication_key(ta, ind_idx)
                ind_market = shared_market.indication_markets.get(key)

                if ind_market is not None:
                    obs[ind_offset] = ind_market.exclusivity_remaining(
                        self.multi_agent_game.time
                    )
                    my_drug_ids = ind_market.active_drugs.get(agent, [])
                    if my_drug_ids:
                        drug_shares = [
                            per_drug_shares.get(did, 0.0) for did in my_drug_ids
                        ]
                        obs[ind_offset + 1] = sum(drug_shares) / len(drug_shares)
                    obs[ind_offset + 2] = float(ind_market.first_mover_agent == agent)
                    my_drugs = 0
                    competitor_drugs = 0
                    for aid, drug_ids in ind_market.active_drugs.items():
                        if aid == agent:
                            my_drugs += len(drug_ids)
                        else:
                            competitor_drugs += len(drug_ids)
                    obs[ind_offset + 3] = my_drugs
                    obs[ind_offset + 4] = competitor_drugs
                    if L.marketing_enabled:
                        obs[ind_offset + 5] = ind_market.demand_multiplier

        offset += (
            len(THERAPEUTIC_AREAS)
            * self.max_indications_per_ta
            * self._indication_features
        )

        # Alert features
        alerts = shared_market.get_alerts_for_agent(agent)
        recent_alerts = alerts[-self.max_alerts :] if self.max_alerts > 0 else []
        for i, alert in enumerate(recent_alerts):
            alert_offset = offset + i * _ALERT_FEATURES
            if alert.event_type == AlertType.DRUG_RELEASE:
                obs[alert_offset] = 1.0
            elif alert.event_type == AlertType.BD_DEAL:
                obs[alert_offset + 1] = 1.0
            elif alert.event_type == AlertType.PIPELINE_LEAK:
                obs[alert_offset + 2] = 1.0
            elif alert.event_type == AlertType.CLINICAL_SITE_DEAL:
                obs[alert_offset + 3] = 1.0
            elif alert.event_type == AlertType.BE_SPEND:
                obs[alert_offset + 4] = 1.0
                obs[alert_offset + 12] = alert.details.get("be_count", 1)
            elif alert.event_type == AlertType.DC_SPEND:
                obs[alert_offset + 5] = 1.0
            if alert.agent_id in self.possible_agents:
                obs[alert_offset + 6] = self.possible_agents.index(alert.agent_id)
            obs[alert_offset + 7] = TA_INDEX.get(alert.therapeutic_area, 0)
            obs[alert_offset + 8] = alert.indication
            obs[alert_offset + 9] = self.multi_agent_game.time - alert.step
            if alert.event_type == AlertType.PIPELINE_LEAK:
                phase_str = alert.details.get("new_phase", "")
                phase_map = {"Phase 2": 0.33, "Phase 3": 0.67, "Approval": 1.0}
                obs[alert_offset + 10] = phase_map.get(phase_str, 0.0)
            elif alert.event_type == AlertType.BD_DEAL:
                obs[alert_offset + 11] = alert.details.get("price", 0.0)
            elif alert.event_type == AlertType.CLINICAL_SITE_DEAL:
                obs[alert_offset + 11] = alert.details.get("price", 0.0)

        return obs

    @property
    def _padding_asset_obs(self):
        """Generate padding asset observation for empty slots."""
        L = self._layout
        trial_pad = {"cost_remaining": 0.0, "time_remaining": 0, "ptrs": 0.0}
        if L.distributional_ptrs_enabled:
            trial_pad["ptrs_expected"] = 0.0
            trial_pad["ptrs_confidence"] = 1.0
            trial_pad["ptrs_range_low"] = 0.0
            trial_pad["ptrs_range_high"] = 0.0
        if L.offset_ptrs_count >= 0:
            trial_pad["ptrs_equiv_n_norm"] = 0.0

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
            "indication": 0,
        }
        if L.interim_obs_enabled:
            obs["interim_signal"] = 0.0
            obs["trial_progress"] = 0.0
        if L.pricing_enabled:
            obs["price_multiplier"] = 1.0
        if L.marketing_enabled:
            obs["brand_score"] = 0.0
        return obs

    @property
    def _PADDING_BD_OBS(self):
        d = {
            "available": 0,
            "max_revenue": 0.0,
            "time_until_max_revenue": 0,
            "time_until_patent_expiry": 0,
            "ta_index": 0,
            "indication": 0,
            "enpv": 0.0,
            "trial_phase": 0,
            "ptrs": 0.0,
            "steps_remaining": 0.0,
            "eroi": 0.0,
        }
        if self.ptrs_readings_config.enabled:
            d["ask_cash_enpv"] = 0.0
            for _ph in range(_BD_NUM_NON_APPROVAL_PHASES):
                d[f"ph{_ph}_ptrs"] = 0.0
                d[f"ph{_ph}_equiv_n_norm"] = 0.0
        return d

    @property
    def _PADDING_INDICATION_OBS(self):
        obs = {
            "exclusivity_remaining": 0.0,
            "my_avg_share": 0.0,
            "first_mover": 0,
            "my_drugs": 0,
            "competitor_drugs": 0,
        }
        if self._layout.marketing_enabled:
            obs["demand_multiplier"] = 1.0
        return obs

    # -1 sentinel distinguishes padding from real alerts
    # (0/1/2/3 = release/bd/leak/clinical_site)
    _PADDING_ALERT_OBS = {
        "event_type": -1,
        "agent_index": 0,
        "ta_index": 0,
        "indication": 0,
        "age": 0,
        "phase": 0.0,
        "bd_price": 0.0,
        "be_count": 0,
    }

    def _get_observation_dict(self, agent: str) -> dict:
        """Build dict-based observation for an agent."""
        L = self._layout
        dist_on = L.distributional_ptrs_enabled
        interim_on = L.interim_obs_enabled
        game_state = self.multi_agent_game.agent_states[agent]
        shared_market = self.multi_agent_game.shared_market
        ptrs_readings_cfg = self.ptrs_readings_config
        ptrs_readings_on = ptrs_readings_cfg is not None and ptrs_readings_cfg.enabled

        off_ptrs_count = L.offset_ptrs_count

        def _make_trial_obs(
            cost,
            time_rem,
            ptrs,
            ptrs_exp,
            ptrs_conf,
            ptrs_lo,
            ptrs_hi,
            ptrs_equiv_n_norm=0.0,
        ):
            t = {
                "cost_remaining": cost,
                "time_remaining": time_rem,
                "ptrs": ptrs,
            }
            if dist_on:
                t["ptrs_expected"] = ptrs_exp
                t["ptrs_confidence"] = ptrs_conf
                t["ptrs_range_low"] = ptrs_lo
                t["ptrs_range_high"] = ptrs_hi
            if ptrs_readings_on:
                t["ptrs_equiv_n_norm"] = ptrs_equiv_n_norm
            return t

        def _get_asset_obs(asset, asset_id):
            trial = asset.trial
            trials_list = []
            for phase in TrialPhase:
                if trial and trial.phase == phase:
                    ptrs_val = (
                        trial.ptrs_sample_mean
                        if ptrs_readings_on and trial.ptrs_sample_mean is not None
                        else trial.ptrs
                    )
                    equiv_n_norm = 0.0
                    if ptrs_readings_on and off_ptrs_count >= 0:
                        equiv_n = (
                            trial.ptrs_total_precision
                            * ptrs_readings_cfg.sigma_logit_base**2
                        )
                        equiv_n_norm = (
                            min(equiv_n, ptrs_readings_cfg.max_sample_obs)
                            / ptrs_readings_cfg.max_sample_obs
                        )
                    trials_list.append(
                        _make_trial_obs(
                            trial.cost_remaining,
                            trial.time_remaining,
                            ptrs_val,
                            trial.ptrs_expected,
                            trial.ptrs_confidence,
                            trial.ptrs_range_low,
                            trial.ptrs_range_high,
                            ptrs_equiv_n_norm=equiv_n_norm,
                        )
                    )
                    trial = trial.next_trial_on_success
                else:
                    trials_list.append(_make_trial_obs(0.0, 0, 0.0, 0.0, 1.0, 0.0, 0.0))

            if asset.state == AssetState.OnMarket:
                pending = 0
            elif asset.trial and asset.trial.state == TrialState.PHASE_FAILED:
                pending = 0
            elif asset.trial:
                pending = asset.trial.phase.integer + 1
            else:
                pending = 0

            obs = {
                "max_revenue": asset.max_revenue,
                "time_until_max_revenue": asset.time_until_max_revenue,
                "time_until_patent_expiry": (asset.time_until_patent_expiry),
                "pending_trial_phase": pending,
                "time_on_market": asset.time_on_market,
                "cost_this_step": asset.cost_this_step,
                "revenue_this_step": asset.revenue_this_step,
                # Cash-adjusted eNPV (aligned to the NCF reward); see the flat
                # observation builder for rationale. Field name kept as "enpv".
                "enpv": asset.cash_enpv(self.reinvestment_percentage),
                "eroi": asset.eroi,
                "trials": tuple(trials_list),
                "state": asset.state.integer,
                "ta_index": TA_INDEX.get(asset.therapeutic_area, 0),
                "indication": asset.indication,
            }
            if interim_on:
                obs["interim_signal"] = asset.interim_signal
                obs["trial_progress"] = asset.trial_progress
            if L.pricing_enabled:
                agent_pricing = self._current_pricing.get(agent, {})
                obs["price_multiplier"] = agent_pricing.get(asset_id, 1.0)
            if L.marketing_enabled:
                obs["brand_score"] = game_state._brand_scores.get(asset_id, 0.0)
            return obs

        # Per-asset observations
        asset_order = self._asset_id_orders[agent]
        asset_obs = []
        for i in range(self.max_num_assets):
            asset_id = asset_order[i] if i < len(asset_order) else None
            if asset_id is None or asset_id not in game_state.assets:
                asset_obs.append(self._padding_asset_obs)
            else:
                asset_obs.append(_get_asset_obs(game_state.assets[asset_id], asset_id))

        # BD market observations
        bd_obs_list = shared_market.get_bd_observations()
        bd_assets = shared_market.current_bd_assets
        bd_clones = game_state._bd_asset_clones
        bd_obs = []
        for slot in range(self.bd_max_slots):
            if slot < len(bd_obs_list):
                bd = bd_obs_list[slot]
                bd_shared = bd_assets[slot] if slot < len(bd_assets) else None
                clone = bd_clones.get(str(bd_shared.id)) if bd_shared else None
                asset_for_obs = clone if clone is not None else bd_shared
                chain = asset_for_obs.pending_trial_chain if asset_for_obs else []
                entry = {
                    "available": 1,
                    "max_revenue": bd["max_revenue"],
                    "time_until_max_revenue": bd["time_until_max_revenue"],
                    "time_until_patent_expiry": bd["time_until_patent_expiry"],
                    "ta_index": TA_INDEX.get(bd["therapeutic_area"], 0),
                    "indication": bd["indication"],
                    # Cash-adjusted eNPV (see the flat BD builder); field name
                    # kept "enpv" for layout stability with the converters.
                    "enpv": (
                        asset_for_obs.cash_enpv(self.reinvestment_percentage)
                        if asset_for_obs
                        else bd["enpv"]
                    ),
                    "trial_phase": bd["trial_phase"],
                    "ptrs": chain[0].ptrs if chain else bd["ptrs"],
                    "steps_remaining": bd.get("steps_remaining", 1.0),
                    "eroi": asset_for_obs.eroi if asset_for_obs else 0.0,
                }
                if ptrs_readings_on:
                    entry["ask_cash_enpv"] = (
                        bd_shared.cash_enpv(self.reinvestment_percentage)
                        if bd_shared
                        else 0.0
                    )
                    for ph in range(_BD_NUM_NON_APPROVAL_PHASES):
                        if ph < len(chain):
                            t = chain[ph]
                            equiv_n = (
                                t.ptrs_total_precision
                                * ptrs_readings_cfg.sigma_logit_base**2
                            )
                            entry[f"ph{ph}_ptrs"] = t.ptrs
                            entry[f"ph{ph}_equiv_n_norm"] = (
                                min(equiv_n, ptrs_readings_cfg.max_sample_obs)
                                / ptrs_readings_cfg.max_sample_obs
                            )
                        else:
                            entry[f"ph{ph}_ptrs"] = 0.0
                            entry[f"ph{ph}_equiv_n_norm"] = 0.0
                bd_obs.append(entry)
            else:
                bd_obs.append(dict(self._PADDING_BD_OBS))

        # Indication market observations
        per_drug_shares = self.multi_agent_game._cached_market_shares.get(agent, {})
        indication_obs: dict[str, list] = {}
        for ta in THERAPEUTIC_AREAS:
            ta_indications = []
            for ind_idx in range(self.max_indications_per_ta):
                key = indication_key(ta, ind_idx)
                ind_market = shared_market.indication_markets.get(key)
                if ind_market is None:
                    ta_indications.append(dict(self._PADDING_INDICATION_OBS))
                else:
                    my_drug_ids = ind_market.active_drugs.get(agent, [])
                    if my_drug_ids:
                        drug_shares = [
                            per_drug_shares.get(did, 0.0) for did in my_drug_ids
                        ]
                        avg_share = sum(drug_shares) / len(drug_shares)
                    else:
                        avg_share = 0.0
                    my_drugs = 0
                    comp_drugs = 0
                    for aid, drug_ids in ind_market.active_drugs.items():
                        if aid == agent:
                            my_drugs += len(drug_ids)
                        else:
                            comp_drugs += len(drug_ids)
                    ind_d: dict = {
                        "exclusivity_remaining": (
                            ind_market.exclusivity_remaining(self.multi_agent_game.time)
                        ),
                        "my_avg_share": avg_share,
                        "first_mover": int(ind_market.first_mover_agent == agent),
                        "my_drugs": my_drugs,
                        "competitor_drugs": comp_drugs,
                    }
                    if L.marketing_enabled:
                        ind_d["demand_multiplier"] = ind_market.demand_multiplier
                    ta_indications.append(ind_d)
            indication_obs[ta] = tuple(ta_indications)

        # Alert observations
        alerts = shared_market.get_alerts_for_agent(agent)
        recent = alerts[-self.max_alerts :] if self.max_alerts > 0 else []
        alert_obs = []
        phase_map = {
            "Phase 2": 0.33,
            "Phase 3": 0.67,
            "Approval": 1.0,
        }
        for alert in recent:
            event_type_idx = {
                AlertType.DRUG_RELEASE: 0,
                AlertType.BD_DEAL: 1,
                AlertType.PIPELINE_LEAK: 2,
                AlertType.CLINICAL_SITE_DEAL: 3,
                AlertType.BE_SPEND: 4,
                AlertType.DC_SPEND: 5,
            }.get(alert.event_type, 0)
            agent_idx = (
                self.possible_agents.index(alert.agent_id)
                if alert.agent_id in self.possible_agents
                else 0
            )
            phase = 0.0
            bd_price = 0.0
            be_count = 0
            if alert.event_type == AlertType.PIPELINE_LEAK:
                phase_str = alert.details.get("new_phase", "")
                phase = phase_map.get(phase_str, 0.0)
            elif alert.event_type == AlertType.BD_DEAL:
                bd_price = float(alert.details.get("price", 0.0))
            elif alert.event_type == AlertType.CLINICAL_SITE_DEAL:
                bd_price = float(alert.details.get("price", 0.0))
            elif alert.event_type == AlertType.BE_SPEND:
                be_count = int(alert.details.get("be_count", 1))
            alert_obs.append({
                "event_type": event_type_idx,
                "agent_index": agent_idx,
                "ta_index": TA_INDEX.get(alert.therapeutic_area, 0),
                "indication": alert.indication,
                "age": self.multi_agent_game.time - alert.step,
                "phase": phase,
                "bd_price": bd_price,
                "be_count": be_count,
            })
        # Pad remaining alert slots
        while len(alert_obs) < self.max_alerts:
            alert_obs.append(dict(self._PADDING_ALERT_OBS))

        result = {
            "cash": np.array([game_state.cash], dtype=np.float32),
            "time": np.array([self.multi_agent_game.time], dtype=np.float32),
            "assets": tuple(asset_obs),
            "bd_market": tuple(bd_obs),
            "indication_markets": {
                ta: tuple(inds) for ta, inds in indication_obs.items()
            },
            "alerts": tuple(alert_obs),
        }

        if L.ta_experience_enabled:
            result["ta_experience"] = {
                ta: game_state.ta_experience.get(ta, 0.0) for ta in THERAPEUTIC_AREAS
            }

        if L.clinical_sites_enabled:
            result["clinical_sites"] = {
                "operational_sites": float(game_state.operational_sites),
                "free_sites": float(game_state.free_sites),
                "sites_in_development": float(
                    len(game_state.sites_in_development)
                ),
                "site_auction_active": int(
                    shared_market.site_auction_available()
                ),
            }

        return result

    def flatten_dict_obs(self, dict_obs: dict) -> np.ndarray:
        """
        Convert a dict observation to flattened array format.

        Parameters
        ----------
        dict_obs: dict
            Dictionary observation from _get_observation_dict().

        Returns
        -------
        np.ndarray
            Flattened observation array.

        """
        L = self._layout
        obs = np.zeros(self._obs_size, dtype=np.float32)
        pos = 0

        # Global features
        obs[pos] = dict_obs["cash"][0]
        obs[pos + 1] = dict_obs["time"][0]
        pos = 2

        if L.ta_experience_enabled:
            ta_exp = dict_obs.get("ta_experience", {})
            for i, ta in enumerate(THERAPEUTIC_AREAS):
                obs[pos + i] = ta_exp.get(ta, 0.0)
            pos += L.num_ta_exp_features

        if L.clinical_sites_enabled:
            sites_d = dict_obs.get("clinical_sites", {})
            site_off = L.offset_clinical_sites
            obs[site_off] = sites_d.get("operational_sites", 0.0)
            obs[site_off + 1] = sites_d.get("free_sites", 0.0)
            obs[site_off + 2] = sites_d.get("sites_in_development", 0.0)
            obs[site_off + 3] = sites_d.get("site_auction_active", 0.0)

        offset = L.global_features
        asset_total = L.asset_total_features
        asset_scalar = L.asset_scalar_features
        trial_feat = L.trial_features

        # Per-asset features
        for asset_d in dict_obs["assets"]:
            obs[offset] = asset_d["max_revenue"]
            obs[offset + 1] = asset_d["time_until_max_revenue"]
            obs[offset + 2] = asset_d["time_until_patent_expiry"]
            obs[offset + 3] = asset_d["pending_trial_phase"]
            obs[offset + 4] = asset_d["time_on_market"]
            obs[offset + 5] = asset_d["cost_this_step"]
            obs[offset + 6] = asset_d["revenue_this_step"]
            obs[offset + 7] = asset_d["enpv"]
            obs[offset + 8] = asset_d["eroi"]
            obs[offset + 9] = asset_d["state"]

            if L.interim_obs_enabled:
                obs[offset + L.offset_interim_signal] = asset_d.get(
                    "interim_signal", 0.0
                )
                obs[offset + L.offset_trial_progress] = asset_d.get(
                    "trial_progress", 0.0
                )

            obs[offset + L.offset_ta_index] = asset_d.get("ta_index", 0)
            obs[offset + L.offset_indication] = asset_d.get("indication", 0)

            if L.pricing_enabled:
                obs[offset + L.offset_pricing] = asset_d.get("price_multiplier", 1.0)

            if L.marketing_enabled:
                obs[offset + L.offset_brand_score] = asset_d.get("brand_score", 0.0)

            # Trial features
            trial_off = offset + asset_scalar
            off_ptrs_count = L.offset_ptrs_count
            for trial in asset_d["trials"]:
                obs[trial_off] = trial["cost_remaining"]
                obs[trial_off + 1] = trial["time_remaining"]
                obs[trial_off + 2] = trial["ptrs"]
                if L.distributional_ptrs_enabled:
                    obs[trial_off + 3] = trial.get("ptrs_expected", trial["ptrs"])
                    obs[trial_off + 4] = trial.get("ptrs_confidence", 1.0)
                    obs[trial_off + 5] = trial.get("ptrs_range_low", trial["ptrs"])
                    obs[trial_off + 6] = trial.get("ptrs_range_high", trial["ptrs"])
                if off_ptrs_count >= 0:
                    obs[trial_off + off_ptrs_count] = trial["ptrs_equiv_n_norm"]
                trial_off += trial_feat

            offset += asset_total

        # BD features
        ptrs_readings_on_flat = (
            self.ptrs_readings_config.enabled
        )
        for bd_d in dict_obs["bd_market"]:
            obs[offset] = bd_d["available"]
            obs[offset + 1] = bd_d["max_revenue"]
            obs[offset + 2] = bd_d["time_until_max_revenue"]
            obs[offset + 3] = bd_d["time_until_patent_expiry"]
            obs[offset + 4] = bd_d["ta_index"]
            obs[offset + 5] = bd_d["indication"]
            obs[offset + 6] = bd_d["enpv"]
            obs[offset + 7] = bd_d["trial_phase"]
            obs[offset + 8] = bd_d["ptrs"]
            obs[offset + 9] = bd_d.get("steps_remaining", 0.0)
            obs[offset + 10] = bd_d.get("eroi", 0.0)
            if ptrs_readings_on_flat:
                obs[offset + _BD_ASK_CASH_ENPV_IDX] = bd_d.get("ask_cash_enpv", 0.0)
                for ph in range(_BD_NUM_NON_APPROVAL_PHASES):
                    base = offset + _BD_PHASE_FEATURES_IDX + ph * 2
                    obs[base] = bd_d.get(f"ph{ph}_ptrs", 0.0)
                    obs[base + 1] = bd_d.get(f"ph{ph}_equiv_n_norm", 0.0)
            offset += self._bd_obs_size

        # Indication market features
        for ta in THERAPEUTIC_AREAS:
            for ind_d in dict_obs["indication_markets"][ta]:
                obs[offset] = ind_d["exclusivity_remaining"]
                obs[offset + 1] = ind_d["my_avg_share"]
                obs[offset + 2] = ind_d["first_mover"]
                obs[offset + 3] = ind_d["my_drugs"]
                obs[offset + 4] = ind_d["competitor_drugs"]
                if L.marketing_enabled:
                    obs[offset + 5] = ind_d.get("demand_multiplier", 1.0)
                offset += self._indication_features

        # Alert features (-1 event_type = padding, all zeros)
        for alert_d in dict_obs["alerts"]:
            event_type = alert_d["event_type"]
            if event_type >= 0:
                # one-hot event type (0..5): drug_release, bd_deal,
                # pipeline_leak, clinical_site_deal, be_spend, dc_spend
                if 0 <= event_type <= 5:
                    obs[offset + event_type] = 1.0
                obs[offset + 6] = alert_d["agent_index"]
                obs[offset + 7] = alert_d["ta_index"]
                obs[offset + 8] = alert_d["indication"]
                obs[offset + 9] = alert_d["age"]
                obs[offset + 10] = alert_d["phase"]
                obs[offset + 11] = alert_d["bd_price"]
                obs[offset + 12] = alert_d["be_count"]
            offset += _ALERT_FEATURES

        return obs

    def unflatten_to_dict_obs(self, flat_obs: np.ndarray) -> dict:
        """
        Convert a flattened observation array to dict format.

        Parameters
        ----------
        flat_obs: np.ndarray
            Flattened observation array.

        Returns
        -------
        dict
            Dictionary observation.

        """
        L = self._layout
        dist_on = L.distributional_ptrs_enabled
        pos = 0

        cash = float(flat_obs[pos])
        time_val = float(flat_obs[pos + 1])
        pos = 2

        result: dict[str, Any] = {
            "cash": np.array([cash], dtype=np.float32),
            "time": np.array([time_val], dtype=np.float32),
        }

        if L.ta_experience_enabled:
            result["ta_experience"] = {
                ta: float(flat_obs[pos + i]) for i, ta in enumerate(THERAPEUTIC_AREAS)
            }
            pos += L.num_ta_exp_features

        offset = L.global_features
        asset_total = L.asset_total_features
        asset_scalar = L.asset_scalar_features
        trial_feat = L.trial_features

        # Assets
        assets = []
        for _ in range(self.max_num_assets):
            trials = []
            trial_off = offset + asset_scalar
            for _ in range(len(TrialPhase)):
                t = {
                    "cost_remaining": float(flat_obs[trial_off]),
                    "time_remaining": int(flat_obs[trial_off + 1]),
                    "ptrs": float(flat_obs[trial_off + 2]),
                }
                if dist_on:
                    t["ptrs_expected"] = float(flat_obs[trial_off + 3])
                    t["ptrs_confidence"] = float(flat_obs[trial_off + 4])
                    t["ptrs_range_low"] = float(flat_obs[trial_off + 5])
                    t["ptrs_range_high"] = float(flat_obs[trial_off + 6])
                trials.append(t)
                trial_off += trial_feat

            asset_d: dict[str, Any] = {
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
                "indication": int(flat_obs[offset + L.offset_indication]),
                "trials": tuple(trials),
            }
            if L.interim_obs_enabled:
                asset_d["interim_signal"] = float(
                    flat_obs[offset + L.offset_interim_signal]
                )
                asset_d["trial_progress"] = float(
                    flat_obs[offset + L.offset_trial_progress]
                )
            if L.pricing_enabled:
                asset_d["price_multiplier"] = float(flat_obs[offset + L.offset_pricing])
            if L.marketing_enabled:
                asset_d["brand_score"] = float(flat_obs[offset + L.offset_brand_score])
            assets.append(asset_d)
            offset += asset_total

        result["assets"] = tuple(assets)

        # BD market
        bd_obs = []
        ptrs_readings_on_unflatten = (
            self.ptrs_readings_config.enabled
        )
        for _ in range(self.bd_max_slots):
            entry = {
                "available": int(flat_obs[offset]),
                "max_revenue": float(flat_obs[offset + 1]),
                "time_until_max_revenue": int(flat_obs[offset + 2]),
                "time_until_patent_expiry": int(flat_obs[offset + 3]),
                "ta_index": int(flat_obs[offset + 4]),
                "indication": int(flat_obs[offset + 5]),
                "enpv": float(flat_obs[offset + 6]),
                "trial_phase": int(flat_obs[offset + 7]),
                "ptrs": float(flat_obs[offset + 8]),
                "steps_remaining": float(flat_obs[offset + 9]),
                "eroi": float(flat_obs[offset + 10]),
            }
            if ptrs_readings_on_unflatten:
                entry["ask_cash_enpv"] = float(flat_obs[offset + _BD_ASK_CASH_ENPV_IDX])
                for ph in range(_BD_NUM_NON_APPROVAL_PHASES):
                    base = offset + _BD_PHASE_FEATURES_IDX + ph * 2
                    entry[f"ph{ph}_ptrs"] = float(flat_obs[base])
                    entry[f"ph{ph}_equiv_n_norm"] = float(flat_obs[base + 1])
            bd_obs.append(entry)
            offset += self._bd_obs_size
        result["bd_market"] = tuple(bd_obs)

        # Indication markets
        indication_obs: dict[str, list] = {}
        for ta in THERAPEUTIC_AREAS:
            ta_inds = []
            for _ in range(self.max_indications_per_ta):
                ind_d: dict = {
                    "exclusivity_remaining": float(flat_obs[offset]),
                    "my_avg_share": float(flat_obs[offset + 1]),
                    "first_mover": int(flat_obs[offset + 2]),
                    "my_drugs": int(flat_obs[offset + 3]),
                    "competitor_drugs": int(flat_obs[offset + 4]),
                }
                if L.marketing_enabled:
                    ind_d["demand_multiplier"] = float(flat_obs[offset + 5])
                ta_inds.append(ind_d)
                offset += self._indication_features
            indication_obs[ta] = tuple(ta_inds)
        result["indication_markets"] = indication_obs

        # Alerts
        alert_obs = []
        for _ in range(self.max_alerts):
            # Decode one-hot event type (0..5: drug_release, bd_deal,
            # pipeline_leak, clinical_site_deal, be_spend, dc_spend; -1 = padding)
            event_type = -1
            for k in range(6):
                if flat_obs[offset + k] == 1.0:
                    event_type = k
                    break
            alert_obs.append({
                "event_type": event_type,
                "agent_index": int(flat_obs[offset + 6]),
                "ta_index": int(flat_obs[offset + 7]),
                "indication": int(flat_obs[offset + 8]),
                "age": int(flat_obs[offset + 9]),
                "phase": float(flat_obs[offset + 10]),
                "bd_price": float(flat_obs[offset + 11]),
                "be_count": int(flat_obs[offset + 12]),
            })
            offset += _ALERT_FEATURES
        result["alerts"] = tuple(alert_obs)

        return result

    def _get_info(self, agent: str) -> dict:
        """Build info dict for an agent."""
        game_state = self.multi_agent_game.agent_states[agent]
        shared_market = self.multi_agent_game.shared_market

        return {
            "cash": game_state.cash,
            "time": self.multi_agent_game.time,
            "num_assets": len(game_state.assets),
            "bankrupt": game_state.bankrupt,
            "ta_experience": dict(game_state.ta_experience),
            "capacity_ratio": game_state.capacity_ratio,
            "indication_names": dict(shared_market.indication_name_map),
            "indications_per_ta": shared_market.indications_per_ta,
        }
