import importlib
import os
from typing import Any

import upath
import yaml
from pydantic import BaseModel, field_validator, model_validator

from pyxis_portfolio_challenge import PROJECT_ROOT


def fibonacci_cost(n: int, base: float) -> float:
    """
    Total cost for n concurrent readings using Fibonacci multipliers.

    Cumulative sums: n=1→1×, n=2→2×, n=3→4×, n=4→7×, n=5→12× base.
    """
    a, b, total = 1, 1, 0
    for _ in range(n):
        total += a
        a, b = b, a + b
    return total * base


def fibonacci_number(n: int) -> int:
    """
    The n-th Fibonacci number (1-indexed): fib(1)=1, fib(2)=1, fib(3)=2, ...

    Used as the *incremental* multiplier for clinical-site purchases, as opposed
    to ``fibonacci_cost`` which sums the sequence for concurrent readings.
    """
    if n < 1:
        raise ValueError("fibonacci_number is 1-indexed; n must be >= 1")
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a


class DropActionConfig(BaseModel):
    """
    Configuration for the standalone drop action.

    When enabled, the investment action space becomes ternary per asset:
      0 = NONE  (don't invest / continue as-is)
      1 = INVEST (start/continue at standard level; only valid for Idle)
      2 = DROP   (remove asset from portfolio entirely; valid for any state)

    Drop fee = round(drop_price_fraction * cost_remaining / drop_price_rounding)
               * drop_price_rounding
    """

    enabled: bool
    # fraction of cost_remaining charged on drop
    drop_price_fraction: float
    # round drop fee to nearest N GBP (1 = no rounding)
    drop_price_rounding: int

    @field_validator("drop_price_fraction")
    @classmethod
    def _fraction_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("drop_price_fraction must be between 0.0 and 1.0")
        return v

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"

    def calculate_drop_fee(self, cost_remaining: float) -> float:
        """Fee charged for dropping an asset whose trial has this cost_remaining."""
        if self.drop_price_fraction <= 0:
            return 0.0
        raw_fee = self.drop_price_fraction * cost_remaining
        rounding = self.drop_price_rounding
        if rounding > 1:
            return round(raw_fee / rounding) * rounding
        return raw_fee


class ClinicalSitesConfig(BaseModel):
    """
    Configuration for the clinical sites capacity feature.

    Clinical sites are concurrency slots that gate active drug trials: each
    operational site can host one InDevelopment asset at a time (Model B — a
    site is freed whenever its asset leaves InDevelopment, including the one-step
    gap between phases). This hard-caps trial throughput so excess cash cannot
    buy unlimited concurrent trials (the runaway-cash problem).

    Agents grow capacity two ways:
      * ``upgrade`` action: buy one site per step at a Fibonacci-scaled cost,
        usable only after a ``site_development_steps`` build delay.
      * site auction: a PvP first-price sealed-bid auction every
        ``auction_interval_steps`` for an immediately-usable site.

    When an agent requests more trial starts than it has free sites, allocation
    is arbitrated per ``agent_priority`` (see design doc §2): False = ascending
    asset-index order (default); True = the agent emits a per-asset ``priority``
    scalar action and sites go to the highest scores. Ungranted requests are
    costless no-ops; the step is never errored.
    """

    enabled: bool

    # Operational sites each agent starts the episode with.
    starting_sites: int
    # Base cost (GBP) for the Fibonacci purchase curve.
    purchase_base_cost: float
    # Round purchase cost to nearest N GBP (1 = no rounding).
    purchase_cost_rounding: int
    # Build delay (steps) before a purchased site becomes operational.
    site_development_steps: int

    # Over-request arbitration: False = pure asset-index order (default);
    # True = agent emits a per-asset priority scalar action.
    agent_priority: bool
    # Entropy weight for the priority Gaussian head (only used when
    # agent_priority is True). 1.0 = neutral / stock-SB3 weighting.
    priority_entropy_weight: float

    # Site auction (PvP)
    auction_enabled: bool
    # A site is auctioned every N steps (= years, since 1 step = 1 year).
    auction_interval_steps: int
    # No site auction before this step (warmup). First auction fires exactly on
    # this step, then every ``auction_interval_steps`` after (e.g. 10, 30, 50).
    auction_min_step: int
    # Continuous bid cap (units matching bd_max_bid).
    site_max_bid: float
    # Auction reserve price as a fraction of the site's value anchor
    # (``purchase_base_cost``, the Fibonacci build-cost base). If the winning
    # bid is below this reserve, no site is sold that round (nobody wins, no
    # cash is charged). 0.0 disables the reserve.
    auction_reserve_fraction: float

    @field_validator("starting_sites")
    @classmethod
    def _starting_sites_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("starting_sites must be >= 0")
        return v

    @field_validator("site_development_steps", "auction_interval_steps")
    @classmethod
    def _positive_step_counts(cls, v: int) -> int:
        if v < 1:
            raise ValueError("step-count fields must be >= 1")
        return v

    @field_validator("auction_reserve_fraction")
    @classmethod
    def _reserve_fraction_in_unit_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("auction_reserve_fraction must be in [0.0, 1.0]")
        return v

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"

    def purchase_cost(self, total_owned: int) -> float:
        """
        Fibonacci-scaled cash cost to buy the next site.

        ``total_owned`` is how many sites the agent already owns (operational +
        in-development; auction-won sites count as owned). The k-th purchase
        beyond the starting endowment costs ``purchase_base_cost * fib(k)``,
        i.e. 1×, 1×, 2×, 3×, 5×, 8×, ... for the 1st, 2nd, 3rd, ... purchase
        (the full Fibonacci sequence, including both leading 1s).
        """
        k = max(1, total_owned - self.starting_sites + 1)
        raw = self.purchase_base_cost * fibonacci_number(k)
        rounding = self.purchase_cost_rounding
        if rounding > 1:
            return round(raw / rounding) * rounding
        return raw


class PtrsReadingsConfig(BaseModel):
    """
    Configuration for per-asset PTRS research readings feature.

    Agents pay Fibonacci-scaled costs to commission
    additional PTRS assessments on individual assets. Each reading returns a
    logit-normal sample centred on the true PTRS, delivered one step later.
    Multiple readings reduce flip probability but cannot analytically recover
    true PTRS. Phase noise scales by distance: next pending trial = 1×σ_base,
    one after = 1.5×, further phases use noise_multipliers[i].
    """

    enabled: bool
    # Fraction of asset.trial.cost_remaining charged per reading (base)
    cost_fraction: float
    # Round base to nearest N GBP before Fibonacci (1 = no rounding)
    cost_rounding: int
    # Upper bound for MultiDiscrete; cash gates the real limit
    action_space_max_readings: int
    sigma_logit_base: float
    # Episode-level truth spread (logit-normal sigma) around the file PTRS: the
    # per-episode _true_ptrs is drawn as sigmoid(logit(file_ptrs) + N(0, sigma_ep^2)).
    # Memorization bound: must equal sigma_logit_base so that knowing the file value
    # is worth exactly one reading (both carry 1/sigma^2 precision). A smaller value
    # makes the file-value lookup worth (sigma_logit_base/sigma_ep)^2 readings.
    # Required (no None fallback).
    sigma_ep: float
    noise_multipliers: list[float]
    # Normalisation cap for sample count in observation
    max_sample_obs: int

    def reading_base_cost(self, trial_cost_remaining: float) -> float:
        """
        Base cash cost of a single PTRS reading, before Fibonacci scaling.

        A fraction of the trial's remaining cost, optionally rounded to the
        nearest ``cost_rounding`` GBP. Shared by the engine step and the HTTP
        response layer so the cost shown next to a reading control can never
        drift from what the step actually charges.
        """
        raw = self.cost_fraction * trial_cost_remaining
        r = self.cost_rounding
        return round(raw / r) * r if r > 1 else raw

    def reading_cost_curve(self, trial_cost_remaining: float) -> list[float]:
        """
        Cumulative cost of commissioning 1..action_space_max_readings readings.

        Entry ``k-1`` is the total cost of ``k`` concurrent readings this step
        (Fibonacci-scaled off the base cost). Drives the live stepper's escalating
        price; shared with the engine (``fibonacci_cost``) so the previewed cost
        matches what the step charges.
        """
        base = self.reading_base_cost(trial_cost_remaining)
        return [
            fibonacci_cost(k, base)
            for k in range(1, self.action_space_max_readings + 1)
        ]

    def effective_readings(self, total_precision: float) -> float:
        """
        Precision-weighted equivalent number of base-σ readings.

        Mirrors the observation feature (``total_precision × sigma_logit_base²``).
        Readings taken at greater phase distance carry more noise (lower
        precision) and so count for less than one whole reading — hence
        "effective". Shared so the value shown matches what the agent observes.
        """
        return total_precision * self.sigma_logit_base**2

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"


class MarketingConfig(BaseModel):
    """
    Configuration for the marketing spend feature.

    Two complementary mechanics per step:
      - Demand creation: agents top up a shared indication-level pool multiplier.
        Revenue for all drugs in the indication is scaled by the multiplier.
        No hard cap — asymptotes naturally via decay. Slow decay each step.
      - Brand equity: agents build a per-drug quality score that boosts their
        drug's quality in the market share calculation. Faster decay; applies
        to any drug regardless of trial phase.

    Both actions are binary per step (0=don't spend, 1=spend fixed cost).
    Cost = fraction of the drug's max_revenue.
    Marketing is a no-op when disable_market_share_competition is true.
    """

    enabled: bool

    # Demand creation parameters
    dc_cost_fraction: float  # cost per top-up as fraction of drug max_revenue
    dc_step_boost: float  # fixed additive increment to demand_multiplier per spend
    dc_decay_rate: float  # per-step multiplicative decay of headroom toward 1.0

    # Brand equity parameters
    be_cost_fraction: float  # cost per step as fraction of drug max_revenue
    be_boost: float  # brand_score increment per spend action
    be_decay_rate: float  # per-step multiplicative decay of brand_score
    be_effectiveness: float  # quality multiplier per unit of (brand_score − floor)

    @field_validator("dc_step_boost", "dc_decay_rate")
    @classmethod
    def _dc_positive(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("dc_step_boost and dc_decay_rate must be positive")
        return v

    @field_validator("be_decay_rate")
    @classmethod
    def _be_decay_in_range(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("be_decay_rate must be in (0, 1)")
        return v

    def dc_cost(self, peak_revenue: float) -> float:
        """
        Flat demand-creation cost to size one indication this step.

        Anchored to the pool-wide static peak max_revenue (passed in) rather
        than any single drug, so the cost is the same for every indication and
        is charged unconditionally when the indication is sized. Single source
        of truth for the DC cost -- the engine step and the HTTP response layer
        both call this so the displayed cost can never drift from the charge.
        """
        return self.dc_cost_fraction * peak_revenue

    def be_cost(self, asset_max_revenue: float) -> float:
        """
        Brand-equity cost to boost one asset this step (scales with drug size).

        Shared by the engine step and the HTTP response layer so the cost shown
        next to a drug always matches what is charged.
        """
        return self.be_cost_fraction * asset_max_revenue

    def next_brand_score(
        self, current_score: float, floor: float, spend: bool
    ) -> float:
        """
        Project a drug's brand score after the next step, given a spend decision.

        Mirrors the engine step (GameState.step): add ``be_boost`` if spending,
        then decay the score multiplicatively toward its floor. Shared with the
        HTTP response layer so the forward-looking value shown before committing
        a brand-equity push matches what the engine will actually produce.
        """
        boosted = current_score + (self.be_boost if spend else 0.0)
        return max(floor, boosted * (1.0 - self.be_decay_rate))

    def next_demand_multiplier(
        self, current_multiplier: float, spend: bool
    ) -> float:
        """
        Project an indication's demand multiplier after the next step.

        Mirrors the engine (apply_demand_creation then SharedMarketState.
        advance_time): add ``dc_step_boost`` if spending, then decay the headroom
        (multiplier − 1) toward the 1.0 base. Shared with the HTTP response layer
        so the forward-looking value shown before committing a demand-creation
        spend matches what the engine will actually produce.
        """
        boosted = current_multiplier + (self.dc_step_boost if spend else 0.0)
        headroom = boosted - 1.0
        return 1.0 + headroom * (1.0 - self.dc_decay_rate)

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"


class ApprovalPhaseConfig(BaseModel):
    """Configuration for the regulatory approval phase after Phase 3."""

    enabled: bool
    duration_min: int  # Minimum approval duration in steps
    duration_max: int  # Maximum approval duration in steps
    success_rate_min: float  # Minimum PTRS for approval
    success_rate_max: float  # Maximum PTRS for approval
    cost: float  # Filing fees

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"


class MultiAgentConfig(BaseModel):
    """Multi-agent specific configuration parameters."""

    enabled: bool
    num_agents: int
    # BD market parameters
    bd_enabled: bool
    bd_assets_dir: upath.UPath
    bd_eval_assets_dir: upath.UPath
    bd_base_lambda: float  # Base Poisson λ for BD appearance
    bd_leak_lambda_boost: float  # Added λ per recent leak
    bd_min_step: int  # No BD before this step
    # Continuous bid: per-slot cash bid in GBP millions. <= 0 = pass; highest
    # bid wins and pays its own bid. bd_max_bid is the action-space upper bound;
    # real affordability is gated by cash (an overbid can bankrupt the winner).
    bd_max_bid: float  # Action-space cap for a BD bid, in GBP millions
    # Auction reserve price as a fraction of the BD asset's value anchor (its
    # ``cash_enpv``). If the winning bid is below this reserve, no one wins the
    # slot: the asset is not sold and stays on the market for its remaining
    # persistence (``bd_persist_steps``). 0.0 disables the reserve.
    bd_reserve_fraction: float
    bd_max_slots: int  # Max BD assets per step (start with 1)
    # Steps an unwon BD asset stays on market (1 = single-step behaviour)
    bd_persist_steps: int
    bd_phase_weights: list[float]  # Phase 1/2/3 sampling weights
    bd_indication_activity_bias: float  # Weight of activity vs uniform
    # Competition parameters
    exclusivity_period: int
    first_mover_bonus: float
    disable_market_share_competition: bool
    # Intelligence parameters
    alert_history_length: int
    alerts_per_agent: int
    # Event-driven leak probabilities: [Phase 1→2, Phase 2→3, Phase 3→Approval]
    leak_phase_probabilities: list[float]
    # Marketing-spend leak probabilities (opponent sees the spend, not the amount):
    # be = brand equity per drug's indication; dc = demand creation per indication
    be_leak_probability: float
    dc_leak_probability: float
    # Minimum agent count for demand-creation leaks to activate. DC boosts a
    # public per-indication demand multiplier that is already in the observation,
    # so with few agents an opponent can attribute a rise to the only other
    # spender for free and the leak is redundant. It self-activates once
    # attribution stops being trivial (num_agents >= this threshold).
    dc_leak_min_agents: int
    # Multi-agent reward type: "absolute", "relative_rank", "zero_sum"
    reward_type: str
    reward_scale: float
    # Indication-based market segmentation
    target_drugs_per_indication: float
    on_market_fraction: float
    max_indications_per_ta: int
    indication_spread: float
    indication_drift_speed: float
    # Market congestion penalty
    congestion_exponent: float  # α in 1/n^α, 0 = disabled
    congestion_ramp_steps: (
        int  # Entry positions to reach full penalty (1 = binary incumbent/challenger)
    )
    # Fraction of full penalty applied to incumbent (0 = protected)
    congestion_incumbent_penalty: float

    @field_validator("bd_reserve_fraction")
    @classmethod
    def _bd_reserve_fraction_in_unit_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("bd_reserve_fraction must be in [0.0, 1.0]")
        return v

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"

    def compute_indications_per_ta(self, equilibrium_num_assets: int) -> int:
        """Compute the number of indications per TA based on game parameters."""
        num_tas = 3
        raw = (
            self.num_agents
            * (equilibrium_num_assets / num_tas)
            * self.on_market_fraction
            / self.target_drugs_per_indication
        )
        return min(max(1, round(raw)), self.max_indications_per_ta)


class Config(BaseModel):
    """Configuration model for the investment game environment."""

    equilibrium_num_assets: int
    max_num_assets: int
    asset_arrival_sensitivity_below: float
    asset_arrival_sensitivity_above: float
    starting_cash: float
    horizon: int
    reinvestment_percentage: float
    trial_cost_multiplier: float
    shuffle_order: bool
    mask_first_order_assets: bool
    mask_negative_enpv_assets: bool
    auto_center_rewards: bool
    auto_center_calibration_steps: int
    flatten_obs: bool
    reward_fn: dict[str, Any]
    training_data_dir: upath.UPath
    evaluation_data_dir: upath.UPath
    evaluation_metrics: list[dict[str, str]]
    num_eval_episodes: int
    eval_initial_seed: int
    warmup_on_reset_steps: int
    warmup_on_reset_policy: str

    # Standalone drop action
    drop_action: DropActionConfig

    # Clinical sites capacity feature (hard throughput cap; runaway-cash sink)
    clinical_sites: ClinicalSitesConfig

    # Marketing spend feature (demand creation + brand equity)
    marketing: MarketingConfig

    # Per-asset PTRS research readings feature
    ptrs_readings: PtrsReadingsConfig

    # Approval phase configuration
    approval_phase: ApprovalPhaseConfig

    # Multi-agent environment configuration
    multi_agent: MultiAgentConfig

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"

    @model_validator(mode="after")
    def resolve_relative_paths(self):
        """Resolve relative asset paths against PROJECT_ROOT."""
        for field in ("training_data_dir", "evaluation_data_dir"):
            path = getattr(self, field)
            if not path.is_absolute():
                object.__setattr__(self, field, PROJECT_ROOT / path)
        ma = self.multi_agent
        if not ma.bd_assets_dir.is_absolute():
            object.__setattr__(ma, "bd_assets_dir", PROJECT_ROOT / ma.bd_assets_dir)
        if not ma.bd_eval_assets_dir.is_absolute():
            object.__setattr__(
                ma, "bd_eval_assets_dir", PROJECT_ROOT / ma.bd_eval_assets_dir
            )
        return self

    @field_validator("reinvestment_percentage", mode="after")
    def validate_reinvestment_percentage(cls, v):
        """Validate that reinvestment_percentage is between 0.0 and 1.0."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("reinvestment_percentage must be between 0.0 and 1.0")
        return v


def from_yaml(
    path: str = f"{PROJECT_ROOT}/pyxis_portfolio_challenge/config.yaml",
) -> Config:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.

    Returns
    -------
    Config
        The loaded configuration as a Config object.

    """
    with open(os.path.abspath(path), "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


def instantiate_from_config(config: Any):
    """
    Instantiate an object from a configuration dictionary.

    This function is now recursive to handle nested objects.

    example generic:
        The following could be defined in the yaml config file:
        ```yaml
        _target_: module.ClassName
        : arg1value
        kwarg1: kwarg1value
        kwarg2: kwarg2value
        ```

        This would come in as the dictionary:
        {"_target_": "module.ClassName", "": arg1value, "kwarg1": kwarg1value, "kwarg2": kwarg2value}

        This function would then import module.ClassName and instantiate it as:
        module.ClassName(arg1value, kwarg1=kwarg1value, kwarg2=kwarg2value)

    specific use case:
        In our use case, this is used to instantiate reward functions for the investment game environment.

        yaml config example:
        ```yaml
        reward_fn:
            _target_: pyxis_portfolio_challenge.environment.reward_functions.LegacyStaticNPVReward

        ```
    """  # noqa: E501
    if isinstance(config, list):
        return [instantiate_from_config(c) for c in config]
    if not isinstance(config, dict) or "_target_" not in config:
        return config

    target = config["_target_"]
    # Split to module and class name
    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    klass = getattr(module, class_name)

    # Recursively instantiate arguments
    kwargs = {
        k: instantiate_from_config(v)
        for k, v in config.items()
        if k != "_target_" and k != ""
    }
    args = [instantiate_from_config(v) for k, v in config.items() if k == ""]

    return klass(*args, **kwargs)


config = from_yaml()
