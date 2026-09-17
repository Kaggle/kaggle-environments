"""Shared market state for multi-agent competitive environment."""

from __future__ import annotations

import logging
import math
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, ConfigDict, PrivateAttr

from pyxis_portfolio_challenge.game.asset import AssetState, DrugAsset
from pyxis_portfolio_challenge.game.asset_generators import AssetGeneratorBase
from pyxis_portfolio_challenge.rng import get_game_rng

if TYPE_CHECKING:
    from pyxis_portfolio_challenge.config import MarketingConfig

logger = logging.getLogger(__name__)

# Human-readable indication names per TA (~25 each for future scaling)
INDICATION_NAMES: dict[str, list[str]] = {
    "oncology": [
        "solid tumors",
        "hematologic malignancies",
        "immuno-oncology",
        "targeted therapy",
        "neuro-oncology",
        "GI oncology",
        "breast cancer",
        "lung cancer",
        "prostate cancer",
        "melanoma",
        "lymphoma",
        "renal cell carcinoma",
        "ovarian cancer",
        "pancreatic cancer",
        "bladder cancer",
        "head and neck cancer",
        "sarcoma",
        "thyroid cancer",
        "liver cancer",
        "endometrial cancer",
        "glioblastoma",
        "mesothelioma",
        "multiple myeloma",
        "CLL/SLL",
        "AML",
    ],
    "respiratory and immunology": [
        "asthma",
        "COPD",
        "pulmonary fibrosis",
        "allergic rhinitis",
        "cystic fibrosis",
        "lupus",
        "rheumatoid arthritis",
        "sarcoidosis",
        "pulmonary hypertension",
        "interstitial lung disease",
        "psoriasis",
        "atopic dermatitis",
        "Crohn's disease",
        "ulcerative colitis",
        "multiple sclerosis",
        "ankylosing spondylitis",
        "vasculitis",
        "myasthenia gravis",
        "bronchiectasis",
        "alpha-1 antitrypsin deficiency",
        "eosinophilic esophagitis",
        "chronic sinusitis",
        "food allergy",
        "graft-versus-host disease",
        "systemic sclerosis",
    ],
    "vaccines and infectious disease": [
        "influenza",
        "RSV",
        "pneumococcal",
        "HIV",
        "malaria",
        "hepatitis B",
        "tuberculosis",
        "COVID",
        "dengue",
        "meningococcal",
        "HPV",
        "Ebola",
        "Zika",
        "norovirus",
        "C. difficile",
        "CMV",
        "shingles",
        "cholera",
        "typhoid",
        "rabies",
        "Lyme disease",
        "chikungunya",
        "hepatitis C",
        "anthrax",
        "Marburg",
    ],
}


def indication_key(ta: str, indication: int) -> str:
    """Create a unique key for a TA-indication pair."""
    return f"{ta}:{indication}"


class AlertType(str, Enum):
    """Types of alerts that can be broadcast to agents."""

    DRUG_RELEASE = "drug_release"
    BD_DEAL = "bd_deal"
    PIPELINE_LEAK = "pipeline_leak"
    CLINICAL_SITE_DEAL = "clinical_site_deal"
    BE_SPEND = "be_spend"
    DC_SPEND = "dc_spend"


class Alert(BaseModel):
    """Intelligence about competitor actions broadcast to all agents."""

    model_config = ConfigDict(frozen=True)

    step: int
    event_type: AlertType
    agent_id: str
    therapeutic_area: str
    indication: int = 0
    details: dict = {}


class TAMarketState(BaseModel):
    """Per-Therapeutic-Area market state tracking."""

    model_config = ConfigDict(validate_assignment=True)

    therapeutic_area: str
    first_mover_agent: Optional[str] = None
    first_mover_drug_id: Optional[uuid.UUID] = None
    exclusivity_start_time: Optional[int] = None
    exclusivity_duration: int = 4
    # Maps agent_id -> list of asset_ids that are OnMarket in this TA
    active_drugs: dict[str, list[uuid.UUID]] = {}
    # FIFO order of on-market drug IDs (position 0 = incumbent)
    entry_order: list[uuid.UUID] = []

    def is_in_exclusivity(self, current_time: int) -> bool:
        """Check if TA is currently in exclusivity period."""
        if self.first_mover_drug_id is None or self.exclusivity_start_time is None:
            return False
        return current_time < self.exclusivity_start_time + self.exclusivity_duration

    def exclusivity_remaining(self, current_time: int) -> int:
        """Return steps remaining in exclusivity period."""
        if not self.is_in_exclusivity(current_time):
            return 0
        return (self.exclusivity_start_time + self.exclusivity_duration) - current_time


class IndicationMarketState(BaseModel):
    """Per-indication market state tracking for granular competition."""

    model_config = ConfigDict(validate_assignment=True)

    therapeutic_area: str
    indication: int
    indication_name: str
    first_mover_agent: Optional[str] = None
    first_mover_drug_id: Optional[uuid.UUID] = None
    exclusivity_start_time: Optional[int] = None
    exclusivity_duration: int = 4
    # Maps agent_id -> list of asset_ids that are OnMarket in this indication
    active_drugs: dict[str, list[uuid.UUID]] = {}
    # FIFO order of on-market drug IDs (position 0 = incumbent)
    entry_order: list[uuid.UUID] = []

    # Shared demand multiplier: all drugs in this indication earn
    # revenue × demand_multiplier.
    # Boosted by demand creation spend; decays slowly toward 1.0 each step.
    demand_multiplier: float = 1.0

    def is_in_exclusivity(self, current_time: int) -> bool:
        """Check if indication is currently in exclusivity period."""
        if self.first_mover_drug_id is None or self.exclusivity_start_time is None:
            return False
        return current_time < self.exclusivity_start_time + self.exclusivity_duration

    def exclusivity_remaining(self, current_time: int) -> int:
        """Return steps remaining in exclusivity period."""
        if not self.is_in_exclusivity(current_time):
            return 0
        return (self.exclusivity_start_time + self.exclusivity_duration) - current_time


# Therapeutic areas in the game
THERAPEUTIC_AREAS = [
    "oncology",
    "respiratory and immunology",
    "vaccines and infectious disease",
]


class SharedMarketState(BaseModel):
    """Shared market state across all agents in multi-agent environment."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Current BD assets available for bidding (one per slot, empty list if none)
    current_bd_assets: list[DrugAsset] = []

    # Age tracking for persistent BD assets:
    # str(asset.id) → steps on market (0 = just spawned)
    bd_asset_ages: dict[str, int] = {}

    # Steps an unwon BD asset stays on market; copied from config at initialize() time
    bd_persist_steps: int = 1

    # Backward-compatible property
    @property
    def current_bd_asset(self) -> Optional[DrugAsset]:
        """First BD asset (backward compat for single-slot code paths)."""
        return self.current_bd_assets[0] if self.current_bd_assets else None

    # Market state per therapeutic area (kept for backward compatibility)
    ta_markets: dict[str, TAMarketState] = {}

    # Market state per indication (keyed by "ta:indication_index")
    indication_markets: dict[str, IndicationMarketState] = {}

    # Number of indications per TA (0 = not using indications)
    indications_per_ta: int = 0

    # Mapping from "ta:indication_index" -> human-readable name
    indication_name_map: dict[str, str] = {}

    # Alert history (last N steps)
    alerts: list[Alert] = []
    alert_history_length: int = 5

    # Competition configuration
    exclusivity_period: int = 4
    first_mover_bonus: float = 0.30
    disable_market_share_competition: bool = False
    congestion_exponent: float
    congestion_ramp_steps: int
    congestion_incumbent_penalty: float

    # BD configuration
    bd_enabled: bool = False
    bd_base_lambda: float = 0.3
    bd_leak_lambda_boost: float = 0.3
    bd_min_step: int = 5
    bd_max_bid: float = 10_000.0  # action-space cap in GBP millions
    bd_phase_weights: list[float] = [0.2, 0.4, 0.4]
    bd_indication_activity_bias: float = 0.8

    # Event-driven leak configuration
    leak_phase_probabilities: list[float] = [0.2, 0.5, 0.7]

    # Marketing-spend leak probabilities: chance an opponent sees that this agent
    # spent brand equity on a drug (be) / demand creation on an indication (dc).
    be_leak_probability: float = 0.8
    dc_leak_probability: float = 0.8

    # Clinical-site auction configuration (PvP). One immediately-usable site is
    # offered every ``site_auction_interval_steps``, gated by a warmup step.
    site_auction_enabled: bool = False
    site_auction_interval_steps: int = 20
    site_auction_min_step: int = 0

    # Current game time
    time: int = 0

    # Private attributes
    _bd_asset_generator: Optional[AssetGeneratorBase] = PrivateAttr(default=None)

    @classmethod
    def initialize(
        cls,
        exclusivity_period: int,
        first_mover_bonus: float,
        alert_history_length: int,
        disable_market_share_competition: bool,
        num_indications_per_ta: int,
        bd_enabled: bool,
        bd_base_lambda: float,
        bd_leak_lambda_boost: float,
        bd_min_step: int,
        bd_max_bid: float,
        bd_phase_weights: list[float] | None,
        bd_indication_activity_bias: float,
        congestion_exponent: float,
        congestion_ramp_steps: int,
        congestion_incumbent_penalty: float,
        bd_persist_steps: int = 1,
        leak_phase_probabilities: list[float] | None = None,
        be_leak_probability: float = 0.8,
        dc_leak_probability: float = 0.8,
        site_auction_enabled: bool = False,
        site_auction_interval_steps: int = 20,
        site_auction_min_step: int = 0,
    ) -> "SharedMarketState":
        """Initialize a new shared market state."""
        rng = get_game_rng()

        if bd_phase_weights is None:
            bd_phase_weights = [0.2, 0.4, 0.4]
        if leak_phase_probabilities is None:
            leak_phase_probabilities = [0.2, 0.5, 0.7]

        # Build indication markets and name map
        ind_markets: dict[str, IndicationMarketState] = {}
        ind_name_map: dict[str, str] = {}

        if num_indications_per_ta > 0:
            for ta in THERAPEUTIC_AREAS:
                pool = list(INDICATION_NAMES[ta])
                sampled = rng.sample(pool, min(num_indications_per_ta, len(pool)))
                for idx, name in enumerate(sampled):
                    key = indication_key(ta, idx)
                    ind_markets[key] = IndicationMarketState(
                        therapeutic_area=ta,
                        indication=idx,
                        indication_name=name,
                        exclusivity_duration=exclusivity_period,
                    )
                    ind_name_map[key] = name

        state = cls(
            exclusivity_period=exclusivity_period,
            first_mover_bonus=first_mover_bonus,
            alert_history_length=alert_history_length,
            disable_market_share_competition=disable_market_share_competition,
            bd_enabled=bd_enabled,
            bd_base_lambda=bd_base_lambda,
            bd_leak_lambda_boost=bd_leak_lambda_boost,
            bd_min_step=bd_min_step,
            bd_max_bid=bd_max_bid,
            bd_phase_weights=bd_phase_weights,
            bd_indication_activity_bias=bd_indication_activity_bias,
            bd_persist_steps=bd_persist_steps,
            leak_phase_probabilities=leak_phase_probabilities,
            be_leak_probability=be_leak_probability,
            dc_leak_probability=dc_leak_probability,
            site_auction_enabled=site_auction_enabled,
            site_auction_interval_steps=site_auction_interval_steps,
            site_auction_min_step=site_auction_min_step,
            congestion_exponent=congestion_exponent,
            congestion_ramp_steps=congestion_ramp_steps,
            congestion_incumbent_penalty=congestion_incumbent_penalty,
            ta_markets={
                ta: TAMarketState(
                    therapeutic_area=ta, exclusivity_duration=exclusivity_period
                )
                for ta in THERAPEUTIC_AREAS
            },
            indication_markets=ind_markets,
            indications_per_ta=num_indications_per_ta,
            indication_name_map=ind_name_map,
        )
        return state

    def add_alert(self, alert: Alert) -> None:
        """Add an alert and prune old ones."""
        self.alerts.append(alert)
        cutoff = self.time - self.alert_history_length
        self.alerts = [a for a in self.alerts if a.step >= cutoff]

    def get_alerts_for_agent(self, agent_id: str) -> list[Alert]:
        """Get alerts visible to a specific agent (all alerts from other agents)."""
        return [a for a in self.alerts if a.agent_id != agent_id]

    def register_drug_release(self, agent_id: str, asset: DrugAsset) -> None:
        """Register a drug reaching the market."""
        ta = asset.therapeutic_area

        # Track in TA market (backward compat)
        ta_market = self.ta_markets.get(ta)
        if ta_market is not None:
            if agent_id not in ta_market.active_drugs:
                ta_market.active_drugs[agent_id] = []
            ta_market.active_drugs[agent_id].append(asset.id)
            ta_market.entry_order.append(asset.id)
            if ta_market.first_mover_agent is None:
                ta_market.first_mover_agent = agent_id
                ta_market.first_mover_drug_id = asset.id
                # Use time + 1 because this is called before advance_time(),
                # but the drug enters the market on the next step
                ta_market.exclusivity_start_time = self.time + 1
            elif ta_market.exclusivity_start_time == self.time + 1:
                # Second drug entered in the same step — void exclusivity for both
                ta_market.first_mover_agent = None
                ta_market.first_mover_drug_id = None
                ta_market.exclusivity_start_time = None

        # Track in indication market (granular competition)
        if self.indications_per_ta > 0:
            key = indication_key(ta, asset.indication)
            ind_market = self.indication_markets.get(key)
            if ind_market is not None:
                if agent_id not in ind_market.active_drugs:
                    ind_market.active_drugs[agent_id] = []
                ind_market.active_drugs[agent_id].append(asset.id)
                ind_market.entry_order.append(asset.id)
                if ind_market.first_mover_agent is None:
                    ind_market.first_mover_agent = agent_id
                    ind_market.first_mover_drug_id = asset.id
                    ind_market.exclusivity_start_time = self.time + 1
                elif ind_market.exclusivity_start_time == self.time + 1:
                    # Second drug entered in the same step — void exclusivity for both
                    ind_market.first_mover_agent = None
                    ind_market.first_mover_drug_id = None
                    ind_market.exclusivity_start_time = None

        # Generate alert
        self.add_alert(
            Alert(
                step=self.time,
                event_type=AlertType.DRUG_RELEASE,
                agent_id=agent_id,
                therapeutic_area=ta,
                indication=asset.indication,
                details={"max_revenue": asset.max_revenue},
            )
        )

    def register_bd_deal(
        self, winner_agent_id: str, asset: DrugAsset, price: float = 0.0
    ) -> None:
        """Register a BD deal completion and generate alert."""
        self.add_alert(
            Alert(
                step=self.time,
                event_type=AlertType.BD_DEAL,
                agent_id=winner_agent_id,
                therapeutic_area=asset.therapeutic_area,
                indication=asset.indication,
                details={"asset_name": asset.name, "price": price},
            )
        )

    def site_auction_available(self) -> bool:
        """
        Whether a clinical site is up for auction at the current step.

        Deterministic cadence: an auction runs on the warmup step and every
        ``site_auction_interval_steps`` thereafter. Evaluated during Phase 0,
        before ``advance_time()``, so it keys off the step being processed.
        """
        if not self.site_auction_enabled:
            return False
        if self.time < self.site_auction_min_step:
            return False
        return (
            self.time - self.site_auction_min_step
        ) % self.site_auction_interval_steps == 0

    def register_site_deal(self, winner_agent_id: str, price: float) -> None:
        """Register a clinical-site auction win and broadcast the price."""
        self.add_alert(
            Alert(
                step=self.time,
                event_type=AlertType.CLINICAL_SITE_DEAL,
                agent_id=winner_agent_id,
                therapeutic_area="",
                details={"price": price},
            )
        )

    def generate_phase_transition_leak(
        self,
        agent_id: str,
        asset: DrugAsset,
        new_phase_index: int,
    ) -> None:
        """
        Generate a pipeline leak alert for a phase transition.

        Called when an asset passes a trial phase. Rolls against
        leak_phase_probabilities to decide whether to leak.

        Parameters
        ----------
        agent_id : str
            The agent whose asset transitioned.
        asset : DrugAsset
            The asset that passed the phase.
        new_phase_index : int
            Index into leak_phase_probabilities (0=Phase 1→2, 1=Phase 2→3,
            2=Phase 3→Approval).

        """
        if new_phase_index >= len(self.leak_phase_probabilities):
            return

        leak_prob = self.leak_phase_probabilities[new_phase_index]
        if get_game_rng().random() < leak_prob:
            phase_names = ["Phase 2", "Phase 3", "Approval"]
            new_phase_name = (
                phase_names[new_phase_index]
                if new_phase_index < len(phase_names)
                else "Unknown"
            )
            self.add_alert(
                Alert(
                    step=self.time,
                    event_type=AlertType.PIPELINE_LEAK,
                    agent_id=agent_id,
                    therapeutic_area=asset.therapeutic_area,
                    indication=asset.indication,
                    details={"new_phase": new_phase_name},
                )
            )
            logger.debug(
                f"Pipeline leak: {agent_id} asset in {asset.therapeutic_area} "
                f"reached {new_phase_name} (prob={leak_prob})"
            )

    def generate_be_spend_leak(
        self,
        agent_id: str,
        therapeutic_area: str,
        indication: int,
        spend_count: int = 1,
    ) -> None:
        """
        Probabilistically leak brand-equity spend in an indication, with a count.

        Rolls ``be_leak_probability`` independently for each of the
        ``spend_count`` brand-equity spends the agent made in this indication
        this step. The alert fires only if at least one roll succeeds and carries
        ``be_count`` = the number of *leaked* spends (an aggregate dominance
        signal, never per-asset attribution). So two spends may surface as a
        count of 1 when one roll fails, or produce no alert at all when both
        fail. The payload still carries no cost or amount — spending BE early
        therefore acts as an early-intel leak. Callers pass one call per
        (agent, indication) per step with the indication's spend count.
        """
        rng = get_game_rng()
        be_count = sum(
            1 for _ in range(spend_count) if rng.random() < self.be_leak_probability
        )
        if be_count > 0:
            self.add_alert(
                Alert(
                    step=self.time,
                    event_type=AlertType.BE_SPEND,
                    agent_id=agent_id,
                    therapeutic_area=therapeutic_area,
                    indication=indication,
                    details={"be_count": be_count},
                )
            )
            logger.debug(
                f"BE-spend leak: {agent_id} spent brand equity in "
                f"{therapeutic_area}:{indication} "
                f"(be_count={be_count}/{spend_count}, prob={self.be_leak_probability})"
            )

    def generate_dc_spend_leak(
        self, agent_id: str, therapeutic_area: str, indication: int
    ) -> None:
        """
        Probabilistically leak that an agent spent demand creation on an indication.

        Rolls against ``dc_leak_probability``. The alert carries only the acting
        agent, the therapeutic area and the indication (no cost/amount).
        """
        if get_game_rng().random() < self.dc_leak_probability:
            self.add_alert(
                Alert(
                    step=self.time,
                    event_type=AlertType.DC_SPEND,
                    agent_id=agent_id,
                    therapeutic_area=therapeutic_area,
                    indication=indication,
                )
            )
            logger.debug(
                f"DC-spend leak: {agent_id} spent demand creation in "
                f"{therapeutic_area}:{indication} (prob={self.dc_leak_probability})"
            )

    def spawn_bd_asset(
        self,
        agent_portfolios: dict[str, dict[uuid.UUID, DrugAsset]],
        max_slots: int = 1,
        won_asset_ids: set | None = None,
    ) -> None:
        """
        Spawn BD assets using Poisson process with leak boost.

        Carries forward unwon assets that have not yet reached bd_persist_steps,
        then fills remaining free slots with newly generated assets.  Assets are
        ordered oldest-first (slot 0 = most urgent) so agents see a stable
        urgency ordering across steps.

        Args:
            agent_portfolios: Per-agent asset dicts for activity weighting.
            max_slots: Hard capacity of the BD market (bd_max_slots).
            won_asset_ids: UUIDs of assets acquired this step; these are removed
                immediately regardless of remaining persistence budget.

        """
        won_ids = won_asset_ids or set()

        # 1. Age surviving (unwon, unexpired) assets from the previous step
        surviving = []
        for asset in self.current_bd_assets:
            if asset.id in won_ids:
                self.bd_asset_ages.pop(str(asset.id), None)
                continue  # acquired — slot freed
            new_age = self.bd_asset_ages.get(str(asset.id), 0) + 1
            if new_age >= self.bd_persist_steps:
                self.bd_asset_ages.pop(str(asset.id), None)
                logger.debug(f"BD asset expired after {new_age} steps: {asset.name}")
                continue  # expired
            self.bd_asset_ages[str(asset.id)] = new_age
            surviving.append(asset)

        # Oldest first → slot 0 is always the most urgent (about to expire)
        surviving.sort(key=lambda a: self.bd_asset_ages.get(str(a.id), 0), reverse=True)
        self.current_bd_assets = surviving

        free_slots = max_slots - len(surviving)
        if free_slots <= 0:
            return  # market is full — no new arrivals this step

        if not self.bd_enabled:
            return
        if self.time < self.bd_min_step:
            return
        if self._bd_asset_generator is None:
            return

        # Count recent leaks per indication
        leak_counts: dict[str, int] = {}
        for alert in self.alerts:
            if alert.event_type == AlertType.PIPELINE_LEAK:
                key = indication_key(alert.therapeutic_area, alert.indication)
                leak_counts[key] = leak_counts.get(key, 0) + 1

        # λ is fixed — leaks only influence indication weighting, not spawn rate
        effective_lambda = self.bd_base_lambda

        # Build indication weights (shared across slots)
        indication_weights: dict[str, float] = {}
        bias = self.bd_indication_activity_bias

        if self.indications_per_ta > 0:
            for ta in THERAPEUTIC_AREAS:
                for idx in range(self.indications_per_ta):
                    key = indication_key(ta, idx)
                    leak_w = leak_counts.get(key, 0)
                    activity_w = 0
                    for portfolio in agent_portfolios.values():
                        for a in portfolio.values():
                            if (
                                a.therapeutic_area == ta
                                and a.indication == idx
                                and a.state
                                in (AssetState.InDevelopment, AssetState.OnMarket)
                            ):
                                activity_w += 1
                    indication_weights[key] = bias * (leak_w + activity_w) + (1 - bias)
        else:
            for ta in THERAPEUTIC_AREAS:
                key = indication_key(ta, 0)
                indication_weights[key] = 1.0

        keys = list(indication_weights.keys())
        weights = [indication_weights[k] for k in keys]
        total_w = sum(weights)

        # Single Poisson draw determines how many new assets spawn
        # (capped by the number of free slots)
        # Manual inverse-CDF sampling: P(X=k) = e^{-λ} λ^k / k!
        u = get_game_rng().random()
        cumulative = 0.0
        num_to_spawn = 0
        p_k = math.exp(-effective_lambda)  # P(X=0)
        for k in range(free_slots + 1):
            cumulative += p_k
            if u < cumulative:
                num_to_spawn = k
                break
            p_k *= effective_lambda / (k + 1)
        else:
            num_to_spawn = free_slots

        for _slot in range(num_to_spawn):
            # Weighted sample for indication
            r = get_game_rng().random() * total_w
            cumulative = 0.0
            selected_key = keys[-1]
            for k, w in zip(keys, weights):
                cumulative += w
                if r <= cumulative:
                    selected_key = k
                    break

            parts = selected_key.split(":")
            selected_ta = parts[0]
            selected_indication = int(parts[1])

            # Sample phase (weighted)
            phase_weights = self.bd_phase_weights
            r = get_game_rng().random() * sum(phase_weights)
            cumulative = 0.0
            target_phase = 0
            for i, w in enumerate(phase_weights):
                cumulative += w
                if r <= cumulative:
                    target_phase = i
                    break

            asset = self._bd_asset_generator.generate_bd_asset(
                therapeutic_area=selected_ta,
                indication=selected_indication,
                target_phase=target_phase,
            )
            self.bd_asset_ages[str(asset.id)] = 0
            self.current_bd_assets.append(asset)

            logger.debug(
                f"BD asset spawned (slot {len(surviving) + _slot}): "
                f"{selected_ta}:{selected_indication} "
                f"Phase {target_phase + 1}, λ_eff={effective_lambda:.2f}"
            )

    def clear_bd_asset(self) -> None:
        """Clear all current BD assets and age tracking (used on reset)."""
        self.current_bd_assets = []
        self.bd_asset_ages = {}

    def remove_expired_drug(self, agent_id: str, asset_id: uuid.UUID) -> None:
        """Remove an expired drug from market tracking."""
        for ta_market in self.ta_markets.values():
            if agent_id in ta_market.active_drugs:
                if asset_id in ta_market.active_drugs[agent_id]:
                    ta_market.active_drugs[agent_id].remove(asset_id)
                    if asset_id in ta_market.entry_order:
                        ta_market.entry_order.remove(asset_id)
                    if ta_market.first_mover_drug_id == asset_id:
                        ta_market.first_mover_drug_id = None

        for ind_market in self.indication_markets.values():
            if agent_id in ind_market.active_drugs:
                if asset_id in ind_market.active_drugs[agent_id]:
                    ind_market.active_drugs[agent_id].remove(asset_id)
                    if asset_id in ind_market.entry_order:
                        ind_market.entry_order.remove(asset_id)
                    if ind_market.first_mover_drug_id == asset_id:
                        ind_market.first_mover_drug_id = None

    def apply_demand_creation(
        self,
        indication_key_str: str,
        boost_from_pre_step: float,
        marketing_config: "MarketingConfig",
    ) -> None:
        """
        Apply a pre-computed demand creation boost to an indication's pool multiplier.

        The boost must be computed from the pre-step snapshot of demand_multiplier
        (before any agent's boost is applied this step) to avoid ordering bias when
        multiple agents spend simultaneously.
        """
        ind_market = self.indication_markets.get(indication_key_str)
        if ind_market is None:
            return
        ind_market.demand_multiplier += boost_from_pre_step

    def advance_time(self, marketing_config: "MarketingConfig | None" = None) -> None:
        """Advance the shared market time by one step."""
        self.time += 1
        if marketing_config is not None and marketing_config.enabled:
            for ind_market in self.indication_markets.values():
                headroom = ind_market.demand_multiplier - 1.0
                ind_market.demand_multiplier = 1.0 + headroom * (
                    1.0 - marketing_config.dc_decay_rate
                )

    def rebase_time_to_zero(self) -> None:
        """
        Shift the market clock so the current step becomes time 0.

        Used at the warmup boundary: warmup runs as a pre-roll, then the
        agent plays a full horizon starting at time 0. Every piece of
        absolute-time-stamped state is shifted back by the elapsed offset
        (``self.time``) so downstream cadence/exclusivity/alert-age checks
        keep working unchanged. Relative state (BD asset ages, current BD
        assets) is intentionally left intact — it represents the warmed
        market the agent inherits.
        """
        offset = self.time
        if offset == 0:
            return

        self.time = 0

        # Alert is frozen — rebuild each with a shifted step index.
        self.alerts = [
            alert.model_copy(update={"step": alert.step - offset})
            for alert in self.alerts
        ]

        # Rebase exclusivity windows (absolute start times) across markets.
        for market in (
            list(self.ta_markets.values()) + list(self.indication_markets.values())
        ):
            if market.exclusivity_start_time is not None:
                market.exclusivity_start_time -= offset

    def set_bd_asset_generator(self, generator: AssetGeneratorBase) -> None:
        """Set the asset generator for BD asset spawning."""
        self._bd_asset_generator = generator

    def get_bd_observations(self) -> list[dict]:
        """Get observation data for all current BD assets."""
        result = []
        for asset in self.current_bd_assets:
            age = self.bd_asset_ages.get(str(asset.id), 0)
            # Steps remaining, including the current one
            steps_left = self.bd_persist_steps - age
            steps_remaining_norm = steps_left / max(1, self.bd_persist_steps)
            result.append({
                "max_revenue": asset.max_revenue,
                "time_until_max_revenue": asset.time_until_max_revenue,
                "time_until_patent_expiry": asset.time_until_patent_expiry,
                "therapeutic_area": asset.therapeutic_area,
                "indication": asset.indication,
                "enpv": asset.enpv,
                "trial_phase": (asset.trial.phase.integer if asset.trial else 3),
                "ptrs": asset.trial.ptrs if asset.trial else 0.0,
                "steps_remaining": steps_remaining_norm,
            })
        return result
