import logging
import uuid
from typing import Literal, Optional

from pydantic import BaseModel

from pyxis_portfolio_challenge.game.asset import AssetState, DrugAsset
from pyxis_portfolio_challenge.game.constants import InvestmentLevel
from pyxis_portfolio_challenge.game.game_state import GameState
from pyxis_portfolio_challenge.game.multi_agent_game import MultiAgentGame
from pyxis_portfolio_challenge.game.shared_market_state import (
    Alert,
    AlertType,
    IndicationMarketState,
    indication_key,
)
from pyxis_portfolio_challenge.game.trial import Trial, TrialPhase, TrialState

logger = logging.getLogger(__name__)

# Action types that can be sent from frontend. "drop" is the voluntary-
# abandonment action available when the drop_action feature is enabled (mutually
# exclusive with investment levels).
ActionType = Literal[
    "invest", "stop", "none", "minimal", "standard", "accelerated", "drop"
]


class StartGameRequest(BaseModel):
    """Request model for starting a new game."""

    num_assets: int
    max_num_assets: int
    horizon: int
    starting_cash: float
    global_seed: int = 42
    level_idx: int = -1  # Use -1 to indicate non-level plays


class TrialResponse(BaseModel):
    """Response model for a trial."""

    cost_remaining: float
    time_remaining: int
    ptrs: float
    # Interim observation data
    interim_result: Optional[Literal["positive", "negative"]]
    has_interim_observation: bool
    # Distributional PTRS fields
    ptrs_expected: Optional[float]
    ptrs_confidence: Optional[float]
    ptrs_range_low: Optional[float]
    ptrs_range_high: Optional[float]
    # PTRS readings (ptrs_readings feature): the number of paid readings
    # commissioned on this trial so far, and the precision-weighted equivalent
    # count of base-σ readings — the exact "effective readings" quantity the
    # observation exposes. Both 0 when the feature is off or the phase is a
    # success/failure placeholder. Always populated so the panel can render.
    ptrs_sample_count: int
    ptrs_effective_readings: float


class DrugAssetResponse(BaseModel):
    """Response model for a drug asset."""

    id: uuid.UUID
    name: str
    therapeutic_area: Literal[
        "oncology", "respiratory and immunology", "vaccines and infectious disease"
    ]
    type: Literal["internal", "BD"]
    indication: int
    indication_name: str
    description: str
    max_revenue: float  # M
    time_until_max_revenue: int  # H
    time_until_patent_expiry: int  # T
    trials: dict[TrialPhase, TrialResponse]
    state: AssetState
    pending_trial_phase: str | None
    time_on_market: int
    cost_this_step: float
    cost_to_invest_this_step: float
    revenue_this_step: float
    enpv: float  # Full expected NPV (business value); NOT the competition score
    cash_enpv: float  # Cash-adjusted eNPV: value that flows to cash and scores
    expected_costs: list[float]
    expected_revenues: list[float]
    eroi: float
    current_investment_level: Literal["none", "minimal", "standard", "accelerated"]
    available_actions: list[ActionType]
    # Marketing (brand equity): the drug's current brand score, its slowly-rising
    # floor, and the cash cost of a brand-equity push on it this step (scales with
    # drug size). All 0.0 when marketing is disabled or the drug is dead. Always
    # populated by the response builder so the frontend can render the panel.
    brand_score: float
    brand_score_floor: float
    be_cost: float
    # Forward-looking projection of the drug's brand score after the next step,
    # for each spend decision: *_if_spend pays the cost this step, *_if_hold skips
    # it (score decays toward its floor). Computed via the shared MarketingConfig
    # helper so the preview matches what the engine will produce. Lets the panel
    # show the impact of committing a push *before* the user commits, rather than
    # the retrospective change. Equal to brand_score for dead drugs (can't push).
    brand_score_if_spend: float
    brand_score_if_hold: float
    # PTRS readings (ptrs_readings feature): cumulative cash cost of commissioning
    # 1..N readings on this drug's current trial this step (escalating Fibonacci
    # curve), so the panel can price each stepper notch. Index i = cost of (i+1)
    # readings. Empty when the feature is off or the drug has no pending trial
    # (dead / on-market), which can no longer be read.
    ptrs_reading_costs: list[float]


class InvestmentLevelConfigResponse(BaseModel):
    """Response model for a single investment level configuration."""

    cost_modifier: float
    speed_modifier: float
    success_modifier: float
    capacity_cost: int
    experience_modifier: float


class InvestmentLevelsConfigResponse(BaseModel):
    """Response model for all investment levels configuration."""

    levels: dict[str, InvestmentLevelConfigResponse]
    base_capacity: float
    overage_max_penalty: float
    overage_cost_max_penalty: float


class GameStateResponse(BaseModel):
    """Response model for the game state."""

    id: uuid.UUID
    cash: float
    time: int
    horizon: int
    assets: dict[uuid.UUID, DrugAssetResponse]
    expired_assets: dict[uuid.UUID, DrugAssetResponse]
    # Voluntarily abandoned assets (drop_action feature); distinct from
    # expired/failed. Serialized in the same shape as expired_assets.
    dropped_assets: dict[uuid.UUID, DrugAssetResponse]
    reinvestment_percentage: float
    realised_costs: list[float]
    realised_revenues: list[float]
    game_ended: bool
    ended_reason: str | None
    capital_over_time: list[float]
    enpv_over_time: list[float]
    eroi_over_time: list[float]
    # TA experience
    ta_experience: dict[str, float]
    experience_to_full_knowledge: float
    max_total_experience: float | None
    # R&D Capacity
    capacity_used: float
    capacity_base: float
    success_modifier: float
    cost_modifier: float
    # Clinical sites (clinical_sites feature). operational_sites host trials;
    # sites_in_development are build-delay timers; free/occupied are derived.
    clinical_sites_enabled: bool
    operational_sites: int
    sites_in_development: list[int]
    free_sites: int
    sites_occupied: int
    # Cash cost of the next site (Fibonacci purchase curve); 0 when the feature
    # is off. Drives the "Buy site" button label/affordability in the UI.
    next_site_purchase_cost: float
    # Feature flags
    ta_experience_enabled: bool
    investment_levels_enabled: bool
    interim_observations_enabled: bool
    distributional_ptrs_enabled: bool
    # Marketing feature: gates the demand-creation and brand-equity panels. Per-
    # asset brand-equity costs are attached to each asset (be_cost); the flat
    # demand-creation cost is on the game response (dc_cost).
    marketing_enabled: bool
    # PTRS readings feature: gates the per-asset readings panel (portfolio) and the
    # BD diligence stepper. Per-asset cost curves are attached to each asset
    # (ptrs_reading_costs); the effective-readings signal is on each trial.
    ptrs_readings_enabled: bool
    # TA quality estimates (distributional PTRS feature)
    ta_quality: dict[str, dict[str, float]]
    # Investment levels configuration (for info popup)
    investment_levels_config: InvestmentLevelsConfigResponse | None


class LevelResponse(BaseModel):
    """Response model for level information."""

    level_idx: int
    user_has_completed: bool
    num_assets: int
    max_num_assets: int
    horizon: int
    starting_cash: float
    global_seed: int


class AgentResponse(BaseModel):
    """Response model for agent information."""

    name: str
    cost: float


class ComparisonDashboardResponse(BaseModel):
    """Response model for comparison dashboard data."""

    game_id: uuid.UUID
    av_enpv: dict[str, float]
    final_enpv: dict[str, float]
    final_eroi: dict[str, float]
    final_capital: dict[str, float]
    realised_eroi: dict[str, float]
    enpv_over_time: dict[str, list[float]]
    eroi_over_time: dict[str, list[float]]


# --- Multi-Agent Request/Response Models ---


class StartMultiAgentGameRequest(BaseModel):
    """Request model for starting a multi-agent game."""

    num_assets: int
    max_num_assets: int
    horizon: int
    starting_cash: float
    global_seed: int = 42
    num_opponents: int  # 1-3
    opponent_agents: list[str]  # e.g. ["knapsack_agent", "random"]


class MultiAgentStepRequest(BaseModel):
    """Request model for stepping a multi-agent game."""

    investment_actions: dict[uuid.UUID, Optional[ActionType]]
    bd_bids: list[float] = []
    # Per-BD-asset cash bids in GBP (0 = pass); length = num BD assets. The
    # highest bid wins and pays its own bid; there is no affordability check,
    # so an overbid can bankrupt the winner.
    # Clinical sites (clinical_sites feature): buy one new site this step
    # (upgrade) and/or bid cash in the PvP site auction (site_bid GBP, 0 = pass).
    # Required, no defaults: the client must send every enabled head each step.
    upgrade: bool
    site_bid: float
    # Marketing (marketing feature): binary spend decisions this step (1 = pay the
    # fixed cost, 0/absent = skip). demand_creation is keyed by indication
    # ("{therapeutic_area}:{indication}") and sizes the shared demand pool;
    # brand_equity is keyed by asset UUID and boosts that drug's brand score.
    # Required, no defaults: the client sends every enabled head each step.
    demand_creation: dict[str, int]
    brand_equity: dict[uuid.UUID, int]
    # PTRS readings (ptrs_readings feature): per-asset count of paid diligence
    # readings to commission this step, keyed by asset UUID. One dict covers both
    # portfolio assets and BD candidates — the engine routes by id (portfolio
    # readings hit the drug directly, BD readings hit a private per-agent clone).
    # 0/absent = no readings. Required, no defaults: the client sends every
    # enabled head each step.
    ptrs_research: dict[uuid.UUID, int]


class BDAssetResponse(BaseModel):
    """Response model for a BD asset available for bidding."""

    asset_id: uuid.UUID
    name: str
    therapeutic_area: str
    indication: int
    indication_name: str
    max_revenue: float
    time_until_max_revenue: int
    time_until_patent_expiry: int
    trial_phase: str
    ptrs: float
    enpv: float
    cash_enpv: float  # cash-adjusted eNPV; a fair-value anchor for a cash bid
    # PTRS readings (ptrs_readings feature): diligence on a BD candidate is private
    # to each bidder, held on a per-agent clone. These reflect *this* player's
    # clone when they have commissioned readings, else the untouched shared asset:
    # readings so far, the effective-readings signal, and the cost curve for buying
    # 1..N more this step. sample_count 0 / effective 0.0 / costs [] when the
    # feature is off or the candidate has no pending trial.
    ptrs_sample_count: int
    ptrs_effective_readings: float
    ptrs_reading_costs: list[float]


class AlertResponse(BaseModel):
    """Response model for a competitive intelligence alert."""

    step: int
    event_type: str
    agent_id: str
    therapeutic_area: str
    indication: int
    indication_name: str
    details: dict


class IndicationMarketResponse(BaseModel):
    """Response model for an indication market."""

    therapeutic_area: str
    indication: int
    indication_name: str
    first_mover_agent: str | None
    incumbent_agent: str | None
    exclusivity_remaining: int
    active_drugs: dict[str, int]  # agent_id -> count
    player_market_share: float
    # Shared demand-creation multiplier for the indication (1.0 = no boost).
    demand_multiplier: float
    # Forward-looking projection of the demand multiplier after the next step,
    # for each spend decision: *_if_spend pays the demand-creation cost this step,
    # *_if_hold skips it (the multiplier decays toward the 1.0 base). Computed via
    # the shared MarketingConfig helper so the preview matches what the engine
    # will produce, letting the panel show the impact of a spend before committing.
    demand_multiplier_if_spend: float
    demand_multiplier_if_hold: float


class OpponentSummaryResponse(BaseModel):
    """Response model for an opponent agent summary."""

    agent_name: str
    display_name: str
    agent_type: str
    cash: float
    num_assets: int
    num_on_market: int
    num_in_development: int
    enpv: float
    cumulative_reward: float
    game_ended: bool
    ended_reason: str | None


class MultiAgentGameStateResponse(BaseModel):
    """Response model for multi-agent game state."""

    game_id: uuid.UUID
    player_agent_name: str
    player_state: GameStateResponse
    bd_assets: list[BDAssetResponse]
    bd_enabled: bool
    # Clinical-site PvP auction (clinical_sites feature): whether a site is up
    # for auction this step. False when the feature or auction is off. The bid
    # is capped by the player's cash in the UI (mirrors the BD bid input).
    site_auction_active: bool
    alerts: list[AlertResponse]
    indication_markets: list[IndicationMarketResponse]
    opponents: list[OpponentSummaryResponse]
    time: int
    horizon: int
    player_cumulative_reward: float
    player_bankrupt: bool
    game_ended: bool  # True only when ALL agents finished or horizon reached
    ended_reason: str | None
    last_bd_acquisitions: dict[str, list[str]]
    # Flat cash cost of sizing one indication via demand creation this step
    # (marketing feature). Same for every indication (anchored to the pool-wide
    # peak revenue); 0 when marketing is off. Per-asset brand-equity costs live on
    # each asset (be_cost). Drives the DC panel's cost label and cash projection.
    dc_cost: float


class OpponentAgentInfo(BaseModel):
    """Response model for available opponent agent types."""

    id: str
    name: str
    description: str


# --- Multi-Agent Converter Functions ---


def bd_asset_to_response(
    asset: DrugAsset,
    indication_name_map: dict[str, str] | None = None,
    reinvestment_percentage: float = 0.10,
    *,
    ptrs_cfg,
    clone,
) -> BDAssetResponse:
    """
    Convert a DrugAsset (BD candidate) to its response format.

    ptrs_cfg is the PtrsReadingsConfig (or None when the feature is off).
    clone is this player's private deep clone of the shared BD asset when they have
    commissioned diligence on it (None otherwise). A bidder's readings are private,
    so ptrs / sample_count / effective_readings come from the clone when present and
    the untouched shared asset otherwise; the cost curve is anchored to the shared
    asset's cost_remaining (what the engine charges), which readings never alter.
    """
    ind_key = indication_key(asset.therapeutic_area, asset.indication)
    ind_name = indication_name_map.get(ind_key, "") if indication_name_map else ""

    diligence = clone if clone is not None else asset
    diligence_trial = diligence.trial
    readings_on = ptrs_cfg is not None and ptrs_cfg.enabled
    ptrs_effective_readings = (
        ptrs_cfg.effective_readings(diligence_trial.ptrs_total_precision)
        if readings_on and diligence_trial is not None
        else 0.0
    )
    if readings_on and asset.pending_trial_chain and asset.trial is not None:
        ptrs_reading_costs = ptrs_cfg.reading_cost_curve(asset.trial.cost_remaining)
    else:
        ptrs_reading_costs = []

    return BDAssetResponse(
        asset_id=asset.id,
        name=asset.name,
        therapeutic_area=asset.therapeutic_area,
        indication=asset.indication,
        indication_name=ind_name,
        max_revenue=asset.max_revenue,
        time_until_max_revenue=asset.time_until_max_revenue,
        time_until_patent_expiry=asset.time_until_patent_expiry,
        trial_phase=asset.trial.phase.value if asset.trial else "unknown",
        ptrs=diligence_trial.ptrs if diligence_trial else 0.0,
        enpv=asset.enpv,
        cash_enpv=asset.cash_enpv(reinvestment_percentage),
        ptrs_sample_count=(
            diligence_trial.ptrs_sample_count if diligence_trial else 0
        ),
        ptrs_effective_readings=ptrs_effective_readings,
        ptrs_reading_costs=ptrs_reading_costs,
    )


def alert_to_response(
    alert: Alert,
    indication_name_map: dict[str, str],
    name_map: dict[str, str] | None = None,
) -> AlertResponse:
    """Convert an Alert to its response format."""
    ind_key = indication_key(alert.therapeutic_area, alert.indication)
    display_agent_id = (
        name_map.get(alert.agent_id, alert.agent_id) if name_map else alert.agent_id
    )
    return AlertResponse(
        step=alert.step,
        event_type=alert.event_type.value,
        agent_id=display_agent_id,
        therapeutic_area=alert.therapeutic_area,
        indication=alert.indication,
        indication_name=indication_name_map.get(ind_key, ""),
        details=alert.details,
    )


def indication_market_to_response(
    market: IndicationMarketState,
    current_time: int,
    player_agent: str,
    name_map: dict[str, str] | None,
    marketing_cfg,
) -> IndicationMarketResponse:
    """
    Convert an IndicationMarketState to its response format.

    marketing_cfg is the MarketingConfig (or None when marketing is off) used to
    project the demand multiplier one step ahead for the spend/hold preview.
    """
    active_drugs_count = {
        (name_map.get(agent_id, agent_id) if name_map else agent_id): len(drug_ids)
        for agent_id, drug_ids in market.active_drugs.items()
    }

    # Calculate player market share (simplified: proportion of drugs)
    total_drugs = sum(active_drugs_count.values())
    player_display = (
        name_map.get(player_agent, player_agent) if name_map else player_agent
    )
    player_drugs = active_drugs_count.get(player_display, 0)
    if total_drugs > 0:
        player_share = player_drugs / total_drugs
    else:
        player_share = 0.0

    # If player has first-mover exclusivity, share is 1.0
    if market.first_mover_agent == player_agent and market.is_in_exclusivity(
        current_time
    ):
        player_share = 1.0
    elif (
        market.first_mover_agent is not None
        and market.first_mover_agent != player_agent
        and market.is_in_exclusivity(current_time)
    ):
        player_share = 0.0

    first_mover_display = (
        name_map.get(market.first_mover_agent, market.first_mover_agent)
        if name_map and market.first_mover_agent
        else market.first_mover_agent
    )

    # Incumbent = agent owning the drug at entry_order[0] (first on market)
    incumbent_agent_id = None
    if market.entry_order:
        incumbent_drug_id = market.entry_order[0]
        for agent_id, drug_ids in market.active_drugs.items():
            if incumbent_drug_id in drug_ids:
                incumbent_agent_id = agent_id
                break
    incumbent_display = (
        name_map.get(incumbent_agent_id, incumbent_agent_id)
        if name_map and incumbent_agent_id
        else incumbent_agent_id
    )

    # Forward-looking demand-multiplier preview. When marketing is on, project
    # one step ahead for both spend decisions via the shared helper (so the panel
    # can show the impact of a demand-creation spend before committing); when off,
    # there is no projection to make, so both mirror the current value.
    if marketing_cfg is not None and marketing_cfg.enabled:
        demand_if_spend = marketing_cfg.next_demand_multiplier(
            market.demand_multiplier, spend=True
        )
        demand_if_hold = marketing_cfg.next_demand_multiplier(
            market.demand_multiplier, spend=False
        )
    else:
        demand_if_spend = market.demand_multiplier
        demand_if_hold = market.demand_multiplier

    return IndicationMarketResponse(
        therapeutic_area=market.therapeutic_area,
        indication=market.indication,
        indication_name=market.indication_name,
        first_mover_agent=first_mover_display,
        incumbent_agent=incumbent_display,
        exclusivity_remaining=market.exclusivity_remaining(current_time),
        active_drugs=active_drugs_count,
        player_market_share=player_share,
        demand_multiplier=market.demand_multiplier,
        demand_multiplier_if_spend=demand_if_spend,
        demand_multiplier_if_hold=demand_if_hold,
    )


def multi_agent_game_to_response(
    game: MultiAgentGame,
    player_agent: str,
    opponent_types: list[str],
    opponent_display_names: list[str] | None = None,
    cumulative_rewards: dict[str, float] | None = None,
) -> MultiAgentGameStateResponse:
    """Convert a MultiAgentGame to its response format."""
    player_state = game.agent_states[player_agent]
    ind_name_map = game.shared_market.indication_name_map
    player_state_response = game_state_to_response(player_state, ind_name_map)

    # Flat demand-creation cost (marketing feature): anchored to the pool-wide
    # peak revenue held on the game, so it lives here rather than per-agent.
    marketing_cfg = player_state._marketing_config
    dc_cost = (
        marketing_cfg.dc_cost(game._be_static_peak_revenue)
        if marketing_cfg is not None and marketing_cfg.enabled
        else 0.0
    )

    # Build name mapping: pharma_X -> display name
    name_map: dict[str, str] = {player_agent: "You"}
    if opponent_display_names:
        for i, display_name in enumerate(opponent_display_names):
            name_map[f"pharma_{i + 1}"] = display_name
    else:
        for i in range(len(opponent_types)):
            name_map[f"pharma_{i + 1}"] = f"pharma_{i + 1}"

    ptrs_cfg = player_state._ptrs_readings_config
    bd_assets_response = [
        bd_asset_to_response(
            asset,
            ind_name_map,
            player_state.reinvestment_percentage,
            ptrs_cfg=ptrs_cfg,
            clone=player_state._bd_asset_clones.get(str(asset.id)),
        )
        for asset in game.shared_market.current_bd_assets
    ]

    alert_responses = [
        alert_to_response(alert, ind_name_map, name_map)
        for alert in game.shared_market.alerts
    ]

    indication_market_responses = [
        indication_market_to_response(
            market, game.time, player_agent, name_map, marketing_cfg
        )
        for market in game.shared_market.indication_markets.values()
    ]

    # Build opponent summaries
    opponent_responses = []
    for i, agent_type in enumerate(opponent_types):
        agent_name = f"pharma_{i + 1}"
        if agent_name not in game.agent_states:
            continue
        state = game.agent_states[agent_name]
        num_on_market = sum(
            1 for a in state.assets.values() if a.state == AssetState.OnMarket
        )
        num_in_dev = sum(
            1 for a in state.assets.values() if a.state == AssetState.InDevelopment
        )
        opponent_responses.append(
            OpponentSummaryResponse(
                agent_name=name_map.get(agent_name, agent_name),
                display_name=name_map.get(agent_name, agent_name),
                agent_type=agent_type,
                cash=state.cash,
                num_assets=len(state.assets),
                num_on_market=num_on_market,
                num_in_development=num_in_dev,
                enpv=sum(a.enpv for a in state.assets.values()),
                cumulative_reward=(
                    cumulative_rewards.get(agent_name, 0.0)
                    if cumulative_rewards
                    else 0.0
                ),
                game_ended=state.game_ended,
                ended_reason=state.ended_reason,
            )
        )

    # Derive last BD acquisitions from BD_DEAL alerts at current step
    last_bd_names: dict[str, list[str]] = {}
    for alert in game.shared_market.alerts:
        if alert.event_type == AlertType.BD_DEAL and alert.step == game.time - 1:
            display_id = name_map.get(alert.agent_id, alert.agent_id)
            asset_name = alert.details.get(
                "asset_name", str(alert.details.get("asset_id", ""))[:8]
            )
            last_bd_names.setdefault(display_id, []).append(asset_name)

    # Player bankrupt = player's game_ended flag (bankruptcy or horizon)
    player_bankrupt = player_state.game_ended and (
        player_state.ended_reason is None
        or "horizon" not in player_state.ended_reason.lower()
    )

    # Game is over when ALL agents have ended or horizon is reached
    all_ended = all(s.game_ended for s in game.agent_states.values())
    horizon_reached = game.time >= game.horizon
    game_ended = all_ended or horizon_reached

    # Use player's ended_reason if they're bankrupt, otherwise check horizon
    if player_bankrupt:
        ended_reason = player_state.ended_reason
    elif horizon_reached:
        ended_reason = "Game horizon reached"
    elif all_ended:
        ended_reason = "All agents eliminated"
    else:
        ended_reason = None

    return MultiAgentGameStateResponse(
        game_id=player_state.id,
        player_agent_name=name_map.get(player_agent, player_agent),
        player_state=player_state_response,
        bd_assets=bd_assets_response,
        bd_enabled=game.shared_market.bd_enabled,
        site_auction_active=game.shared_market.site_auction_available(),
        alerts=alert_responses,
        indication_markets=indication_market_responses,
        opponents=opponent_responses,
        time=game.time,
        horizon=game.horizon,
        player_cumulative_reward=(
            cumulative_rewards.get(player_agent, 0.0) if cumulative_rewards else 0.0
        ),
        player_bankrupt=player_bankrupt,
        game_ended=game_ended,
        ended_reason=ended_reason,
        last_bd_acquisitions=last_bd_names,
        dc_cost=dc_cost,
    )


def trial_to_response(trial: Trial, ptrs_cfg) -> dict[TrialPhase, TrialResponse]:
    """
    Convert the Trial and prev/subsequent trials to response.

    ptrs_cfg is the PtrsReadingsConfig (or None when the feature is off), used to
    convert a trial's accumulated precision into the effective-readings signal the
    observation exposes.
    """
    _trial = trial
    readings_on = ptrs_cfg is not None and ptrs_cfg.enabled
    response_dict = {}
    failure_detected = False
    for phase in TrialPhase:
        if failure_detected:
            # A failure was detected in a previous phase,
            # so all subsequent ptrs are 0.
            response_dict[phase.value] = TrialResponse(
                cost_remaining=0.0,
                time_remaining=0,
                ptrs=0.0,
                interim_result=None,
                has_interim_observation=False,
                ptrs_expected=0.0,
                ptrs_confidence=1.0,
                ptrs_range_low=0.0,
                ptrs_range_high=0.0,
                ptrs_sample_count=0,
                ptrs_effective_readings=0.0,
            )
            continue

        if _trial and _trial.phase == phase:
            # Check if the current trial has failed
            if _trial.state == TrialState.PHASE_FAILED:
                failure_detected = True

            # Check for interim observation
            interim_result = None
            has_interim = False
            if hasattr(_trial, "_interim_observation_result"):
                interim_obs = _trial._interim_observation_result
                if interim_obs is not None:
                    has_interim = True
                    interim_result = "positive" if interim_obs else "negative"

            response_dict[phase.value] = TrialResponse(
                cost_remaining=_trial.cost_remaining,
                time_remaining=_trial.time_remaining,
                ptrs=_trial.ptrs,
                interim_result=interim_result,
                has_interim_observation=has_interim,
                ptrs_expected=_trial.ptrs_expected,
                ptrs_confidence=_trial.ptrs_confidence,
                ptrs_range_low=_trial.ptrs_range_low,
                ptrs_range_high=_trial.ptrs_range_high,
                ptrs_sample_count=_trial.ptrs_sample_count,
                ptrs_effective_readings=(
                    ptrs_cfg.effective_readings(_trial.ptrs_total_precision)
                    if readings_on
                    else 0.0
                ),
            )
            _trial = _trial.next_trial_on_success
        else:
            # previous trial must have succeeded, so use success values
            response_dict[phase.value] = TrialResponse(
                cost_remaining=0.0,
                time_remaining=0,
                ptrs=1.0,
                interim_result=None,
                has_interim_observation=False,
                ptrs_expected=1.0,
                ptrs_confidence=1.0,
                ptrs_range_low=1.0,
                ptrs_range_high=1.0,
                ptrs_sample_count=0,
                ptrs_effective_readings=0.0,
            )

    return response_dict


def asset_to_response(
    drug_asset: DrugAsset,
    investment_levels_enabled: bool = False,
    indication_name_map: dict[str, str] | None = None,
    *,
    reinvestment_percentage: float,
    drop_action_enabled: bool,
    ptrs_cfg,
) -> DrugAssetResponse:
    """
    Convert the asset to a response format for the frontend.

    ptrs_cfg is the PtrsReadingsConfig (or None when the feature is off), used to
    price the per-drug readings cost curve and the per-trial effective-readings
    signal.
    """
    base = drug_asset.model_dump()
    base["id"] = drug_asset.id

    # Add indication name from map
    ind_key = indication_key(drug_asset.therapeutic_area, drug_asset.indication)
    base["indication_name"] = (
        indication_name_map.get(ind_key, "") if indication_name_map else ""
    )

    # Convert trials to response format
    del base["trial"]
    base["trials"] = trial_to_response(drug_asset.trial, ptrs_cfg)
    # handle pending trial phase logic
    if drug_asset.state == AssetState.OnMarket:
        pending_trial_phase = None
    elif drug_asset.trial.state == TrialState.PHASE_FAILED:
        pending_trial_phase = None
    else:
        pending_trial_phase = drug_asset.trial.phase.value

    # Add properties to response
    base["cost_this_step"] = drug_asset.cost_this_step
    # cost_to_invest_this_step is the cost if you invest this step
    # (only valid for Idle assets)
    if drug_asset.state == AssetState.Idle:
        base["cost_to_invest_this_step"] = drug_asset.cost_to_invest_this_step
    else:
        base["cost_to_invest_this_step"] = 0.0
    base["revenue_this_step"] = drug_asset.revenue_this_step
    # Full eNPV is the real-world business value; cash_enpv is the score-aligned
    # value (revenues scaled by reinvestment_percentage, matching the NCF reward).
    base["enpv"] = drug_asset.enpv
    base["cash_enpv"] = drug_asset.cash_enpv(reinvestment_percentage)
    (
        base["expected_costs"],
        base["expected_revenues"],
    ) = drug_asset.expected_costs_and_revenues
    base["eroi"] = drug_asset.eroi
    base["pending_trial_phase"] = pending_trial_phase

    # PTRS readings cost curve for this drug's current trial (see field docs). A
    # reading only applies to a pending trial, so dead / on-market drugs get [].
    # Computed via the shared config helper so the priced stepper can never drift
    # from what the engine charges.
    if (
        ptrs_cfg is not None
        and ptrs_cfg.enabled
        and drug_asset.pending_trial_chain
        and drug_asset.trial is not None
    ):
        base["ptrs_reading_costs"] = ptrs_cfg.reading_cost_curve(
            drug_asset.trial.cost_remaining
        )
    else:
        base["ptrs_reading_costs"] = []

    # Add investment level info
    level_map = {
        InvestmentLevel.NONE: "none",
        InvestmentLevel.MINIMAL: "minimal",
        InvestmentLevel.STANDARD: "standard",
        InvestmentLevel.ACCELERATED: "accelerated",
    }
    current_level = drug_asset.current_investment_level
    base["current_investment_level"] = level_map[current_level]

    # Determine available actions based on asset state
    available_actions: list[ActionType] = []
    if drug_asset.state == AssetState.Idle:
        if investment_levels_enabled:
            available_actions = ["none", "minimal", "standard", "accelerated"]
        else:
            available_actions = ["none", "invest"]
    elif drug_asset.state == AssetState.InDevelopment:
        if investment_levels_enabled:
            available_actions = ["minimal", "standard", "accelerated", "stop"]
        else:
            available_actions = ["invest", "stop"]
    # On Market, Failed, Expired, Dropped have no actions
    # Voluntary drop is available on any live (Idle / In Development) asset when
    # the feature is enabled (mutually exclusive with investment levels).
    if drop_action_enabled and drug_asset.state in (
        AssetState.Idle,
        AssetState.InDevelopment,
    ):
        available_actions.append("drop")
    base["available_actions"] = available_actions

    return base


def game_state_to_response(
    game_state: GameState,
    indication_name_map: dict[str, str] | None = None,
) -> GameStateResponse:
    """Convert the game state to a response format for the frontend."""
    logger.debug("Dumping game state model for response.")
    base = game_state.model_dump()
    base["id"] = game_state.id
    if "running_enpv" in base:
        del base["running_enpv"]
    if "running_eroi" in base:
        del base["running_eroi"]

    # Check if features are enabled
    investment_levels_enabled = (
        hasattr(game_state, "_investment_levels_config")
        and game_state._investment_levels_config is not None
        and game_state._investment_levels_config.enabled
    )
    interim_observations_enabled = (
        hasattr(game_state, "_interim_trial_observations_config")
        and game_state._interim_trial_observations_config is not None
        and game_state._interim_trial_observations_config.enabled
    )
    distributional_ptrs_enabled = (
        hasattr(game_state, "_distributional_ptrs_config")
        and game_state._distributional_ptrs_config is not None
        and game_state._distributional_ptrs_config.enabled
    )
    ta_experience_enabled = (
        hasattr(game_state, "_ta_experience_config")
        and game_state._ta_experience_config is not None
        and game_state._ta_experience_config.enabled
    )
    drop_action_enabled = game_state.drop_action_enabled
    marketing_cfg = game_state._marketing_config
    marketing_enabled = marketing_cfg is not None and marketing_cfg.enabled
    ptrs_cfg = game_state._ptrs_readings_config
    ptrs_readings_enabled = ptrs_cfg is not None and ptrs_cfg.enabled

    logger.debug("Converting assets to response.")
    # Convert assets to response format
    base["assets"] = {
        asset_id: asset_to_response(
            asset,
            investment_levels_enabled,
            indication_name_map,
            reinvestment_percentage=game_state.reinvestment_percentage,
            drop_action_enabled=drop_action_enabled,
            ptrs_cfg=ptrs_cfg,
        )
        for asset_id, asset in game_state.assets.items()
    }
    base["expired_assets"] = {
        asset_id: asset_to_response(
            asset,
            investment_levels_enabled,
            indication_name_map,
            reinvestment_percentage=game_state.reinvestment_percentage,
            drop_action_enabled=drop_action_enabled,
            ptrs_cfg=ptrs_cfg,
        )
        for asset_id, asset in {
            **game_state.expired_assets,
            **game_state.failed_assets,
        }.items()
    }
    # Dropped assets (distinct from Failed/Expired): serialize in the same shape
    # as live/expired assets so the frontend can render a Dropped bucket.
    base["dropped_assets"] = {
        asset_id: asset_to_response(
            asset,
            investment_levels_enabled,
            indication_name_map,
            reinvestment_percentage=game_state.reinvestment_percentage,
            drop_action_enabled=drop_action_enabled,
            ptrs_cfg=ptrs_cfg,
        )
        for asset_id, asset in game_state.dropped_assets.items()
    }

    # Brand-equity scores live in a PrivateAttr on the game state (excluded from
    # model_dump), so attach them per-asset. These fields are required on the
    # response model, so every asset bucket (live, expired/failed, dropped) must
    # set them or validation fails. be_cost is the cash cost of a brand-equity
    # spend on that drug this step (scales with drug size); computed via the
    # shared MarketingConfig helper so it can never drift from what the engine
    # charges. It is 0 when marketing is off, and for dead drugs (expired/failed/
    # dropped) which can no longer be pushed. Live drugs carry the real cost;
    # brand_score/floor carry over from _brand_scores for any asset that has one.
    for asset_id, asset_resp in base["assets"].items():
        score = game_state._brand_scores.get(asset_id, 0.0)
        floor = game_state._brand_score_floors.get(asset_id, 0.0)
        asset_resp["brand_score"] = score
        asset_resp["brand_score_floor"] = floor
        asset_resp["be_cost"] = (
            marketing_cfg.be_cost(game_state.assets[asset_id].max_revenue)
            if marketing_enabled
            else 0.0
        )
        # Forward-looking preview: project this drug's brand score one step ahead
        # for each spend decision so the panel can show the impact of committing a
        # push before the user commits. When marketing is off there is nothing to
        # project, so both mirror the current score.
        if marketing_enabled:
            asset_resp["brand_score_if_spend"] = marketing_cfg.next_brand_score(
                score, floor, spend=True
            )
            asset_resp["brand_score_if_hold"] = marketing_cfg.next_brand_score(
                score, floor, spend=False
            )
        else:
            asset_resp["brand_score_if_spend"] = score
            asset_resp["brand_score_if_hold"] = score
    for bucket in (base["expired_assets"], base["dropped_assets"]):
        for asset_id, asset_resp in bucket.items():
            score = game_state._brand_scores.get(asset_id, 0.0)
            asset_resp["brand_score"] = score
            asset_resp["brand_score_floor"] = game_state._brand_score_floors.get(
                asset_id, 0.0
            )
            asset_resp["be_cost"] = 0.0
            # Dead drugs can no longer be pushed, so there is no projection.
            asset_resp["brand_score_if_spend"] = score
            asset_resp["brand_score_if_hold"] = score

    logger.debug("Adding properties to response.")
    # Add properties to response
    base["game_ended"] = game_state.game_ended
    base["capital_over_time"] = game_state.capital_over_time
    base["enpv_over_time"] = game_state.enpv_over_time
    base["eroi_over_time"] = game_state.eroi_over_time

    # Add TA experience
    base["ta_experience"] = dict(game_state.ta_experience)

    # Add TA experience config values
    ta_exp_config = game_state._ta_experience_config
    if ta_exp_config is not None:
        base["experience_to_full_knowledge"] = (
            ta_exp_config.experience_to_full_knowledge
        )
        base["max_total_experience"] = ta_exp_config.max_total_experience
    else:
        base["experience_to_full_knowledge"] = 0.0
        base["max_total_experience"] = None

    # Add R&D capacity info
    base["capacity_used"] = game_state.capacity_used
    base["capacity_base"] = game_state.capacity_base
    base["success_modifier"] = game_state.success_modifier
    base["cost_modifier"] = game_state.cost_modifier

    # Add clinical-sites info. operational_sites and sites_in_development are
    # public fields (already in model_dump); free_sites/sites_occupied are
    # properties and must be attached explicitly.
    base["clinical_sites_enabled"] = game_state.clinical_sites_enabled
    base["free_sites"] = game_state.free_sites
    base["sites_occupied"] = game_state.sites_occupied
    base["next_site_purchase_cost"] = game_state.next_site_purchase_cost()

    # Add feature flags
    base["investment_levels_enabled"] = investment_levels_enabled
    base["interim_observations_enabled"] = interim_observations_enabled
    base["distributional_ptrs_enabled"] = distributional_ptrs_enabled
    base["ta_experience_enabled"] = ta_experience_enabled
    base["marketing_enabled"] = marketing_enabled
    base["ptrs_readings_enabled"] = ptrs_readings_enabled

    # Add TA quality estimates (distributional PTRS feature)
    if distributional_ptrs_enabled:
        base["ta_quality"] = {
            ta: {
                "estimate": game_state.ta_quality_estimates[ta],
                "confidence": game_state.ta_quality_confidences[ta],
            }
            for ta in [
                "oncology",
                "respiratory and immunology",
                "vaccines and infectious disease",
            ]
        }
    else:
        base["ta_quality"] = {}

    # Add investment levels configuration for info popup
    if investment_levels_enabled:
        inv_config = game_state._investment_levels_config
        base["investment_levels_config"] = InvestmentLevelsConfigResponse(
            levels={
                level_name: InvestmentLevelConfigResponse(
                    cost_modifier=level_params.cost_modifier,
                    speed_modifier=level_params.speed_modifier,
                    success_modifier=level_params.success_modifier,
                    capacity_cost=level_params.capacity_cost,
                    experience_modifier=level_params.experience_modifier,
                )
                for level_name, level_params in inv_config.levels.items()
            },
            base_capacity=game_state._rd_capacity_config.base_capacity
            if game_state._rd_capacity_config
            else 0.0,
            overage_max_penalty=game_state._rd_capacity_config.overage_max_penalty
            if game_state._rd_capacity_config
            else 0.0,
            overage_cost_max_penalty=game_state._rd_capacity_config.overage_cost_max_penalty
            if game_state._rd_capacity_config
            else 0.0,
        )
    else:
        base["investment_levels_config"] = None

    return base
