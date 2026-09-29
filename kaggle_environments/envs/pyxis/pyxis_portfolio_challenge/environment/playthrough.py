"""
Playthrough capture for multi-agent replay viewer.

Captures full game state at each step in a format matching the frontend
PlaythroughData Zod schema (definitionsGameZ.ts).
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any, Callable

from pydantic import BaseModel

# Re-use the existing serialization helpers from the app layer
from pyxis_portfolio_challenge.app.endpoint_datamodels import (
    alert_to_response,
    bd_asset_to_response,
    game_state_to_response,
    indication_market_to_response,
)
from pyxis_portfolio_challenge.game.asset import AssetState
from pyxis_portfolio_challenge.game.multi_agent_game import MultiAgentGame
from pyxis_portfolio_challenge.game.shared_market_state import (
    THERAPEUTIC_AREAS,
    AlertType,
    SharedMarketState,
    indication_key,
)

# ---- Pydantic models matching the frontend schema ----


class AgentActionRecord(BaseModel):
    """AgentActionRecord."""

    investment_decisions: dict[str, str]  # asset_id -> action ("invest"/"drop"/...)
    bd_bids: list[int]
    bd_assets_at_bid: list[dict[str, Any]]  # serialized BDAssetResponse dicts
    # PTRS research readings commissioned this step (0-value slots omitted).
    ptrs_research: dict[str, int]  # asset_id -> reading count
    bd_ptrs_research: list[int]  # reading count per BD slot
    # Marketing spend this step (only spent slots recorded, value == 1).
    demand_creation: dict[str, int]  # indication_key -> 1
    brand_equity: dict[str, int]  # asset_id -> 1
    # Clinical-site actions this step.
    upgrade: bool  # bought a new clinical site
    site_bid: float  # GBP cash bid in the PvP site auction (0.0 = pass)


class BDAcquisitionRecord(BaseModel):
    """BDAcquisitionRecord."""

    name: str
    price: float


class SharedMarketSnapshot(BaseModel):
    """SharedMarketSnapshot."""

    bd_assets: list[dict[str, Any]]  # serialized BDAssetResponse dicts
    alerts: list[dict[str, Any]]  # serialized AlertResponse dicts
    indication_markets: list[dict[str, Any]]
    last_bd_acquisitions: dict[str, list[BDAcquisitionRecord]]


class StepRecord(BaseModel):
    """StepRecord."""

    step: int
    actions: dict[str, AgentActionRecord]  # agent_id -> actions
    agent_states: dict[str, dict[str, Any]]  # agent_id -> serialized game state
    shared_market: SharedMarketSnapshot
    rewards: dict[str, float]
    cumulative_rewards: dict[str, float]


class PlaythroughMetadata(BaseModel):
    """PlaythroughMetadata."""

    num_agents: int
    agent_ids: list[str]
    agent_names: dict[str, str]  # agent_id -> display name
    horizon: int
    seed: int | None
    captured_at: str


class PlaythroughConfig(BaseModel):
    """Pydantic model config."""

    bd_enabled: bool
    bd_max_bid: float
    reinvestment_percentage: float
    investment_levels_enabled: bool
    interim_observations_enabled: bool
    distributional_ptrs_enabled: bool
    ta_experience_enabled: bool
    congestion_exponent: float
    congestion_ramp_steps: int
    congestion_incumbent_penalty: float
    rd_capacity_enabled: bool
    rd_capacity_base: float
    marketing_enabled: bool
    clinical_sites_enabled: bool
    site_auction_enabled: bool
    ptrs_readings_enabled: bool
    drop_action_enabled: bool


class PlaythroughData(BaseModel):
    """PlaythroughData."""

    metadata: PlaythroughMetadata
    config: PlaythroughConfig
    initial_agent_states: dict[str, dict[str, Any]]
    initial_shared_market: SharedMarketSnapshot
    steps: list[StepRecord]


# ---- Capture functions ----


def _serialize_game_state(game_state, indication_name_map=None) -> dict[str, Any]:
    """Serialize a GameState to the frontend-compatible dict format."""
    response = game_state_to_response(game_state, indication_name_map)
    # game_state_to_response returns a dict (despite its type hint)
    if isinstance(response, dict):
        return _json_serialize(response)
    return response.model_dump(mode="json")


def _json_serialize(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable types (UUIDs, enums, etc.)."""
    import enum

    if isinstance(obj, dict):
        return {str(k): _json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serialize(v) for v in obj]
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def capture_agent_states(
    game: MultiAgentGame,
) -> dict[str, dict[str, Any]]:
    """Capture all agent states as serialized dicts."""
    ind_name_map = game.shared_market.indication_name_map
    return {
        agent_id: _serialize_game_state(state, ind_name_map)
        for agent_id, state in game.agent_states.items()
    }


def _serialize_shared_market(
    shared_market: SharedMarketState,
    current_time: int,
    agent_ids: list[str],
    reinvestment_percentage: float,
    cached_market_shares: dict[str, dict] | None = None,
    agent_portfolios: dict | None = None,
) -> SharedMarketSnapshot:
    """Serialize SharedMarketState to frontend-compatible snapshot."""
    ind_name_map = shared_market.indication_name_map

    bd_assets = [
        bd_asset_to_response(
            a,
            ind_name_map,
            reinvestment_percentage,
            ptrs_cfg=None,
            clone=None,
        ).model_dump(mode="json")
        for a in shared_market.current_bd_assets
    ]

    alerts = [
        alert_to_response(alert, ind_name_map).model_dump(mode="json")
        for alert in shared_market.alerts
    ]

    # Build drug->indication lookup from agent portfolios
    drug_indication: dict[uuid.UUID, tuple[str, int]] = {}
    if agent_portfolios:
        for agent_state in agent_portfolios.values():
            for asset in agent_state.assets.values():
                if asset.state == AssetState.OnMarket:
                    drug_indication[asset.id] = (
                        asset.therapeutic_area,
                        asset.indication,
                    )

    # For replay, show all agents equally (no "You" vs competitors)
    indication_markets = []
    for market in shared_market.indication_markets.values():
        # Use first agent as dummy "player" — the replay viewer shows all equally
        # Replay is retrospective: no forward projection is needed, so pass no
        # marketing config (both preview branches then mirror the current value).
        resp = indication_market_to_response(
            market,
            current_time,
            agent_ids[0] if agent_ids else "",
            name_map=None,
            marketing_cfg=None,
        )
        market_dict = resp.model_dump(mode="json")

        # Compute per-agent aggregate market share for this indication
        if cached_market_shares:
            agent_shares: dict[str, float] = {}
            for agent_id in agent_ids:
                agent_drug_shares = cached_market_shares.get(agent_id, {})
                total = 0.0
                for drug_id, share in agent_drug_shares.items():
                    if isinstance(drug_id, str):
                        drug_id = uuid.UUID(drug_id)
                    ind = drug_indication.get(drug_id)
                    if (
                        ind
                        and ind[0] == market.therapeutic_area
                        and ind[1] == market.indication
                    ):
                        total += share
                if total > 0:
                    agent_shares[agent_id] = total
            market_dict["market_shares"] = agent_shares
        else:
            market_dict["market_shares"] = {}

        indication_markets.append(market_dict)

    # Derive last BD acquisitions from BD_DEAL alerts at current step
    last_bd_acquisitions: dict[str, list[BDAcquisitionRecord]] = {}
    for alert in shared_market.alerts:
        if alert.event_type == AlertType.BD_DEAL and alert.step == current_time - 1:
            name = alert.details.get(
                "asset_name", str(alert.details.get("asset_id", ""))[:8]
            )
            price = float(alert.details.get("price", 0))
            last_bd_acquisitions.setdefault(alert.agent_id, []).append(
                BDAcquisitionRecord(name=name, price=price)
            )

    return SharedMarketSnapshot(
        bd_assets=bd_assets,
        alerts=alerts,
        indication_markets=indication_markets,
        last_bd_acquisitions=last_bd_acquisitions,
    )


def capture_shared_market(game: MultiAgentGame) -> SharedMarketSnapshot:
    """Capture the shared market state as a serialized snapshot."""
    # The reinvestment rate is a global config value, identical across agents.
    reinvestment_percentage = next(
        iter(game.agent_states.values())
    ).reinvestment_percentage
    return _serialize_shared_market(
        game.shared_market,
        game.time,
        list(game.agent_states.keys()),
        reinvestment_percentage,
        cached_market_shares=game._cached_market_shares,
        agent_portfolios=game.agent_states,
    )


def capture_actions(
    raw_actions: dict[str, Any],
    asset_id_orders: dict[str, list[uuid.UUID | None]],
    use_investment_levels: bool,
    pre_step_bd_assets: list[Any] | None = None,
    bd_bid_decoder: Callable[[Any], list[float]] | None = None,
    env=None,
) -> dict[str, AgentActionRecord]:
    """
    Convert raw env actions to AgentActionRecord dicts.

    raw_actions: {agent_id: {"investments": np.ndarray, "bd_bids": np.ndarray, ...}}
        The action dict may also carry the ``ptrs_research``, ``demand_creation``,
        ``brand_equity``, ``upgrade`` and ``site_bid`` heads; agents that omit a
        head (e.g. the knapsack heuristic) simply record no activity for it.
    asset_id_orders: {agent_id: [asset_id_or_None, ...]} — maps indices to asset IDs
    bd_bid_decoder: optional env decoder (e.g. ``env._decode_bd_bids``) that maps
        the raw BD action (GBP millions) to the GBP cash amount the auction
        actually uses. When provided, bids are stored in GBP so replays display
        the real amount (a £25M bid stored as 25_000_000, not 25). When None the
        raw action value is stored unchanged (legacy behaviour).
    env: the training-gym env, used to decode the marketing / clinical-site /
        PTRS-reading heads (slot→asset-id / slot→indication-key mappings and the
        site-bid decoder). When None those heads are recorded as empty.
    """
    result: dict[str, AgentActionRecord] = {}

    # Env-derived shapes and feature flags (all heads default to inactive when
    # env is absent or the feature is disabled).
    max_num_assets = getattr(env, "max_num_assets", 0)
    bd_max_slots = getattr(env, "bd_max_slots", 0)
    indications_per_ta = getattr(env, "_indications_per_ta", 0)
    max_indications_per_ta = getattr(env, "max_indications_per_ta", 0)
    marketing_on = (
        env is not None
        and env.marketing_config is not None
        and env.marketing_config.enabled
    )
    ptrs_on = (
        env is not None
        and env.ptrs_readings_config is not None
        and env.ptrs_readings_config.enabled
    )
    sites_on = env is not None and env._sites_on
    site_auction_on = env is not None and env._site_auction_on

    for agent_id, action in raw_actions.items():
        investments = action.get("investments", [])
        bd_bids = action.get("bd_bids", [])
        order = asset_id_orders.get(agent_id, [])

        investment_decisions: dict[str, str] = {}
        for i, asset_id in enumerate(order):
            if asset_id is None:
                continue
            if i < len(investments):
                val = int(investments[i])
                if use_investment_levels:
                    level_map = {
                        0: "none",
                        1: "minimal",
                        2: "standard",
                        3: "accelerated",
                        4: "stop",
                    }
                    investment_decisions[str(asset_id)] = level_map.get(val, "none")
                elif val == 1:
                    investment_decisions[str(asset_id)] = "invest"
                elif val == 2:
                    # drop_action mode: MultiDiscrete([3]) with 2 == drop.
                    investment_decisions[str(asset_id)] = "drop"
                else:
                    investment_decisions[str(asset_id)] = "none"

        if not (hasattr(bd_bids, "__len__") and len(bd_bids) > 0):
            bd_bid_list = [0]
        elif bd_bid_decoder is not None:
            # Store the decoded GBP amount used by the auction, not GBP millions.
            bd_bid_list = [int(round(b)) for b in bd_bid_decoder(bd_bids)]
        else:
            bd_bid_list = [int(b) for b in bd_bids]

        bd_assets_serialized = []
        if pre_step_bd_assets:
            for ba in pre_step_bd_assets:
                if hasattr(ba, "model_dump"):
                    bd_assets_serialized.append(ba.model_dump(mode="json"))
                elif isinstance(ba, dict):
                    bd_assets_serialized.append(ba)

        # PTRS research readings: first max_num_assets slots map to portfolio
        # assets (via the asset order), the remaining bd_max_slots to BD offers.
        ptrs_research: dict[str, int] = {}
        bd_ptrs_research: list[int] = []
        ptr_arr = action.get("ptrs_research") if ptrs_on else None
        if ptr_arr is not None and len(ptr_arr) > 0:
            for i, asset_id in enumerate(order):
                if asset_id is None or i >= max_num_assets or i >= len(ptr_arr):
                    continue
                n = int(ptr_arr[i])
                if n > 0:
                    ptrs_research[str(asset_id)] = n
            for slot in range(bd_max_slots):
                idx = max_num_assets + slot
                bd_ptrs_research.append(int(ptr_arr[idx]) if idx < len(ptr_arr) else 0)

        # Demand creation: slot = ta_idx * max_indications_per_ta + ind_idx.
        demand_creation: dict[str, int] = {}
        dc_arr = action.get("demand_creation") if marketing_on else None
        if dc_arr is not None and len(dc_arr) > 0:
            for ta_idx, ta in enumerate(THERAPEUTIC_AREAS):
                for ind_idx in range(indications_per_ta):
                    bucket = ta_idx * max_indications_per_ta + ind_idx
                    if bucket < len(dc_arr) and int(dc_arr[bucket]) == 1:
                        demand_creation[indication_key(ta, ind_idx)] = 1

        # Brand equity: per-asset spend, indexed like investments.
        brand_equity: dict[str, int] = {}
        be_arr = action.get("brand_equity") if marketing_on else None
        if be_arr is not None and len(be_arr) > 0:
            for i, asset_id in enumerate(order):
                if asset_id is None or i >= len(be_arr):
                    continue
                if int(be_arr[i]) == 1:
                    brand_equity[str(asset_id)] = 1

        # Clinical-site actions.
        upgrade = False
        if sites_on:
            upgrade = bool(int(round(float(action.get("upgrade", 0)))))
        site_bid = 0.0
        if site_auction_on and "site_bid" in action:
            site_bid = float(env._decode_site_bid(action["site_bid"]))

        result[agent_id] = AgentActionRecord(
            investment_decisions=investment_decisions,
            bd_bids=bd_bid_list,
            bd_assets_at_bid=bd_assets_serialized,
            ptrs_research=ptrs_research,
            bd_ptrs_research=bd_ptrs_research,
            demand_creation=demand_creation,
            brand_equity=brand_equity,
            upgrade=upgrade,
            site_bid=site_bid,
        )

    return result


def _delta_compress_expired_assets(
    initial_agent_states: dict[str, dict[str, Any]],
    steps: list[StepRecord],
) -> None:
    """
    Replace full expired_assets dicts with per-step deltas (in-place).

    Each step's agent_states[agent].expired_assets is replaced with only the
    assets that are *newly* expired/failed since the previous step. The FE
    reconstructs full sets by accumulating deltas from step 0 onwards.
    """
    agent_ids = list(initial_agent_states.keys())

    # Track the set of expired asset IDs seen so far per agent
    prev_expired: dict[str, set[str]] = {}
    for agent_id in agent_ids:
        prev_expired[agent_id] = set(
            initial_agent_states[agent_id].get("expired_assets", {}).keys()
        )

    for step in steps:
        for agent_id in agent_ids:
            state = step.agent_states[agent_id]
            full_expired = state.get("expired_assets", {})
            current_ids = set(full_expired.keys())

            # Only keep newly expired assets
            new_ids = current_ids - prev_expired[agent_id]
            state["expired_assets"] = {k: full_expired[k] for k in new_ids}

            prev_expired[agent_id] = current_ids


def build_playthrough_data(
    env,
    seed: int | None,
    initial_agent_states: dict[str, dict[str, Any]],
    initial_shared_market: SharedMarketSnapshot,
    steps: list[StepRecord],
    agent_names: dict[str, str] | None = None,
) -> PlaythroughData:
    """Build the complete PlaythroughData from captured data."""
    game = env.multi_agent_game
    agent_ids = list(game.agent_states.keys())
    shared_market = game.shared_market

    # Delta-compress expired assets to reduce JSON size
    _delta_compress_expired_assets(initial_agent_states, steps)

    # Detect feature flags from first agent's state
    first_state = list(game.agent_states.values())[0]

    investment_levels_enabled = (
        hasattr(first_state, "_investment_levels_config")
        and first_state._investment_levels_config is not None
        and first_state._investment_levels_config.enabled
    )
    interim_observations_enabled = (
        hasattr(first_state, "_interim_trial_observations_config")
        and first_state._interim_trial_observations_config is not None
        and first_state._interim_trial_observations_config.enabled
    )
    distributional_ptrs_enabled = (
        hasattr(first_state, "_distributional_ptrs_config")
        and first_state._distributional_ptrs_config is not None
        and first_state._distributional_ptrs_config.enabled
    )
    ta_experience_enabled = (
        hasattr(first_state, "_ta_experience_config")
        and first_state._ta_experience_config is not None
        and first_state._ta_experience_config.enabled
    )
    rd_capacity_enabled = (
        hasattr(first_state, "_rd_capacity_config")
        and first_state._rd_capacity_config is not None
        and first_state._rd_capacity_config.enabled
    )
    rd_capacity_base = (
        first_state._rd_capacity_config.base_capacity if rd_capacity_enabled else 0.0
    )

    marketing_enabled = (
        env.marketing_config is not None and env.marketing_config.enabled
    )
    ptrs_readings_enabled = (
        env.ptrs_readings_config is not None and env.ptrs_readings_config.enabled
    )
    drop_action_enabled = (
        env.drop_action_config is not None and env.drop_action_config.enabled
    )

    return PlaythroughData(
        metadata=PlaythroughMetadata(
            num_agents=len(agent_ids),
            agent_ids=agent_ids,
            agent_names=agent_names or {aid: aid for aid in agent_ids},
            horizon=game.horizon,
            seed=seed,
            captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ),
        config=PlaythroughConfig(
            bd_enabled=shared_market.bd_enabled,
            bd_max_bid=shared_market.bd_max_bid,
            reinvestment_percentage=first_state.reinvestment_percentage,
            investment_levels_enabled=investment_levels_enabled,
            interim_observations_enabled=interim_observations_enabled,
            distributional_ptrs_enabled=distributional_ptrs_enabled,
            ta_experience_enabled=ta_experience_enabled,
            congestion_exponent=shared_market.congestion_exponent,
            congestion_ramp_steps=shared_market.congestion_ramp_steps,
            congestion_incumbent_penalty=shared_market.congestion_incumbent_penalty,
            rd_capacity_enabled=rd_capacity_enabled,
            rd_capacity_base=rd_capacity_base,
            marketing_enabled=marketing_enabled,
            clinical_sites_enabled=env._sites_on,
            site_auction_enabled=env._site_auction_on,
            ptrs_readings_enabled=ptrs_readings_enabled,
            drop_action_enabled=drop_action_enabled,
        ),
        initial_agent_states=initial_agent_states,
        initial_shared_market=initial_shared_market,
        steps=steps,
    )
