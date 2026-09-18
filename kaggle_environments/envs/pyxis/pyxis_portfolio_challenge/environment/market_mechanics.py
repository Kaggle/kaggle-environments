"""Market mechanics for multi-agent competitive environment."""

from __future__ import annotations

import logging
import random
import uuid
from typing import TYPE_CHECKING

from pyxis_portfolio_challenge.game.asset import AssetState, DrugAsset
from pyxis_portfolio_challenge.game.game_state import GameState
from pyxis_portfolio_challenge.game.shared_market_state import (
    THERAPEUTIC_AREAS,
    SharedMarketState,
    indication_key,
)

if TYPE_CHECKING:
    from pyxis_portfolio_challenge.config import MarketingConfig

logger = logging.getLogger(__name__)


def resolve_bd_bid(
    bids: dict[str, float],
    asset: DrugAsset,
    rng: random.Random,
) -> tuple[str | None, float]:
    """
    Resolve a single BD asset auction (first-price sealed-bid).

    Highest cash bid wins and the winner pays their own bid.

    Bids are raw cash amounts (GBP). There is no affordability check here — a
    winner whose bid exceeds their cash simply pays it and may go bankrupt.
    The shared asset's ``cash_enpv`` is still exposed in the observation as
    value guidance, but it no longer parameterises pricing.

    Args:
        bids: Dict mapping agent_id -> cash bid in GBP (``<= 0`` = pass).
        asset: The BD asset being auctioned.
        rng: Random number generator for tie-breaking.

    Returns:
        Tuple of (winner_agent_id, price_paid) or (None, 0.0) if no bids.

    """
    active_bids: list[tuple[str, float]] = [
        (agent_id, float(bid)) for agent_id, bid in bids.items() if bid > 0
    ]

    if not active_bids:
        return None, 0.0

    # Highest cash bid wins (ties broken at random)
    active_bids.sort(key=lambda x: x[1], reverse=True)
    highest_bid = active_bids[0][1]
    top_bidders = [b for b in active_bids if b[1] == highest_bid]

    if len(top_bidders) > 1:
        rng.shuffle(top_bidders)

    winner_agent, winner_price = top_bidders[0]
    logger.debug(
        f"BD Auction: {winner_agent} wins {asset.name} (bid ${winner_price:,.0f})"
    )
    return winner_agent, winner_price


def resolve_site_bid(
    bids: dict[str, float],
    rng: random.Random,
) -> tuple[str | None, float]:
    """
    Resolve a single clinical-site auction. Highest cash bid wins.

    First-price sealed-bid: the winner pays their own bid. Bids of zero or less
    are treated as passes. Ties are broken by ``rng.shuffle`` (like the BD
    auction). No affordability mask — an overbid may bankrupt the winner.

    Args:
        bids: Dict mapping agent_id -> cash bid (£).
        rng: Random number generator for tie-breaking.

    Returns:
        Tuple of (winner_agent_id, price_paid) or (None, 0.0) if no bids.

    """
    active_bids = [(agent_id, bid) for agent_id, bid in bids.items() if bid > 0]
    if not active_bids:
        return None, 0.0

    highest = max(bid for _, bid in active_bids)
    top_bidders = [b for b in active_bids if b[1] == highest]
    if len(top_bidders) > 1:
        rng.shuffle(top_bidders)

    winner_agent, winner_price = top_bidders[0]
    logger.debug(
        f"Site Auction: {winner_agent} wins a clinical site "
        f"(price ${winner_price:,.0f})"
    )
    return winner_agent, winner_price


def calculate_per_drug_indication_shares(
    therapeutic_area: str,
    indication: int,
    shared_market: SharedMarketState,
    agent_portfolios: dict[str, GameState],
    current_time: int,
    pricing_multipliers: dict[uuid.UUID, float] | None = None,
    pricing_elasticity: float = 1.0,
    brand_scores: dict[uuid.UUID, float] | None = None,
    brand_floors: dict[uuid.UUID, float] | None = None,
    marketing_config: "MarketingConfig | None" = None,
) -> dict[uuid.UUID, float]:
    """
    Compute market share for each on-market drug in an indication.

    Each drug competes individually against all other drugs (own + rival).
    First mover bonus applies to the specific first mover drug, not the agent.
    When the first mover drug expires, the bonus disappears.
    """
    if shared_market.disable_market_share_competition:
        shares: dict[uuid.UUID, float] = {}
        for portfolio in agent_portfolios.values():
            for asset in portfolio.assets.values():
                if (
                    asset.therapeutic_area == therapeutic_area
                    and asset.indication == indication
                    and asset.state == AssetState.OnMarket
                ):
                    shares[asset.id] = 1.0
        return shares

    key = indication_key(therapeutic_area, indication)
    ind_market = shared_market.indication_markets.get(key)
    if ind_market is None:
        return {}

    # Collect all on-market drugs and their qualities
    drug_qualities: dict[uuid.UUID, float] = {}
    for portfolio in agent_portfolios.values():
        for asset in portfolio.assets.values():
            if (
                asset.therapeutic_area == therapeutic_area
                and asset.indication == indication
                and asset.state == AssetState.OnMarket
            ):
                tenure_bonus = 1.0 + asset.time_on_market * 0.05
                # demand elasticity: quality = max_rev * (1/price^elast) * tenure
                price_mult = 1.0
                if pricing_multipliers is not None:
                    price_mult = pricing_multipliers.get(asset.id, 1.0)
                price_quality = 1.0 / (price_mult**pricing_elasticity)
                brand_mult = 1.0
                if brand_scores is not None and marketing_config is not None:
                    score = brand_scores.get(asset.id, 0.0)
                    floor = (
                        brand_floors.get(asset.id, 0.0)
                        if brand_floors is not None
                        else 0.0
                    )
                    # Underdog-weighted brand equity: the (1 - floor) factor drives
                    # a large drug's multiplier toward 1.0 (little headroom) while
                    # keeping BE strong for small drugs (large headroom), so a big
                    # drug cannot cheaply combat a small rival's catch-up by
                    # spending itself. score is already max(0, brand_score - floor).
                    brand_mult = 1.0 + marketing_config.be_effectiveness * (
                        1.0 - floor
                    ) * score
                drug_qualities[asset.id] = (
                    asset.max_revenue * price_quality * tenure_bonus * brand_mult
                )

    if not drug_qualities:
        return {}

    # During exclusivity: first mover drug gets 1.0, all others 0.0
    first_mover_drug_id = ind_market.first_mover_drug_id
    if ind_market.is_in_exclusivity(current_time):
        return {
            drug_id: (1.0 if drug_id == first_mover_drug_id else 0.0)
            for drug_id in drug_qualities
        }

    # Post-exclusivity: per-drug quality-weighted shares
    total_quality = sum(drug_qualities.values())
    first_mover_bonus = shared_market.first_mover_bonus

    # First mover bonus only applies if the first mover drug is still on market
    first_mover_active = (
        first_mover_drug_id is not None and first_mover_drug_id in drug_qualities
    )

    shares = {}
    for drug_id, quality in drug_qualities.items():
        proportional = quality / total_quality
        if first_mover_active and drug_id == first_mover_drug_id:
            shares[drug_id] = first_mover_bonus + (1 - first_mover_bonus) * proportional
        elif first_mover_active:
            shares[drug_id] = (1 - first_mover_bonus) * proportional
        else:
            shares[drug_id] = proportional

    # Apply congestion penalty: reduces total revenue when many drugs compete
    # Penalty scales with position in entry_order
    congestion_exp = shared_market.congestion_exponent
    if len(drug_qualities) > 1 and congestion_exp > 0:
        n_drugs = len(drug_qualities)
        ramp_steps = shared_market.congestion_ramp_steps
        incumbent_base = shared_market.congestion_incumbent_penalty
        entry_order = ind_market.entry_order
        position_map = {drug_id: i for i, drug_id in enumerate(entry_order)}
        for drug_id in shares:
            pos = position_map.get(drug_id, len(entry_order))
            ramp = min(pos / ramp_steps, 1.0) if ramp_steps > 0 else 1.0
            penalty_fraction = incumbent_base + (1.0 - incumbent_base) * ramp
            exp = congestion_exp * penalty_fraction
            shares[drug_id] *= 1.0 / (n_drugs**exp)

    return shares


def calculate_per_drug_ta_shares(
    therapeutic_area: str,
    shared_market: SharedMarketState,
    agent_portfolios: dict[str, GameState],
    current_time: int,
    pricing_multipliers: dict[uuid.UUID, float] | None = None,
    pricing_elasticity: float = 1.0,
    brand_scores: dict[uuid.UUID, float] | None = None,
    brand_floors: dict[uuid.UUID, float] | None = None,
    marketing_config: "MarketingConfig | None" = None,
) -> dict[uuid.UUID, float]:
    """Compute per-drug market shares within a TA."""
    if shared_market.disable_market_share_competition:
        shares: dict[uuid.UUID, float] = {}
        for portfolio in agent_portfolios.values():
            for asset in portfolio.assets.values():
                if (
                    asset.therapeutic_area == therapeutic_area
                    and asset.state == AssetState.OnMarket
                ):
                    shares[asset.id] = 1.0
        return shares

    ta_market = shared_market.ta_markets.get(therapeutic_area)
    if ta_market is None:
        return {}

    drug_qualities: dict[uuid.UUID, float] = {}
    for portfolio in agent_portfolios.values():
        for asset in portfolio.assets.values():
            if (
                asset.therapeutic_area == therapeutic_area
                and asset.state == AssetState.OnMarket
            ):
                tenure_bonus = 1.0 + asset.time_on_market * 0.05
                price_mult = 1.0
                if pricing_multipliers is not None:
                    price_mult = pricing_multipliers.get(asset.id, 1.0)
                price_quality = 1.0 / (price_mult**pricing_elasticity)
                brand_mult = 1.0
                if brand_scores is not None and marketing_config is not None:
                    score = brand_scores.get(asset.id, 0.0)
                    floor = (
                        brand_floors.get(asset.id, 0.0)
                        if brand_floors is not None
                        else 0.0
                    )
                    # Underdog-weighted brand equity: the (1 - floor) factor drives
                    # a large drug's multiplier toward 1.0 (little headroom) while
                    # keeping BE strong for small drugs (large headroom), so a big
                    # drug cannot cheaply combat a small rival's catch-up by
                    # spending itself. score is already max(0, brand_score - floor).
                    brand_mult = 1.0 + marketing_config.be_effectiveness * (
                        1.0 - floor
                    ) * score
                drug_qualities[asset.id] = (
                    asset.max_revenue * price_quality * tenure_bonus * brand_mult
                )

    if not drug_qualities:
        return {}

    first_mover_drug_id = ta_market.first_mover_drug_id
    if ta_market.is_in_exclusivity(current_time):
        return {
            drug_id: (1.0 if drug_id == first_mover_drug_id else 0.0)
            for drug_id in drug_qualities
        }

    total_quality = sum(drug_qualities.values())
    first_mover_bonus = shared_market.first_mover_bonus
    first_mover_active = (
        first_mover_drug_id is not None and first_mover_drug_id in drug_qualities
    )

    shares = {}
    for drug_id, quality in drug_qualities.items():
        proportional = quality / total_quality
        if first_mover_active and drug_id == first_mover_drug_id:
            shares[drug_id] = first_mover_bonus + (1 - first_mover_bonus) * proportional
        elif first_mover_active:
            shares[drug_id] = (1 - first_mover_bonus) * proportional
        else:
            shares[drug_id] = proportional

    # Apply congestion penalty (position-based ramp using entry order)
    congestion_exp = shared_market.congestion_exponent
    if len(drug_qualities) > 1 and congestion_exp > 0:
        n_drugs = len(drug_qualities)
        ramp_steps = shared_market.congestion_ramp_steps
        incumbent_base = shared_market.congestion_incumbent_penalty
        entry_order = ta_market.entry_order
        position_map = {drug_id: i for i, drug_id in enumerate(entry_order)}
        for drug_id in shares:
            pos = position_map.get(drug_id, len(entry_order))
            ramp = min(pos / ramp_steps, 1.0) if ramp_steps > 0 else 1.0
            penalty_fraction = incumbent_base + (1.0 - incumbent_base) * ramp
            exp = congestion_exp * penalty_fraction
            shares[drug_id] *= 1.0 / (n_drugs**exp)

    return shares


def calculate_agent_market_shares(
    agent_id: str,
    shared_market: SharedMarketState,
    agent_portfolios: dict[str, GameState],
    current_time: int,
    all_pricing_multipliers: dict[uuid.UUID, float] | None = None,
    pricing_elasticity: float = 1.0,
    brand_scores: dict[uuid.UUID, float] | None = None,
    brand_floors: dict[uuid.UUID, float] | None = None,
    marketing_config: "MarketingConfig | None" = None,
) -> dict[uuid.UUID, float]:
    """
    Calculate per-drug market shares for a specific agent's on-market drugs.

    Returns a dict mapping asset_id -> market share for each of this agent's
    on-market drugs. Each drug competes individually against all other drugs
    in its indication (including the agent's own drugs).

    Args:
        agent_id: Identifier of the agent whose market shares are being calculated.
        shared_market: Shared market state containing indication markets and TA info.
        agent_portfolios: Dict mapping agent_id -> GameState for all agents.
        current_time: Current simulation time step.
        all_pricing_multipliers: Merged pricing multipliers from ALL agents'
            on-market drugs (asset_id -> price_mult). Used in quality formula.
        pricing_elasticity: Demand elasticity for price-share tradeoff.
        brand_scores: Per-drug brand-equity contribution (asset_id -> value),
            already reduced to max(0, brand_score - floor). If None, no
            brand-equity effect is applied.
        brand_floors: Per-drug brand-score floor (asset_id -> floor in [0, 1]),
            i.e. raw_max_rev / pool_peak. Weights the multiplier by (1 - floor)
            so BE is strong for small drugs and near-inert for large ones.
        marketing_config: Marketing feature configuration. If None, marketing
            mechanics are disabled.

    """
    agent_drug_ids = {
        asset.id
        for asset in agent_portfolios[agent_id].assets.values()
        if asset.state == AssetState.OnMarket
    }

    shares: dict[uuid.UUID, float] = {}

    if shared_market.indications_per_ta > 0:
        for ind_market in shared_market.indication_markets.values():
            all_drug_shares = calculate_per_drug_indication_shares(
                ind_market.therapeutic_area,
                ind_market.indication,
                shared_market,
                agent_portfolios,
                current_time,
                pricing_multipliers=all_pricing_multipliers,
                pricing_elasticity=pricing_elasticity,
                brand_scores=brand_scores,
                brand_floors=brand_floors,
                marketing_config=marketing_config,
            )
            for drug_id, share in all_drug_shares.items():
                if drug_id in agent_drug_ids:
                    shares[drug_id] = share
    else:
        for ta in THERAPEUTIC_AREAS:
            all_drug_shares = calculate_per_drug_ta_shares(
                ta,
                shared_market,
                agent_portfolios,
                current_time,
                pricing_multipliers=all_pricing_multipliers,
                pricing_elasticity=pricing_elasticity,
                brand_scores=brand_scores,
                brand_floors=brand_floors,
                marketing_config=marketing_config,
            )
            for drug_id, share in all_drug_shares.items():
                if drug_id in agent_drug_ids:
                    shares[drug_id] = share

    return shares
