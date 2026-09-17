"""Multi-agent game orchestrator wrapping N GameState instances."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import Literal

import upath
from pydantic import BaseModel, ConfigDict, PrivateAttr

from pyxis_portfolio_challenge.game.asset import AssetState
from pyxis_portfolio_challenge.game.asset_generators import JSONAssetGenerator
from pyxis_portfolio_challenge.game.constants import InvestmentLevel
from pyxis_portfolio_challenge.game.game_state import _PACKAGE_CODE_HASH, GameState
from pyxis_portfolio_challenge.game.shared_market_state import (
    THERAPEUTIC_AREAS,
    SharedMarketState,
)
from pyxis_portfolio_challenge.game.shared_market_state import (
    indication_key as _ind_key,
)
from pyxis_portfolio_challenge.game.trial import TrialPhase
from pyxis_portfolio_challenge.rng import get_game_rng, init_game_rng

logger = logging.getLogger(__name__)


class MultiAgentGame(BaseModel):
    """
    Immutable orchestrator for multi-agent competitive investment game.

    Wraps N GameState instances (one per agent) and a SharedMarketState.
    All single-player game dynamics are delegated to GameState.step().
    This class only adds cross-agent interactions:
    - BD (Business Development) auction resolution
    - Market share competition (revenue splitting)
    - Event-driven pipeline leak alerts (competitive intelligence)
    - Drug release / expiry tracking across agents

    Immutable: step() returns a new MultiAgentGame instance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_states: dict[str, GameState]
    shared_market: SharedMarketState
    time: int
    horizon: int
    num_agents: int
    # DC-spend leaks activate only from this agent count (redundant below it —
    # see step()). Defaulted so direct constructions need not set it.
    dc_leak_min_agents: int = 3
    disable_market_share_competition: bool
    bd_max_slots: int
    bd_persist_steps: int = 1
    pricing_elasticity: float

    # Cached per-drug market shares from last step, keyed by agent.
    # Avoids recomputing in observation building.
    _cached_market_shares: dict[str, dict] = PrivateAttr(default_factory=dict)
    # Optional display names for agents (agent_id -> display_name)
    _display_names: dict[str, str] = PrivateAttr(default_factory=dict)
    # Global peak max_revenue across the full static asset pool (computed once at init).
    # Used to normalize BE initialization scores against a stable,
    # episode-independent reference.
    _be_static_peak_revenue: float = PrivateAttr(default=0.0)

    @classmethod
    def initialise(
        cls,
        num_agents: int,
        seed: int,
        starting_cash: float,
        horizon: int,
        equilibrium_num_assets: int,
        max_num_assets: int,
        asset_arrival_sensitivity_below: float,
        asset_arrival_sensitivity_above: float,
        reinvestment_percentage: float,
        assets_dir: upath.UPath,
        exclusivity_period: int,
        first_mover_bonus: float,
        disable_market_share_competition: bool,
        alert_history_length: int,
        reward_fn_config: dict,
        distributional_ptrs_config,
        ta_experience_config,
        uncertain_ptrs_config,
        investment_levels_config,
        interim_trial_observations_config,
        rd_capacity_config,
        drop_action_config,
        ptrs_readings_config,
        clinical_sites_config,
        marketing_config,
        indications_per_ta: int,
        indication_spread: float,
        indication_drift_speed: float,
        trial_cost_multiplier: float,
        approval_phase_config,
        # BD configuration
        bd_enabled: bool,
        bd_assets_dir: upath.UPath | None,
        bd_base_lambda: float,
        bd_leak_lambda_boost: float,
        bd_min_step: int,
        bd_max_bid: float,
        bd_phase_weights: list[float],
        bd_indication_activity_bias: float,
        bd_max_slots: int,
        # Leak configuration
        leak_phase_probabilities: list[float],
        be_leak_probability: float,
        dc_leak_probability: float,
        # Congestion penalty
        congestion_exponent: float,
        congestion_ramp_steps: int,
        congestion_incumbent_penalty: float,
        # Pricing elasticity
        pricing_elasticity: float,
        bd_persist_steps: int,
        dc_leak_min_agents: int,
    ) -> "MultiAgentGame":
        """
        Create initial multi-agent game with N GameState instances.

        Each agent gets its own GameState via GameState.initialise_new_game().
        """
        agent_names = [f"pharma_{i}" for i in range(num_agents)]
        rng = init_game_rng(seed)

        # Build indications_per_ta dict for asset generators
        indications_per_ta_dict = None
        indication_permutation = None
        if indications_per_ta > 0:
            indications_per_ta_dict = {
                ta: indications_per_ta for ta in THERAPEUTIC_AREAS
            }
            # Random permutation per TA so drift order != observed index
            indication_permutation = {}
            for ta in THERAPEUTIC_AREAS:
                perm = list(range(indications_per_ta))
                rng.shuffle(perm)
                indication_permutation[ta] = perm

        # Create per-agent GameState instances
        agent_states = {}
        for i, agent_name in enumerate(agent_names):
            game_state = GameState.initialise_new_game(
                asset_generator_cls=JSONAssetGenerator,
                num_assets=equilibrium_num_assets,
                max_num_assets=max_num_assets,
                cash=starting_cash,
                horizon=horizon,
                asset_arrival_sensitivity_below=asset_arrival_sensitivity_below,
                asset_arrival_sensitivity_above=asset_arrival_sensitivity_above,
                reinvestment_percentage=reinvestment_percentage,
                seed=None,
                assets_dir=assets_dir,
                ta_experience_config=ta_experience_config,
                uncertain_ptrs_config=uncertain_ptrs_config,
                investment_levels_config=investment_levels_config,
                interim_trial_observations_config=interim_trial_observations_config,
                distributional_ptrs_config=distributional_ptrs_config,
                rd_capacity_config=rd_capacity_config,
                drop_action_config=drop_action_config,
                ptrs_readings_config=ptrs_readings_config,
                clinical_sites_config=clinical_sites_config,
                marketing_config=marketing_config,
                indications_per_ta=indications_per_ta_dict,
                indication_spread=indication_spread,
                indication_drift_speed=indication_drift_speed,
                trial_cost_multiplier=trial_cost_multiplier,
                approval_phase_config=approval_phase_config,
                generator_index=i,
            )
            if indication_permutation:
                game_state._asset_generator.set_indication_permutation(
                    indication_permutation
                )
            agent_states[agent_name] = game_state

        # Create shared market state
        shared_market = SharedMarketState.initialize(
            exclusivity_period=exclusivity_period,
            first_mover_bonus=first_mover_bonus,
            alert_history_length=alert_history_length,
            disable_market_share_competition=disable_market_share_competition,
            num_indications_per_ta=indications_per_ta,
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
            site_auction_enabled=(
                clinical_sites_config is not None
                and clinical_sites_config.enabled
                and clinical_sites_config.auction_enabled
            ),
            site_auction_interval_steps=(
                clinical_sites_config.auction_interval_steps
                if clinical_sites_config is not None
                else 20
            ),
            site_auction_min_step=(
                clinical_sites_config.auction_min_step
                if clinical_sites_config is not None
                else 0
            ),
            congestion_exponent=congestion_exponent,
            congestion_ramp_steps=congestion_ramp_steps,
            congestion_incumbent_penalty=congestion_incumbent_penalty,
        )

        # Create BD asset generator
        if bd_enabled and bd_assets_dir is not None:
            ta_quality_modifiers = agent_states[agent_names[0]]._ta_quality_modifiers
            bd_generator = JSONAssetGenerator(
                bd_assets_dir,
                distributional_ptrs_config=distributional_ptrs_config,
                ta_quality_modifiers=ta_quality_modifiers,
                ta_experience_config=ta_experience_config,
                uncertain_ptrs_config=uncertain_ptrs_config,
                ptrs_readings_config=ptrs_readings_config,
                indications_per_ta=indications_per_ta_dict,
                indication_spread=indication_spread,
                indication_drift_speed=indication_drift_speed,
                trial_cost_multiplier=trial_cost_multiplier,
                approval_phase_config=approval_phase_config,
                generator_index=len(agent_names),
            )
            if indication_permutation:
                bd_generator.set_indication_permutation(indication_permutation)
            shared_market.set_bd_asset_generator(bd_generator)

        # Compute global peak max_revenue from all static asset pools (main + BD)
        # for BE init.
        be_static_peak_revenue = 0.0
        gens_to_scan = [agent_states[agent_names[0]]._asset_generator]
        if shared_market._bd_asset_generator is not None:
            gens_to_scan.append(shared_market._bd_asset_generator)
        for gen in gens_to_scan:
            for stage_assets in gen._all_assets.values():
                for asset_data in stage_assets:
                    rev = asset_data.get("max_revenue", 0.0)
                    if rev > be_static_peak_revenue:
                        be_static_peak_revenue = rev

        game = cls(
            agent_states=agent_states,
            shared_market=shared_market,
            time=0,
            horizon=horizon,
            num_agents=num_agents,
            dc_leak_min_agents=dc_leak_min_agents,
            disable_market_share_competition=disable_market_share_competition,
            bd_max_slots=bd_max_slots,
            bd_persist_steps=bd_persist_steps,
            pricing_elasticity=pricing_elasticity,
        )
        game._be_static_peak_revenue = be_static_peak_revenue
        return game

    def rebase_time_to_zero(self) -> "MultiAgentGame":
        """
        Return a copy of this game with the clock reset to time 0.

        Treats everything played so far (warmup) as a pre-roll: the agent
        then experiences a full ``horizon`` starting at time 0. The game
        clock, the shared market clock, and every per-agent ``GameState``
        clock are shifted back by the current ``time`` offset. All existing
        ``time >= horizon`` / progress / cadence checks therefore keep
        working unchanged, and the agent gets the full horizon to act.

        Only the absolute game clock is rebased; relative durations (asset
        timers, BD asset ages) are left intact — they describe the warmed
        market state the agent inherits.
        """
        if self.time == 0:
            return self

        # Each sub-model rebases the fields it owns (co-located with their
        # definitions), so this orchestrator never enumerates time fields.
        # Both are mutated in place — safe because the pre-rebase game is
        # discarded and the copy below reuses the same sub-model objects.
        self.shared_market.rebase_time_to_zero()
        for state in self.agent_states.values():
            state.rebase_time_to_zero()

        # MultiAgentGame is frozen, so its own clock is reset via model_copy;
        # private attrs (cached shares, display names) are copied too.
        new_game = self.model_copy(update={"time": 0})
        new_game._cached_market_shares = self._cached_market_shares
        new_game._display_names = self._display_names
        return new_game

    def with_horizon(self, horizon: int) -> "MultiAgentGame":
        """
        Return a copy of this game whose horizon is ``horizon``.

        Sets both the game-level clock and every per-agent ``GameState``
        horizon so ``time >= horizon`` termination fires consistently. Used
        by the warmup wrapper to make ``warmup_steps`` and ``horizon``
        additive: the pre-roll runs on an extended horizon (so it never trips
        horizon termination, however long the warmup), then the configured
        horizon is restored once ``rebase_time_to_zero`` has handed the agent
        a fresh clock at time 0. Only call while ``time <= horizon`` holds for
        the new horizon (the wrapper only calls it at time 0).
        """
        # Per-agent states are mutated in place (GameState is mutable);
        # MultiAgentGame is frozen so its own horizon is reset via model_copy.
        # Safe because the pre-update game is discarded and the copy reuses the
        # same sub-model objects (mirrors rebase_time_to_zero above).
        for state in self.agent_states.values():
            state.horizon = horizon
        new_game = self.model_copy(update={"horizon": horizon})
        new_game._cached_market_shares = self._cached_market_shares
        new_game._display_names = self._display_names
        return new_game

    def content_fingerprint(self, seed: int | None = None) -> str:
        """
        Compute a deterministic fingerprint for this initial multi-agent game state.

        Serializes all agent portfolios and the shared market, strips the random
        GameState.id from each agent (asset UUIDs are deterministic/seeded so they
        stay in), then hashes with seed and package code hash.
        """
        data = self.model_dump(mode="json")
        for agent_data in data.get("agent_states", {}).values():
            agent_data.pop("id", None)
        fingerprint_input = {
            "state": data,
            "seed": seed,
            "code_hash": _PACKAGE_CODE_HASH,
        }
        serialized = json.dumps(fingerprint_input, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    @property
    def possible_agents(self) -> list[str]:
        """Get list of all agent names."""
        return list(self.agent_states.keys())

    @property
    def active_agents(self) -> list[str]:
        """Get list of non-bankrupt agents."""
        return [
            agent for agent, state in self.agent_states.items() if not state.game_ended
        ]

    def _label(self, agent_id: str) -> str:
        """Return display label for an agent, falling back to agent_id."""
        return self._display_names.get(agent_id, agent_id)

    def step(
        self,
        investor_actions: dict[
            str,
            dict[uuid.UUID, InvestmentLevel | Literal["invest"] | None],
        ],
        bd_bids: dict[str, list[float]] | None = None,
        pricing_actions: dict[str, dict[uuid.UUID, float]] | None = None,
        research_actions: dict[str, dict[uuid.UUID, int]] | None = None,
        marketing_actions: dict[str, dict] | None = None,
        site_bids: dict[str, float] | None = None,
        buy_site_actions: dict[str, bool] | None = None,
        site_priorities: dict[str, dict[uuid.UUID, float]] | None = None,
    ) -> "MultiAgentGame":
        """
        Advance all agents by one step.

        Args:
            investor_actions: Per-agent investment decisions.
                {agent_id: {asset_uuid: InvestmentLevel}}
            bd_bids: Per-agent BD cash bids per slot.
                {agent_id: [slot_0_cash, slot_1_cash, ...]}
                Raw GBP amounts; <= 0 = pass. Highest bid wins, pays own bid.
            pricing_actions: Per-agent pricing multipliers for on-market drugs.
                {agent_id: {asset_uuid: price_multiplier}}
                If None, all drugs use 1.0x pricing.
            research_actions: Per-agent PTRS reading counts to purchase this step.
                {agent_id: {asset_uuid: num_readings}}
                If None, no readings are taken.
            marketing_actions: Per-agent demand-creation and brand-equity spend
                decisions. {agent_id: {"demand_creation": ..., "brand_equity": ...}}
                If None, no marketing spend occurs.
            site_bids: Per-agent cash bid for the clinical-site auction this step.
                {agent_id: cash_bid}. Only used when a site is up for auction.
            buy_site_actions: Per-agent clinical-site "upgrade" decision.
                {agent_id: bool}. When True, the agent buys one site (Fibonacci
                cost, build delay) inside its GameState.step.
            site_priorities: Per-agent, per-asset priority scores for clinical-
                site arbitration when ``agent_priority`` is enabled.
                {agent_id: {asset_uuid: priority}}.

        Returns:
            New MultiAgentGame with updated state.

        """
        from pyxis_portfolio_challenge.environment.market_mechanics import (
            calculate_agent_market_shares,
            resolve_bd_bid,
            resolve_site_bid,
        )

        new_agent_states = dict(self.agent_states)
        # Mutate shared_market in place — the old MultiAgentGame reference
        # is only used post-step for agent_states (reward calc), not shared_market.
        # This avoids a costly deep copy (~4ms per step).
        new_shared_market = self.shared_market

        # Phase 0: Resolve BD Auctions (one per slot, highest bid wins each)
        won_asset_ids: set = set()
        if (
            bd_bids
            and new_shared_market.current_bd_assets
            and new_shared_market.bd_enabled
        ):
            for slot_idx, bd_asset in enumerate(new_shared_market.current_bd_assets):
                # Extract per-slot cash bids from each agent. Skip agents that
                # have already ended (e.g. bankrupted by winning an earlier slot
                # this step): they cannot acquire the asset, so they must not win
                # the slot and strand it away from the live bidders.
                slot_bids: dict[str, float] = {}
                for agent_id, agent_bid_list in bd_bids.items():
                    if new_agent_states[agent_id].game_ended:
                        continue
                    if slot_idx < len(agent_bid_list):
                        slot_bids[agent_id] = agent_bid_list[slot_idx]

                winner, price = resolve_bd_bid(
                    bids=slot_bids,
                    asset=bd_asset,
                    rng=get_game_rng(),
                )

                if winner is not None:
                    state = new_agent_states[winner]
                    if state.game_ended:
                        # Winner can no longer acquire; leave the asset in the
                        # market rather than removing it unpaid-for.
                        continue
                    # Mark as won (and thus removed from the market) only once
                    # the winner is confirmed able to acquire it.
                    won_asset_ids.add(bd_asset.id)
                    new_cash = state.cash - price
                    new_assets = dict(state.assets)
                    new_assets[bd_asset.id] = bd_asset
                    new_shared_market.register_bd_deal(winner, bd_asset, price=price)
                    logger.debug(
                        f"{winner} acquired BD asset {bd_asset.name} "
                        f"(slot {slot_idx}) for ${price:,.0f}"
                    )

                    updated_realised_costs = list(state.realised_costs)
                    if updated_realised_costs:
                        updated_realised_costs[-1] += price
                    else:
                        updated_realised_costs.append(price)

                    new_agent_states[winner] = GameState(
                        id=state.id,
                        cash=new_cash,
                        time=state.time,
                        horizon=state.horizon,
                        equilibrium_num_assets=state.equilibrium_num_assets,
                        max_num_assets=state.max_num_assets,
                        asset_arrival_sensitivity_below=state.asset_arrival_sensitivity_below,
                        asset_arrival_sensitivity_above=state.asset_arrival_sensitivity_above,
                        reinvestment_percentage=state.reinvestment_percentage,
                        initial_cash=state.initial_cash,
                        assets=new_assets,
                        failed_assets=dict(state.failed_assets),
                        expired_assets=dict(state.expired_assets),
                        dropped_assets=dict(state.dropped_assets),
                        realised_costs=updated_realised_costs,
                        realised_revenues=list(state.realised_revenues),
                        running_enpv=list(state.running_enpv),
                        running_eroi=list(state.running_eroi),
                        game_ended=new_cash < 0 or state.time >= state.horizon,
                        ended_reason=(
                            state.ended_reason
                            if state.ended_reason
                            else (
                                "horizon_reached"
                                if state.time >= state.horizon
                                else ("bankrupt" if new_cash < 0 else None)
                            )
                        ),
                        ta_experience=dict(state.ta_experience),
                        capacity_used=state.capacity_used,
                        capacity_base=state.capacity_base,
                        ta_quality_estimates=dict(state.ta_quality_estimates),
                        ta_quality_confidences=dict(state.ta_quality_confidences),
                        operational_sites=state.operational_sites,
                        sites_in_development=list(state.sites_in_development),
                    )
                    ns = new_agent_states[winner]
                    ns._asset_generator = state._asset_generator
                    ns._ta_experience_config = state._ta_experience_config
                    ns._uncertain_ptrs_config = state._uncertain_ptrs_config
                    ns._investment_levels_config = state._investment_levels_config
                    ns._interim_trial_observations_config = (
                        state._interim_trial_observations_config
                    )
                    ns._distributional_ptrs_config = state._distributional_ptrs_config
                    ns._rd_capacity_config = state._rd_capacity_config
                    ns._drop_action_config = state._drop_action_config
                    ns._ptrs_readings_config = state._ptrs_readings_config
                    ns._clinical_sites_config = state._clinical_sites_config
                    ns._ta_quality_modifiers = state._ta_quality_modifiers.copy()
                    ns._marketing_config = state._marketing_config
                    ns._brand_scores = dict(state._brand_scores)
                    ns._brand_score_floors = dict(state._brand_score_floors)
                    # Preserve clones for other BD slots; drop the won asset's clone
                    # since the asset is now in the portfolio.
                    # (All other clones are already for live assets — the end-of-step
                    # prune after spawn_bd_asset() guarantees no stale entries survive
                    # into Phase 0 of the next step.)
                    ns._bd_asset_clones = {
                        k: v
                        for k, v in state._bd_asset_clones.items()
                        if k != str(bd_asset.id)
                    }

        # Phase 0b: Resolve the clinical-site auction (PvP). One immediately-
        # usable site is offered on the auction cadence; highest cash bid wins
        # and pays its own bid. No affordability mask (overbid -> bankruptcy),
        # mirroring the BD auction.
        if site_bids and new_shared_market.site_auction_available():
            live_bids = {
                agent_id: bid
                for agent_id, bid in site_bids.items()
                if agent_id in new_agent_states
                and not new_agent_states[agent_id].game_ended
            }
            winner, price = resolve_site_bid(live_bids, get_game_rng())
            if winner is not None:
                new_agent_states[winner] = new_agent_states[
                    winner
                ].with_auction_site_win(price)
                new_shared_market.register_site_deal(winner, price)
                logger.debug(
                    f"{winner} won a clinical site at auction for ${price:,.0f}"
                )

        # Phase 1-3: Step each agent's GameState
        # Detect phase transitions for event-driven leaks
        _PHASE_INDEX = {
            TrialPhase.PHASE_1: 0,
            TrialPhase.PHASE_2: 1,
            TrialPhase.PHASE_3: 2,
        }
        if hasattr(TrialPhase, "APPROVAL"):
            _PHASE_INDEX[TrialPhase.APPROVAL] = 3

        all_market_shares: dict[str, dict] = {}

        # Merge all agents' pricing multipliers into a single dict for market share calc
        all_pricing_multipliers: dict[uuid.UUID, float] | None = None
        if pricing_actions is not None:
            all_pricing_multipliers = {}
            for agent_pricing in pricing_actions.values():
                if agent_pricing:
                    all_pricing_multipliers.update(agent_pricing)

        step_num = self.time + 1
        active = [a for a in self.active_agents if not new_agent_states[a].game_ended]
        active_labels = [self._label(a) for a in active]
        logger.info(
            f"Step {step_num}/{self.horizon} — active agents: "
            f"{', '.join(active_labels)}"
        )

        # Aggregate marketing state for this step
        marketing_config = None
        all_brand_scores: dict[uuid.UUID, float] = {}
        all_brand_floors: dict[uuid.UUID, float] = {}
        pre_step_demand_multipliers: dict[str, float] = {}
        for agent_state in new_agent_states.values():
            mc = agent_state._marketing_config
            if mc is not None and mc.enabled:
                marketing_config = mc
                break
        indication_peak_revenues: dict[str, float] = {}
        if marketing_config is not None:
            for agent_state in new_agent_states.values():
                for aid, score in agent_state._brand_scores.items():
                    floor = agent_state._brand_score_floors.get(aid, 0.0)
                    # Pass the raw contribution and the floor separately; the
                    # underdog (1 - floor) weighting is applied in market_mechanics.
                    all_brand_scores[aid] = max(0.0, score - floor)
                    all_brand_floors[aid] = floor
            for key, ind_market in new_shared_market.indication_markets.items():
                pre_step_demand_multipliers[key] = ind_market.demand_multiplier
            for agent_state in new_agent_states.values():
                for asset in agent_state.assets.values():
                    if asset.state == AssetState.OnMarket:
                        k = _ind_key(asset.therapeutic_area, asset.indication)
                        if asset.raw_max_revenue > indication_peak_revenues.get(k, 0.0):
                            indication_peak_revenues[k] = asset.raw_max_revenue

        for agent in self.active_agents:
            if new_agent_states[agent].game_ended:
                continue

            # Snapshot pre-step trial phases for leak detection
            pre_phases: dict[uuid.UUID, TrialPhase | None] = {}
            pre_states: dict[uuid.UUID, AssetState] = {}
            for asset_id, asset in new_agent_states[agent].assets.items():
                pre_states[asset_id] = asset.state
                pre_phases[asset_id] = asset.trial.phase if asset.trial else None

            # Calculate market shares for this agent
            market_shares = None
            if not self.disable_market_share_competition:
                market_shares = calculate_agent_market_shares(
                    agent,
                    new_shared_market,
                    new_agent_states,
                    self.time,
                    all_pricing_multipliers=all_pricing_multipliers,
                    pricing_elasticity=self.pricing_elasticity,
                    brand_scores=all_brand_scores
                    if marketing_config is not None
                    else None,
                    brand_floors=all_brand_floors
                    if marketing_config is not None
                    else None,
                    marketing_config=marketing_config,
                )
                all_market_shares[agent] = market_shares or {}

            # Get this agent's actions and pricing multipliers
            actions = investor_actions.get(agent, {})
            agent_pricing = None
            if pricing_actions is not None:
                agent_pricing = pricing_actions.get(agent)

            # Build demand_multipliers for this agent's on-market drugs from the
            # pre-step snapshot
            agent_demand_multipliers: dict[uuid.UUID, float] | None = None
            agent_brand_equity: dict[uuid.UUID, int] | None = None
            agent_demand_creation: dict[str, int] | None = None
            if marketing_config is not None:
                agent_demand_multipliers = {}
                for asset_id, asset in new_agent_states[agent].assets.items():
                    if asset.state == AssetState.OnMarket:
                        k = _ind_key(asset.therapeutic_area, asset.indication)
                        raw_mult = pre_step_demand_multipliers.get(k, 1.0)
                        peak_rev = indication_peak_revenues.get(
                            k, asset.raw_max_revenue
                        )
                        scale = (
                            peak_rev / asset.raw_max_revenue
                            if asset.raw_max_revenue > 0
                            else 1.0
                        )
                        agent_demand_multipliers[asset_id] = (
                            1.0 + (raw_mult - 1.0) * scale
                        )
            if marketing_actions is not None:
                agent_mktg = marketing_actions.get(agent, {})
                agent_brand_equity = agent_mktg.get("brand_equity")
                agent_demand_creation = agent_mktg.get("demand_creation")

                # Marketing spend leaks: brand-equity or demand-creation spend
                # reveals to the opponent (probabilistically) which indication an
                # agent is investing in. Leaks are indication-level: one alert per
                # indication per step. A DC action always sizes the market (the
                # shared demand boost is applied for every action==1, regardless
                # of whether the agent yet holds a drug there — see the
                # demand-creation loop below), so the leak fires on every DC
                # action. BE is keyed by asset UUID, so we resolve each asset to
                # its indication and count how many BE spends land in it; the
                # generator rolls per spend and reports the leaked count as a
                # dominance signal (be_count), never per-asset attribution.
                #
                # DC leaks only carry information when attribution is non-trivial.
                # The demand multiplier a DC action boosts is a public
                # per-indication value already exposed in the observation, so with
                # few agents an opponent can attribute any rise to the only other
                # spender for free and the leak is redundant. It therefore
                # activates only from ``dc_leak_min_agents`` upward, so future
                # multi-agent (>2) iterations get DC leaks automatically. BE has
                # no public analogue and always leaks.
                pre_step_assets = new_agent_states[agent].assets
                if agent_brand_equity:
                    be_spend_counts: dict[tuple[str, int], int] = {}
                    for asset_id, be_action in agent_brand_equity.items():
                        if be_action != 1:
                            continue
                        asset = pre_step_assets.get(asset_id)
                        if asset is None:
                            continue
                        ind = (asset.therapeutic_area, asset.indication)
                        be_spend_counts[ind] = be_spend_counts.get(ind, 0) + 1
                    for (ta, ind_idx), count in be_spend_counts.items():
                        new_shared_market.generate_be_spend_leak(
                            agent, ta, ind_idx, count
                        )
                if agent_demand_creation and self.num_agents >= self.dc_leak_min_agents:
                    for ind_key_str, dc_action in agent_demand_creation.items():
                        if dc_action != 1:
                            continue
                        ta, _, ind_str = ind_key_str.partition(":")
                        new_shared_market.generate_dc_spend_leak(
                            agent, ta, int(ind_str)
                        )

            # Step the GameState
            agent_research = (
                research_actions.get(agent) if research_actions is not None else None
            )
            agent_buy_site = (
                bool(buy_site_actions.get(agent))
                if buy_site_actions is not None
                else False
            )
            agent_site_priorities = (
                site_priorities.get(agent) if site_priorities is not None else None
            )
            new_state = new_agent_states[agent].step(
                actions,
                market_shares=market_shares,
                pricing_multipliers=agent_pricing,
                research_actions=agent_research,
                demand_multipliers=agent_demand_multipliers,
                brand_equity_actions=agent_brand_equity,
                demand_creation_actions=agent_demand_creation,
                demand_creation_cost_base=self._be_static_peak_revenue,
                bd_current_assets=new_shared_market.current_bd_assets,
                site_priorities=agent_site_priorities,
                buy_site=agent_buy_site,
            )
            new_agent_states[agent] = new_state

            if new_state.game_ended and not self.agent_states[agent].game_ended:
                if new_state.bankrupt:
                    logger.info(
                        f"  {self._label(agent)} bankrupt at step {step_num}: "
                        f"{new_state.ended_reason}"
                    )

            # Detect phase transitions and generate leaks
            for asset_id, new_asset in new_state.assets.items():
                old_phase = pre_phases.get(asset_id)
                old_state = pre_states.get(asset_id)
                if old_phase is None or old_state != AssetState.InDevelopment:
                    continue

                new_phase = new_asset.trial.phase if new_asset.trial else None
                new_asset_state = new_asset.state

                # Detect phase advancement
                if new_phase is not None and old_phase != new_phase:
                    # Phase changed while still in development
                    old_idx = _PHASE_INDEX.get(old_phase)
                    if old_idx is not None:
                        new_shared_market.generate_phase_transition_leak(
                            agent, new_asset, old_idx
                        )
                elif (
                    new_asset_state == AssetState.OnMarket
                    and old_state == AssetState.InDevelopment
                ):
                    # Asset went directly to market (final phase passed)
                    # This is detected by DRUG_RELEASE alert, no leak needed
                    pass

            # Register drug releases and remove expired drugs
            old_assets = self.agent_states[agent].assets
            for asset_id, new_asset in new_state.assets.items():
                if new_asset.state == AssetState.OnMarket:
                    old_asset = old_assets.get(asset_id)
                    if old_asset is None or old_asset.state != AssetState.OnMarket:
                        new_shared_market.register_drug_release(agent, new_asset)

            # Check for expired drugs
            for asset_id, old_asset in old_assets.items():
                if asset_id not in new_state.assets:
                    if old_asset.state == AssetState.OnMarket:
                        new_shared_market.remove_expired_drug(agent, asset_id)
                elif asset_id in new_state.expired_assets:
                    if old_asset.state == AssetState.OnMarket:
                        new_shared_market.remove_expired_drug(agent, asset_id)

        # Set permanent BE floor scores for drugs newly entered OnMarket.
        # The floor is proportional to the drug's raw (pre-scaling) max_revenue
        # relative to the global peak across the full static asset pool. Stronger
        # drugs get a higher floor (spending gives diminishing returns); weaker
        # drugs get a lower floor (more headroom, steeper part of the curve —
        # catch-up mechanism). Decay never goes below the floor, so the floor is a
        # permanent property of the drug.
        if marketing_config is not None:
            peak = self._be_static_peak_revenue
            for agent in self.active_agents:
                if new_agent_states[agent].game_ended:
                    continue
                old_assets = self.agent_states[agent].assets
                for asset_id, new_asset in new_agent_states[agent].assets.items():
                    if new_asset.state == AssetState.OnMarket:
                        old_asset = old_assets.get(asset_id)
                        if old_asset is None or old_asset.state != AssetState.OnMarket:
                            floor = (
                                min(new_asset.raw_max_revenue / peak, 1.0)
                                if peak > 0
                                else 0.0
                            )
                            new_agent_states[agent]._brand_score_floors[asset_id] = (
                                floor
                            )
                            new_agent_states[agent]._brand_scores[asset_id] = floor

        # Process demand creation actions using pre-step multiplier snapshot
        if marketing_config is not None and marketing_actions is not None:
            boosts: dict[str, float] = {}
            for agent in self.active_agents:
                if new_agent_states[agent].game_ended:
                    continue
                agent_mktg = marketing_actions.get(agent, {})
                dc_actions = agent_mktg.get("demand_creation") or {}
                for ind_key_str, action in dc_actions.items():
                    if action == 1:
                        boosts[ind_key_str] = (
                            boosts.get(ind_key_str, 0.0)
                            + marketing_config.dc_step_boost
                        )
            for ind_key_str, total_boost in boosts.items():
                new_shared_market.apply_demand_creation(
                    ind_key_str, total_boost, marketing_config
                )

        # Advance shared market time
        new_shared_market.advance_time(marketing_config=marketing_config)

        # Spawn BD asset for next step (uses recent leaks to boost λ)
        agent_portfolios = {
            agent: new_agent_states[agent].assets for agent in self.active_agents
        }
        new_shared_market.spawn_bd_asset(
            agent_portfolios, max_slots=self.bd_max_slots, won_asset_ids=won_asset_ids
        )

        # Prune BD asset clones for assets that have left the market
        live_bd_ids = {str(a.id) for a in new_shared_market.current_bd_assets}
        for agent in self.active_agents:
            gs = new_agent_states[agent]
            gs._bd_asset_clones = {
                k: v for k, v in gs._bd_asset_clones.items() if k in live_bd_ids
            }

        # Recompute market shares post-step so the cache reflects the
        # current on-market drugs (pre-step shares, used for revenue, may
        # include drugs that expired during the step).
        post_step_shares: dict[str, dict] = {}
        if not self.disable_market_share_competition:
            # Aggregate updated brand scores for post-step share calc
            post_step_brand_scores: dict[uuid.UUID, float] = {}
            post_step_brand_floors: dict[uuid.UUID, float] = {}
            if marketing_config is not None:
                for agent_state in new_agent_states.values():
                    for aid, score in agent_state._brand_scores.items():
                        floor = agent_state._brand_score_floors.get(aid, 0.0)
                        # Raw contribution + floor; the (1 - floor) underdog
                        # weighting is applied in market_mechanics (see above).
                        post_step_brand_scores[aid] = max(0.0, score - floor)
                        post_step_brand_floors[aid] = floor
            for agent in self.active_agents:
                if new_agent_states[agent].game_ended:
                    continue
                post_step_shares[agent] = (
                    calculate_agent_market_shares(
                        agent,
                        new_shared_market,
                        new_agent_states,
                        self.time + 1,
                        all_pricing_multipliers=all_pricing_multipliers,
                        pricing_elasticity=self.pricing_elasticity,
                        brand_scores=post_step_brand_scores
                        if marketing_config is not None
                        else None,
                        brand_floors=post_step_brand_floors
                        if marketing_config is not None
                        else None,
                        marketing_config=marketing_config,
                    )
                    or {}
                )

        # Log final summary when game reaches horizon
        if step_num >= self.horizon:
            logger.info("Game complete — final standings:")
            for agent, state in new_agent_states.items():
                status = "BANKRUPT" if state.bankrupt else f"cash={state.cash:,.0f}"
                logger.info(
                    f"  {self._label(agent)}: {status}, eNPV={state.enpv():,.0f}"
                )

        new_game = MultiAgentGame(
            agent_states=new_agent_states,
            shared_market=new_shared_market,
            time=self.time + 1,
            horizon=self.horizon,
            num_agents=self.num_agents,
            dc_leak_min_agents=self.dc_leak_min_agents,
            disable_market_share_competition=self.disable_market_share_competition,
            bd_max_slots=self.bd_max_slots,
            pricing_elasticity=self.pricing_elasticity,
        )
        new_game._cached_market_shares = post_step_shares
        new_game._display_names = self._display_names
        new_game._be_static_peak_revenue = self._be_static_peak_revenue
        return new_game
