"""Metrics for multi-agent competitive environment."""

from __future__ import annotations

import logging
from collections import defaultdict

from pyxis_portfolio_challenge.environment.metrics import (
    MetricsContext,
    PerEpisodeMetric,
    PerStepMetric,
)
from pyxis_portfolio_challenge.game.asset import AssetState
from pyxis_portfolio_challenge.game.game_state import GameState
from pyxis_portfolio_challenge.game.shared_market_state import (
    AlertType,
    indication_key,
)

logger = logging.getLogger(__name__)


def compute_agent_rankings(
    agent_portfolios: dict[str, GameState],
    metric: str = "enpv",
) -> dict[str, int]:
    """
    Compute agent rankings based on a metric.

    Args:
        agent_portfolios: Dict mapping agent_id -> GameState
        metric: Metric to rank by ("enpv", "cash", "eroi", "realised_roi")

    Returns:
        Dict mapping agent_id -> rank (1 = best)

    """
    metric_values = {}
    for agent_id, portfolio in agent_portfolios.items():
        if metric == "enpv":
            metric_values[agent_id] = portfolio.enpv()
        elif metric == "cash":
            metric_values[agent_id] = portfolio.cash
        elif metric == "eroi":
            metric_values[agent_id] = portfolio.eroi()
        elif metric == "realised_roi":
            metric_values[agent_id] = portfolio.realised_roi()
        else:
            raise ValueError(f"Unknown metric: {metric}")

    sorted_agents = sorted(
        metric_values.keys(), key=lambda x: metric_values[x], reverse=True
    )

    return {agent: rank + 1 for rank, agent in enumerate(sorted_agents)}


def compute_market_dominance(
    agent_portfolios: dict[str, GameState],
) -> dict[str, float]:
    """
    Compute market dominance as fraction of total portfolio value.

    Returns:
        Dict mapping agent_id -> dominance fraction (0 to 1)

    """
    enpvs = {aid: max(0, p.enpv()) for aid, p in agent_portfolios.items()}
    total = sum(enpvs.values())
    if total == 0:
        n = len(agent_portfolios)
        return {aid: 1.0 / n for aid in agent_portfolios}
    return {aid: v / total for aid, v in enpvs.items()}


# ---------------------------------------------------------------------------
# Indication metrics
# ---------------------------------------------------------------------------


class PerStepIndicationDiversity(PerStepMetric):
    """Number of distinct indications this agent has assets in (all states)."""

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> int:
        seen = set()
        for asset in context.game_state.assets.values():
            seen.add(indication_key(asset.therapeutic_area, asset.indication))
        return len(seen)

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerStepIndicationConcentration(PerStepMetric):
    """HHI of this agent's assets across indications (0=spread, 1=single)."""

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> float:
        counts: dict[str, int] = defaultdict(int)
        for asset in context.game_state.assets.values():
            key = indication_key(asset.therapeutic_area, asset.indication)
            counts[key] += 1
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return sum((c / total) ** 2 for c in counts.values())

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerStepOnMarketPerIndication(PerStepMetric):
    """
    Number of this agent's on-market drugs per indication.

    Stored as a dict mapping indication key to count at each step.
    """

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[dict[str, int]]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for asset in context.game_state.assets.values():
            if asset.state == AssetState.OnMarket:
                key = indication_key(asset.therapeutic_area, asset.indication)
                counts[key] += 1
        return dict(counts)

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerStepFirstMoverExclusivities(PerStepMetric):
    """
    Number of indications where this agent holds first-mover exclusivity.

    Requires shared_market_state on the MetricsContext.
    """

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[int]] = {}
        self._agent_id: str | None = None

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> int:
        sm = context.shared_market_state
        if sm is None:
            return 0
        # Identify agent by matching game_state id against market data
        agent_id = self._find_agent_id(context)
        if agent_id is None:
            return 0
        count = 0
        for ind_market in sm.indication_markets.values():
            if (
                ind_market.first_mover_agent == agent_id
                and ind_market.is_in_exclusivity(sm.time)
            ):
                count += 1
        return count

    def _find_agent_id(self, context: MetricsContext) -> str | None:
        sm = context.shared_market_state
        if sm is None:
            return None
        # Check indication markets for any agent that has assets matching
        # this game_state's asset ids
        asset_ids = set(context.game_state.assets.keys())
        for ind_market in sm.indication_markets.values():
            for agent_id, drug_ids in ind_market.active_drugs.items():
                if asset_ids & set(drug_ids):
                    return agent_id
        # Fallback: check TA markets
        for ta_market in sm.ta_markets.values():
            for agent_id, drug_ids in ta_market.active_drugs.items():
                if asset_ids & set(drug_ids):
                    return agent_id
        return None

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerStepContestedIndications(PerStepMetric):
    """
    Number of indications with on-market drugs from multiple agents.

    Requires shared_market_state on the MetricsContext.
    """

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> int:
        sm = context.shared_market_state
        if sm is None:
            return 0
        count = 0
        for ind_market in sm.indication_markets.values():
            agents_with_drugs = sum(
                1 for drugs in ind_market.active_drugs.values() if len(drugs) > 0
            )
            if agents_with_drugs > 1:
                count += 1
        return count

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerStepTotalOnMarketPerIndication(PerStepMetric):
    """
    Total on-market drugs across all agents per indication.

    Requires shared_market_state on the MetricsContext.
    """

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[dict[str, int]]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> dict[str, int]:
        sm = context.shared_market_state
        if sm is None:
            return {}
        result = {}
        for key, ind_market in sm.indication_markets.items():
            total = sum(len(drugs) for drugs in ind_market.active_drugs.values())
            if total > 0:
                result[key] = total
        return result

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerStepIndicationSpread(PerStepMetric):
    """Fraction of available indications that the agent occupies."""

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> float:
        sm = context.shared_market_state
        total_indications = len(sm.indication_markets) if sm is not None else 0
        if total_indications == 0:
            return 0.0
        occupied = set()
        for asset in context.game_state.assets.values():
            occupied.add(indication_key(asset.therapeutic_area, asset.indication))
        return len(occupied) / total_indications

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerEpisodeIndicationSpread(PerEpisodeMetric):
    """Fraction of available indications that the agent occupies at episode end."""

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, float] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_end(self, context: MetricsContext) -> None:
        """Record final value at episode end."""
        key = context.episode_key
        sm = context.shared_market_state
        total_indications = len(sm.indication_markets) if sm is not None else 0
        if total_indications == 0:
            self.history[key] = 0.0
            return
        occupied = set()
        for asset in context.game_state.assets.values():
            occupied.add(indication_key(asset.therapeutic_area, asset.indication))
        self.history[key] = len(occupied) / total_indications

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


# ---------------------------------------------------------------------------
# General multi-agent metrics
# ---------------------------------------------------------------------------


class PerStepNonBankruptAgents(PerStepMetric):
    """Number of agents that are not bankrupt."""

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> int:
        if context.all_agent_states is None:
            return 0
        return sum(1 for gs in context.all_agent_states.values() if not gs.bankrupt)

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerStepAgentRank(PerStepMetric):
    """Agent rank by eNPV (1 = best)."""

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> int:
        if context.all_agent_states is None or context.agent_id is None:
            return 0
        rankings = compute_agent_rankings(context.all_agent_states, metric="enpv")
        return rankings.get(context.agent_id, 0)

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerStepRelativeEnpv(PerStepMetric):
    """Agent eNPV as fraction of total across all agents."""

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> float:
        if context.all_agent_states is None or context.agent_id is None:
            return 0.0
        dominance = compute_market_dominance(context.all_agent_states)
        return dominance.get(context.agent_id, 0.0)

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


# ---------------------------------------------------------------------------
# BD deal metrics
# ---------------------------------------------------------------------------


class PerStepBDAssetAvailable(PerStepMetric):
    """Whether a BD asset is available for auction this step (0 or 1)."""

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> int:
        sm = context.shared_market_state
        if sm is None:
            return 0
        return len(sm.current_bd_assets)

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerEpisodeDrugReleases(PerEpisodeMetric):
    """
    Total drugs reaching OnMarket across ALL agents during the episode.

    Counts asset state transitions to OnMarket by tracking previous states
    for every agent each step.
    """

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, int] = {}
        self._count: int = 0
        # Keyed by (agent_id, asset_id) to track all agents
        self._prev_states: dict[tuple[str, str], str] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        self._count = 0
        self._prev_states = {}
        if context.all_agent_states:
            for agent_id, state in context.all_agent_states.items():
                for aid, asset in state.assets.items():
                    self._prev_states[(agent_id, str(aid))] = asset.state

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        if not context.all_agent_states:
            return
        for agent_id, state in context.all_agent_states.items():
            for aid, asset in state.assets.items():
                key = (agent_id, str(aid))
                if asset.state == AssetState.OnMarket:
                    prev = self._prev_states.get(key)
                    if prev != AssetState.OnMarket:
                        self._count += 1
                self._prev_states[key] = asset.state

    def on_episode_end(self, context: MetricsContext) -> None:
        """Record final value at episode end."""
        key = context.episode_key
        self.history[key] = self._count

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerStepDrugsOnMarket(PerStepMetric):
    """Total number of on-market drugs across ALL agents at each step."""

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> int:
        if not context.all_agent_states:
            return 0
        count = 0
        for state in context.all_agent_states.values():
            for asset in state.assets.values():
                if asset.state == AssetState.OnMarket:
                    count += 1
        return count

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerEpisodeBDDealsWon(PerEpisodeMetric):
    """
    Total BD deals won by this agent during the episode.

    Counts BD_DEAL alerts where this agent is the buyer.
    """

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, int] = {}
        self._count: int = 0

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        self._count = 0

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        sm = context.shared_market_state
        agent_id = context.agent_id
        if sm is None or agent_id is None:
            return
        # sm.time has already been advanced by advance_time(), so alerts
        # created this step have step == sm.time - 1
        current_step = sm.time - 1
        for alert in sm.alerts:
            if (
                alert.event_type == AlertType.BD_DEAL
                and alert.agent_id == agent_id
                and alert.step == current_step
            ):
                self._count += 1

    def on_episode_end(self, context: MetricsContext) -> None:
        """Record final value at episode end."""
        key = context.episode_key
        self.history[key] = self._count

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerEpisodeBDBidLevelDistribution(PerEpisodeMetric):
    """
    Distribution of BD bid levels chosen by this agent during the episode.

    Tracks counts per bid level (0=pass, 1-10=bid levels) across all BD slots
    and all steps. Only counts steps where a BD asset was available.
    Reports as a dict mapping level -> count.
    """

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, dict[int, int]] = {}
        self._counts: dict[int, int] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        self._counts = defaultdict(int)

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        if context.bd_bid_levels is None:
            return
        for level in context.bd_bid_levels:
            self._counts[level] += 1

    def on_episode_end(self, context: MetricsContext) -> None:
        """Record final value at episode end."""
        key = context.episode_key
        self.history[key] = dict(self._counts)

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerStepMeanBDBidLevel(PerStepMetric):
    """
    Mean BD bid level per step (across all slots).

    Only counts non-zero bids (passes excluded). Returns 0 if no bids placed.
    """

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = []

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        if context.bd_bid_levels is None:
            self.history[key] = self.history[key] + [0.0]
            return
        non_zero = [lvl for lvl in context.bd_bid_levels if lvl > 0]
        mean_level = sum(non_zero) / len(non_zero) if non_zero else 0.0
        self.history[key] = self.history[key] + [mean_level]

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerStepBDBidRate(PerStepMetric):
    """Fraction of BD slots where a non-zero bid was placed per step."""

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = []

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        if context.bd_bid_levels is None:
            self.history[key] = self.history[key] + [0.0]
            return
        n_total = len(context.bd_bid_levels)
        n_bids = sum(1 for lvl in context.bd_bid_levels if lvl > 0)
        rate = n_bids / n_total if n_total > 0 else 0.0
        self.history[key] = self.history[key] + [rate]

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerEpisodePhaseTransitionLeaks(PerEpisodeMetric):
    """Total pipeline leak alerts generated during the episode (event-driven)."""

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, int] = {}
        self._count: int = 0
        self._prev_alert_count: int = 0

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        self._count = 0
        sm = context.shared_market_state
        self._prev_alert_count = (
            sum(1 for a in sm.alerts if a.event_type == AlertType.PIPELINE_LEAK)
            if sm is not None
            else 0
        )

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        sm = context.shared_market_state
        if sm is None:
            return
        current = sum(1 for a in sm.alerts if a.event_type == AlertType.PIPELINE_LEAK)
        self._count += current - self._prev_alert_count
        self._prev_alert_count = current

    def on_episode_end(self, context: MetricsContext) -> None:
        """Record final value at episode end."""
        key = context.episode_key
        self.history[key] = self._count

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


# ---------------------------------------------------------------------------
# Market share metrics
# ---------------------------------------------------------------------------


class PerStepMeanMarketShare(PerStepMetric):
    """
    Mean per-drug market share across this agent's on-market drugs.

    Uses the actual per-drug share computation from market_mechanics,
    which accounts for quality-weighted competition and first mover bonus.
    """

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> float:
        from pyxis_portfolio_challenge.environment.market_mechanics import (
            calculate_agent_market_shares,
        )

        sm = context.shared_market_state
        agent_id = context.agent_id
        if sm is None or agent_id is None:
            return 0.0
        if context.all_agent_states is None:
            return 0.0

        per_drug_shares = calculate_agent_market_shares(
            agent_id,
            sm,
            context.all_agent_states,
            sm.time,
        )
        if not per_drug_shares:
            return 0.0
        return sum(per_drug_shares.values()) / len(per_drug_shares)

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerEpisodeMarketShareDistribution(PerEpisodeMetric):
    """
    Distribution of per-drug market shares across an episode.

    Collects every on-market drug's share at every step and reports:
    - mean, median share
    - fraction of drug-steps at each share bucket (100%, >70%, 50-70%, 30-50%, <30%)
    - total drug-step observations
    """

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, dict] = {}
        self._shares: list[float] = []

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        self._shares = []

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        from pyxis_portfolio_challenge.environment.market_mechanics import (
            calculate_agent_market_shares,
        )

        sm = context.shared_market_state
        agent_id = context.agent_id
        if sm is None or agent_id is None or context.all_agent_states is None:
            return

        per_drug_shares = calculate_agent_market_shares(
            agent_id,
            sm,
            context.all_agent_states,
            sm.time,
        )
        self._shares.extend(per_drug_shares.values())

    def on_episode_end(self, context: MetricsContext) -> None:
        """Record final value at episode end."""
        key = context.episode_key
        n = len(self._shares)
        if n == 0:
            self.history[key] = {
                "n_observations": 0,
                "mean": 0.0,
                "median": 0.0,
                "pct_uncontested": 0.0,
                "pct_above_70": 0.0,
                "pct_50_to_70": 0.0,
                "pct_30_to_50": 0.0,
                "pct_below_30": 0.0,
            }
            return

        import numpy as np

        shares = np.array(self._shares)
        self.history[key] = {
            "n_observations": n,
            "mean": float(np.mean(shares)),
            "median": float(np.median(shares)),
            "pct_uncontested": float((shares >= 0.99).mean()),
            "pct_above_70": float((shares > 0.7).mean()),
            "pct_50_to_70": float(((shares >= 0.5) & (shares <= 0.7)).mean()),
            "pct_30_to_50": float(((shares >= 0.3) & (shares < 0.5)).mean()),
            "pct_below_30": float((shares < 0.3).mean()),
        }

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerStepMarketShareDistribution(PerStepMetric):
    """
    Per-step snapshot of market share distribution across on-market drugs.

    At each step, reports: mean share, fraction uncontested, fraction below 30%.
    """

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[dict]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> dict:
        from pyxis_portfolio_challenge.environment.market_mechanics import (
            calculate_agent_market_shares,
        )

        sm = context.shared_market_state
        agent_id = context.agent_id
        if sm is None or agent_id is None or context.all_agent_states is None:
            return {"n": 0, "mean": 0.0, "pct_uncontested": 0.0, "pct_below_30": 0.0}

        per_drug_shares = calculate_agent_market_shares(
            agent_id,
            sm,
            context.all_agent_states,
            sm.time,
        )
        if not per_drug_shares:
            return {"n": 0, "mean": 0.0, "pct_uncontested": 0.0, "pct_below_30": 0.0}

        shares = list(per_drug_shares.values())
        n = len(shares)
        return {
            "n": n,
            "mean": sum(shares) / n,
            "pct_uncontested": sum(1 for s in shares if s >= 0.99) / n,
            "pct_below_30": sum(1 for s in shares if s < 0.3) / n,
        }

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerStepIndicationsWithExclusivity(PerStepMetric):
    """Total number of indications currently under any agent's exclusivity."""

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> int:
        sm = context.shared_market_state
        if sm is None:
            return 0
        return sum(
            1
            for ind_market in sm.indication_markets.values()
            if ind_market.is_in_exclusivity(sm.time)
        )

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerStepEmptyIndications(PerStepMetric):
    """Number of indications with 0 on-market drugs from any agent."""

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> int:
        sm = context.shared_market_state
        if sm is None:
            return 0
        count = 0
        for ind_market in sm.indication_markets.values():
            total = sum(len(drugs) for drugs in ind_market.active_drugs.values())
            if total == 0:
                count += 1
        return count

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerStepNovelIndications(PerStepMetric):
    """
    Number of indications that have never had a drug on market.

    An indication is considered "explored" once any agent's drug reaches
    the market there (first_mover_agent is set). Novel = total - explored.
    """

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> int:
        sm = context.shared_market_state
        if sm is None:
            return 0
        count = 0
        for ind_market in sm.indication_markets.values():
            if ind_market.first_mover_agent is None:
                count += 1
        return count

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerStepStaleIndications(PerStepMetric):
    """
    Number of indications that are empty but previously had a drug on market.

    Stale = empty and first_mover_agent is set (indication was explored but
    all drugs have since expired). These indications offer no first-mover
    opportunity to new entrants.
    """

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> int:
        sm = context.shared_market_state
        if sm is None:
            return 0
        count = 0
        for ind_market in sm.indication_markets.values():
            total = sum(len(drugs) for drugs in ind_market.active_drugs.values())
            if total == 0 and ind_market.first_mover_agent is not None:
                count += 1
        return count

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


# ---------------------------------------------------------------------------
# Alert metrics
# ---------------------------------------------------------------------------


class PerStepAlertCount(PerStepMetric):
    """Total number of alerts in the shared market history."""

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> int:
        sm = context.shared_market_state
        if sm is None:
            return 0
        return len(sm.alerts)

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerStepDrugReleaseAlerts(PerStepMetric):
    """Number of drug release alerts in history."""

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> int:
        sm = context.shared_market_state
        if sm is None:
            return 0
        return sum(1 for a in sm.alerts if a.event_type == AlertType.DRUG_RELEASE)

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerStepBDDealAlerts(PerStepMetric):
    """Number of BD deal alerts in history."""

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> int:
        sm = context.shared_market_state
        if sm is None:
            return 0
        return sum(1 for a in sm.alerts if a.event_type == AlertType.BD_DEAL)

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerStepPipelineLeakAlerts(PerStepMetric):
    """Number of pipeline leak alerts in history."""

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        key = context.episode_key
        self.history[key] = [self._compute(context)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        key = context.episode_key
        self.history[key] = self.history[key] + [self._compute(context)]

    def _compute(self, context: MetricsContext) -> int:
        sm = context.shared_market_state
        if sm is None:
            return 0
        return sum(1 for a in sm.alerts if a.event_type == AlertType.PIPELINE_LEAK)

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


# ---------------------------------------------------------------------------
# Strategic timing metrics — detect leak-informed investment behavior
# ---------------------------------------------------------------------------


class PerEpisodeExclusivityCollisionRate(PerEpisodeMetric):
    """
    Fraction of drug releases landing where a competitor holds exclusivity.

    Lower = better timing. An agent using leak intel to avoid competitor
    exclusivity windows should have a lower collision rate than a naive agent.
    """

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, float] = {}
        self._releases: int = 0
        self._collisions: int = 0
        self._prev_on_market: set[str] = set()

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        self._releases = 0
        self._collisions = 0
        self._prev_on_market = set()
        for asset in context.game_state.assets.values():
            if asset.state == AssetState.OnMarket:
                self._prev_on_market.add(str(asset.id))

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        sm = context.shared_market_state
        agent_id = context.agent_id
        if sm is None or agent_id is None:
            return

        for asset in context.game_state.assets.values():
            aid = str(asset.id)
            if asset.state == AssetState.OnMarket and aid not in self._prev_on_market:
                # New release this step
                self._releases += 1
                key = indication_key(asset.therapeutic_area, asset.indication)
                ind_market = sm.indication_markets.get(key)
                if ind_market is not None:
                    if (
                        ind_market.is_in_exclusivity(sm.time)
                        and ind_market.first_mover_agent != agent_id
                    ):
                        self._collisions += 1

        # Update tracked set
        self._prev_on_market = set()
        for asset in context.game_state.assets.values():
            if asset.state == AssetState.OnMarket:
                self._prev_on_market.add(str(asset.id))

    def on_episode_end(self, context: MetricsContext) -> None:
        """Record final value at episode end."""
        key = context.episode_key
        self.history[key] = (
            self._collisions / self._releases if self._releases > 0 else 0.0
        )

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerEpisodeFirstMoverRate(PerEpisodeMetric):
    """
    Fraction of this agent's drug releases that achieve first-mover status.

    Higher = agent is targeting empty/novel indications or racing to beat
    competitors. Indicates proactive use of market intelligence.
    """

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, float] = {}
        self._releases: int = 0
        self._first_movers: int = 0
        self._prev_on_market: set[str] = set()

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        self._releases = 0
        self._first_movers = 0
        self._prev_on_market = set()
        for asset in context.game_state.assets.values():
            if asset.state == AssetState.OnMarket:
                self._prev_on_market.add(str(asset.id))

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        sm = context.shared_market_state
        agent_id = context.agent_id
        if sm is None or agent_id is None:
            return

        for asset in context.game_state.assets.values():
            aid = str(asset.id)
            if asset.state == AssetState.OnMarket and aid not in self._prev_on_market:
                self._releases += 1
                key = indication_key(asset.therapeutic_area, asset.indication)
                ind_market = sm.indication_markets.get(key)
                if ind_market is not None:
                    if ind_market.first_mover_drug_id == asset.id:
                        self._first_movers += 1

        self._prev_on_market = set()
        for asset in context.game_state.assets.values():
            if asset.state == AssetState.OnMarket:
                self._prev_on_market.add(str(asset.id))

    def on_episode_end(self, context: MetricsContext) -> None:
        """Record final value at episode end."""
        key = context.episode_key
        self.history[key] = (
            self._first_movers / self._releases if self._releases > 0 else 0.0
        )

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerEpisodeRevenueLostToCompetition(PerEpisodeMetric):
    """
    Cumulative revenue lost due to market share < 1.0 across all on-market drugs.

    Computed as sum of (1 - share) * revenue_this_step for each on-market drug
    each step. Lower = agent avoids contested indications or dominates them.
    """

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, float] = {}
        self._lost: float = 0.0

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        self._lost = 0.0

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        from pyxis_portfolio_challenge.environment.market_mechanics import (
            calculate_agent_market_shares,
        )

        sm = context.shared_market_state
        agent_id = context.agent_id
        if sm is None or agent_id is None or context.all_agent_states is None:
            return

        per_drug_shares = calculate_agent_market_shares(
            agent_id,
            sm,
            context.all_agent_states,
            sm.time,
        )

        for asset in context.game_state.assets.values():
            if asset.state == AssetState.OnMarket:
                share = per_drug_shares.get(asset.id, 1.0)
                if share < 1.0:
                    # revenue_this_step is the FULL pre-share revenue (property)
                    self._lost += asset.revenue_this_step * (1.0 - share)

    def on_episode_end(self, context: MetricsContext) -> None:
        """Record final value at episode end."""
        key = context.episode_key
        self.history[key] = self._lost

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerEpisodeLeakInformedBDDeals(PerEpisodeMetric):
    """
    Fraction of BD deals won in indications with recent competitor leaks.

    Higher = agent is using leak intel to target BD acquisitions. BD assets
    spawn more frequently in leaked indications, so a baseline correlation
    exists — but a strategic agent should have a higher rate than baseline.
    """

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, float] = {}
        self._total_deals: int = 0
        self._leak_correlated_deals: int = 0

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        self._total_deals = 0
        self._leak_correlated_deals = 0

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        sm = context.shared_market_state
        agent_id = context.agent_id
        if sm is None or agent_id is None:
            return

        # Check for BD_DEAL alerts created this step where this agent won
        current_step = sm.time - 1  # advance_time() already called
        for alert in sm.alerts:
            if (
                alert.event_type != AlertType.BD_DEAL
                or alert.step != current_step
                or alert.agent_id != agent_id
            ):
                continue

            # This agent won a BD deal this step
            self._total_deals += 1
            deal_ind = indication_key(alert.therapeutic_area, alert.indication)

            # Check if there was a recent leak in the same indication
            alert_window = sm.alert_history_length
            for leak_alert in sm.alerts:
                if leak_alert.event_type != AlertType.PIPELINE_LEAK:
                    continue
                if leak_alert.agent_id == agent_id:
                    continue  # Own leaks aren't intel
                age = current_step - leak_alert.step
                if age < 0 or age > alert_window:
                    continue
                leak_ind = indication_key(
                    leak_alert.therapeutic_area, leak_alert.indication
                )
                if leak_ind == deal_ind:
                    self._leak_correlated_deals += 1
                    break

    def on_episode_end(self, context: MetricsContext) -> None:
        """Record final value at episode end."""
        key = context.episode_key
        self.history[key] = (
            self._leak_correlated_deals / self._total_deals
            if self._total_deals > 0
            else 0.0
        )

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerEpisodeInvestmentPnL(PerEpisodeMetric):
    """
    Track per-drug profit/loss for all invested drugs during an episode.

    For each drug that transitions from Idle to InDevelopment, accumulates:
    - total cost (cost_this_step each step while InDevelopment)
    - total revenue received as cash (revenue_this_step * market_share * reinv_pct)

    When the drug disappears (trial failure, replaced) or the episode ends,
    records it as profitable or unprofitable.

    Reports:
    - n_invested: total drugs that received investment
    - n_profitable: drugs where revenue >= cost
    - n_unprofitable: drugs where revenue < cost
    - pct_unprofitable: fraction of invested drugs that lost money
    - n_failed: drugs that failed trials (never reached market)
    - n_reached_market: drugs that reached OnMarket
    - n_still_in_progress: drugs still InDevelopment at episode end
    - mean_pnl: average profit/loss per invested drug
    - mean_pnl_market: average P&L for drugs that reached market
    - mean_pnl_failed: average P&L for drugs that failed
    """

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, dict] = {}
        self._drug_ledger: dict[str, dict] = {}
        self._prev_assets: dict[str, str] = {}
        self._reinv_pct: float = 0.15
        self._agent_id: str | None = None

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        self._drug_ledger = {}
        self._prev_assets = {}
        self._agent_id = context.agent_id
        gs = context.game_state
        self._reinv_pct = gs.reinvestment_percentage
        for aid, asset in gs.assets.items():
            self._prev_assets[str(aid)] = asset.state

    def _get_bd_price(self, context: MetricsContext, asset_id_str: str) -> float:
        """Try to find the BD acquisition price from alerts."""
        sm = context.shared_market_state
        agent_id = context.agent_id
        if sm is None or agent_id is None:
            return 0.0
        current_step = sm.time - 1
        for alert in sm.alerts:
            if (
                alert.event_type == AlertType.BD_DEAL
                and alert.step == current_step
                and alert.agent_id == agent_id
            ):
                return alert.details.get("price", 0.0)
        return 0.0

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        from pyxis_portfolio_challenge.environment.market_mechanics import (
            calculate_agent_market_shares,
        )

        gs = context.game_state
        sm = context.shared_market_state
        agent_id = context.agent_id

        per_drug_shares = {}
        if (
            sm is not None
            and agent_id is not None
            and context.all_agent_states is not None
        ):
            per_drug_shares = calculate_agent_market_shares(
                agent_id,
                sm,
                context.all_agent_states,
                sm.time,
            )

        current_asset_ids = set()
        for aid, asset in gs.assets.items():
            aid_str = str(aid)
            current_asset_ids.add(aid_str)
            prev_state = self._prev_assets.get(aid_str)

            # Detect new investment: was Idle, now InDevelopment
            if (
                prev_state == AssetState.Idle
                and asset.state == AssetState.InDevelopment
            ):
                is_bd = asset.name.startswith("BD-")
                bd_cost = 0.0
                if is_bd and aid_str not in self._drug_ledger:
                    bd_cost = self._get_bd_price(context, aid_str)
                self._drug_ledger[aid_str] = {
                    "cost": bd_cost,
                    "revenue": 0.0,
                    "reached_market": False,
                    "failed": False,
                    "is_bd": is_bd,
                }

            if aid_str in self._drug_ledger:
                if asset.state == AssetState.InDevelopment:
                    self._drug_ledger[aid_str]["cost"] += asset.cost_this_step
                elif asset.state == AssetState.OnMarket:
                    self._drug_ledger[aid_str]["reached_market"] = True
                    share = per_drug_shares.get(aid, 1.0)
                    cash_rev = asset.revenue_this_step * share * self._reinv_pct
                    self._drug_ledger[aid_str]["revenue"] += cash_rev

            self._prev_assets[aid_str] = asset.state

        for aid_str in list(self._prev_assets.keys()):
            if aid_str not in current_asset_ids:
                if aid_str in self._drug_ledger:
                    if not self._drug_ledger[aid_str]["reached_market"]:
                        self._drug_ledger[aid_str]["failed"] = True
                del self._prev_assets[aid_str]

    def on_episode_end(self, context: MetricsContext) -> None:
        """Record final value at episode end."""
        key = context.episode_key

        n_invested = len(self._drug_ledger)
        if n_invested == 0:
            self.history[key] = {
                "n_invested": 0,
                "n_profitable": 0,
                "n_unprofitable": 0,
                "pct_unprofitable": 0.0,
                "n_failed": 0,
                "n_reached_market": 0,
                "n_still_in_progress": 0,
                "mean_pnl": 0.0,
                "mean_pnl_market": 0.0,
                "mean_pnl_failed": 0.0,
                "revenue_to_cost_ratio_all": 0.0,
                "revenue_to_cost_ratio_market": 0.0,
            }
            return

        pnls = []
        pnls_market = []
        pnls_failed = []
        n_profitable = 0
        n_unprofitable = 0
        n_failed = 0
        n_reached_market = 0
        n_still_in_progress = 0
        n_market_profitable = 0
        n_market_unprofitable = 0
        n_bd = 0
        n_bd_reached_market = 0
        n_bd_market_profitable = 0
        n_bd_market_unprofitable = 0
        total_revenue_all = 0.0
        total_cost_all = 0.0
        total_revenue_market = 0.0
        total_cost_market = 0.0

        for drug in self._drug_ledger.values():
            pnl = drug["revenue"] - drug["cost"]
            pnls.append(pnl)
            is_bd = drug["is_bd"]
            total_revenue_all += drug["revenue"]
            total_cost_all += drug["cost"]

            if is_bd:
                n_bd += 1

            if drug["failed"]:
                n_failed += 1
                pnls_failed.append(pnl)
            elif drug["reached_market"]:
                n_reached_market += 1
                pnls_market.append(pnl)
                total_revenue_market += drug["revenue"]
                total_cost_market += drug["cost"]
                if pnl >= 0:
                    n_market_profitable += 1
                else:
                    n_market_unprofitable += 1
                if is_bd:
                    n_bd_reached_market += 1
                    if pnl >= 0:
                        n_bd_market_profitable += 1
                    else:
                        n_bd_market_unprofitable += 1
            else:
                n_still_in_progress += 1

            if pnl >= 0:
                n_profitable += 1
            else:
                n_unprofitable += 1

        import numpy as np

        self.history[key] = {
            "n_invested": n_invested,
            "n_profitable": n_profitable,
            "n_unprofitable": n_unprofitable,
            "pct_unprofitable": n_unprofitable / n_invested,
            "n_failed": n_failed,
            "n_reached_market": n_reached_market,
            "n_still_in_progress": n_still_in_progress,
            "mean_pnl": float(np.mean(pnls)),
            "mean_pnl_market": float(np.mean(pnls_market)) if pnls_market else 0.0,
            "mean_pnl_failed": float(np.mean(pnls_failed)) if pnls_failed else 0.0,
            "n_market_profitable": n_market_profitable,
            "n_market_unprofitable": n_market_unprofitable,
            "pct_market_unprofitable": (
                n_market_unprofitable / n_reached_market
                if n_reached_market > 0
                else 0.0
            ),
            "n_bd": n_bd,
            "n_bd_reached_market": n_bd_reached_market,
            "n_bd_market_profitable": n_bd_market_profitable,
            "n_bd_market_unprofitable": n_bd_market_unprofitable,
            "pct_bd_market_unprofitable": (
                n_bd_market_unprofitable / n_bd_reached_market
                if n_bd_reached_market > 0
                else 0.0
            ),
            "revenue_to_cost_ratio_all": (
                total_revenue_all / total_cost_all if total_cost_all > 0 else 0.0
            ),
            "revenue_to_cost_ratio_market": (
                total_revenue_market / total_cost_market
                if total_cost_market > 0
                else 0.0
            ),
        }

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerEpisodeWinLoss(PerEpisodeMetric):
    """
    Win/loss outcome per episode with bankruptcy-aware conditions.

    Records 1.0 for a win, 0.0 for a loss, 0.5 for a draw.

    Win conditions:
    - Bankrupt agents automatically lose.
    - If both agents go bankrupt, the one that survived longer wins
      (higher game_state.time). Same step → draw.
    - If neither is bankrupt, the agent with higher NCF wins.
    """

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, float] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_end(self, context: MetricsContext) -> None:
        """Record win/loss/draw at episode end."""
        key = context.episode_key
        if (
            context.all_agent_rewards is None
            or context.all_agent_states is None
            or context.agent_id is None
            or len(context.all_agent_rewards) < 2
        ):
            return

        my_state = context.all_agent_states[context.agent_id]
        my_bankrupt = my_state.bankrupt
        my_reward = context.all_agent_rewards[context.agent_id]

        opponent_ids = [
            aid for aid in context.all_agent_rewards if aid != context.agent_id
        ]
        # For 2-player: single opponent
        opp_id = opponent_ids[0]
        opp_state = context.all_agent_states[opp_id]
        opp_bankrupt = opp_state.bankrupt
        opp_reward = context.all_agent_rewards[opp_id]

        if my_bankrupt and not opp_bankrupt:
            self.history[key] = 0.0
        elif not my_bankrupt and opp_bankrupt:
            self.history[key] = 1.0
        elif my_bankrupt and opp_bankrupt:
            # Both bankrupt: whoever survived longer wins
            if my_state.time > opp_state.time:
                self.history[key] = 1.0
            elif my_state.time < opp_state.time:
                self.history[key] = 0.0
            else:
                self.history[key] = 0.5
        else:
            # Neither bankrupt: compare NCF
            if my_reward > opp_reward:
                self.history[key] = 1.0
            elif my_reward < opp_reward:
                self.history[key] = 0.0
            else:
                self.history[key] = 0.5

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}


class PerEpisodeAssetLifecycle(PerEpisodeMetric):
    """Track per-asset idle time before first investment and inter-phase pauses."""

    def __init__(self) -> None:
        """Initialize metric."""
        self.history: dict[str, dict] = {}
        self._tracker: dict[str, dict] = {}
        self._prev_states: dict[str, str] = {}
        self._step: int = 0

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Reset state for new evaluation."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Record initial value at episode start."""
        self._tracker.clear()
        self._prev_states.clear()
        self._step = 0
        for aid, asset in context.game_state.assets.items():
            aid_str = str(aid)
            self._tracker[aid_str] = {
                "first_seen": 0,
                "first_invested": None,
                "pauses": [],
                "became_idle_at": 0 if asset.state == AssetState.Idle else None,
            }
            self._prev_states[aid_str] = asset.state

    def on_step_end(self, context: MetricsContext) -> None:
        """Record value at step end."""
        self._step += 1
        gs = context.game_state
        current_ids = set()

        for aid, asset in gs.assets.items():
            aid_str = str(aid)
            current_ids.add(aid_str)
            prev = self._prev_states.get(aid_str)

            if prev is None:
                self._tracker[aid_str] = {
                    "first_seen": self._step,
                    "first_invested": None,
                    "pauses": [],
                    "became_idle_at": self._step
                    if asset.state == AssetState.Idle
                    else None,
                }

            tracker = self._tracker.get(aid_str)
            if tracker is None:
                continue

            if prev == AssetState.Idle and asset.state == AssetState.InDevelopment:
                if tracker["first_invested"] is None:
                    tracker["first_invested"] = self._step
                elif tracker["became_idle_at"] is not None:
                    pause = self._step - tracker["became_idle_at"]
                    tracker["pauses"].append(pause)
                tracker["became_idle_at"] = None

            if prev == AssetState.InDevelopment and asset.state == AssetState.Idle:
                tracker["became_idle_at"] = self._step

            self._prev_states[aid_str] = asset.state

        for aid_str in list(self._prev_states.keys()):
            if aid_str not in current_ids:
                del self._prev_states[aid_str]

    def on_episode_end(self, context: MetricsContext) -> None:
        """Record final value at episode end."""
        import numpy as np

        key = context.episode_key

        idle_before_first = []
        all_pauses = []

        for tracker in self._tracker.values():
            if tracker["first_invested"] is not None:
                idle_before_first.append(
                    tracker["first_invested"] - tracker["first_seen"]
                )
                all_pauses.extend(tracker["pauses"])

        self.history[key] = {
            "mean_idle_before_first_invest": (
                float(np.mean(idle_before_first)) if idle_before_first else 0.0
            ),
            "median_idle_before_first_invest": (
                float(np.median(idle_before_first)) if idle_before_first else 0.0
            ),
            "mean_inter_phase_pause": (
                float(np.mean(all_pauses)) if all_pauses else 0.0
            ),
            "max_inter_phase_pause": max(all_pauses) if all_pauses else 0,
            "n_pauses": len(all_pauses),
        }

    def report(self) -> dict:
        """Return collected metric data."""
        return {self.__class__.__name__: self.history}
