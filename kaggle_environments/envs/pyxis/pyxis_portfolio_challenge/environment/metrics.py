import statistics
import uuid
from dataclasses import dataclass
from itertools import accumulate

# Avoid circular import — SharedMarketState only used as optional context
from typing import TYPE_CHECKING, Any, Protocol

from pyxis_portfolio_challenge.game.asset import AssetState
from pyxis_portfolio_challenge.game.game_state import GameState

if TYPE_CHECKING:
    from pyxis_portfolio_challenge.game.shared_market_state import SharedMarketState


@dataclass
class MetricsContext:
    """Context for metrics."""

    game_state: GameState
    reward: float
    investment_decisions: dict[uuid.UUID, str] | None = None
    shared_market_state: "SharedMarketState | None" = None
    agent_id: str | None = None
    all_agent_states: dict[str, GameState] | None = None
    all_agent_rewards: dict[str, float] | None = None
    bd_bid_levels: list[int] | None = None
    episode_id: str | None = None

    @property
    def episode_key(self) -> str:
        """Return a prefixed key for the episode identifier."""
        if self.episode_id is not None:
            return f"episode_id_{self.episode_id}"
        return f"game_state_id_{self.game_state.id}"


class EvaluationMetric(Protocol):
    """Protocol for evaluation metrics."""

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        pass

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called at the beginning of each episode."""
        pass

    def on_step_begin(self, context: MetricsContext) -> None:
        """Called at the start of every step."""
        pass

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        pass

    def on_episode_end(self, context: MetricsContext) -> None:
        """Called at the end of each episode."""
        pass

    def on_evaluation_end(self, context: MetricsContext) -> None:
        """Called once after a multi-episode evaluation run ends."""
        pass

    def report(self) -> dict[str, Any]:
        """Returns the collected metrics in a dictionary format."""
        pass


class MergeHistoryMixin:
    """
    Mixin to add merging functionality to metrics with history.

    Used to merge the history of two metrics of the same type that are collected
    separately in different processes.
    """

    def merge(self, other):
        """Merge the history of another metric into this one."""
        if type(self) is not type(other):
            raise TypeError("Metric type mismatch")

        for k, v in other.history.items():
            if k in self.history:
                raise ValueError(f"Duplicate metric key: {k}")
            self.history[k] = v


class PerEvaluationMetric(MergeHistoryMixin, EvaluationMetric):
    """Metric to keep track of a metric per evaluation."""


class PerEpisodeMetric(MergeHistoryMixin, EvaluationMetric):
    """Metric to keep track of a metric per episode."""


class PerStepMetric(MergeHistoryMixin, EvaluationMetric):
    """Metric to keep track of a metric per step."""


class PerEvaluationCumulativeReward(PerEvaluationMetric):
    """Metric to keep track of a cumulative reward per evaluation."""

    def __init__(self) -> None:
        """Initialize PerEvaluationCumulativeReward."""
        self.history: dict[str, float] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = 0.0

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        self.history[key] = self.history[key] + context.reward

    def report(self) -> dict[str, Any]:
        """Returns the collected metrics in a dictionary format."""
        combined = [cum_reward for cum_reward in self.history.values()]
        # Guard against empty history (e.g. workers that received 0 episodes)
        if not combined:
            return {
                self.__class__.__name__: {
                    "mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0,
                    "median": 0.0, "q25": 0.0, "q75": 0.0, "iqr": 0.0,
                }
            }
        sorted_combined = sorted(combined)
        n = len(sorted_combined)
        half = n // 2
        fallback = sorted_combined[0]
        q25 = statistics.median(sorted_combined[:half]) if n >= 2 else fallback
        q75 = statistics.median(sorted_combined[(n + 1) // 2 :]) if n >= 2 else fallback

        return {
            self.__class__.__name__: {
                "mean": statistics.mean(combined),
                "stdev": statistics.stdev(combined) if len(combined) > 1 else 0.0,
                "min": min(combined),
                "max": max(combined),
                "median": statistics.median(combined),
                "q25": q25,
                "q75": q75,
                "iqr": q75 - q25,
            }
        }


class PerEvaluationCumulativeNCF(PerEvaluationMetric):
    """Cumulative net cash flow per evaluation (mean/std across episodes)."""

    def __init__(self) -> None:
        """Initialize PerEvaluationCumulativeNCF."""
        self.history: dict[str, float] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = context.game_state.cash

    def on_episode_end(self, context: MetricsContext) -> None:
        """Store total NCF = final_cash - starting_cash."""
        key = context.episode_key
        starting_cash = self.history[key]
        self.history[key] = context.game_state.cash - starting_cash

    def report(self) -> dict[str, Any]:
        """Returns the collected metrics in a dictionary format."""
        combined = list(self.history.values())
        return {
            self.__class__.__name__: {
                "mean": statistics.mean(combined),
                "stdev": statistics.stdev(combined) if len(combined) > 1 else 0.0,
                "min": min(combined),
                "max": max(combined),
                "median": statistics.median(combined),
            }
        }


class PerEvaluationBankruptcyRate(PerEvaluationMetric):
    """Metric to keep track of the rate of bankruptcy per evaluation."""

    def __init__(self) -> None:
        """Initialize PerEvaluationBankruptcyRate."""
        self.history: dict[str, bool] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = False

    def on_episode_end(self, context: MetricsContext) -> None:
        """Called once after a multi-episode evaluation run ends."""
        key = context.episode_key
        if context.game_state.game_ended and context.game_state.bankrupt:
            self.history[key] = True

    def report(self) -> dict[str, Any]:
        """Returns the collected metrics in a dictionary format."""
        combined = [bankrupt for bankrupt in self.history.values()]

        return {
            self.__class__.__name__: {
                "bankruptcy_rate": sum(combined) / len(combined),
            }
        }


class PerEpisodeBankruptcyStep(PerEpisodeMetric):
    """Records the time step at which bankruptcy occurs for each episode."""

    def __init__(self) -> None:
        """Initialize PerEpisodeBankruptcyStep."""
        self.history: dict[str, int | None] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = f"episode_id_{str(context.game_state.id)}"
        self.history[key] = None

    def on_episode_end(self, context: MetricsContext) -> None:
        """Called once after a multi-episode evaluation run ends."""
        key = f"episode_id_{str(context.game_state.id)}"
        if context.game_state.game_ended and context.game_state.bankrupt:
            self.history[key] = context.game_state.time

    def report(self) -> dict[str, Any]:
        """Returns the collected metrics in a dictionary format."""
        steps = [s for s in self.history.values() if s is not None]
        return {
            self.__class__.__name__: {
                "bankruptcy_steps": steps,
                "count": len(steps),
                "total_episodes": len(self.history),
            }
        }


class PerEpisodeFinalEnpv(PerEpisodeMetric):
    """Metric to keep track of the number of the final eNPV."""

    def __init__(self) -> None:
        """Initialize PerEpisodeFinalEnpv."""
        self.history: dict[str, float] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = 0.0

    def on_episode_end(self, context: MetricsContext) -> None:
        """Called once after a multi-episode evaluation run ends."""
        key = context.episode_key
        self.history[key] = context.game_state.enpv_over_time[-1]

    def report(self) -> dict[str, Any]:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerEpisodeFinalEroi(PerEpisodeMetric):
    """Metric to keep track of the number of the final eROI."""

    def __init__(self) -> None:
        """Initialize PerEpisodeFinalEnpv."""
        self.history: dict[str, float] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = 0.0

    def on_episode_end(self, context: MetricsContext) -> None:
        """Called once after a multi-episode evaluation run ends."""
        key = context.episode_key
        self.history[key] = context.game_state.eroi_over_time[-1]

    def report(self) -> dict[str, Any]:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerEpisodeNumSteps(PerEpisodeMetric):
    """Metric to keep track of the number of steps per episode."""

    def __init__(self) -> None:
        """Initialize PerEpisodeNumSteps."""
        self.history: dict[str, int] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = 0

    def on_episode_end(self, context: MetricsContext) -> None:
        """Called once after a multi-episode evaluation run ends."""
        key = context.episode_key
        self.history[key] = context.game_state.time

    def report(self) -> dict[str, Any]:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerEpisodeCumulativeReward(PerEpisodeMetric):
    """Metric callback to keep track of the cumulative reward per episode."""

    def __init__(self) -> None:
        """Initialize PerEpisodeCumulativeReward."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        if key not in self.history:
            self.history[key] = []

        self.history[key].append(context.reward)

    def report(self) -> dict[str, Any]:
        """Returns the collected metrics in a dictionary format."""
        summary = {
            episode_name: {
                "cumulative": sum(rewards),
                "mean": statistics.mean(rewards),
                "stdev": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
            }
            for episode_name, rewards in self.history.items()
        }
        return {self.__class__.__name__: summary}


class PerEpisodeRealisedRoi(PerEpisodeMetric):
    """Metric callback to keep track of the roi per episode."""

    def __init__(self) -> None:
        """Initialize PerEpisodeRealisedRoi."""
        self.history: dict[str, float] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = 0.0

    def on_episode_end(self, context: MetricsContext) -> None:
        """Called after every episode."""
        key = context.episode_key
        self.history[key] = context.game_state.realised_roi()

    def report(self) -> dict[str, Any]:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepReward(PerStepMetric):
    """Metric to keep track of the reward per step."""

    def __init__(self) -> None:
        """Initialize PerStepReward."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = [0.0]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        self.history[key].append(context.reward)

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepCumulativeReward(PerStepMetric):
    """Metric to keep track of the cumulative reward per step."""

    def __init__(self) -> None:
        """Initialize PerStepReward."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = [0.0]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        self.history[key].append(self.history[key][-1] + context.reward)

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepEnpv(PerStepMetric):
    """Metric to keep track of the eNPV per step."""

    def __init__(self) -> None:
        """Initialize PerStepEnpv."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = []

    def on_episode_end(self, context: MetricsContext) -> None:
        """Called once before when episode run ends."""
        key = context.episode_key
        self.history[key] = context.game_state.running_enpv

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepEroi(PerStepMetric):
    """Metric to keep track of the eROI per step."""

    def __init__(self) -> None:
        """Initialize PerStepEroi."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = []

    def on_episode_end(self, context: MetricsContext) -> None:
        """Called once before when episode run ends."""
        key = context.episode_key
        self.history[key] = context.game_state.running_eroi

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepCash(PerStepMetric):
    """Metric to keep track of the cash per step."""

    def __init__(self) -> None:
        """Initialize PerStepCash."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = [context.game_state.cash]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        self.history[key] = self.history[key] + [context.game_state.cash]

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepRevenue(PerStepMetric):
    """Metric to keep track of the revenue per step."""

    def __init__(self) -> None:
        """Initialize PerStepRevenue."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = []

    def on_episode_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        self.history[key] = self.history[key] + context.game_state.realised_revenues

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepCost(PerStepMetric):
    """Metric to keep track of the cost per step."""

    def __init__(self) -> None:
        """Initialize PerStepCost."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = []

    def on_episode_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        self.history[key] = self.history[key] + context.game_state.realised_costs

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepNetCashFlow(PerStepMetric):
    """Metric to keep track of the net cash flow per step."""

    def __init__(self) -> None:
        """Initialize PerStepNetCashFlow."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = []

    def on_episode_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        revenues = context.game_state.realised_revenues
        costs = context.game_state.realised_costs
        reinvestment_percentage = context.game_state.reinvestment_percentage
        # Apply reinvestment_percentage to get actual cash collected
        cashflows = [
            revenue * reinvestment_percentage - cost
            for revenue, cost in zip(revenues, costs)
        ]
        self.history[key] = self.history[key] + cashflows

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepCumulativeNetCashFlow(PerStepMetric):
    """Metric to keep track of the cumulative net cash flow per step."""

    def __init__(self) -> None:
        """Initialize PerStepCumulativeNetCashFlow."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = []

    def on_episode_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        revenues = context.game_state.realised_revenues
        costs = context.game_state.realised_costs
        reinvestment_percentage = context.game_state.reinvestment_percentage
        # Apply reinvestment_percentage to get actual cash collected
        cashflows = [
            revenue * reinvestment_percentage - cost
            for revenue, cost in zip(revenues, costs)
        ]
        self.history[key] = self.history[key] + list(accumulate(cashflows))

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepNumAssetsIdleState(PerStepMetric):
    """Metric to keep track of number of assets in Idle state."""

    def __init__(self) -> None:
        """Initialize PerStepNumAssetsIdleState."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        num_idle_assets = len([
            asset
            for asset in context.game_state.assets.values()
            if asset.state == AssetState.Idle
        ])
        self.history[key] = [num_idle_assets]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        num_idle_assets = len([
            asset
            for asset in context.game_state.assets.values()
            if asset.state == AssetState.Idle
        ])
        self.history[key] = self.history[key] + [num_idle_assets]

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepNumAssetsInDevelopmentState(PerStepMetric):
    """Metric to keep track of number of assets in InDevelopment state."""

    def __init__(self) -> None:
        """Initialize PerStepNumAssetsInDevelopmentState."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        num_in_dev_assets = len([
            asset
            for asset in context.game_state.assets.values()
            if asset.state == AssetState.InDevelopment
        ])
        self.history[key] = [num_in_dev_assets]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        num_in_dev_assets = len([
            asset
            for asset in context.game_state.assets.values()
            if asset.state == AssetState.InDevelopment
        ])
        self.history[key] = self.history[key] + [num_in_dev_assets]

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepNumAssetsOnMarketState(PerStepMetric):
    """Metric to keep track of number of assets in OnMarket state."""

    def __init__(self) -> None:
        """Initialize PerStepNumAssetsOnMarketState."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        num_on_mkt_assets = len([
            asset
            for asset in context.game_state.assets.values()
            if asset.state == AssetState.OnMarket
        ])
        self.history[key] = [num_on_mkt_assets]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        num_on_mkt_assets = len([
            asset
            for asset in context.game_state.assets.values()
            if asset.state == AssetState.OnMarket
        ])
        self.history[key] = self.history[key] + [num_on_mkt_assets]

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepNumAssetsFailedState(PerStepMetric):
    """Metric to keep track of number of assets in Failed state."""

    def __init__(self) -> None:
        """Initialize PerStepNumAssetsFailedState."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = [len(context.game_state.failed_assets)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        self.history[key] = self.history[key] + [len(context.game_state.failed_assets)]

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepNumAssetsExpiredState(PerStepMetric):
    """Metric to keep track of number of assets in Expired state."""

    def __init__(self) -> None:
        """Initialize PerStepNetCashFlow."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = [len(context.game_state.expired_assets)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        self.history[key] = self.history[key] + [len(context.game_state.expired_assets)]

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepFractionOfPossibleInvestments(PerStepMetric):
    """Metric to keep track fraction of possible investments that were invested."""

    def __init__(self) -> None:
        """Initialize PerStepFractionOfPossibleInvestments."""
        self.history: dict[str, list[float]] = {}
        self._step_idle_assets: list[str] = []

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = []

    def on_step_begin(self, context: MetricsContext) -> None:
        """Called after every step."""
        # get number of assets available for investment at start of step
        assets_available_for_inv = [
            asset_id
            for asset_id, asset in context.game_state.assets.items()
            if asset.state == AssetState.Idle
        ]

        investment_decisions = context.investment_decisions
        # calculate number of investments made this step
        num_investments_made = sum(
            1
            for asset_id in assets_available_for_inv
            if investment_decisions.get(asset_id) == "invest"
        )
        fraction = 0.0
        if len(assets_available_for_inv) > 0:
            fraction = num_investments_made / len(assets_available_for_inv)
        key = context.episode_key
        self.history[key] = self.history[key] + [fraction]

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepFractionOfPossibleInvestmentsPosEnpv(PerStepMetric):
    """Metric to keep track FOPI that were invested that have positive eNPV."""

    def __init__(self) -> None:
        """Initialize PerStepFractionOfPossibleInvestmentsPosEnpv."""
        self.history: dict[str, list[float]] = {}
        self._step_idle_assets: list[str] = []

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = []

    def on_step_begin(self, context: MetricsContext) -> None:
        """Called after every step."""
        # get number of assets available for investment at start of step
        assets_available_for_inv = [
            asset_id
            for asset_id, asset in context.game_state.assets.items()
            if (asset.state == AssetState.Idle and asset.enpv > 0)
        ]

        investment_decisions = context.investment_decisions
        # calculate number of investments made this step
        num_investments_made = sum(
            1
            for asset_id in assets_available_for_inv
            if investment_decisions.get(asset_id) == "invest"
        )

        fraction = 0.0
        if len(assets_available_for_inv) > 0:
            fraction = num_investments_made / len(assets_available_for_inv)
        key = context.episode_key
        self.history[key] = self.history[key] + [fraction]

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepNumSkippedPosEnpvAssets(PerStepMetric):
    """
    Count of positive-eNPV idle assets the agent chose not to invest in each step.

    Fires at on_step_begin so it only counts assets the agent saw and skipped,
    not assets that returned to Idle mid-step from a completed trial phase.
    """

    def __init__(self) -> None:
        """Initialize PerStepNumSkippedPosEnpvAssets."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = f"episode_id_{str(context.game_state.id)}"
        assert key not in self.history
        self.history[key] = []

    def on_step_begin(self, context: MetricsContext) -> None:
        """Called at the start of every step."""
        pos_enpv_idle = [
            asset_id
            for asset_id, asset in context.game_state.assets.items()
            if asset.state == AssetState.Idle and asset.enpv > 0
        ]
        investment_decisions = context.investment_decisions
        skipped = sum(
            1
            for asset_id in pos_enpv_idle
            if investment_decisions.get(asset_id) != "invest"
        )
        key = f"episode_id_{str(context.game_state.id)}"
        self.history[key] = self.history[key] + [skipped]

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


# =============================================================================
# Uncertain PTRS Feature Metrics
# =============================================================================


class PerStepTAExperienceOncology(PerStepMetric):
    """Metric to track TA experience for Oncology per step."""

    def __init__(self) -> None:
        """Initialize PerStepTAExperienceOncology."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        experience = context.game_state.ta_experience.get("oncology", 0.0)
        self.history[key] = [experience]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        experience = context.game_state.ta_experience.get("oncology", 0.0)
        self.history[key].append(experience)

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepTAExperienceRespiratoryImmunology(PerStepMetric):
    """Metric to track TA experience for Respiratory and Immunology per step."""

    def __init__(self) -> None:
        """Initialize PerStepTAExperienceRespiratoryImmunology."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        experience = context.game_state.ta_experience.get(
            "respiratory and immunology", 0.0
        )
        self.history[key] = [experience]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        experience = context.game_state.ta_experience.get(
            "respiratory and immunology", 0.0
        )
        self.history[key].append(experience)

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepTAExperienceVaccines(PerStepMetric):
    """Metric to track TA experience for Vaccines and Infectious Disease per step."""

    def __init__(self) -> None:
        """Initialize PerStepTAExperienceVaccines."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        experience = context.game_state.ta_experience.get(
            "vaccines and infectious disease", 0.0
        )
        self.history[key] = [experience]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        experience = context.game_state.ta_experience.get(
            "vaccines and infectious disease", 0.0
        )
        self.history[key].append(experience)

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepMeanPTRSError(PerStepMetric):
    """
    Metric to track mean absolute error between observed and true PTRS.

    Lower values indicate better PTRS knowledge (convergence toward true PTRS).
    Only considers assets with uncertain PTRS enabled (those with _true_ptrs set).
    """

    def __init__(self) -> None:
        """Initialize PerStepMeanPTRSError."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = [self._compute_mean_error(context.game_state)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        self.history[key].append(self._compute_mean_error(context.game_state))

    def _compute_mean_error(self, game_state: GameState) -> float:
        """Compute mean absolute PTRS error across all assets with uncertain PTRS."""
        errors = []
        for asset in game_state.assets.values():
            trial = asset.trial
            while trial is not None:
                if trial._true_ptrs is not None:
                    error = abs(trial.ptrs - trial._true_ptrs)
                    errors.append(error)
                trial = trial.next_trial_on_success
        return statistics.mean(errors) if errors else 0.0

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepMeanExpertiseBoost(PerStepMetric):
    """
    Metric to track mean expertise boost applied to true PTRS.

    Shows how much PTRS improvement agents get from TA specialization.
    """

    def __init__(self) -> None:
        """Initialize PerStepMeanExpertiseBoost."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = [self._compute_mean_boost(context.game_state)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        self.history[key].append(self._compute_mean_boost(context.game_state))

    def _compute_mean_boost(self, game_state: GameState) -> float:
        """Compute mean expertise boost across all assets with uncertain PTRS."""
        boosts = []
        for asset in game_state.assets.values():
            trial = asset.trial
            while trial is not None:
                if (
                    trial._true_ptrs is not None
                    and trial._effective_true_ptrs is not None
                ):
                    boost = trial._effective_true_ptrs - trial._true_ptrs
                    boosts.append(boost)
                trial = trial.next_trial_on_success
        return statistics.mean(boosts) if boosts else 0.0

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepNumAssetsOncology(PerStepMetric):
    """Metric to track number of assets in Oncology TA per step."""

    def __init__(self) -> None:
        """Initialize PerStepNumAssetsOncology."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        count = sum(
            1
            for asset in context.game_state.assets.values()
            if asset.therapeutic_area == "oncology"
        )
        self.history[key] = [count]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        count = sum(
            1
            for asset in context.game_state.assets.values()
            if asset.therapeutic_area == "oncology"
        )
        self.history[key].append(count)

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepNumAssetsRespiratoryImmunology(PerStepMetric):
    """Metric to track number of assets in Respiratory and Immunology TA per step."""

    def __init__(self) -> None:
        """Initialize PerStepNumAssetsRespiratoryImmunology."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        count = sum(
            1
            for asset in context.game_state.assets.values()
            if asset.therapeutic_area == "respiratory and immunology"
        )
        self.history[key] = [count]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        count = sum(
            1
            for asset in context.game_state.assets.values()
            if asset.therapeutic_area == "respiratory and immunology"
        )
        self.history[key].append(count)

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepNumAssetsVaccines(PerStepMetric):
    """Metric to track assets in Vaccines and Infectious Disease TA per step."""

    def __init__(self) -> None:
        """Initialize PerStepNumAssetsVaccines."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        count = sum(
            1
            for asset in context.game_state.assets.values()
            if asset.therapeutic_area == "vaccines and infectious disease"
        )
        self.history[key] = [count]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        count = sum(
            1
            for asset in context.game_state.assets.values()
            if asset.therapeutic_area == "vaccines and infectious disease"
        )
        self.history[key].append(count)

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


# =============================================================================
# Investment Levels Feature Metrics
# =============================================================================


class PerStepCapacityUsed(PerStepMetric):
    """Metric to track capacity used per step."""

    def __init__(self) -> None:
        """Initialize PerStepCapacityUsed."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = [context.game_state.capacity_used]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        self.history[key].append(context.game_state.capacity_used)

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepCapacityRatio(PerStepMetric):
    """Metric to track capacity ratio (used/base) per step."""

    def __init__(self) -> None:
        """Initialize PerStepCapacityRatio."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = [context.game_state.capacity_ratio]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        self.history[key].append(context.game_state.capacity_ratio)

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepGlobalSuccessModifier(PerStepMetric):
    """Metric to track global success modifier from capacity overage per step."""

    def __init__(self) -> None:
        """Initialize PerStepGlobalSuccessModifier."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = [context.game_state.success_modifier]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        self.history[key].append(context.game_state.success_modifier)

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepGlobalCostModifier(PerStepMetric):
    """Metric to track global cost modifier from capacity overage per step."""

    def __init__(self) -> None:
        """Initialize PerStepGlobalCostModifier."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = [context.game_state.cost_modifier]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        self.history[key].append(context.game_state.cost_modifier)

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepNumAssetsMinimalLevel(PerStepMetric):
    """Metric to track number of assets at MINIMAL investment level per step."""

    def __init__(self) -> None:
        """Initialize PerStepNumAssetsMinimalLevel."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        from pyxis_portfolio_challenge.game.constants import InvestmentLevel

        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        count = sum(
            1
            for asset in context.game_state.assets.values()
            if asset.current_investment_level == InvestmentLevel.MINIMAL
        )
        self.history[key] = [count]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        from pyxis_portfolio_challenge.game.constants import InvestmentLevel

        key = context.episode_key
        count = sum(
            1
            for asset in context.game_state.assets.values()
            if asset.current_investment_level == InvestmentLevel.MINIMAL
        )
        self.history[key].append(count)

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepNumAssetsStandardLevel(PerStepMetric):
    """Metric to track number of assets at STANDARD investment level per step."""

    def __init__(self) -> None:
        """Initialize PerStepNumAssetsStandardLevel."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        from pyxis_portfolio_challenge.game.constants import InvestmentLevel

        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        count = sum(
            1
            for asset in context.game_state.assets.values()
            if asset.current_investment_level == InvestmentLevel.STANDARD
        )
        self.history[key] = [count]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        from pyxis_portfolio_challenge.game.constants import InvestmentLevel

        key = context.episode_key
        count = sum(
            1
            for asset in context.game_state.assets.values()
            if asset.current_investment_level == InvestmentLevel.STANDARD
        )
        self.history[key].append(count)

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepNumAssetsAcceleratedLevel(PerStepMetric):
    """Metric to track number of assets at ACCELERATED investment level per step."""

    def __init__(self) -> None:
        """Initialize PerStepNumAssetsAcceleratedLevel."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        from pyxis_portfolio_challenge.game.constants import InvestmentLevel

        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        count = sum(
            1
            for asset in context.game_state.assets.values()
            if asset.current_investment_level == InvestmentLevel.ACCELERATED
        )
        self.history[key] = [count]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        from pyxis_portfolio_challenge.game.constants import InvestmentLevel

        key = context.episode_key
        count = sum(
            1
            for asset in context.game_state.assets.values()
            if asset.current_investment_level == InvestmentLevel.ACCELERATED
        )
        self.history[key].append(count)

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


# =============================================================================
# Interim Trial Observations Metrics
# =============================================================================


class PerStepMeanInterimSignal(PerStepMetric):
    """
    Metric to track mean interim signal across in-development assets.

    Interim signals indicate noisy observations of trial quality during development.
    Lower signals suggest the trial may fail; higher signals suggest success.
    """

    def __init__(self) -> None:
        """Initialize PerStepMeanInterimSignal."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = [self._compute_mean_signal(context.game_state)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        self.history[key].append(self._compute_mean_signal(context.game_state))

    def _compute_mean_signal(self, game_state: GameState) -> float:
        """Compute mean interim signal across in-development assets."""
        signals = []
        for asset in game_state.assets.values():
            if (
                asset.state == AssetState.InDevelopment
                and asset.interim_observations_enabled
            ):
                signal = asset.interim_signal
                if signal is not None:
                    signals.append(signal)
        return statistics.mean(signals) if signals else 0.0

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepMeanTrialProgress(PerStepMetric):
    """
    Metric to track mean trial progress across in-development assets.

    Progress ranges from 0.0 (just started) to 1.0 (about to complete).
    Higher progress means more accurate interim signals.
    """

    def __init__(self) -> None:
        """Initialize PerStepMeanTrialProgress."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = [self._compute_mean_progress(context.game_state)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        self.history[key].append(self._compute_mean_progress(context.game_state))

    def _compute_mean_progress(self, game_state: GameState) -> float:
        """Compute mean trial progress across in-development assets."""
        progress_values = []
        for asset in game_state.assets.values():
            if asset.state == AssetState.InDevelopment:
                progress = asset.trial_progress
                if progress is not None:
                    progress_values.append(progress)
        return statistics.mean(progress_values) if progress_values else 0.0

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepMinInterimSignal(PerStepMetric):
    """
    Metric to track minimum interim signal across in-development assets.

    Useful for identifying the worst-performing trial at each step.
    Low signals are candidates for early stopping (STOP action).
    """

    def __init__(self) -> None:
        """Initialize PerStepMinInterimSignal."""
        self.history: dict[str, list[float]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = [self._compute_min_signal(context.game_state)]

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step."""
        key = context.episode_key
        self.history[key].append(self._compute_min_signal(context.game_state))

    def _compute_min_signal(self, game_state: GameState) -> float:
        """Compute minimum interim signal across in-development assets."""
        signals = []
        for asset in game_state.assets.values():
            if (
                asset.state == AssetState.InDevelopment
                and asset.interim_observations_enabled
            ):
                signal = asset.interim_signal
                if signal is not None:
                    signals.append(signal)
        return min(signals) if signals else 1.0

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerEpisodeNumStopActions(PerEpisodeMetric):
    """
    Metric to track total STOP actions taken per episode.

    STOP actions terminate in-development trials early to free capacity.
    Higher counts indicate more aggressive trial management.
    """

    def __init__(self) -> None:
        """Initialize PerEpisodeNumStopActions."""
        self.history: dict[str, int] = {}
        self._current_count: int = 0

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self._current_count = 0

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step - checks for STOP actions in investment decisions."""
        from pyxis_portfolio_challenge.game.constants import InvestmentLevel

        if context.investment_decisions:
            for decision in context.investment_decisions.values():
                if decision == InvestmentLevel.STOP:
                    self._current_count += 1

    def on_episode_end(self, context: MetricsContext) -> None:
        """Called at the end of each episode."""
        key = context.episode_key
        self.history[key] = self._current_count

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


class PerStepNumStopActions(PerStepMetric):
    """
    Metric to track STOP actions taken at each step.

    Records count of STOP actions taken, useful for understanding
    when agents choose to stop trials during episodes.
    """

    def __init__(self) -> None:
        """Initialize PerStepNumStopActions."""
        self.history: dict[str, list[int]] = {}

    def on_evaluation_begin(self, context: MetricsContext) -> None:
        """Called once before a multi-episode evaluation run starts."""
        self.history.clear()

    def on_episode_begin(self, context: MetricsContext) -> None:
        """Called once before an episode run starts."""
        key = context.episode_key
        assert key not in self.history, "Got a duplicate episode_id, shouldn't happen."
        self.history[key] = []

    def on_step_end(self, context: MetricsContext) -> None:
        """Called after every step - checks for STOP actions in investment decisions."""
        from pyxis_portfolio_challenge.game.constants import InvestmentLevel

        key = context.episode_key
        stop_count = 0
        if context.investment_decisions:
            for decision in context.investment_decisions.values():
                if decision == InvestmentLevel.STOP:
                    stop_count += 1
        self.history[key].append(stop_count)

    def report(self) -> dict:
        """Returns the collected metrics in a dictionary format."""
        return {self.__class__.__name__: self.history}


def collect_metrics(
    collection_fn: str,
    metrics: list[EvaluationMetric],
    context: MetricsContext | None,
) -> None:
    """Collect metric data for all metrics for collection_state function."""
    for metric in metrics:
        # Skip metrics that are in warmup mode
        if getattr(metric, "_warmup_mode", False):
            continue
        metric.__getattribute__(collection_fn)(context=context)


def merge_all_metrics(
    metrics_sets: list[list[EvaluationMetric]],
) -> list[EvaluationMetric]:
    """Merge all instances of the same metric, for multiprocessing."""
    merged = metrics_sets[0]

    for metrics in metrics_sets[1:]:
        for m_target, m_src in zip(merged, metrics):
            m_target.merge(m_src)

    return merged


def report_all_metrics(
    metrics: list[EvaluationMetric],
) -> list[dict[str, Any]]:
    """Report all metrics as a jsonable dict."""
    per_evaluation_report = {"PerEvaluationMetrics": []}
    per_episode_report = {"PerEpisodeMetrics": []}
    per_step_report = {"PerStepMetrics": []}
    for metric in metrics:
        if isinstance(metric, PerEvaluationMetric):
            per_evaluation_report["PerEvaluationMetrics"].append(metric.report())
        elif isinstance(metric, PerEpisodeMetric):
            per_episode_report["PerEpisodeMetrics"].append(metric.report())
        elif isinstance(metric, PerStepMetric):
            per_step_report["PerStepMetrics"].append(metric.report())
    return [
        per_evaluation_report,
        per_episode_report,
        per_step_report,
    ]


def legacy_static_npv(game_state: GameState) -> float:
    """
    Calculate the Net Present Value (NPV) of the game state.

    This assumes a 'smart' investor, so if an asset is Idle we only include its NPV if
    it is non-negative.

    Parameters
    ----------
    game_state: MetricsContext
        The game state of which to calculate the NPV.

    Returns
    -------
    float
        The calculated NPV of the game state.

    """
    total_npv = game_state.cash
    for asset in game_state.assets.values():
        if asset.state == "Idle" and asset.enpv < 0:
            continue
        total_npv += asset.enpv
    return total_npv
