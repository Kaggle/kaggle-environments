"""Configurable reward functions for multi-agent competitive environment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyxis_portfolio_challenge.game.game_state import GameState


class RewardType(str, Enum):
    """Types of reward functions available."""

    ABSOLUTE = "absolute"
    RELATIVE_RANK = "relative_rank"
    ZERO_SUM = "zero_sum"


class RewardFunction(ABC):
    """Base class for multi-agent reward functions."""

    @abstractmethod
    def compute(
        self,
        agent_id: str,
        pre_enpvs: dict[str, float],
        post_enpvs: dict[str, float],
        portfolios: dict[str, "GameState"],
    ) -> dict[str, float]:
        """Compute rewards for all agents."""
        pass


class AbsolutePerformanceReward(RewardFunction):
    """Absolute performance reward: delta_eNPV / scale_factor."""

    def __init__(self, scale_factor: float = 1.0):
        """Initialize."""
        self.scale_factor = scale_factor

    def compute(
        self,
        agent_id: str,
        pre_enpvs: dict[str, float],
        post_enpvs: dict[str, float],
        portfolios: dict[str, "GameState"],
    ) -> dict[str, float]:
        """Compute the reward value."""
        rewards = {}
        for aid in pre_enpvs:
            if portfolios[aid].bankrupt:
                rewards[aid] = -1.0
            else:
                delta = post_enpvs[aid] - pre_enpvs[aid]
                rewards[aid] = delta / self.scale_factor
        return rewards


class RelativeRankReward(RewardFunction):
    """Relative rank-based reward (1st: 1.0, 2nd: 0.5, etc.)."""

    def __init__(self, first_place: float = 1.0, decay_factor: float = 0.5):
        """Initialize."""
        self.first_place = first_place
        self.decay_factor = decay_factor

    def compute(
        self,
        agent_id: str,
        pre_enpvs: dict[str, float],
        post_enpvs: dict[str, float],
        portfolios: dict[str, "GameState"],
    ) -> dict[str, float]:
        """Compute the reward value."""
        deltas = {}
        for aid in pre_enpvs:
            if portfolios[aid].bankrupt:
                deltas[aid] = float("-inf")
            else:
                deltas[aid] = post_enpvs[aid] - pre_enpvs[aid]

        ranked = sorted(deltas.keys(), key=lambda x: deltas[x], reverse=True)

        rewards = {}
        for rank, aid in enumerate(ranked):
            if portfolios[aid].bankrupt:
                rewards[aid] = -1.0
            else:
                rewards[aid] = self.first_place * (self.decay_factor**rank)

        return rewards


class ZeroSumReward(RewardFunction):
    """Zero-sum competitive reward: my_delta - mean(others_delta)."""

    def __init__(self, scale_factor: float = 1.0):
        """Initialize."""
        self.scale_factor = scale_factor

    def compute(
        self,
        agent_id: str,
        pre_enpvs: dict[str, float],
        post_enpvs: dict[str, float],
        portfolios: dict[str, "GameState"],
    ) -> dict[str, float]:
        """Compute the reward value."""
        deltas = {}
        active_agents = []
        for aid in pre_enpvs:
            if portfolios[aid].bankrupt:
                deltas[aid] = 0.0
            else:
                deltas[aid] = post_enpvs[aid] - pre_enpvs[aid]
                active_agents.append(aid)

        if len(active_agents) == 0:
            return {aid: 0.0 for aid in pre_enpvs}

        total_delta = sum(deltas[aid] for aid in active_agents)

        rewards = {}
        for aid in pre_enpvs:
            if portfolios[aid].bankrupt:
                rewards[aid] = -1.0
            else:
                my_delta = deltas[aid]
                others_total = total_delta - my_delta
                others_count = len(active_agents) - 1

                if others_count > 0:
                    others_mean = others_total / others_count
                    rewards[aid] = (my_delta - others_mean) / self.scale_factor
                else:
                    rewards[aid] = my_delta / self.scale_factor

        return rewards


def create_reward_function(
    reward_type: RewardType | str,
    scale_factor: float = 1.0,
    first_place: float = 1.0,
    decay_factor: float = 0.5,
) -> RewardFunction:
    """Factory function to create reward functions."""
    if isinstance(reward_type, str):
        reward_type = RewardType(reward_type)

    if reward_type == RewardType.ABSOLUTE:
        return AbsolutePerformanceReward(scale_factor=scale_factor)
    elif reward_type == RewardType.RELATIVE_RANK:
        return RelativeRankReward(first_place=first_place, decay_factor=decay_factor)
    elif reward_type == RewardType.ZERO_SUM:
        return ZeroSumReward(scale_factor=scale_factor)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
