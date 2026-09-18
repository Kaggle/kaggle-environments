"""Utility functions for working with agents."""

import uuid
from typing import Literal, Optional

from pyxis_portfolio_challenge.config import (
    CapacityConfig,
    DistributionalPtrsConfig,
    InterimTrialObservationsConfig,
    InvestmentLevelParams,
    InvestmentLevelsConfig,
    TAExperienceConfig,
    UncertainPtrsConfig,
    config,
)
from pyxis_portfolio_challenge.environment.reward import LegacyStaticNPVReward
from pyxis_portfolio_challenge.environment.training_gym import InvestmentGameEnv
from pyxis_portfolio_challenge.game.game_state import GameState

_DISABLED_DISTRIBUTIONAL_PTRS = DistributionalPtrsConfig(
    enabled=False,
    ta_quality_variance={
        "oncology": 0.08,
        "respiratory and immunology": 0.05,
        "vaccines and infectious disease": 0.03,
    },
    asset_noise_std=0.03,
    prior_concentration=5.0,
    observation_noise=0.1,
)

_DISABLED_TA_EXPERIENCE = TAExperienceConfig(
    enabled=False,
    experience_to_full_knowledge=30.0,
    max_expertise_boost=0.05,
    experience_to_max_boost=40.0,
    experience_decay_rate=0.98,
    max_total_experience=60.0,
    phase_experience_weights={
        "phase_1": 0.5,
        "phase_2": 1.0,
        "phase_3": 1.5,
        "approval": 0.5,
    },
    asset_arrival_temperature=0.1,
)

_DISABLED_UNCERTAIN_PTRS = UncertainPtrsConfig(
    enabled=False,
    ta_noise_config={
        "oncology": 0.12,
        "respiratory and immunology": 0.10,
        "vaccines and infectious disease": 0.08,
    },
    phase_noise_multipliers={
        "phase_1": 1.5,
        "phase_2": 1.0,
        "phase_3": 0.75,
        "approval": 0.5,
    },
)

_DISABLED_INVESTMENT_LEVELS = InvestmentLevelsConfig(
    enabled=False,
    levels={
        "none": InvestmentLevelParams(
            cost_modifier=0.0,
            speed_modifier=0.0,
            success_modifier=1.0,
            capacity_cost=0,
            experience_modifier=0.0,
        ),
        "standard": InvestmentLevelParams(
            cost_modifier=1.0,
            speed_modifier=1.0,
            success_modifier=1.0,
            capacity_cost=2,
            experience_modifier=1.0,
        ),
    },
)

_DISABLED_INTERIM_TRIAL_OBSERVATIONS = InterimTrialObservationsConfig(
    enabled=False,
    latent_quality_concentration=10.0,
    initial_noise_scale=0.3,
)

_DISABLED_RD_CAPACITY = CapacityConfig(
    enabled=False,
    base_capacity=80.0,
    overage_max_penalty=0.5,
    overage_cost_max_penalty=0.5,
    overage_scaling="linear",
)

# The single-agent InvestmentGameEnv cannot model the multi-agent-only features
# (marketing, clinical sites, PTRS readings, approval phase) and raises if any is
# enabled. This reasoning env only extracts investment decisions, so we feed it
# disabled copies of the shipped configs — matching main's behaviour of ignoring
# these features here. Copying the live config (rather than hand-building) keeps
# every now-required field valid; the values are never read while disabled.
_DISABLED_MARKETING = config.marketing.model_copy(update={"enabled": False})
_DISABLED_CLINICAL_SITES = config.clinical_sites.model_copy(update={"enabled": False})
_DISABLED_PTRS_READINGS = config.ptrs_readings.model_copy(update={"enabled": False})
_DISABLED_APPROVAL_PHASE = config.approval_phase.model_copy(update={"enabled": False})


def get_agent_investment_decisions(
    agent,
    game_state: GameState,
) -> dict[uuid.UUID, Optional[Literal["invest"]]]:
    """
    Get investment decisions from an agent using the environment-based approach.

    This function creates an environment from the game state, gets observations,
    and retrieves agent actions. It then converts the action array back to
    investment decisions.

    This approach works for all agent types: KnapsackAgent can use either the
    environment or direct game state access.

    Parameters
    ----------
    agent : Agent
        The agent instance (KnapsackAgent, etc.).
    game_state : GameState
        The current game state to get recommendations for. The assets_dir is
        extracted from the game state's asset generator.

    Returns
    -------
    dict[uuid.UUID, Optional[Literal["invest"]]]
        A dictionary mapping asset IDs to investment decisions ("invest" or None).

    Examples
    --------
    >>> from pyxis_portfolio_challenge.agents import get_agent
    >>> agent = get_agent("Knapsack")
    >>> decisions = get_agent_investment_decisions(agent, game_state)
    >>> print(decisions)
    {UUID('...'): 'invest', UUID('...'): 'invest'}

    """
    # Extract assets_dir from the game state's asset generator
    assets_dir = game_state._asset_generator.assets_dir

    # Create environment from game state
    # Feature configs are extracted from the game state to ensure consistency
    env = InvestmentGameEnv(
        assets_dir=assets_dir,
        initial_game_state=game_state,
        equilibrium_num_assets=20,
        reinvestment_percentage=1.0,
        starting_cash=10_000_000,
        max_num_assets=game_state.max_num_assets,
        asset_arrival_sensitivity_below=1.5,
        asset_arrival_sensitivity_above=3.0,
        horizon=20,
        reward_fn=LegacyStaticNPVReward(),
        flatten_obs=True,
        shuffle_order=False,  # Don't shuffle for consistent ordering
        mask_first_order_assets=False,
        mask_negative_enpv_assets=False,
        distributional_ptrs_config=game_state._distributional_ptrs_config
        or _DISABLED_DISTRIBUTIONAL_PTRS,
        ta_experience_config=game_state._ta_experience_config
        or _DISABLED_TA_EXPERIENCE,
        uncertain_ptrs_config=game_state._uncertain_ptrs_config
        or _DISABLED_UNCERTAIN_PTRS,
        investment_levels_config=game_state._investment_levels_config
        or _DISABLED_INVESTMENT_LEVELS,
        interim_trial_observations_config=game_state._interim_trial_observations_config
        or _DISABLED_INTERIM_TRIAL_OBSERVATIONS,
        rd_capacity_config=game_state._rd_capacity_config or _DISABLED_RD_CAPACITY,
        drop_action_config=game_state._drop_action_config,
        # These four are multi-agent-only: the single-agent env rejects them when
        # enabled, so force them disabled unconditionally (a competition game_state
        # carries them enabled). This matches main, which ignored them here.
        marketing_config=_DISABLED_MARKETING,
        clinical_sites_config=_DISABLED_CLINICAL_SITES,
        ptrs_readings_config=_DISABLED_PTRS_READINGS,
        approval_phase_config=_DISABLED_APPROVAL_PHASE,
        metrics=[],
    )

    # Set environment on agent
    agent.set_env(env)

    # Reset to get initial observation
    obs, _ = env.reset()

    # Get agent's action
    action = agent(obs)

    # Convert action array to investment decisions
    investment_decisions = env._action_to_investment_decision(action)

    return investment_decisions


def get_all_agents_investment_decisions(
    agents: dict,
    game_state: GameState,
) -> dict[str, dict[uuid.UUID, Optional[Literal["invest"]]]]:
    """
    Get investment decisions from multiple agents.

    Parameters
    ----------
    agents : dict
        Dictionary mapping agent names to agent instances.
    game_state : GameState
        The current game state to get recommendations for.

    Returns
    -------
    dict[str, dict[uuid.UUID, Optional[Literal["invest"]]]]
        Dictionary mapping agent names to their investment decisions.

    Examples
    --------
    >>> agents = {"Knapsack": get_agent("Knapsack")}
    >>> all_decisions = get_all_agents_investment_decisions(agents, game_state)
    >>> print(all_decisions["Knapsack"])
    {UUID('...'): 'invest', UUID('...'): 'invest'}

    """
    all_decisions = {}

    for agent_name, agent_instance in agents.items():
        decisions = get_agent_investment_decisions(agent_instance, game_state)
        all_decisions[agent_name] = decisions

    return all_decisions
