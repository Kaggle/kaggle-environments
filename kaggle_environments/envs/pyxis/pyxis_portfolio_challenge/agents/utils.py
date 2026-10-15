"""Utility functions for working with agents."""

import uuid
from typing import Literal, Optional

from pyxis_portfolio_challenge.config import config
from pyxis_portfolio_challenge.environment.reward import LegacyStaticNPVReward
from pyxis_portfolio_challenge.environment.training_gym import InvestmentGameEnv
from pyxis_portfolio_challenge.game.game_state import GameState

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
