"""Environment factory functions for single-agent and multi-agent environments."""

from __future__ import annotations

from typing import Any

import upath

from pyxis_portfolio_challenge.config import (
    config,
    instantiate_from_config,
)
from pyxis_portfolio_challenge.environment.training_gym import (  # noqa: E501
    InvestmentGameEnv,
)
from pyxis_portfolio_challenge.environment.warmup_wrapper import (
    MultiAgentWarmupOnResetWrapper,
    WarmupOnResetWrapper,
)
from pyxis_portfolio_challenge.environment.wrappers import AutoCenterWrapper


def _prepare_envs(
    n_envs: int,
    norm_obs: bool,
    norm_reward: bool,
    monitor_path: str | None = None,
):
    """
    Prepare the training and evaluation environments.

    Environment parameters are loaded from the central config (config.yaml).
    Only training-specific parameters (n_envs, normalisation, monitor path)
    are passed as arguments.

    Parameters
    ----------
    n_envs : int
        Number of parallel environments to create.
    norm_obs : bool
        Whether to normalize observations.
    norm_reward : bool
        Whether to normalize rewards in the training environment.
    monitor_path : Optional[str]
        Path to the monitor directory.

    """
    # Imported here rather than at module scope: stable-baselines3 pulls in
    # torch/matplotlib, and only this training path needs them.
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

    from pyxis_portfolio_challenge.environment.vec_wrappers import (
        VecAutoCenterWrapper,
    )

    cfg = config

    reward_fn = instantiate_from_config(cfg.reward_fn)

    def train_env():
        """Create the training environment."""
        env = InvestmentGameEnv(
            equilibrium_num_assets=cfg.equilibrium_num_assets,
            max_num_assets=cfg.max_num_assets,
            asset_arrival_sensitivity_below=cfg.asset_arrival_sensitivity_below,
            asset_arrival_sensitivity_above=cfg.asset_arrival_sensitivity_above,
            starting_cash=cfg.starting_cash,
            horizon=cfg.horizon,
            reward_fn=reward_fn,
            assets_dir=cfg.training_data_dir,
            reinvestment_percentage=cfg.reinvestment_percentage,
            shuffle_order=cfg.shuffle_order,
            mask_first_order_assets=cfg.mask_first_order_assets,
            mask_negative_enpv_assets=cfg.mask_negative_enpv_assets,
            flatten_obs=cfg.flatten_obs,
            ta_experience_config=cfg.ta_experience,
            uncertain_ptrs_config=cfg.uncertain_ptrs,
            investment_levels_config=cfg.investment_levels,
            interim_trial_observations_config=cfg.interim_trial_observations,
            distributional_ptrs_config=cfg.distributional_ptrs,
            rd_capacity_config=cfg.rd_capacity,
            drop_action_config=cfg.drop_action,
            marketing_config=cfg.marketing,
            clinical_sites_config=cfg.clinical_sites,
            ptrs_readings_config=cfg.ptrs_readings,
            approval_phase_config=cfg.approval_phase,
            metrics=[],
            initial_game_state=None,
        )
        if cfg.warmup_on_reset_steps > 0:
            env = WarmupOnResetWrapper(
                env,
                warmup_steps=cfg.warmup_on_reset_steps,
                policy=cfg.warmup_on_reset_policy,
                verbose=False,
            )
        return env

    def eval_env():
        """Create the evaluation environment."""
        env = InvestmentGameEnv(
            equilibrium_num_assets=cfg.equilibrium_num_assets,
            max_num_assets=cfg.max_num_assets,
            asset_arrival_sensitivity_below=cfg.asset_arrival_sensitivity_below,
            asset_arrival_sensitivity_above=cfg.asset_arrival_sensitivity_above,
            starting_cash=cfg.starting_cash,
            horizon=cfg.horizon,
            reward_fn=reward_fn,
            assets_dir=cfg.evaluation_data_dir,
            reinvestment_percentage=cfg.reinvestment_percentage,
            shuffle_order=cfg.shuffle_order,
            mask_first_order_assets=cfg.mask_first_order_assets,
            mask_negative_enpv_assets=cfg.mask_negative_enpv_assets,
            flatten_obs=cfg.flatten_obs,
            ta_experience_config=cfg.ta_experience,
            uncertain_ptrs_config=cfg.uncertain_ptrs,
            investment_levels_config=cfg.investment_levels,
            interim_trial_observations_config=cfg.interim_trial_observations,
            distributional_ptrs_config=cfg.distributional_ptrs,
            rd_capacity_config=cfg.rd_capacity,
            drop_action_config=cfg.drop_action,
            marketing_config=cfg.marketing,
            clinical_sites_config=cfg.clinical_sites,
            ptrs_readings_config=cfg.ptrs_readings,
            approval_phase_config=cfg.approval_phase,
            metrics=[],
            initial_game_state=None,
        )
        if cfg.warmup_on_reset_steps > 0:
            env = WarmupOnResetWrapper(
                env,
                warmup_steps=cfg.warmup_on_reset_steps,
                policy=cfg.warmup_on_reset_policy,
                verbose=False,
            )
        return env

    train_env = SubprocVecEnv([train_env for _ in range(n_envs)])
    if cfg.auto_center_rewards:
        train_env = VecAutoCenterWrapper(
            train_env, calibration_steps=cfg.auto_center_calibration_steps
        )
    if monitor_path is not None:
        train_env = VecMonitor(train_env, monitor_path)
    train_env = VecNormalize(train_env, norm_obs=norm_obs, norm_reward=norm_reward)

    eval_env = SubprocVecEnv([eval_env for _ in range(n_envs)])
    if cfg.auto_center_rewards:
        eval_env = VecAutoCenterWrapper(
            eval_env, calibration_steps=cfg.auto_center_calibration_steps
        )
    if monitor_path is not None:
        eval_env = VecMonitor(eval_env, None)
    eval_env = VecNormalize(
        eval_env, norm_obs=norm_obs, norm_reward=False, training=False
    )

    return train_env, eval_env


def make_train_env(flatten_obs: bool = True) -> InvestmentGameEnv:
    """
    Create a training environment based on the fixed YAML configuration.

    Args:
        flatten_obs (bool): Whether to flatten the observations. Default is True.

    Returns:
        InvestmentGameEnv: The created training environment.

    """
    cfg = config
    env = InvestmentGameEnv(
        equilibrium_num_assets=cfg.equilibrium_num_assets,
        max_num_assets=cfg.max_num_assets,
        asset_arrival_sensitivity_below=cfg.asset_arrival_sensitivity_below,
        asset_arrival_sensitivity_above=cfg.asset_arrival_sensitivity_above,
        starting_cash=cfg.starting_cash,
        horizon=cfg.horizon,
        reward_fn=instantiate_from_config(cfg.reward_fn),
        assets_dir=cfg.training_data_dir,
        reinvestment_percentage=cfg.reinvestment_percentage,
        shuffle_order=cfg.shuffle_order,
        mask_first_order_assets=cfg.mask_first_order_assets,
        mask_negative_enpv_assets=cfg.mask_negative_enpv_assets,
        flatten_obs=flatten_obs,
        uncertain_ptrs_config=cfg.uncertain_ptrs,
        investment_levels_config=cfg.investment_levels,
        interim_trial_observations_config=cfg.interim_trial_observations,
        distributional_ptrs_config=cfg.distributional_ptrs,
        ta_experience_config=cfg.ta_experience,
        rd_capacity_config=cfg.rd_capacity,
        drop_action_config=cfg.drop_action,
        marketing_config=cfg.marketing,
        clinical_sites_config=cfg.clinical_sites,
        ptrs_readings_config=cfg.ptrs_readings,
        approval_phase_config=cfg.approval_phase,
        metrics=[],
        initial_game_state=None,
    )
    if cfg.auto_center_rewards:
        env = AutoCenterWrapper(
            env, calibration_steps=cfg.auto_center_calibration_steps
        )

    # Apply warmup wrapper if enabled
    if cfg.warmup_on_reset_steps > 0:
        env = WarmupOnResetWrapper(
            env,
            warmup_steps=cfg.warmup_on_reset_steps,
            policy=cfg.warmup_on_reset_policy,
            verbose=False,
        )

    return env


def _build_multi_agent_env_kwargs(
    flatten_obs: bool,
    num_agents: int | None = None,
    assets_dir: upath.UPath | None = None,
    bd_assets_dir: upath.UPath | None = None,
) -> dict[str, Any]:
    """
    Build keyword arguments for MultiAgentInvestmentGameEnv from config.

    Parameters
    ----------
    flatten_obs : bool
        Whether to flatten observations into a single array.
    num_agents : int | None
        Override number of agents. If None, uses config value.
    assets_dir : upath.UPath | None
        Override portfolio assets directory. Defaults to training_data_dir.
    bd_assets_dir : upath.UPath | None
        Override BD assets directory. Defaults to multi_agent.bd_assets_dir.

    Returns
    -------
    dict[str, Any]
        Keyword arguments for MultiAgentInvestmentGameEnv constructor.

    """
    cfg = config
    ma = cfg.multi_agent
    return dict(
        assets_dir=assets_dir if assets_dir is not None else cfg.training_data_dir,
        num_agents=num_agents if num_agents is not None else ma.num_agents,
        starting_cash=cfg.starting_cash,
        max_num_assets=cfg.max_num_assets,
        horizon=cfg.horizon,
        equilibrium_num_assets=cfg.equilibrium_num_assets,
        asset_arrival_sensitivity_below=cfg.asset_arrival_sensitivity_below,
        asset_arrival_sensitivity_above=cfg.asset_arrival_sensitivity_above,
        reinvestment_percentage=cfg.reinvestment_percentage,
        bd_enabled=ma.bd_enabled,
        bd_assets_dir=bd_assets_dir if bd_assets_dir is not None else ma.bd_assets_dir,
        bd_base_lambda=ma.bd_base_lambda,
        bd_leak_lambda_boost=ma.bd_leak_lambda_boost,
        bd_min_step=ma.bd_min_step,
        bd_max_bid=ma.bd_max_bid,
        bd_max_slots=ma.bd_max_slots,
        bd_persist_steps=ma.bd_persist_steps,
        bd_phase_weights=list(ma.bd_phase_weights),
        bd_indication_activity_bias=ma.bd_indication_activity_bias,
        exclusivity_period=ma.exclusivity_period,
        first_mover_bonus=ma.first_mover_bonus,
        disable_market_share_competition=ma.disable_market_share_competition,
        alert_history_length=ma.alert_history_length,
        leak_phase_probabilities=list(ma.leak_phase_probabilities),
        be_leak_probability=ma.be_leak_probability,
        dc_leak_probability=ma.dc_leak_probability,
        dc_leak_min_agents=ma.dc_leak_min_agents,
        alerts_per_agent=ma.alerts_per_agent,
        reward_fn=instantiate_from_config(cfg.reward_fn),
        shuffle_order=cfg.shuffle_order,
        mask_first_order_assets=cfg.mask_first_order_assets,
        mask_negative_enpv_assets=cfg.mask_negative_enpv_assets,
        flatten_obs=flatten_obs,
        distributional_ptrs_config=cfg.distributional_ptrs,
        ta_experience_config=cfg.ta_experience,
        uncertain_ptrs_config=cfg.uncertain_ptrs,
        investment_levels_config=cfg.investment_levels,
        rd_capacity_config=cfg.rd_capacity,
        drop_action_config=cfg.drop_action,
        ptrs_readings_config=cfg.ptrs_readings,
        marketing_config=cfg.marketing,
        clinical_sites_config=cfg.clinical_sites,
        interim_trial_observations_config=cfg.interim_trial_observations,
        approval_phase_config=cfg.approval_phase,
        reward_type=ma.reward_type,
        reward_scale=ma.reward_scale,
        max_indications_per_ta=ma.max_indications_per_ta,
        target_drugs_per_indication=ma.target_drugs_per_indication,
        on_market_fraction=ma.on_market_fraction,
        indication_spread=ma.indication_spread,
        indication_drift_speed=ma.indication_drift_speed,
        trial_cost_multiplier=cfg.trial_cost_multiplier,
        congestion_exponent=ma.congestion_exponent,
        congestion_ramp_steps=ma.congestion_ramp_steps,
        congestion_incumbent_penalty=ma.congestion_incumbent_penalty,
        pricing_config=cfg.pricing,
        render_mode=None,
    )


def make_multi_agent_train_env(
    flatten_obs: bool = True,
    num_agents: int | None = None,
):
    """
    Create a multi-agent training environment from YAML configuration.

    Returns a PettingZoo ``MultiAgentInvestmentGameEnv`` (optionally
    warmup-wrapped) with a ``.train()`` method for creating a gym-like
    single-agent ``Trainer``.

    Usage::

        env = make_multi_agent_train_env()

        # Use as PettingZoo env directly
        obs, infos = env.reset(seed=42)

        # Or get a gym-like trainer
        trainer = env.train([None, "knapsack"])
        obs, info = trainer.reset()

    Parameters
    ----------
    flatten_obs : bool
        Whether to flatten observations into a single numpy array.
        Default ``True`` for faster processing. Set to ``False`` to get
        structured dict observations.
    num_agents : int | None
        Override the number of agents. If ``None``, uses the value from
        ``config.multi_agent.num_agents``.

    Returns
    -------
    MultiAgentInvestmentGameEnv
        The created multi-agent environment with ``.train()`` attached.

    """
    from types import MethodType

    from pyxis_portfolio_challenge.environment.competition import run, train
    from pyxis_portfolio_challenge.environment.multi_agent_training_gym import (
        MultiAgentInvestmentGameEnv,
    )

    cfg = config
    env_kwargs = _build_multi_agent_env_kwargs(flatten_obs, num_agents)
    env = MultiAgentInvestmentGameEnv(**env_kwargs)

    if cfg.warmup_on_reset_steps > 0:
        env = MultiAgentWarmupOnResetWrapper(
            env,
            warmup_steps=cfg.warmup_on_reset_steps,
            policy=cfg.warmup_on_reset_policy,
            verbose=False,
        )

    # Attach .train() and .run() methods so the env supports the Kaggle-like pattern:
    #   trainer = env.train([None, "knapsack"])
    #   reports, playthrough = env.run(["knapsack", "random"], seed=42)
    env.train = MethodType(train, env)
    env.run = MethodType(run, env)

    return env
