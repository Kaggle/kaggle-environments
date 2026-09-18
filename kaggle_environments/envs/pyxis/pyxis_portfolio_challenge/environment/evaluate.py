from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable

from tqdm import tqdm

from pyxis_portfolio_challenge.config import config, instantiate_from_config
from pyxis_portfolio_challenge.environment.metrics import (
    EvaluationMetric,
    collect_metrics,
    merge_all_metrics,
    report_all_metrics,
)
from pyxis_portfolio_challenge.environment.training_gym import InvestmentGameEnv
from pyxis_portfolio_challenge.environment.warmup_wrapper import WarmupOnResetWrapper


def parallel_evaluate(agent, num_workers: int, **evaluate_kwargs) -> dict[str, Any]:
    """
    Evaluate the given agent in parallel using multiple worker processes.

    Args:
        agent (Callable): The agent to evaluate. It should be callable,
        num_workers (int): The number of parallel worker processes to use.
            If num_workers=1, runs in main process to avoid pickling issues.
        **evaluate_kwargs: Additional keyword arguments to pass to the
         evaluate function.

    """
    cfg = config
    episodes_per_worker = cfg.num_eval_episodes // num_workers

    # Single-worker mode: run in main process to avoid pickling issues
    # This is useful for agents that can't be pickled (e.g., models with closures)
    if num_workers == 1:
        metrics = evaluate(
            agent,
            worker_id=0,
            episodes_per_worker=cfg.num_eval_episodes,
            progress_queue=None,
            **evaluate_kwargs,
        )
        return report_all_metrics(metrics)

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(
                evaluate,
                agent,
                worker_id=i,
                episodes_per_worker=episodes_per_worker,
                progress_queue=None,  # Removed queue - not needed
                **evaluate_kwargs,
            )
            for i in range(num_workers)
        ]

        # Collect results as they complete
        all_metrics = []
        with tqdm(total=num_workers, desc="Workers", unit="worker") as pbar:
            for future in as_completed(futures):
                all_metrics.append(future.result())
                pbar.update(1)

    merged = merge_all_metrics(all_metrics)
    return report_all_metrics(merged)


def evaluate(
    agent: Callable,
    worker_id: int,
    episodes_per_worker: int,
    progress_queue=None,
    flatten_obs=None,
    mask_first_order_assets=None,
    mask_negative_enpv_assets=None,
    warmup_on_reset_steps=None,
    warmup_on_reset_policy=None,
    distributional_ptrs_config=None,
) -> list[EvaluationMetric]:
    """
    Evaluate the given agent in the investment game environment.

    Args:
        agent (Callable): The agent to evaluate. It should be callable,
        worker_id (int): The ID of the worker process.
        episodes_per_worker (int): Number of episodes to evaluate per worker.
        progress_queue (multiprocessing.Queue): Queue to report progress.
        flatten_obs (bool): Whether to flatten the observation.
        mask_first_order_assets (bool): Whether to mask the first order assets.
        mask_negative_enpv_assets (bool): Whether to mask negative eNPV assets.
        warmup_on_reset_steps (int): Number of warmup steps after each reset.
        warmup_on_reset_policy (str): Warmup policy ("do_nothing" or "random").
        distributional_ptrs_config: Configuration for distributional PTRS feature.

    Returns:
        list[EvaluationMetric]: A list of evaluation metrics collected during
         evaluation.

    """
    cfg = config

    metrics = [instantiate_from_config(metric) for metric in cfg.evaluation_metrics]

    collect_metrics(collection_fn="on_evaluation_begin", metrics=metrics, context=None)

    mask_first_order_assets = (
        mask_first_order_assets
        if mask_first_order_assets is not None
        else cfg.mask_first_order_assets
    )
    mask_negative_enpv_assets = (
        mask_negative_enpv_assets
        if mask_negative_enpv_assets is not None
        else cfg.mask_negative_enpv_assets
    )
    flatten_obs = flatten_obs if flatten_obs is not None else cfg.flatten_obs
    warmup_on_reset_steps = (
        warmup_on_reset_steps
        if warmup_on_reset_steps is not None
        else cfg.warmup_on_reset_steps
    )
    warmup_on_reset_policy = (
        warmup_on_reset_policy
        if warmup_on_reset_policy is not None
        else cfg.warmup_on_reset_policy
    )
    distributional_ptrs_config = (
        distributional_ptrs_config
        if distributional_ptrs_config is not None
        else cfg.distributional_ptrs
    )

    env = InvestmentGameEnv(
        equilibrium_num_assets=cfg.equilibrium_num_assets,
        max_num_assets=cfg.max_num_assets,
        asset_arrival_sensitivity_below=cfg.asset_arrival_sensitivity_below,
        asset_arrival_sensitivity_above=cfg.asset_arrival_sensitivity_above,
        starting_cash=cfg.starting_cash,
        horizon=cfg.horizon,
        reward_fn=instantiate_from_config(cfg.reward_fn),
        assets_dir=cfg.evaluation_data_dir,
        reinvestment_percentage=cfg.reinvestment_percentage,
        shuffle_order=False,  # fixed to false for eval
        flatten_obs=flatten_obs,
        metrics=metrics,
        mask_first_order_assets=mask_first_order_assets,
        mask_negative_enpv_assets=mask_negative_enpv_assets,
        uncertain_ptrs_config=cfg.uncertain_ptrs,
        investment_levels_config=cfg.investment_levels,
        interim_trial_observations_config=cfg.interim_trial_observations,
        distributional_ptrs_config=distributional_ptrs_config,
        ta_experience_config=cfg.ta_experience,
        rd_capacity_config=cfg.rd_capacity,
        drop_action_config=cfg.drop_action,
        marketing_config=cfg.marketing,
        clinical_sites_config=cfg.clinical_sites,
        ptrs_readings_config=cfg.ptrs_readings,
        approval_phase_config=cfg.approval_phase,
        initial_game_state=None,
    )

    # Apply warmup wrapper if enabled
    if warmup_on_reset_steps > 0:
        env = WarmupOnResetWrapper(
            env,
            warmup_steps=warmup_on_reset_steps,
            policy=warmup_on_reset_policy,
            verbose=False,
        )

    # Neded for compatibility with LegacyPyxieAgent
    if callable(getattr(agent, "set_env", None)):
        agent.set_env(env)

    base_seed = cfg.eval_initial_seed
    for local_idx in range(episodes_per_worker):
        global_episode_idx = worker_id * episodes_per_worker + local_idx
        seed = base_seed + global_episode_idx
        obs, _ = env.reset(seed=seed)
        # Neded for compatibility with LegacyPyxieAgent
        if callable(getattr(agent, "set_env", None)):
            agent.set_env(env)

        terminated = False
        while not terminated:
            action = agent(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            # Neded for compatibility with LegacyPyxieAgent
            if callable(getattr(agent, "set_env", None)):
                agent.set_env(env)

        # Report a single step to the parent
        if progress_queue is not None:
            progress_queue.put(1)

    collect_metrics(collection_fn="on_evaluation_end", metrics=metrics, context=None)
    return metrics
