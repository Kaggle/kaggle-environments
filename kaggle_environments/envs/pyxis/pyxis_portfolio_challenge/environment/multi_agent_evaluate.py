"""Evaluation functions for multi-agent competitive environment."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable

from tqdm import tqdm

from pyxis_portfolio_challenge.app.endpoint_datamodels import bd_asset_to_response
from pyxis_portfolio_challenge.config import config, instantiate_from_config
from pyxis_portfolio_challenge.environment.metrics import (
    EvaluationMetric,
    MetricsContext,
    collect_metrics,
    merge_all_metrics,
    report_all_metrics,
)
from pyxis_portfolio_challenge.environment.multi_agent_training_gym import (
    MultiAgentInvestmentGameEnv,
)
from pyxis_portfolio_challenge.environment.playthrough import (
    PlaythroughData,
    StepRecord,
    build_playthrough_data,
    capture_actions,
    capture_agent_states,
    capture_shared_market,
)
from pyxis_portfolio_challenge.environment.warmup_wrapper import (
    MultiAgentWarmupOnResetWrapper,
)

logger = logging.getLogger(__name__)


def evaluate_multi_agent(
    agents: dict[str, Callable],
    worker_id: int,
    episodes_per_worker: int,
    env_kwargs: dict[str, Any],
    warmup_steps: int = 0,
    progress_queue=None,
    capture_playthrough: bool = False,
    agent_names: dict[str, str] | None = None,
    agent_labels: dict[str, str] | None = None,
    seed: int | None = None,
    seed_offset: int | None = None,
) -> tuple[
    dict[str, list[EvaluationMetric]],
    list[EvaluationMetric],
    PlaythroughData | None,
]:
    """
    Evaluate agents in the multi-agent environment (single worker).

    Args:
        agents: Dict mapping agent_id -> callable(obs) -> action
        worker_id: Worker index for seed calculation.
        episodes_per_worker: Number of episodes this worker evaluates.
        env_kwargs: Keyword arguments to construct MultiAgentInvestmentGameEnv.
        warmup_steps: Number of warmup steps after each reset.
        progress_queue: Optional queue for progress reporting.
        capture_playthrough: If True, capture a full playthrough for replay.
            Only valid with episodes_per_worker=1.
        agent_names: Optional mapping of agent_id -> display name for playthrough data.
        agent_labels: Optional display labels for agents.
        seed: Optional explicit seed. If provided, overrides the
            config-based seed calculation.
        seed_offset: Optional offset added to the computed seed, used to
            de-correlate the two halves of a symmetric evaluation.

    Returns:
        Tuple of:
        - Per-agent metrics: {agent_id: [EvaluationMetric, ...]}
        - PlaythroughData if capture_playthrough=True, else None

    """
    if capture_playthrough and episodes_per_worker > 1:
        raise ValueError(
            "Playthrough capture requires exactly 1 episode, "
            f"got episodes_per_worker={episodes_per_worker}"
        )
    cfg = config

    env = MultiAgentInvestmentGameEnv(**env_kwargs)

    if warmup_steps > 0:
        env = MultiAgentWarmupOnResetWrapper(
            env, warmup_steps=warmup_steps, verbose=False
        )

    # Create per-agent metric instances
    agent_metrics: dict[str, list[EvaluationMetric]] = {}
    for agent_id in env.possible_agents:
        agent_metrics[agent_id] = [
            instantiate_from_config(m) for m in cfg.evaluation_metrics
        ]

    # Begin evaluation
    for metrics_list in agent_metrics.values():
        collect_metrics("on_evaluation_begin", metrics=metrics_list, context=None)

    base_seed = cfg.eval_initial_seed
    for local_idx in range(episodes_per_worker):
        if seed is not None:
            episode_seed = seed
        elif seed_offset is not None:
            episode_seed = base_seed + seed_offset + local_idx
        else:
            global_episode_idx = worker_id * episodes_per_worker + local_idx
            episode_seed = base_seed + global_episode_idx

        observations, infos = env.reset(seed=episode_seed)

        if agent_labels:
            env.multi_agent_game._display_names = agent_labels

        # Capture initial state for playthrough
        playthrough_steps: list[StepRecord] = []
        if capture_playthrough:
            initial_agent_states = capture_agent_states(env.multi_agent_game)
            initial_shared_market = capture_shared_market(env.multi_agent_game)
            use_levels = (
                env.investment_levels_config is not None
                and env.investment_levels_config.enabled
            )

        # Attach env to agents that need it (e.g. for action masks)
        for agent_id, agent in agents.items():
            if callable(getattr(agent, "set_env", None)):
                agent.set_env(env)

        # Begin episode for per-agent metrics
        shared_market = env.multi_agent_game.shared_market
        all_states = env.agent_portfolios
        for agent_id in env.possible_agents:
            game_state = all_states[agent_id]
            ctx = MetricsContext(
                game_state=game_state,
                reward=0.0,
                shared_market_state=shared_market,
                agent_id=agent_id,
                all_agent_states=all_states,
            )
            collect_metrics(
                "on_episode_begin",
                metrics=agent_metrics[agent_id],
                context=ctx,
            )

        episode_rewards = {agent: 0.0 for agent in env.possible_agents}
        episode_steps = 0

        while env.agents:
            actions = {}
            for agent_id in env.agents:
                obs = observations[agent_id]
                actions[agent_id] = agents[agent_id](obs)

            # Capture actions before stepping (for playthrough)
            if capture_playthrough:
                pre_step_bd_assets = [
                    bd_asset_to_response(
                        a,
                        env.multi_agent_game.shared_market.indication_name_map,
                        env.reinvestment_percentage,
                        ptrs_cfg=None,
                        clone=None,
                    )
                    for a in env.multi_agent_game.shared_market.current_bd_assets
                ]
                captured_actions = capture_actions(
                    actions,
                    env._asset_id_orders,
                    use_levels,
                    pre_step_bd_assets=pre_step_bd_assets,
                    bd_bid_decoder=env._decode_bd_bids,
                    env=env,
                )

            # Fire on_step_begin with investment decisions before env steps
            pre_step_portfolios = env.agent_portfolios
            pre_step_shared = env.multi_agent_game.shared_market
            for agent_id in env.possible_agents:
                agent_action = actions.get(agent_id, {})
                inv_array = (
                    agent_action.get("investments", [])
                    if isinstance(agent_action, dict)
                    else []
                )
                asset_order = env._asset_id_orders.get(agent_id, [])
                investment_decisions = {
                    asset_id: "invest"
                    for idx, asset_id in enumerate(asset_order)
                    if asset_id is not None and idx < len(inv_array) and inv_array[idx]
                }
                ctx = MetricsContext(
                    game_state=pre_step_portfolios[agent_id],
                    reward=0.0,
                    shared_market_state=pre_step_shared,
                    agent_id=agent_id,
                    all_agent_states=pre_step_portfolios,
                    investment_decisions=investment_decisions,
                )
                collect_metrics(
                    "on_step_begin",
                    metrics=agent_metrics[agent_id],
                    context=ctx,
                )

            observations, rewards, terminations, truncations, infos = env.step(actions)

            for agent_id, agent in agents.items():
                if callable(getattr(agent, "set_env", None)):
                    agent.set_env(env)

            for agent_id, reward in rewards.items():
                episode_rewards[agent_id] += reward

            episode_steps += 1

            # Capture post-step state for playthrough
            if capture_playthrough:
                playthrough_steps.append(
                    StepRecord(
                        step=episode_steps,
                        actions=captured_actions,
                        agent_states=capture_agent_states(env.multi_agent_game),
                        shared_market=capture_shared_market(env.multi_agent_game),
                        rewards={aid: float(r) for aid, r in rewards.items()},
                        cumulative_rewards={
                            aid: float(r) for aid, r in episode_rewards.items()
                        },
                    )
                )

            # Per-agent step metrics
            shared_market = env.multi_agent_game.shared_market
            all_states = env.agent_portfolios
            for agent_id in env.possible_agents:
                game_state = all_states[agent_id]
                # Extract BD bid levels from raw actions
                agent_action = actions.get(agent_id, {})
                bd_bid_levels = None
                if isinstance(agent_action, dict) and "bd_bids" in agent_action:
                    import numpy as np

                    raw = np.asarray(agent_action["bd_bids"])
                    if raw.dtype in (np.int64, np.int32, np.int8):
                        bd_bid_levels = raw.tolist()
                ctx = MetricsContext(
                    game_state=game_state,
                    reward=rewards.get(agent_id, 0.0),
                    shared_market_state=shared_market,
                    agent_id=agent_id,
                    all_agent_states=all_states,
                    bd_bid_levels=bd_bid_levels,
                )
                collect_metrics(
                    "on_step_end",
                    metrics=agent_metrics[agent_id],
                    context=ctx,
                )

            if all(terminations.values()) or all(truncations.values()):
                break

        # End episode for per-agent metrics
        shared_market = env.multi_agent_game.shared_market
        all_states = env.agent_portfolios
        for agent_id in env.possible_agents:
            game_state = all_states[agent_id]
            ctx = MetricsContext(
                game_state=game_state,
                reward=episode_rewards[agent_id],
                shared_market_state=shared_market,
                agent_id=agent_id,
                all_agent_states=all_states,
                all_agent_rewards=episode_rewards,
            )
            collect_metrics(
                "on_episode_end",
                metrics=agent_metrics[agent_id],
                context=ctx,
            )

        if progress_queue is not None:
            progress_queue.put(1)

    # End evaluation
    for metrics_list in agent_metrics.values():
        collect_metrics("on_evaluation_end", metrics=metrics_list, context=None)

    # Build playthrough data if capturing
    playthrough: PlaythroughData | None = None
    if capture_playthrough:
        playthrough = build_playthrough_data(
            env=env,
            seed=seed,
            initial_agent_states=initial_agent_states,
            initial_shared_market=initial_shared_market,
            steps=playthrough_steps,
            agent_names=agent_names,
        )

    return agent_metrics, playthrough


def _parallel_evaluate_raw(
    agents: dict[str, Callable],
    num_workers: int,
    env_kwargs: dict[str, Any],
    warmup_steps: int = 0,
    num_episodes: int | None = None,
) -> dict[str, list[EvaluationMetric]]:
    """
    Evaluate agents in parallel and return raw (unformatted) metric objects.

    This is the internal workhorse used by both
    ``parallel_evaluate_multi_agent`` and the symmetric evaluation path in
    ``competition.evaluate()``.

    Returns:
        Per-agent raw metrics: {agent_id: [EvaluationMetric, ...]}

    """
    cfg = config
    total_episodes = num_episodes if num_episodes is not None else cfg.num_eval_episodes
    # Don't spawn more workers than episodes
    num_workers = min(num_workers, total_episodes)

    if num_workers == 1:
        agent_metrics, _ = evaluate_multi_agent(
            agents,
            worker_id=0,
            episodes_per_worker=total_episodes,
            env_kwargs=env_kwargs,
            warmup_steps=warmup_steps,
        )
        return agent_metrics

    # Distribute episodes: base count per worker, remainder goes to the last worker
    base, remainder = divmod(total_episodes, num_workers)
    counts = [base] * num_workers
    counts[-1] += remainder
    offsets = []
    cumulative = 0
    for c in counts:
        offsets.append(cumulative)
        cumulative += c

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(
                evaluate_multi_agent,
                agents,
                worker_id=i,
                episodes_per_worker=counts[i],
                env_kwargs=env_kwargs,
                warmup_steps=warmup_steps,
                seed_offset=offsets[i],
            )
            for i in range(num_workers)
        ]

        all_agent_metrics: dict[str, list[list[EvaluationMetric]]] = {}

        with tqdm(total=num_workers, desc="Workers", unit="worker") as pbar:
            for future in as_completed(futures):
                agent_metrics, _ = future.result()
                for agent_id, metrics in agent_metrics.items():
                    if agent_id not in all_agent_metrics:
                        all_agent_metrics[agent_id] = []
                    all_agent_metrics[agent_id].append(metrics)
                pbar.update(1)

    # Merge per-agent metrics across workers
    merged_metrics = {}
    for agent_id, metrics_lists in all_agent_metrics.items():
        merged_metrics[agent_id] = merge_all_metrics(metrics_lists)

    return merged_metrics


def parallel_evaluate_multi_agent(
    agents: dict[str, Callable],
    num_workers: int,
    env_kwargs: dict[str, Any],
    warmup_steps: int = 0,
    capture_playthrough: bool = False,
    agent_names: dict[str, str] | None = None,
    num_episodes: int | None = None,
) -> tuple[dict[str, dict[str, Any]], dict | None]:
    """
    Evaluate agents in parallel using multiple worker processes.

    Args:
        agents: Dict mapping agent_id -> callable(obs) -> action
        num_workers: Number of parallel workers.
        env_kwargs: Keyword arguments to construct MultiAgentInvestmentGameEnv.
        warmup_steps: Number of warmup steps after each reset.
        capture_playthrough: If True, capture a full playthrough for replay.
            Requires num_eval_episodes=1 and num_workers=1.
        agent_names: Optional mapping of agent_id -> display name for playthrough data.
        num_episodes: Total number of evaluation episodes. If None, uses
            config.num_eval_episodes.

    Returns:
        Tuple of:
        - Per-agent reported metrics: {agent_id: {metric_name: value}}
        - Playthrough dict (JSON-serializable) if capture_playthrough=True, else None

    """
    cfg = config
    total_episodes = num_episodes if num_episodes is not None else cfg.num_eval_episodes

    if capture_playthrough:
        if total_episodes > 1:
            raise ValueError(
                "Playthrough capture requires num_eval_episodes=1, "
                f"got {total_episodes}"
            )
        if num_workers > 1:
            raise ValueError(
                f"Playthrough capture requires num_workers=1, got {num_workers}"
            )
        # Playthrough path: must use single worker with full API
        agent_metrics, playthrough = evaluate_multi_agent(
            agents,
            worker_id=0,
            episodes_per_worker=total_episodes,
            env_kwargs=env_kwargs,
            warmup_steps=warmup_steps,
            capture_playthrough=True,
            agent_names=agent_names,
        )
        per_agent_reports = {}
        for agent_id, metrics in agent_metrics.items():
            per_agent_reports[agent_id] = report_all_metrics(metrics)
        playthrough_dict = playthrough.model_dump(mode="json") if playthrough else None
        return per_agent_reports, playthrough_dict

    # Standard path: delegate to _parallel_evaluate_raw and report
    raw_metrics = _parallel_evaluate_raw(
        agents=agents,
        num_workers=num_workers,
        env_kwargs=env_kwargs,
        warmup_steps=warmup_steps,
        num_episodes=num_episodes,
    )
    per_agent_reports = {}
    for agent_id, metrics in raw_metrics.items():
        per_agent_reports[agent_id] = report_all_metrics(metrics)

    return per_agent_reports, None
