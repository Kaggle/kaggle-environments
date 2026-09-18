"""CLI for running multi-agent investment game matches."""

import importlib.util
import logging

import click
import yaml

from pyxis_portfolio_challenge.logging_utils import setup_logging

NUM_AGENTS = 2


def _load_agent_from_script(script_path, agent_name, **kwargs):
    """
    Load a multi-agent from a custom Python script.

    The script must define ``create_agent(agent_name, **kwargs)``
    returning a callable with an optional ``set_env`` method.
    """
    import upath

    path = upath.UPath(script_path)
    if not path.exists():
        raise FileNotFoundError(f"Agent script not found: {path}")

    spec = importlib.util.spec_from_file_location("custom_agent", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "create_agent"):
        return module.create_agent(agent_name, **kwargs)

    raise AttributeError(
        f"Script {script_path} must define a 'create_agent(agent_name, **kwargs)' "
        "function that returns a callable agent."
    )


def resolve_agent(spec, agent_name, **kwargs):
    """Resolve an agent spec (name or script path) to a callable."""
    from pyxis_portfolio_challenge.environment.competition import (
        NAMED_AGENTS,
    )

    factory = NAMED_AGENTS.get(spec.lower())
    if factory is not None:
        return factory(agent_name)
    return _load_agent_from_script(spec, agent_name, **kwargs)


def _run_simple(agents_dict, env_kwargs, seed, agent_labels=None):
    """Run a single episode without replay capture (no app dependency)."""
    from pyxis_portfolio_challenge.environment.multi_agent_training_gym import (
        MultiAgentInvestmentGameEnv,
    )

    env = MultiAgentInvestmentGameEnv(**env_kwargs)
    observations, _infos = env.reset(seed=seed)

    if agent_labels:
        env.multi_agent_game._display_names = agent_labels

    for agent in agents_dict.values():
        if callable(getattr(agent, "set_env", None)):
            agent.set_env(env)

    while env.agents:
        actions = {
            aid: agents_dict[aid](observations[aid]) for aid in env.agents
        }
        observations, _rewards, terminations, truncations, _infos = env.step(
            actions
        )
        for agent in agents_dict.values():
            if callable(getattr(agent, "set_env", None)):
                agent.set_env(env)
        if all(terminations.values()) or all(truncations.values()):
            break

    return env


def _run_with_replay(agents_dict, env_kwargs, seed, agent_names, agent_labels=None):
    """Run a single episode with replay capture via evaluate_multi_agent."""
    from pyxis_portfolio_challenge.environment.multi_agent_evaluate import (
        evaluate_multi_agent,
    )

    _per_agent_metrics, playthrough = evaluate_multi_agent(
        agents=agents_dict,
        worker_id=0,
        episodes_per_worker=1,
        env_kwargs=env_kwargs,
        capture_playthrough=True,
        agent_names=agent_names,
        agent_labels=agent_labels,
        seed=seed,
    )
    return playthrough


@click.command()
@click.argument("agents", nargs=-1, required=True)
@click.option(
    "--output",
    "-o",
    type=str,
    default=None,
    help="Path to write replay JSON file.",
)
@click.option(
    "--horizon", "-H", type=int, default=None, help="Override episode length."
)
@click.option("--seed", "-s", type=int, default=None, help="Random seed.")
@click.option(
    "--agent-kwargs",
    "agent_kwargs_list",
    type=str,
    multiple=True,
    help="Per-agent YAML kwargs (positional, repeatable).",
)
@click.option(
    "--names",
    "-n",
    type=str,
    multiple=True,
    help="Display names for agents in replay (repeatable).",
)
@click.option("--log-level", type=str, default="INFO", help="Logging level.")
def main(agents, output, horizon, seed, agent_kwargs_list, names, log_level):
    """
    Run a multi-agent investment game match.

    AGENTS: Exactly 2 agent specifications. Each is either a built-in name
    (knapsack, random, do_nothing) or a path to a custom agent
    script.

    \b
    Examples:
        pyxis knapsack random --seed 42
        pyxis ./my_bot.py knapsack -o replay.json
        pyxis knapsack knapsack --horizon 50
    """  # noqa: D301
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)

    if len(agents) != NUM_AGENTS:
        raise click.UsageError(
            f"Exactly {NUM_AGENTS} agents required, got {len(agents)}."
        )

    # Parse per-agent kwargs
    per_agent_kwargs = []
    for i in range(NUM_AGENTS):
        if i < len(agent_kwargs_list):
            per_agent_kwargs.append(yaml.safe_load(agent_kwargs_list[i]) or {})
        else:
            per_agent_kwargs.append({})

    # Build agent display names and labels
    agent_ids = [f"pharma_{i}" for i in range(NUM_AGENTS)]
    agent_names = {}
    agent_labels = {}  # agent_id -> "display_name (agent_type)" for logs
    for i, aid in enumerate(agent_ids):
        spec = agents[i]
        # Derive short agent type from spec (script filename or built-in name)
        if "/" in spec or spec.endswith(".py"):
            import os
            agent_type = os.path.basename(spec)
        else:
            agent_type = spec
        if i < len(names):
            agent_names[aid] = names[i]
            agent_labels[aid] = f"{names[i]} ({agent_type})"
        else:
            agent_names[aid] = aid
            agent_labels[aid] = f"{aid} ({agent_type})"

    # Resolve agents
    agents_dict = {}
    for i, (aid, spec) in enumerate(zip(agent_ids, agents)):
        logger.info(f"  {agent_labels[aid]}: {spec}")
        agents_dict[aid] = resolve_agent(spec, aid, **per_agent_kwargs[i])

    # Build env kwargs from config
    from pyxis_portfolio_challenge.environment.env_factory import (
        _build_multi_agent_env_kwargs,
    )

    env_kwargs = _build_multi_agent_env_kwargs(
        flatten_obs=True, num_agents=NUM_AGENTS
    )
    if horizon is not None:
        env_kwargs["horizon"] = horizon
        logger.info(f"Overriding horizon to {horizon}")

    p0, p1 = agent_labels[agent_ids[0]], agent_labels[agent_ids[1]]
    logger.info(f"Starting match: {p0} vs {p1}")

    if output:
        playthrough = _run_with_replay(
            agents_dict, env_kwargs, seed, agent_names, agent_labels
        )
        with open(output, "w") as f:
            f.write(playthrough.model_dump_json(indent=2))
        logger.info(f"Replay written to {output}")
    else:
        env = _run_simple(agents_dict, env_kwargs, seed, agent_labels)
        game = env.multi_agent_game
        logger.info("Match complete.")
        for aid in agent_ids:
            state = game.agent_states[aid]
            logger.info(
                f"  {agent_labels[aid]}: "
                f"cash={state.cash:,.0f}, "
                f"eNPV={state.enpv():,.0f}, "
                f"bankrupt={state.bankrupt}"
            )


if __name__ == "__main__":
    main()
