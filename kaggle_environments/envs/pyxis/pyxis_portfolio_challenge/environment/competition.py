"""
Competition API for multi-agent environment.

Adds a ``.train()`` method to the multi-agent env and provides a standalone
``evaluate()`` function

Example usage::

    from pyxis_portfolio_challenge.environment import make_multi_agent_train_env
    from pyxis_portfolio_challenge.environment.competition import evaluate

    # Create the PettingZoo env
    env = make_multi_agent_train_env()

    # Use as a PettingZoo env directly
    obs, infos = env.reset(seed=42)
    obs, rewards, terms, truncs, infos = env.step(actions)

    # Or get a gym-like trainer (like env.train())
    trainer = env.train([None, "knapsack"])
    obs, info = trainer.reset()
    obs, reward, term, trunc, info = trainer.step(action)
    masks = trainer.action_masks()

    # Evaluate agents
    results = evaluate(
        agents=[my_agent, "knapsack"],
        num_episodes=100,
        flat_obs={0: True},  # my_agent at index 0 expects flat obs
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from pyxis_portfolio_challenge.environment.playthrough import PlaythroughData

import gymnasium as gym
import numpy as np

from pyxis_portfolio_challenge.config import config
from pyxis_portfolio_challenge.environment.env_factory import (
    _build_multi_agent_env_kwargs,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named agent registry
# ---------------------------------------------------------------------------

# Whether each named agent expects flat (True) or dict (False) observations.
NAMED_AGENT_FLAT_OBS: dict[str, bool] = {
    "knapsack": False,
    "random": False,
    "do_nothing": False,
}


def _make_knapsack_agent(agent_name: str):
    """
    Create a MultiAgentKnapsackAgent for competition use.

    Uncapped (``capacity=None``): concurrent-trial throughput is limited
    naturally by the clinical-sites feature, so the agent no longer
    imposes its own hard capacity cap.
    """
    from pyxis_portfolio_challenge.agents.multi_agent_knapsack import (
        MultiAgentKnapsackAgent,
    )

    return MultiAgentKnapsackAgent(
        agent_name=agent_name,
        capacity=None,
        enable_bd_bidding=True,
    )


def _make_random_agent(agent_name: str):
    """Create a random agent for the multi-agent environment."""
    from pyxis_portfolio_challenge.agents.multi_agent_random import (
        MultiAgentRandomAgent,
    )

    return MultiAgentRandomAgent(agent_name=agent_name)


def _make_do_nothing_agent(agent_name: str):
    """Create a do-nothing agent for the multi-agent environment."""
    from pyxis_portfolio_challenge.agents.multi_agent_do_nothing import (
        MultiAgentDoNothingAgent,
    )

    return MultiAgentDoNothingAgent(agent_name=agent_name)


# Map of named agent strings to factory functions
NAMED_AGENTS: dict[str, Callable[[str], Any]] = {
    "knapsack": _make_knapsack_agent,
    "random": _make_random_agent,
    "do_nothing": _make_do_nothing_agent,
}


def _resolve_agent(agent_spec, agent_name: str):
    """
    Resolve an agent specification to a callable.

    Parameters
    ----------
    agent_spec : str | Callable | None
        Agent specification. A string is looked up in ``NAMED_AGENTS``,
        a callable is returned as-is, and ``None`` marks the trainee slot.
    agent_name : str
        The PettingZoo agent identifier (e.g. ``"pharma_0"``).

    Returns
    -------
    Callable
        The resolved agent callable.

    Raises
    ------
    ValueError
        If the string is not a recognised named agent.

    """
    if isinstance(agent_spec, str):
        factory = NAMED_AGENTS.get(agent_spec)
        if factory is None:
            raise ValueError(
                f"Unknown agent '{agent_spec}'. "
                f"Available: {sorted(NAMED_AGENTS.keys())}"
            )
        return factory(agent_name)
    return agent_spec


# ---------------------------------------------------------------------------
# Observation format helpers
# ---------------------------------------------------------------------------


class _FlatObsAgentWrapper:
    """Wraps an agent so it receives flattened obs from a dict-obs env."""

    def __init__(self, agent, env):
        self._agent = agent
        self._env = env

    def set_env(self, env):
        """Update env reference on both wrapper and inner agent."""
        self._env = env
        if callable(getattr(self._agent, "set_env", None)):
            self._agent.set_env(env)

    def __call__(self, obs):
        """Flatten dict obs then forward to inner agent."""
        if isinstance(obs, np.ndarray):
            return self._agent(obs)
        flat_obs = self._env.flatten_dict_obs(obs)
        return self._agent(flat_obs)

    def __getattr__(self, name):
        """Proxy attribute access to inner agent."""
        if name in ("_agent", "_env"):
            raise AttributeError(name)
        return getattr(self._agent, name)


class _DeferredFlatObsWrapper:
    """Wraps an agent to receive flat obs; binds env on first set_env call."""

    def __init__(self, agent):
        self._agent = agent
        self._env = None

    def set_env(self, env):
        """Bind env for flatten_dict_obs and forward to inner agent."""
        self._env = env
        if callable(getattr(self._agent, "set_env", None)):
            self._agent.set_env(env)

    def __call__(self, obs):
        """Flatten dict obs then forward to inner agent."""
        if isinstance(obs, np.ndarray):
            return self._agent(obs)
        flat_obs = self._env.flatten_dict_obs(obs)
        return self._agent(flat_obs)

    def __getattr__(self, name):
        """Proxy attribute access to inner agent."""
        if name in ("_agent", "_env"):
            raise AttributeError(name)
        return getattr(self._agent, name)


def _needs_flat_obs(agent_spec, index: int, flat_obs: dict[int, bool] | None) -> bool:
    """Determine whether an agent needs flat observations."""
    if isinstance(agent_spec, str):
        return NAMED_AGENT_FLAT_OBS.get(agent_spec, False)
    if flat_obs is not None and index in flat_obs:
        return flat_obs[index]
    return getattr(agent_spec, "flat_obs", False)


def _validate_flat_obs_overrides(
    agents: list,
    flat_obs: dict[int, bool] | None,
) -> None:
    """Raise if the user tries to override obs format for a named agent."""
    if flat_obs is None:
        return
    for idx, spec in enumerate(agents):
        if isinstance(spec, str) and idx in flat_obs:
            raise ValueError(
                f"Cannot override flat_obs for named agent '{spec}' "
                f"at index {idx}. Named agents have a fixed observation "
                f"format (flat_obs={NAMED_AGENT_FLAT_OBS[spec]})."
            )


def _wrap_agents_for_obs_format(
    agents_list: list,
    resolved: dict[str, Any],
    possible_agents: list[str],
    flat_obs: dict[int, bool] | None,
    env,
) -> dict[str, Any]:
    """Wrap resolved agents that need flat obs with _FlatObsAgentWrapper."""
    wrapped = {}
    for i, agent_name in enumerate(possible_agents):
        if agent_name not in resolved:
            continue
        agent = resolved[agent_name]
        if _needs_flat_obs(agents_list[i], i, flat_obs):
            wrapped[agent_name] = _FlatObsAgentWrapper(agent, env)
        else:
            wrapped[agent_name] = agent
    return wrapped


# ---------------------------------------------------------------------------
# Trainer — single-agent gym.Env wrapping the multi-agent env
# ---------------------------------------------------------------------------


class Trainer(gym.Env):
    """
    Single-agent Gymnasium wrapper around the multi-agent PettingZoo env.

    Handles opponent actions automatically so the trainee sees a standard
    ``reset() → step() → action_masks()`` loop. Used for evaluation/competition
    with dict-action agents; RL self-play training uses
    :class:`~pyxis_portfolio_challenge.environment.self_play.SelfPlayWrapper`.

    The action space is a ``Dict`` matching the multi-agent env:
    ``{"investments": MultiDiscrete/MultiBinary, "bd_bids": Box}``, where BD
    bids are continuous raw-cash amounts (GBP millions) rather than discrete
    levels.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        multi_env,
        trainee_index: int,
        opponents: dict,
        flatten_trainee_obs: bool = False,
    ):
        """
        Initialise the Trainer wrapper.

        Parameters
        ----------
        multi_env : MultiAgentInvestmentGameEnv
            The PettingZoo parallel env (possibly warmup-wrapped).
        trainee_index : int
            Index into ``multi_env.possible_agents`` for the trainee.
        opponents : dict[str, Callable]
            Mapping of opponent agent names to their callables.
            Opponents that need flat obs should already be wrapped.
        flatten_trainee_obs : bool
            If True, the trainee receives flattened observations.

        """
        super().__init__()
        self.env = multi_env
        self.trainee = multi_env.possible_agents[trainee_index]
        self.opponents = opponents
        self._flatten_trainee_obs = flatten_trainee_obs

        self.observation_space = multi_env.observation_space(self.trainee)
        self.action_space = multi_env.action_space(self.trainee)

        self._current_obs: dict = {}

    def _trainee_obs(self, obs_dict: dict):
        """Extract trainee obs, flattening if configured."""
        obs = obs_dict[self.trainee]
        if self._flatten_trainee_obs:
            obs = self.env.flatten_dict_obs(obs)
        return obs

    def reset(self, seed=None, options=None):
        """Reset the environment and return the trainee's observation."""
        obs_dict, info_dict = self.env.reset(seed=seed)
        self._current_obs = obs_dict

        for opp in self.opponents.values():
            if callable(getattr(opp, "set_env", None)):
                opp.set_env(self.env)

        return self._trainee_obs(obs_dict), info_dict.get(self.trainee, {})

    def step(self, action):
        """Step the environment with the trainee's action and opponent actions."""
        actions = {self.trainee: action}

        for name, opp in self.opponents.items():
            actions[name] = opp(self._current_obs[name])

        obs_dict, rewards, terms, truncs, infos = self.env.step(actions)
        self._current_obs = obs_dict

        # Re-attach env after step (opponents may need updated state)
        for opp in self.opponents.values():
            if callable(getattr(opp, "set_env", None)):
                opp.set_env(self.env)

        return (
            self._trainee_obs(obs_dict),
            rewards[self.trainee],
            terms[self.trainee],
            truncs[self.trainee],
            infos.get(self.trainee, {}),
        )

    def action_masks(self):
        """Return action masks for the trainee agent."""
        return self.env.action_masks(self.trainee)

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        return self.env.close()


# ---------------------------------------------------------------------------
# train() — attached to the env by make_multi_agent_train_env
# ---------------------------------------------------------------------------


def train(env, agents: list, flat_obs: dict[int, bool] | None = None) -> Trainer:
    """
    Create a single-agent ``Trainer`` for SB3-compatible training.

    This function is attached to the env returned by
    ``make_multi_agent_train_env()`` as a bound method.

    Parameters
    ----------
    env : MultiAgentInvestmentGameEnv
        The PettingZoo parallel env.
    agents : list
        List of agent specifications, one per player slot. Use ``None``
        for the trainee and strings or callables for opponents.
        Exactly one ``None`` is required.

        Named agents: ``"knapsack"``, ``"random"``,
        ``"do_nothing"``.
    flat_obs : dict[int, bool] | None
        Mapping of agent index to whether it needs flat observations.
        The trainee index controls what format ``Trainer.reset()`` and
        ``Trainer.step()`` return.  Opponent indices control what each
        opponent callable receives.  Named agents already have a fixed
        format and cannot be overridden.

    Returns
    -------
    Trainer
        A ``gym.Env`` wrapping the multi-agent env as single-agent.

    """
    possible = env.possible_agents
    if len(agents) != len(possible):
        raise ValueError(
            f"Expected {len(possible)} agents, got {len(agents)}. "
            f"Agent slots: {possible}"
        )

    none_indices = [i for i, a in enumerate(agents) if a is None]
    if len(none_indices) != 1:
        raise ValueError(
            "Exactly one None (trainee slot) is required. "
            f"Got {len(none_indices)}."
        )

    _validate_flat_obs_overrides(agents, flat_obs)

    trainee_index = none_indices[0]
    flatten_trainee = False
    if flat_obs is not None and trainee_index in flat_obs:
        flatten_trainee = flat_obs[trainee_index]

    # Resolve and wrap opponents
    resolved = {}
    for i, spec in enumerate(agents):
        if spec is None:
            continue
        agent_name = possible[i]
        resolved[agent_name] = _resolve_agent(spec, agent_name)

    opponents = _wrap_agents_for_obs_format(
        agents, resolved, possible, flat_obs, env,
    )

    return Trainer(
        env, trainee_index, opponents,
        flatten_trainee_obs=flatten_trainee,
    )


# ---------------------------------------------------------------------------
# evaluate() — standalone function
# ---------------------------------------------------------------------------


def evaluate(
    agents: list,
    num_episodes: int | None = None,
    num_workers: int = 1,
    num_agents: int | None = None,
    capture_playthrough: bool = False,
    flat_obs: dict[int, bool] | None = None,
    symmetric: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict | None]:
    """
    Evaluate agents in the multi-agent environment.

    Creates the environment from config and runs evaluation episodes.

    When ``symmetric=True`` (default), each seed is played twice with
    agent positions swapped to control for positional asymmetry.
    Results are keyed by original agent identity (``agent_0``,
    ``agent_1``) rather than seat position.

    Parameters
    ----------
    agents : list
        List of agent specifications (strings or callables), one per
        player slot. No ``None`` slots — all agents must be specified.
    num_episodes : int | None
        Number of unique seeds to evaluate. If ``None``, uses
        ``config.num_eval_episodes``. When ``symmetric=True``, the
        actual number of episodes played is ``2 * num_episodes``.
    num_workers : int
        Number of parallel workers. Default 1.
    num_agents : int | None
        Override the number of agents. If ``None``, uses config value.
    capture_playthrough : bool
        If True, capture a full playthrough for replay (requires
        ``num_episodes=1`` and ``num_workers=1``). Not supported with
        ``symmetric=True``.
    flat_obs : dict[int, bool] | None
        Mapping of agent index to whether it needs flat observations.
        Named agents already have a fixed format and cannot be
        overridden.  User agents without an entry here (and without a
        ``flat_obs`` attribute) receive dict observations.
    symmetric : bool
        If True (default), run seed-symmetric evaluation: each seed is
        played with both agent orderings. Results are reported under
        ``agent_0`` / ``agent_1`` keys (original identity, not seat).

    Returns
    -------
    tuple
        ``(per_agent_reports, playthrough_dict)``
        where per_agent_reports maps agent key to metric dicts.

    """
    from pyxis_portfolio_challenge.environment.metrics import (
        merge_all_metrics,
        report_all_metrics,
    )
    from pyxis_portfolio_challenge.environment.multi_agent_evaluate import (
        _parallel_evaluate_raw,
        parallel_evaluate_multi_agent,
    )

    # Build env kwargs to determine possible_agents
    env_kwargs = _build_multi_agent_env_kwargs(
        flatten_obs=False,
        num_agents=num_agents,
        assets_dir=config.evaluation_data_dir,
        bd_assets_dir=config.multi_agent.bd_eval_assets_dir,
    )
    n = env_kwargs["num_agents"]
    possible = [f"pharma_{i}" for i in range(n)]

    if len(agents) != len(possible):
        raise ValueError(
            f"Expected {len(possible)} agents, got {len(agents)}. "
            f"Agent slots: {possible}"
        )

    if any(a is None for a in agents):
        raise ValueError(
            "All agent slots must be specified for evaluation. "
            "Use env.train() for training with a trainee slot."
        )

    _validate_flat_obs_overrides(agents, flat_obs)

    if symmetric and capture_playthrough:
        raise ValueError(
            "Playthrough capture is not supported with symmetric=True."
        )

    # Non-symmetric path: original behaviour
    if not symmetric:
        resolved = {}
        for i, spec in enumerate(agents):
            agent_name = possible[i]
            resolved[agent_name] = _resolve_agent(spec, agent_name)

        agents_with_format = {}
        for i, agent_name in enumerate(possible):
            agent = resolved[agent_name]
            if _needs_flat_obs(agents[i], i, flat_obs):
                agents_with_format[agent_name] = _DeferredFlatObsWrapper(agent)
            else:
                agents_with_format[agent_name] = agent

        cfg = config
        warmup_steps = cfg.warmup_on_reset_steps
        return parallel_evaluate_multi_agent(
            agents=agents_with_format,
            num_workers=num_workers,
            env_kwargs=env_kwargs,
            warmup_steps=warmup_steps,
            capture_playthrough=capture_playthrough,
            num_episodes=num_episodes,
        )

    # --- Symmetric evaluation ---
    cfg = config
    warmup_steps = cfg.warmup_on_reset_steps

    # Resolve agents for both orderings.
    # Each ordering needs fresh agent instances (they carry env state).
    def _build_agents_dict(agent_specs, position_order):
        """Build {pharma_id: wrapped_agent} for a given position order."""
        resolved = {}
        for pos_idx, agent_idx in enumerate(position_order):
            agent_name = possible[pos_idx]
            resolved[agent_name] = _resolve_agent(
                agent_specs[agent_idx], agent_name,
            )
        agents_fmt = {}
        for pos_idx, agent_idx in enumerate(position_order):
            agent_name = possible[pos_idx]
            agent = resolved[agent_name]
            if _needs_flat_obs(agent_specs[agent_idx], agent_idx, flat_obs):
                agents_fmt[agent_name] = _DeferredFlatObsWrapper(agent)
            else:
                agents_fmt[agent_name] = agent
        return agents_fmt

    # Run 1: original order [0, 1]
    agents_original = _build_agents_dict(agents, [0, 1])
    raw_original = _parallel_evaluate_raw(
        agents=agents_original,
        num_workers=num_workers,
        env_kwargs=env_kwargs,
        warmup_steps=warmup_steps,
        num_episodes=num_episodes,
    )

    # Run 2: swapped order [1, 0]
    agents_swapped = _build_agents_dict(agents, [1, 0])
    raw_swapped = _parallel_evaluate_raw(
        agents=agents_swapped,
        num_workers=num_workers,
        env_kwargs=env_kwargs,
        warmup_steps=warmup_steps,
        num_episodes=num_episodes,
    )

    # Remap swapped results: pharma_0 in run2 was agent_1, pharma_1 was agent_0
    # Merge under canonical keys: agent_0, agent_1
    agent_keys = [f"agent_{i}" for i in range(n)]

    # original: pharma_0 -> agent_0, pharma_1 -> agent_1
    # swapped:  pharma_0 -> agent_1, pharma_1 -> agent_0
    per_agent_reports = {}
    for agent_idx in range(n):
        agent_key = agent_keys[agent_idx]
        # From original run: agent_idx sat in pharma_{agent_idx}
        original_metrics = raw_original[possible[agent_idx]]
        # From swapped run: agent_idx sat in pharma_{1-agent_idx} (for 2 agents)
        swapped_pos = possible[1 - agent_idx]
        swapped_metrics = raw_swapped[swapped_pos]
        # Merge the two sets of metrics
        merged = merge_all_metrics([original_metrics, swapped_metrics])
        per_agent_reports[agent_key] = report_all_metrics(merged)

    return per_agent_reports, None


def run(
    env,
    agents: list,
    seed: int | None = None,
    flat_obs: dict[int, bool] | None = None,
) -> tuple[dict[str, dict[str, Any]], PlaythroughData]:
    """
    Run two agents head-to-head for a single episode.

    Convenience wrapper around ``evaluate_multi_agent`` that always
    captures a playthrough and runs exactly one episode.

    Parameters
    ----------
    env : MultiAgentInvestmentGameEnv
        The multi-agent environment (ignored internally — a fresh env
        is created from config, but the method is bound to the env for
        a consistent API).
    agents : list
        List of agent specifications (strings or callables), one per
        player slot. No ``None`` slots.
    seed : int | None
        Random seed for the episode. If ``None``, uses the config
        default.
    flat_obs : dict[int, bool] | None
        Mapping of agent index to whether it needs flat observations.

    Returns
    -------
    tuple
        ``(per_agent_reports, playthrough)`` where per_agent_reports
        maps agent_id to metric dicts and playthrough is the full
        ``PlaythroughData``.

    """
    from pyxis_portfolio_challenge.environment.metrics import (
        report_all_metrics,
    )
    from pyxis_portfolio_challenge.environment.multi_agent_evaluate import (
        evaluate_multi_agent,
    )

    env_kwargs = _build_multi_agent_env_kwargs(
        flatten_obs=False,
        assets_dir=config.evaluation_data_dir,
        bd_assets_dir=config.multi_agent.bd_eval_assets_dir,
    )
    n = env_kwargs["num_agents"]
    possible = [f"pharma_{i}" for i in range(n)]

    if len(agents) != len(possible):
        raise ValueError(
            f"Expected {len(possible)} agents, got {len(agents)}. "
            f"Agent slots: {possible}"
        )

    if any(a is None for a in agents):
        raise ValueError(
            "All agent slots must be specified. "
            "Use env.train() for training with a trainee slot."
        )

    _validate_flat_obs_overrides(agents, flat_obs)

    resolved = {}
    for i, spec in enumerate(agents):
        agent_name = possible[i]
        resolved[agent_name] = _resolve_agent(spec, agent_name)

    agents_with_format = {}
    for i, agent_name in enumerate(possible):
        agent = resolved[agent_name]
        if _needs_flat_obs(agents[i], i, flat_obs):
            agents_with_format[agent_name] = _DeferredFlatObsWrapper(agent)
        else:
            agents_with_format[agent_name] = agent

    cfg = config
    agent_metrics, playthrough = evaluate_multi_agent(
        agents=agents_with_format,
        worker_id=0,
        episodes_per_worker=1,
        env_kwargs=env_kwargs,
        warmup_steps=cfg.warmup_on_reset_steps,
        capture_playthrough=True,
        seed=seed,
    )

    per_agent_reports = {
        agent_id: report_all_metrics(metrics)
        for agent_id, metrics in agent_metrics.items()
    }

    return per_agent_reports, playthrough
