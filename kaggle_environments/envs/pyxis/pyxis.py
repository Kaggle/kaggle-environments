"""Kaggle wiring for the Pyxis Portfolio Challenge (2-player investment game).

Kaggle scores an env by the terminal ``state.reward`` its interpreter sets, so
this drives the engine step-by-step and writes win/loss (1.0/0.5/0.0) at the
end. The live engine is held in a module-level cache keyed by ``env.id`` since
it can't be serialized into the replay state.
"""

import json
import os
import sys

from kaggle_environments.utils import resolve_episode_seed

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    # The bundled package uses absolute ``pyxis_portfolio_challenge`` imports.
    sys.path.insert(0, _DIR)

NUM_AGENTS = 2
AGENT_IDS = ["pharma_0", "pharma_1"]

# env.id -> {"env": pyenv, "cum": {aid: float}, "baselines": {(seat, spec): agent}}
_LIVE = {}

_FORFEIT_STATUSES = ("ERROR", "INVALID", "TIMEOUT")

with open(os.path.join(_DIR, "pyxis.json")) as _f:
    specification = json.load(_f)


def _to_jsonable(value):
    """Recursively convert numpy / engine types into JSON-safe Python types."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _build_pyxis_env():
    """Build the live engine on evaluation assets, as ``competition.evaluate`` does."""
    # Import the stable-baselines3 top-level package before the engine pulls in
    # its submodules: a submodule-first import order trips a circular import
    # (stable_baselines3.common.utils.get_device) in some sb3 builds.
    import stable_baselines3  # noqa: F401

    from pyxis_portfolio_challenge.config import config
    from pyxis_portfolio_challenge.environment.env_factory import (
        _build_multi_agent_env_kwargs,
    )
    from pyxis_portfolio_challenge.environment.multi_agent_training_gym import (
        MultiAgentInvestmentGameEnv,
    )
    from pyxis_portfolio_challenge.environment.warmup_wrapper import (
        MultiAgentWarmupOnResetWrapper,
    )

    env_kwargs = _build_multi_agent_env_kwargs(
        flatten_obs=True,
        num_agents=NUM_AGENTS,
        assets_dir=config.evaluation_data_dir,
        bd_assets_dir=config.multi_agent.bd_eval_assets_dir,
    )
    env = MultiAgentInvestmentGameEnv(**env_kwargs)
    if config.warmup_on_reset_steps > 0:
        env = MultiAgentWarmupOnResetWrapper(env, warmup_steps=config.warmup_on_reset_steps, verbose=False)
    return env


def _write_observations(state, pyenv, observations):
    """Copy per-agent engine observation + book-keeping into the Kaggle state."""
    portfolios = pyenv.agent_portfolios
    for i, aid in enumerate(AGENT_IDS):
        obs = state[i].observation
        obs.obs = _to_jsonable(observations.get(aid, []))
        obs.actionMask = _to_jsonable(pyenv.action_masks(aid))
        gs = portfolios.get(aid)
        if gs is not None:
            obs.cash = float(gs.cash)
            obs.enpv = float(gs.enpv())
            obs.bankrupt = bool(gs.bankrupt)


def _outcomes(pyenv, cum):
    """Win/loss/draw per agent, read from the engine's own PerEpisodeWinLoss metric."""
    from pyxis_portfolio_challenge.environment.metrics import MetricsContext
    from pyxis_portfolio_challenge.environment.multi_agent_metrics import (
        PerEpisodeWinLoss,
    )

    all_states = pyenv.agent_portfolios
    outcomes = {}
    for aid in AGENT_IDS:
        metric = PerEpisodeWinLoss()
        metric.on_episode_end(
            MetricsContext(
                game_state=all_states[aid],
                reward=cum[aid],
                agent_id=aid,
                all_agent_states=all_states,
                all_agent_rewards=cum,
            )
        )
        (outcomes[aid],) = metric.report()["PerEpisodeWinLoss"].values()
    return outcomes


def _finish(state, outcomes):
    """Assign final rewards and mark both agents DONE."""
    for i, aid in enumerate(AGENT_IDS):
        state[i].reward = outcomes[aid]
        state[i].status = "DONE"


def _illegal_investments(pyenv, aid, action):
    """True if ``action`` picks an investment level the mask forbids.

    An out-of-mask investment makes the engine raise, which would crash the
    whole episode, so the interpreter forfeits the offender instead. Level 0 is
    always safe.
    """
    if not isinstance(action, dict):
        return False
    inv = action.get("investments")
    if inv is None:
        return False
    inv_mask = pyenv.action_masks(aid).get("investments")
    if inv_mask is None:
        return False
    for i, choice in enumerate(inv):
        if i >= len(inv_mask):
            break
        try:
            c = int(choice)
        except (TypeError, ValueError):
            return True
        if c == 0:
            continue
        m = inv_mask[i]
        if isinstance(m, (list, tuple)):
            if c < 0 or c >= len(m) or not m[c]:
                return True
        elif not m or c != 1:
            return True
    return False


def _forfeit(state, env, loser_seats):
    """End the episode: the given seats lose (INVALID), the others win."""
    for i in range(len(state)):
        if i in loser_seats:
            if state[i].status == "ACTIVE":
                state[i].status = "INVALID"
        else:
            state[i].reward = 1.0
            state[i].status = "DONE"
    _LIVE.pop(env.id, None)


def interpreter(state, env):
    # Init runs during reset() (env.done is True), before the env.done return.
    if not state[0].observation.initialized:
        seed = resolve_episode_seed(env)
        pyenv = _build_pyxis_env()
        observations, _ = pyenv.reset(seed=seed)
        _LIVE[env.id] = {"env": pyenv, "cum": {aid: 0.0 for aid in AGENT_IDS}, "baselines": {}}
        _write_observations(state, pyenv, observations)
        for i in range(len(state)):
            state[i].observation.initialized = True
        state[0].observation.kaggleEnvId = env.id
        return state

    if env.done:
        return state

    live = _LIVE.get(env.id)
    if live is None:
        _finish(state, {aid: 0.5 for aid in AGENT_IDS})
        return state

    pyenv = live["env"]

    losers = {i for i in range(len(state)) if state[i].status in _FORFEIT_STATUSES}
    losers |= {
        i for i, aid in enumerate(AGENT_IDS) if i not in losers and _illegal_investments(pyenv, aid, state[i].action)
    }
    if losers:
        _forfeit(state, env, losers)
        return state

    actions = {aid: state[i].action for i, aid in enumerate(AGENT_IDS)}
    try:
        observations, rewards, terminations, truncations, _ = pyenv.step(actions)
    except Exception:
        # An action slipped past validation and broke the engine; draw it out.
        _finish(state, {aid: 0.5 for aid in AGENT_IDS})
        _LIVE.pop(env.id, None)
        return state

    for aid in AGENT_IDS:
        live["cum"][aid] += float(rewards.get(aid, 0.0))

    _write_observations(state, pyenv, observations)

    if all(terminations.values()) or all(truncations.values()):
        _finish(state, _outcomes(pyenv, live["cum"]))
        _LIVE.pop(env.id, None)

    return state


def renderer(state, env):
    lines = []
    for i, aid in enumerate(AGENT_IDS):
        obs = state[i].observation
        lines.append(
            f"{aid}: cash={obs.cash:,.0f} enpv={obs.enpv:,.0f} "
            f"bankrupt={obs.bankrupt} status={state[i].status} reward={state[i].reward}"
        )
    return "\n".join(lines)


def html_renderer():
    return ""


def _make_baseline(spec_name):
    """Builtin agent that delegates to a live-engine baseline (runs in-process)."""

    def agent(observation, configuration):
        from pyxis_portfolio_challenge.environment.competition import _resolve_agent

        live = _LIVE.get(observation["kaggleEnvId"])
        if live is None:
            return {}

        seat = int(observation["agentIndex"])
        key = (seat, spec_name)
        impl = live["baselines"].get(key)
        if impl is None:
            impl = _resolve_agent(spec_name, AGENT_IDS[seat])
            live["baselines"][key] = impl
        impl.set_env(live["env"])

        return _to_jsonable(impl(observation.get("obs", [])))

    return agent


agents = {
    "knapsack": _make_baseline("knapsack"),
    "random": _make_baseline("random"),
    "do_nothing": _make_baseline("do_nothing"),
}
