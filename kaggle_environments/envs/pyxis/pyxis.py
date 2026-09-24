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
    # The bundled package uses absolute ``pyxis_portfolio_challenge`` imports,
    # so its parent has to be importable. This leaks exactly one top-level
    # name, which is distinctive enough not to shadow a competitor's module.
    # (The bundle also shipped a top-level ``app``; that one was generic enough
    # to collide, so it now lives under ``pyxis_portfolio_challenge.app``.)
    # Removing the path hack entirely means rewriting ~160 absolute imports in
    # vendored code — worth doing only alongside an upstream change.
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


# Fixed therapeutic-area order, mirrored in the visualizer. Snapshots store the
# index rather than repeating these strings once per asset per step.
_THERAPEUTIC_AREAS = [
    "oncology",
    "respiratory and immunology",
    "vaccines and infectious disease",
]


def _therapeutic_area_index(area):
    """Index of ``area``, or -1 when the event isn't tied to one.

    Clinical-site auctions are TA-agnostic, so their alerts carry an empty
    therapeutic area. The visualizer already falls back to no label on an
    unknown index.
    """
    try:
        return _THERAPEUTIC_AREAS.index(area)
    except ValueError:
        return -1


def _asset_key(asset):
    """Short stable id for an asset, long enough not to collide within a match."""
    return str(asset.id)[:8]


def _render_snapshot(game, known):
    """Compact per-step portfolio + market state for the web visualizer.

    The engine's own ``playthrough.capture_agent_states`` emits ~215 KB per step
    (21 MB for a match), most of it static prose and per-phase trial detail the
    renderer never reads. This keeps the ~4 KB the visualizer actually draws: an
    asset's immutable identity is emitted once into ``assetMeta`` the step it
    first appears, and every later step carries only the mutable row. ``known``
    is the caller's accumulator of already-described assets.

    Asset and market rows stay positional because they repeat ~60x per step;
    naming their fields would add 16% to the whole replay. Everything that
    appears once per step is spelled out -- measured at 1.7% of the replay, not
    worth the illegibility. ``visualizer/default/src/types.ts`` labels the tuple
    slots.
    """
    market = game.shared_market
    asset_meta = {}
    agents = {}
    for aid, gs in game.agent_states.items():
        rows = []
        for asset in gs.assets.values():
            key = _asset_key(asset)
            if key not in known:
                known.add(key)
                asset_meta[key] = [
                    asset.name,
                    _THERAPEUTIC_AREAS.index(asset.therapeutic_area),
                    int(asset.indication),
                    0 if asset.type == "internal" else 1,
                    round(float(asset.max_revenue)),
                ]
            trial = asset.trial
            rows.append(
                [
                    key,
                    asset.state.integer,
                    trial.phase.integer if trial else -1,
                    int(trial.time_remaining) if trial else 0,
                    round(float(trial.ptrs), 3) if trial else 0,
                    int(asset.current_investment_level),
                    int(asset.time_on_market),
                ]
            )
        agents[aid] = {
            "cash": round(float(gs.cash)),
            "enpv": round(float(gs.enpv())),
            "eroi": round(float(gs.eroi()), 3),
            "bankrupt": bool(gs.bankrupt),
            "operationalSites": int(gs.operational_sites),
            "buildingSites": len(gs.sites_in_development),
            # ``expired_assets`` is the unreleased asset pool, not expired drugs —
            # it holds hundreds of entries at reset — so it is deliberately absent.
            "failedCount": len(gs.failed_assets),
            "droppedCount": len(gs.dropped_assets),
            "assets": rows,
        }

    snapshot = {
        "time": int(game.time),
        "agents": agents,
        "bdOffers": [
            {
                "name": a.name,
                "therapeuticArea": _THERAPEUTIC_AREAS.index(a.therapeutic_area),
                "phase": a.trial.phase.integer if a.trial else -1,
                "maxRevenue": round(float(a.max_revenue)),
            }
            for a in market.current_bd_assets
        ],
        # Already pruned by the engine to a rolling 5-step window.
        "alerts": [
            {
                "step": al.step,
                "eventType": al.event_type.value,
                "agentId": al.agent_id,
                "therapeuticArea": _therapeutic_area_index(al.therapeutic_area),
                "indication": int(al.indication),
                "details": _to_jsonable(al.details),
            }
            for al in market.alerts
        ],
        "indicationMarkets": [
            [
                key,
                m.indication_name,
                m.first_mover_agent,
                round(float(m.demand_multiplier), 3),
                sum(len(ids) for ids in m.active_drugs.values()),
            ]
            for key, m in market.indication_markets.items()
        ],
    }
    if asset_meta:
        snapshot["assetMeta"] = asset_meta
    return snapshot


def _write_observations(state, pyenv, observations, known_assets):
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
    # Hidden + shared: recorded once on seat 0 for the replay, stripped from both
    # agents' runtime observations so neither can read the other's portfolio.
    state[0].observation.render = _render_snapshot(pyenv.multi_agent_game, known_assets)


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


def _coerce_for_space(space, action):
    """Cast a JSON action to the dtypes and shapes ``space`` expects.

    A JSON round-trip loses dtype: a ``Box(float32)`` head arrives as a list of
    Python floats, and ``contains()`` rejects the float64 array ``asarray``
    would build. Casting first means validation judges the values, not the
    encoding.
    """
    import numpy as np

    out = {}
    for head, value in action.items():
        sub = space.get(head) if hasattr(space, "get") else None
        if sub is None:
            out[head] = value
            continue
        arr = np.asarray(value, dtype=sub.dtype)
        # Discrete heads are scalars; gymnasium wants the numpy scalar, not a
        # 0-d array.
        out[head] = arr if arr.shape else arr[()]
    return out


def _masked_head_illegal(mask, values):
    """True if any entry of ``values`` picks a choice ``mask`` forbids.

    Masks are shaped ``(*slots, num_choices)`` against an action of shape
    ``(*slots,)`` — including scalar heads like ``upgrade``, whose mask is a
    flat per-choice vector. So the check is a per-slot ``mask[slot][choice]``
    lookup, and a slot-count mismatch is itself illegal.
    """
    import numpy as np

    if mask is None:
        return False
    try:
        m = np.asarray(mask, dtype=bool)
        choices = np.asarray(values).reshape(-1)
        flat = m.reshape(-1, m.shape[-1])
        if choices.shape[0] != flat.shape[0]:
            return True
        idx = choices.astype(np.int64)
        if np.any(idx < 0) or np.any(idx >= flat.shape[1]):
            return True
        return not bool(flat[np.arange(flat.shape[0]), idx].all())
    except (ValueError, TypeError, IndexError, OverflowError):
        return True


def _normalize_action(pyenv, aid, action):
    """Coerce a submitted action into one the engine accepts, or reject it.

    Returns ``(action, illegal)``. Isolated submissions can only send JSON, and
    the engine demands every enabled head each step, so missing heads are
    filled from ``noop_action()`` — an agent that ignores a feature gets its
    no-op rather than a crash.

    Rejection is reserved for actions the engine would raise on: a non-dict, an
    unknown head, or a choice the mask forbids. Those forfeit the offender
    rather than ending the match in a draw.
    """
    noop = pyenv.noop_action()
    if action is None:
        return noop, False
    if not isinstance(action, dict):
        return noop, True
    if any(head not in noop for head in action):
        return noop, True

    masks = pyenv.action_masks(aid)
    merged = dict(noop)
    for head, value in action.items():
        if value is None:
            continue
        if _masked_head_illegal(masks.get(head), value):
            return noop, True
        merged[head] = value

    # The masks only cover discrete heads. Let the engine's own action space
    # reject anything else malformed (wrong length, wrong dtype, out of range)
    # before it reaches step() and raises.
    try:
        space = pyenv.action_space(aid)
        coerced = _coerce_for_space(space, merged)
        if not space.contains(coerced):
            return noop, True
    except Exception:
        return noop, True
    return coerced, False


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
        _LIVE[env.id] = {
            "env": pyenv,
            "cum": {aid: 0.0 for aid in AGENT_IDS},
            "baselines": {},
            # Assets already described in a snapshot's ``meta``; see _render_snapshot.
            "known_assets": set(),
        }
        _write_observations(state, pyenv, observations, _LIVE[env.id]["known_assets"])
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
    actions = {}
    for i, aid in enumerate(AGENT_IDS):
        action, illegal = _normalize_action(pyenv, aid, state[i].action)
        if illegal:
            losers.add(i)
        actions[aid] = action
    if losers:
        _forfeit(state, env, losers)
        return state

    try:
        observations, rewards, terminations, truncations, _ = pyenv.step(actions)
    except Exception:
        # Validation should have caught this. Forfeiting both seats keeps a
        # malformed action from being a cheap way to escape a losing position;
        # a draw would reward whoever sent it.
        _forfeit(state, env, set(range(len(state))))
        return state

    for aid in AGENT_IDS:
        live["cum"][aid] += float(rewards.get(aid, 0.0))

    _write_observations(state, pyenv, observations, live["known_assets"])

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


def html_renderer(env, mode):
    jspath = os.path.join(_DIR, "visualizer", "default", "dist", "index.html")
    if os.path.exists(jspath):
        with open(jspath, encoding="utf-8") as f:
            return f.read()
    # Unbuilt visualizer; ``renderer`` above is the text fallback.
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
