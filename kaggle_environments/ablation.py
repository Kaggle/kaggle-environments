"""Prompt-ablation runner for Kaggle game environments.

Cross-game tool for running parallel leaderboards across prompt variants.
Each game's harness directory contributes a ``prompt_variants.py`` whose
top-level ``VARIANTS: dict[str, GameHarness]`` is the experiment surface.
This runner discovers that module, schedules paired-seat round-robin
games per variant, and writes ``games.csv`` + ``summary.md``.

CLI
---

::

    python -m kaggle_environments.ablation check --env open_spiel_bargaining
    python -m kaggle_environments.ablation run \\
        --env open_spiel_bargaining \\
        --models gpt-5,claude-opus-4-7,gemini-2.5-pro \\
        --games 20 \\
        --concurrency 16 \\
        --out results/bargaining/

``MODEL_PROXY_KEY`` and ``MODEL_PROXY_URL`` come from env. The same key
authenticates every model (proxy fans out internally).

Design notes
------------

* Matrix shape (per the design doc): K parallel leaderboards, where each
  leaderboard runs a round-robin among the M models with both players
  using the same variant. Self-play is opt-in (``--self-play``).
* Every (A, B) matchup is scheduled in **seat-flipped pairs sharing one
  chance seed** -- the same instance is played twice with seats swapped,
  cancelling first-mover advantage and instance luck. ``--games`` counts
  individual games and must be even.
* Concurrency uses a ``ProcessPoolExecutor`` with the ``fork`` start
  method (one game per worker process). Threads turn out to deadlock
  inside ``env.run`` + the full harness path -- some shared state in
  the env or pyspiel interpreter is not thread-safe. Processes are also
  the right semantic match: each game has its own pyspiel state and its
  own model selection via ``model_override``, with zero cross-game
  interference. Fork preserves monkey-patched ``core_harness._call_llm``
  for smoke tests, so the in-process mocking pattern still works.
* No per-model concurrency cap yet -- a ``threading.Semaphore`` doesn't
  cross process boundaries, so the simple total ``--concurrency`` cap is
  the only knob. (Future: ``multiprocessing.Manager().Semaphore``.)
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib
import json
import logging
import math
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

from kaggle_environments import make
from kaggle_environments.core_harness import GameHarness, create_agent_fn

_log = logging.getLogger("kaggle_environments.ablation")


# ---------------------------------------------------------------------------
# Variant discovery
# ---------------------------------------------------------------------------


def _variants_module_path(env_name: str) -> str:
    """Map env name to the dotted path of its ``prompt_variants`` module."""
    if env_name.startswith("open_spiel_"):
        game = env_name[len("open_spiel_") :]
        return f"kaggle_environments.envs.open_spiel_env.games.{game}.prompt_variants"
    return f"kaggle_environments.envs.{env_name}.prompt_variants"


def load_variants(env_name: str) -> dict[str, GameHarness]:
    """Return the ``VARIANTS`` registry for ``env_name``, or raise."""
    module_path = _variants_module_path(env_name)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"No prompt_variants.py found for env '{env_name}' "
            f"(expected at {module_path}). Bootstrap one with the "
            f"run-ablation skill."
        ) from e
    variants = getattr(module, "VARIANTS", None)
    if not isinstance(variants, dict) or not variants:
        raise SystemExit(
            f"{module_path} must expose a non-empty VARIANTS "
            f"dict[str, GameHarness]."
        )
    if "baseline" not in variants:
        raise SystemExit(
            f"{module_path}.VARIANTS must include a 'baseline' entry "
            f"(byte-identical to harness.py)."
        )
    return variants


# ---------------------------------------------------------------------------
# Model wiring
# ---------------------------------------------------------------------------


def build_model_setup(
    model_name: str,
    api_key: str,
    api_base: str,
) -> tuple[str, dict[str, Any]]:
    """Compose ``(model_name, litellm_kwargs)`` for one agent.

    Mirrors :func:`core_harness._setup_model` but takes explicit args so
    callers can pick a model per-agent without touching the global env.
    """
    if api_base and api_base != "dummy_url":
        return f"openai/{model_name}", {
            "api_base": f"{api_base.rstrip('/')}/openapi",
            "api_key": api_key,
            "reasoning_effort": "high",
        }
    if "gemini" in model_name.lower() and not model_name.startswith("gemini/"):
        return f"gemini/{model_name}", {}
    return model_name, {}


# ---------------------------------------------------------------------------
# Cell scheduling
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GameCell:
    """One scheduled game.

    ``seed`` is shared across the two seat-flipped games of a pair; the
    pair as a whole is identified by ``(variant, frozenset({model_p0,
    model_p1}), seed)``. ``pair_role`` is ``"AB"`` or ``"BA"`` for the
    flipped games, or ``"SELF"`` for self-play (which doesn't pair).
    """

    variant: str
    model_p0: str
    model_p1: str
    seed: int
    pair_role: str


def build_cells(
    variants: Iterable[str],
    models: list[str],
    games_per_matchup: int,
    *,
    include_self_play: bool,
) -> list[GameCell]:
    """Enumerate every cell to play (K * M(M-1)/2 * games + optional self-play).

    For each variant, each unordered model pair contributes
    ``games_per_matchup`` games -- ``games_per_matchup / 2`` seat-flipped
    pairs sharing one chance seed each.
    """
    if games_per_matchup % 2 != 0:
        raise SystemExit(
            f"--games must be even (got {games_per_matchup}); each "
            f"matchup is scheduled in seat-flipped pairs."
        )
    pairs_per_matchup = games_per_matchup // 2
    cells: list[GameCell] = []
    for v in variants:
        # Round-robin over unordered model pairs.
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                for pair_idx in range(pairs_per_matchup):
                    cells.append(GameCell(v, a, b, seed=pair_idx, pair_role="AB"))
                    cells.append(GameCell(v, b, a, seed=pair_idx, pair_role="BA"))
            if include_self_play:
                for game_idx in range(games_per_matchup):
                    cells.append(
                        GameCell(v, a, a, seed=game_idx, pair_role="SELF")
                    )
    return cells


# ---------------------------------------------------------------------------
# Game execution
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GameResult:
    """Per-game row written to games.csv."""

    variant: str
    model_p0: str
    model_p1: str
    pair_role: str
    seed: int
    score_p0: float
    score_p1: float
    winner: int  # -1 P1, 0 draw, +1 P0
    length_moves: int
    crash_p0: bool
    crash_p1: bool
    error: str
    duration_s: float


def _extract_scores(env: Any) -> tuple[float, float, int]:
    """Return (score_p0, score_p1, winner) from a terminated env."""
    p0 = float(env.state[0].reward or 0.0)
    p1 = float(env.state[1].reward or 0.0)
    if p0 > p1:
        winner = 1
    elif p1 > p0:
        winner = -1
    else:
        winner = 0
    return p0, p1, winner


def _agent_crashed(env: Any, idx: int) -> bool:
    """True if agent ``idx`` ended in a non-DONE/INACTIVE status."""
    status = env.state[idx].status if env.state[idx] else None
    if not status:
        return False
    return str(status).upper() not in {"DONE", "INACTIVE", "ACTIVE"}


# ---------------------------------------------------------------------------
# Status files (real-time progress for the monitor thread)
# ---------------------------------------------------------------------------


def _status_filename(cell: GameCell) -> str:
    """Sanitize cell coords into a single filesystem-safe filename."""
    def s(x: str) -> str:
        return str(x).replace("/", "_").replace(" ", "_")
    return (
        f"{s(cell.variant)}__{s(cell.model_p0)}__vs__{s(cell.model_p1)}"
        f"__{s(cell.pair_role)}__seed{cell.seed}.json"
    )


def _write_status(status_dir: str, cell: GameCell, state: str,
                  started_at: float, **extra: Any) -> None:
    """Atomically write a per-cell status snapshot.

    Workers call this on each agent invocation so the parent's monitor
    thread can show move-by-move progress for in-flight games.
    """
    if not status_dir:
        return
    path = os.path.join(status_dir, _status_filename(cell))
    payload = {
        "variant": cell.variant,
        "model_p0": cell.model_p0,
        "model_p1": cell.model_p1,
        "seed": cell.seed,
        "pair_role": cell.pair_role,
        "state": state,
        "started_at": started_at,
        "updated_at": time.time(),
        **extra,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def _clear_status(status_dir: str, cell: GameCell) -> None:
    if not status_dir:
        return
    try:
        os.remove(os.path.join(status_dir, _status_filename(cell)))
    except FileNotFoundError:
        pass


def run_one_game(
    env_name: str,
    cell: GameCell,
    variant_obj: GameHarness,
    api_key: str,
    api_base: str,
    *,
    configuration_extras: Mapping[str, Any] | None = None,
    status_dir: str | None = None,
    started_at: float | None = None,
) -> GameResult:
    """Play one cell to completion and return its row.

    Called in a worker process by :func:`cmd_run`. Safe to call directly
    for testing -- no shared state beyond what's passed in.

    If ``status_dir`` is provided, each agent invocation triggers a
    status-file write so the parent's monitor thread can show live
    per-game progress.
    """
    start = time.perf_counter()
    if started_at is None:
        started_at = time.time()
    config: dict[str, Any] = {"seed": int(cell.seed)}
    if configuration_extras:
        config.update(configuration_extras)
    error = ""

    # Per-seat move counters used by the status writer below.
    moves = [0, 0]

    def _wrap(seat: int, real_agent):
        def tracked(obs: Any, cfg: dict[str, Any]) -> dict[str, Any]:
            moves[seat] += 1
            _write_status(
                status_dir or "", cell, "running", started_at,
                moves_p0=moves[0], moves_p1=moves[1],
            )
            return real_agent(obs, cfg)
        return tracked

    agent_p0 = _wrap(0, create_agent_fn(
        variant_obj,
        model_override=build_model_setup(cell.model_p0, api_key, api_base),
    ))
    agent_p1 = _wrap(1, create_agent_fn(
        variant_obj,
        model_override=build_model_setup(cell.model_p1, api_key, api_base),
    ))

    env = make(env_name, configuration=config, debug=False)
    try:
        env.run([agent_p0, agent_p1])
        p0, p1, winner = _extract_scores(env)
        length = len(env.steps)
        crash_p0 = _agent_crashed(env, 0)
        crash_p1 = _agent_crashed(env, 1)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _log.warning("Game crashed (%s): %s", cell, exc)
        p0 = p1 = 0.0
        winner = 0
        length = 0
        crash_p0 = crash_p1 = True

    return GameResult(
        variant=cell.variant,
        model_p0=cell.model_p0,
        model_p1=cell.model_p1,
        pair_role=cell.pair_role,
        seed=cell.seed,
        score_p0=p0,
        score_p1=p1,
        winner=winner,
        length_moves=length,
        crash_p0=crash_p0,
        crash_p1=crash_p1,
        error=error,
        duration_s=time.perf_counter() - start,
    )


class _AblationTimeout(BaseException):
    """Raised by the SIGALRM handler to enforce --game-timeout.

    Subclasses BaseException (not Exception) on purpose: the LLM agent loop
    in core_harness.agent_fn catches Exception broadly and retries on
    failure, which would swallow a plain TimeoutError. BaseException
    subclasses propagate through that handler so the timeout actually
    fires through to the worker's outer catch.
    """


def _alarm_handler(signum, frame):
    raise _AblationTimeout("game exceeded --game-timeout")


def _worker_entry(args: tuple) -> GameResult:
    """Picklable worker entrypoint for ProcessPoolExecutor.map / submit.

    Wraps :func:`run_one_game` in a SIGALRM watchdog so a single stuck LLM
    call doesn't block the worker process forever (the staging proxy is
    known to drop connections mid-stream, and litellm's retry loop can
    spin indefinitely). On timeout the worker returns a crash result and
    is free to take the next cell.
    """
    env_name, cell, variant_obj, api_key, api_base, timeout_secs, status_dir = args
    start = time.perf_counter()
    started_at = time.time()
    _write_status(status_dir or "", cell, "starting", started_at,
                  moves_p0=0, moves_p1=0)
    if timeout_secs and timeout_secs > 0:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(int(timeout_secs))
    try:
        return run_one_game(
            env_name, cell, variant_obj, api_key, api_base,
            status_dir=status_dir, started_at=started_at,
        )
    except _AblationTimeout:
        return GameResult(
            variant=cell.variant, model_p0=cell.model_p0, model_p1=cell.model_p1,
            pair_role=cell.pair_role, seed=cell.seed,
            score_p0=0.0, score_p1=0.0, winner=0, length_moves=0,
            crash_p0=True, crash_p1=True,
            error=f"TimeoutError: exceeded {timeout_secs}s",
            duration_s=time.perf_counter() - start,
        )
    except Exception as exc:  # noqa: BLE001
        return GameResult(
            variant=cell.variant, model_p0=cell.model_p0, model_p1=cell.model_p1,
            pair_role=cell.pair_role, seed=cell.seed,
            score_p0=0.0, score_p1=0.0, winner=0, length_moves=0,
            crash_p0=True, crash_p1=True,
            error=f"{type(exc).__name__}: {exc}",
            duration_s=time.perf_counter() - start,
        )
    finally:
        if timeout_secs and timeout_secs > 0:
            signal.alarm(0)
        _clear_status(status_dir or "", cell)


# ---------------------------------------------------------------------------
# CSV / summary writers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Real-time monitor (parent-side)
# ---------------------------------------------------------------------------


def _read_status_dir(status_dir: str) -> list[dict[str, Any]]:
    """Snapshot of all in-flight cells. Tolerates partial writes / races."""
    out: list[dict[str, Any]] = []
    try:
        names = os.listdir(status_dir)
    except FileNotFoundError:
        return out
    for name in names:
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(status_dir, name)) as f:
                out.append(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return out


def _classify_csv(csv_path: Path) -> tuple[int, int, int]:
    """Return (done, timed_out, crashed_other) from games.csv."""
    if not csv_path.exists():
        return 0, 0, 0
    done = timed_out = crashed = 0
    try:
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                done += 1
                err = row.get("error") or ""
                if "TimeoutError" in err:
                    timed_out += 1
                elif err:
                    crashed += 1
                else:
                    # Even with no error string, crash_p* may be true
                    # (agent ERROR detected by env without raising).
                    if (str(row.get("crash_p0", "")).lower() == "true"
                            or str(row.get("crash_p1", "")).lower() == "true"):
                        crashed += 1
    except FileNotFoundError:
        pass
    return done, timed_out, crashed


def _fmt_elapsed(secs: float) -> str:
    if secs < 60:
        return f"{int(secs):2d}s"
    return f"{int(secs // 60):2d}m{int(secs % 60):02d}s"


def _print_snapshot(status_dir: str, csv_path: Path, total: int,
                    overall_start: float) -> None:
    done, timed_out, crashed = _classify_csv(csv_path)
    in_flight = _read_status_dir(status_dir)
    in_flight.sort(key=lambda r: r.get("started_at", 0))
    now = time.time()
    overall_elapsed = _fmt_elapsed(now - overall_start)
    print(
        f"\n[{overall_elapsed}] {done}/{total} done "
        f"({timed_out} timeouts, {crashed} crashes) | {len(in_flight)} running:",
        flush=True,
    )
    for r in in_flight:
        cell_elapsed = now - r.get("started_at", now)
        moves = (r.get("moves_p0", 0) or 0) + (r.get("moves_p1", 0) or 0)
        m0 = (r.get("model_p0") or "")[-25:]
        m1 = (r.get("model_p1") or "")[-25:]
        print(
            f"  {r.get('variant',''):18s} {m0:25s} vs {m1:25s}"
            f"  seed={r.get('seed')} {r.get('pair_role',''):4s}"
            f"  moves={moves:2d}  {_fmt_elapsed(cell_elapsed)}"
            f"  ({r.get('state','?')})",
            flush=True,
        )


def _monitor_thread_target(
    status_dir: str, csv_path: Path, total: int,
    stop_event: threading.Event, overall_start: float,
    interval: float,
) -> None:
    while not stop_event.wait(interval):
        try:
            _print_snapshot(status_dir, csv_path, total, overall_start)
        except Exception as e:  # noqa: BLE001
            _log.warning("monitor snapshot failed: %s", e)


_CSV_FIELDS = [
    "variant", "model_p0", "model_p1", "pair_role", "seed",
    "score_p0", "score_p1", "winner", "length_moves",
    "crash_p0", "crash_p1", "error", "duration_s",
]


def _read_existing_cells(csv_path: Path) -> set[tuple]:
    """Return the set of cell keys already present in games.csv (for --resume)."""
    if not csv_path.exists():
        return set()
    done: set[tuple] = set()
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            done.add((
                row["variant"], row["model_p0"], row["model_p1"],
                row["pair_role"], int(row["seed"]),
            ))
    return done


def _cell_key(c: GameCell) -> tuple:
    return (c.variant, c.model_p0, c.model_p1, c.pair_role, c.seed)


def _open_csv_for_append(csv_path: Path) -> tuple[Any, csv.DictWriter]:
    """Open ``games.csv`` for appending; write header if new."""
    new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    f = csv_path.open("a", newline="")
    writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
    if new_file:
        writer.writeheader()
        f.flush()
    return f, writer


def _row_dict(r: GameResult) -> dict[str, Any]:
    return dataclasses.asdict(r)


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the per-variant / per-model statistics shown in summary.md."""
    # Per (variant, model) accumulators.
    games_played: dict[tuple[str, str], int] = defaultdict(int)
    game_wins: dict[tuple[str, str], float] = defaultdict(float)
    score_sum: dict[tuple[str, str], float] = defaultdict(float)
    score_n: dict[tuple[str, str], int] = defaultdict(int)
    crash_count: dict[tuple[str, str], int] = defaultdict(int)

    # Per (variant, pair_key) for pair-win aggregation. pair_key is the
    # unordered pair tuple + seed, so we can sum both seat-flipped games.
    # pair_totals[(variant, frozenset({A,B}), seed)][model] = total_score
    pair_totals: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    pair_games: dict[tuple, int] = defaultdict(int)

    for row in rows:
        v = row["variant"]
        m0, m1 = row["model_p0"], row["model_p1"]
        s0, s1 = float(row["score_p0"]), float(row["score_p1"])
        winner = int(row["winner"])
        seed = int(row["seed"])

        for model, score, crashed, won_flag in (
            (m0, s0, row["crash_p0"], 1.0 if winner == 1 else 0.5 if winner == 0 else 0.0),
            (m1, s1, row["crash_p1"], 1.0 if winner == -1 else 0.5 if winner == 0 else 0.0),
        ):
            games_played[(v, model)] += 1
            game_wins[(v, model)] += won_flag
            score_sum[(v, model)] += score
            score_n[(v, model)] += 1
            if str(crashed).lower() == "true":
                crash_count[(v, model)] += 1

        if row["pair_role"] in ("AB", "BA") and m0 != m1:
            pair_key = (v, frozenset((m0, m1)), seed)
            pair_totals[pair_key][m0] += s0
            pair_totals[pair_key][m1] += s1
            pair_games[pair_key] += 1

    # Per (variant, model) pair-wins. A pair counts when both seat-flipped
    # games for that pair_key are complete (pair_games[k] == 2).
    pair_wins: dict[tuple[str, str], float] = defaultdict(float)
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for pair_key, totals in pair_totals.items():
        if pair_games[pair_key] != 2:
            continue
        v = pair_key[0]
        models_in_pair = list(totals.keys())
        if len(models_in_pair) != 2:
            continue
        a, b = models_in_pair
        ta, tb = totals[a], totals[b]
        for m in (a, b):
            pair_counts[(v, m)] += 1
        if ta > tb:
            pair_wins[(v, a)] += 1.0
        elif tb > ta:
            pair_wins[(v, b)] += 1.0
        else:
            pair_wins[(v, a)] += 0.5
            pair_wins[(v, b)] += 0.5

    return {
        "games_played": games_played,
        "game_wins": game_wins,
        "score_sum": score_sum,
        "score_n": score_n,
        "crash_count": crash_count,
        "pair_wins": pair_wins,
        "pair_counts": pair_counts,
    }


def _format_leaderboard(
    variant: str,
    models: list[str],
    agg: dict[str, Any],
) -> str:
    """Render one leaderboard section for summary.md."""
    rows: list[tuple[str, float, float, float, float, float]] = []
    for m in models:
        gp = agg["games_played"].get((variant, m), 0)
        if gp == 0:
            continue
        pair_n = agg["pair_counts"].get((variant, m), 0)
        pair_win_pct = (
            100.0 * agg["pair_wins"].get((variant, m), 0.0) / pair_n
            if pair_n else float("nan")
        )
        game_win_pct = 100.0 * agg["game_wins"].get((variant, m), 0.0) / gp
        mean_score = agg["score_sum"].get((variant, m), 0.0) / max(gp, 1)
        crash_pct = 100.0 * agg["crash_count"].get((variant, m), 0) / gp
        rows.append((m, pair_win_pct, game_win_pct, mean_score, crash_pct, gp))
    rows.sort(
        # Sort by pair-win% desc, fall back to game-win% if all-self-play.
        key=lambda r: (-r[1] if not math.isnan(r[1]) else -r[2], -r[2]),
    )
    lines = [
        f"## Leaderboard: {variant}",
        "",
        "| model | pair-win% | game-win% | mean score | crash% | games |",
        "|-------|-----------|-----------|------------|--------|-------|",
    ]
    for m, pwin, gwin, sc, cr, gp in rows:
        pwin_str = "—" if math.isnan(pwin) else f"{pwin:.1f}"
        lines.append(
            f"| {m} | {pwin_str} | {gwin:.1f} | {sc:+.2f} | {cr:.1f} | {gp} |"
        )
    return "\n".join(lines)


def _format_rank_shifts(
    variants: list[str],
    models: list[str],
    agg: dict[str, Any],
) -> str:
    """Cross-variant rank-of-each-model table -- the artifact worth reading."""
    # Within each variant, rank models by pair-win% (game-win% fallback).
    ranks: dict[tuple[str, str], int] = {}
    for v in variants:
        scored: list[tuple[str, float]] = []
        for m in models:
            gp = agg["games_played"].get((v, m), 0)
            if gp == 0:
                continue
            pair_n = agg["pair_counts"].get((v, m), 0)
            primary = (
                agg["pair_wins"].get((v, m), 0.0) / pair_n if pair_n
                else agg["game_wins"].get((v, m), 0.0) / gp
            )
            scored.append((m, primary))
        scored.sort(key=lambda x: -x[1])
        for rank, (m, _) in enumerate(scored, start=1):
            ranks[(v, m)] = rank

    if not ranks:
        return ""

    width = max(len(m) for m in models)
    header = "| " + "model".ljust(width) + " | " + " | ".join(v for v in variants) + " |"
    sep = "|" + "-" * (width + 2) + "|" + "|".join("-" * (len(v) + 2) for v in variants) + "|"
    lines = ["## Cross-variant rank shifts", "", header, sep]
    for m in models:
        cells = [str(ranks.get((v, m), "—")).center(len(v)) for v in variants]
        lines.append("| " + m.ljust(width) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_anomalies(rows: list[dict[str, Any]]) -> str:
    """Flag variants with high crash or error rates."""
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)
    lines: list[str] = []
    for v, rs in sorted(by_variant.items()):
        crashes = sum(
            1 for r in rs
            if str(r["crash_p0"]).lower() == "true"
            or str(r["crash_p1"]).lower() == "true"
        )
        errors = sum(1 for r in rs if r["error"])
        total = len(rs)
        if total == 0:
            continue
        crash_pct = 100.0 * crashes / total
        if crash_pct >= 5.0 or errors > 0:
            lines.append(
                f"- **{v}**: {crashes}/{total} games with crash "
                f"({crash_pct:.1f}%), {errors} hard errors"
            )
    if not lines:
        return "## Anomalies\n\n(none)"
    return "## Anomalies\n\n" + "\n".join(lines)


def render_summary(
    env_name: str,
    variants: list[str],
    models: list[str],
    rows: list[dict[str, Any]],
) -> str:
    """Render the full summary.md content for ``rows``."""
    agg = _aggregate(rows)
    parts = [
        f"# Ablation results: {env_name}",
        "",
        f"Variants: {', '.join(variants)}",
        f"Models: {', '.join(models)}",
        f"Total games: {len(rows)}",
        "",
    ]
    for v in variants:
        parts.append(_format_leaderboard(v, models, agg))
        parts.append("")
    rank_shifts = _format_rank_shifts(variants, models, agg)
    if rank_shifts:
        parts.append(rank_shifts)
        parts.append("")
    parts.append(_format_anomalies(rows))
    parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("--models must list at least one model.", file=sys.stderr)
        return 2

    variants_all = load_variants(args.env)
    if args.variants:
        names = [v.strip() for v in args.variants.split(",") if v.strip()]
        missing = [n for n in names if n not in variants_all]
        if missing:
            print(
                f"Unknown variant(s): {missing}. "
                f"Available: {sorted(variants_all)}",
                file=sys.stderr,
            )
            return 2
        variants_to_run = {n: variants_all[n] for n in names}
    else:
        variants_to_run = variants_all

    cells = build_cells(
        variants_to_run.keys(),
        models,
        args.games,
        include_self_play=args.self_play,
    )

    if args.dry_run:
        print(
            f"{len(cells)} games would be scheduled "
            f"({len(variants_to_run)} variants × {len(models)} models × "
            f"{args.games} games/matchup"
            f"{' + self-play' if args.self_play else ''})."
        )
        return 0

    api_key = os.environ.get("MODEL_PROXY_KEY", "")
    api_base = os.environ.get("MODEL_PROXY_URL", "dummy_url")
    if not api_key and api_base != "dummy_url":
        print(
            "MODEL_PROXY_KEY env var is required (or set "
            "MODEL_PROXY_URL=dummy_url for offline testing).",
            file=sys.stderr,
        )
        return 2

    # Propagate the per-call timeout via env var so forked workers (and the
    # core_harness._call_llm inside them) inherit it without an API change.
    if args.llm_call_timeout and args.llm_call_timeout > 0:
        os.environ["LLM_CALL_TIMEOUT"] = str(int(args.llm_call_timeout))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "games.csv"
    summary_path = out_dir / "summary.md"

    done = _read_existing_cells(csv_path) if args.resume else set()
    todo = [c for c in cells if _cell_key(c) not in done]
    if args.resume:
        print(f"Resume: {len(done)} cells already done, {len(todo)} to go.")
    else:
        print(f"Scheduling {len(todo)} games.")

    csv_file, writer = _open_csv_for_append(csv_path)

    # Status dir for the live monitor. Workers write per-cell JSON files
    # here on each agent invocation; the parent's monitor thread reads
    # them every --status-interval seconds and prints a snapshot.
    status_dir = str(out_dir / "status")
    os.makedirs(status_dir, exist_ok=True)
    # Stale status files from a previous run would otherwise show as
    # "in-flight" cells that don't exist. Clear on launch.
    for fname in os.listdir(status_dir):
        try:
            os.remove(os.path.join(status_dir, fname))
        except OSError:
            pass

    # Build the work items eagerly so workers receive a self-contained
    # tuple they can run without any reference to the parent process.
    work_items = [
        (args.env, c, variants_to_run[c.variant], api_key, api_base,
         args.game_timeout, status_dir)
        for c in todo
    ]

    completed = 0
    overall_start = time.time()
    stop_event = threading.Event()
    monitor: threading.Thread | None = None
    if args.status_interval > 0:
        monitor = threading.Thread(
            target=_monitor_thread_target,
            args=(status_dir, csv_path, len(todo), stop_event,
                  overall_start, args.status_interval),
            daemon=True,
        )
        monitor.start()

    try:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=args.concurrency,
            mp_context=ctx,
        ) as pool:
            futures = {pool.submit(_worker_entry, item): item for item in work_items}
            for fut in as_completed(futures):
                result = fut.result()
                writer.writerow(_row_dict(result))
                csv_file.flush()
                completed += 1
                if (
                    completed % max(1, len(todo) // 20) == 0
                    or completed == len(todo)
                ):
                    print(
                        f"  [{completed}/{len(todo)}] {result.variant}"
                        f" {result.model_p0} vs {result.model_p1}"
                        f" pair={result.pair_role} seed={result.seed}"
                        f" -> ({result.score_p0:+.2f}, {result.score_p1:+.2f})"
                        f" in {result.duration_s:.1f}s"
                    )
    finally:
        stop_event.set()
        if monitor is not None:
            monitor.join(timeout=2.0)
        csv_file.close()

    # Re-read everything to build the summary (works under --resume).
    with csv_path.open() as f:
        all_rows = list(csv.DictReader(f))
    summary = render_summary(
        args.env, list(variants_to_run.keys()), models, all_rows,
    )
    summary_path.write_text(summary)
    print(f"\nWrote {len(all_rows)} rows to {csv_path}")
    print(f"Wrote summary to {summary_path}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: check
# ---------------------------------------------------------------------------


def _seeded_observations(env_name: str, n: int = 5) -> list[dict[str, Any]]:
    """Build N observations from a few seeded steps of the env for parity checks."""
    obs_list: list[dict[str, Any]] = []
    for seed in range(n):
        env = make(env_name, configuration={"seed": seed}, debug=False)
        # Trigger interpreter to populate per-agent observation.
        env.reset()
        # First state with a real current player. For OpenSpiel-backed
        # envs the proxy obs lives on env.state[player].observation.
        for player_idx in range(len(env.state)):
            agent_state = env.state[player_idx]
            obs = agent_state.observation
            if not obs:
                continue
            d = dict(obs) if isinstance(obs, dict) else dict(vars(obs))
            d.setdefault("playerId", player_idx)
            obs_list.append(d)
            break
    return obs_list


def cmd_check(args: argparse.Namespace) -> int:
    variants = load_variants(args.env)
    print(f"env: {args.env}")
    print(f"variants: {sorted(variants)}")

    # Baseline parity: BASELINE.make_prompt must equal harness.generate_prompt
    # for the same observation. Loaded via the per-env harness module.
    try:
        harness_mod_path = _variants_module_path(args.env).rsplit(".", 1)[0] + ".harness"
        harness_mod = importlib.import_module(harness_mod_path)
        prod_make_prompt = getattr(harness_mod, "generate_prompt", None)
    except ModuleNotFoundError:
        prod_make_prompt = None

    obs_list = _seeded_observations(args.env, n=args.parity_seeds)
    if not obs_list:
        print("  (could not build seeded observations -- skipping render check)")
        return 0

    failed = False

    baseline = variants["baseline"]
    if prod_make_prompt is None:
        print("  (no harness.generate_prompt found -- skipping baseline parity)")
    else:
        for i, obs in enumerate(obs_list):
            try:
                a = prod_make_prompt(obs, [])
                b = baseline.make_prompt(obs, [])
            except Exception as e:
                print(f"  baseline parity obs[{i}]: ERROR {e}")
                failed = True
                continue
            if a != b:
                print(f"  baseline parity obs[{i}]: MISMATCH")
                failed = True
            else:
                print(f"  baseline parity obs[{i}]: ok ({len(a)} chars)")

    # Render check: every variant must render without error on each obs.
    for name, variant in variants.items():
        for i, obs in enumerate(obs_list):
            try:
                prompt = variant.make_prompt(obs, [])
                assert isinstance(prompt, str) and prompt
            except Exception as e:
                print(f"  variant '{name}' obs[{i}]: RENDER ERROR {e}")
                traceback.print_exc()
                failed = True
        print(f"  variant '{name}': rendered ok across {len(obs_list)} observations")

    if failed:
        print("FAIL")
        return 1
    print("OK")
    return 0


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m kaggle_environments.ablation",
        description="Prompt-ablation runner for Kaggle game environments.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run a paired-seat ablation tournament.")
    pr.add_argument("--env", required=True, help="Env name, e.g. open_spiel_bargaining.")
    pr.add_argument("--models", required=True,
                    help="Comma-separated model names (proxy-side, no openai/ prefix).")
    pr.add_argument("--games", type=int, default=20,
                    help="Games per matchup (must be even; default 20).")
    pr.add_argument("--variants", default="",
                    help="Comma-separated variant names; default all.")
    pr.add_argument("--concurrency", type=int, default=8,
                    help="Max concurrent games (worker process count).")
    pr.add_argument("--llm-call-timeout", type=int, default=900,
                    help="Per-LLM-call timeout in seconds. Passed to "
                         "litellm via the LLM_CALL_TIMEOUT env var, which "
                         "core_harness._call_llm reads. Single hung calls "
                         "fail at this limit so the agent's retry loop "
                         "can recover. Default 900s = 15 min.")
    pr.add_argument("--game-timeout", type=int, default=0,
                    help="Per-game SIGALRM watchdog (seconds). Default 0 "
                         "(disabled) -- the per-call timeout above is the "
                         "primary protection. Set to e.g. 3600 if you want "
                         "a hard ceiling on cumulative game wall time on "
                         "top of the per-call limit.")
    pr.add_argument("--status-interval", type=float, default=10.0,
                    help="Seconds between live status snapshots printed "
                         "by the monitor thread. Set to 0 to disable.")
    pr.add_argument("--self-play", action="store_true",
                    help="Add the M self-play cells per leaderboard.")
    pr.add_argument("--resume", action="store_true",
                    help="Skip cells already present in games.csv.")
    pr.add_argument("--dry-run", action="store_true",
                    help="Print the cell count and exit.")
    pr.add_argument("--out", default="ablation_results",
                    help="Output directory for games.csv + summary.md.")
    pr.set_defaults(func=cmd_run)

    pc = sub.add_parser("check", help="Verify baseline parity + variant rendering.")
    pc.add_argument("--env", required=True)
    pc.add_argument("--parity-seeds", type=int, default=5,
                    help="How many seeded observations to test.")
    pc.set_defaults(func=cmd_check)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
