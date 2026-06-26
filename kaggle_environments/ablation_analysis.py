"""Post-hoc statistical analysis of ablation results.

Reads a ``games.csv`` produced by :mod:`kaggle_environments.ablation` and
quantifies how much each prompt variant actually shifts the leaderboard,
relative to a noise floor estimated from a duplicated "null" variant.

The headline statistic is ``Sum |delta rank|`` (Σ|Δrank|) -- for a given
variant, the sum across all models of how many ranks each model moved
versus baseline. Higher = the variant reorders who wins. The null
variant (byte-identical to baseline) gives us the distribution of this
statistic under the null hypothesis that prompts have no effect; the
permutation test compares each real variant's observed value to that
distribution to produce a p-value.

Usage::

    python -m kaggle_environments.ablation_analysis \\
        --csv results/bargaining/games.csv \\
        --baseline baseline --null null \\
        --permutations 2000
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _load(csv_path: Path) -> list[dict]:
    """Load games.csv, dropping rows with errors (timeout/crash/etc).

    Permutation logic depends on each cell having a real pair-outcome,
    so half-played or crashed games would contaminate the test.
    """
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    clean = [
        r for r in rows
        if not r.get("error")
        and str(r.get("crash_p0", "")).lower() != "true"
        and str(r.get("crash_p1", "")).lower() != "true"
    ]
    dropped = len(rows) - len(clean)
    if dropped:
        print(f"# dropped {dropped} rows with errors/crashes", file=sys.stderr)
    return clean


def _models_in(rows: list[dict]) -> list[str]:
    seen = set()
    for r in rows:
        seen.add(r["model_p0"])
        seen.add(r["model_p1"])
    return sorted(seen)


def _pair_id(row: dict) -> tuple:
    """Canonical pair id: unordered model pair + chance seed.

    Two seat-flipped games share this id (and share the same instance via
    the pinned chance seed). The permutation test shuffles labels at this
    granularity so seat-flip pairing is preserved.
    """
    a, b = sorted((row["model_p0"], row["model_p1"]))
    return (a, b, int(row["seed"]))


def _pair_score_per_model(
    rows: list[dict],
) -> dict[tuple, dict[str, float]]:
    """Aggregate raw rows into ``{pair_id: {model: total_score_in_pair}}``."""
    totals: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[tuple, int] = defaultdict(int)
    for r in rows:
        pid = _pair_id(r)
        totals[pid][r["model_p0"]] += float(r["score_p0"])
        totals[pid][r["model_p1"]] += float(r["score_p1"])
        counts[pid] += 1
    # Only keep complete pairs (both seat-flipped games present).
    return {pid: ts for pid, ts in totals.items() if counts[pid] == 2}


def _leaderboard_from_pairs(
    pair_scores: Iterable[dict[str, float]],
    models: list[str],
) -> list[tuple[str, float]]:
    """Score each model = sum of its pair-totals. Returns sorted descending."""
    cum: dict[str, float] = defaultdict(float)
    seen: dict[str, int] = defaultdict(int)
    for totals in pair_scores:
        for m, s in totals.items():
            cum[m] += s
            seen[m] += 1
    scored = [
        (m, cum.get(m, 0.0) / max(seen.get(m, 1), 1))
        for m in models
    ]
    scored.sort(key=lambda x: -x[1])
    return scored


def _ranks_from_leaderboard(
    leaderboard: list[tuple[str, float]],
) -> dict[str, int]:
    return {m: i + 1 for i, (m, _) in enumerate(leaderboard)}


def _sum_abs_delta_rank(
    ranks_a: dict[str, int],
    ranks_b: dict[str, int],
) -> int:
    return sum(abs(ranks_a[m] - ranks_b[m]) for m in ranks_a if m in ranks_b)


def _variant_pairs(rows: list[dict], variant: str) -> dict[tuple, dict[str, float]]:
    return _pair_score_per_model([r for r in rows if r["variant"] == variant])


def _observed_delta(
    baseline_pairs: dict[tuple, dict[str, float]],
    variant_pairs: dict[tuple, dict[str, float]],
    models: list[str],
) -> int:
    lb_baseline = _leaderboard_from_pairs(baseline_pairs.values(), models)
    lb_variant = _leaderboard_from_pairs(variant_pairs.values(), models)
    return _sum_abs_delta_rank(
        _ranks_from_leaderboard(lb_baseline),
        _ranks_from_leaderboard(lb_variant),
    )


def _permutation_distribution(
    baseline_pairs: dict[tuple, dict[str, float]],
    variant_pairs: dict[tuple, dict[str, float]],
    models: list[str],
    n_permutations: int,
    rng: random.Random,
) -> list[int]:
    """Shuffle labels across the pooled pairs, recompute Σ|Δrank| each time.

    Under H0 ("baseline and variant come from the same distribution"),
    the label is arbitrary. The fraction of permutations producing a
    statistic at least as extreme as observed is the p-value.

    Both arms must have the same number of pairs for the permutation to
    be well-defined; we subsample the larger to match the smaller.
    """
    a_ids = list(baseline_pairs.keys())
    b_ids = list(variant_pairs.keys())
    n = min(len(a_ids), len(b_ids))
    if n == 0:
        return []

    # Pool of (pair_id, source-arm) tuples. Each pair_id may appear once
    # per source arm (the two arms are independent draws from
    # comparable matchups, not the literal same pair).
    pool: list[tuple[str, dict[str, float]]] = []
    for pid in a_ids[:n]:
        pool.append(("A", baseline_pairs[pid]))
    for pid in b_ids[:n]:
        pool.append(("B", variant_pairs[pid]))

    dist: list[int] = []
    for _ in range(n_permutations):
        rng.shuffle(pool)
        a_sample = [p[1] for p in pool[:n]]
        b_sample = [p[1] for p in pool[n:2 * n]]
        lb_a = _leaderboard_from_pairs(a_sample, models)
        lb_b = _leaderboard_from_pairs(b_sample, models)
        dist.append(_sum_abs_delta_rank(
            _ranks_from_leaderboard(lb_a),
            _ranks_from_leaderboard(lb_b),
        ))
    return dist


def _summary_stats(xs: list[int]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    return statistics.mean(xs), statistics.pstdev(xs)


def _p_value(observed: int, dist: list[int]) -> float:
    """P(permuted >= observed). One-sided; Σ|Δrank| can only be positive."""
    if not dist:
        return float("nan")
    return sum(1 for x in dist if x >= observed) / len(dist)


def analyze(
    csv_path: Path,
    baseline_name: str,
    null_name: str | None,
    n_permutations: int,
    seed: int,
) -> str:
    """Run the full analysis; return a Markdown-formatted report."""
    rows = _load(csv_path)
    if not rows:
        return "No clean rows in CSV."

    variants = sorted({r["variant"] for r in rows})
    if baseline_name not in variants:
        raise SystemExit(
            f"baseline variant '{baseline_name}' not found. "
            f"Available: {variants}"
        )
    if null_name and null_name not in variants:
        print(
            f"# warning: null variant '{null_name}' not in data; "
            f"no calibrated noise floor",
            file=sys.stderr,
        )
        null_name = None

    models = _models_in(rows)
    baseline_pairs = _variant_pairs(rows, baseline_name)
    rng = random.Random(seed)

    lines: list[str] = []
    lines.append(f"# Permutation-test analysis")
    lines.append("")
    lines.append(f"- CSV: `{csv_path}`")
    lines.append(f"- Baseline: `{baseline_name}` ({len(baseline_pairs)} complete pairs)")
    if null_name:
        null_pairs = _variant_pairs(rows, null_name)
        lines.append(f"- Null: `{null_name}` ({len(null_pairs)} complete pairs)")
    lines.append(f"- Permutations per test: {n_permutations}")
    lines.append(f"- Models ({len(models)}): {', '.join(models)}")
    lines.append("")

    # Noise floor: null vs baseline observed + its permutation distribution.
    noise_floor_observed: float | None = None
    if null_name:
        null_observed = _observed_delta(
            baseline_pairs, null_pairs, models,
        )
        null_dist = _permutation_distribution(
            baseline_pairs, null_pairs, models, n_permutations, rng,
        )
        nm, ns = _summary_stats(null_dist)
        noise_floor_observed = float(null_observed)
        lines.append(f"## Noise floor")
        lines.append("")
        lines.append(
            f"Observed Σ|Δrank| of `{null_name}` vs `{baseline_name}`: "
            f"**{null_observed}**"
        )
        lines.append("")
        lines.append(
            f"Distribution under label-permutation: mean **{nm:.2f}**, "
            f"sd **{ns:.2f}**"
        )
        lines.append("")
        lines.append(
            f"This is the Σ|Δrank| we expect from LLM-sampling noise alone, "
            f"when prompts are byte-identical. Any real variant whose "
            f"observed Σ|Δrank| sits comfortably above this is a "
            f"statistically detectable shift."
        )
        lines.append("")

    # Each real variant: observed Σ|Δrank|, permutation p-value.
    lines.append(f"## Per-variant tests vs `{baseline_name}`")
    lines.append("")
    header = (
        "| variant | n pairs | obs Σ|Δrank| | perm mean ± sd | p-value | "
        "vs noise floor |"
    )
    sep = (
        "|---------|--------:|-------------:|---------------:|--------:|"
        "----------------|"
    )
    lines.append(header)
    lines.append(sep)

    real_variants = [v for v in variants if v not in {baseline_name, null_name}]
    for v in real_variants:
        v_pairs = _variant_pairs(rows, v)
        if not v_pairs:
            lines.append(f"| {v} | 0 | — | — | — | (no pairs) |")
            continue
        observed = _observed_delta(baseline_pairs, v_pairs, models)
        dist = _permutation_distribution(
            baseline_pairs, v_pairs, models, n_permutations, rng,
        )
        m, s = _summary_stats(dist)
        p = _p_value(observed, dist)
        if noise_floor_observed is not None:
            comp = (
                "above" if observed > noise_floor_observed
                else ("at" if observed == noise_floor_observed else "below")
            )
            comp_str = f"{comp} ({observed - noise_floor_observed:+.0f})"
        else:
            comp_str = "—"
        lines.append(
            f"| {v} | {len(v_pairs)} | {observed} | "
            f"{m:.2f} ± {s:.2f} | {p:.4f} | {comp_str} |"
        )

    lines.append("")
    lines.append(
        "*p-value = fraction of label-permutations producing Σ|Δrank| ≥ "
        "observed. Small p (e.g. <0.05) means the variant reorders the "
        "leaderboard more than chance would.*"
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m kaggle_environments.ablation_analysis",
        description="Permutation test for prompt-ablation rank shifts.",
    )
    p.add_argument("--csv", required=True, help="Path to games.csv.")
    p.add_argument("--baseline", default="baseline",
                   help="Name of the baseline variant (default: baseline).")
    p.add_argument("--null", default="null",
                   help="Name of the null variant (byte-identical to "
                        "baseline). Set to empty string to disable.")
    p.add_argument("--permutations", type=int, default=2000,
                   help="Number of label-shuffles per test (default 2000).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the permutation shuffles.")
    p.add_argument("--out", default="",
                   help="Optional output file for the Markdown report; "
                        "default prints to stdout.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    null_name = args.null or None
    report = analyze(
        Path(args.csv), args.baseline, null_name,
        args.permutations, args.seed,
    )
    if args.out:
        Path(args.out).write_text(report)
        print(f"Wrote {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
