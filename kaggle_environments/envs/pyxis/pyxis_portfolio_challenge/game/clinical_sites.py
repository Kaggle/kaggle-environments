"""
Clinical sites arbitration helpers.

Pure resolution logic for the clinical sites capacity feature, kept separate
from ``GameState`` so it can be unit-tested in isolation. Given the set of
new-trial requests competing for a limited number of free sites in a single
step, decide which requests are granted.
"""

from __future__ import annotations

from typing import Hashable, Mapping, Sequence, TypeVar

K = TypeVar("K", bound=Hashable)


def resolve_site_grants(
    requested: Sequence[K],
    free_sites: int,
    priorities: Mapping[K, float] | None = None,
) -> set[K]:
    """
    Decide which new-trial requests receive one of the free clinical sites.

    Parameters
    ----------
    requested:
        Keys (e.g. asset IDs) whose action would start a new
        Idle -> InDevelopment trial this step, each consuming one site. The
        sequence order is the deterministic fallback ordering (ascending asset
        index) used for pure-index arbitration and as the tiebreak when
        priorities collide.
    free_sites:
        Number of sites available to grant this step
        (``operational_sites - occupied``, never negative).
    priorities:
        Per-key priority scores. When provided (``agent_priority`` mode) requests
        are granted in descending score; because the sort is stable, the
        ``requested`` order (asset index) breaks any measure-zero ties. When
        ``None`` (default) grants follow ``requested`` order directly (pure
        asset-index arbitration).

    Returns
    -------
    set
        The subset of ``requested`` granted a site. Requests not in this set are
        the caller's responsibility to turn into costless no-ops (the asset stays
        Idle, no cost charged). Nothing is ever errored.

    """
    if free_sites <= 0:
        return set()
    if len(requested) <= free_sites:
        return set(requested)
    if priorities is None:
        ordered = list(requested)
    else:
        ordered = sorted(requested, key=lambda k: -priorities.get(k, 0.0))
    return set(ordered[:free_sites])
