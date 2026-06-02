"""Render a grid of orbit_wars starting positions for visual review.

Usage:
    python3 scripts/preview_orbit_wars_starts.py [num_seeds] [start_seed]

Defaults to 16 seeds starting at 0. Saves to orbit_wars_starts.png.
"""
import math
import random
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    BOARD_SIZE,
    CENTER,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    generate_planets,
)

PLAYER_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f"]


def pick_home_group(planets, num_agents, rng):
    num_groups = len(planets) // 4
    if num_groups == 0:
        return None
    return rng.randint(0, num_groups - 1)


def render(ax, seed, num_agents=4):
    rng = random.Random(seed)
    # Match interpreter: angular_velocity drawn first, then planets.
    rng.uniform(0.025, 0.05)
    planets = generate_planets(rng)
    home_group = pick_home_group(planets, num_agents, rng)

    ax.set_xlim(0, BOARD_SIZE)
    ax.set_ylim(0, BOARD_SIZE)
    ax.set_aspect("equal")
    ax.set_facecolor("#0b1020")
    ax.set_title(f"seed={seed}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    # Rotation radius boundary
    ax.add_patch(Circle((CENTER, CENTER), ROTATION_RADIUS_LIMIT,
                        fill=False, edgecolor="#333", linestyle="--", linewidth=0.5))
    # Sun
    ax.add_patch(Circle((CENTER, CENTER), SUN_RADIUS, color="#f39c12"))

    home_ids = set()
    if home_group is not None:
        base = home_group * 4
        if num_agents == 2:
            home_ids = {base, base + 3}
        elif num_agents == 4:
            home_ids = {base + j for j in range(4)}

    for i, p in enumerate(planets):
        pid, _, x, y, r, _, _ = p
        orbital = math.hypot(x - CENTER, y - CENTER)
        is_orbiting = orbital + r < ROTATION_RADIUS_LIMIT
        if pid in home_ids:
            if num_agents == 4:
                base = (home_group or 0) * 4
                player = pid - base
            else:
                player = 0 if pid == (home_group or 0) * 4 else 1
            color = PLAYER_COLORS[player]
            edge = "white"
        else:
            color = "#7f8c8d" if is_orbiting else "#bdc3c7"
            edge = None
        ax.add_patch(Circle((x, y), max(r, 0.8), color=color,
                            ec=edge, linewidth=0.8 if edge else 0))


def main():
    num_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    start_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    cols = int(math.ceil(math.sqrt(num_seeds)))
    rows = int(math.ceil(num_seeds / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = axes.flatten() if num_seeds > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < num_seeds:
            render(ax, start_seed + i, num_agents=4)
        else:
            ax.axis("off")

    fig.suptitle(f"orbit_wars starts (4p, seeds {start_seed}..{start_seed + num_seeds - 1})",
                 color="black")
    fig.tight_layout()
    out = "orbit_wars_starts.png"
    fig.savefig(out, dpi=120, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
