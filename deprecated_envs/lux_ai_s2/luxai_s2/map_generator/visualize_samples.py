from math import prod
from pathlib import Path

import numpy as np
import pygame

from luxai_s2.env import LuxAI_S2

if __name__ == "__main__":
    env = LuxAI_S2()
    N = 100
    Path("./map_samples").mkdir(parents=True, exist_ok=True)
    stats = dict(rubble_count=[], ice=[], ore=[])
    for seed in range(10000, 10000 + N):
        env.reset(seed)
        env.render()
        env.py_visualizer.update_scene(env.state)

        pygame.image.save(env.py_visualizer.screen, f"map_samples/{seed}.jpg")
        stats["ice"] += [env.state.board.ice.sum()]
        stats["ore"] += [env.state.board.ore.sum()]
        stats["rubble_count"] += [env.state.board.rubble.sum()]

    for k in stats:
        stats[k] = np.mean(stats[k])
    stats["rubble_count"] /= prod(env.state.board.rubble.shape)
    print(stats)
