# A quick visualizer to aid in map generating algorithms.
try:
    import pygame
except:
    pass
import numpy as np


def viz(game_map, screen=None):
    N = min(500 // game_map.width, 750 // game_map.height)
    if screen is None:
        screen = pygame.display.set_mode((3 * N * game_map.width, N * game_map.height))

    for x in range(game_map.width):
        for y in range(game_map.height):
            rubble_color = [255 - game_map.rubble[y][x] * 255 / 100] * 3
            ice_color = [0, 0, game_map.ice[y][x] * 255 / 100]
            ore_color = [game_map.ore[y][x] * 255 / 100, 0, 0]
            screen.fill(rubble_color, (N * x, N * y, N, N))
            screen.fill(ice_color, (N * x + N * game_map.width, N * y, N, N))
            screen.fill(ore_color, (N * x + 2 * N * game_map.width, N * y, N, N))
    pygame.display.update()
