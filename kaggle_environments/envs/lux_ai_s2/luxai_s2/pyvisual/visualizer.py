import numpy as np

try:
    import pygame
    from pygame import gfxdraw
except:
    print("No pygame installed, ignoring import")
from luxai_s2.map.board import Board
from luxai_s2.state import State
from luxai_s2.unit import UnitType

try:
    import matplotlib.pyplot as plt

    color_to_rgb = dict(
        yellow=[236, 238, 126],
        green=[173, 214, 113],
        blue=[154, 210, 203],
        red=[164, 74, 63],
    )
    strain_colors = plt.colormaps["Pastel1"]
except:
    pass


class Visualizer:
    def __init__(self, state: State) -> None:
        # self.screen = pygame.display.set_mode((3*N*game_map.width, N*game_map.height))
        self.screen_size = (1000, 1000)
        self.board = state.board
        self.tile_width = min(
            self.screen_size[0] // self.board.width,
            self.screen_size[1] // self.board.height,
        )
        self.WINDOW_SIZE = (
            self.tile_width * self.board.width,
            self.tile_width * self.board.height,
        )
        self.surf = pygame.Surface(self.WINDOW_SIZE)
        self.surf.fill([239, 120, 79])
        self.state = state
        pygame.font.init()
        self.screen = None

    def init_window(self):
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode(self.WINDOW_SIZE)

    def rubble_color(self, rubble):
        opacity = 0.2 + min(rubble / 100, 1) * 0.8
        return [96, 32, 9, opacity * 255]

    def ore_color(self, rubble):
        return [218, 167, 48, 255]

    def ice_color(self, rubble):
        return [44, 158, 211, 255]

    def update_scene(self, state: State):
        self.state = state
        self.surf.fill([239, 120, 79, 255])
        for x in range(self.board.width):
            for y in range(self.board.height):
                rubble_amt = self.state.board.rubble[x][y]
                rubble_color = self.rubble_color(
                    rubble_amt
                )  # [255 - self.state.board.rubble[y][x] * 255 / 100] * 3
                # import ipdb;ipdb.set_trace()
                gfxdraw.box(
                    self.surf,
                    (
                        self.tile_width * x,
                        self.tile_width * y,
                        self.tile_width,
                        self.tile_width,
                    ),
                    rubble_color,
                )
                if self.state.board.ice[x, y] > 0:
                    pygame.draw.rect(
                        self.surf,
                        self.ice_color(rubble_amt),
                        pygame.Rect(
                            self.tile_width * x,
                            self.tile_width * y,
                            self.tile_width,
                            self.tile_width,
                        ),
                    )
                # if self.state.valid_spawns_mask
                if self.state.board.ore[x, y] > 0:
                    pygame.draw.rect(
                        self.surf,
                        self.ore_color(rubble_amt),
                        pygame.Rect(
                            self.tile_width * x,
                            self.tile_width * y,
                            self.tile_width,
                            self.tile_width,
                        ),
                    )
                if self.state.board.lichen_strains[x, y] != -1:
                    c = strain_colors.colors[
                        self.state.board.lichen_strains[x, y]
                        % len(strain_colors.colors)
                    ]
                    pygame.draw.rect(
                        self.surf,
                        [int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)],
                        pygame.Rect(
                            self.tile_width * x,
                            self.tile_width * y,
                            self.tile_width,
                            self.tile_width,
                        ),
                    )
        if len(state.teams) > 0:
            team_color = dict(player_0=[224, 65, 40], player_1=[0, 112, 81])
            for agent in state.factories:
                if agent not in state.teams:
                    continue
                team = state.teams[agent]
                for factory in state.factories[agent].values():
                    x = factory.pos.x
                    y = factory.pos.y
                    pygame.draw.rect(
                        self.surf,
                        # color_to_rgb[team.faction.value.color],
                        team_color[agent],
                        pygame.Rect(
                            self.tile_width * (x - 1),
                            self.tile_width * (y - 1),
                            self.tile_width * 3,
                            self.tile_width * 3,
                        ),
                        border_radius=int(self.tile_width / 2),
                    )
                    self.sans_font = pygame.font.SysFont("Open Sans", 30)
                    self.surf.blit(
                        self.sans_font.render("F", False, [51, 56, 68]),
                        (self.tile_width * x, self.tile_width * y),
                    )
            for agent in state.units:
                if agent not in state.teams:
                    continue
                team = state.teams[agent]
                for unit in state.units[agent].values():
                    x = unit.pos.x
                    y = unit.pos.y
                    h = 1
                    pygame.draw.rect(
                        self.surf,
                        [51, 56, 68],
                        pygame.Rect(
                            self.tile_width * (x),
                            self.tile_width * (y),
                            self.tile_width * 1,
                            self.tile_width * 1,
                        ),
                    )
                    pygame.draw.rect(
                        self.surf,
                        # color_to_rgb[team.faction.value.color],
                        team_color[agent],
                        pygame.Rect(
                            self.tile_width * (x) + h,
                            self.tile_width * (y) + h,
                            (self.tile_width) * 1 - h * 2,
                            (self.tile_width) * 1 - h * 2,
                        ),
                    )

                    label = "H"
                    if unit.unit_type == UnitType.LIGHT:
                        label = "L"
                    self.sans_font = pygame.font.SysFont("Open Sans", 20)
                    self.surf.blit(
                        self.sans_font.render(label, False, [51, 56, 68]),
                        (self.tile_width * x + 2, self.tile_width * y + 2),
                    )

    def render(self):
        pygame.display.update()
        self.screen.blit(self.surf, (0, 0))

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )
