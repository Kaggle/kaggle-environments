try:
    import pygame
    from pygame import gfxdraw
except:
    print("No pygame installed, ignoring import")
from luxai2022.map.board import Board
from luxai2022.state import State
from luxai2022.unit import UnitType
try:
    import matplotlib.pyplot as plt
    color_to_rgb = dict(yellow=[236, 238, 126], green=[173, 214, 113], blue=[154, 210, 203], red=[164, 74, 63])
    strain_colors=plt.colormaps['Pastel1']
except:
    pass
class Visualizer:
    def __init__(self, state: State) -> None:
        # self.screen = pygame.display.set_mode((3*N*game_map.width, N*game_map.height))
        self.screen_size = (1000, 1000)
        self.board = state.board
        self.tile_width = min(self.screen_size[0] // self.board.width, self.screen_size[1] // self.board.height)
        self.screen = pygame.display.set_mode((self.tile_width * self.board.width, self.tile_width * self.board.height))
        self.screen.fill([239, 120, 79])
        self.state = state
        pygame.font.init() # you have to call this at the start

    def rubble_color(self, rubble):
        opacity = 0.2 + min(rubble / 100, 1) * 0.8
        return [96, 32, 9, opacity * 255]
    def ore_color(self, rubble):
        return [218, 167, 48, 255]
    def ice_color(self, rubble):
        return [44, 158, 211, 255]


    def update_scene(self, state: State):
        self.state = state
        self.screen.fill([239, 120, 79, 255])
        for x in range(self.board.width):
            for y in range(self.board.height):
                rubble_amt = self.state.board.rubble[y][x]
                rubble_color = self.rubble_color(rubble_amt) #[255 - self.state.board.rubble[y][x] * 255 / 100] * 3
                # import ipdb;ipdb.set_trace()
                gfxdraw.box(self.screen, (self.tile_width * x, self.tile_width * y, self.tile_width, self.tile_width), rubble_color)
                if self.state.board.ice[y, x] > 0:
                    pygame.draw.rect(
                        self.screen,
                        self.ice_color(rubble_amt),
                        pygame.Rect(self.tile_width * x, self.tile_width * y, self.tile_width, self.tile_width),
                    )
                # print(self.state.board.ore[y, x])
                if self.state.board.ore[y, x] > 0:
                    pygame.draw.rect(
                        self.screen,
                        self.ore_color(rubble_amt),
                        pygame.Rect(self.tile_width * x, self.tile_width * y, self.tile_width, self.tile_width),
                    )
                if self.state.board.lichen_strains[y, x] != -1:
                    c = strain_colors.colors[self.state.board.lichen_strains[y, x] % len(strain_colors.colors)]
                    pygame.draw.rect(
                        self.screen,
                        [int(c[0]*255),int(c[1]*255),int(c[2]*255)],
                        pygame.Rect(self.tile_width * x, self.tile_width * y, self.tile_width, self.tile_width),
                    )
                # screen.fill(ice_color, (N*x+N*game_map.width, N*y, N, N))
                # screen.fill(ore_color, (N*x+2*N*game_map.width, N*y, N, N))
        if len(state.teams) > 0:
            for agent in state.factories:
                if agent not in state.teams: continue
                team = state.teams[agent]
                for factory in state.factories[agent].values():
                    x = factory.pos.x
                    y = factory.pos.y
                    pygame.draw.rect(
                        self.screen,
                        color_to_rgb[team.faction.value.color],
                        pygame.Rect(
                            self.tile_width * (x - 1),
                            self.tile_width * (y - 1),
                            self.tile_width * 3,
                            self.tile_width * 3,
                        ),
                        border_radius=int(self.tile_width / 2)
                    )
                    self.sans_font = pygame.font.SysFont('Open Sans', 30)
                    self.screen.blit(self.sans_font.render('F', False, [51,56,68]), (self.tile_width * x, self.tile_width * y))
            for agent in state.units:
                if agent not in state.teams: continue
                team = state.teams[agent]
                for unit in state.units[agent].values():
                    x = unit.pos.x
                    y = unit.pos.y
                    h=1
                    pygame.draw.rect(
                        self.screen,
                        [51,56,68],
                        
                        pygame.Rect(
                            self.tile_width * (x),
                            self.tile_width * (y),
                            self.tile_width * 1,
                            self.tile_width * 1,
                        ),
                    )
                    pygame.draw.rect(
                        self.screen,
                        color_to_rgb[team.faction.value.color],
                        pygame.Rect(
                            self.tile_width * (x)+h,
                            self.tile_width * (y)+ h,
                            (self.tile_width) * 1 - h * 2,
                            (self.tile_width) * 1 - h * 2,
                        ),
                    )
                    
                    label = "H"
                    if unit.unit_type == UnitType.LIGHT:
                        label = "L"
                    self.sans_font = pygame.font.SysFont('Open Sans', 20)
                    self.screen.blit(self.sans_font.render(label, False, [51,56,68]), (self.tile_width * x+2, self.tile_width * y+2))
    def render(self):
        pygame.display.update()
