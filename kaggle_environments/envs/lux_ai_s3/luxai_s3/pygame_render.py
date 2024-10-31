from luxai_s3.params import EnvParams
from luxai_s3.state import ASTEROID_TILE, NEBULA_TILE, EnvState
import numpy as np

try:
    import pygame
except:
    pass

TILE_SIZE = 64


class LuxAIPygameRenderer:
    def __init__(self):
        pass

    def render(self, state: EnvState, params: EnvParams):
        """Render the environment."""

        # Initialize Pygame if not already done
        if not pygame.get_init():
            pygame.init()
            self.clock = pygame.time.Clock()
            # Set up the display
            screen_width = params.map_width * TILE_SIZE
            screen_height = params.map_height * TILE_SIZE
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            self.surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            pygame.display.set_caption("Lux AI Season 3")

            self.display_options = {
                "show_grid": True,
                "show_relic_spots": False,
                "show_sensor_mask": True,
                "show_vision_power_map": True,
                "show_energy_field": False,
            }

        # Handle events to keep the window responsive
        render_state = "running"
        while True:
            self._update_display(state, params)
            for event in pygame.event.get():
                if event.type == pygame.TEXTINPUT:
                    if event.text == " ":
                        if render_state == "running":
                            render_state = "paused"
                        else:
                            render_state = "running"
                    elif event.text == "r":
                        self.display_options["show_relic_spots"] = (
                            not self.display_options["show_relic_spots"]
                        )
                    elif event.text == "s":
                        self.display_options["show_sensor_mask"] = (
                            not self.display_options["show_sensor_mask"]
                        )
                    elif event.text == "e":
                        self.display_options["show_energy_field"] = (
                            not self.display_options["show_energy_field"]
                        )
            else:
                if render_state == "paused":
                    self.clock.tick(60)
                    continue
                break

    def _update_display(self, state: EnvState, params: EnvParams):
        # Fill the screen with a background color
        self.screen.fill((200, 200, 200))
        self.surface.fill((200, 200, 200, 255))  # Light gray background

        # Draw the grid of tiles
        for x in range(params.map_width):
            for y in range(params.map_height):
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                tile_type = state.map_features.tile_type[x, y]
                if tile_type == NEBULA_TILE:
                    color = (166, 177, 225, 255)  # Light blue (a6b1e1) for tile type 1
                elif tile_type == ASTEROID_TILE:
                    color = (51, 56, 68, 255)
                else:
                    color = (255, 255, 255, 255)  # White for other tile types
                pygame.draw.rect(self.surface, color, rect)  # Draw filled squares

        # Draw relic node configs if display option is enabled
        def draw_rect_alpha(surface, color, rect):
            shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
            surface.blit(shape_surf, rect)

        if self.display_options["show_relic_spots"]:
            for i in range(params.max_relic_nodes):
                if state.relic_nodes_mask[i]:
                    x, y = state.relic_nodes[i, :2]
                    config_size = params.relic_config_size
                    half_size = config_size // 2
                    for dx in range(-half_size, half_size + 1):
                        for dy in range(-half_size, half_size + 1):
                            config_x = x + dx
                            config_y = y + dy

                            if (
                                0 <= config_x < params.map_width
                                and 0 <= config_y < params.map_height
                            ):

                                config_value = state.relic_node_configs[
                                    i, dy + half_size, dx + half_size
                                ]

                                if config_value > 0:
                                    rect = pygame.Rect(
                                        config_x * TILE_SIZE,
                                        config_y * TILE_SIZE,
                                        TILE_SIZE,
                                        TILE_SIZE,
                                    )
                                    draw_rect_alpha(
                                        self.surface, (255, 215, 0, 50), rect
                                    )  # Semi-transparent gold

        # Draw energy nodes
        for i in range(params.max_energy_nodes):
            if state.energy_nodes_mask[i]:
                x, y = state.energy_nodes[i, :2]
                center_x = (x + 0.5) * TILE_SIZE
                center_y = (y + 0.5) * TILE_SIZE
                radius = (
                    TILE_SIZE // 4
                )  # Adjust this value to change the size of the circle
                pygame.draw.circle(
                    self.surface,
                    (0, 255, 0, 255),
                    (int(center_x), int(center_y)),
                    radius,
                )
        # Draw relic nodes
        for i in range(params.max_relic_nodes):
            if state.relic_nodes_mask[i]:
                x, y = state.relic_nodes[i, :2]
                rect_size = TILE_SIZE // 2  # Make the square smaller than the tile
                rect_x = x * TILE_SIZE + (TILE_SIZE - rect_size) // 2
                rect_y = y * TILE_SIZE + (TILE_SIZE - rect_size) // 2
                rect = pygame.Rect(rect_x, rect_y, rect_size, rect_size)
                pygame.draw.rect(
                    self.surface, (173, 151, 32, 255), rect
                )  # Light blue color

        # Draw sensor mask
        if self.display_options["show_sensor_mask"]:
            for team in range(params.num_teams):
                for x in range(params.map_width):
                    for y in range(params.map_height):
                        if state.sensor_mask[team, x, y]:
                            draw_rect_alpha(
                                self.surface,
                                (255, 0, 0, 25),
                                pygame.Rect(
                                    x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE
                                ),
                            )

        if self.display_options["show_energy_field"]:
            font = pygame.font.Font(None, 32)  # You may need to adjust the font size
            for x in range(params.map_width):
                for y in range(params.map_height):
                    energy_field_value = state.map_features.energy[x, y]
                    text = font.render(str(energy_field_value), True, (255, 255, 255))
                    text_rect = text.get_rect(
                        center=((x + 0.5) * TILE_SIZE, (y + 0.5) * TILE_SIZE)
                    )
                    self.surface.blit(text, text_rect)
                    if energy_field_value > 0:
                        draw_rect_alpha(
                            self.surface,
                            (
                                0,
                                255,
                                0,
                                255 * energy_field_value / params.max_energy_per_tile,
                            ),
                            pygame.Rect(
                                x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE
                            ),
                        )
                    else:
                        draw_rect_alpha(
                            self.surface,
                            (
                                255,
                                0,
                                0,
                                255 * energy_field_value / params.min_energy_per_tile,
                            ),
                            pygame.Rect(
                                x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE
                            ),
                        )
        # if self.display_options["show_vision_power_map"]:
        #     print(state.vision_power_map.shape)
        #     font = pygame.font.Font(None, 32)  # You may need to adjust the font size
        #     # vision_power_map = vision_power_map - (state.map_features.tile_type == NEBULA_TILE)[..., 0] * params.nebula_tile_vision_reduction
        #     for team in range(0, 1):
        #         for x in range(params.map_width):
        #             for y in range(params.map_height):
        #                 vision_power_value = state.vision_power_map[team, x, y]
        #                 vision_power_value -= state.map_features.tile_type[x, y] == NEBULA_TILE
        #                 text = font.render(str(vision_power_value), True, (0, 255, 255))
        #                 text_rect = text.get_rect(
        #                     center=((x + 0.5) * TILE_SIZE, (y + 0.5) * TILE_SIZE)
        #                 )
        #                 self.surface.blit(text, text_rect)

        # Draw units
        for team in range(2):
            for i in range(params.max_units):
                if state.units_mask[team, i]:
                    x, y = state.units.position[team, i]
                    center_x = (x + 0.5) * TILE_SIZE
                    center_y = (y + 0.5) * TILE_SIZE
                    radius = (
                        TILE_SIZE // 3
                    )  # Adjust this value to change the size of the circle
                    color = (
                        (255, 0, 0, 255) if team == 0 else (0, 0, 255, 255)
                    )  # Red for team 0, Blue for team 1
                    pygame.draw.circle(
                        self.surface, color, (int(center_x), int(center_y)), radius
                    )
        # Draw unit counts
        unit_counts = {}
        for team in range(2):
            for i in range(params.max_units):
                if state.units_mask[team, i]:
                    x, y = np.array(state.units.position[team, i])
                    pos = (x, y)
                    if pos not in unit_counts:
                        unit_counts[pos] = 0
                    unit_counts[pos] += 1

        font = pygame.font.Font(None, 32)  # You may need to adjust the font size
        for pos, count in unit_counts.items():
            if count >= 1:
                x, y = pos
                text = font.render(str(count), True, (255, 255, 255))  # White text
                text_rect = text.get_rect(
                    center=((x + 0.5) * TILE_SIZE, (y + 0.5) * TILE_SIZE)
                )
                self.surface.blit(text, text_rect)

        # Draw the grid lines
        for x in range(params.map_width + 1):
            pygame.draw.line(
                self.surface,
                (100, 100, 100),
                (x * TILE_SIZE, 0),
                (x * TILE_SIZE, params.map_height * TILE_SIZE),
            )
        for y in range(params.map_height + 1):
            pygame.draw.line(
                self.surface,
                (100, 100, 100),
                (0, y * TILE_SIZE),
                (params.map_width * TILE_SIZE, y * TILE_SIZE),
            )

        self.screen.blit(self.surface, (0, 0))
        # Update the display
        pygame.display.flip()
