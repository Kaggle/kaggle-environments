import math
from sys import stderr
import numpy as np
from dataclasses import dataclass
if __package__ == "":
    from lux.weather import get_weather_config
    from lux.cargo import UnitCargo
    from lux.config import EnvConfig
else:
    from .weather import get_weather_config
    from .cargo import UnitCargo
    from .config import EnvConfig
@dataclass
class Factory:
    team_id: int
    unit_id: str
    strain_id: int
    power: int
    cargo: UnitCargo
    pos: np.ndarray
    # lichen tiles connected to this factory
    # lichen_tiles: np.ndarray
    env_cfg: EnvConfig


    def build_heavy_metal_cost(self, game_state):
        unit_cfg = self.env_cfg.ROBOTS["HEAVY"]
        return unit_cfg.METAL_COST
    def build_heavy_power_cost(self, game_state):
        current_weather = game_state.weather_schedule[game_state.real_env_steps]
        weather_cfg = get_weather_config(current_weather, self.env_cfg)
        unit_cfg = self.env_cfg.ROBOTS["HEAVY"]
        return math.ceil(unit_cfg.POWER_COST * weather_cfg["power_loss_factor"])
    def can_build_heavy(self, game_state):
        return self.power >= self.build_heavy_power_cost(game_state) and self.cargo.metal >= self.build_heavy_metal_cost(game_state)
    def build_heavy(self):
        return 1

    def build_light_metal_cost(self, game_state):
        unit_cfg = self.env_cfg.ROBOTS["LIGHT"]
        return unit_cfg.METAL_COST
    def build_light_power_cost(self, game_state):
        current_weather = game_state.weather_schedule[game_state.real_env_steps]
        weather_cfg = get_weather_config(current_weather, self.env_cfg)
        unit_cfg = self.env_cfg.ROBOTS["LIGHT"]
        return math.ceil(unit_cfg.POWER_COST * weather_cfg["power_loss_factor"])
    def can_build_light(self, game_state):
        return self.power >= self.build_light_power_cost(game_state) and self.cargo.metal >= self.build_light_metal_cost(game_state)

    def build_light(self):
        return 0

    def water_cost(self, game_state):
        """
        Water required to perform water action
        """
        owned_lichen_tiles = (game_state.board.lichen_strains == self.strain_id).sum()
        return np.ceil(owned_lichen_tiles / self.env_cfg.LICHEN_WATERING_COST_FACTOR) + 1
    def can_water(self, game_state):
        return self.cargo.water >= self.water_cost(game_state)
    def water(self):
        return 2
