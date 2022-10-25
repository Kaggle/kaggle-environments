from __future__ import annotations
from collections import defaultdict
from typing import Dict, List
import numpy as np
from typing import TYPE_CHECKING
from luxai2022.config import EnvConfig
if TYPE_CHECKING:
    from luxai2022.factory import Factory
from luxai2022.map.position import Position
from luxai2022.map_generator.generator import GameMap
from luxai2022.unit import Unit


class Board:
    def __init__(self, seed=None, env_cfg: EnvConfig = None) -> None:
        self.env_cfg = env_cfg
        self.height =  self.env_cfg.map_size
        self.width =  self.env_cfg.map_size
        self.seed = seed
        rng = np.random.RandomState(seed=seed)
        map_type = rng.choice(["Cave", "Mountain"])
        symmetry = rng.choice(["horizontal", "vertical"])
        self.map = GameMap.random_map(seed=seed, symmetry=symmetry, map_type=map_type, width=self.width, height=self.height)
        self.factories_per_team = rng.randint(env_cfg.MIN_FACTORIES, env_cfg.MAX_FACTORIES + 1)

        # remove bottom once generator is fully ready
        self.map.rubble = self.map.rubble.astype(int)
        self.map.ore = self.map.ore.astype(int)
        self.map.ice = self.map.ice.astype(int)

        self.lichen = np.zeros((self.height, self.width), dtype=int)
        # ownership of lichen by factory id, a simple mask
        # -1 = no ownership
        self.lichen_strains = -np.ones((self.height, self.width), dtype=int)
        # self.units_map: np.ndarray = np.zeros((self.height, self.width))
        self.units_map: Dict[str, List[Unit]] = defaultdict(list)
        # TODO consider unit occupancy map, may speed up computation by storing more info in a form indexable in numpy arrays

        # maps center of factory to the factory
        self.factory_map: Dict[str, 'Factory'] = dict()
        # != -1 if a factory tile is on the location. Equals factory number id
        self.factory_occupancy_map = -np.ones((self.height, self.width), dtype=int)


        # Valid spawn locations
        player_0_spawn_info = self.get_valid_spawns(0)
        player_1_spawn_info = self.get_valid_spawns(1)
        self.spawns = {"player_0": player_0_spawn_info[0],
                       "player_1": player_1_spawn_info[0]}
        self.spawn_masks = {"player_0": player_0_spawn_info[1],
                       "player_1": player_1_spawn_info[1]}
    def pos_hash(self, pos: Position):
        return f"{pos.x},{pos.y}"
    def get_units_at(self, pos: Position):
        pos_hash = self.pos_hash(pos)
        if pos_hash in self.units_map:
            return self.units_map[pos_hash]
        return None
    def get_factory_at(self, pos: Position):
        pos_hash = self.pos_hash(pos)
        if pos_hash in self.factory_map:
            return self.factory_map[pos_hash]
        return None
    def get_valid_spawns(self, team_id):
        xx, yy = np.mgrid[:self.width, :self.height]
        if self.map.symmetry == "vertical":
            if team_id == 0:
                spawns_mask = xx < (self.width - 2) / 2
            else:
                spawns_mask = xx > (self.width + 2) / 2
        if self.map.symmetry == "horizontal":
            if team_id == 0:
                spawns_mask = yy < (self.height - 2) / 2

            else:
                spawns_mask = yy > (self.height + 2) / 2

        # if self.map.symmetry == "rotational":
        #     if team_id == 0:
        #         spawns_mask = xx < (self.width - 1) / 2

        #     else:
        #         spawns_mask = xx > (self.width - 1) / 2

        # if self.map.symmetry == "/":
        #     if team_id == 0:
        #         spawns_mask = xx - yy < 0

        #     else:
        #         spawns_mask = xx - yy > 0

        # if self.map.symmetry == "\\":
        #     if team_id == 0:
        #         spawns_mask = xx + yy < (self.width + self.height) / 2 - 1

        #     else:
        #         spawns_mask = xx + yy > (self.width + self.height) / 2 - 1

        x, y = np.where(spawns_mask)

        spawns = np.array([*zip(x, y)])
        spawns = spawns[(spawns[:, 0] > 0) & (spawns[:, 1] > 0)]
        spawns = spawns[(spawns[:, 0] < self.width - 1) & (spawns[:, 1] < self.height - 1)]
        return spawns, spawns_mask

    @property
    def rubble(self) -> np.ndarray:
        return self.map.rubble
    @property
    def ice(self) -> np.ndarray:
        return self.map.ice
    @property
    def ore(self):
        return self.map.ore
    def state_dict(self):
        return dict(
            rubble=self.rubble.copy(),
            ore=self.ore.copy(),
            ice=self.ice.copy(),
            lichen=self.lichen.copy(),
            lichen_strains=self.lichen_strains.copy(),
            spawns=self.spawns.copy(),
            factories_per_team=self.factories_per_team,
        )
