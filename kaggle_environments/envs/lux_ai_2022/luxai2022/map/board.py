from __future__ import annotations
from collections import defaultdict
from typing import Dict, List
import numpy as np
from typing import TYPE_CHECKING
from luxai2022.config import EnvConfig
if TYPE_CHECKING:
    from luxai2022.factory import Factory
    from luxai2022.state import State
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
        self.valid_spawns_mask = np.ones((self.height, self.width), dtype=bool)
        resource_mask = (self.map.ice != 0) | (self.map.ore != 0)
        # generate all 9 shifts of the resource mask. Below is a simple solution to add 1s around any existing 1s in resource_mask
        init_mask = np.zeros((self.height + 2, self.width + 2), dtype=bool)
        for delta in np.array([[0,0],[0, 1], [0, -1], [1, 1], [1,0], [1, -1], [-1, -1], [-1, 0], [-1, 1]]):
            s0 = delta[0] + 1
            s1 = delta[1] + 1
            e0 = -1 + delta[0]
            e1 = -1 + delta[1]
            if e0 == 0: e0 = None
            if e1 == 0: e1 = None
            init_mask[s0:e0, s1:e1] = init_mask[s0:e0, s1:e1] | resource_mask
        resource_mask = init_mask[1:-1,1:-1]
        self.valid_spawns_mask[resource_mask] = False
        self.valid_spawns_mask[0] = False
        self.valid_spawns_mask[-1] = False
        self.valid_spawns_mask[:,0] = False
        self.valid_spawns_mask[:,-1] = False
        # grow the 0s by 1 radius

    def pos_hash(self, pos: Position):
        return f"{pos.x},{pos.y}"
    def get_units_at(self, pos: Position):
        pos_hash = self.pos_hash(pos)
        if pos_hash in self.units_map:
            return self.units_map[pos_hash]
        return None
    def get_factory_at(self, state: State, pos: Position):
        f_id = self.factory_occupancy_map[pos.x, pos.y]
        if f_id != -1:
            unit_id = f"factory_{f_id}"
            if unit_id in state.factories["player_0"]:
                return state.factories["player_0"][unit_id]
            else:
                return state.factories["player_1"][unit_id]
        return None

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
            valid_spawns_mask=self.valid_spawns_mask.copy(),
            factories_per_team=self.factories_per_team,
        )
