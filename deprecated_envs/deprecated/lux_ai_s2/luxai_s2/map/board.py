from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List
try:
    from typing import TypedDict    
except:
    from typing_extensions import TypedDict

import numpy as np
import numpy.typing as npt

from luxai_s2.config import EnvConfig

if TYPE_CHECKING:
    from luxai_s2.factory import Factory
    from luxai_s2.state import State

from luxai_s2.map.position import Position
from luxai_s2.map_generator.generator import GameMap
from luxai_s2.unit import Unit


class BoardStateDict(TypedDict):
    rubble: npt.NDArray[np.int_]
    ore: npt.NDArray[np.bool_]
    ice: npt.NDArray[np.bool_]
    lichen: npt.NDArray[np.int_]
    lichen_strains: npt.NDArray[np.int_]
    valid_spawns_mask: npt.NDArray[np.bool_]
    factories_per_team: int


class Board:
    def __init__(
        self, seed=None, env_cfg: EnvConfig = None, existing_map: GameMap = None
    ) -> None:
        self.env_cfg = env_cfg
        self.height = self.env_cfg.map_size
        self.width = self.env_cfg.map_size
        self.seed = seed
        rng = np.random.RandomState(seed=seed)
        if existing_map is None:
            self.gen_map(seed, rng, env_cfg)
        else:
            self.map = existing_map
        self.post_map_gen(env_cfg, rng)

    def gen_map(self, seed, rng, env_cfg):

        map_type = rng.choice(["Cave", "Mountain"])
        map_distribution_type = rng.choice(
            [
                "high_ice_high_ore",
                "high_ice_low_ore",
                "low_ice_high_ore",
                "low_ice_low_ore",
            ]
        )
        symmetry = rng.choice(["horizontal", "vertical"])
        self.map = GameMap.random_map(
            seed=seed,
            symmetry=symmetry,
            map_type=map_type,
            map_distribution_type=map_distribution_type,
            width=self.width,
            height=self.height,
        )
        # remove bottom once generator is fully ready
        self.map.rubble = self.map.rubble.astype(int)
        self.map.ore = self.map.ore.astype(int)
        self.map.ice = self.map.ice.astype(int)

    def post_map_gen(self, env_cfg, rng):
        self.factories_per_team = rng.randint(
            env_cfg.MIN_FACTORIES, env_cfg.MAX_FACTORIES + 1
        )

        self.lichen = np.zeros((self.height, self.width), dtype=int)
        # ownership of lichen by factory id, a simple mask
        # -1 = no ownership
        self.lichen_strains = -np.ones((self.height, self.width), dtype=int)
        # self.units_map: np.ndarray = np.zeros((self.height, self.width))
        self.units_map: Dict[str, List[Unit]] = defaultdict(list)
        # TODO consider unit occupancy map, may speed up computation by storing more info in a form indexable in numpy arrays

        # maps center of factory to the factory
        self.factory_map: Dict[str, "Factory"] = dict()
        # != -1 if a factory tile is on the location. Equals factory number id
        self.factory_occupancy_map = -np.ones((self.height, self.width), dtype=int)

        # Valid spawn locations
        self.valid_spawns_mask = np.ones((self.height, self.width), dtype=bool)
        resource_mask = (self.map.ice != 0) | (self.map.ore != 0)
        # generate all 9 shifts of the resource mask. Below is a simple solution to add 1s around any existing 1s in resource_mask
        init_mask = np.zeros((self.height + 2, self.width + 2), dtype=bool)
        for delta in np.array(
            [
                [0, 0],
                [0, 1],
                [0, -1],
                [1, 1],
                [1, 0],
                [1, -1],
                [-1, -1],
                [-1, 0],
                [-1, 1],
            ]
        ):
            s0 = delta[0] + 1
            s1 = delta[1] + 1
            e0 = -1 + delta[0]
            e1 = -1 + delta[1]
            if e0 == 0:
                e0 = None
            if e1 == 0:
                e1 = None
            init_mask[s0:e0, s1:e1] = init_mask[s0:e0, s1:e1] | resource_mask
        resource_mask = init_mask[1:-1, 1:-1]
        self.valid_spawns_mask[resource_mask] = False
        self.valid_spawns_mask[0] = False
        self.valid_spawns_mask[-1] = False
        self.valid_spawns_mask[:, 0] = False
        self.valid_spawns_mask[:, -1] = False
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

    def state_dict(self) -> BoardStateDict:
        return dict(
            rubble=self.rubble.copy(),
            ore=self.ore.copy(),
            ice=self.ice.copy(),
            lichen=self.lichen.copy(),
            lichen_strains=self.lichen_strains.copy(),
            valid_spawns_mask=self.valid_spawns_mask.copy(),
            factories_per_team=self.factories_per_team,
        )
