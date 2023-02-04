import random
from typing import Any, List

import numpy as np
from gym import spaces

from luxai_s2.config import EnvConfig
from luxai_s2.spaces.act_space import ActionsQueue, FactionString


class DynamicArray(spaces.Space):
    def __init__(self, element_space: spaces.Space, max_length: int) -> None:
        super().__init__((), float)
        self.max_length = max_length
        self.element_space = element_space

    def sample(self):
        queue_size = self.np_random.randint(0, self.max_length) + 1
        arr = []
        for _ in range(queue_size):
            arr.append(self.element_space.sample())
        return arr

    def contains(self, x: Any) -> bool:
        if isinstance(x, list) and len(x) > self.max_length:
            return False
        elif (
            isinstance(x, np.ndarray) and len(x.shape) == 2 and len(x) > self.max_length
        ):
            return False
        elif isinstance(x, np.ndarray) and len(x.shape) == 1:
            x = [x]
        # if (not isinstance(x, list) and not isinstance(x, np.ndarray)) or len(x) > self.max_length:
        # return False
        for a in x:
            # fix issue where multidiscrete space does not do 100% check on type of value
            try:
                if not self.element_space.contains(a):
                    return False
            except:
                return False
        return True


class UnitTypeSpace(spaces.Space):
    def __init__(
        self,
    ):
        pass

    def sample(self):
        return random.choice(["LIGHT", "HEAVY"])

    def contains(self, x):
        return type(x) == "str" and (x == "LIGHT" or x == "HEAVY")


class FactoryIDSpace(spaces.Space):
    def __init__(
        self,
    ):
        pass

    def sample(self):
        return "factory_x"

    def contains(self, x):
        return type(x) == "str" and "factory" in x


class UnitIDSpace(spaces.Space):
    def __init__(
        self,
    ):
        pass

    def sample(self):
        return "unit_x"

    def contains(self, x):
        return type(x) == "str" and "unit" in x


def get_obs_space(config: EnvConfig, agent_names: List[str], agent: int = 0):
    max_space = 2**32 - 1
    obs_space = dict()

    obs_space["real_env_steps"] = spaces.Discrete(config.max_episode_length)
    obs_space["global_id"] = spaces.Discrete(99999999)

    # teams obs space
    teams_obs_space = dict()
    for agent_name in agent_names:
        teams_obs_space[agent_name] = spaces.Dict(
            team_id=spaces.Discrete(2),
            faction=FactionString(),
            water=spaces.Discrete(max_space),
            metal=spaces.Discrete(max_space),
            factories_to_place=spaces.Discrete(config.MAX_FACTORIES),
            factory_strains=DynamicArray(
                spaces.Discrete(max_space), max_length=(config.MAX_FACTORIES + 1)
            ),
            place_first=spaces.Discrete(2),
            bid=spaces.Discrete(150),
        )
    obs_space["teams"] = spaces.Dict(teams_obs_space)

    # board obs space
    map_shape = (config.map_size, config.map_size)
    board_obs_space = dict(
        ice=spaces.Box(low=0, high=1, shape=map_shape, dtype=int),
        ore=spaces.Box(low=0, high=1, shape=map_shape, dtype=int),
        rubble=spaces.Box(low=0, high=config.MAX_RUBBLE, shape=map_shape, dtype=int),
        lichen=spaces.Box(
            low=0, high=config.MAX_LICHEN_PER_TILE, shape=map_shape, dtype=int
        ),
        lichen_strains=spaces.Box(low=-1, high=99999, shape=map_shape, dtype=int),
        valid_spawns_mask=spaces.Box(low=0, high=1, shape=map_shape, dtype=int),
        factories_per_team=spaces.Discrete(config.MAX_FACTORIES),
    )
    obs_space["board"] = spaces.Dict(board_obs_space)

    # Unit obs space
    units_obs_space = dict()
    for agent_name in agent_names:
        # defines what each unit's info looks like since its variable
        # up to user to do any kind of padding or reshaping
        obs_dict = dict(
            power=spaces.Discrete(config.ROBOTS["HEAVY"].BATTERY_CAPACITY + 1),
            pos=spaces.Box(0, config.map_size - 1, shape=(2,), dtype=int),
            cargo=spaces.Dict(
                ice=spaces.Discrete(config.ROBOTS["HEAVY"].CARGO_SPACE + 1),
                water=spaces.Discrete(config.ROBOTS["HEAVY"].CARGO_SPACE + 1),
                ore=spaces.Discrete(config.ROBOTS["HEAVY"].CARGO_SPACE + 1),
                metal=spaces.Discrete(config.ROBOTS["HEAVY"].CARGO_SPACE + 1),
            ),
            unit_id=UnitIDSpace(),
            team_id=spaces.Discrete(2),
            unit_type=UnitTypeSpace(),
        )
        if config.UNIT_ACTION_QUEUE_SIZE != 1:
            # same as unit action space
            obs_dict["action_queue"] = ActionsQueue(
                spaces.Box(
                    low=np.array([0, 0, 0, 0, 0, 1]),
                    high=np.array([5, 4, 4, config.max_transfer_amount + 1, 9999, 9999]),
                    shape=(6,),
                    dtype=np.int64,
                ),
                config.UNIT_ACTION_QUEUE_SIZE,
            )
        units_obs_space[agent_name] = spaces.Dict(obs_dict)

    obs_space["units"] = spaces.Dict(units_obs_space)

    # Factory obs space
    factories_obs_space = dict()
    for agent_name in agent_names:
        # defines what each factory's info looks like since its variable
        # up to user to do any kind of padding or reshaping
        obs_dict = dict(
            power=spaces.Discrete(max_space),
            pos=spaces.Box(0, config.map_size - 1, shape=(2,), dtype=int),
            cargo=spaces.Dict(
                ice=spaces.Discrete(max_space),
                water=spaces.Discrete(max_space),
                ore=spaces.Discrete(max_space),
                metal=spaces.Discrete(max_space),
            ),
            unit_id=FactoryIDSpace(),
            team_id=spaces.Discrete(2),
            strain_id=spaces.Discrete(max_space),
        )
        factories_obs_space[agent_name] = spaces.Dict(obs_dict)
    obs_space["factories"] = spaces.Dict(factories_obs_space)

    return spaces.Dict(obs_space)


if __name__ == "__main__":
    cfg = EnvConfig()
    obs_space = get_obs_space(cfg, ["player_0", "player_1"])
    import ipdb

    ipdb.set_trace()
