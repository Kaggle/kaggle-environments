import random
from typing import Any, Dict, List

import gym
import numpy as np
from gym import spaces

from luxai_s2.config import EnvConfig
from luxai_s2.factory import Factory
from luxai_s2.team import FactionTypes
from luxai_s2.unit import Unit


class FactionString(gym.Space):
    def __init__(
        self,
    ):
        self.valid_factions = ["Null"] + [x.name for x in FactionTypes]

    def sample(self):
        return random.choice(self.valid_factions)

    def contains(self, x):
        return type(x) == "str" and x in self.valid_factions


class PartialDict(spaces.Dict):
    def contains(self, x: Any) -> bool:
        if not isinstance(x, dict) or len(x) > len(self.spaces):
            return False, "invalid"
        for k, v in x.items():
            if k in self.spaces:
                space = self.spaces[k]
                if not space.contains(v):
                    return False, f"invalid action {v} for {k}"
            else:
                return False, f"{k} is not on your team or does not exist"
        return True, None


class ActionsQueue(spaces.Space):
    def __init__(self, action_space: spaces.Space, max_length: int) -> None:
        super().__init__((), float)
        self.max_length = max_length
        self.action_space = action_space

    def sample(self):
        queue_size = self.np_random.randint(0, self.max_length) + 1
        action_q = []
        for _ in range(queue_size):
            action_q.append(self.action_space.sample())
        return action_q

    def contains(self, x: Any) -> bool:
        if isinstance(x, list) and len(x) > self.max_length:
            return False
        elif (
            isinstance(x, np.ndarray) and len(x.shape) == 2 and len(x) > self.max_length
        ):
            return False
        elif isinstance(x, np.ndarray) and x.shape[0] == 0:
            # empty action
            return True
        elif isinstance(x, np.ndarray) and len(x.shape) == 1:
            x = [x]
        # if (not isinstance(x, list) and not isinstance(x, np.ndarray)) or len(x) > self.max_length:
        # return False
        for a in x:
            # fix issue where multidiscrete space does not do 100% check on type of value
            try:
                if not self.action_space.contains(a):
                    return False
            except:
                return False
        return True


def get_act_space_init(config: EnvConfig, agent: str):
    # Get action space for turn 0 initialization
    act_space = dict()
    act_space["faction"] = FactionString()
    act_space["spawns"] = spaces.Box(
        low=0, high=config.map_size, shape=(config.MAX_FACTORIES, 2), dtype=int
    )
    return spaces.Dict(act_space)


def get_act_space_bid(config: EnvConfig, agent: str):
    act_space = dict()
    act_space["faction"] = FactionString()
    act_space["bid"] = spaces.Discrete(100000)
    return spaces.Dict(act_space)


def get_act_space_placement(config: EnvConfig, agent: str):
    # Get action space for turn 0 initialization
    act_space = dict()
    act_space["spawn"] = spaces.Box(low=0, high=config.map_size, shape=(2,), dtype=int)
    act_space["water"] = spaces.Discrete(100000)
    act_space["metal"] = spaces.Discrete(100000)
    return spaces.Dict(act_space)


def get_act_space(
    units: Dict[str, Dict[str, Unit]],
    factories: Dict[str, Dict[str, Factory]],
    config: EnvConfig,
    agent: str,
):
    act_space = dict()

    # for consistency, every action space per unit is fixed, makes it easier to work out of the box.
    # we further annotate dimensions, in many places for clarity

    # for those who are programming rule-based bots, we provide helper functions to generate the action vector to store in an actions dict, as well as generate
    # a human readable version of an action vector

    # TODO - verify speed of building action spaces like this.
    for u in units[agent].values():
        # Each action for any mobile unit, (light, heavy) is an array A of shape (max_queue, 6)
        # if max_queue is 1, we remove this dimension
        # let a be some element in A
        # Then

        # a[0] = action type
        # (0 = move, 1 = transfer X amount of R, 2 = pickup X amount of R, 3 = dig, 4 = self destruct, 5 = recharge X)

        # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)

        # a[2] = R = resource type (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 power)

        # a[3] = X, amount of resources transferred or picked up if action is transfer or pickup.
        # If action is recharge, it is how much energy to store before executing the next action in queue

        # a[4] = repeat. If repeat == 0, then action is not recycled and removed once we have executed it a[5] = n times. 
        # Otherwise if repeat > 0 we recycle this action to the back of the action queue and set n = repeat.

        # a[5] = n, number of times to execute this action before exhausting it and removing it from the front of the action queue. Minimum is 1.
        act_space[u.unit_id] = ActionsQueue(
            spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 1]),
                high=np.array([5, 4, 4, config.max_transfer_amount + 1, 9999, 9999]),
                shape=(6,),
                dtype=np.int64,
            ),
            config.UNIT_ACTION_QUEUE_SIZE,
        )

    for factory in factories[agent].values():
        # action type (0 = build light robot, 1 = build heavy robot, 2 = water lichen)
        act_space[factory.unit_id] = spaces.Discrete(3)

    return PartialDict(act_space)
