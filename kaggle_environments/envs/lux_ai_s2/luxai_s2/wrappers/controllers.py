from typing import Dict

import numpy as np
import numpy.typing as npt
from gym import spaces

from luxai_s2.actions import format_action_vec
from luxai_s2.config import EnvConfig
from luxai_s2.state import ObservationStateDict


class Controller:
    def __init__(self, action_space: spaces.Space) -> None:
        self.action_space = action_space

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, ObservationStateDict], action: npt.NDArray
    ):
        """
        Takes as input the current "raw observation" and the parameterized action and returns
        an action formatted for the Lux env
        """
        raise NotImplementedError()


class SimpleSingleUnitDiscreteController(Controller):
    def __init__(self, env_cfg: EnvConfig) -> None:
        """
        A simple controller that controls only the heavy unit that will get spawned. This assumes for whichever environment wrapper you use
        you have defined a policy to generate the first factory action

        For the heavy unit
        - 4 cardinal direction movement (4 dims)
        - a move center no-op action (1 dim)
        - transfer action just for transferring ice in 4 cardinal directions or center (5)
        - pickup action for each resource (5 dims)
        - dig action (1 dim)

        It does not include
        - self destruct action
        - recharge action
        - planning (via actions executing multiple times or repeating actions)
        - factory actions
        - transferring power or resources other than ice
        """
        self.env_cfg = env_cfg
        self.move_act_dims = 5
        self.transfer_act_dims = 5  # 5 * 5
        self.pickup_act_dims = 5
        self.dig_act_dims = 1

        self.move_dim_high = self.move_act_dims
        self.transfer_dim_high = self.move_dim_high + self.transfer_act_dims
        self.pickup_dim_high = self.transfer_dim_high + self.pickup_act_dims
        self.dig_dim_high = self.pickup_dim_high + self.dig_act_dims

        total_act_dims = self.dig_dim_high
        # action_space = spaces.Box(0, 1, shape=(total_act_dims,))
        action_space = spaces.Discrete(total_act_dims)
        super().__init__(action_space)

    def _is_move_action(self, id):
        return id < self.move_dim_high

    def _get_move_action(self, id):
        return np.array([0, id, 0, 0, 0, 1])

    def _is_transfer_action(self, id):
        return id < self.transfer_dim_high

    def _get_transfer_action(self, id):
        id = id - self.move_dim_high
        transfer_dir = id % 5
        # resource_type = id // 5
        return np.array([1, transfer_dir, 0, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_pickup_action(self, id):
        return id < self.pickup_dim_high

    def _get_pickup_action(self, id):
        id = id - self.transfer_dim_high
        return np.array([2, 0, id % 5, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_dig_action(self, id):
        return id < self.dig_dim_high

    def _get_dig_action(self, id):
        return np.array([3, 0, 0, 0, 0, 1])

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, ObservationStateDict], action: npt.NDArray
    ):
        shared_obs = obs["player_0"]
        lux_action = dict()
        factories = shared_obs["factories"][agent]
        units = shared_obs["units"][agent]
        for unit_id in units.keys():
            unit = units[unit_id]
            pos = unit["pos"]
            unit_related_action = action
            choice = action  # unit_related_action.argmax()
            action_queue = []
            if self._is_move_action(choice):
                action_queue = [self._get_move_action(choice)]
            elif self._is_transfer_action(choice):
                action_queue = [self._get_transfer_action(choice)]
            elif self._is_pickup_action(choice):
                action_queue = [self._get_pickup_action(choice)]

            elif self._is_dig_action(choice):
                action_queue = [self._get_dig_action(choice)]
            lux_action[unit_id] = action_queue
            # only control the first unit!
            break
        return lux_action


class SimpleDiscreteController(Controller):
    def __init__(self, env_cfg: EnvConfig) -> None:
        """
        A simple controller that uses a discrete action parameterization for Lux AI S2. It includes

        For units
        - 4 cardinal direction movement (4 dims)
        - a move center no-op action (1 dim)
        - transfer action for each combination of the (4 cardinal directions plus center) x (resource type or power) (5*5 = 25 dims)
        - pickup action for each resource (5 dims)
        - dig action (1 dim)

        For factories
        - all actions (build light, heavy, or water) (3 dims)


        It does not include
        - self destruct action
        - recharge action
        - planning (via actions executing multiple times or repeating actions)

        Sampling from this controller will always result in a valid action, albeit sometimes disastrous
        """
        self.env_cfg = env_cfg
        self.move_act_dims = 5
        self.transfer_act_dims = 5 * 5
        self.pickup_act_dims = 5
        self.dig_act_dims = 1
        # self.self_destruct_act_dims = 1
        # self.recharge_act_dims = 1
        self.factory_act_dims = 3  # 0 = light, 1 = heavy, 2 = water

        self.move_dim_high = self.move_act_dims
        self.transfer_dim_high = self.move_dim_high + self.transfer_act_dims
        self.pickup_dim_high = self.transfer_dim_high + self.pickup_act_dims
        self.dig_dim_high = self.pickup_dim_high + self.dig_act_dims

        self.factory_dim_high = 3  # self.dig_dim_high + self.factory_act_dims

        total_act_dims = self.factory_dim_high
        # action_space = spaces.Discrete(total_act_dims)
        action_space = spaces.Box(
            0, 1, shape=(env_cfg.map_size, env_cfg.map_size, total_act_dims)
        )

        super().__init__(action_space)

    # note that all the _is_x_action are meant to be called in a if, elseif... cascade/waterfall
    # to understand how _get_x_action works to map the parameterization back to the original action space see luxai_s2/actions.py
    def _is_move_action(self, id):
        return id < self.move_dim_high

    def _get_move_action(self, id):
        return np.array([0, id, 0, 0, 0, 1])

    def _is_transfer_action(self, id):
        return id < self.transfer_dim_high

    def _get_transfer_action(self, id):
        id = id - self.move_dim_high
        transfer_dir = id % 5
        resource_type = id // 5
        return np.array(
            [1, transfer_dir, resource_type, self.env_cfg.max_transfer_amount, 0, 1]
        )

    def _is_pickup_action(self, id):
        return id < self.pickup_dim_high

    def _get_pickup_action(self, id):
        id = id - self.transfer_dim_high
        return np.array([2, 0, id % 5, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_dig_action(self, id):
        return id < self.dig_dim_high

    def _get_dig_action(self, id):
        return np.array([3, 0, 0, 0, 0, 1])

    # def _is_self_destruct_action(self, id):
    # return id < self.move_act_dims + self.transfer_act_dims + self.self_destruct_dims
    # def _get_self_destruct_action(self, id):
    #     return [2, 0, 0, 0, 0, 1]

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, ObservationStateDict], action: npt.NDArray
    ):
        """
        Generate an action compatible with LuxAI_S2 engine for a single player
        """
        shared_obs = obs["player_0"]
        lux_action = dict()
        factories = shared_obs["factories"][agent]
        units = shared_obs["units"][agent]
        for unit_id in units.keys():
            unit = units[unit_id]
            pos = unit["pos"]
            action_here = action[pos[0], pos[1]]
            unit_related_action = action_here[
                : -self.factory_act_dims
            ]  # assuming factory action is always the final few dimensions
            choice = unit_related_action.argmax()
            action_queue = []
            # if self._is_move_action(choice):
            #     action_queue = [self._get_move_action(choice)]
            # elif self._is_transfer_action(choice):
            #     action_queue = [self._get_transfer_action(choice)]
            # elif self._is_pickup_action(choice):
            #     action_queue = [self._get_pickup_action(choice)]
            # elif self._is_dig_action(choice):
            #     action_queue = [self._get_dig_action(choice)]

            lux_action[unit_id] = action_queue

        for unit_id in factories.keys():
            factory = factories[unit_id]
            pos = factory["pos"]

            action_here = action[pos[0], pos[1]]
            factory_related_action = action_here[
                -self.factory_act_dims :
            ]  # assuming factory action is always the final few dimensions
            choice = factory_related_action.argmax()
            lux_action[unit_id] = choice
        return lux_action
