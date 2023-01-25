from typing import Callable, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces

import luxai_s2.env
from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict
from luxai_s2.unit import ActionType, BidActionType, FactoryPlacementActionType
from luxai_s2.utils import my_turn_to_place_factory
from luxai_s2.wrappers.controllers import (
    Controller,
    SimpleDiscreteController,
    SimpleSingleUnitDiscreteController,
)


class SingleUnitObservationWrapper(gym.ObservationWrapper):
    """
    A state based observation to work with in pair with the SimpleSingleUnitDiscreteController

    It contains info only on the first heavy unit, the first factory you own, and some useful features. If there are no owned heavy units the observation is just zero.
    No information about the opponent is included


    Included features:
    - First heavy unit's stats
    - Position of closest ice tile
    - First factory

    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(-999, 999, shape=(13,))

    def observation(
        self, obs: Dict[str, ObservationStateDict]
    ) -> Dict[str, npt.NDArray]:
        observation = dict()
        shared_obs = obs["player_0"]
        ice_map = shared_obs["board"]["ice"]
        ice_tile_locations = np.argwhere(ice_map == 1)
        for agent in obs.keys():
            factories = shared_obs["factories"][agent]
            factory_vec = np.zeros(2)
            for k in factories.keys():
                factory = factories[k]
                factory_vec = np.array(factory["pos"]) / self.env.state.env_cfg.map_size
                break
            units = shared_obs["units"][agent]
            for k in units.keys():
                unit = units[k]
                cargo_space = self.env.state.env_cfg.ROBOTS[
                    unit["unit_type"]
                ].CARGO_SPACE
                battery_cap = self.env.state.env_cfg.ROBOTS[
                    unit["unit_type"]
                ].BATTERY_CAPACITY
                cargo_vec = np.array(
                    [
                        unit["power"] / battery_cap,
                        unit["cargo"]["ice"] / cargo_space,
                        unit["cargo"]["ore"] / cargo_space,
                        unit["cargo"]["water"] / cargo_space,
                        unit["cargo"]["metal"] / cargo_space,
                    ]
                )
                unit_type = (
                    0 if unit["unit_type"] == "LIGHT" else 1
                )  # note that build actions use 0 to encode Light

                pos = np.array(unit["pos"]) / self.env.state.env_cfg.map_size
                unit_vec = np.concatenate(
                    [pos, [unit_type], cargo_vec, [unit["team_id"]]], axis=-1
                )
                # engineered features

                # compute closest ice tile
                ice_tile_distances = np.mean(
                    (ice_tile_locations - np.array(unit["pos"])) ** 2, 1
                )
                closest_ice_tile = (
                    ice_tile_locations[np.argmin(ice_tile_distances)]
                    / self.env.state.env_cfg.map_size
                )
                obs_vec = np.concatenate(
                    [unit_vec, factory_vec - pos, closest_ice_tile - pos], axis=-1
                )

                observation[agent] = obs_vec
                break
            if agent not in observation:
                observation[agent] = np.zeros(13)
        return observation


class ImageObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        obs_dims = 23  # see _convert_obs function for how this is computed
        self.map_size = self.env.env_cfg.map_size
        self.observation_space = spaces.Box(
            -999, 999, shape=(self.map_size, self.map_size, obs_dims)
        )

    def observation(
        self, obs: Dict[str, ObservationStateDict]
    ) -> Dict[str, npt.NDArray]:
        shared_obs = obs["player_0"]
        unit_mask = np.zeros((self.map_size, self.map_size, 1))
        unit_data = np.zeros(
            (self.map_size, self.map_size, 9)
        )  # power(1) + cargo(4) + unit_type(1) + unit_pos(2) + team(1)
        factory_mask = np.zeros_like(unit_mask)
        factory_data = np.zeros(
            (self.map_size, self.map_size, 8)
        )  # power(1) + cargo(4) + factory_pos(2) + team(1)
        for agent in obs.keys():
            factories = shared_obs["factories"][agent]
            units = shared_obs["units"][agent]

            for unit_id in units.keys():
                unit = units[unit_id]
                # we encode everything but unit_id or action queue
                cargo_space = self.env.state.env_cfg.ROBOTS[
                    unit["unit_type"]
                ].CARGO_SPACE
                battery_cap = self.env.state.env_cfg.ROBOTS[
                    unit["unit_type"]
                ].BATTERY_CAPACITY
                cargo_vec = np.array(
                    [
                        unit["power"] / battery_cap,
                        unit["cargo"]["ice"] / cargo_space,
                        unit["cargo"]["ore"] / cargo_space,
                        unit["cargo"]["water"] / cargo_space,
                        unit["cargo"]["metal"] / cargo_space,
                    ]
                )
                unit_type = (
                    0 if unit["unit_type"] == "LIGHT" else 1
                )  # note that build actions use 0 to encode Light
                unit_vec = np.concatenate(
                    [unit["pos"], [unit_type], cargo_vec, [unit["team_id"]]], axis=-1
                )
                unit_vec[:2] /= self.env.state.env_cfg.map_size

                # note that all data is stored as map[x, y] format
                unit_data[unit["pos"][0], unit["pos"][1]] = unit_vec
                unit_mask[unit["pos"][0], unit["pos"][1]] = 1

            for unit_id in factories.keys():
                factory = factories[unit_id]
                # we encode everything but strain_id or unit_id
                cargo_vec = np.array(
                    [
                        factory["power"],
                        factory["cargo"]["ice"],
                        factory["cargo"]["ore"],
                        factory["cargo"]["water"],
                        factory["cargo"]["metal"],
                    ]
                )
                cargo_vec = cargo_vec * 1 / 1000

                factory_vec = np.concatenate(
                    [factory["pos"], cargo_vec, [factory["team_id"]]], axis=-1
                )
                factory_vec[:2] /= self.env.state.env_cfg.map_size
                factory_data[factory["pos"][0], factory["pos"][1]] = factory_vec
                factory_mask[factory["pos"][0], factory["pos"][1]] = 1

            image_features = np.concatenate(
                [
                    np.expand_dims(shared_obs["board"]["lichen"], -1)
                    / self.env.state.env_cfg.MAX_LICHEN_PER_TILE,
                    np.expand_dims(shared_obs["board"]["rubble"], -1)
                    / self.env.state.env_cfg.MAX_RUBBLE,
                    np.expand_dims(shared_obs["board"]["ice"], -1),
                    np.expand_dims(shared_obs["board"]["ore"], -1),
                    unit_mask,
                    unit_data,
                    factory_mask,
                    factory_data,
                ],
                axis=-1,
            )

        new_obs = dict()
        for agent in self.all_agents:
            new_obs[agent] = image_features
        return new_obs
