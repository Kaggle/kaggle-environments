# TODO (stao): Add lux ai s3 env to gymnax api wrapper, which is the old gym api
import json
import os
from typing import Any, SupportsFloat
import flax
import flax.serialization
import gymnasium as gym
import gymnax
import gymnax.environments.spaces
import jax
import numpy as np
import dataclasses
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams, env_params_ranges
from luxai_s3.state import serialize_env_actions, serialize_env_states
from luxai_s3.utils import to_numpy


class LuxAIS3GymEnv(gym.Env):
    def __init__(self, numpy_output: bool = False):
        self.numpy_output = numpy_output
        self.rng_key = jax.random.key(0)
        self.jax_env = LuxAIS3Env(auto_reset=False)
        self.env_params: EnvParams = EnvParams()

        low = np.zeros((self.env_params.max_units, 3))
        low[:, 1:] = -self.env_params.unit_sap_range
        high = np.ones((self.env_params.max_units, 3)) * 6
        high[:, 1:] = self.env_params.unit_sap_range
        self.action_space = gym.spaces.Dict(
            dict(
                player_0=gym.spaces.Box(low=low, high=high, dtype=np.int16),
                player_1=gym.spaces.Box(low=low, high=high, dtype=np.int16),
            )
        )

    def render(self):
        self.jax_env.render(self.state, self.env_params)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            self.rng_key = jax.random.key(seed)
        self.rng_key, reset_key = jax.random.split(self.rng_key)
        # generate random game parameters
        # TODO (stao): check why this keeps recompiling when marking structs as static args
        randomized_game_params = dict()
        for k, v in env_params_ranges.items():
            self.rng_key, subkey = jax.random.split(self.rng_key)
            randomized_game_params[k] = jax.random.choice(
                subkey, jax.numpy.array(v)
            ).item()
        params = EnvParams(**randomized_game_params)
        if options is not None and "params" in options:
            params = options["params"]

        self.env_params = params
        obs, self.state = self.jax_env.reset(reset_key, params=params)
        if self.numpy_output:
            obs = to_numpy(flax.serialization.to_state_dict(obs))

        # only keep the following game parameters available to the agent
        params_dict = dataclasses.asdict(params)
        params_dict_kept = dict()
        for k in [
            "max_units",
            "match_count_per_episode",
            "max_steps_in_match",
            "map_height",
            "map_width",
            "num_teams",
            "unit_move_cost",
            "unit_sap_cost",
            "unit_sap_range",
            "unit_sensor_range",
        ]:
            params_dict_kept[k] = params_dict[k]
        return obs, dict(
            params=params_dict_kept, full_params=params_dict, state=self.state
        )

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self.rng_key, step_key = jax.random.split(self.rng_key)
        obs, self.state, reward, terminated, truncated, info = self.jax_env.step(
            step_key, self.state, action, self.env_params
        )
        if self.numpy_output:
            obs = to_numpy(flax.serialization.to_state_dict(obs))
            reward = to_numpy(reward)
            terminated = to_numpy(terminated)
            truncated = to_numpy(truncated)
            # info = to_numpy(flax.serialization.to_state_dict(info))
        return obs, reward, terminated, truncated, info


# TODO: vectorized gym wrapper


class RecordEpisode(gym.Wrapper):
    def __init__(
        self,
        env: LuxAIS3GymEnv,
        save_dir: str = None,
        save_on_close: bool = True,
        save_on_reset: bool = True,
    ):
        super().__init__(env)
        self.episode = dict(states=[], actions=[], metadata=dict())
        self.episode_id = 0
        self.save_dir = save_dir
        self.save_on_close = save_on_close
        self.save_on_reset = save_on_reset
        self.episode_steps = 0
        if save_dir is not None:
            from pathlib import Path

            Path(save_dir).mkdir(parents=True, exist_ok=True)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        if self.save_on_reset and self.episode_steps > 0:
            self._save_episode_and_reset()
        obs, info = self.env.reset(seed=seed, options=options)

        self.episode["metadata"]["seed"] = seed
        self.episode["params"] = flax.serialization.to_state_dict(info["full_params"])
        self.episode["states"].append(info["state"])
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        self.episode["states"].append(info["final_state"])
        self.episode["actions"].append(action)
        return obs, reward, terminated, truncated, info

    def serialize_episode_data(self, episode=None):
        if episode is None:
            episode = self.episode
        ret = dict()
        ret["observations"] = serialize_env_states(episode["states"])
        if "actions" in episode:
            ret["actions"] = serialize_env_actions(episode["actions"])
        ret["metadata"] = episode["metadata"]
        ret["params"] = episode["params"]
        return ret

    def save_episode(self, save_path: str):
        episode = self.serialize_episode_data()
        with open(save_path, "w") as f:
            json.dump(episode, f)
        self.episode = dict(states=[], actions=[], metadata=dict())

    def _save_episode_and_reset(self):
        """saves to generated path based on self.save_dir and episoe id and updates relevant counters"""
        self.save_episode(
            os.path.join(self.save_dir, f"episode_{self.episode_id}.json")
        )
        self.episode_id += 1
        self.episode_steps = 0

    def close(self):
        if self.save_on_close and self.episode_steps > 0:
            self._save_episode_and_reset()
