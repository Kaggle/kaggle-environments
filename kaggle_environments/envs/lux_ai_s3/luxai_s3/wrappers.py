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
        
        # auto run compiling steps here:
        # print("Running compilation steps")
        key = jax.random.key(0)
        # Reset the environment
        dummy_env_params = EnvParams(map_type=0)
        key, reset_key = jax.random.split(key)
        obs, state = self.jax_env.reset(reset_key, params=dummy_env_params)
        # Take a random action
        key, subkey = jax.random.split(key)
        action = self.jax_env.action_space(dummy_env_params).sample(subkey)
        # Step the environment and compile. Not sure why 2 steps? are needed
        for _ in range(2):
            key, subkey = jax.random.split(key)
            obs, state, reward, terminated, truncated, info = self.jax_env.step(
                subkey, state, action, params=dummy_env_params
            )
        # print("Finish compilation steps")
        self.action_space = gym.spaces.Dict(dict(
            player_0=gym.spaces.MultiDiscrete(np.ones(self.env_params.max_units) * 5),
            player_1=gym.spaces.MultiDiscrete(np.ones(self.env_params.max_units) * 5)
        ))
    
    def render(self):
        self.jax_env.render(self.state, self.env_params)
        
        
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            self.rng_key = jax.random.key(seed)
        self.rng_key, reset_key = jax.random.split(self.rng_key)
        # generate random game parameters
        # TODO (stao): check why this keeps recompiling when marking structs as static args
        params = EnvParams(max_steps_in_match=50)
        if options is not None and "params" in options:
            params = options["params"]
        
        self.env_params = params
        obs, self.state = self.jax_env.reset(reset_key, params=params)
        if self.numpy_output:
            obs = to_numpy(flax.serialization.to_state_dict(obs))
        return obs, dict(params=params, state=self.state)
    
    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self.rng_key, step_key = jax.random.split(self.rng_key)
        obs, self.state, reward, terminated, truncated, info = self.jax_env.step(step_key, self.state, action, self.env_params)
        if self.numpy_output:
            # obs = to_numpy(obs)
            obs = to_numpy(flax.serialization.to_state_dict(obs))
            reward = to_numpy(reward)
            terminated = to_numpy(terminated)
            truncated = to_numpy(truncated)
            # info = to_numpy(flax.serialization.to_state_dict(info))
        return obs, reward, terminated, truncated, info

# TODO: vectorized gym wrapper

class RecordEpisode(gym.Wrapper):
    def __init__(self, env: LuxAIS3GymEnv, save_dir: str = None, save_on_close: bool = True, save_on_reset: bool = True):
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

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        if self.save_on_reset and self.episode_steps > 0:
            self._save_episode_and_reset()
        obs, info = self.env.reset(seed=seed, options=options)
        
        self.episode["metadata"]["seed"] = seed
        self.episode["params"] = flax.serialization.to_state_dict(info["params"])
        self.episode["states"].append(info["state"])
        return obs, info
    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        self.episode["states"].append(info["final_state"])
        self.episode["actions"].append(action)
        return obs, reward, terminated, truncated, info
        
    def serialize_episode_data(self):
        episode = dict()
        episode["observations"] = serialize_env_states(self.episode["states"])
        episode["actions"] = serialize_env_actions(self.episode["actions"])
        episode["metadata"] = self.episode["metadata"]
        episode["params"] = self.episode["params"]
        return episode
    
    def save_episode(self, save_path: str):
        episode = self.serialize_episode_data()
        with open(save_path, "w") as f:
            json.dump(episode, f)
        self.episode = dict(states=[], actions=[], metadata=dict())
        
    def _save_episode_and_reset(self):
        """saves to generated path based on self.save_dir and episoe id and updates relevant counters"""
        self.save_episode(os.path.join(self.save_dir, f"episode_{self.episode_id}.json"))
        self.episode_id += 1
        self.episode_steps = 0
        
    def close(self):
        if self.save_on_close and self.episode_steps > 0:
            self._save_episode_and_reset()
        