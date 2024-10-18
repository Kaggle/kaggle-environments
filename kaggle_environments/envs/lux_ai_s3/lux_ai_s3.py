import dataclasses
import json
import math
import os
import random
import sys
from os import path
from .agents import all_agents
from sys import path as syspath
from os import path as osp

# next two lines enables importing local packages e.g. luxai_s2
__dir__ = osp.dirname(__file__)
syspath.append(__dir__)


# import vec_noise

from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from luxai_s3.state import serialize_env_actions, serialize_env_states
import numpy as np

import copy
import json
def to_json(state):
    if isinstance(state, np.ndarray):
        return state.tolist()
    elif isinstance(state, np.int64):
        return state.tolist()
    elif isinstance(state, list):
        return [to_json(s) for s in state]
    elif isinstance(state, dict):
        out = {}
        for k in state:
            out[k] = to_json(state[k])
        return out
    else:
        return state
prev_step = 0
luxenv: RecordEpisode = None # LuxAIS3GymEnv(numpy_output=True)
prev_obs = None
state_obs = None
default_env_cfg = None
def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()
    
def interpreter(state, env):
    global luxenv, prev_obs, state_obs, default_env_cfg
    player_0 = state[0]
    player_1 = state[1]
    # filter out actions such as debug annotations so they aren't saved
    # filter_actions(state, env)
    ### 1.2: Initialize a blank state game if new episode is starting ###
    if env.done:
        if "seed" in env.configuration:
            seed = int(env.configuration["seed"])
        else:
            seed = math.floor(random.random() * 1e9);
            env.configuration["seed"] = seed
        if "max_episode_length" in env.configuration:
            max_episode_length = int(env.configuration["max_episode_length"])
        else:
            max_episode_length = 505

        
        if default_env_cfg is None:
            # if this is the first time creating the environment, env.configuration contains the kaggle competition configurations used to override 
            # the default env config in LuxAI_S2. env.configuration is later populated by the merge of the kaggle competition configs with the 
            # env config in LuxAI_S2 so we only run this branch once and save the result.
            parsed_env_config = copy.deepcopy(env.configuration)
            parsed_env_config["max_episode_length"] = max_episode_length
            delete_keys = ["seed", "episodeSteps", "actTimeout", "runTimeout", "env_cfg"]
            env_cfg_override = dict()
            if "env_cfg" in parsed_env_config:
                env_cfg_override = copy.deepcopy(parsed_env_config["env_cfg"])
            for k in delete_keys:
                if k in parsed_env_config: del parsed_env_config[k]
            parsed_env_config = {**parsed_env_config, **env_cfg_override}
            default_env_cfg = parsed_env_config
        else:
            parsed_env_config = default_env_cfg
        # luxenv = LuxAIS3GymEnv(numpy_output=True, **parsed_env_config)
        luxenv = LuxAIS3GymEnv(numpy_output=True)
        luxenv = RecordEpisode(luxenv, save_on_close=False, save_on_reset=False)
        obs, info = luxenv.reset(seed=seed)

        env_cfg_json = info["params"]

        env.configuration.env_cfg = env_cfg_json
        
        player_0.observation.player = "player_0"
        player_1.observation.player = "player_1"
        player_0.observation.obs = json.dumps(to_json(obs["player_0"]))
        player_1.observation.obs = json.dumps(to_json(obs["player_1"]))
        
        replay_frame = luxenv.serialize_episode_data(dict(
            states=[luxenv.episode["states"][-1]],
            metadata=luxenv.episode["metadata"],
            params=luxenv.episode["params"]
        ))
        # don't need to keep metadata/params beyond first step
        player_0.info = dict(replay=replay_frame)
        return state
    
    new_state_obs, rewards, terminations, truncations, infos = luxenv.step({
        "player_0": np.array(player_0.action["action"]),
        "player_1": np.array(player_1.action["action"])
    })
    
    # cannot store np arrays in replay jsons so must convert to list
    player_0.action = player_0.action["action"]
    player_1.action = player_1.action["action"]
    
    dones = dict()
    for k in terminations:
        dones[k] = terminations[k] | truncations[k]

    player_0.observation.player = "player_0"
    player_1.observation.player = "player_1"

    player_0.observation.obs = json.dumps(to_json(new_state_obs["player_0"]))
    player_1.observation.obs = json.dumps(to_json(new_state_obs["player_1"]))
    

    player_0.reward = int(rewards["player_0"])
    player_1.reward = int(rewards["player_1"])

    player_0.observation.reward = int(player_0.reward)
    player_1.observation.reward = int(player_1.reward)
    replay_frame = luxenv.serialize_episode_data(dict(
        states=[luxenv.episode["states"][-2]],
        actions=[luxenv.episode["actions"][-1]],
        metadata=luxenv.episode["metadata"],
        params=luxenv.episode["params"]
    ))
    # don't need to keep metadata/params beyond first step
    del replay_frame["metadata"]
    del replay_frame["params"]
    player_0.info = dict(replay=replay_frame)

    if np.all([dones[k] for k in dones]):
        if player_0.status == "ACTIVE":
            player_0.status = "DONE"
        if player_1.status == "ACTIVE":
            player_1.status = "DONE"
    return state

def renderer(state, env):
    raise NotImplementedError("To render the replay, please set the render mode to json or html")


dir_path = path.dirname(__file__)
json_path = path.abspath(path.join(dir_path, "lux_ai_s3.json"))
with open(json_path) as json_file:
    specification = json.load(json_file)


def html_renderer():
    html_path = path.abspath(path.join(dir_path, "index.html"))
    return ("html_path", html_path)

agents = all_agents