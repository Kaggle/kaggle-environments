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


import vec_noise

from luxai_s2.env import LuxAI_S2
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
luxenv: LuxAI_S2 = LuxAI_S2(verbose=0, validate_action_space=True, collect_stats=True)
prev_obs = None
state_obs = None
def cleanup_dimensions():
    global dimension_process
    if dimension_process is not None:
        dimension_process.kill()
def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()
def interpreter(state, env):
    global luxenv, prev_obs, state_obs
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
            max_episode_length = 1000

        parsed_env_config = copy.deepcopy(env.configuration)
        parsed_env_config["max_episode_length"] = max_episode_length
        delete_keys = ["seed", "episodeSteps", "actTimeout", "runTimeout", "env_cfg"]
        
        for k in delete_keys:
            if k in parsed_env_config: del parsed_env_config[k]
        luxenv = LuxAI_S2(validate_action_space=True, collect_stats=True, **parsed_env_config)
        _ = luxenv.reset(seed=seed)
        state_obs = luxenv.state.get_compressed_obs()

        env_cfg_json = dataclasses.asdict(luxenv.env_cfg)
        # del env_cfg_json["WEATHER_ID_TO_NAME"]
        env.configuration.env_cfg = env_cfg_json
        
        player_0.observation.player = "player_0"
        player_1.observation.player = "player_1"
        # TODO add observation optimizations here later
        player_0.observation.obs = json.dumps(to_json(state_obs))
        
        player_0.observation.width = luxenv.state.board.width
        player_0.observation.height = luxenv.state.board.height
        return state

    new_state_obs, rewards, dones, infos = luxenv.step({
        "player_0": player_0.action,
        "player_1": player_1.action
    })

    player_0.observation.player = "player_0"
    player_1.observation.player = "player_1"

    player_0.observation.obs = json.dumps(to_json(luxenv.state.get_change_obs(state_obs)))
    state_obs = new_state_obs["player_0"]
    
    player_0.observation.width = luxenv.state.board.width
    player_0.observation.height = luxenv.state.board.height

    player_0.reward = int(rewards["player_0"])
    player_1.reward = int(rewards["player_1"])
    player_0.observation.reward = int(player_0.reward)
    player_1.observation.reward = int(player_1.reward)
    if np.all([dones[k] for k in dones]):
        if player_0.status == "ACTIVE":
            player_0.status = "DONE"
        if player_1.status == "ACTIVE":
            player_1.status = "DONE"
        
        player_0.observation.stats = to_json(luxenv.state.stats["player_0"])
        player_1.observation.stats = to_json(luxenv.state.stats["player_1"])
    return state

def renderer(state, env):
    raise NotImplementedError("To render the replay, please set the render mode to json or html")


dir_path = path.dirname(__file__)
json_path = path.abspath(path.join(dir_path, "lux_ai_s2.json"))
with open(json_path) as json_file:
    specification = json.load(json_file)


def html_renderer():
    html_path = path.abspath(path.join(dir_path, "index.html"))
    return ("html_path", html_path)

agents = all_agents