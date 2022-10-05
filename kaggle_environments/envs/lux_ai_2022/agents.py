
import sys
import numpy as np
import json
def process_action(action):
    return to_json(action)
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

def from_json(state):
    if isinstance(state, list):
        return np.array(state)
    elif isinstance(state, dict):
        out = {}
        for k in state:
            out[k] = from_json(state[k])
        return out
    else:
        return state 

game_state = dict()
def process_obs(agent, step, obs):
    global game_state
    if step == 0:
        # at step 0 we get the entire map information
        game_state[agent] = from_json(obs)
    else:
        # use delta changes to board to update game state
        obs = from_json(obs)
        for k in obs:
            if k != 'board':
                game_state[agent][k] = obs[k]
        for item in ["rubble", "lichen", "lichen_strains"]:
            for k, v in obs["board"][item].items():
                k = k.split(",")
                x, y = int(k[0]), int(k[1])
                game_state[agent]["board"][item][y, x] = v

def random_agent(observation, configurations):
    global game_state
    step = observation["step"]
    agent = observation.player
    process_obs(agent, step, json.loads(observation.obs))
    
    if step == 0:
        if agent == "player_0":
            return process_action(dict(faction="MotherMars", spawns=np.array([[4, 4], [15, 5]])))
        else:
            return process_action(dict(faction="AlphaStrike", spawns=np.array([[56, 55], [40, 42]])))
    else:
        obs = game_state[agent]
        factories = obs["factories"][agent]
        actions = dict()
        if step % 4 == 0 and step > 1:
            for unit_id, factory in factories.items():
                actions[unit_id] = np.random.randint(0,2)
        else:
            for unit_id, factory in factories.items():
                actions[unit_id] = 2
        for unit_id, unit in obs["units"][agent].items():
            # actions[unit_id] = np.array([0, np.random.randint(5), 0, 0, 0])
            # make units go to 0, 0
            pos = unit['pos']
            target_pos = np.array([32 + np.random.randint(-10, 10), 32 + np.random.randint(-10, 10)])
            diff = target_pos - pos
            # print(pos, diff)
            direc = 0
            if np.random.randint(0, 2) == 0:
                if diff[0] != 0:
                    if diff[0] > 0:
                        direc = 2
                    else:
                        direc = 4
                elif diff[1] != 0:
                    if diff[1] > 0:
                        direc = 3
                    else:
                        direc = 1
            else:
                direc = np.random.randint(0,5)
            actions[unit_id] = []
            for i in range(10):
                actions[unit_id] += [np.array([0, direc, 0, 0, 0])]
        return process_action(actions)

def simple_agent(observation, configuration):
    return {}


all_agents = {
    "random": random_agent,
    "simple": simple_agent
}
