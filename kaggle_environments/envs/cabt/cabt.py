import random
import os
import json

from .cg.sim import Battle
from .cg.game import battle_start, battle_finish, battle_select, visualize_data


def random_agent(obs: dict) -> list[int]:
    return random.sample(list(range(len(obs["select"]["option"]))), obs["select"]["maxCount"])

def first_agent(obs: dict) -> list[int]:
    return list(range(obs["select"]["maxCount"]))

agents = {"random": random_agent, "first": first_agent}


def finish(env):
    if len(env.steps) > 0:
        env.steps[0][0]["visualize"] = json.loads(visualize_data())
    battle_finish()

def interpreter(state, env):
    if env.done:
        decks = env.configuration.decks
        battle_start(decks[0], decks[1])
    else:
        error = False
        select_player = Battle.obs["current"]["yourIndex"]
        if state[select_player].status == "TIMEOUT":
            error = True
        else:
            try:
                battle_select(state[select_player].action)
            except:
                state[select_player].status = "INVALID"
                error = True

        if error:
            state[select_player].reward = -1
            state[1 - select_player].status = "DONE"
            state[1 - select_player].reward = 1
            finish(env)
            return state

    obs = Battle.obs
    s = obs["current"]
    if s["result"] >= 0:
        state[0].status = "DONE"
        state[1].status = "DONE"
        if s["result"] == 0:
            state[0].reward = 1
            state[1].reward = -1
        elif s["result"] == 1:
            state[0].reward = -1
            state[1].reward = 1
        else:
            state[0].reward = 0
            state[1].reward = 0
        finish(env)
    else:
        index = s["yourIndex"]
        state[index].status = "ACTIVE"
        state[1 - index].status = "INACTIVE"
        o = state[index].observation
        o["select"] = obs["select"]
        o["logs"] = obs["logs"]
        o["current"] = obs["current"]
        o["search_begin_input"] = obs["search_begin_input"]
    return state

def renderer(state, env):
    return json.dumps(Battle.obs)

def html_renderer():
    jspath = os.path.abspath(os.path.join(os.path.dirname(__file__), "cabt.js"))
    with open(jspath, encoding="utf-8") as f:
        return f.read()


jsonpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "cabt.json"))
with open(jsonpath) as f:
    specification = json.load(f)
