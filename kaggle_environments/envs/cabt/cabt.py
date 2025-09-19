import json
import os
import random

from .cg.game import battle_finish, battle_select, battle_start, visualize_data
from .cg.sim import Battle

deck = [
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    9,
    9,
    77,
    77,
    77,
    77,
    156,
    156,
    156,
    156,
    157,
    157,
    157,
    157,
    331,
    331,
    331,
    331,
    408,
    408,
    408,
    408,
    474,
    474,
    474,
    474,
    528,
    528,
    528,
    528,
    530,
    530,
    530,
    530,
    532,
    554,
    554,
    554,
    576,
    576,
    576,
    576,
    585,
    585,
    585,
    585,
    630,
    630,
    630,
    630,
]


def random_agent(obs: dict) -> list[int]:
    if obs["select"] == None:
        return deck
    return random.sample(list(range(len(obs["select"]["option"]))), obs["select"]["maxCount"])


def first_agent(obs: dict) -> list[int]:
    if obs["select"] == None:
        return deck
    return list(range(obs["select"]["maxCount"]))


agents = {"random": random_agent, "first": first_agent}


def finish(env):
    if len(env.steps) > 0:
        env.steps[0][0]["visualize"] = json.loads(visualize_data())
    battle_finish()


def interpreter(state, env):
    if env.done:
        Battle.battle_ptr = None
        for i in range(2):
            state[i].status = "ACTIVE"
            o = state[i].observation
            o["select"] = None
            o["logs"] = []
            o["current"] = None
            o["search_begin_input"] = None
        return state
    elif Battle.battle_ptr == None:
        decks = [state[0].action, state[1].action]
        error = False
        for i in range(2):
            if state[i].status == "TIMEOUT" or state[i].status == "ERROR":
                error = True
                continue
            if len(decks[i]) != 60:
                state[i].status = "INVALID"
                env.steps[0][0]["error"] = f"Player {i}'s deck does not have 60 cards."
                error = True
        if not error:
            _, start_data = battle_start(state[0].action, state[1].action)
            if start_data.errorPlayer >= 0:
                state[start_data.errorPlayer].status = "INVALID"
                env.steps[0][0]["error"] = f"Player {i}'s deck error."
                error = True
        if error:
            for i in range(2):
                if state[i].status == "ACTIVE":
                    state[i].status = "DONE"
            return state
        if Battle.battle_ptr == None:
            raise ValueError("battle_ptr None.")
    else:
        error = False
        select_player = Battle.obs["current"]["yourIndex"]
        if state[select_player].status == "TIMEOUT" or state[select_player].status == "ERROR":
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
