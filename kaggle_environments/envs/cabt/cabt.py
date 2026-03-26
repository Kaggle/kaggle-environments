import copy
import json
import os
import random

from .cg.game import battle_finish, battle_select, battle_start, visualize_data
from .cg.sim import Battle

deck = [
    721,
    721,
    722,
    722,
    722,
    722,
    723,
    723,
    723,
    723,
    1092,
    1121,
    1121,
    1145,
    1145,
    1163,
    1163,
    1219,
    1219,
    1219,
    1219,
    1227,
    1227,
    1227,
    1227,
    1262,
    1262,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
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


def finish(state, env):
    if len(env.steps) > 0:
        vis = json.loads(visualize_data())
        for i in range(len(vis)):
            obs = ""
            action = None
            if len(env.steps) > i:
                index = 1
                if env.steps[i][0].status == 'ACTIVE':
                    index = 0
                obs = copy.copy(env.steps[i][index].observation)
                obs.pop("search_begin_input")
                if len(env.steps) > i + 1:
                    action = [env.steps[i + 1][0].action, env.steps[i + 1][1].action]
                else:
                    action = [state[0].action, state[1].action]
            vis[i]["obs"] = obs
            vis[i]["action"] = action
        env.steps[0][0]["visualize"] = vis
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
            finish(state, env)
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
        finish(state, env)
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
