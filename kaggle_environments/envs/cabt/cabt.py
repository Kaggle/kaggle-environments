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


def round_finish(state, env):
    steps = env.steps[Battle.last_step:]
    if len(steps) > 0:
        vis = json.loads(visualize_data())
        for i in range(len(vis)):
            obs = ""
            action = None
            if len(steps) > i:
                index = 1
                if steps[i][0].status == 'ACTIVE':
                    index = 0
                obs = copy.copy(steps[i][index].observation)
                obs.pop("search_begin_input")
                if len(steps) > i + 1:
                    action = [steps[i + 1][0].action, steps[i + 1][1].action]
                else:
                    action = [state[0].action, state[1].action]
            vis[i]["obs"] = obs
            vis[i]["action"] = action
            for j in range(2):
                vis[i]["current"]["players"][j]["remainingTime"] = steps[i][j]["observation"]["remainingOverageTime"]
        Battle.vis += vis
    battle_finish()


def finish(state, env):
    round_finish(state, env)
    env.steps[0][0]["visualize"] = Battle.vis


def interpreter(state, env):
    if env.done:
        Battle.battle_ptr = None
        Battle.decks = None
        Battle.result = [0, 0, 0]
        Battle.vis = []
        Battle.last_step = 0
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
        Battle.decks = decks
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
        if s["result"] == 0:
            Battle.result[0] += 1
        elif s["result"] == 1:
            Battle.result[1] += 1
        else:
            Battle.result[2] += 1

        count = Battle.result[0] + Battle.result[1] + Battle.result[2]
        remain = env.configuration.bo - count
        result = -1
        if Battle.result[0] > Battle.result[1] + remain:
            result = 0
        elif Battle.result[1] > Battle.result[0] + remain:
            result = 1
        elif remain <= 0:
            result = 2

        env.result = Battle.result
        if result >= 0:
            state[0].status = "DONE"
            state[1].status = "DONE"
            if result == 0:
                state[0].reward = 1
                state[1].reward = -1
            elif result == 1:
                state[0].reward = -1
                state[1].reward = 1
            else:
                state[0].reward = 0
                state[1].reward = 0
            finish(state, env)
            return state
        else:
            round_finish(state, env)
            Battle.last_step = len(env.steps) - 1
            battle_start(Battle.decks[0], Battle.decks[1], count % 2 != 0)
            obs = Battle.obs
            s = obs["current"]
    index = s["yourIndex"]
    state[index].status = "ACTIVE"
    state[1 - index].status = "INACTIVE"
    o = state[index].observation
    o["select"] = obs["select"]
    o["logs"] = obs["logs"]
    o["current"] = obs["current"]
    o["search_begin_input"] = obs["search_begin_input"]
    s["players"][0]["win"] = Battle.result[0]
    s["players"][1]["win"] = Battle.result[1]
    s["draw"] = Battle.result[2]
    s["round"] = Battle.result[0] + Battle.result[1] + Battle.result[2] + 1
    return state


def renderer(state, env):
    return json.dumps(Battle.obs)


def html_renderer():
    dir_path = os.path.dirname(__file__)
    htmlpath = os.path.join(dir_path, "visualizer", "default", "dist", "index.html")
    if os.path.exists(htmlpath):
        with open(htmlpath, encoding="utf-8") as f:
            return f.read()
    jspath = os.path.abspath(os.path.join(dir_path, "cabt.js"))
    if os.path.exists(jspath):
        with open(jspath, encoding="utf-8") as f:
            return f.read()
    return ""


jsonpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "cabt.json"))
with open(jsonpath) as f:
    specification = json.load(f)
