import json
from os import path
from random import Random

DIRECTIONS = {
    "STAY": (0, 0),
    "NORTH": (-1, 0),
    "SOUTH": (1, 0),
    "EAST": (0, 1),
    "WEST": (0, -1),
}


def _is_own_territory(row, mark, rows):
    if mark == 1:
        return row < rows // 2
    return row >= (rows + 1) // 2


def _initialize(obs, rows, cols):
    mid = cols // 2
    obs.p1_base = [0, mid]
    obs.p2_base = [rows - 1, mid]
    obs.p1_pos = [0, mid]
    obs.p2_pos = [rows - 1, mid]
    obs.p1_flag_pos = [0, mid]
    obs.p2_flag_pos = [rows - 1, mid]
    obs.p1_has_flag = False
    obs.p2_has_flag = False


def interpreter(state, env):
    rows = env.configuration.rows
    cols = env.configuration.cols
    obs = state[0].observation

    if list(obs.p1_base) == list(obs.p2_base):
        _initialize(obs, rows, cols)

    if env.done:
        return state

    # The "mover" is the agent whose action just resolved. It may be ACTIVE
    # (valid action) or INVALID/ERROR/TIMEOUT (framework flagged before us).
    mover_i = 0 if state[0].status != "INACTIVE" else 1
    other_i = 1 - mover_i
    active = state[mover_i]
    inactive = state[other_i]

    if active.status != "ACTIVE":
        # Mover failed; opponent wins. Framework will null the mover's reward.
        inactive.reward = 1
        inactive.status = "DONE"
        return state

    action = active.action
    dr, dc = DIRECTIONS[action]

    mark = 1 if mover_i == 0 else 2
    my_pos = list(obs.p1_pos) if mark == 1 else list(obs.p2_pos)
    opp_pos = list(obs.p2_pos) if mark == 1 else list(obs.p1_pos)
    my_base = list(obs.p1_base) if mark == 1 else list(obs.p2_base)

    new_r, new_c = my_pos[0] + dr, my_pos[1] + dc
    off_board = new_r < 0 or new_r >= rows or new_c < 0 or new_c >= cols
    if off_board:
        new_r, new_c = my_pos[0], my_pos[1]

    tagged = False
    if [new_r, new_c] == opp_pos and (dr, dc) != (0, 0):
        if _is_own_territory(new_r, mark, rows):
            tagged = True
        else:
            new_r, new_c = my_pos[0], my_pos[1]

    if mark == 1:
        obs.p1_pos = [new_r, new_c]
    else:
        obs.p2_pos = [new_r, new_c]

    if tagged:
        if mark == 1:
            obs.p2_pos = list(obs.p2_base)
            if obs.p2_has_flag:
                obs.p2_has_flag = False
                obs.p1_flag_pos = list(obs.p1_base)
        else:
            obs.p1_pos = list(obs.p1_base)
            if obs.p1_has_flag:
                obs.p1_has_flag = False
                obs.p2_flag_pos = list(obs.p2_base)

    if mark == 1:
        if not obs.p1_has_flag and list(obs.p1_pos) == list(obs.p2_flag_pos):
            obs.p1_has_flag = True
        if obs.p1_has_flag:
            obs.p2_flag_pos = list(obs.p1_pos)
    else:
        if not obs.p2_has_flag and list(obs.p2_pos) == list(obs.p1_flag_pos):
            obs.p2_has_flag = True
        if obs.p2_has_flag:
            obs.p1_flag_pos = list(obs.p2_pos)

    my_flag_home = (
        list(obs.p1_flag_pos) == list(obs.p1_base) if mark == 1 else list(obs.p2_flag_pos) == list(obs.p2_base)
    )
    carrying = obs.p1_has_flag if mark == 1 else obs.p2_has_flag
    at_home = [new_r, new_c] == my_base
    if carrying and at_home and my_flag_home:
        active.reward = 1
        inactive.reward = -1
        active.status = "DONE"
        inactive.status = "DONE"
        return state

    active.status = "INACTIVE"
    inactive.status = "ACTIVE"
    return state


def _cell_str(r, c, obs):
    p1 = list(obs.p1_pos) == [r, c]
    p2 = list(obs.p2_pos) == [r, c]
    if p1:
        return "1F" if obs.p1_has_flag else "1 "
    if p2:
        return "2F" if obs.p2_has_flag else "2 "
    if list(obs.p1_flag_pos) == [r, c]:
        return "f1"
    if list(obs.p2_flag_pos) == [r, c]:
        return "f2"
    if list(obs.p1_base) == [r, c]:
        return "b1"
    if list(obs.p2_base) == [r, c]:
        return "b2"
    return ". "


def renderer(state, env):
    rows = env.configuration.rows
    cols = env.configuration.cols
    obs = state[0].observation

    if list(obs.p1_base) == list(obs.p2_base):
        _initialize(obs, rows, cols)

    header = (
        f"Step {obs.step}  P1 {list(obs.p1_pos)}{' (has flag)' if obs.p1_has_flag else ''}  "
        f"P2 {list(obs.p2_pos)}{' (has flag)' if obs.p2_has_flag else ''}\n"
    )
    row_bar = "+" + "+".join(["--"] * cols) + "+\n"
    out = header + row_bar
    for r in range(rows):
        line = "|" + "|".join(_cell_str(r, c, obs) for c in range(cols)) + "|\n"
        out += line + row_bar
    return out


def _dir_toward(src, dst):
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    candidates = []
    if dr < 0:
        candidates.append("NORTH")
    elif dr > 0:
        candidates.append("SOUTH")
    if dc > 0:
        candidates.append("EAST")
    elif dc < 0:
        candidates.append("WEST")
    return candidates


def random_agent(obs, config):
    rng = Random()
    return rng.choice(list(DIRECTIONS.keys()))


def greedy_agent(obs, config):
    mark = obs.mark
    my_pos = list(obs.p1_pos) if mark == 1 else list(obs.p2_pos)
    my_base = list(obs.p1_base) if mark == 1 else list(obs.p2_base)
    opp_flag_pos = list(obs.p2_flag_pos) if mark == 1 else list(obs.p1_flag_pos)
    my_has_flag = obs.p1_has_flag if mark == 1 else obs.p2_has_flag
    target = my_base if my_has_flag else opp_flag_pos
    if my_pos == target:
        return "STAY"
    options = _dir_toward(my_pos, target)
    if not options:
        return "STAY"
    return options[0]


def defender_agent(obs, config):
    mark = obs.mark
    my_pos = list(obs.p1_pos) if mark == 1 else list(obs.p2_pos)
    opp_pos = list(obs.p2_pos) if mark == 1 else list(obs.p1_pos)
    my_flag_pos = list(obs.p1_flag_pos) if mark == 1 else list(obs.p2_flag_pos)
    my_base = list(obs.p1_base) if mark == 1 else list(obs.p2_base)
    opp_has_my_flag = obs.p2_has_flag if mark == 1 else obs.p1_has_flag

    if opp_has_my_flag:
        target = opp_pos
    else:
        target = my_flag_pos if my_flag_pos != my_base else [my_base[0], my_base[1]]

    if my_pos == target:
        return "STAY"
    options = _dir_toward(my_pos, target)
    if not options:
        return "STAY"
    return options[0]


agents = {"random": random_agent, "greedy": greedy_agent, "defender": defender_agent}


dirpath = path.dirname(__file__)
jsonpath = path.abspath(path.join(dirpath, "capture_the_flag.json"))
with open(jsonpath) as f:
    specification = json.load(f)


def html_renderer():
    htmlpath = path.join(dirpath, "visualizer", "default", "dist", "index.html")
    if path.exists(htmlpath):
        with open(htmlpath, encoding="utf-8") as f:
            return f.read()
    return ""
