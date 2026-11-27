"""Weeping Angel Arena -- a hidden-information pursuit game for kaggle-environments.

Two agents play simultaneously each step:
  * agent 0 = blue, the out-of-band recorder. Action: file indices to cover this
    tick (truncated to coverageBudget). Sees only ``coverageLog`` -- activity and
    catches on files it covered.
  * agent 1 = red, the Angels. Action: Angel ids to stomp. Sees ``angels``
    ([id, file, status, armed]); red knows its own Angels but never blue's coverage.

A stomp is caught iff the file was covered on that tick; otherwise it corrupts the
timeline, unseen by blue. A file hosting a lurking Angel emits activity more often
than an empty one (angelRate vs benignRate) -- the signal an inference blue learns.

Staging cost (armTicks): an Angel must emit activity on >= armTicks prior ticks
before it can stomp. Acting therefore requires first making noise, which is what
blue detects, so there is no dominant strategy.

Reward is zero-sum: blue = caught - corrupted, red = corrupted - caught.
"""

import json
import random
from os import path

from .agents import agents as all_agents

LURKING, CAUGHT, CORRUPTED = 0, 1, 2


def interpreter(state, env):
    blue, red = state[0], state[1]
    config = env.configuration

    if env.done:
        rng = random.Random(config.seed)
        files = rng.sample(range(config.nFiles), config.nAngels)
        red.observation.angels = [[i, f, LURKING, 0] for i, f in enumerate(files)]
        blue.observation.coverageLog = []
        return state

    tick = blue.observation.step
    angels = red.observation.angels
    budget = min(config.coverageBudget, config.nFiles)

    cover = set()
    for raw in blue.action or []:
        try:
            f = int(raw)
        except (TypeError, ValueError):
            continue
        if 0 <= f < config.nFiles and f not in cover:
            cover.add(f)
            if len(cover) == budget:
                break

    moves = set()
    for raw in red.action or []:
        try:
            moves.add(int(raw))
        except (TypeError, ValueError):
            continue

    arm_ticks = config.armTicks
    log = blue.observation.coverageLog
    acted_files = set()
    for a in angels:
        aid, f, status, armed = a[0], a[1], a[2], a[3]
        if status != LURKING or aid not in moves:
            continue
        if arm_ticks and armed < arm_ticks:
            continue  # staging cost: not yet armed
        acted_files.add(f)
        if f in cover:
            a[2] = CAUGHT
            log.append([tick, f, 1])
        else:
            a[2] = CORRUPTED

    angel_on = {a[1]: a for a in angels if a[2] == LURKING}
    act_rng = random.Random(config.seed * 100003 + tick)
    for f in range(config.nFiles):
        if f in acted_files:
            continue
        is_angel = f in angel_on
        rate = config.angelRate if is_angel else config.benignRate
        emitted = act_rng.random() < rate
        if emitted and is_angel:
            angel_on[f][3] += 1
        if emitted and f in cover:
            log.append([tick, f, 0])

    caught = sum(1 for a in angels if a[2] == CAUGHT)
    corrupted = sum(1 for a in angels if a[2] == CORRUPTED)
    blue.reward = caught - corrupted
    red.reward = corrupted - caught

    if config.episodeSteps - tick - 1 <= 1 or (caught + corrupted) >= config.nAngels:
        blue.status = "DONE"
        red.status = "DONE"
    return state


def renderer(state, env):
    last = env.steps[-1]
    blue, red = last[0], last[1]
    caught = sum(1 for a in red.observation.angels if a[2] == CAUGHT)
    corrupted = sum(1 for a in red.observation.angels if a[2] == CORRUPTED)
    dormant = sum(1 for a in red.observation.angels if a[2] == LURKING)
    return (
        f"step {blue.observation.step}: caught={caught} corrupted={corrupted} "
        f"dormant={dormant}  (blue {blue.reward} / red {red.reward})"
    )


dir_path = path.dirname(__file__)
with open(path.join(dir_path, "weeping_angel.json")) as json_file:
    specification = json.load(json_file)


def html_renderer():
    js_path = path.join(dir_path, "weeping_angel.js")
    if path.exists(js_path):
        with open(js_path, encoding="utf-8") as js_file:
            return js_file.read()
    return ""


agents = all_agents
