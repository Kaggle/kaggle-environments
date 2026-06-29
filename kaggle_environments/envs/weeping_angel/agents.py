"""Built-in baseline agents for the Weeping Angel Arena.

Blue agents return file indices to cover; red agents return Angel ids to stomp.
All are stateless (recomputed from the observation each step) and deterministic.
"""

import random

LURKING = 0


def _angel_ids(obs):
    return [a[0] for a in obs["angels"] if a[2] == LURKING]


# -- blue (the recorder) -------------------------------------------------------
def random_blue(obs, config):
    rng = random.Random(obs["step"] * 7919 + 1)
    k = min(config["coverageBudget"], config["nFiles"])
    return rng.sample(range(config["nFiles"]), k)


def sweep_blue(obs, config):
    k = min(config["coverageBudget"], config["nFiles"])
    start = (obs["step"] * k) % config["nFiles"]
    return [(start + i) % config["nFiles"] for i in range(k)]


def inference_blue(obs, config):
    """Cover the files with the most observed activity (Angel files are noisier),
    rotating exploration in on ties."""
    n = config["nFiles"]
    activity = [0] * n
    caught = set()
    for _tick, f, kind in obs["coverageLog"]:
        if kind == 1:
            caught.add(f)
        else:
            activity[f] += 1
    rot = obs["step"]
    candidates = [f for f in range(n) if f not in caught]
    candidates.sort(key=lambda f: (-activity[f], (f + rot) % n))
    return candidates[: min(config["coverageBudget"], len(candidates))]


def bayes_blue(obs, config):
    """Thompson sampling over each file's activity-based posterior."""
    n = config["nFiles"]
    activity = [0] * n
    caught = set()
    for _tick, f, kind in obs["coverageLog"]:
        if kind == 1:
            caught.add(f)
        else:
            activity[f] += 1
    rng = random.Random(config["seed"] * 131 + obs["step"])
    candidates = [f for f in range(n) if f not in caught]
    candidates.sort(key=lambda f: rng.betavariate(activity[f] + 1, 1), reverse=True)
    return candidates[: min(config["coverageBudget"], len(candidates))]


# -- red (the Angels) ----------------------------------------------------------
def rush_red(obs, config):
    return _angel_ids(obs) if obs["step"] <= 1 else []


def evasive_red(obs, config):
    strike = 1 + (config["seed"] % 3)
    return _angel_ids(obs) if obs["step"] == strike else []


def random_red(obs, config):
    n_ticks = config["episodeSteps"] - 1
    seed = config["seed"]
    return [
        a[0]
        for a in obs["angels"]
        if a[2] == LURKING and (seed * 31 + a[0] * 2654435761) % n_ticks == obs["step"]
    ]


def mixed_red(obs, config):
    """Randomized strike timing, then keep attempting -- waits out the staging
    cost without knowing when each Angel is armed."""
    n_ticks = config["episodeSteps"] - 1
    seed = config["seed"]
    out = []
    for a in obs["angels"]:
        if a[2] != LURKING:
            continue
        strike = (seed * 131 + a[0] * 2654435761) % n_ticks
        if obs["step"] >= strike:
            out.append(a[0])
    return out


agents = {
    "random_blue": random_blue,
    "sweep_blue": sweep_blue,
    "inference_blue": inference_blue,
    "bayes_blue": bayes_blue,
    "rush_red": rush_red,
    "evasive_red": evasive_red,
    "random_red": random_red,
    "mixed_red": mixed_red,
}
