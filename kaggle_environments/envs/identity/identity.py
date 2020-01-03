# Copyright 2020 Kaggle Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from os import path
from random import choice, gauss


def random_agent(obs, config):
    return choice(range(config.min, config.max))


def max_agent(obs, config):
    return config.max


def min_agent(obs, config):
    return config.min


def avg_agent(obs, config):
    return (config.min + config.max) // 2


agents = {
    "random": random_agent,
    "max": max_agent,
    "min": min_agent,
    "avg": avg_agent,
}


def interpreter(state, env):
    if env.done:
        return state

    # Validate and assign actions as rewards !(min <= action <= max).
    for agent in state:
        if agent.action < env.configuration.min or agent.action > env.configuration.max:
            agent.status = f"Invalid action: {agent.action}"
        else:
            agent.reward = agent.action + \
                gauss(0, 1) * env.configuration.noise // 1
            agent.status = "DONE"

    return state


def renderer(state, env):
    return json.dumps([{"action": a.action, "reward": a.reward} for a in state])


dirpath = path.dirname(__file__)
jsonpath = path.abspath(path.join(dirpath, "identity.json"))
with open(jsonpath) as f:
    specification = json.load(f)
