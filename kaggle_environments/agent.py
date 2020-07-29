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

import inspect
import json
import os
import requests
import sys
from io import StringIO
from time import perf_counter
from urllib.parse import urlparse
from .errors import DeadlineExceeded, InvalidArgument
from .utils import read_file, structify


def is_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def get_last_callable(raw, fallback=None):
    orig_out = sys.stdout
    buffer = StringIO()
    sys.stdout = buffer

    try:
        code_object = compile(raw, "<string>", "exec")
        env = {}
        exec(code_object, env)
        sys.stdout = orig_out
        output = buffer.getvalue()
        if output:
            print(output)
        return [v for v in env.values() if callable(v)][-1]
    except Exception as e:
        sys.stdout = orig_out
        output = buffer.getvalue()
        if output:
            print(output)
        if fallback is not None:
            return fallback
        raise InvalidArgument("Invalid raw Python: " + repr(e))


class UrlAgent:
    def __init__(self, raw, environment_name):
        self.raw = raw
        self.environment_name = environment_name

    def __call__(self, observation, configuration):
        data = {
            "action": "act",
            "configuration": configuration,
            "environment": self.environment_name,
            "state": {
                "observation": observation,
            },
        }
        response = requests.post(url=self.raw, data=json.dumps(data))
        response_json = response.json()
        action = response_json["action"]
        if action == "DeadlineExceeded":
            action = DeadlineExceeded()
        elif isinstance(action, str) and action.startswith("BaseException::"):
            # Deserialize the exception message
            parts = action.split("::", 1)
            action = BaseException(parts[1])
        return action


def build_agent(raw, builtin_agents, environment_name):
    """
    Returns the agent and whether the agent is parallelizable.
    """
    if raw in builtin_agents:
        return builtin_agents[raw], False

    # Already callable.
    if callable(raw):
        return raw, False

    # Not a string, static action.
    if not isinstance(raw, str):
        return lambda: raw, False

    # A URL and will be initialized on the calling server.
    if is_url(raw):
        return UrlAgent(raw, environment_name), True

    # A path exists and attempt to grab the source (fallback to the original string).
    if os.path.exists(raw):
        raw = read_file(raw, raw)

    # Attempt to execute the last callable or just return the string.
    agent = None

    def callable_agent(observation, configuration):
        nonlocal agent
        if agent is None:
            agent = get_last_callable(raw) or raw
        return \
            agent(observation, configuration) \
            if callable(agent) \
            else agent

    return callable_agent, False


class Agent:
    def __init__(self, raw, environment):
        self.builtin_agents = environment.agents
        self.configuration = environment.configuration
        self.environment_name = environment.name
        self.raw = raw
        self.agent, self.is_parallelizable = build_agent(self.raw, self.builtin_agents, self.environment_name)
        self.is_initialized = False

    def act(self, observation):
        timeout = self.configuration.actTimeout
        if not self.is_initialized:
            # Add in the initialization timeout since this is the first time this agent is called
            timeout += self.configuration.agentTimeout
            self.is_initialized = True

        args = [
            structify(observation),
            structify(self.configuration)
        ]

        if hasattr(self.agent, "__code__"):
            args = args[:self.agent.__code__.co_argcount]

        # Start the timer.
        start = perf_counter()
        try:
            action = self.agent(*args)
        except Exception as e:
            action = e

        # Timeout reached, throw an error.
        if perf_counter() - start > timeout:
            return DeadlineExceeded()

        return action
