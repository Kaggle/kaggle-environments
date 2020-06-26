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
import os
import requests
import sys
from io import StringIO
from time import time
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


def build_agent(raw, environment):
    if raw in environment.agents:
        return environment.agents[raw]

    # Already callable.
    if callable(raw):
        return raw

    # Not a string, static action.
    if not isinstance(raw, str):
        return lambda: raw

    # A URL and will be initialized on the calling server.
    if is_url(raw):
        def url_agent(o, c, r, i):
            data = {
                "action": "act",
                "configuration": {
                    **c,
                    "agentExec": "LOCAL"
                },
                "environment": environment.name,
                "state": {
                    "observation": o,
                    "reward": r,
                    "info": i
                }
            }
            response = requests.post(url=raw, data=json.dumps(data))
            response_json = response.json()
            action = response_json["action"]
            if action == "DeadlineExceeded":
                action = DeadlineExceeded()
            elif isinstance(action, str) and action.startswith("BaseException::"):
                # Deserialize the exception message
                parts = action.split("::", 1)
                action = BaseException(parts[1])
            return action
        return url_agent

    # A path exists and attempt to grab the source (fallback to the original string).
    if os.path.exists(raw):
        raw = read_file(raw, raw)

    # Attempt to execute the last callable or just return the string.
    return get_last_callable(raw) or (lambda: raw)


class Agent:
    def __init__(self, raw, configuration, environment):
        self.configuration = configuration
        self.environment = environment
        self.raw = raw
        self.agent = None

    def act(self, state, timeout=10):
        # Start the timer.
        start = time()

        if self.agent is None:
            self.agent = build_agent(self.raw, self.environment)
            # Add in the initialization timeout since this is the first time this agent is called
            timeout += self.configuration.agentTimeout

        if state is not None:
            args = [
               structify(state["observation"]),
               structify(self.configuration),
               state["reward"],
               structify(state["info"])
            ][:self.agent.__code__.co_argcount]

            try:
                action = self.agent(*args)
            except Exception as e:
                action = e

        # Timeout reached, throw an error.
        if time() - start > timeout:
            return DeadlineExceeded()

        return action
