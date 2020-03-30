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

import os
import requests
import json
from multiprocessing import Manager, Process
from time import time
import uuid
from .errors import DeadlineExceeded
from .utils import get_exec, has, is_url, read_file, structify


def build_agent(raw):
    # Already callable.
    if callable(raw):
        return raw

    # Not a string, static action.
    if not has(raw, str):
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
                "state": {
                    "observation": o,
                    "reward": r,
                    "info": i
                }
            }
            action = requests.post(
                url=raw, data=json.dumps(data)).json()["action"]
            if action == "DeadlineExceeded":
                action = DeadlineExceeded()
            elif action == "BaseException":
                action = BaseException()
            return action
        return url_agent

    # A path exists and attempt to grab the source (fallback to the original string).
    if os.path.exists(raw):
        raw = read_file(raw, raw)

    # Attempt to execute the last callable or just return the string.
    try:
        local = get_exec(raw)
        callables = [v for v in local.values() if callable(v)]
        if len(callables) > 0:
            return callables[-1]
        raise "Nope"
    except:
        return lambda: raw


def run_agent(agent, message):
    if message.state != None and message:
        args = [
            structify(message.state["observation"]),
            structify(message.configuration),
            message.state["reward"],
            structify(message.state["info"])
        ][:agent.__code__.co_argcount]
        try:
            message.action = agent(*args)
        except Exception as e:
            message.action = e
        message.state = None


def runner(raw, message):
    try:
        agent = build_agent(raw)
    except Exception as e:
        message.action = e
    while True:
        run_agent(agent, message)


class Agent():

    def __init__(self, raw, configuration, id=None):
        self.id = id or str(uuid.uuid1())
        self.configuration = configuration
        self.raw = raw
        self.use_process = configuration["agentExec"] == "PROCESS"

        if self.use_process:
            self.manager = Manager()
            self.message = self.manager.Namespace()
            self.message.action = None
            self.message.state = None
            self.message.configuration = configuration
            self.process = Process(target=runner, args=(raw, self.message))
            self.process.daemon = True
            self.process.start()
        else:
            self.message = structify(
                {"action": None, "state": None, "configuration": configuration})
            self.agent = None

    def act(self, state, timeout=10):
        # Start the timer.
        start = time()

        # If an action is already set (uncleared), there is an error.
        if self.message.action != None:
            return self.message.action

        # Inform the agent an action is requested.
        self.message.state = state

        if self.use_process:
            # Timeout or Action Returned (will be processed below).
            while True:
                if time() - start > timeout or self.message.action != None:
                    break
        else:
            if self.agent is None:
                try:
                    self.agent = build_agent(self.raw)
                    # Update the timeout to add the agentTimeout (incase set to "act").
                    timeout = self.configuration.agentTimeout + self.configuration.actTimeout
                except Exception as e:
                    return e
            run_agent(self.agent, self.message)

        # Timeout reached, destroy the agent, and throw an error.
        if time() - start > timeout:
            self.destroy()
            return DeadlineExceeded()

        # Return and clear the action.
        action = self.message.action
        self.message.action = None
        return action

    def destroy(self):
        if self.id == None:
            return
        self.id = None
        if self.use_process:
            self.process.join(0.1)
            self.process.terminate()
            self.manager.shutdown()
