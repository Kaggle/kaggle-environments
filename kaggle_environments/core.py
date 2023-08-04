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

import traceback
import copy
import json
import uuid
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from multiprocessing import Pool
from time import perf_counter
from .agent import Agent
from .errors import DeadlineExceeded, FailedPrecondition, Internal, InvalidArgument
from .utils import get, has, get_player, process_schema, schemas, structify

# Registered Environments.
environments = {}

# Registered Interactive Sessions.
interactives = {}


def register(name, environment):
    """
    Register an environment by name.  An environment contains the following:
     * specification - JSON Schema representing the environment.
     * interpreter - Function(state, environment) -> new_state
     * renderer - Function(state, environment) -> string
     * html_renderer - Function(environment) -> JavaScript HTML renderer function.
     * agents(optional) - List of default agents [Function(observation, config) -> action]
    """
    environments[name] = environment


def evaluate(environment, agents=None, configuration=None, steps=None, num_episodes=1, debug=False, state=None):
    """
    Evaluate and return the rewards of one or more episodes (environment and agents combo).

    Args:
        environment (str|Environment):
        agents (list):
        configuration (dict, optional):
        steps (list, optional):
        num_episodes (int=1, optional): How many episodes to execute (run until done).
        debug (bool=False, optional): Render print() statments to stdout
        state (optional)

    Returns:
        list of list of int: List of final rewards for all agents for all episodes.
    """
    if agents is None:
        agents = []
    if configuration is None:
        configuration = {}
    if steps is None:
        steps = []

    e = make(environment, configuration, steps, debug=debug, state=state)
    rewards = [[] for i in range(num_episodes)]
    for i in range(num_episodes):
        last_state = e.run(agents)[-1]
        rewards[i] = [state.reward for state in last_state]
    return rewards


def make(environment, configuration=None, info=None, steps=None, logs=None, debug=False, state=None):
    """
    Creates an instance of an Environment.

    Args:
        environment (str|Environment):
        configuration (dict, optional):
        info (dict, optional):
        steps (list, optional):
        debug (bool=False, optional): Render print() statments to stdout
        state (optional):

    Returns:
        Environment: Instance of a specific environment.
    """
    if configuration is None:
        configuration = {}
    if info is None:
        info = {}
    if steps is None:
        steps = []
    if logs is None:
        logs = []


    if has(environment, str) and has(environments, dict, path=[environment]):
        return Environment(**environments[environment], configuration=configuration, info=info, steps=steps, logs=logs, debug=debug, state=state)
    elif callable(environment):
        return Environment(interpreter=environment, configuration=configuration, info=info, steps=steps, logs=logs, debug=debug, state=state)
    elif has(environment, path=["interpreter"], is_callable=True):
        return Environment(**environment, configuration=configuration, info=info, steps=steps, logs=logs, debug=debug, state=state)
    raise InvalidArgument("Unknown Environment Specification")


def act_agent(args):
    agent, state, configuration, none_action = args
    if state["status"] != "ACTIVE":
        return None, {}
    elif agent is None:
        return none_action, {}
    else:
        return agent.act(state["observation"])


class Environment:
    def __init__(
        self,
        specification=None,
        configuration=None,
        info=None,
        steps=None,
        logs=None,
        agents=None,
        interpreter=None,
        renderer=None,
        html_renderer=None,
        debug=False,
        state=None,
    ):
        if specification is None:
            specification = {}
        if configuration is None:
            configuration = {}
        if info is None:
            info = {}
        if steps is None:
            steps = []
        if logs is None:
            logs = []
        if agents is None:
            agents = {}

        self.logs = logs
        self.id = str(uuid.uuid1())
        self.debug = debug
        self.info = info
        self.pool = None

        err, specification = self.__process_specification(specification)
        if err:
            raise InvalidArgument("Specification Invalid: " + err)
        self.specification = structify(specification)

        err, configuration = process_schema(
            {"type": "object", "properties": self.specification.configuration},
            {} if configuration is None else configuration,
        )
        if err:
            raise InvalidArgument("Configuration Invalid: " + err)
        self.configuration = structify(configuration)

        if not callable(interpreter):
            raise InvalidArgument("Interpreter is not Callable.")
        self.interpreter = interpreter

        if not callable(renderer):
            raise InvalidArgument("Renderer is not Callable.")
        self.renderer = renderer

        if not callable(html_renderer):
            raise InvalidArgument("Html_renderer is not Callable.")
        self.html_renderer = html_renderer

        if not all([callable(a) for a in agents.values()]):
            raise InvalidArgument("Default agents must be Callable.")
        self.agents = structify(agents)

        if steps is not None and len(steps) > 0:
            self.__set_state(steps[-1])
            self.steps = steps[0:-1] + self.steps
        elif state is not None:
            step = [{}] * self.specification.agents[0]
            step[0] = state
            self.__set_state(step)
        else:
            self.reset()

    def step(self, actions, logs=None):
        """
        Execute the environment interpreter using the current state and a list of actions.

        Args:
            actions (list): Actions to pair up with the current agent states.
            logs (list): Logs to pair up with each agent for the current step.

        Returns:
            list of dict: The agents states after the step.
        """
        if logs is None:
            logs = []

        if self.done:
            raise FailedPrecondition("Environment done, reset required.")
        if not actions or len(actions) != len(self.state):
            raise InvalidArgument(f"{len(self.state)} actions required.")

        action_state = [0] * len(self.state)
        for index, action in enumerate(actions):
            action_state[index] = {**self.state[index], "action": None}

            if isinstance(action, DeadlineExceeded):
                self.debug_print(f"Timeout: {str(action)}")
                action_state[index]["status"] = "TIMEOUT"
            elif isinstance(action, BaseException):
                self.debug_print(f"Error: {traceback.format_exception(None, action, action.__traceback__)}")
                action_state[index]["status"] = "ERROR"
            else:
                err, data = process_schema(
                    self.__state_schema.properties.action, action)
                if err:
                    self.debug_print(f"Invalid Action: {str(err)}")
                    action_state[index]["status"] = "INVALID"
                else:
                    action_state[index]["action"] = data

        self.state = self.__run_interpreter(action_state, logs)

        # Max Steps reached. Mark ACTIVE/INACTIVE agents as DONE.
        if self.state[0].observation.step >= self.configuration.episodeSteps - 1:
            for s in self.state:
                if s.status == "ACTIVE" or s.status == "INACTIVE":
                    s.status = "DONE"

        self.steps.append(self.state)
        if logs is not None:
            self.logs.append(logs)

        return self.state

    def run(self, agents):
        """
        Steps until the environment is "done" or the runTimeout was reached.

        Args:
            agents (list of any): List of agents to obtain actions from.

        Returns:
            tuple of:
                list of list of dict: The agent states of all steps executed.
                list of list of dict: The agent logs of all steps executed.
        """
        if self.state is None or len(self.steps) == 1 or self.done:
            self.reset(len(agents))
        if len(self.state) != len(agents):
            raise InvalidArgument(
                f"{len(self.state)} agents were expected, but {len(agents)} was given.")

        runner = self.__agent_runner(agents)
        start = perf_counter()
        while not self.done and perf_counter() - start < self.configuration.runTimeout:
            actions, logs = runner.act()
            self.step(actions, logs)
        return self.steps

    def reset(self, num_agents=None):
        """
        Resets the environment state to the initial step.

        Args:
            num_agents (int): Resets the state assuming a fixed number of agents.

        Returns:
            list of dict: The agents states after the reset.
        """

        if num_agents is None:
            num_agents = self.specification.agents[0]

        # Get configuration default state.
        self.__set_state([{} for _ in range(num_agents)])
        # Reset all agents to status=INACTIVE (copy out values to reset afterwards).
        statuses = [a.status for a in self.state]
        for agent in self.state:
            agent.status = "INACTIVE"
        # Give the interpreter an opportunity to make any initializations.
        logs = []
        self.__set_state(self.__run_interpreter(self.state, logs))
        self.logs.append(logs)
        # Replace the starting "status" if still "done".
        if self.done and len(self.state) == len(statuses):
            for i in range(len(self.state)):
                self.state[i].status = statuses[i]
        return self.state

    def render(self, **kwargs):
        """
        Renders a visual representation of the current state of the environment.

        Args:
            mode (str): html, ipython, ansi, human (default)
            **kwargs (dict): Other args are directly passed into the html player.

        Returns:
            str: html if mode=html or ansi if mode=ansi.
            None: prints ansi if mode=human or prints html if mode=ipython
        """
        mode = get(kwargs, str, "human", path=["mode"])
        if mode == "ansi" or mode == "human":
            args = [self.state, self]
            out = self.renderer(*args[:self.renderer.__code__.co_argcount])
            if mode == "ansi":
                return out
        elif mode == "html" or mode == "ipython":
            is_playing = get(kwargs, bool, self.done, path=["playing"])
            window_kaggle = {
                "debug": get(kwargs, bool, self.debug, path=["debug"]),
                "playing": is_playing,
                "step": 0 if is_playing else len(self.steps) - 1,
                "controls": get(kwargs, bool, self.done, path=["controls"]),
                "environment": self.toJSON(),
                "logs": self.logs,
                **kwargs,
            }
            args = [self]
            player_html = get_player(window_kaggle,
                                     self.html_renderer(*args[:self.html_renderer.__code__.co_argcount]))
            if mode == "html":
                return player_html

            from IPython.display import display, HTML
            player_html = player_html.replace('"', '&quot;')
            width = get(kwargs, int, 300, path=["width"])
            height = get(kwargs, int, 300, path=["height"])
            html = f'<iframe srcdoc="{player_html}" width="{width}" height="{height}" frameborder="0"></iframe> '
            display(HTML(html))
        elif mode == "json":
            return json.dumps(self.toJSON(), sort_keys=True, indent=2 if self.debug else None)
        else:
            raise InvalidArgument("Available render modes: human, ansi, html, ipython")

    def play(self, agents=None, **kwargs):
        """
        Renders a visual representation of the environment and allows interactive action selection.

        Args:
            **kwargs (dict): Args directly passed into render().  Mode is fixed to ipython.

        Returns:
            None: prints directly to an IPython notebook
        """
        if agents is None:
            agents = []

        env = self.clone()
        trainer = env.train(agents)
        interactives[env.id] = (env, trainer)
        env.render(mode="ipython", interactive=True, **kwargs)

    def train(self, agents=None):
        """
        Setup a lightweight training environment for a single agent.
        Note: This is designed to be a lightweight starting point which can
              be integrated with other frameworks (i.e. gym, stable-baselines).
              The reward returned by the "step" function here is a diff between the
              current and the previous step.

        Example:
            env = make("tictactoe")
            # Training agent in first position (player 1) against the default random agent.
            trainer = env.train([None, "random"])

            obs = trainer.reset()
            done = False
            while not done:
                action = 0 # Action for the agent being trained.
                obs, reward, done, info = trainer.step(action)
            env.render()

        Args:
            agents (list): List of agents to obtain actions from while training.
                           The agent to train (in position), should be set to "None".

        Returns:
            `dict`.reset: Reset def that reset the environment, then advances until the agents turn.
            `dict`.step: Steps using the agent action, then advance until agents turn again.
        """
        if agents is None:
            agents = []

        runner = None
        position = None
        for index, agent in enumerate(agents):
            if agent is None:
                if position is not None:
                    raise InvalidArgument(
                        "Only one agent can be marked 'None'")
                position = index

        if position is None:
            raise InvalidArgument("One agent must be marked 'None' to train.")

        def advance():
            while not self.done and self.state[position].status == "INACTIVE":
                actions, logs = runner.act()
                self.step(actions, logs)

        def reset():
            nonlocal runner
            self.reset(len(agents))
            runner = self.__agent_runner(agents)
            advance()
            return self.__get_shared_state(position).observation

        def step(action):
            actions, logs = runner.act(action)
            self.step(actions, logs)
            advance()
            agent = self.__get_shared_state(position)
            reward = agent.reward
            if len(self.steps) > 1 and reward is not None:
                reward -= self.steps[-2][position].reward
            return [
                agent.observation, reward, agent.status != "ACTIVE", agent.info
            ]

        reset()

        return structify({"step": step, "reset": reset})

    @property
    def name(self):
        """str: The name from the specification."""
        return get(self.specification, str, "", ["name"])

    @property
    def version(self):
        """str: The version from the specification."""
        return get(self.specification, str, "", ["version"])

    @property
    def done(self):
        """bool: If any agents have an ACTIVE status."""
        return all(s.status != "ACTIVE" for s in self.state)

    def toJSON(self):
        """
        Returns:
            dict: Specifcation and current state of the Environment instance.
        """
        spec = self.specification
        return copy.deepcopy(
            {
                "id": self.id,
                "name": spec.name,
                "title": spec.title,
                "description": spec.description,
                "version": spec.version,
                "configuration": self.configuration,
                "specification": {
                    "action": spec.action,
                    "agents": spec.agents,
                    "configuration": spec.configuration,
                    "info": spec.info,
                    "observation": spec.observation,
                    "reward": spec.reward
                },
                "steps": self.steps,
                "rewards": [state.reward for state in self.steps[-1]],
                "statuses": [state.status for state in self.steps[-1]],
                "schema_version": 1,
                "info": self.info,
            }
        )

    def clone(self):
        """
        Returns:
            Environment: A copy of the current environment.
        """
        return Environment(
            specification=self.specification,
            configuration=self.configuration,
            steps=self.steps,
            agents=self.agents,
            interpreter=self.interpreter,
            renderer=self.renderer,
            html_renderer=self.html_renderer,
            debug=self.debug,
        )

    @property
    def __state_schema(self):
        if not hasattr(self, "__state_schema_value"):
            spec = self.specification
            self.__state_schema_value = {
                **schemas["state"],
                "properties": {
                    "action": {
                        **schemas.state.properties.action,
                        **get(spec, dict, path=["action"], fallback={})
                    },
                    "reward": {
                        **schemas.state.properties.reward,
                        **get(spec, dict, path=["reward"], fallback={})
                    },
                    "info": {
                        **schemas.state.properties.info,
                        "properties": get(spec, dict, path=["info"], fallback={})
                    },
                    "observation": {
                        **schemas.state.properties.observation,
                        "properties": get(spec, dict, path=["observation"], fallback={})
                    },
                    "status": {
                        **schemas.state.properties.status,
                        **get(spec, dict, path=["status"], fallback={})
                    },
                },
            }
        return structify(self.__state_schema_value)

    def __set_state(self, state=None):
        if state is None:
            state = []

        if len(state) not in self.specification.agents:
            raise InvalidArgument(
                f"{len(state)} is not a valid number of agent(s).")

        self.state = structify([self.__get_state(index, s)
                                for index, s in enumerate(state)])
        self.steps = [self.state]
        return self.state

    def __get_state(self, position, state):
        key = f"__state_schema_{position}"
        if not hasattr(self, key):

            # Update a property default value based on position in defaults.
            # Remove shared properties from non-first agents.
            def update_props(props):
                for k, prop in list(props.items()):
                    if get(prop, bool, path=["shared"], fallback=False) and position > 0:
                        del props[k]
                        continue
                    if has(prop, list, path=["defaults"]) and len(prop["defaults"]) > position:
                        prop["default"] = prop["defaults"][position]
                        del prop["defaults"]
                    if has(prop, dict, path=["properties"]):
                        update_props(prop["properties"])
                return props

            props = structify(update_props(
                copy.deepcopy(self.__state_schema.properties)))

            setattr(self, key, {**self.__state_schema, "properties": props})

        err, data = process_schema(getattr(self, key), state)
        if err:
            raise InvalidArgument(
                f"Default state generation failed for #{position}: " + err
            )
        return data

    def __run_interpreter(self, state, logs):
        out = None
        err = None
        # Append any environmental logs to any agent logs we collected.
        try:
            with StringIO() as out_buffer, StringIO() as err_buffer, redirect_stdout(out_buffer), redirect_stderr(err_buffer):
                try:
                    args = [structify(state), self]
                    new_state = structify(self.interpreter(
                        *args[:self.interpreter.__code__.co_argcount]))
                    new_state[0].observation.step = (
                        0 if self.done
                        else len(self.steps)
                    )

                    for index, agent in enumerate(new_state):
                        if index < len(logs) and "duration" in logs[index]:
                            duration = logs[index]["duration"]
                            overage_time_consumed = max(0, duration - self.configuration.actTimeout)
                            agent.observation.remainingOverageTime -= overage_time_consumed
                        if agent.status not in self.__state_schema.properties.status.enum:
                            self.debug_print(f"Invalid Action: {agent.status}")
                            agent.status = "INVALID"
                        if agent.status in ["ERROR", "INVALID", "TIMEOUT"]:
                            agent.reward = None
                    return new_state
                except Exception as e:
                    # Print the exception stack trace to our log
                    traceback.print_exc(file=err_buffer)
                    # Reraise e to ensure that the program exits
                    raise e
                finally:
                    # Allow up to 1k log characters per step which is ~1MB per 600 step episode
                    max_log_length = 1024
                    out = out_buffer.getvalue()
                    err = err_buffer.getvalue()
                    if out or err:
                        logs.append({
                            "stdout": out[0:max_log_length],
                            "stderr": err[0:max_log_length]
                        })
        finally:
            if out:
                while out.endswith('\n'):
                    out = out[:-1]
                self.debug_print(out)
            if err:
                while err.endswith('\n'):
                    err = err[:-1]
                self.debug_print(err)

    def __process_specification(self, spec):
        if has(spec, path=["reward"]):
            reward = spec["reward"]
            reward_type = get(reward, str, "number", ["type"])
            if reward_type not in ["integer", "number"]:
                return ("type must be an integer or number", None)
            reward["type"] = [reward_type, "null"]

        # Allow environments to extend various parts of the specification.
        def extend_specification(source, field_name):
            field = copy.deepcopy(source[field_name]["properties"])
            for k, v in get(spec, dict, {}, [field_name]).items():
                # Set a new default value.
                if not isinstance(v, dict):
                    if not has(field, path=[k]):
                        raise InvalidArgument(
                            f"Field {field} was unable to set default of missing property: {k}")
                    field[k]["default"] = v
                # Add a new field.
                elif not has(field, path=[k]):
                    field[k] = v
                # Override an existing field if types match.
                elif field[k]["type"] == get(v, path=["type"]):
                    field[k] = v
                # Types don't match - unable to extend.
                else:
                    raise InvalidArgument(
                        f"Field {field} was unable to extend: {k}")

            spec[field_name] = field

        extend_specification(schemas, "configuration")
        extend_specification(schemas["state"]["properties"], "observation")

        return process_schema(schemas.specification, spec)

    def __agent_runner(self, agents):
        # Generate the agents.
        agents = [
            Agent(agent, self)
            if agent is not None
            else None
            for agent in agents
        ]

        def act(none_action=None):
            if len(agents) != len(self.state):
                raise InvalidArgument(
                    "Number of agents must match the state length")

            act_args = [
                (
                    agent,
                    self.__get_shared_state(i),
                    self.configuration,
                    none_action,
                )
                for i, agent in enumerate(agents)
            ]

            if all((agent is None or agent.is_parallelizable) for agent in agents):
                if self.pool is None:
                    self.pool = Pool(processes=len(agents))
                results = self.pool.map(act_agent, act_args)
            else:
                results = list(map(act_agent, act_args))

            # results is a list of tuples where the first element is an agent action and the second is the agent log
            # This destructures into two lists, a list of actions and a list of logs.
            actions, logs = zip(*results)
            return list(actions), list(logs)

        return structify({"act": act})

    def __get_shared_state(self, position):
        # Note: state and schema are required to be in sync (apart from shared ones).
        def update_props(shared_state, state, schema_props):
            for k, prop in schema_props.items():
                # Hidden fields are tracked in the episode replay but are not provided to the agent at runtime
                if get(prop, bool, path=["hidden"], fallback=False):
                    if k in state:
                        del state[k]
                elif get(prop, bool, path=["shared"], fallback=False):
                    state[k] = shared_state[k]
                elif has(prop, dict, path=["properties"]):
                    update_props(shared_state[k], state[k], prop["properties"])
            return state

        return update_props(
            self.state[0],
            copy.deepcopy(self.state[position]),
            self.__state_schema.properties
        )

    def debug_print(self, message):
        if self.debug:
            print(message)
