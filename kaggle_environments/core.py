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

import copy
import json
from time import time
import uuid
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
     * html_renderer(optional) - JavaScript HTML renderer function.
     * agents(optional) - List of default agents [Function(observation, config) -> action]
    """
    environments[name] = environment


def evaluate(environment, agents=[], configuration={}, steps=[], num_episodes=1):
    """
    Evaluate and return the rewards of one or more episodes (environment and agents combo).

    Args:
        environment (str|Environment): 
        agents (list):
        configuration (dict, optional):
        steps (list, optional):
        num_episodes (int=1, optional): How many episodes to execute (run until done).

    Returns:
        list of list of int: List of final rewards for all agents for all episodes.
    """
    e = make(environment, configuration, steps)
    rewards = [[]] * num_episodes
    for i in range(num_episodes):
        last_state = e.run(agents)[-1]
        rewards[i] = [state.reward for state in last_state]
    return rewards


def make(environment, configuration={}, steps=[], debug=False):
    """
    Creates an instance of an Environment.

    Args:
        environment (str|Environment): 
        configuration (dict, optional):
        steps (list, optional):
        debug (bool=False, optional):

    Returns:
        Environment: Instance of a specific environment.
    """
    if has(environment, str) and has(environments, dict, path=[environment]):
        return Environment(**environments[environment], configuration=configuration, steps=steps, debug=debug)
    elif callable(environment):
        return Environment(interpreter=environment, configuration=configuration, steps=steps, debug=debug)
    elif has(environment, path=["interpreter"], is_callable=True):
        return Environment(**environment, configuration=configuration, steps=steps, debug=debug)
    raise InvalidArgument("Unknown Environment Specification")


class Environment:

    def __init__(
        self,
        specification={},
        configuration={},
        steps=[],
        agents={},
        interpreter=None,
        renderer=None,
        html_renderer=None,
        debug=False,
    ):
        self.id = str(uuid.uuid1())
        self.debug = debug

        err, specification = self.__process_specification(specification)
        if err:
            raise InvalidArgument("Specification Invalid: " + err)
        self.specification = structify(specification)

        err, configuration = process_schema(
            {"type": "object", "properties": self.specification.configuration},
            {} if configuration == None else configuration,
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

        if callable(html_renderer):
            html_renderer = html_renderer()
        self.html_renderer = get(html_renderer, str, "")

        if not all([callable(a) for a in agents.values()]):
            raise InvalidArgument("Default agents must be Callable.")
        self.agents = structify(agents)

        if steps == None or len(steps) == 0:
            self.reset()
        else:
            self.__set_state(steps[-1])
            self.steps = steps[0:-1] + self.steps

    def step(self, actions):
        """
        Execute the environment interpreter using the current state and a list of actions.

        Args:
            actions (list): Actions to pair up with the current agent states.

        Returns:
            list of dict: The agents states after the step.
        """

        if self.done:
            raise FailedPrecondition("Environment done, reset required.")
        if not actions or len(actions) != len(self.state):
            raise InvalidArgument(f"{len(self.state)} actions required.")

        action_state = [0] * len(self.state)
        for index, action in enumerate(actions):
            action_state[index] = {**self.state[index], "action": None}

            if isinstance(action, DeadlineExceeded):
                self.__debug_print(f"Timeout: {str(action)}")
                action_state[index]["status"] = "TIMEOUT"
            elif isinstance(action, BaseException):
                self.__debug_print(f"Error: {str(action)}")
                action_state[index]["status"] = "ERROR"
            else:
                err, data = process_schema(
                    self.__state_schema.properties.action, action)
                if err:
                    self.__debug_print(f"Invalid Action: {str(err)}")
                    action_state[index]["status"] = "INVALID"
                else:
                    action_state[index]["action"] = data

        self.state = self.__run_interpreter(action_state)

        # Max Steps reached. Mark ACTIVE/INACTIVE agents as DONE.
        if len(self.steps) == self.configuration.episodeSteps - 1:
            for s in self.state:
                if s.status == "ACTIVE" or s.status == "INACTIVE":
                    s.status = "DONE"

        self.steps.append(self.state)

        return self.state

    def run(self, agents):
        """
        Steps until the environment is "done" or the runTimeout was reached.

        Args:
            agents (list of any): List of agents to obtain actions from.

        Returns:
            list of list of dict: The agent states of all steps executed.
        """
        if self.state == None or len(self.steps) == 1 or self.done:
            self.reset(len(agents))
        if len(self.state) != len(agents):
            raise InvalidArgument(
                f"{len(self.state)} agents were expected, but {len(agents)} was given.")

        runner = self.__agent_runner(agents)
        start = time()
        while not self.done and time() - start < self.configuration.runTimeout:
            self.step(runner.act())
        runner.destroy()
        return self.steps

    def reset(self, num_agents=None):
        """
        Resets the environment state to the initial step.

        Args:
            num_agents (int): Resets the state assuming a fixed number of agents.

        Returns:
            list of dict: The agents states after the reset.
        """

        if num_agents == None:
            num_agents = self.specification.agents[0]

        # Get configuration default state.
        self.__set_state([{} for _ in range(num_agents)])
        # Reset all agents to status=INACTIVE (copy out values to reset afterwards).
        statuses = [a.status for a in self.state]
        for agent in self.state:
            agent.status = "INACTIVE"
        # Give the interpreter an opportunity to make any initializations.
        self.__set_state(self.__run_interpreter(self.state))
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
            print(out)
        elif mode == "html" or mode == "ipython":
            window_kaggle = {
                "debug": get(kwargs, bool, self.debug, path=["debug"]),
                "autoplay": get(kwargs, bool, self.done, path=["autoplay"]),
                "step": 0 if get(kwargs, bool, self.done, path=["autoplay"]) else (len(self.steps) - 1),
                "controls": get(kwargs, bool, self.done, path=["controls"]),
                "environment": self.toJSON(),
                **kwargs,
            }
            player_html = get_player(window_kaggle, self.html_renderer)
            if mode == "html":
                return player_html
            from IPython.display import display, HTML
            player_html = player_html.replace('"', '&quot;')
            width = get(kwargs, int, 300, path=["width"])
            height = get(kwargs, int, 300, path=["height"])
            html = f'<iframe srcdoc="{player_html}" width="{width}" height="{height}" frameborder="0"></iframe> '
            display(HTML(html))
        elif mode == "json":
            return json.dumps(self.toJSON(), sort_keys=True)
        else:
            raise InvalidArgument(
                "Available render modes: human, ansi, html, ipython")

    def play(self, agents=[], **kwargs):
        """
        Renders a visual representation of the environment and allows interactive action selection.

        Args:
            **kwargs (dict): Args directly passed into render().  Mode is fixed to ipython.

        Returns:
            None: prints directly to an IPython notebook
        """
        env = self.clone()
        trainer = env.train(agents)
        interactives[env.id] = (env, trainer)
        env.render(mode="ipython", interactive=True, **kwargs)

    def train(self, agents=[]):
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
        runner = None
        position = None
        for index, agent in enumerate(agents):
            if agent == None:
                if position != None:
                    raise InvalidArgument(
                        "Only one agent can be marked 'None'")
                position = index

        if position == None:
            raise InvalidArgument("One agent must be marked 'None' to train.")

        def advance():
            while not self.done and self.state[position].status == "INACTIVE":
                self.step(runner.act())

        def reset():
            nonlocal runner
            self.reset(len(agents))
            if runner != None:
                runner.destroy()
            runner = self.__agent_runner(agents)
            advance()
            return self.state[position].observation

        def step(action):
            self.step(runner.act(action))
            advance()
            agent = self.state[position]
            reward = agent.reward
            if len(self.steps) > 1 and reward != None:
                reward -= self.steps[-2][position].reward
            if self.done:
                runner.destroy()
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

    def __set_state(self, state=[]):
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

    def __run_interpreter(self, state):
        try:
            args = [structify(state), self]
            new_state = structify(self.interpreter(
                *args[:self.interpreter.__code__.co_argcount]))
            for agent in new_state:
                if agent.status not in self.__state_schema.properties.status.enum:
                    self.__debug_print(f"Invalid Action: {agent.status}")
                    agent.status = "INVALID"
                if agent.status in ["ERROR", "INVALID", "TIMEOUT"]:
                    agent.reward = None
            return new_state
        except Exception as err:
            raise Internal("Error running environment: " + str(err))

    def __process_specification(self, spec):
        if has(spec, path=["reward"]):
            reward = spec["reward"]
            reward_type = get(reward, str, "number", ["type"])
            if reward_type not in ["integer", "number"]:
                return ("type must be an integer or number", None)
            reward["type"] = [reward_type, "null"]

        # Allow environments to extend the default configuration.
        configuration = copy.deepcopy(
            schemas["configuration"]["properties"])

        for k, v in get(spec, dict, {}, ["configuration"]).items():
            # Set a new default value.
            if not isinstance(v, dict):
                if not has(configuration, path=[k]):
                    raise InvalidArgument(
                        f"Configuration was unable to set default of missing property: {k}")
                configuration[k]["default"] = v
            # Add a new configuration.
            elif not has(configuration, path=[k]):
                configuration[k] = v
            # Override an existing configuration if types match.
            elif configuration[k]["type"] == get(v, path=["type"]):
                configuration[k] = v
            # Types don't match - unable to extend.
            else:
                raise InvalidArgument(
                    f"Configuration was unable to extend: {k}")

        spec["configuration"] = configuration
        return process_schema(schemas.specification, spec)

    def __agent_runner(self, agents):
        # Replace default agents with their source.
        for i, agent in enumerate(agents):
            if has(self.agents, path=[agent]):
                agents[i] = self.agents[agent]

        # Generate the agents.
        agents = [Agent(a, self.configuration) if a !=
                  None else None for a in agents]

        # Have the agents had a chance to initialize (first non-empty act).
        initialized = [False] * len(agents)

        def act(none_action=None):
            if len(agents) != len(self.state):
                raise InvalidArgument(
                    "Number of agents must match the state length")

            actions = [0] * len(agents)
            for i, agent in enumerate(agents):
                if self.state[i]["status"] != "ACTIVE":
                    actions[i] = None
                elif agent == None:
                    actions[i] = none_action
                else:
                    timeout = self.configuration.actTimeout
                    if not initialized[i]:
                        initialized[i] = True
                        timeout += self.configuration.agentTimeout
                    state = self.__get_shared_state(i)
                    actions[i] = agent.act(state, timeout)
            return actions

        def destroy():
            for a in agents:
                a.destroy()

        return structify({"act": act, "destroy": destroy})

    def __get_shared_state(self, position):
        if position == 0:
            return self.state[0]
        state = copy.deepcopy(self.state[position])

        # Note: state and schema are required to be in sync (apart from shared ones).
        def update_props(shared_state, state, schema_props):
            for k, prop in schema_props.items():
                if get(prop, bool, path=["shared"], fallback=False):
                    state[k] = shared_state[k]
                elif has(prop, dict, path=["properties"]):
                    update_props(shared_state[k], state[k], prop["properties"])
            return state

        return update_props(self.state[0], state, self.__state_schema.properties)

    def __debug_print(self, message):
        if self.debug:
            print(message)
