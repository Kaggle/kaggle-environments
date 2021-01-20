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
from sys import path
from time import perf_counter
from typing import *

from jsonschema import SchemaError, ValidationError

from .agent import AgentRunner
from .errors import DeadlineExceeded, FailedPrecondition, Internal, InvalidArgument
from .helpers import Agent, State, TState, TConfiguration, TObservation, TAction, Log, with_print, AgentStatus, Environment
from .utils import get, has, get_player, process_schema, schemas, structify, process_properties

# Registered Interactive Sessions.
interactives = {}
# Registered Environments.
environments: Dict[str, Environment] = {}


def register(name: str, environment: Environment):
    """
    Register an environment by name.  An environment contains the following:
     * specification - JSON Schema representing the environment.
     * interpreter - Function(state, environment) -> new_state
     * renderer - Function(state, environment) -> string
     * html_renderer - Function(environment) -> JavaScript HTML renderer function.
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


def act_agent(args):
    agent, state, configuration = args
    if state["status"] != AgentStatus.ACTIVE:
        return None, {}
    else:
        return agent.act(state["observation"])


class EnvironmentRunner(Generic[TState, TConfiguration]):
    def __init__(
        self,
        environment: Environment,
        configuration: Optional[TConfiguration] = None,
        steps: Optional[List[List[TState]]] = None,
        logs: Optional[List[List[Log]]] = None,
        debug: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ):
        self.logs = logs or []
        self.id = str(uuid.uuid1())
        self.debug = debug
        self.info = info
        self.pool = None
        self.environment = environment

        # TODO: Move configuration creation out of the env runner.
        self.configuration = process_properties(environment.specification["configuration"], configuration)
        self.interpreter = environment.step
        self.renderer = environment.render_text
        self.html_renderer = environment.render_html
        self.agents = environment.builtin_agents()

        if steps is not None and len(steps) > 0:
            self.__set_state(steps[-1])
            self.steps = steps[0:-1] + self.steps
        else:
            self.reset()

    @property
    def state(self):
        return self.steps[-1]

    @property
    def specification(self):
        return self.environment.specification

    @property
    def name(self):
        """str: The name from the specification."""
        return get(self.environment.specification, str, "", ["name"])

    @property
    def version(self):
        """str: The version from the specification."""
        return get(self.environment.specification, str, "", ["version"])

    @property
    def done(self):
        """bool: If all agents are in a terminal status."""
        return all(s.status.is_terminal for s in self.state)

    @property
    def shared_state(self):
        return self.state[0]

    def step(self, actions: List[TAction], logs: List[Log] = None):
        """
        Execute the environment interpreter using the current state and a list of actions.

        Args:
            actions (list): Actions to pair up with the current agent states.
            logs (list): Logs to pair up with each agent for the current step.

        Returns:
            list of dict: The agents states after the step.
        """
        logs = logs or []
        if self.done:
            # No agents can act, nothing left to simulate.
            return self.state
        if not actions or len(actions) != len(self.state):
            raise InvalidArgument(f"{len(self.state)} actions required.")

        action_states = [State()] * len(self.state)
        for index, action in enumerate(actions):
            action_state = action_states[index] = copy.deepcopy(self.state[index])
            action_state.action = None

            if isinstance(action, DeadlineExceeded):
                action_state.status = AgentStatus.TIMEOUT
            elif isinstance(action, BaseException):
                action_state.status = AgentStatus.ERROR
            else:
                try:
                    self.specification.action.validate(action)
                except ValidationError as e:
                    action_state.status = AgentStatus.INVALID
                    raise e
                action_state.action = action

        def run():
            new_state = self.environment.step(action_states, self.configuration)
            new_state[0].observation.step += 1

            for index, agent in enumerate(new_state):
                if index < len(logs) and "duration" in logs[index]:
                    duration = logs[index]["duration"]
                    overage_time_consumed = max(0, duration - self.configuration.actTimeout)
                    agent.observation.remainingOverageTime -= overage_time_consumed
                if agent.status.is_error:
                    agent.reward = None

            return new_state

        result, log = Log.collect(run)
        logs.append(log)

        if isinstance(result, Exception):
            raise result
        else:
            self.steps.append(result)

        # Max Steps reached. Mark ACTIVE/INACTIVE agents as DONE.
        if self.state[0].observation.step >= self.configuration.episodeSteps - 1:
            for s in self.state:
                if s.status == AgentStatus.ACTIVE or s.status == AgentStatus.INACTIVE:
                    s.status = AgentStatus.DONE

        self.logs.append(logs)
        return self.state

    def run(self, agents: List[Agent]):
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
            self.reset()
        if self.configuration.agent_count != len(agents):
            raise InvalidArgument(
                f"{self.configuration.agent_count} agents were expected, but {len(agents)} were given.")

        runner = self.__agent_runner(agents)
        start = perf_counter()
        while not self.done and perf_counter() - start < self.configuration.runTimeout:
            actions, logs = runner()
            self.step(actions, logs)
        return self.steps

    def __agent_runner(self, agents: List[Agent]):
        if len(agents) != len(self.state):
            raise InvalidArgument("Number of agents must match the state length")

        # Generate the agents.
        agent_runners = [
            AgentRunner(agent, self.agents, self.configuration, self.debug, self.name)
            for agent in agents
        ]

        def act():
            act_args = [
                (agent_runner, observation, self.configuration)
                for i, agent_runner in enumerate(agent_runners)
                for observation in [self.get_observation(i)]
            ]

            if all(agent_runner.is_parallelizable for agent_runner in agent_runners):
                if self.pool is None:
                    self.pool = Pool(processes=len(agents))
                results = self.pool.map(act_agent, act_args)
            else:
                results = list(map(act_agent, act_args))

            # results is a list of tuples where the first element is an agent action and the second is the agent log
            # This destructures into two lists, a list of actions and a list of logs.
            actions, logs = zip(*results)
            return list(actions), list(logs)

        return act

    def reset(self):
        """
        Resets the environment state to the initial step.

        Returns:
            list of dict: The agents states after the reset.
        """
        self.__set_state(self.environment.reset(self.configuration))

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
            window_kaggle = {
                "debug": get(kwargs, bool, self.debug, path=["debug"]),
                "autoplay": get(kwargs, bool, self.done, path=["autoplay"]),
                "step": 0 if get(kwargs, bool, self.done, path=["autoplay"]) else (len(self.steps) - 1),
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

    def train(self, agents = None):
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
        runner: Callable[[Optional[TAction]], Tuple[List[TAction], List[Log]]] = None
        position = None
        agents = agents or []
        for index, agent in enumerate(agents):
            if agent is None:
                if position is not None:
                    raise InvalidArgument("Only one agent can be marked 'None'")
                position = index

        if position is None:
            raise InvalidArgument("One agent must be marked 'None' to train.")

        def advance():
            while not self.done and self.state[position].status == AgentStatus.INACTIVE:
                actions, logs = runner(None)
                self.step(actions, logs)

        def reset():
            nonlocal runner
            self.reset()
            runner = self.__agent_runner(agents)
            advance()
            return self.get_observation(position).observation

        def step(action):
            actions, logs = runner(action)
            self.step(actions, logs)
            advance()
            agent = self.get_observation(position)
            reward = agent.reward
            if len(self.steps) > 1 and reward is not None:
                reward -= self.steps[-2][position].reward
            return [
                agent.observation, reward, agent.status != AgentStatus.ACTIVE, agent.info
            ]

        reset()

        return structify({"step": step, "reset": reset})

    def toJSON(self):
        """
        Returns:
            dict: Specifcation and current state of the Environment instance.
        """
        spec = self.environment.specification
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
        return EnvironmentRunner(
            environment=self.environment,
            configuration=self.configuration,
            steps=self.steps,
            debug=self.debug,
        )

    @property
    def __state_schema(self):
        if not hasattr(self, "__state_schema_value"):
            spec = self.environment.specification
            self.__state_schema_value = {
                **schemas["state"],
                "properties": {
                    "action": {
                        **schemas.state.properties.action,
                        **spec.action
                    },
                    "reward": {
                        **schemas.state.properties.reward,
                        **spec.reward
                    },
                    "info": {
                        **schemas.state.properties.info,
                        "properties": spec.info
                    },
                    "observation": {
                        **schemas.state.properties.observation,
                        "properties": spec.observation
                    },
                    "status": {
                        **schemas.state.properties.status,
                        **spec.status
                    },
                },
            }
        return structify(self.__state_schema_value)

    def __set_state(self, state: List[TState]) -> None:
        if len(state) != self.configuration.agent_count:
            raise InvalidArgument(f"{len(state)} is not a valid number of agent(s).")

        def get_state(position, state):
            key = f"__state_schema_{position}"
            if not hasattr(self, key):
                # Update a property default value based on position in defaults.
                # Remove shared properties from non-first agents.
                def update_props(props):
                    for k, prop in props.items():
                        if get(prop, bool, path=["shared"], fallback=False) and position > 0:
                            del props[k]
                            continue
                        if has(prop, list, path=["defaults"]) and len(prop["defaults"]) > position:
                            prop["default"] = prop["defaults"][position]
                            del prop["defaults"]
                        if has(prop, dict, path=["properties"]):
                            update_props(prop["properties"])
                    return props

                props = structify(update_props(copy.deepcopy(self.__state_schema.properties)))

                setattr(self, key, {**self.__state_schema, "properties": props})

            return State(process_schema(getattr(self, key), state))

        self.steps = [[
            structify(get_state(index, s))
            for index, s in enumerate(state)
        ]]

    def get_observation(self, position):
        """An observation consists of each agent's individual state merged over the shared state with hidden properties removed."""
        return self.specification.observation.state_to_observation(self.state, position)


def make(environment, configuration = None, info = None, steps = None, logs = None, debug = False) -> EnvironmentRunner:
    return EnvironmentRunner(environments[environment], configuration=configuration, info=info, steps=steps, logs=logs, debug=debug)