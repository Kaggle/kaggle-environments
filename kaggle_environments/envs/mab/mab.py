from kaggle_environments.helpers import *
from os import path
from random import SystemRandom
from typing import List
from .agents import agents as all_agents


# SystemRandom is used to provide stronger randoms than builtin twister
random = SystemRandom()


class MabObservation(Observation):
    @property
    def agent_index(self) -> float:
        """The current agent's index within observation.last_actions."""
        return self["agentIndex"]

    """This provides bindings for the observation type described at https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/mab/mab.json"""
    @property
    def last_actions(self) -> List[int]:
        """Bandit chosen by opponent last step. None on the first step."""
        return self["lastActions"]

    @last_actions.setter
    def last_actions(self, value):
        self["lastActions"] = value

    @property
    def reward(self) -> float:
        """Current reward of the agent."""
        return self["reward"]

    @reward.setter
    def reward(self, value):
        self["reward"] = value

    @property
    def thresholds(self) -> List[float]:
        """Probability values for each machine payout on this step. This value is None at agent runtime."""
        return (
            self["thresholds"]
            if "thresholds" in self
            else None
        )

    @thresholds.setter
    def thresholds(self, value):
        self["thresholds"] = value


class MabConfiguration(Configuration):
    """This provides bindings for the configuration type described at https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/mab/mab.json"""
    @property
    def bandit_count(self) -> int:
        """Number of bandits available to choose from. Max action is this number -1."""
        return self["banditCount"]

    @property
    def decay_rate(self) -> float:
        """Rate that reward chance threshold increases per step that a bandit is chosen by an agent."""
        return self["decayRate"]

    @property
    def sample_resolution(self) -> int:
        """Maximum value that can be returned by a bandit."""
        return self["sampleResolution"]

    @property
    def initial_thresholds(self) -> List[float]:
        if "initialThresholds" not in self:
            self["initialThresholds"] = [
                random.randint(0, self.sample_resolution)
                for _ in range(self.bandit_count)
            ]
        return self["initialThresholds"]


MabState = State[MabObservation, int]


def renderer(steps, env):
    rounds_played = len(env.steps)
    board = ""

    for i in range(1, rounds_played):
        actions = [agent.action for agent in steps[i]]
        rewards = [agent.reward for agent in steps[i]]
        board += f"Round {i} Actions: {actions}, Rewards: {rewards}\n"

    return board


dir_path = path.dirname(__file__)


def html_renderer():
    js_path = path.abspath(path.join(dir_path, "mab.js"))
    with open(js_path, encoding="utf-8") as js_file:
        return js_file.read()


agents = all_agents


class MabEnvironment(Environment[MabState, MabConfiguration]):
    def __init__(self):
        json_path = path.abspath(path.join(dir_path, "mab.json"))
        self._specification = Environment.load_specification(json_path)

    @property
    def specification(self) -> Specification:
        return self._specification

    def reset(self, configuration: MabConfiguration) -> List[MabState]:
        states = [
            # Scrub shared fields from non-1st agents.
            state if i == 0 else self.specification.unshare(state)
            for i in range(configuration.agent_count)
            for state in [MabState(self.specification.default)]
        ]
        states[0].observation.thresholds = configuration.initial_thresholds
        return states

    def step(self, state: List[MabState], configuration: MabConfiguration) -> List[MabState]:
        shared_observation = state[0].observation
        # Provide actions in the next observation so agents can monitor opponents.
        shared_observation.last_actions = [agent.action for agent in state]
        thresholds = shared_observation.thresholds

        for agent in state:
            if (
                agent.action is not None and
                0 <= agent.action < configuration.bandit_count
            ):
                sample = random.randint(0, configuration.sample_resolution)
                is_win = sample < thresholds[agent.action]
                agent.reward += 1 if is_win else 0
                agent.observation.reward = agent.reward
            else:
                agent.status = "INVALID"
                agent.reward = -1

        action_histogram = histogram(shared_observation.last_actions)

        for index, threshold in enumerate(thresholds):
            # Every time a threshold is selected it is multiplied by (decay_rate) for each agent that selected it.
            # When a threshold is not selected it is reduced by (decay_rate) ^ 0 (i.e. no recovery).
            action_count = action_histogram[index] if index in action_histogram else 0
            update_rate = configuration.decay_rate ** action_count
            thresholds[index] = min(threshold * update_rate, configuration.initial_thresholds[index])

        active_agents = [
            agent for agent in state
            if not agent.status.is_terminal
        ]

        if len(active_agents) <= 1:
            for agent in active_agents:
                agent.status = "DONE"

        return state