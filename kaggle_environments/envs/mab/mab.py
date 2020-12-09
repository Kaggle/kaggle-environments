import json
import kaggle_environments.helpers
from os import path
from random import SystemRandom
from typing import List
from .agents import agents as all_agents


class Observation(kaggle_environments.helpers.Observation):
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


class Configuration(kaggle_environments.helpers.Configuration):
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


# SystemRandom is used to provide stronger randoms than builtin twister
random = SystemRandom()


def interpreter(agents, env):
    configuration = Configuration(env.configuration)
    shared_agent = agents[0]
    # Assign shared_agent.observation so that changes that we make to the shared observation are propagated back to the agent state.
    shared_agent.observation = shared_observation = Observation(shared_agent.observation)

    def sample():
        """Obtain a value between 0 and sampleResolution to check against a bandit threshold."""
        return random.randint(0, configuration.sample_resolution)

    if env.done:
        # Initialize thresholds
        shared_observation.last_actions = None
        shared_observation.thresholds = [sample() for _ in range(configuration.bandit_count)]
        return agents

    # Provide actions in the next observation so agents can monitor opponents.
    shared_observation.last_actions = [agent.action for agent in agents]
    thresholds = shared_observation.thresholds

    for agent in agents:
        if (
            agent.action is not None and
            isinstance(agent.action, int) and
            0 <= agent.action < configuration.bandit_count
        ):
            # If the sample is less than the threshold the agent gains reward, otherwise nothing
            agent.reward += 1 if sample() < thresholds[agent.action] else 0
            agent.observation.reward = agent.reward
        else:
            agent.status = "INVALID"
            agent.reward = -1

    initial_thresholds = env.steps[0][0].observation.thresholds
    action_histogram = kaggle_environments.helpers.histogram(shared_observation.last_actions)

    for index, threshold in enumerate(thresholds):
        # Every time a threshold is selected it is multiplied by (decay_rate) for each agent that selected it.
        # When a threshold is not selected it is reduced by (decay_rate) ^ 0 (i.e. no recovery).
        action_count = action_histogram[index] if index in action_histogram else 0
        update_rate = (configuration.decay_rate) ** action_count
        thresholds[index] = min(threshold * update_rate, initial_thresholds[index])

    active_agents = [
        agent for agent in agents
        if agent.status == "ACTIVE" or agent.status == "INACTIVE"
    ]

    if len(active_agents) <= 1:
        for agent in active_agents:
            agent.status = "DONE"

    return agents


def renderer(steps, env):
    rounds_played = len(env.steps)
    board = ""

    for i in range(1, rounds_played):
        actions = [agent.action for agent in steps[i]]
        rewards = [agent.reward for agent in steps[i]]
        board += f"Round {i} Actions: {actions}, Rewards: {rewards}\n"

    return board


dir_path = path.dirname(__file__)
json_path = path.abspath(path.join(dir_path, "mab.json"))
with open(json_path) as json_file:
    specification = json.load(json_file)


def html_renderer():
    js_path = path.abspath(path.join(dir_path, "mab.js"))
    with open(js_path, encoding="utf-8") as js_file:
        return js_file.read()


agents = all_agents
