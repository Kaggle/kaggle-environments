import json
import numpy as np
import random
from os import path
from .agents import agents as all_agents
from ..helpers import *


class Observation(Observation):
    """This provides bindings for the observation type described at https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/mab/mab.json"""
    @property
    def last_opponent_action(self):
        """Bandit chosen by opponent last step. None on the first step."""
        return self["lastOpponentAction"]

    @property
    def reward(self):
        """Current reward of the agent."""
        return self["reward"]


class Configuration(Configuration):
    """This provides bindings for the configuration type described at https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/mab/mab.json"""
    @property
    def bandit_count(self):
        """Number of bandits available to choose from. Max bandit is this number -1."""
        return self["banditCount"]

    @property
    def seed(self):
        """Seed value used to initialize bandits for this episode."""
        return self["seed"]

    @seed.setter
    def seed(self, value):
        self._data["seed"] = value


def interpreter(state, env):
    configuration = Configuration(env.configuration)

    if env.done:
        if not hasattr(configuration, "seed"):
            max_int_32 = (1 << 31) - 1
            configuration.seed = random.randrange(max_int_32)

        np.random.seed(configuration.seed)
        random.seed(configuration.seed)

        return state

    player1 = state[0]
    player2 = state[1]

    def is_valid_action(player):
        return (
            player.action is not None and
            isinstance(player.action, int) and
            0 <= player.action < configuration.bandit_count
        )

    # Check for validity of actions
    is_player1_valid, is_player2_valid = is_valid_action(player1), is_valid_action(player2)
    if not is_player2_valid:
        player2.status = "INVALID"
        player2.reward = 0

        if is_player1_valid:
            player1.status = "DONE"
            player1.reward = 1
            return state

    if not is_player1_valid:
        player1.status = "INVALID"
        player1.reward = 0

        if is_player2_valid:
            player2.status = "DONE"
            player2.reward = 1

        return state

    score = get_score(player1.action, player2.action)
    player1.observation.lastOpponentAction = player2.action
    player1.reward += score
    player2.observation.lastOpponentAction = player1.action
    player2.reward -= score
    player1.observation.reward = int(player1.reward)
    player2.observation.reward = int(player2.reward)
    remaining_steps = env.configuration.episodeSteps - step - 1
    if remaining_steps <= 0:
        player1.status = "DONE"
        player2.status = "DONE"
        # Player performance too similar, consider the match a tie.
        if abs(player1.reward) < env.configuration.tieRewardThreshold:
            player1.reward = 0
            player2.reward = 0
    return state


def renderer(state, env):
    return "Hello!"


dir_path = path.dirname(__file__)
json_path = path.abspath(path.join(dir_path, "rps.json"))
with open(json_path) as json_file:
    specification = json.load(json_file)


def html_renderer():
    js_path = path.abspath(path.join(dir_path, "rps.js"))
    with open(js_path, encoding="utf-8") as js_file:
        return js_file.read()


agents = all_agents
