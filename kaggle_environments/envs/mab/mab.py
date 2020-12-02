import json
from functools import reduce
from os import path
from random import SystemRandom
from .agents import agents as all_agents
from ...helpers import *


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
    
    @property
    def last_action(self):
        """Bandit chosen by this agent last step. None on the first step."""
        return self["lastAction"]    


class Configuration(Configuration):
    """This provides bindings for the configuration type described at https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/mab/mab.json"""
    @property
    def bandit_count(self):
        """Number of bandits available to choose from. Max bandit is this number -1."""
        return self["banditCount"]


random = SystemRandom()
initial_thresholds = None
current_thresholds = None

def interpreter(state, env):
    configuration = Configuration(env.configuration)

    global current_thresholds    
    
    if env.done:
        global initial_thresholds
        initial_thresholds = [
            random.randint(0, 100)
            for _ in range(configuration.bandit_count)
        ]
        
        current_thresholds = initial_thresholds.copy()     
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

    actions = [[agent.action for agent in step] for step in env.steps]
    
    decay_rate = 1.03 ## should probably set from config?        
    recovery_rate = 2.0 - decay_rate #0.9
            
    player1.observation.lastOpponentAction = player2.action
    player1.observation.lastAction = player1.action
    player1.reward += 1 if random.randint(0, 100) > current_thresholds[player1.action] else 0
    player2.observation.lastOpponentAction = player1.action
    player2.observation.lastAction = player2.action    
    player2.reward += 1 if random.randint(0, 100) > current_thresholds[player2.action] else 0
    player1.observation.reward = int(player1.reward)
    player2.observation.reward = int(player2.reward)

    current_thresholds = reduce(
        lambda thresholds, current_actions:
            [
                min(max(initial_thresholds[i], threshold * (decay_rate if i in current_actions else recovery_rate)), 100.0)
                for i, threshold in enumerate(thresholds)
            ],
        actions, initial_thresholds)
    
    remaining_steps = env.configuration.episodeSteps - player1.observation.step - 1
    if remaining_steps <= 0:
        player1.status = "DONE"
        player2.status = "DONE"
    return state


def renderer(state, env):
    rounds_played = len(env.steps)
    board = ""

    # This line prints results each round, good for debugging
    for i in range(1, rounds_played):
        step = env.steps[i]
        right_move = step[0].observation.lastOpponentAction
        left_move = step[1].observation.lastOpponentAction
        board += f"Round {i}: {left_move} vs {right_move}, Score: {step[0].reward} to {step[1].reward}\n"

    board += f"Game ended on round {rounds_played - 1}, final score: {state[0].reward} to {state[0].reward}\n"
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
