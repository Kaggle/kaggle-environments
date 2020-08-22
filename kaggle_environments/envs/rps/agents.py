import random
from .utils import get_score


def random_agent(observation, configuration):
    return random.randrange(0, configuration.signs)


def rock(observation, configuration):
    return 0


def paper(observation, configuration):
    return 1


def scissors(observation, configuration):
    return 2


def rockish(observation, configuration):
    rand = random.random()

    if rand > 0.5:
        return random.randrange(0, configuration.signs)
    else:
        return 0


def paperish(observation, configuration):
    rand = random.random()

    if rand > 0.5:
        return random.randrange(0, configuration.signs)
    else:
        return 1


def scissorsish(observation, configuration):
    rand = random.random()

    if rand > 0.5:
        return random.randrange(0, configuration.signs)
    else:
        return 2


def copy_opponent(observation, configuration):
    if observation.step > 0:
        return observation.last_opponent_action
    else:
        return random.randrange(0, configuration.signs)


last_react_action = None


def reactionary(observation, configuration):
    global last_action
    if observation.step == 0:
        last_action = random.randrange(0, configuration.signs)
    elif get_score(last_action, observation.last_opponent_action) <= 1:
        last_action = (observation.last_opponent_action + 1) % configuration.signs

    return last_action


last_counter_reaction = None


def counter_reactionary(observation, configuration):
    global last_counter_reaction
    if observation.step == 0:
        last_counter_reaction = random.randrange(0, configuration.signs)
    elif get_score(last_counter_reaction, observation.last_opponent_action) == 1:
        last_counter_reaction = (last_counter_reaction + 2) % configuration.signs
    else:
        last_counter_reaction = (observation.last_opponent_action + 1) % configuration.signs

    return last_counter_reaction


agents = {
    "random": random_agent,
    "rock": rock,
    "paper": paper,
    "scissors": scissors,
    "rockish": rockish,
    "paperish": paperish,
    "scissorsish": scissorsish,
    "copy_opponent": copy_opponent,
    "reactionary": reactionary,
    "counter_reactionary": counter_reactionary
}