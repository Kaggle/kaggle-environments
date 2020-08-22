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
    global last_react_action
    if observation.step == 0:
        last_react_action = random.randrange(0, configuration.signs)
    elif get_score(last_react_action, observation.last_opponent_action) <= 1:
        last_react_action = (observation.last_opponent_action + 1) % configuration.signs

    return last_react_action


last_counter_action = None


def counter_reactionary(observation, configuration):
    global last_counter_action
    if observation.step == 0:
        last_counter_action = random.randrange(0, configuration.signs)
    elif get_score(last_counter_action, observation.last_opponent_action) == 1:
        last_counter_action = (last_counter_action + 2) % configuration.signs
    else:
        last_counter_action = (observation.last_opponent_action + 1) % configuration.signs

    return last_counter_action


action_histogram = {}


def statistical(observation, configuration):
    global action_histogram
    if observation.step == 0:
        action_histogram = {}
        return
    action = observation.last_opponent_action
    if action not in action_histogram:
        action_histogram[action] = 0
    action_histogram[action] += 1
    mode_action = None
    mode_action_count = None
    for k, v in action_histogram.items():
        if mode_action_count is None or v > mode_action_count:
            mode_action = k
            mode_action_count = v
            continue

    return (mode_action + 1) % configuration.signs


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
    "counter_reactionary": counter_reactionary,
    "statistical": statistical
}