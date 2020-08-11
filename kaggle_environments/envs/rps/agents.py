import random


def random_agent(observation, configuration):
    return random.randint(0, configuration.weapons - 1)


def rock(observation, configuration):
    return 0


def paper(observation, configuration):
    return 1


def scissors(observation, configuration):
    return 2


def rockish(observation, configuration):
    rand = random.random()

    if rand > 0.5:
        return random.randint(0, configuration.weapons - 1)
    else:
        return 0


def paperish(observation, configuration):
    rand = random.random()

    if rand > 0.5:
        return random.randint(0, configuration.weapons - 1)
    else:
        return 1


def scissorsish(observation, configuration):
    rand = random.random()

    if rand > 0.5:
        return random.randint(0, configuration.weapons - 1)
    else:
        return 2


def copy_opponent(observation, configuration):
    if observation.round > 0:
        return observation.opponent_last_action
    else:
        return random.randint(0, configuration.weapons - 1)


def reactionary(observation, configuration):
    if observation.round == 0:
        return random.randint(0, configuration.weapons - 1)
    
    if observation.your_last_score == 1:
        return observation.your_last_action
    else:
        return (observation.opponent_last_action + 1) % configuration.weapons


def counter_reactionary(observation, configuration):
    if observation.round == 0:
        return random.randint(0, configuration.weapons - 1)

    if observation.your_last_score == 1:
        return (observation.your_last_action + 2) % configuration.weapons
    else:
        return (observation.opponent_last_action + 1) % configuration.weapons


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