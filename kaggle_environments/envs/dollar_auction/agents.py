import random


def random_agent(observation, configuration):
    return bool(random.getrandbits(1))


def true_agent(observation, configuration):
    return True


def false_agent(observation, configuration):
    return False


agents = {
    "random": random_agent,
    "true": true_agent,
    "false": false_agent
}