import random


def random_agent(observation, configuration):
    return random.randrange(configuration.banditCount - 1)


agents = {
    "random": random_agent
}