import random


def random_agent(observation, configuration):
    return random.randrange(configuration.banditCount - 1)


def round_robin_agent(observation, configuration):
    return observation.step % configuration.banditCount


agents = {
    "random": random_agent,
    "round_robin": round_robin_agent
}