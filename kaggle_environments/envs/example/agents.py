# Every environment should include at least one builtin agent for local testing
# Typically a good default is to choose a random sample from the action space on each step
# You can include as many builtin agents as you like


import random


def random_agent(observation, configuration):
    return random.randrange(configuration.banditCount - 1)


def round_robin_agent(observation, configuration):
    return observation.step % configuration.banditCount


# Builtin agents must be included in the `agents` dict to be referenced by users from the command line
# e.g. kaggle-environments run --environment example --agents random round_robin
# The names passed to the --agents flag correspond to the keys of this dict
agents = {
    "random": random_agent,
    "round_robin": round_robin_agent
}