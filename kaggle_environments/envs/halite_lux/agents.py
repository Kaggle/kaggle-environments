import random
from .utils import get_score


def random_agent(observation, configuration):
    return ["m 0 n", "m 1 n"]

agents = {
    "random_agent": random_agent
}