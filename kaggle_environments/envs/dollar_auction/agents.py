import random

from kaggle_environments.envs.dollar_auction.helpers import Observation, Configuration, Agent, make_cached_agent


class RandomAgent(Agent):
    def __init__(self, configuration: Configuration):
        self.configuration = configuration

    def __call__(self, observation: Observation) -> bool:
        return bool(random.getrandbits(1))


class TrueAgent(Agent):
    def __init__(self, configuration: Configuration):
        self.configuration = configuration

    def __call__(self, observation: Observation) -> bool:
        return True


class FalseAgent(Agent):
    def __init__(self, configuration: Configuration):
        self.configuration = configuration

    def __call__(self, observation: Observation) -> bool:
        return False


class RetirementAgent(Agent):
    def __init__(self, configuration: Configuration):
        self.configuration = configuration

    def __call__(self, observation: Observation) -> bool:
        risk = int(self.configuration.auction_reward * (self.configuration.episode_steps - observation.step) / self.configuration.episode_steps)
        return risk > observation.current_bid


class HistogramAgent(Agent):
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.histogram = {}
        self.total_auctions = 1
        self.last_bid = 0

    def __call__(self, observation: Observation) -> bool:
        current_bid = observation.current_bid
        if current_bid not in self.histogram:
            self.histogram[current_bid] = 0
        self.histogram[current_bid] += 1

        if current_bid + 1 not in self.histogram:
            return False

        probability = self.histogram[current_bid + 1] / self.total_auctions
        gain = self.configuration.auction_reward * probability
        loss = (1 - probability) * current_bid

        print(probability, gain, loss)
        if self.last_bid > current_bid:
            self.total_auctions += 1
        self.last_bid = current_bid

        return gain > loss


agents = {
    "random": make_cached_agent(RandomAgent),
    "true": make_cached_agent(TrueAgent),
    "false": make_cached_agent(FalseAgent),
    "retirement": make_cached_agent(RetirementAgent),
    "histogram": make_cached_agent(HistogramAgent)
}