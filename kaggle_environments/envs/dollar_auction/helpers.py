import kaggle_environments.helpers

from typing import *


class Configuration(kaggle_environments.helpers.Configuration):
    """This provides bindings for the configuration type described at https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/dollar_auction/dollar_auction.json"""
    @property
    def auction_reward(self) -> int:
        """Reward provided by winning an auction."""
        return self["auctionReward"]


class Observation(kaggle_environments.helpers.Observation):
    """This provides bindings for the observation type described at https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/dollar_auction/dollar_auction.json"""
    @property
    def agent_index(self) -> int:
        """The index of the current agent."""
        return self["agentIndex"]

    @property
    def current_bid(self) -> int:
        """The index of the current agent."""
        return self["currentBid"]

    @current_bid.setter
    def current_bid(self, value: int) -> None:
        self["currentBid"] = value

    @property
    def ultimate_bidder_index(self) -> int:
        """The index of the last agent who bid. This is not passed to agents."""
        return self["ultimateBidderIndex"]

    @ultimate_bidder_index.setter
    def ultimate_bidder_index(self, value: int) -> None:
        self["ultimateBidderIndex"] = value

    @property
    def penultimate_bidder_index(self) -> int:
        # REVIEW: I wanted an expressive name for the bidder index fields, but is this too wordy? It's only used in the environment and replays, not agents.
        """The index of the second-last agent who bid. This is not passed to agents."""
        return self["penultimateBidderIndex"]

    @penultimate_bidder_index.setter
    def penultimate_bidder_index(self, value: int) -> None:
        self["penultimateBidderIndex"] = value

    @property
    def reward(self) -> int:
        """The index of the current agent."""
        return self["reward"]

    @reward.setter
    def reward(self, value: int) -> None:
        self["reward"] = value


Action = NewType('Action', bool)
Agent = kaggle_environments.helpers.Agent[Configuration, Observation, Action]


def make_cached_agent(agent_constructor: Callable[[Configuration], Agent]):
    return kaggle_environments.helpers.make_cached_agent(agent_constructor, Configuration, Observation)