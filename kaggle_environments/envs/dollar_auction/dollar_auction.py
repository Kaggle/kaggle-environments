import json
import kaggle_environments.helpers
from os import path
from random import SystemRandom
from typing import List
from .agents import agents as all_agents


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


# REVIEW: If an agent passes on a bid should they be knocked out?
# REVIEW: Should we randomize order of agents between auctions?


def interpreter(agents, env):
    configuration = Configuration(env.configuration)
    shared_agent = agents[0]
    # Assign shared_agent.observation so that changes that we make to the shared observation are propagated back to the agent state.
    shared_agent.observation = shared_observation = Observation(shared_agent.observation)

    if env.done:
        # Initialize environment
        for agent in agents:
            agent.status = "INACTIVE"
        shared_agent.status = "ACTIVE"
        shared_observation.current_bid = 0
        shared_observation.penultimate_bidder_index = None
        shared_observation.ultimate_bidder_index = None
        return agents

    next_agent_index = None
    for index, agent in enumerate(agents):
        if agent.status == "ACTIVE":
            if shared_observation.ultimate_bidder_index == index:
                # All agents have had a chance to bid, end the auction and start a new one.
                agents[shared_observation.ultimate_bidder_index].reward += configuration.auction_reward - shared_observation.current_bid
                if shared_observation.penultimate_bidder_index is not None:
                    agents[shared_observation.penultimate_bidder_index].reward -= shared_observation.current_bid - 1
                shared_observation.current_bid = 0
                shared_observation.penultimate_bidder_index = None
                shared_observation.ultimate_bidder_index = None
            elif agent.action and agent.reward > shared_observation.current_bid:
                shared_observation.current_bid += 1
                shared_observation.penultimate_bidder_index = shared_observation.ultimate_bidder_index
                shared_observation.ultimate_bidder_index = index
            next_agent_index = (index + 1) % len(agents)
            agent.status = "INACTIVE"

    for agent in agents:
        agent.observation.reward = agent.reward
        if agent.reward <= 0:
            agent.status = "DONE"

    next_agent = agents[next_agent_index]
    next_agent.status = "ACTIVE"

    active_agents = [
        agent for agent in agents
        if agent.status == "ACTIVE" or agent.status == "INACTIVE"
    ]

    if len(active_agents) <= 1:
        for agent in active_agents:
            agent.status = "DONE"

    return agents


def renderer(steps, env):
    rounds_played = len(env.steps)
    board = ""

    for i in range(1, rounds_played):
        actions = [agent.action for agent in steps[i]]
        rewards = [agent.reward for agent in steps[i]]
        board += f"Round {i} Actions: {actions}, Rewards: {rewards}\n"

    return board


dir_path = path.dirname(__file__)
json_path = path.abspath(path.join(dir_path, "dollar_auction.json"))
with open(json_path) as json_file:
    specification = json.load(json_file)


def html_renderer():
    js_path = path.abspath(path.join(dir_path, "dollar_auction.js"))
    with open(js_path, encoding="utf-8") as js_file:
        return js_file.read()


agents = all_agents
