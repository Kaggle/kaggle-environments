from pyxis_portfolio_challenge.agents.knapsack import KnapsackAgent
from pyxis_portfolio_challenge.agents.multi_agent_do_nothing import (
    MultiAgentDoNothingAgent as MultiAgentDoNothingAgent,
)
from pyxis_portfolio_challenge.agents.multi_agent_knapsack import (
    MultiAgentKnapsackAgent,
)
from pyxis_portfolio_challenge.agents.multi_agent_random import (
    MultiAgentRandomAgent as MultiAgentRandomAgent,
)

AGENTS_LIST = [
    {"name": "Knapsack", "cost": 500_000.0},
]
AGENTS = {agent["name"]: agent for agent in AGENTS_LIST}


def get_agent(name: str, **kwargs) -> object:
    """Retrieve an investment agent instance by passing its name."""
    if name == "Knapsack":
        agent = KnapsackAgent()
    elif name == "MultiAgentKnapsack":
        if "agent_name" not in kwargs:
            raise ValueError("agent_name must be provided for MultiAgentKnapsack.")
        agent = MultiAgentKnapsackAgent(
            agent_name=kwargs["agent_name"],
            capacity=kwargs.get("capacity", 12),
            enable_bd_bidding=kwargs.get("enable_bd_bidding", True),
        )
    elif name == "MultiAgentRandom":
        if "agent_name" not in kwargs:
            raise ValueError("agent_name must be provided for MultiAgentRandom.")
        agent = MultiAgentRandomAgent(agent_name=kwargs["agent_name"])
    elif name == "MultiAgentDoNothing":
        if "agent_name" not in kwargs:
            raise ValueError(
                "agent_name must be provided for MultiAgentDoNothing."
            )
        agent = MultiAgentDoNothingAgent(agent_name=kwargs["agent_name"])
    else:
        raise ValueError(
            f"Unknown agent name: {name}. Available agents: {list(AGENTS.keys())}"
        )
    return agent
