import random
from .test_agents.js_agent_test_1.agent import js_agent as js_agent_random
from .test_agents.js_agent_test_2.agent import js_agent as js_agent_slow_expand
from .test_agents.python.agents import random_agent, collector_agent

agents = {
    "random_agent": random_agent,
    "collector_agent": collector_agent,
    "js_agent_random": js_agent_random,
    "js_agent_slow_expand": js_agent_slow_expand
}