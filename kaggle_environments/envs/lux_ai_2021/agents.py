from kaggle_environments.envs.lux_ai_2021.test_agents.python.random_agent import random_agent
import random
from .test_agents.js_agent_test_1.main import js_agent as js_agent_random
from .test_agents.js_agent_test_2.main import js_agent as js_agent_slow_expand
from .test_agents.python.random_agent import random_agent
from .test_agents.python.organic_agent import organic_agent
from .test_agents.python.simple_agent import simple_agent
agents = {
    "random_agent": random_agent,
    "simple_agent": simple_agent,
    "organic_agent": organic_agent,
    "js_agent_random": js_agent_random,
    "js_agent_slow_expand": js_agent_slow_expand
}