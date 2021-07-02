from kaggle_environments.envs.lux_ai_2021.test_agents.python.random_agent import random_agent
import random
from .test_agents.js_simple.main import js_agent as js_simple_agent
from .test_agents.python.random_agent import random_agent
from .test_agents.python.simple_agent import agent as simple_agent
agents = {
    "random_agent": random_agent,
    "simple_agent": simple_agent,
    "js_simple_agent": js_simple_agent,
}