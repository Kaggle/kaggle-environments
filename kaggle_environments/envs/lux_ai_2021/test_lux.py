from kaggle_environments import make
from .agents import random_agent, js_agent_random, js_agent_slow_expand

def test_lux_completes():
    env = make("lux_ai_2021", configuration={})
    env.run([random_agent, random_agent])
    json = env.toJSON()
    assert json["name"] == "lux_ai_2021"
    assert json["statuses"] == ["DONE", "DONE"]

def test_js_agents():
    env = make("lux_ai_2021", configuration={})
    env.run([js_agent_slow_expand, js_agent_random])
    json = env.toJSON()
    assert json["name"] == "lux_ai_2021"
    assert json["statuses"] == ["DONE", "DONE"]