import sys
from kaggle_environments import make
from .agents import random_agent

def test_lux_completes():
    env = make("lux_ai_2022", configuration={})
    env.run([random_agent, random_agent])
    json = env.toJSON()
    assert json["name"] == "lux_ai_2022"
    assert json["statuses"] == ["DONE", "DONE"]