import sys
from kaggle_environments import make
from .agents import random_agent

def test_lux_completes():
    env = make("lux_ai_s2", debug=True)
    env.run([random_agent, random_agent])
    json = env.toJSON()
    assert json["name"] == "lux_ai_s2"
    assert json["statuses"] == ["DONE", "DONE"]