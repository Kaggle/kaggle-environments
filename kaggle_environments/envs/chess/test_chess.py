from kaggle_environments import make
from .agents import random_agent

env = None

def test_chess_completes():
    env = make("chess", debug=True)
    env.run([random_agent, random_agent])
    json = env.toJSON()
    assert json["name"] == "chess"
    assert json["statuses"] == ["DONE", "DONE"]