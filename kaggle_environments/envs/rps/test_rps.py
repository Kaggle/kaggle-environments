from kaggle_environments import make
from .agents import random_agent, rock, paper

def negative_move_agent(observation, configuration):
    return -1


def too_big_weapon_agent(observation, configuration):
    return 1000000


def non_integer_agent(observation, configuration):
    return 0.3


def none_agent(observation, configuration):
    return None


def test_rps_completes():
    env = make("rps", configuration={"episodeSteps": 10})
    env.run([random_agent, random_agent])
    json = env.toJSON()
    assert json["name"] == "rps"
    assert json["statuses"] == ["DONE", "DONE"]


def test_tie():
    env = make("rps", configuration={"episodeSteps": 10})
    env.run([rock, rock])
    json = env.toJSON()
    assert json["rewards"] == [0.5, 0.5]
    assert json["statuses"] == ["DONE", "DONE"]


def test_win():
    env = make("rps", configuration={"episodeSteps": 10})
    env.run([paper, rock])
    json = env.toJSON()
    print(json)
    assert json["rewards"] == [1, 0]
    assert json["statuses"] == ["DONE", "DONE"]


def test_loss():
    env = make("rps", configuration={"episodeSteps": 10})
    env.run([rock, paper])
    json = env.toJSON()
    assert json["rewards"] == [0, 1]
    assert json["statuses"] == ["DONE", "DONE"]


def test_negative_move():
    env = make("rps", configuration={"episodeSteps": 10})
    env.run([negative_move_agent, rock])
    json = env.toJSON()
    assert json["rewards"] == [None, 1]
    assert json["statuses"] == ['INVALID', 'DONE']


def test_non_integer_move():
    env = make("rps", configuration={"episodeSteps": 10})
    env.run([non_integer_agent, rock])
    json = env.toJSON()
    assert json["rewards"] == [None, 1]
    assert json["statuses"] == ['INVALID', 'DONE']


def test_too_big_move():
    env = make("rps", configuration={"episodeSteps": 10})
    env.run([paper, too_big_weapon_agent])
    json = env.toJSON()
    assert json["rewards"] == [1, None]
    assert json["statuses"] == ['DONE', 'INVALID']
