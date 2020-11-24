from kaggle_environments import make
from .agents import rock, paper, agents


def negative_move_agent(observation, configuration):
    return -1


def too_big_sign_agent(observation, configuration):
    return 1000000


def non_integer_agent(observation, configuration):
    return 0.3


def none_agent(observation, configuration):
    return None


def test_rps_completes():
    env = make("rps", configuration={"episodeSteps": 10, "tieRewardThreshold": 1})
    env.run([rock, rock])
    json = env.toJSON()
    assert json["name"] == "rps"
    assert json["statuses"] == ["DONE", "DONE"]


def test_all_agents():
    env = make("rps", configuration={"episodeSteps": 3, "tieRewardThreshold": 1})
    for agent in agents:
        env.run([agent, agent])
        json = env.toJSON()
        assert json["statuses"] == ["DONE", "DONE"]


def test_tie():
    env = make("rps", configuration={"episodeSteps": 3, "tieRewardThreshold": 1})
    env.run([rock, rock])
    assert env.render(mode='ansi') == "Round 1: Rock vs Rock, Score: 0 to 0\nRound 2: Rock vs Rock, Score: 0 to 0\nGame ended on round 2, final score: 0 to 0\n"
    json = env.toJSON()
    assert json["rewards"] == [0, 0]
    assert json["statuses"] == ["DONE", "DONE"]


def test_threshold_tie():
    env = make("rps", configuration={"episodeSteps": 3, "tieRewardThreshold": 4})
    env.run([rock, paper])
    assert env.render(mode='ansi') == "Round 1: Rock vs Paper, Score: -1.0 to 1.0\nRound 2: Rock vs Paper, Score: 0 to 0\nGame ended on round 2, final score: 0 to 0\n"
    json = env.toJSON()
    assert json["rewards"] == [0, 0]
    assert json["statuses"] == ["DONE", "DONE"]


def test_win():
    env = make("rps", configuration={"episodeSteps": 2, "tieRewardThreshold": 1})
    env.run([paper, rock])
    json = env.toJSON()
    print(json)
    assert json["rewards"] == [1, -1]
    assert json["statuses"] == ["DONE", "DONE"]


def test_loss():
    env = make("rps", configuration={"episodeSteps": 2, "tieRewardThreshold": 1})
    env.run([rock, paper])
    json = env.toJSON()
    assert json["rewards"] == [-1, 1]
    assert json["statuses"] == ["DONE", "DONE"]


def test_negative_move():
    env = make("rps", configuration={"episodeSteps": 10, "tieRewardThreshold": 1})
    env.run([negative_move_agent, rock])
    json = env.toJSON()
    assert json["rewards"] == [None, 1]
    assert json["statuses"] == ['INVALID', 'DONE']


def test_non_integer_move():
    env = make("rps", configuration={"episodeSteps": 10, "tieRewardThreshold": 1})
    env.run([non_integer_agent, rock])
    json = env.toJSON()
    assert json["rewards"] == [None, 1]
    assert json["statuses"] == ['INVALID', 'DONE']


def test_too_big_move():
    env = make("rps", configuration={"episodeSteps": 10, "tieRewardThreshold": 1})
    env.run([paper, too_big_sign_agent])
    json = env.toJSON()
    assert json["rewards"] == [1, None]
    assert json["statuses"] == ['DONE', 'INVALID']

def test_agent_reward():
    env = make("rps", configuration={"episodeSteps": 2, "tieRewardThreshold": 1})
    env.run([paper, rock])
    json = env.toJSON()
    last_step = json["steps"][-1]
    assert last_step[0]["observation"]["step"] == last_step[0]["observation"]["reward"]
    assert last_step[0]["observation"]["reward"] == -last_step[1]["observation"]["reward"]
    assert json["statuses"] == ["DONE", "DONE"]
