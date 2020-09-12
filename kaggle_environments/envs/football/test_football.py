from kaggle_environments import make, evaluate
from kaggle_environments.errors import DeadlineExceeded
import copy
import os

env = None


def before_each(state=None, configuration=None, info={}):
    global env
    steps = [] if state == None else [state]
    env = make("football", steps=steps,
               configuration=configuration, info=info, debug=True)


def test_to_json():
    before_each()
    json = env.toJSON()
    assert json["name"] == "football"
    assert json["rewards"] == [0.0, 0.0]
    assert json["statuses"] == ["ACTIVE", "ACTIVE"]
    assert json["specification"]["reward"]["type"] == ["number", "null"]

def clear_players_raw(state):
  state = copy.deepcopy(state)
  for entry in state:
    entry.observation.players_raw = []
  return state


def test_single_agent():
    before_each(configuration={"team_1": 1, "team_2": 0, "scenario_name": "11_vs_11_stochastic"})
    x = env.reset()

    assert clear_players_raw(env.reset()) == [
        {
            "action": [],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 1,
                "players_raw": []
            }
        },
        {
            "action": [],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 0,
                "players_raw": []
            }
        }
    ]

    # Correct step from agent 0.
    assert clear_players_raw(env.step([[0],[]])) == [
        {
            "action": [0],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 1,
                "players_raw": []
            }
        },
        {
            "action": [],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 0,
                "players_raw": []
            }
        }
    ]

    # Incorrect step from agent 1 (it is forbidden to act in this scenario,
    # as 'team_2' players is set to 0).
    assert clear_players_raw(env.step([[0],[1]])) == [
        {
            "action": [0],
            "status": "DONE",
            "reward": 100,
            'info': {'debug_info': 'Oponnent forfeited. You win.'},
            "observation": {
                "controlled_players": 1,
                "players_raw": []
            }
        },
        {
            "action": [1],
            "status": "INVALID",
            "reward": None,
            'info': {'debug_info': 'Invalid number of actions provided: Expected 0, got 1.'},
            "observation": {
                "controlled_players": 0,
                "players_raw": []
            }
        }
    ]

    # Incorrect step from agent 1 (out of range).
    before_each(configuration={"team_1": 1, "team_2": 0, "scenario_name": "11_vs_11_stochastic", "save_video": True})
    x = env.reset()
    assert clear_players_raw(env.step([[100],[]])) == [
        {
            "action": [100],
            "status": "INVALID",
            "reward": None,
            'info': {'debug_info': 'Invalid action provided: 100.'},
            "observation": {
                "controlled_players": 1,
                "players_raw": []
            }
        },
        {
            "action": [],
            "status": "DONE",
            "reward": 100,
            'info': {'debug_info': 'Oponnent forfeited. You win.'},
            "observation": {
                "controlled_players": 0,
                "players_raw": []
            }
        }
    ]
    # We can render even an "empty" episode...
    env.render(mode="human", width=800, height=600)


def test_multi_agent():
    before_each(configuration={"team_1": 2, "team_2": 1, "scenario_name": "11_vs_11_stochastic"})
    x = env.reset()

    assert clear_players_raw(env.reset()) == [
        {
            "action": [],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 2,
                "players_raw": []
            }
        },
        {
            "action": [],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 1,
                "players_raw": []
            }
        }
    ]

    # Correct step from both agents.
    assert clear_players_raw(env.step([[0, 2],[4]])) == [
        {
            "action": [0, 2],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 2,
                "players_raw": []
            }
        },
        {
            "action": [4],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 1,
                "players_raw": []
            }
        }
    ]

    # Incorrect step from agent 1  - too many actions passed.
    assert clear_players_raw(env.step([[0, 1, 2], [1]])) == [
        {
            "action": [0, 1, 2],
            "status": "INVALID",
            "reward": None,
            'info': {'debug_info': 'Invalid number of actions provided: Expected 2, got 3.'},
            "observation": {
                "controlled_players": 2,
                "players_raw": []
            }
        },
        {
            "action": [1],
            "status": "DONE",
            "reward": 100,
            'info': {'debug_info': 'Oponnent forfeited. You win.'},
            "observation": {
                "controlled_players": 1,
                "players_raw": []
            }
        }
    ]

def test_deadline():
    before_each(configuration={"team_1": 1, "team_2": 0, "scenario_name": "11_vs_11_stochastic"})
    x = env.reset()

    assert clear_players_raw(env.reset()) == [
        {
            "action": [],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 1,
                "players_raw": []
            }
        },
        {
            "action": [],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 0,
                "players_raw": []
            }
        }
    ]

    # Correct step from agent 0.
    assert clear_players_raw(env.step([DeadlineExceeded(),[]])) == [
        {
            "action": None,
            "status": "TIMEOUT",
            "reward": None,
            "info": {},
            "observation": {
                "controlled_players": 1,
                "players_raw": []
            }
        },
        {
            "action": [],
            "status": "DONE",
            "reward": 100,
            "info": {'debug_info': 'Oponnent forfeited. You win.'},
            "observation": {
                "controlled_players": 0,
                "players_raw": []
            }
        }
    ]


def test_render():
    video_file = "/tmp/video.webm"
    before_each(configuration={"team_1": 1, "team_2": 1, "scenario_name": "tests.penalty", "save_video": True},
        info={"LiveVideoPath": video_file})
    env.step([[0],[0]])
    env.render(mode="human", width=800, height=600)
    output = env.step([[0],[0]])
    env.render(mode="human", width=800, height=600)
    while output[0]['status'] == 'ACTIVE':
        output = env.step([[0],[0]])
    assert output[0]['reward'] == 0
    assert output[1]['reward'] == 0
    assert output[0]['status'] == "DONE"
    assert output[1]['status'] == "DONE"
    assert os.path.isfile(video_file)
