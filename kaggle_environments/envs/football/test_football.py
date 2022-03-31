from kaggle_environments import make, evaluate
from kaggle_environments.envs.football import helpers
from helpers import Action
from kaggle_environments.errors import DeadlineExceeded
import copy
import os

env = None

# Temporarily disable these tests until we can fix the gfootball env
"""
def before_each(state=None, configuration=None, info=None):
    if info is None:
        info = {}
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


@helpers.human_readable_agent
def human_readable_agent(obs):
    if Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint
    controlled_player_pos = obs['left_team'][obs['active']]
    if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:
        if controlled_player_pos[0] > 0.5:
            return Action.Shot
        return Action.Right
    else:
        return Action.Slide


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
                "remainingOverageTime": 60,
                "controlled_players": 1,
                "players_raw": [],
                "step": 0
            }
        },
        {
            "action": [],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "remainingOverageTime": 60,
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
                "remainingOverageTime": 60,
                "controlled_players": 1,
                "players_raw": [],
                "step": 1
            }
        },
        {
            "action": [],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "remainingOverageTime": 60,
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
            'info': {'debug_info': 'Opponent forfeited. You win.'},
            "observation": {
                "remainingOverageTime": 60,
                "controlled_players": 1,
                "players_raw": [],
                "step": 2
            }
        },
        {
            "action": [1],
            "status": "INVALID",
            "reward": None,
            'info': {'debug_info': 'Invalid number of actions provided: Expected 0, got 1.'},
            "observation": {
                "remainingOverageTime": 60,
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
            'info': {'debug_info': 'Invalid action provided: [100].'},
            "observation": {
                "remainingOverageTime": 60,
                "controlled_players": 1,
                "players_raw": [],
                "step": 1
            }
        },
        {
            "action": [],
            "status": "DONE",
            "reward": 100,
            'info': {'debug_info': 'Opponent forfeited. You win.'},
            "observation": {
                "remainingOverageTime": 60,
                "controlled_players": 0,
                "players_raw": []
            }
        }
    ]
    # We can render even an "empty" episode...
    env.render(mode="human", width=800, height=600)

    # Incorrect step from agent 1 (not a list).
    before_each(configuration={"team_1": 1, "team_2": 0, "scenario_name": "11_vs_11_stochastic"})
    x = env.reset()
    assert clear_players_raw(env.step([100, []])) == [
        {
            "action": [],
            "status": "INVALID",
            "reward": None,
            'info': {'debug_info': 'Invalid number of actions provided: Expected 1, got 0.'},
            "observation": {
                "remainingOverageTime": 60,
                "controlled_players": 1,
                "players_raw": [],
                "step": 1
            }
        },
        {
            "action": [],
            "status": "DONE",
            "reward": 100,
            'info': {'debug_info': 'Opponent forfeited. You win.'},
            "observation": {
                "remainingOverageTime": 60,
                "controlled_players": 0,
                "players_raw": []
            }
        }
    ]


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
                "remainingOverageTime": 60,
                "controlled_players": 2,
                "players_raw": [],
                "step": 0
            }
        },
        {
            "action": [],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "remainingOverageTime": 60,
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
                "remainingOverageTime": 60,
                "controlled_players": 2,
                "players_raw": [],
                "step": 1
            }
        },
        {
            "action": [4],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "remainingOverageTime": 60,
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
                "remainingOverageTime": 60,
                "controlled_players": 2,
                "players_raw": [],
                "step": 2
            }
        },
        {
            "action": [1],
            "status": "DONE",
            "reward": 100,
            'info': {'debug_info': 'Opponent forfeited. You win.'},
            "observation": {
                "remainingOverageTime": 60,
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
                "remainingOverageTime": 60,
                "controlled_players": 1,
                "players_raw": [],
                "step": 0
            }
        },
        {
            "action": [],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "remainingOverageTime": 60,
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
                "remainingOverageTime": 60,
                "controlled_players": 1,
                "players_raw": [],
                "step": 1
            }
        },
        {
            "action": [],
            "status": "DONE",
            "reward": 100,
            "info": {'debug_info': 'Opponent forfeited. You win.'},
            "observation": {
                "remainingOverageTime": 60,
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


def test_human_readable_agent():
    before_each(configuration={"team_1": 1, "team_2": 1, "scenario_name": "tests.penalty"})
    action = [0]
    for _ in range(10):
      obs = env.step([action, [0]])
      action  = human_readable_agent(obs[0]['observation'])


def test_score():
    before_each(configuration={"team_1": 1, "team_2": 1, "scenario_name": "tests.goal_test", "save_video": True, "episode_number": 0})
    res = env.run(["run_right", "run_right"])
    assert res[-1][0]['reward'] == -1
    assert res[-1][1]['reward'] == 1
"""