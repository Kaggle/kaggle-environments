from kaggle_environments import make, evaluate
from kaggle_environments.errors import DeadlineExceeded
import copy

env = None


def before_each(state=None, configuration=None):
    global env
    steps = [] if state == None else [state]
    env = make("football", steps=steps,
               configuration=configuration, debug=True)


def test_to_json():
    before_each()
    json = env.toJSON()
    assert json["name"] == "football"
    assert json["rewards"] == [0.0, 0.0]
    assert json["statuses"] == ["ACTIVE", "ACTIVE"]
    assert json["specification"]["reward"]["type"] == ["number", "null"]

def clear_minimap(state):
  state = copy.deepcopy(state)
  for entry in state:
    entry.observation.minimap = []
  return state


def test_single_agent():
    before_each()
    x = env.reset()

    assert clear_minimap(env.reset()) == [
        {
            "action": [],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 1,
                "minimap": []
            }
        },
        {
            "action": [],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 0,
                "minimap": []
            }
        }
    ]

    # Correct step from agent 0.
    assert clear_minimap(env.step([[0],[]])) == [
        {
            "action": [0],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 1,
                "minimap": []
            }
        },
        {
            "action": [],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 0,
                "minimap": []
            }
        }
    ]

    # Incorrect step from agent 1 (it is forbidden to act in this scenario,
    # as 'team_2' players is set to 0).
    assert clear_minimap(env.step([[0],[1]])) == [
        {
            "action": [0],
            "status": "DONE",
            "reward": 100,
            'info': {'debug_info': 'Opponent made invalid move. You win.'},
            "observation": {
                "controlled_players": 1,
                "minimap": []
            }
        },
        {
            "action": [1],
            "status": "INVALID",
            "reward": None,
            'info': {'debug_info': 'Too many actions passed: Expected 0, got 1.'},
            "observation": {
                "controlled_players": 0,
                "minimap": []
            }
        }
    ]


def test_multi_agent():
    before_each(configuration={"team_1": 2, "team_2": 1})
    x = env.reset()

    assert clear_minimap(env.reset()) == [
        {
            "action": [],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 2,
                "minimap": []
            }
        },
        {
            "action": [],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 1,
                "minimap": []
            }
        }
    ]

    # Correct step from both agents.
    assert clear_minimap(env.step([[0, 2],[4]])) == [
        {
            "action": [0, 2],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 2,
                "minimap": []
            }
        },
        {
            "action": [4],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 1,
                "minimap": []
            }
        }
    ]

    # Incorrect step from agent 1  - too many actions passed.
    assert clear_minimap(env.step([[0, 1, 2], [1]])) == [
        {
            "action": [0, 1, 2],
            "status": "INVALID",
            "reward": None,
            'info': {'debug_info': 'Too many actions passed: Expected 2, got 3.'},
            "observation": {
                "controlled_players": 2,
                "minimap": []
            }
        },
        {
            "action": [1],
            "status": "DONE",
            "reward": 100,
            'info': {'debug_info': 'Opponent made invalid move. You win.'},
            "observation": {
                "controlled_players": 1,
                "minimap": []
            }
        }
    ]

def test_deadline():
    before_each()
    x = env.reset()

    assert clear_minimap(env.reset()) == [
        {
            "action": [],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 1,
                "minimap": []
            }
        },
        {
            "action": [],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 0,
                "minimap": []
            }
        }
    ]

    # Correct step from agent 0.
    assert clear_minimap(env.step([DeadlineExceeded(),[]])) == [
        {
            "action": None,
            "status": "TIMEOUT",
            "reward": None,
            "info": {},
            "observation": {
                "controlled_players": 1,
                "minimap": []
            }
        },
        {
            "action": [],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
            "observation": {
                "controlled_players": 0,
                "minimap": []
            }
        }
    ]


