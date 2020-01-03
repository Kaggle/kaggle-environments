# Copyright 2020 Kaggle Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from kaggle_environments import make, evaluate

env = None


def custom1(obs):
    step = sum(1 for mark in obs.board if mark == obs.mark)
    return [0, 2, 4, 6, 8][step]


def custom2(obs):
    step = sum(1 for mark in obs.board if mark == obs.mark)
    return [1, 3, 5, 7][step]


def custom3():
    time.sleep(5)
    return 3


def custom4():
    raise Exception("Foo Bar")


def custom5():
    return -1


def before_each(state=None):
    global env
    steps = [] if state == None else [state]
    env = make("tictactoe", steps=steps, debug=True)


def test_to_json():
    before_each()
    json = env.toJSON()
    assert json["name"] == "tictactoe"
    assert json["rewards"] == [0.5, 0.5]
    assert json["statuses"] == ["ACTIVE", "INACTIVE"]
    assert json["specification"]["reward"]["type"] == ["number", "null"]


def test_can_reset():
    before_each()
    assert env.reset() == [
        {
            "action": 0,
            "status": "ACTIVE",
            "info": {},
            "observation": {"mark": 1, "board": [0, 0, 0, 0, 0, 0, 0, 0, 0]},
            "reward": 0.5,
        },
        {
            "action": 0,
            "status": "INACTIVE",
            "info": {},
            "observation": {"mark": 2, "board": [0, 0, 0, 0, 0, 0, 0, 0, 0]},
            "reward": 0.5,
        },
    ]


def test_can_place_valid_mark():
    before_each()

    assert env.step([4, None]) == [
        {
            "action": 4,
            "status": "INACTIVE",
            "info": {},
            "observation": {"mark": 1, "board": [0, 0, 0, 0, 1, 0, 0, 0, 0]},
            "reward": 0.5,
        },
        {
            "action": 0,  # None caused the default action to be applied.
            "status": "ACTIVE",
            "info": {},
            "observation": {"mark": 2, "board": [0, 0, 0, 0, 1, 0, 0, 0, 0]},
            "reward": 0.5,
        },
    ]


def test_can_place_invalid_mark():
    before_each()

    env.step([4, None])

    assert env.step([None, 4]) == [
        {
            "action": 0,
            "status": "DONE",
            "info": {},
            "observation": {"mark": 1, "board": [0, 0, 0, 0, 1, 0, 0, 0, 0]},
            "reward": 0.5,
        },
        {
            "action": 4,
            "status": "INVALID",
            "info": {},
            "observation": {"mark": 2, "board": [0, 0, 0, 0, 1, 0, 0, 0, 0]},
            "reward": None,
        },
    ]


def test_can_place_winning_mark():
    obs = {"observation": {"board": [2, 1, 0, 1, 1, 0, 2, 0, 2]}}
    before_each([obs, obs])

    assert env.step([7, None]) == [
        {
            "action": 7,
            "status": "DONE",
            "info": {},
            "observation": {"mark": 1, "board": [2, 1, 0, 1, 1, 0, 2, 1, 2]},
            "reward": 1,
        },
        {
            "action": 0,
            "status": "DONE",
            "info": {},
            "observation": {"mark": 2, "board": [2, 1, 0, 1, 1, 0, 2, 1, 2]},
            "reward": 0,
        },
    ]


def test_can_render():
    obs = {"observation": {"board": [0, 1, 0, 2, 1, 2, 0, 0, 2]}}
    before_each([obs, obs])
    out = "   | X |   \n---+---+---\n O | X | O \n---+---+---\n   |   | O "
    assert env.render(mode="ansi") == out


def test_can_step_through_agents():
    before_each()
    while not env.done:
        action1 = env.agents.random(env.state[0].observation)
        action2 = env.agents.reaction(env.state[1].observation)
        env.step([action1, action2])
    assert env.state[0].reward + env.state[1].reward == 1


def test_can_run_agents():
    before_each()
    state = env.run(["random", "reaction"])[-1]
    assert state[0].reward + state[1].reward == 1


def test_can_evaluate():
    rewards = evaluate("tictactoe", ["random", "reaction"], num_episodes=2)
    assert (rewards[0][0] + rewards[0][1] ==
            1) and rewards[1][0] + rewards[1][1] == 1


def test_can_run_custom_agents():
    state = env.run([custom1, custom2])[-1]
    assert state == [
        {
            "action": 6,
            "reward": 1,
            "info": {},
            "observation": {"board": [1, 2, 1, 2, 1, 2, 1, 0, 0], "mark": 1},
            "status": "DONE",
        },
        {
            "action": 0,
            "reward": 0,
            "info": {},
            "observation": {"board": [1, 2, 1, 2, 1, 2, 1, 0, 0], "mark": 2},
            "status": "DONE",
        },
    ]


def test_agents_can_timeout():
    state = env.run([custom1, custom3])[-1]
    assert state == [
        {
            "action": 0,
            "reward": 0.5,
            "info": {},
            "observation": {"board": [1, 0, 0, 0, 0, 0, 0, 0, 0], "mark": 1},
            "status": "DONE",
        },
        {
            "action": None,
            "reward": None,
            "info": {},
            "observation": {"board": [1, 0, 0, 0, 0, 0, 0, 0, 0], "mark": 2},
            "status": "TIMEOUT",
        },
    ]


def test_agents_can_error():
    state = env.run([custom1, custom4])[-1]
    assert state == [
        {
            "action": 0,
            "reward": 0.5,
            "info": {},
            "observation": {"board": [1, 0, 0, 0, 0, 0, 0, 0, 0], "mark": 1},
            "status": "DONE",
        },
        {
            "action": None,
            "reward": None,
            "info": {},
            "observation": {"board": [1, 0, 0, 0, 0, 0, 0, 0, 0], "mark": 2},
            "status": "ERROR",
        },
    ]


def test_agents_can_have_invalid_actions():
    state = env.run([custom1, custom5])[-1]
    assert state == [
        {
            "action": 0,
            "reward": 0.5,
            "info": {},
            "observation": {"board": [1, 0, 0, 0, 0, 0, 0, 0, 0], "mark": 1},
            "status": "DONE",
        },
        {
            "action": -1,
            "reward": None,
            "info": {},
            "observation": {"board": [1, 0, 0, 0, 0, 0, 0, 0, 0], "mark": 2},
            "status": "INVALID",
        },
    ]
