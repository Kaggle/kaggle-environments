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
from kaggle_environments import make, evaluate, utils

env = None


def custom1(obs):
    step = sum(1 for mark in obs.board if mark == obs.mark)
    return [0, 2, 4, 6, 8][step]


def custom2(obs):
    step = sum(1 for mark in obs.board if mark == obs.mark)
    return [1, 3, 5, 7][step]


def custom3(obs):
    step = sum(1 for mark in obs.board if mark == obs.mark)
    time.sleep(4)
    return [1, 3, 5, 7][step]


def custom4():
    raise Exception("Foo Bar")


def custom5():
    return -1


def custom6(obs):
    step = sum(1 for mark in obs.board if mark == obs.mark)
    time.sleep(2)
    return [1, 3, 5, 7][step]


def before_each(state=None):
    global env
    steps = [] if state == None else [state]
    env = make("tictactoe", steps=steps, debug=True)


def test_to_json():
    before_each()
    json = env.toJSON()
    assert json["name"] == "tictactoe"
    assert json["rewards"] == [0, 0]
    assert json["statuses"] == ["ACTIVE", "INACTIVE"]
    assert json["specification"]["reward"]["type"] == ["number", "null"]


def test_can_reset():
    before_each()
    assert env.reset() == [
        {
            "action": 0,
            "status": "ACTIVE",
            "info": {},
            "observation": {"remainingOverageTime": 2, "mark": 1, "board": [0, 0, 0, 0, 0, 0, 0, 0, 0], "step": 0},
            "reward": 0,
        },
        {
            "action": 0,
            "status": "INACTIVE",
            "info": {},
            "observation": {"remainingOverageTime": 2, "mark": 2},
            "reward": 0,
        },
    ]


def test_can_place_valid_mark():
    before_each()

    assert env.step([4, None]) == [
        {
            "action": 4,
            "status": "INACTIVE",
            "info": {},
            "observation": {"remainingOverageTime": 2, "mark": 1, "board": [0, 0, 0, 0, 1, 0, 0, 0, 0], "step": 1},
            "reward": 0,
        },
        {
            "action": 0,  # None caused the default action to be applied.
            "status": "ACTIVE",
            "info": {},
            "observation": {"remainingOverageTime": 2, "mark": 2},
            "reward": 0,
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
            "observation": {"remainingOverageTime": 2, "mark": 1, "board": [0, 0, 0, 0, 1, 0, 0, 0, 0], "step": 2},
            "reward": 0,
        },
        {
            "action": 4,
            "status": "INVALID",
            "info": {},
            "observation": {"remainingOverageTime": 2, "mark": 2},
            "reward": None,
        },
    ]


def test_can_place_winning_mark():
    state1 = {"observation": {"board": [2, 1, 0, 1, 1, 0, 2, 0, 2]}}
    state2 = {}
    before_each([state1, state2])

    assert env.step([7, None]) == [
        {
            "action": 7,
            "status": "DONE",
            "info": {},
            "observation": {"remainingOverageTime": 2, "mark": 1, "board": [2, 1, 0, 1, 1, 0, 2, 1, 2], "step": 1},
            "reward": 1,
        },
        {
            "action": 0,
            "status": "DONE",
            "info": {},
            "observation": {"remainingOverageTime": 2, "mark": 2},
            "reward": -1,
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
        action2 = env.agents.reaction(
            utils.structify({"board": env.state[0].observation.board, "mark": 2}))
        env.step([action1, action2])
    assert env.state[0].reward + env.state[1].reward == 0


def test_can_run_agents():
    before_each()
    state = env.run(["random", "reaction"])[-1]
    assert state[0].reward + state[1].reward == 0


def test_can_evaluate():
    rewards = evaluate("tictactoe", ["random", "reaction"], num_episodes=2)
    assert (rewards[0][0] + rewards[0][1] ==
            0) and rewards[1][0] + rewards[1][1] == 0


def test_can_run_custom_agents():
    before_each()
    state = env.run([custom1, custom2])[-1]
    assert state == [
        {
            "action": 6,
            "reward": 1,
            "info": {},
            "observation": {"remainingOverageTime": 2, "board": [1, 2, 1, 2, 1, 2, 1, 0, 0], "mark": 1, "step": 7},
            "status": "DONE",
        },
        {
            "action": 0,
            "reward": -1,
            "info": {},
            "observation": {"remainingOverageTime": 2, "mark": 2},
            "status": "DONE",
        },
    ]


def test_agents_can_timeout_on_init():
    env = make("tictactoe", debug=True)
    state = env.run([custom1, custom3])[-1]
    assert state[1]["status"] == "TIMEOUT"
    assert state[1]["observation"]["remainingOverageTime"] < 0


def test_agents_can_timeout_on_act():
    env = make("tictactoe", debug=True)
    state = env.run([custom1, custom6])[-1]
    print(state)
    assert state[1]["status"] == "TIMEOUT"
    assert state[1]["observation"]["remainingOverageTime"] < 0


def test_run_timeout():
    env = make("tictactoe", debug=True, configuration={"actTimeout": 10, "runTimeout": 1})
    state = env.run([custom1, custom3])[-1]
    assert state == [
        {
            "action": 0,
            "reward": 0,
            "info": {},
            "observation": {"remainingOverageTime": 2, "board": [1, 2, 0, 0, 0, 0, 0, 0, 0], "mark": 1, "step": 2},
            "status": "ACTIVE",
        },
        {
            "action": 1,
            "reward": 0,
            "info": {},
            "observation": {"remainingOverageTime": 2, "mark": 2},
            "status": "INACTIVE",
        },
    ]


def test_agents_can_error():
    before_each()
    state = env.run([custom1, custom4])[-1]
    assert state == [
        {
            "action": 0,
            "reward": 0,
            "info": {},
            "observation": {"remainingOverageTime": 2, "board": [1, 0, 0, 0, 0, 0, 0, 0, 0], "mark": 1, "step": 2},
            "status": "DONE",
        },
        {
            "action": None,
            "reward": None,
            "info": {},
            "observation": {"remainingOverageTime": 2, "mark": 2},
            "status": "ERROR",
        },
    ]


def test_agents_can_have_invalid_actions():
    before_each()
    state = env.run([custom1, custom5])[-1]
    assert state == [
        {
            "action": 0,
            "reward": 0,
            "info": {},
            "observation": {"remainingOverageTime": 2, "board": [1, 0, 0, 0, 0, 0, 0, 0, 0], "mark": 1, "step": 2},
            "status": "DONE",
        },
        {
            "action": None,
            "reward": None,
            "info": {},
            "observation": {"remainingOverageTime": 2, "mark": 2},
            "status": "INVALID",
        },
    ]
