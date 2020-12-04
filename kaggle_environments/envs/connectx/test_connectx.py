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

from kaggle_environments import make, evaluate

env = None


def before_each(state=None, configuration=None):
    global env
    steps = [] if state == None else [state]
    env = make("connectx", steps=steps,
               configuration=configuration, debug=True)


def test_has_correct_timeouts():
    before_each()
    assert env.configuration.actTimeout == 2


def test_can_train_first():
    before_each()
    trainer = env.train([None, 'random'])
    obs = trainer.reset()
    assert "board" in obs
    obs, _, _, _ = trainer.step(0)
    assert "board" in obs


def test_can_train_second():
    before_each()
    trainer = env.train(['random', None])
    obs = trainer.reset()
    assert "board" in obs
    obs, _, _, _ = trainer.step(0)
    assert "board" in obs


def test_to_json():
    before_each()
    json = env.toJSON()
    assert json["name"] == "connectx"
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
            "observation": {"remainingOverageTime": 60, "board": [0] * 42, "mark": 1, "step": 0},
            "reward": 0,
        },
        {
            "action": 0,
            "status": "INACTIVE",
            "info": {},
            "observation": {"remainingOverageTime": 60, "mark": 2},
            "reward": 0,
        },
    ]


def test_can_mark():
    before_each(configuration={"rows": 4, "columns": 5, "inarow": 3})
    assert env.step([2, None]) == [
        {
            "action": 2,
            "status": "INACTIVE",
            "info": {},
            "observation": {
                "remainingOverageTime": 60,
                "board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                "mark": 1,
                "step": 1,
            },
            "reward": 0,
        },
        {
            "action": 0,
            "status": "ACTIVE",
            "info": {},
            "observation": {"remainingOverageTime": 60, "mark": 2},
            "reward": 0,
        },
    ]


def test_can_mark_out_of_bounds():
    before_each(configuration={"rows": 4, "columns": 5, "inarow": 3})
    assert env.step([10, None]) == [
        {
            "action": 10,
            "status": "INVALID",
            "info": {},
            "observation": {
                "remainingOverageTime": 60,
                "board": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "mark": 1,
                "step": 1,
            },
            "reward": None,
        },
        {
            "action": 0,
            "status": "DONE",
            "info": {},
            "observation": {"remainingOverageTime": 60, "mark": 2},
            "reward": 0,
        },
    ]


def test_can_mark_a_full_column():
    board = [1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0]
    before_each(
        configuration={"rows": 4, "columns": 5, "inarow": 3},
        state=[{"observation": {"board": board}},
               {"observation": {}}],
    )
    assert env.step([1, None]) == [
        {
            "action": 1,
            "status": "INVALID",
            "info": {},
            "observation": {"remainingOverageTime": 60, "board": board, "mark": 1, "step": 1},
            "reward": None,
        },
        {
            "action": 0,
            "status": "DONE",
            "info": {},
            "observation": {"remainingOverageTime": 60, "mark": 2},
            "reward": 0,
        },
    ]


def test_can_win():
    board = [0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0]
    board_post_move = board[:]
    board_post_move[0] = 1
    before_each(
        configuration={"rows": 4, "columns": 5, "inarow": 3},
        state=[{"observation": {"board": board}},
               {"observation": {}}],
    )
    assert env.step([0, None]) == [
        {
            "action": 0,
            "status": "DONE",
            "info": {},
            "observation": {"remainingOverageTime": 60, "board": board_post_move, "mark": 1, "step": 1},
            "reward": 1,
        },
        {
            "action": 0,
            "status": "DONE",
            "info": {},
            "observation": {"remainingOverageTime": 60, "mark": 2},
            "reward": -1,
        },
    ]


def test_can_tie():
    board = [0, 0, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
    board_post_move = board[:]
    board_post_move[0] = 1
    board_post_move[1] = 2
    before_each(
        configuration={"rows": 4, "columns": 5, "inarow": 3},
        state=[{"observation": {"remainingOverageTime": 60, "board": board}}, {"observation": {}}],
    )
    env.step([0, None])
    assert env.step([None, 1]) == [
        {
            "action": 0,
            "status": "DONE",
            "info": {},
            "observation": {"remainingOverageTime": 60, "board": board_post_move, "mark": 1, "step": 2},
            "reward": 0,
        },
        {
            "action": 1,
            "status": "DONE",
            "info": {},
            "observation": {"remainingOverageTime": 60, "mark": 2},
            "reward": 0,
        },
    ]


def test_can_render():
    board = [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 2, 1, 0, 1, 2, 1, 2, 1]
    before_each(
        configuration={"rows": 4, "columns": 5, "inarow": 3},
        state=[{"observation": {"remainingOverageTime": 60, "board": board}}, {"observation": {}}],
    )
    assert env.render(mode="ansi").strip() == """
+---+---+---+---+---+
| 0 | 0 | 0 | 0 | 0 |
+---+---+---+---+---+
| 0 | 0 | 2 | 0 | 0 |
+---+---+---+---+---+
| 0 | 1 | 2 | 1 | 0 |
+---+---+---+---+---+
| 1 | 2 | 1 | 2 | 1 |
+---+---+---+---+---+
""".strip()


def test_can_run_agents():
    def custom1():
        return 1

    def custom2():
        return 2
    before_each(
        configuration={"rows": 4, "columns": 5, "inarow": 3},
    )
    board = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0]
    assert env.run([custom1, custom2])[-1] == [
        {
            "action": 1,
            "status": "DONE",
            "info": {},
            "observation": {"remainingOverageTime": 60, "board": board, "mark": 1, "step": 5},
            "reward": 1,
        },
        {
            "action": 0,
            "status": "DONE",
            "info": {},
            "observation": {"remainingOverageTime": 60, "mark": 2},
            "reward": -1,
        },
    ]


def test_can_evaluate():
    rewards = evaluate("connectx", ["random", "random"], num_episodes=2)
    assert (rewards[0][0] + rewards[0][1] ==
            0) and rewards[1][0] + rewards[1][1] == 0
