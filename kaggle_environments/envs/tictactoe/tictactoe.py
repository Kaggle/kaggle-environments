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

import json
from os import path
from random import choice

checks = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
]

EMPTY = 0


def random_agent(obs):
    return choice([c for c in range(len(obs.board)) if obs.board[c] == EMPTY])


def reaction_agent(obs):
    # Connect 3 in a row to win.
    for check in checks:
        left = list(filter(lambda c: obs.board[c] != obs.mark, check))
        if len(left) == 1 and obs.board[left[0]] == EMPTY:
            return left[0]

    # Block 3 in a row to prevent loss.
    opponent = 2 if obs.mark == 1 else 1
    for check in checks:
        left = list(filter(lambda c: obs.board[c] != opponent, check))
        if len(left) == 1 and obs.board[left[0]] == EMPTY:
            return left[0]

    # No 3-in-a-rows, return random unmarked.
    return choice(list(filter(lambda m: m[1] == EMPTY, enumerate(obs.board))))[0]


agents = {"random": random_agent, "reaction": reaction_agent}


def interpreter(state, env):
    if env.done:
        return state

    # Isolate the active and inactive agents.
    active = state[0] if state[0].status == "ACTIVE" else state[1]
    inactive = state[0] if state[0].status == "INACTIVE" else state[1]
    if active.status != "ACTIVE" or inactive.status != "INACTIVE":
        active.status = "DONE" if active.status == "ACTIVE" else active.status
        inactive.status = "DONE" if inactive.status == "INACTIVE" else inactive.status
        return state

    # The board is shared, only update the first state.
    board = state[0].observation.board

    # Illegal move by the active agent.
    if board[active.action] != EMPTY:
        active.status = f"Invalid move: {active.action}"
        inactive.status = "DONE"
        return state

    # Mark the position.
    board[active.action] = active.observation.mark

    # Check for a win.
    if any(all(board[p] == active.observation.mark for p in c) for c in checks):
        active.reward = 1
        active.status = "DONE"
        inactive.reward = -1
        inactive.status = "DONE"
        return state

    # Check for a tie.
    if all(mark != EMPTY for mark in board):
        active.status = "DONE"
        inactive.status = "DONE"
        return state

    # Swap active and inactive agents to switch turns.
    active.status = "INACTIVE"
    inactive.status = "ACTIVE"

    return state


def renderer(state, env):
    row_bar = "\n---+---+---\n"
    marks = [" ", "X", "O"]

    def print_pos(pos):
        str = ""
        if pos % 3 == 0 and pos > 0:
            str += row_bar
        if pos % 3 != 0:
            str += "|"
        return str + f" {marks[state[0].observation.board[pos]]} "

    return "".join(print_pos(p) for p in range(9))


jsonpath = path.abspath(path.join(path.dirname(__file__), "tictactoe.json"))
with open(jsonpath) as f:
    specification = json.load(f)


def html_renderer():
    jspath = path.abspath(path.join(path.dirname(__file__), "tictactoe.js"))
    with open(jspath) as f:
        return f.read()
