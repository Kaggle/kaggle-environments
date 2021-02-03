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
import kaggle_environments.helpers
from enum import auto, Enum
from kaggle_environments.helpers import histogram, with_print
from os import path
from random import choice, sample
from typing import *


class Observation(kaggle_environments.helpers.Observation):
    @property
    def geese(self) -> List[List[int]]:
        return self["geese"]

    @property
    def food(self) -> List[int]:
        return self["food"]

    @property
    def index(self) -> int:
        return self["index"]


class Configuration(kaggle_environments.helpers.Configuration):
    @property
    def columns(self) -> int:
        return self["columns"]

    @property
    def rows(self) -> int:
        return self["rows"]

    @property
    def hunger_rate(self) -> int:
        return self["hunger_rate"]

    @property
    def min_food(self) -> int:
        return self["min_food"]

    @property
    def max_length(self) -> int:
        return self["max_length"]


class Action(Enum):
    NORTH = auto()
    EAST = auto()
    SOUTH = auto()
    WEST = auto()

    def to_row_col(self):
        if self == Action.NORTH:
            return -1, 0
        if self == Action.SOUTH:
            return 1, 0
        if self == Action.EAST:
            return 0, 1
        if self == Action.WEST:
            return 0, -1
        return 0, 0

    def opposite(self):
        if self == Action.NORTH:
            return Action.SOUTH
        if self == Action.SOUTH:
            return Action.NORTH
        if self == Action.EAST:
            return Action.WEST
        if self == Action.WEST:
            return Action.EAST
        raise TypeError(str(self) + " is not a valid Action.")


def row_col(position: int, columns: int) -> Tuple[int, int]:
    return position // columns, position % columns


def translate(position: int, direction: Action, columns: int, rows: int) -> int:
    row, column = row_col(position, columns)
    row_offset, column_offset = direction.to_row_col()
    row = (row + row_offset) % rows
    column = (column + column_offset) % columns
    return row * columns + column


def adjacent_positions(position: int, columns: int, rows: int) -> List[int]:
    return [
        translate(position, action, columns, rows)
        for action in Action
    ]


def min_distance(position: int, food: List[int], columns: int):
    row, column = row_col(position, columns)
    return min(
        abs(row - food_row) + abs(column - food_column)
        for food_position in food
        for food_row, food_column in [row_col(food_position, columns)]
    )


def random_agent():
    return choice([action for action in Action]).name


class GreedyAgent:
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.last_action = None

    def __call__(self, observation: Observation):
        rows, columns = self.configuration.rows, self.configuration.columns

        food = observation.food
        geese = observation.geese
        opponents = [
            goose
            for index, goose in enumerate(geese)
            if index != observation.index and len(goose) > 0
        ]

        # Don't move adjacent to any heads
        head_adjacent_positions = {
            opponent_head_adjacent
            for opponent in opponents
            for opponent_head in [opponent[0]]
            for opponent_head_adjacent in adjacent_positions(opponent_head, columns, rows)
        }
        # Don't move into any bodies
        bodies = {position for goose in geese for position in goose}

        # Move to the closest food
        position = geese[observation.index][0]
        actions = {
            action: min_distance(new_position, food, columns)
            for action in Action
            for new_position in [translate(position, action, columns, rows)]
            if (
                new_position not in head_adjacent_positions and
                new_position not in bodies and
                (self.last_action is None or action != self.last_action.opposite())
            )
        }

        action = min(actions, key=actions.get) if any(actions) else choice([action for action in Action])
        self.last_action = action
        return action.name


cached_greedy_agents = {}


def greedy_agent(obs, config):
    index = obs["index"]
    if index not in cached_greedy_agents:
        cached_greedy_agents[index] = GreedyAgent(Configuration(config))
    return cached_greedy_agents[index](Observation(obs))


agents = {"random": random_agent, "greedy": greedy_agent}


def interpreter(state, env):
    configuration = Configuration(env.configuration)
    columns = configuration.columns
    rows = configuration.rows
    min_food = configuration.min_food
    state[0].observation = shared_observation = Observation(state[0].observation)

    # Reset the environment.
    if env.done:
        agent_count = len(state)
        heads = sample(range(columns * rows), agent_count)
        shared_observation["geese"] = [[head] for head in heads]
        food_candidates = set(range(columns * rows)).difference(heads)
        # Ensure we only place as many food as there are open squares
        min_food = min(min_food, len(food_candidates))
        shared_observation["food"] = sample(food_candidates, min_food)
        return state

    geese = shared_observation.geese
    food = shared_observation.food

    # If there is no last state, reuse current state so that current action is never the opposite of the last action.
    last_state = env.steps[-1] if len(env.steps) > 1 else state
    # Apply the actions from active agents.
    for index, agent in enumerate(state):
        if agent.status != "ACTIVE":
            if agent.status != "INACTIVE" and agent.status != "DONE":
                # ERROR, INVALID, or TIMEOUT, remove the goose.
                geese[index] = []
            continue

        action = Action[agent.action]

        # Check action direction
        last_agent = last_state[index]
        last_action = Action[last_agent["action"]] if "action" in last_agent else action
        if last_action == action.opposite():
            env.debug_print(f"Opposite action: {agent.observation.index, action, last_action}")
            agent.status = "DONE"
            geese[index] = []
            continue

        goose = geese[index]
        head = translate(goose[0], action, columns, rows)

        # Consume food or drop a tail piece.
        if head in food:
            food.remove(head)
        else:
            goose.pop()

        # Self collision.
        if head in goose:
            env.debug_print(f"Body Hit: {agent.observation.index, action, head, goose}")
            agent.status = "DONE"
            geese[index] = []
            continue

        while len(goose) >= configuration.max_length:
            # Free a spot for the new head if needed
            goose.pop()
        # Add New Head to the Goose.
        goose.insert(0, head)

        # If hunger strikes remove from the tail.
        if len(env.steps) % configuration.hunger_rate == 0:
            if len(goose) > 0:
                goose.pop()
            if len(goose) == 0:
                env.debug_print(f"Goose Starved: {action}")
                agent.status = "DONE"
                continue

    goose_positions = histogram(
        position
        for goose in geese
        for position in goose
    )

    # Check for collisions.
    for index, agent in enumerate(state):
        goose = geese[index]
        if len(goose) > 0:
            head = geese[index][0]
            if goose_positions[head] > 1:
                env.debug_print(f"Goose Collision: {agent.action}")
                agent.status = "DONE"
                geese[index] = []

    # Add food if min_food threshold reached.
    needed_food = min_food - len(food)
    if needed_food > 0:
        collisions = {
            position
            for goose in geese
            for position in goose
        }
        available_positions = set(range(rows * columns)).difference(collisions).difference(food)
        # Ensure we don't sample more food than available positions.
        needed_food = min(needed_food, len(available_positions))
        food.extend(sample(available_positions, needed_food))

    # Set rewards after deleting all geese to ensure that geese don't receive a reward on the turn they perish.
    for index, agent in enumerate(state):
        if agent.status == "ACTIVE":
            # Adding 1 to len(env.steps) ensures that if an agent gets reward 4507, it died on turn 45 with length 7.
            agent.reward = (len(env.steps) + 1) * (configuration.max_length + 1) + len(geese[index])

    # If only one ACTIVE agent left, set it to DONE.
    active_agents = [a for a in state if a.status == "ACTIVE"]
    if len(active_agents) == 1:
        agent = active_agents[0]
        agent.status = "DONE"

    return state


def renderer(state, env):
    config = env.configuration
    columns = config.columns
    rows = config.rows

    food_symbol = "F"
    column_divider = "|"
    row_divider = "+" + "+".join(["---"] * columns) + "+\n"

    board = [" "] * (rows * columns)
    for pos in state[0].observation.food:
        board[pos] = food_symbol

    for index, goose in enumerate(state[0].observation.geese):
        for position in goose:
            board[position] = index

    out = row_divider
    for row in range(rows):
        for col in range(columns):
            out += column_divider + f" {board[(row * columns) + col]} "
        out += column_divider + "\n" + row_divider

    return out


dirpath = path.dirname(__file__)
jsonpath = path.abspath(path.join(dirpath, "hungry_geese.json"))
with open(jsonpath) as f:
    specification = json.load(f)


def html_renderer():
    jspath = path.abspath(path.join(dirpath, "hungry_geese.js"))
    with open(jspath) as f:
        return f.read()
