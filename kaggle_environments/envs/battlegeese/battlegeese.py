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
from random import choice, sample


def get_pos(from_pos, direction, columns, rows):
    if direction == "N":
        if from_pos - columns < 0:
            return -1
        return from_pos - columns
    if direction == "S":
        if from_pos + columns >= columns * rows:
            return -1
        return from_pos + columns
    if direction == "E":
        if from_pos // columns != (from_pos + 1) // columns:
            return -1
        return from_pos + 1
    if direction == "W":
        if from_pos // columns != (from_pos - 1) // columns:
            return -1
        return from_pos - 1


def min_distance(pos, food, config):
    cols = config.columns
    return min([abs(pos % cols - fpos % cols) + abs(pos // cols - fpos // cols) for fpos in food])


def random_agent():
    return choice(["N", "S", "E", "W"])


def shortest_path_agent(obs, config):
    columns = config.columns
    rows = config.rows
    goose = obs.geese[obs.index]
    head = goose[0]
    max_value = columns * rows
    directions = ["N", "S", "E", "W"]

    geese = [g for g in obs.geese if len(g) > 0]
    heads = [g[0] for g in geese]
    tails = [g[-1] for g in geese]
    bodies = [p for g in geese for p in g[:-1]]

    actions = {}
    for d in directions:
        # Get new position - or -1 if run into wall.
        pos = get_pos(head, d, columns, rows)
        if (
            pos in bodies or  # Hit a body.
            (len(goose) > 1 and goose[1] == pos) or  # Backwards.
            # Hit a tail when head over food.
            (pos in tails and heads[tails.index(pos)] in obs.food)
        ):
            pos = -1
        actions[d] = max_value if pos == - \
            1 else min_distance(pos, obs.food, config)
        # Possibility of a collision, devalue a valid action.
        for h in heads:
            if h == head:
                continue
            for dh in directions:
                posh = get_pos(h, dh, columns, rows)
                if posh == pos:
                    actions[d] += 1

    return min(actions, key=actions.get)


agents = {"random": random_agent, "shortest": shortest_path_agent}


def interpreter(state, env):
    config = env.configuration
    columns = config.columns
    rows = config.rows
    hunger_rate = config.hunger_rate
    min_food = config.min_food
    num_agents = len(state)

    # Clone the geese and food observation between all the agents.
    geese = state[0].observation.geese
    food = state[0].observation.food
    for agent in state:
        agent.observation.geese = geese
        agent.observation.food = food

    # Reset the environment.
    if env.done:
        # Distribute food and geese randomly.
        starting_positions = sample(range(columns * rows), num_agents * 2)
        for index in range(num_agents):
            geese.append([starting_positions[index]])
            food.append(starting_positions[index + num_agents])
        return state

    # Update active agents rewards.
    for index, agent in enumerate(state):
        if agent.status == "ACTIVE":
            agent.reward = len(env.steps) + len(geese[index])

    # Apply the actions from active agents.
    for index, agent in enumerate(state):
        if agent.status != "ACTIVE":
            continue
        action = agent.action
        goose = geese[index]
        head = goose[0]
        new_head = get_pos(head, action, columns, rows)

        # Wall Hit.
        if new_head == -1:
            env.debug_print(f"Wall Hit: {action}")
            agent.status = "INACTIVE"
            geese[index] = []
            continue

        # Last Body Hit.
        if len(goose) > 1 and goose[1] == new_head:
            env.debug_print(f"Body Hit: {action}")
            agent.status = "INACTIVE"
            geese[index] = []
            continue

        # Add New Head to the Goose.
        goose.insert(0, new_head)

        # Check Food.
        if head in food:
            food.remove(head)
        else:
            goose.pop()

        # If hunger strikes remove from the tail.
        if len(env.steps) % hunger_rate == 0:
            goose.pop()
            if len(goose) == 0:
                env.debug_print(f"Goose Starved: {action}")
                agent.status = "INACTIVE"
                geese[index] = []
                continue

    # Check for collisions.
    collisions = {}
    for goose in geese:
        for pos in goose:
            collisions[pos] = collisions.get(pos, 0) + 1
    for index, agent in enumerate(state):
        for pos in geese[index]:
            if collisions[pos] > 1:
                env.debug_print(f"Goose Collision: {agent.action}")
                agent.status = "INACTIVE"
                geese[index] = []
                continue

    # Add food if min_food threshold reached.
    if len(food) < min_food:
        available_positions = list(range(rows * columns))
        for goose in geese:
            for pos in goose:
                available_positions.remove(pos)
        food.extend(sample(available_positions, min_food - len(food)))

    # If only one ACTIVE agent left, set it to INACTIVE.
    active_agents = [a for a in state if a.status == "ACTIVE"]
    if len(active_agents) == 1:
        active_agents[0].status = "INACTIVE"

    return state


def renderer(state, env):
    config = env.configuration
    columns = config.columns
    rows = config.rows

    foodSymbol = "F"
    colDivider = "|"
    rowDivider = "+" + "+".join(["---"] * columns) + "+\n"

    board = [" "] * (rows * columns)
    for pos in state[0].observation.food:
        board[pos] = foodSymbol

    for index, goose in enumerate(state[0].observation.geese):
        for pos in goose:
            board[pos] = index

    out = rowDivider
    for row in range(rows):
        for col in range(columns):
            out += colDivider + f" {board[(row * columns) + col]} "
        out += colDivider + "\n" + rowDivider

    return out


dirpath = path.dirname(__file__)
jsonpath = path.abspath(path.join(dirpath, "battlegeese.json"))
with open(jsonpath) as f:
    specification = json.load(f)


def html_renderer():
    jspath = path.abspath(path.join(dirpath, "battlegeese.js"))
    with open(jspath) as f:
        return f.read()
