# Copyright 2022 Kaggle Inc
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

import copy
import json
import math
import numpy as np
from os import path
from random import choice, randint, randrange, sample, seed, random
from .helpers import board_agent, Board, ShipyardAction
from kaggle_environments import utils
from kaggle_environments.helpers import Point, Direction


def get_col_row(size, pos):
    return pos % size, pos // size


def get_to_pos(size, pos, direction):
    col, row = get_col_row(size, pos)
    if direction == "NORTH":
        return pos - size if pos >= size else size ** 2 - size + col
    elif direction == "SOUTH":
        return col if pos + size >= size ** 2 else pos + size
    elif direction == "EAST":
        return pos + 1 if col < size - 1 else row * size
    elif direction == "WEST":
        return pos - 1 if col > 0 else (row + 1) * size - 1

def check_path(board, start, dirs, dist_a, dist_b, collection_rate):
    kore = 0
    npv = .98
    current = start
    steps = 2 * (dist_a + dist_b + 2)
    for idx, d in enumerate(dirs):
        for _ in range((dist_a if idx % 2 == 0 else dist_b) + 1):
            current = current.translate(d.to_point(), board.configuration.size)
            kore += int((board.cells.get(current).kore or 0) * collection_rate)
    return math.pow(npv, steps) * kore / (2 * (dist_a + dist_b + 2))

def check_location(board, loc, me):
    if board.cells.get(loc).shipyard and board.cells.get(loc).shipyard.player.id == me.id:
        return 0
    kore = 0
    for i in range(-3, 4):
        for j in range(-3, 4):
            pos = loc.translate(Point(i, j), board.configuration.size)
            kore += board.cells.get(pos).kore or 0
    return kore

def get_closest_enemy_shipyard(board, position, me):
    min_dist = 1000000
    enemy_shipyard = None
    for shipyard in board.shipyards.values():
        if shipyard.player_id == me.id:
            continue
        dist = position.distance_to(shipyard.position, board.configuration.size)
        if dist < min_dist:
            min_dist = dist
            enemy_shipyard = shipyard
    return enemy_shipyard
    
def get_shortest_flight_path_between(position_a, position_b, size, trailing_digits=False):
    mag_x = 1 if position_b.x > position_a.x else -1
    abs_x = abs(position_b.x - position_a.x)
    dir_x = mag_x if abs_x < size/2 else -mag_x
    mag_y = 1 if position_b.y > position_a.y else -1
    abs_y = abs(position_b.y - position_a.y)
    dir_y = mag_y if abs_y < size/2 else -mag_y
    flight_path_x = ""
    if abs_x > 0:
        flight_path_x += "E" if dir_x == 1 else "W"
        flight_path_x += str(abs_x - 1) if (abs_x - 1) > 0 else ""
    flight_path_y = ""
    if abs_y > 0:
        flight_path_y += "N" if dir_y == 1 else "S"
        flight_path_y += str(abs_y - 1) if (abs_y - 1) > 0 else ""
    if not len(flight_path_x) == len(flight_path_y):
        if len(flight_path_x) < len(flight_path_y):
            return flight_path_x + (flight_path_y if trailing_digits else flight_path_y[0])
        else:
            return flight_path_y + (flight_path_x if trailing_digits else flight_path_x[0])
    return flight_path_y + (flight_path_x if trailing_digits or not flight_path_x else flight_path_x[0]) if random() < .5 else flight_path_x + (flight_path_y if trailing_digits or not flight_path_y else flight_path_y[0])

@board_agent
def attacker_agent(board):
    me = board.current_player
    remaining_kore = me.kore
    shipyards = me.shipyards
    size = board.configuration.size
    spawn_cost = board.configuration.spawn_cost
    # randomize shipyard order
    shipyards = sample(shipyards, len(shipyards))
    for idx, shipyard in enumerate(shipyards):
        closest_enemy_shipyard = get_closest_enemy_shipyard(board, shipyard.position, me)
        if closest_enemy_shipyard and (remaining_kore >= spawn_cost or shipyard.ship_count >= 50):
            if shipyard.ship_count >= 50:
                flight_plan = get_shortest_flight_path_between(shipyard.position, closest_enemy_shipyard.position, size)
                shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(50, flight_plan)
            elif remaining_kore >= spawn_cost:
                shipyard.next_action = ShipyardAction.spawn_ships(min(shipyard.max_spawn, int(remaining_kore/spawn_cost)))
        elif shipyard.ship_count >= 21:
            best_h = 0
            best_gap1 = 5
            best_gap2 = 5
            start_dir = board.step % 4
            dirs = Direction.list_directions()[start_dir:] + Direction.list_directions()[:start_dir]
            for gap1 in range(0, 10):
                for gap2 in range(0, 10):
                    h = check_path(board, shipyard.position, dirs, gap1, gap2, .2)
                    if h > best_h:
                        best_h = h
                        best_gap1 = gap1
                        best_gap2 = gap2
            gap1 = str(best_gap1)
            gap2 = str(best_gap2)
            flight_plan = Direction.list_directions()[start_dir].to_char()
            if int(gap1):
                flight_plan += gap1
            next_dir = (start_dir + 1) % 4
            flight_plan += Direction.list_directions()[next_dir].to_char()
            if int(gap2):
                flight_plan += gap2
            next_dir = (next_dir + 1) % 4
            flight_plan += Direction.list_directions()[next_dir].to_char()
            if int(gap1):
                flight_plan += gap1
            next_dir = (next_dir + 1) % 4
            flight_plan += Direction.list_directions()[next_dir].to_char()
            shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(21, flight_plan)
        elif shipyard.ship_count > 0 and len(shipyards) > 1:
            flight_plan = get_shortest_flight_path_between(shipyard.position, shipyards[(idx + 1) % len(shipyards)].position, size)
            shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(shipyard.ship_count, flight_plan)



@board_agent
def simp_agent(board):
    me = board.current_player
    if board.step % 100  == 0 and me.id == 0:
        import json
        obs = board.observation
    me = board.current_player
    remaining_kore = me.kore
    shipyards = me.shipyards
    convert_cost = board.configuration.convert_cost
    size = board.configuration.size
    spawn_cost = board.configuration.spawn_cost
    # randomize shipyard order
    shipyards = sample(shipyards, len(shipyards))
    for shipyard in shipyards:
        closest_enemy_shipyard = get_closest_enemy_shipyard(board, shipyard.position, me)
        invading_fleet_size = 100
        dist_to_closest_enemy_shipyard = 100 if not closest_enemy_shipyard else shipyard.position.distance_to(closest_enemy_shipyard.position, size)
        if closest_enemy_shipyard and (closest_enemy_shipyard.ship_count < 20 or dist_to_closest_enemy_shipyard < 15) and (remaining_kore >= spawn_cost or shipyard.ship_count >= invading_fleet_size):
            if shipyard.ship_count >= invading_fleet_size:
                flight_plan = get_shortest_flight_path_between(shipyard.position, closest_enemy_shipyard.position, size)
                shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(invading_fleet_size, flight_plan)
            elif remaining_kore >= spawn_cost:
                shipyard.next_action = ShipyardAction.spawn_ships(min(shipyard.max_spawn, int(remaining_kore/spawn_cost)))

        elif remaining_kore > 500 and shipyard.max_spawn > 5:
            if shipyard.ship_count >= convert_cost + 7:
                start_dir = randint(0, 3)
                next_dir = (start_dir + 1) % 4
                best_kore = 0
                best_gap1 = 0
                best_gap2 = 0
                for gap1 in range(5, 15, 3):
                    for gap2 in range(5, 15, 3):
                        gap2 = randint(3, 9)
                        diff1 = Direction.from_index(start_dir).to_point() * gap1
                        diff2 = Direction.from_index(next_dir).to_point() * gap2
                        diff = diff1 + diff2
                        pos = shipyard.position.translate(diff, board.configuration.size)
                        h = check_location(board, pos, me)
                        if h > best_kore:
                            best_kore = h
                            best_gap1 = gap1
                            best_gap2 = gap2
                gap1 = str(best_gap1)
                gap2 = str(best_gap2)
                flight_plan = Direction.list_directions()[start_dir].to_char() + gap1
                flight_plan += Direction.list_directions()[next_dir].to_char() + gap2
                flight_plan += "C"
                shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(max(convert_cost + 7, int(shipyard.ship_count/2)), flight_plan)
            elif remaining_kore >= spawn_cost:
                shipyard.next_action = ShipyardAction.spawn_ships(min(shipyard.max_spawn, int(remaining_kore/spawn_cost)))

        # launch a large fleet if able
        elif shipyard.ship_count >= 21:
            best_h = 0
            best_gap1 = 5
            best_gap2 = 5
            start_dir = board.step % 4
            dirs = Direction.list_directions()[start_dir:] + Direction.list_directions()[:start_dir]
            for gap1 in range(0, 10):
                for gap2 in range(0, 10):
                    h = check_path(board, shipyard.position, dirs, gap1, gap2, .2)
                    if h > best_h:
                        best_h = h
                        best_gap1 = gap1
                        best_gap2 = gap2
            gap1 = str(best_gap1)
            gap2 = str(best_gap2)
            flight_plan = Direction.list_directions()[start_dir].to_char()
            if int(gap1):
                flight_plan += gap1
            next_dir = (start_dir + 1) % 4
            flight_plan += Direction.list_directions()[next_dir].to_char()
            if int(gap2):
                flight_plan += gap2
            next_dir = (next_dir + 1) % 4
            flight_plan += Direction.list_directions()[next_dir].to_char()
            if int(gap1):
                flight_plan += gap1
            next_dir = (next_dir + 1) % 4
            flight_plan += Direction.list_directions()[next_dir].to_char()
            shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(21, flight_plan)
        # else spawn if possible
        elif remaining_kore > board.configuration.spawn_cost * shipyard.max_spawn:
            remaining_kore -= board.configuration.spawn_cost
            if remaining_kore >= spawn_cost:
                shipyard.next_action = ShipyardAction.spawn_ships(min(shipyard.max_spawn, int(remaining_kore/spawn_cost)))

@board_agent
def do_nothing_agent(board):
    pass

@board_agent
def random_agent(board):
    me = board.current_player
    remaining_kore = me.kore
    shipyards = me.shipyards
    # randomize shipyard order
    shipyards = sample(shipyards, len(shipyards))
    for shipyard in shipyards:
        # 25% chance to launch a large fleet
        if randint(0, 3) == 0 and shipyard.ship_count > 10:
            dir_str = Direction.random_direction().to_char()
            dir2_str = Direction.random_direction().to_char()
            flight_plan = dir_str + str(randint(1, 10)) + dir2_str
            shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(min(10, math.floor(shipyard.ship_count / 2)), flight_plan)
        # else spawn if possible
        elif remaining_kore > board.configuration.spawn_cost * shipyard.max_spawn:
            remaining_kore -= board.configuration.spawn_cost
            shipyard.next_action = ShipyardAction.spawn_ships(shipyard.max_spawn)
        # else launch a small fleet
        elif shipyard.ship_count >= 2:
            dir_str = Direction.random_direction().to_char()
            shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(2, dir_str)

@board_agent
def simple_agent(board):
    me = board.current_player
    remaining_kore = me.kore
    shipyards = me.shipyards
    convert_cost = board.configuration.convert_cost
    spawn_cost = board.configuration.spawn_cost
    # randomize shipyard order
    shipyards = sample(shipyards, len(shipyards))
    for shipyard in shipyards:
        if remaining_kore > 1000 and shipyard.max_spawn > 5:
            if shipyard.ship_count >= convert_cost + 10:
                gap1 = str(randint(3, 9))
                gap2 = str(randint(3, 9))
                start_dir = randint(0, 3)
                flight_plan = Direction.list_directions()[start_dir].to_char() + gap1
                next_dir = (start_dir + 1) % 4
                flight_plan += Direction.list_directions()[next_dir].to_char() + gap2
                next_dir = (next_dir + 1) % 4
                flight_plan += "C"
                shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(max(convert_cost + 10, int(shipyard.ship_count/2)), flight_plan)
            elif remaining_kore >= spawn_cost:
                shipyard.next_action = ShipyardAction.spawn_ships(min(shipyard.max_spawn, int(remaining_kore/spawn_cost)))

        # launch a large fleet if able

        elif shipyard.ship_count >= 21:
            gap1 = str(randint(3, 9))
            gap2 = str(randint(3, 9))
            start_dir = randint(0, 3)
            flight_plan = Direction.list_directions()[start_dir].to_char() + gap1
            next_dir = (start_dir + 1) % 4
            flight_plan += Direction.list_directions()[next_dir].to_char() + gap2
            next_dir = (next_dir + 1) % 4
            flight_plan += Direction.list_directions()[next_dir].to_char() + gap1
            next_dir = (next_dir + 1) % 4
            flight_plan += Direction.list_directions()[next_dir].to_char()
            shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(21, flight_plan)
        # else spawn if possible
        elif remaining_kore > board.configuration.spawn_cost * shipyard.max_spawn:
            remaining_kore -= board.configuration.spawn_cost
            if remaining_kore >= spawn_cost:
                shipyard.next_action = ShipyardAction.spawn_ships(min(shipyard.max_spawn, int(remaining_kore/spawn_cost)))
        # else launch a small fleet
        elif shipyard.ship_count >= 2:
            dir_str = Direction.random_direction().to_char()
            shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(2, dir_str)

        
agents = {"random": random_agent, "simple": simple_agent, "do_nothing": do_nothing_agent, "simp": simp_agent, "attacker": attacker_agent}


def populate_board(state, env):
    obs = state[0].observation
    config = env.configuration
    size = env.configuration.size
    uid_counter = 0

    # Set seed for random number generators
    if not hasattr(config, "randomSeed"):
        max_int_32 = (1 << 31) - 1
        config.randomSeed = randrange(max_int_32)

    np.random.seed(config.randomSeed)
    seed(config.randomSeed)

    # This is a consistent way to generate unique strings to form ship and shipyard ids
    def create_uid():
        nonlocal uid_counter
        uid_counter += 1
        return f"{obs.step}-{uid_counter}"

    # Distribute Kore evenly into quartiles.
    half = math.ceil(size / 2)
    grid = [[0] * half for _ in range(half)]

    # Randomly place a few kore "seeds".
    for i in range(half):
        # random distribution across entire quartile
        grid[randint(0, half - 1)][randint(0, half - 1)] = i ** 2

        # as well as a particular distribution weighted toward the center of the map
        grid[randint(half // 2, half - 1)][randint(half // 2, half - 1)] = i ** 2

    # Spread the seeds radially.
    radius_grid = copy.deepcopy(grid)
    for r in range(half):
        for c in range(half):
            value = grid[r][c]
            if value == 0:
                continue

            # keep initial seed values, but constrain radius of clusters
            radius = min(round((value / half) ** 0.5), 1)
            for r2 in range(r - radius + 1, r + radius):
                for c2 in range(c - radius + 1, c + radius):
                    if 0 <= r2 < half and 0 <= c2 < half:
                        distance = (abs(r2 - r) ** 2 + abs(c2 - c) ** 2) ** 0.5
                        radius_grid[r2][c2] += int(value / max(1, distance) ** distance)

    # add some random sprouts of kore
    radius_grid = np.asarray(radius_grid)
    add_grid = np.random.gumbel(0, 300.0, size=(half, half)).astype(int)
    sparse_radius_grid = np.random.binomial(1, 0.5, size=(half, half))
    add_grid = np.clip(add_grid, 0, a_max=None) * sparse_radius_grid
    radius_grid += add_grid

    # add another set of random locations to the center corner
    corner_grid = np.random.gumbel(0, 500.0, size=(half // 4, half // 4)).astype(int)
    corner_grid = np.clip(corner_grid, 0, a_max=None)
    radius_grid[half - (half // 4):, half - (half // 4):] += corner_grid

    # make it assomptocially symmetric
    for i in range(half):
        for j in range(half):
            if i + j < half:
                radius_grid[i][j] = radius_grid[j][i]

    # Normalize the available kore against the defined configuration starting kore.
    total = sum([sum(row) for row in radius_grid])
    obs.kore = [0] * (size ** 2)
    for r, row in enumerate(radius_grid):
        for c, val in enumerate(row):
            val = int(val * config.startingKore / total / 4)
            obs.kore[size * r + c] = val
            obs.kore[size * r + (size - c - 1)] = val
            obs.kore[size * (size - 1) - (size * r) + c] = val
            obs.kore[size * (size - 1) - (size * r) + (size - c - 1)] = val

    # Distribute the starting shipyards evenly.
    num_agents = len(state)
    starting_positions = [0] * num_agents
    if num_agents == 1:
        starting_positions[0] = size * (size // 2) + size // 2
    elif num_agents == 2:
        starting_positions[0] = size * (size // 2 - size // 4) + size // 4
        starting_positions[1] = size * (size // 2 + size // 4) + math.ceil(3 * size / 4) - 1
    elif num_agents == 4:
        starting_positions[0] = size * (size // 4 + 1) + size // 4 - 1
        starting_positions[1] = size * (size // 4 - 1) + 3 * size // 4 - 1
        starting_positions[2] = size * (3 * size // 4 + 1) + size // 4 + 1
        starting_positions[3] = size * (3 * size // 4 - 1) + 3 * size // 4 + 1
    
    # clear the kore on the starting square
    for pos in starting_positions:
        obs.kore[pos] = 0

    # Initialize the players.
    obs.players = []
    for i in range(num_agents):
        shipyards = {create_uid(): [starting_positions[i], 0, 0]}
        obs.players.append([state[0].reward, shipyards, {}])

    return state


def interpreter(state, env):
    obs = state[0].observation
    config = env.configuration

    # Initialize the board (place cell kore and starting ships).
    if env.done:
        return populate_board(state, env)

    # Interpreter invoked here
    actions = [agent.action for agent in state]
    board = Board(obs, config, actions)
    board = board.next()
    state[0].observation = obs = utils.structify(board.observation)

    # Remove players with invalid status or insufficient potential.
    for index, agent in enumerate(state):
        player_kore, shipyards, fleets = obs.players[index]
        ships_in_shipyards = [int(s[1]) for s in shipyards.values()]
        can_spawn = len(shipyards) > 0 and player_kore >= config.spawnCost
        if agent.status == "ACTIVE" and len(shipyards) == 0 and len(fleets) == 0:
            # Agent can no longer gather any kore
            agent.status = "DONE"
            agent.reward = board.step - board.configuration.episode_steps - 1
        if agent.status == "ACTIVE" and ships_in_shipyards == 0 and len(fleets) == 0 and not can_spawn:
            # Agent can no longer gather any kore
            agent.status = "DONE"
            agent.reward = board.step - board.configuration.episode_steps - 1
        if agent.status != "ACTIVE" and agent.status != "DONE":
            obs.players[index] = [0, {}, {}]

    # Check if done (< 2 players and num_agents > 1)
    if len(state) > 1 and sum(1 for agent in state if agent.status == "ACTIVE") < 2:
        for agent in state:
            if agent.status == "ACTIVE":
                agent.status = "DONE"

    # Update Rewards.
    for index, agent in enumerate(state):
        if agent.status == "ACTIVE":
            agent.reward = obs.players[index][0]
        elif agent.status != "DONE":
            agent.reward = 0

    return state


def renderer(state, env):
    config = env.configuration
    size = config.size
    obs = state[0].observation

    board = [[h, -1, -1, -1] for h in obs.kore]
    for index, player in enumerate(obs.players):
        _, shipyards, fleets = player
        for shipyard in shipyards.values():
            shipyard_pos, _, _ = shipyard
            board[shipyard_pos][1] = index
        for fleet in fleets.values():
            fleet_pos, fleet_kore, ship_count, _, _ = fleet
            board[fleet_pos][2] = index
            board[fleet_pos][3] = ship_count

    col_divider = "|"
    row_divider = "+" + "+".join(["----"] * size) + "+\n"

    out = row_divider
    for row in range(size):
        for col in range(size):
            _, _, fleet, fleet_kore = board[col + row * size]
            out += col_divider + (
                f"{min(int(fleet_kore), 99)}S{fleet}" if fleet > -1 else ""
            ).ljust(4)
        out += col_divider + "\n"
        for col in range(size):
            kore, shipyard, _, _ = board[col + row * size]
            if shipyard > -1:
                out += col_divider + f"SY{shipyard}".ljust(4)
            else:
                out += col_divider + str(min(int(kore), 9999)).rjust(4)
        out += col_divider + "\n" + row_divider

    return out


dir_path = path.dirname(__file__)
json_path = path.abspath(path.join(dir_path, "kore_fleets.json"))
with open(json_path) as json_file:
    specification = json.load(json_file)


def html_renderer():
    js_path = path.abspath(path.join(dir_path, "kore_fleets.js"))
    with open(js_path) as js_file:
        return js_file.read()
