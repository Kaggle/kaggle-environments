import sys
import random
from .lux.game import Game
from .lux.game_map import Cell, Position
from .lux.constants import Constants
from kaggle_environments.envs.lux_ai_2021.test_agents.python.lux import game_map
DIRECTIONS = Constants.DIRECTIONS
game_state = None

def organic_agent(observation, configuration):
    global game_state

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
    else:
        game_state._update(observation["updates"])
    
    actions = []

    ### AI Code goes down here! ### 
    player = game_state.players[observation.player]
    
    width, height = game_state.game_map.width, game_state.game_map.height

    resource_tiles: list[Cell] = []
    for y in range(height):
        for x in range(width):
            cell = game_state.game_map.get_cell(x, y)
            if cell.resource is not None:
                resource_tiles.append(cell)




    # loop over entire map and find closest resources to city center
    
    cities_to_build = 0
    for k, city in player.cities.items():
        if (city.get_light_upkeep() < city.fuel + 200):
            cities_to_build += 1;

    for unit in player.units:
        if unit.is_worker():
            closest_dist = 999999999
            closest_resource_tile = None
            if unit.get_cargo_space_left() > 0:
                for resource_tile in resource_tiles:
                    dist = resource_tile.pos.distance_to(unit.pos)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_resource_tile = resource_tile
                if closest_resource_tile is not None:
                    actions.append(unit.move(unit.pos.direction_to(closest_resource_tile.pos)))
            else:
                # if we have cities, return to them
                if len(player.cities) > 0:
                    closest_dist = 999999
                    closest_city_tile = None
                    for k, city in player.cities.items():
                        for city_tile in city.citytiles:
                            dist = city_tile.pos.distance_to(unit.pos)
                            if dist < closest_dist:
                                closest_dist = dist
                                closest_city_tile = city_tile
                    if closest_city_tile is not None:
                        move_dir = unit.pos.direction_to(closest_city_tile.pos)
                        # print(game_state.turn, "Can build", unit.can_build(game_state.game_map))
                        # print(game_state.game_map.get_cell(14, 2).resource.amount)
                        if cities_to_build > 0 and unit.pos.is_adjacent(closest_city_tile.pos) and unit.can_build(game_state.game_map):
                            actions.append(unit.build_city())        
                        else:
                            actions.append(unit.move(move_dir))
    
    return actions