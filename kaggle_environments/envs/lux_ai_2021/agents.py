import random
from .utils import get_score
from .kit.game import Game
from .kit.game_map import Position
from .kit.constants import Constants
DIRECTIONS = Constants.DIRECTIONS
game_state = None

def random_agent(observation, configuration):
    """
    a blank, completely empty agent, usually incapable of surviving past the first night
    """
    global game_state

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
    else:
        game_state._update(observation["updates"])
    
    actions = []

    ### AI Code goes down here! ### 
    player = game_state.players[observation.player]
    for unit in player.units:
        dirs = [DIRECTIONS.NORTH, DIRECTIONS.WEST, DIRECTIONS.EAST, DIRECTIONS.SOUTH]
        action = unit.move(random.choice(dirs))
        actions.append(action)
    
    return actions

def collector_agent(observation, configuration):
    global game_state

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
    else:
        game_state._update(observation["updates"])
    
    actions = []

    ### AI Code goes down here! ### 
    player = game_state.players[observation.player]


    # loop over entire map and find closest resources to city center
    width, height = game_state.game_map.width, game_state.game_map.height
    avg_city_x = 0
    avg_city_y = 0
    for k, v in player.cities.items():
        for city_tile in v.citytiles:
            avg_city_x += city_tile.pos.x
            avg_city_y += city_tile.pos.y
        avg_city_x /= len(v.citytiles)
        avg_city_y /= len(v.citytiles)
        break
    city_center_pos = Position(avg_city_x, avg_city_y)
    
    closest_resource_pos = None
    closest_resource_pos_dist = 99999999
    for y in range(height):
        for x in range(width):
            cell = game_state.game_map.get_cell(x, y)
            if cell.resource is Constants.RESOURCE_TYPES.WOOD:
                dist = cell.pos.distance_to(city_center_pos)
                if dist < closest_resource_pos_dist:
                    closest_resource_pos_dist = dist
                    closest_resource_pos = cell.pos
                pass
    for unit in player.units:
        dirs = [DIRECTIONS.NORTH, DIRECTIONS.WEST, DIRECTIONS.EAST, DIRECTIONS.SOUTH]
        # action = unit.move(random.choice(dirs))
        action = unit.move(unit.pos.direction_to(closest_resource_pos))
        actions.append(action)
    
    return actions

agents = {
    "random_agent": random_agent,
    "collector_agent": collector_agent
}