import random
from .utils import get_score
from .kit.game import Game
from .kit.constants import Constants
DIRECTIONS = Constants.DIRECTIONS
game_state = None

def random_agent(observation, configuration):
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

agents = {
    "random_agent": random_agent
}