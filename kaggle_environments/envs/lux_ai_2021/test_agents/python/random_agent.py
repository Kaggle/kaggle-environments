import random

if __package__ == "":
    # for kaggle-environments
    from lux.constants import Constants
    from lux.game import Game
else:
    # for CLI tool
    from .lux.constants import Constants
    from .lux.game import Game
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
        game_state._update(observation["updates"][2:])
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
