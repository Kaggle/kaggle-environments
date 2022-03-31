from kaggle_environments.envs.kore_fleets.helpers import *
from random import randint

def agent(obs, config):
    board = Board(obs, config)
    me=board.current_player

    me = board.current_player
    turn = board.step
    spawn_cost = board.configuration.spawn_cost
    kore_left = me.kore

    for shipyard in me.shipyards:
        if shipyard.ship_count > 10:
            direction = Direction.from_index(turn % 4)
            action = ShipyardAction.launch_fleet_with_flight_plan(2, direction.to_char())
            shipyard.next_action = action
        elif kore_left > spawn_cost * shipyard.max_spawn:
            action = ShipyardAction.spawn_ships(shipyard.max_spawn)
            shipyard.next_action = action
            kore_left -= spawn_cost * shipyard.max_spawn
        elif kore_left > spawn_cost:
            action = ShipyardAction.spawn_ships(1)
            shipyard.next_action = action
            kore_left -= spawn_cost

    return me.next_actions
