from kaggle_environments import make
from .halite import random_agent
from .helpers import *


def test_halite_no_repeated_steps():
    step_count = 10
    actual_steps = []

    def step_appender_agent(obs, config):
        actual_steps.append(obs.step)
        return {}

    env = make("halite", configuration={"episodeSteps": step_count}, debug=True)
    env.run({step_appender_agent})
    assert actual_steps == list(range(step_count - 1))


def test_halite_completes():
    env = make("halite", debug=True)
    env.run([random_agent, random_agent])
    json = env.toJSON()
    assert json["name"] == "halite"
    assert json["statuses"] == ["DONE", "DONE"]


def test_halite_exception_action_has_error_status():
    env = make("halite", debug=True)

    def error_agent(obs, config):
        raise Exception("An exception occurred!")
    env.run([error_agent, random_agent])
    json = env.toJSON()
    assert json["name"] == "halite"
    assert json["statuses"] == ["ERROR", "DONE"]


def test_halite_helpers():
    env = make("halite", debug=True)

    def helper_agent(obs, config):
        board = Board(obs, config)
        for ship in board.current_player.ships:
            ship.try_set_pending_action(ShipAction.NORTH)
        for shipyard in board.current_player.shipyards:
            shipyard.try_set_pending_spawn(True)
        return board.pending_actions

    env.run([helper_agent, helper_agent])
    json = env.toJSON()
    assert json["name"] == "halite"
    assert json["statuses"] == ["DONE", "DONE"]