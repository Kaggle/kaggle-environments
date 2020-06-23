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
    env.run([step_appender_agent])
    assert actual_steps == list(range(step_count - 1))


def test_halite_completes():
    env = make("halite", configuration={"episodeSteps": 100})
    env.run([random_agent, random_agent])
    json = env.toJSON()
    assert json["name"] == "halite"
    assert json["statuses"] == ["DONE", "DONE"]


def test_halite_exception_action_has_error_status():
    env = make("halite")

    def error_agent(obs, config):
        raise Exception("An exception occurred!")
    env.run([error_agent, random_agent])
    json = env.toJSON()
    assert json["name"] == "halite"
    assert json["statuses"] == ["ERROR", "DONE"]


def test_halite_helpers():
    env = make("halite", configuration={"size": 3})

    @board_agent
    def helper_agent(board):
        for ship in board.current_player.ships:
            ship.next_action = ShipAction.NORTH
        for shipyard in board.current_player.shipyards:
            shipyard.next_action = ShipyardAction.SPAWN

    env.run([helper_agent, helper_agent])

    json = env.toJSON()
    assert json["name"] == "halite"
    assert json["statuses"] == ["DONE", "DONE"]


def create_board(size=3, starting_halite=0, agent_count=2, random_seed=0):
    env = make("halite", configuration={
        "size": size,
        "startingHalite": starting_halite,
        "randomSeed": random_seed
    })
    return Board(env.reset(agent_count)[0].observation, env.configuration)


def test_move_moves_ship():
    size = 3
    board = create_board(size, agent_count=1)
    for ship in board.current_player.ships:
        ship.next_action = ShipAction.SOUTH
    next_board = board.next()
    for ship in board.ships.values():
        next_position = ship.position.translate(Point(0, -1), size)
        next_ship = next_board.ships[ship.id]
        assert next_ship.position == next_position


def move_toward(ship, target: Point):
    (x1, y1) = ship.position
    (x2, y2) = target
    if x2 > x1:
        return ShipAction.EAST
    elif x2 < x1:
        return ShipAction.WEST
    elif y2 > y1:
        return ShipAction.NORTH
    elif y2 < y1:
        return ShipAction.SOUTH


def test_equal_ship_collision_destroys_both_ships():
    size = 3
    board = create_board(size, agent_count=2)
    for ship in board.ships.values():
        ship.next_action = move_toward(ship, Point(1, 1))
    next_board = board.next()
    assert len(next_board.ships) == 0


def test_unequal_ship_collision_destroys_weaker_ship():
    board = create_board(agent_count=2)
    for opponent in board.opponents:
        for ship in opponent.ships:
            # Make the opponents' ships have more halite so they'll be destroyed
            ship._halite = 1000
    for ship in board.ships.values():
        ship.next_action = move_toward(ship, Point(1, 1))
    next_board = board.next()
    assert len(next_board.current_player.ships) == 1
    assert len(next_board.ships) == 1


def first(iterable):
    return next(iter(iterable))


def test_ship_shipyard_collision_destroys_both():
    board = create_board(agent_count=2)
    player_ship = first(board.current_player.ships)
    opponent_ship = first(first(board.opponents).ships)
    opponent_ship.next_action = ShipAction.CONVERT
    board = board.next()
    assert len(board.ships) == 1
    assert len(board.shipyards) == 1
    while player_ship.id in board.ships:
        board.ships[player_ship.id].next_action = move_toward(player_ship, opponent_ship.position)
        board = board.next()
    assert len(board.ships) == 0
    assert len(board.shipyards) == 0


def test_cells_regen_halite():
    board = create_board(starting_halite=1000, agent_count=1)
    cell = first(board.cells.values())
    next_board = board.next()
    next_cell = next_board[cell.position]
    expected_regen = round(cell.halite * board.configuration.regen_rate, 3)
    # We compare to a floating point value here to handle float rounding errors
    assert next_cell.halite - cell.halite - expected_regen < .000001


def test_no_move_on_halite_gathers_halite():
    board = create_board(starting_halite=1000, agent_count=1)
    ship = first(board.ships.values())
    expected_delta = int(ship.cell.halite * board.configuration.collect_rate)
    next_board = board.next()
    next_ship = next_board.ships[ship.id]
    ship_delta = next_ship.halite - ship.halite
    cell_delta = round(ship.cell.halite - next_ship.cell.halite, 3)
    assert ship_delta == expected_delta
    assert cell_delta == expected_delta


def test_move_on_halite_gathers_no_halite():
    board = create_board(starting_halite=1000, agent_count=1)
    ship = first(board.ships.values())
    ship.next_action = ShipAction.NORTH
    next_board = board.next()
    next_ship = next_board.ships[ship.id]
    ship_delta = next_ship.halite - ship.halite
    assert ship_delta == 0


def test_failed_convert_gathers_halite():
    board = create_board(starting_halite=1000, agent_count=1)
    board.current_player._halite = board.configuration.convert_cost - 1
    ship = first(board.ships.values())
    ship.next_action = ShipAction.CONVERT
    expected_delta = int(ship.cell.halite * board.configuration.collect_rate)
    next_board = board.next()
    next_ship = next_board.ships[ship.id]
    ship_delta = next_ship.halite - ship.halite
    cell_delta = round(ship.cell.halite - next_ship.cell.halite, 3)
    assert ship_delta == expected_delta
    assert cell_delta == expected_delta


def test_shipyard_ids_not_reused():
    board = create_board(starting_halite=1000, agent_count=1)
    ship = first(board.ships.values())
    ship.next_action = ShipAction.CONVERT
    board = board.next()
    shipyard = board.cells[ship.position].shipyard
    assert ship.id != shipyard.id


def test_seed_parameter():
    seed = 9

    def aggregate_halite_for_board(seed):
        board = create_board(starting_halite=1000, agent_count=1, random_seed=seed)
        return sum(map(lambda c: c.halite, board.cells.values()))

    assert aggregate_halite_for_board(seed) == aggregate_halite_for_board(seed)
