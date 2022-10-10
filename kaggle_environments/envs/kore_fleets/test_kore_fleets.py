from kaggle_environments import make
from .kore_fleets import random_agent
from .helpers import *

def test_shipyard_action_class_serialization():
    launch = ShipyardAction.launch_fleet_in_direction(10, Direction.NORTH)
    assert str(launch) == "LAUNCH_10_N", "Launch: " + str(launch)

    parsed = ShipyardAction.from_str(str(launch))
    assert launch.action_type == parsed.action_type, "type"
    assert launch.num_ships == parsed.num_ships, "num_ships"
    assert launch.flight_plan == parsed.flight_plan, "flight_plan"

    spawn = ShipyardAction.spawn_ships(1)
    assert str(spawn) == "SPAWN_1", "Spawn: " + str(launch)

    parsed = ShipyardAction.from_str(str(spawn))
    assert spawn.action_type == parsed.action_type, "type"
    assert spawn.num_ships == parsed.num_ships, "num_ships"

def test_kore_no_repeated_steps():
    step_count = 10
    actual_steps = []

    def step_appender_agent(obs, config):
        actual_steps.append(obs.step)
        return {}

    env = make("kore_fleets", configuration={"episodeSteps": step_count}, debug=True)
    env.run([step_appender_agent])
    assert actual_steps == list(range(step_count - 1))

def test_kore_helpers():
    env = make("kore_fleets", configuration={"size": 5, "episodeSteps": 100}, debug=True)

    @board_agent
    def helper_agent(board):
        for shipyard in board.current_player.shipyards:
            if shipyard.ship_count >= 10:
                shipyard.next_action = ShipyardAction.launch_fleet_in_direction(10, Direction.NORTH)
            else:
                shipyard.next_action = ShipyardAction.spawn_ships(1)

    env.run([helper_agent, helper_agent])

    json = env.toJSON()
    assert json["name"] == "kore_fleets"
    assert json["statuses"] == ["DONE", "DONE"]


def test_start_with_one_shipyard_and_no_fleets():
    env = make("kore_fleets", configuration={
        "size": 3,
        "startingKore": 100,
        "randomSeed": 0 
    })
    obs = env.reset(2)[0].observation
    players = obs.get('players')
    assert len(players) == 2
    assert len(players[0][1].items()) == 1
    assert len(players[0][2].items()) == 0

def test_shipyards_make_ships():
    board = create_board()
    for shipyard in board.shipyards.values():
        shipyard.next_action = ShipyardAction.spawn_ships(1)

    board = board.next()

    for shipyard in board.shipyards.values():
        assert shipyard.ship_count == 1, "Should have spawned a ship"

def test_shipyards_launch_fleet():
    board = create_board()
    for shipyard in board.shipyards.values():
        shipyard.next_action = ShipyardAction.spawn_ships(1)

    board = board.next()

    for shipyard in board.shipyards.values():
        shipyard.next_action = ShipyardAction.launch_fleet_in_direction(1, Direction.NORTH)

    board = board.next()

    assert len(board.current_player.fleets) == 1, "should have one fleet"

def test_flight_plan_truncated():
    board = create_board()
    for shipyard in board.shipyards.values():
        shipyard.next_action = ShipyardAction.spawn_ships(1)

    board = board.next()

    for shipyard in board.shipyards.values():
        shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(1, "NESW")

    board = board.next()

    assert len(board.current_player.fleets) == 1, "should have one fleet"
    print(board.current_player.fleets[0].flight_plan)
    assert board.current_player.fleets[0].flight_plan == "", "should have an empty flight plan"

def test_fleets_launched_with_direction():
    board = create_board()
    for shipyard in board.shipyards.values():
        shipyard.next_action = ShipyardAction.spawn_ships(1)

    board = board.next()

    for shipyard in board.shipyards.values():
        shipyard.next_action = ShipyardAction.launch_fleet_in_direction(1, Direction.NORTH)

    board = board.next()

    assert len(board.current_player.fleets) == 1, "should have one fleet"
    assert board.current_player.fleets[0].direction == Direction.NORTH, "should go NORTH"

def test_fleets_move_in_direction():
    board = create_board(size=10)
    for shipyard in board.shipyards.values():
        shipyard.next_action = ShipyardAction.spawn_ships(1)

    print("board.next()")
    board = board.next()

    for shipyard in board.shipyards.values():
        shipyard.next_action = ShipyardAction.launch_fleet_in_direction(1, Direction.NORTH)

    for fleet in board.fleets.values():
        print(fleet.id, fleet.direction, fleet.position)

    board = board.next()

    for fleet in board.fleets.values():
        print(fleet.id, fleet.direction, fleet.position)

    positions = {
        fleet.id: fleet.position
        for fleet in board.fleets.values()
    }

    board = board.next()

    new_positions = {
        fleet.id: fleet.position
        for fleet in board.fleets.values()
    }

    assert len(positions) == 2 and len(positions) == len(new_positions), "there should be two fleets"

    for key in positions.keys():
        assert positions.get(key) + Direction.NORTH.to_point() == new_positions.get(key)

def test_kore_regenerates():
    board = create_board(size=31)

    p = Point(10, 10)

    board.get_cell_at_point(p)._kore = 100

    next_board = board.next()

    assert board.get_cell_at_point(p).kore < next_board.get_cell_at_point(p).kore, "should have regenerated kore"

def test_spawn_zero_has_no_effect():
    board = create_board(size=31)

    shipyard_id = board.players[0].shipyard_ids[0]
    shipyard = board.shipyards.get(shipyard_id)

    shipyard.next_action = ShipyardAction.spawn_ships(0)

    next_board = board.next()
    next_shipyard = next_board.shipyards.get(shipyard_id)

    assert shipyard.ship_count == 0, 'should have 0 ships'
    assert next_shipyard.ship_count == 0, 'should not have spawned a ship'
    assert board.players[0].kore == next_board.players[0].kore, 'kore should be the same'
    
def test_fleet_coalescence_works():
    board = create_board(size=31)

    p = Point(10, 10)

    f1 = Fleet("f1", 100, Direction.SOUTH, p + Direction.NORTH.to_point(), 100, "", 0, board)
    board._add_fleet(f1)
    f2 = Fleet("f2", 60, Direction.NORTH, p + Direction.SOUTH.to_point(), 100, "", 0, board)
    board._add_fleet(f2)
    f3 = Fleet("f3", 60, Direction.WEST, p + Direction.EAST.to_point(), 100, "", 0, board)
    board._add_fleet(f3)

    next_board = board.next()

    next_fleet = next_board.get_fleet_at_point(p)
    assert next_fleet.id == "f1", "biggest fleet should absord others"

def test_fleets_pick_up_kore():
    board = create_board(size=31)

    p = Point(10, 10)
    board.get_cell_at_point(p)._kore = 100
    board.get_cell_at_point(p + Direction.SOUTH.to_point())._kore = 100

    fleet = Fleet("test", 100, Direction.SOUTH, p, 100, "8N", 0, board)

    board._add_fleet(fleet)

    next_board = board.next()

    next_fleet = next_board.get_fleet_at_point(p + Direction.SOUTH.to_point())
    assert next_fleet.kore > fleet.kore, "ships should pick up kore"

def test_updates_flight_plan_decrements():
    board = create_board(size=31)

    p = Point(10, 11)
    f = Fleet("test", 10, Direction.SOUTH, p, 100, "8N", 0, board)

    board._add_fleet(f)

    next_board = board.next()
    next_fleet = next_board.get_fleet_at_point(p + Direction.SOUTH.to_point())
    assert Direction.SOUTH.to_char() == next_fleet.direction.to_char(), "should not change direction"
    assert "7N" == next_fleet.flight_plan, "should update flight plan"

def test_updates_flight_plan_changes_direction():
    board = create_board(size=31)

    p = Point(10, 11)
    f = Fleet("test", 10, Direction.NORTH, p, 100, "S", 0, board)

    board._add_fleet(f)

    next_board = board.next()
    next_fleet = next_board.get_fleet_at_point(p + Direction.SOUTH.to_point())
    assert Direction.SOUTH.to_char() == next_fleet.direction.to_char(), "should change direction"
    assert "" == next_fleet.flight_plan, "should update flight plan"

def test_updates_flight_plan_converts_to_shipyard():
    board = create_board(size=31)

    p = Point(10, 11)
    f = Fleet("test", 100, Direction.NORTH, p, 100, "C", 0, board)

    board._add_fleet(f)

    next_board = board.next()
    shipyard = next_board.get_shipyard_at_point(p)
    assert shipyard is not None, "should have built a shipyard"
    assert shipyard.player_id == f.player_id, "should belong to the player"
    assert shipyard.ship_count == 50, "should have the right number of ships"

def test_adjacent_combat_dumps_all_kore_when_both_die():
    board = create_board(size=31)

    p1 = Point(10, 11)
    f1 = Fleet("f1", 100, Direction.NORTH, p1, 100, "", 0, board)
    p1_kore = board.cells.get(p1 + Direction.NORTH.to_point()).kore
    board._add_fleet(f1)

    p2 = p1 + Direction.NORTH.to_point() + Direction.NORTH.to_point() + Direction.EAST.to_point()
    f2 = Fleet("f2", 100, Direction.SOUTH, p2, 100, "", 1, board)
    board._add_fleet(f2)

    next_board = board.next()
    f1_next = next_board.fleets.get("f1")
    p1_kore_next = next_board.cells.get(p1 + Direction.NORTH.to_point()).kore
    assert f1_next is None, "should have been destroyed"
    assert p1_kore + 100 < p1_kore_next, "should dump all kore"

def test_adjacent_combat_dumps_half_kore_when_one_dies():
    board = create_board(size=31)

    p1 = Point(10, 11)
    f1 = Fleet("f1", 50, Direction.NORTH, p1, 100, "", 0, board)
    p1_kore = board.cells.get(p1 + Direction.NORTH.to_point()).kore
    board._add_fleet(f1)

    p2 = p1 + Direction.NORTH.to_point() + Direction.NORTH.to_point() + Direction.EAST.to_point()
    f2 = Fleet("f2", 100, Direction.SOUTH, p2, 100, "", 1, board)
    board._add_fleet(f2)

    next_board = board.next()
    f1_next = next_board.fleets.get("f1")
    p1_kore_next = next_board.cells.get(p1 + Direction.NORTH.to_point()).kore
    assert f1_next is None, "should have been destroyed"
    assert p1_kore + 50 < p1_kore_next and p1_kore + 55 > p1_kore_next, "should dump half kore"

    f2_next = next_board.fleets.get("f2")
    assert f2.kore + 50 <= f2_next.kore and f2.kore + 55 > f2_next.kore, "Should have picked up half"

def test_updates_flight_plan_does_not_convert_if_not_enough_ships():
    board = create_board(size=31)

    p = Point(10, 11)
    f = Fleet("test", 10, Direction.NORTH, p, 100, "C", 0, board)

    board._add_fleet(f)

    next_board = board.next()
    assert board.get_shipyard_at_point(p) is None, "Should not have made a shipyard"
    next_fleet = next_board.get_fleet_at_point(p + Direction.NORTH.to_point())
    assert next_fleet is not None, "should have kept going"
    assert f.id == next_fleet.id, "should have the same id"
    assert "" == next_fleet.flight_plan, "should update flight plan"

def test_updates_flight_plan_works_with_two_converts():
    board = create_board(size=31)

    p = Point(10, 11)
    f = Fleet("test", 10, Direction.NORTH, p, 100, "SCCN", 0, board)

    board._add_fleet(f)

    board = board.next()
    board = board.next()
    board = board.next()


def create_board(size=21, starting_kore=100, agent_count=2, random_seed=0):
    env = make("kore_fleets", configuration={
        "size": size,
        "startingKore": starting_kore,
        "randomSeed": random_seed
    })
    return Board(env.reset(agent_count)[0].observation, env.configuration)

