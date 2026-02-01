from kaggle_environments import evaluate, make

env = None


def before_each(configuration=None):
    global env
    if configuration is None:
        configuration = {"mapSeed": 42}
    elif "mapSeed" not in configuration:
        configuration["mapSeed"] = 42
    env = make("reinforce_tactics", configuration=configuration, debug=False)


# ---- Basic lifecycle tests ----


def test_can_create():
    before_each()
    assert env is not None
    assert env.configuration.mapWidth == 20
    assert env.configuration.mapHeight == 20


def test_has_correct_timeouts():
    before_each()
    assert env.configuration.actTimeout == 5
    assert env.configuration.runTimeout == 1200


def test_to_json():
    before_each()
    json = env.toJSON()
    assert json["name"] == "reinforce_tactics"
    assert json["rewards"] == [0, 0]
    assert json["statuses"] == ["ACTIVE", "INACTIVE"]


def test_can_reset():
    before_each()
    state = env.reset()
    assert len(state) == 2
    assert state[0]["status"] == "ACTIVE"
    assert state[1]["status"] == "INACTIVE"
    assert len(state[0]["observation"]["board"]) == 20
    assert len(state[0]["observation"]["board"][0]) == 20
    assert state[0]["observation"]["gold"] == [250, 250]
    assert state[0]["observation"]["player"] == 0
    assert state[1]["observation"]["player"] == 1


def test_defaults_to_beginner_map():
    """When no mapName is provided, the beginner map should be used."""
    before_each()
    assert env.configuration.mapName == "beginner"


def test_random_map_has_ocean_default():
    """Verify a randomly generated map uses ocean as the default tile."""
    before_each(configuration={"mapName": "", "mapSeed": 42})
    state = env.reset()
    board = state[0]["observation"]["board"]
    flat = [cell for row in board for cell in row]
    assert flat.count("o") > 0


def test_board_has_hq_and_buildings():
    """Verify the map contains HQ and building structures for both players."""
    before_each()
    state = env.reset()
    structures = state[0]["observation"]["structures"]
    types = {s["type"] for s in structures}
    assert "h" in types, "Map should contain headquarters"
    assert "b" in types, "Map should contain buildings"
    owners = {s["owner"] for s in structures if s["type"] == "h"}
    assert 1 in owners, "Player 1 should have a headquarters"
    assert 2 in owners, "Player 2 should have a headquarters"


# ---- Running with built-in agents ----


def test_can_run_random_agents():
    before_each(configuration={"mapSeed": 42, "episodeSteps": 20})
    result = env.run(["random", "random"])
    final = result[-1]
    assert final[0].status == "DONE"
    assert final[1].status == "DONE"


def test_can_run_aggressive_agents():
    before_each(configuration={"mapSeed": 42, "episodeSteps": 30})
    result = env.run(["aggressive", "aggressive"])
    final = result[-1]
    assert final[0].status == "DONE"
    assert final[1].status == "DONE"


def test_draw_on_max_steps():
    """Game should end in a draw when max steps reached."""
    before_each(configuration={"mapSeed": 42, "episodeSteps": 5})
    result = env.run(["random", "random"])
    final = result[-1]
    assert final[0].status == "DONE"
    assert final[1].status == "DONE"
    assert final[0].reward == 0
    assert final[1].reward == 0


# ---- Rendering ----


def test_can_render():
    before_each()
    env.reset()
    rendered = env.render(mode="ansi")
    assert "Turn" in rendered
    assert "Gold" in rendered


# ---- Configuration ----


def test_custom_starting_gold():
    before_each(configuration={"mapSeed": 42, "startingGold": 500})
    state = env.reset()
    assert state[0]["observation"]["gold"] == [500, 500]


def test_fog_of_war_config():
    before_each(configuration={"mapSeed": 42, "fogOfWar": True})
    state = env.reset()
    # With fog of war, units observation should only show own units
    # (no units at start, so both should be empty)
    assert isinstance(state[0]["observation"]["units"], list)


def test_enabled_units_config():
    """Restricting enabled units should still work."""
    before_each(configuration={"mapSeed": 42, "enabledUnits": "W,A"})
    state = env.reset()
    assert state[0]["status"] == "ACTIVE"


# ---- Built-in maps ----


def test_builtin_map_beginner():
    before_each(configuration={"mapName": "beginner"})
    state = env.reset()
    board = state[0]["observation"]["board"]
    assert len(board) == 20
    assert len(board[0]) == 20


def test_builtin_map_crossroads():
    before_each(configuration={"mapName": "crossroads"})
    state = env.reset()
    assert state[0]["status"] == "ACTIVE"


def test_builtin_map_tower_rush():
    before_each(configuration={"mapName": "tower_rush"})
    state = env.reset()
    assert state[0]["status"] == "ACTIVE"


def test_unknown_map_falls_back_to_random():
    """An unrecognised mapName should fall back to random generation."""
    before_each(configuration={"mapName": "nonexistent_map", "mapSeed": 42})
    state = env.reset()
    assert state[0]["status"] == "ACTIVE"
    assert len(state[0]["observation"]["board"]) == 20


def test_run_on_builtin_map():
    before_each(configuration={"mapName": "beginner", "episodeSteps": 20})
    result = env.run(["random", "random"])
    final = result[-1]
    assert final[0].status == "DONE"
    assert final[1].status == "DONE"


# ---- Evaluate ----


def test_can_evaluate():
    rewards = evaluate(
        "reinforce_tactics",
        ["random", "random"],
        num_episodes=2,
        configuration={"mapSeed": 42, "episodeSteps": 10},
    )
    assert len(rewards) == 2
    for r in rewards:
        assert r[0] + r[1] == 0


# ---- Action handling ----


def test_invalid_action_dict_loses():
    """An agent returning an action list with an invalid dict should lose."""
    def bad_agent(obs, config):
        return [{"type": "move", "from_x": -99, "from_y": -99, "to_x": -1, "to_y": -1}]

    before_each(configuration={"mapSeed": 42, "episodeSteps": 20})
    result = env.run([bad_agent, "random"])
    final = result[-1]
    assert final[0].reward == -1
    assert final[1].reward == 1
