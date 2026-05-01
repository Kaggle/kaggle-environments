"""
Functional tests for the Reinforce Tactics Kaggle environment exercised
through the public ``kaggle_environments.make()`` API.
"""
# pylint: disable=missing-function-docstring,redefined-outer-name
import pytest

from kaggle_environments import evaluate, make


@pytest.fixture
def make_env():
    """Factory: build an env with mapSeed defaulting to 42 for determinism."""
    def _make(configuration=None):
        if configuration is None:
            configuration = {"mapSeed": 42}
        elif "mapSeed" not in configuration:
            configuration["mapSeed"] = 42
        return make("reinforce_tactics", configuration=configuration, debug=False)

    return _make


# ---- Basic lifecycle tests --------------------------------------------------


def test_can_create(make_env):
    env = make_env()
    assert env is not None
    assert env.configuration.mapWidth == 20
    assert env.configuration.mapHeight == 20


def test_has_correct_timeouts(make_env):
    env = make_env()
    assert env.configuration.actTimeout == 5
    assert env.configuration.runTimeout == 1200


def test_to_json(make_env):
    env = make_env()
    payload = env.toJSON()
    assert payload["name"] == "reinforce_tactics"
    assert payload["rewards"] == [0, 0]
    assert payload["statuses"] == ["ACTIVE", "INACTIVE"]


def test_can_reset(make_env):
    env = make_env()
    state = env.reset()
    assert len(state) == 2
    assert state[0]["status"] == "ACTIVE"
    assert state[1]["status"] == "INACTIVE"
    assert len(state[0]["observation"]["board"]) == 20
    assert len(state[0]["observation"]["board"][0]) == 20
    assert state[0]["observation"]["gold"] == [250, 250]
    assert state[0]["observation"]["player"] == 0
    assert state[1]["observation"]["player"] == 1


def test_defaults_to_beginner_map(make_env):
    env = make_env()
    assert env.configuration.mapName == "beginner"


def test_random_map_has_ocean_default(make_env):
    env = make_env(configuration={"mapName": "", "mapSeed": 42})
    state = env.reset()
    board = state[0]["observation"]["board"]
    flat = [cell for row in board for cell in row]
    assert flat.count("o") > 0


def test_board_has_hq_and_buildings(make_env):
    env = make_env()
    state = env.reset()
    structures = state[0]["observation"]["structures"]
    types = {s["type"] for s in structures}
    assert "h" in types, "Map should contain headquarters"
    assert "b" in types, "Map should contain buildings"
    owners = {s["owner"] for s in structures if s["type"] == "h"}
    assert 1 in owners
    assert 2 in owners


# ---- Running with built-in agents -------------------------------------------


def test_can_run_random_agents(make_env):
    env = make_env(configuration={"mapSeed": 42, "episodeSteps": 20})
    result = env.run(["random", "random"])
    final = result[-1]
    assert final[0].status == "DONE"
    assert final[1].status == "DONE"


def test_can_run_simple_bot_agents(make_env):
    env = make_env(configuration={"mapSeed": 42, "episodeSteps": 30})
    result = env.run(["simple_bot", "simple_bot"])
    final = result[-1]
    assert final[0].status == "DONE"
    assert final[1].status == "DONE"


def test_draw_on_max_steps(make_env):
    env = make_env(configuration={"mapSeed": 42, "episodeSteps": 5})
    result = env.run(["random", "random"])
    final = result[-1]
    assert final[0].status == "DONE"
    assert final[1].status == "DONE"
    assert final[0].reward == 0
    assert final[1].reward == 0


# ---- Rendering --------------------------------------------------------------


def test_can_render(make_env):
    env = make_env()
    env.reset()
    rendered = env.render(mode="ansi")
    assert "Turn" in rendered
    assert "Gold" in rendered


# ---- Configuration ----------------------------------------------------------


def test_custom_starting_gold(make_env):
    env = make_env(configuration={"mapSeed": 42, "startingGold": 500})
    state = env.reset()
    assert state[0]["observation"]["gold"] == [500, 500]


def test_fog_of_war_config(make_env):
    env = make_env(configuration={"mapSeed": 42, "fogOfWar": True})
    state = env.reset()
    assert isinstance(state[0]["observation"]["units"], list)


def test_enabled_units_config(make_env):
    env = make_env(configuration={"mapSeed": 42, "enabledUnits": "W,A"})
    state = env.reset()
    assert state[0]["status"] == "ACTIVE"


# ---- Built-in maps ----------------------------------------------------------


def test_builtin_map_beginner(make_env):
    env = make_env(configuration={"mapName": "beginner"})
    state = env.reset()
    board = state[0]["observation"]["board"]
    assert len(board) == 20
    assert len(board[0]) == 20


def test_builtin_map_crossroads(make_env):
    env = make_env(configuration={"mapName": "crossroads"})
    state = env.reset()
    assert state[0]["status"] == "ACTIVE"


def test_builtin_map_tower_rush(make_env):
    env = make_env(configuration={"mapName": "tower_rush"})
    state = env.reset()
    assert state[0]["status"] == "ACTIVE"


def test_unknown_map_falls_back_to_random(make_env):
    env = make_env(configuration={"mapName": "nonexistent_map", "mapSeed": 42})
    state = env.reset()
    assert state[0]["status"] == "ACTIVE"
    assert len(state[0]["observation"]["board"]) == 20


def test_run_on_builtin_map(make_env):
    env = make_env(configuration={"mapName": "beginner", "episodeSteps": 20})
    result = env.run(["random", "random"])
    final = result[-1]
    assert final[0].status == "DONE"
    assert final[1].status == "DONE"


# ---- Evaluate ---------------------------------------------------------------


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


# ---- Action handling --------------------------------------------------------


def test_invalid_action_dict_loses(make_env):
    """An agent returning an out-of-bounds move should lose immediately."""
    def bad_agent(_obs, _config):
        return [{"type": "move", "from_x": -99, "from_y": -99, "to_x": -1, "to_y": -1}]

    env = make_env(configuration={"mapSeed": 42, "episodeSteps": 20})
    result = env.run([bad_agent, "random"])
    final = result[-1]
    assert final[0].reward == -1
    assert final[1].reward == 1