from kaggle_environments import make
from kaggle_environments.envs.cabt.cabt import deck, first_agent, random_agent
from kaggle_environments.envs.cabt.cg.sim import Battle, get_lib


def test_cabt_inits():
    """
    Test that a game runs to completion with random agents.
    """
    env = make("cabt", debug=True)
    env.run(["random", "random"])
    env_map = env.toJSON()
    assert env_map["name"] == "cabt"
    assert env_map["statuses"] == ["DONE", "DONE"]
    assert sorted(env_map["rewards"]) in [[-1, 1], [0, 0]]


def test_cabt_first_agent_run():
    """
    Test that a game runs to completion with first_agent.
    """
    env = make("cabt", debug=True)
    env.run(["first", "first"])
    env_map = env.toJSON()
    assert env_map["name"] == "cabt"
    assert env_map["statuses"] == ["DONE", "DONE"]
    assert sorted(env_map["rewards"]) in [[-1, 1], [0, 0]]


def test_cabt_random_vs_first():
    """
    Test a game between a random and a first agent.
    """
    env = make("cabt", debug=True)
    env.run(["random", "first"])
    env_map = env.toJSON()
    assert env_map["name"] == "cabt"
    assert env_map["statuses"] == ["DONE", "DONE"]
    assert sorted(env_map["rewards"]) in [[-1, 1], [0, 0]]


def test_random_agent_deck_submission():
    """
    Test random_agent's deck submission phase.
    """
    obs = {"select": None}
    action = random_agent(obs)
    assert action == deck


def test_random_agent_selection():
    """
    Test random_agent's selection phase.
    """
    obs = {"select": {"option": ["a", "b", "c", "d"], "maxCount": 2}}
    action = random_agent(obs)
    assert isinstance(action, list)
    assert len(action) == 2
    assert all(isinstance(i, int) for i in action)
    assert all(0 <= i < 4 for i in action)
    assert len(set(action)) == 2  # no duplicates


def test_first_agent_deck_submission():
    """
    Test first_agent's deck submission phase.
    """
    obs = {"select": None}
    action = first_agent(obs)
    assert action == deck


def test_first_agent_selection():
    """
    Test first_agent's selection phase.
    """
    obs = {"select": {"option": ["a", "b", "c", "d"], "maxCount": 2}}
    action = first_agent(obs)
    assert action == [0, 1]


def invalid_deck_agent(obs, config):
    """An agent that submits an invalid deck."""
    if obs["select"] is None:
        return [1, 2, 3]  # Invalid deck (not 60 cards)
    return [0]


def test_invalid_deck():
    """
    Test the interpreter's handling of an invalid deck submission.
    """
    env = make("cabt", debug=True)
    env.run([invalid_deck_agent, "random"])
    env_map = env.toJSON()
    assert env_map["statuses"] == ["INVALID", "DONE"]
    assert env_map["rewards"] == [None, 0]
    assert "deck does not have 60 cards" in env_map["steps"][0][0]["error"]


def invalid_selection_agent(obs, config):
    """An agent that makes an invalid selection."""
    if obs["select"] is None:
        return deck
    # Return more items than maxCount, which may cause an error in battle_select
    return list(range(obs["select"]["maxCount"] + 1))


def test_invalid_selection():
    """
    Test the interpreter's handling of an invalid selection during a turn.
    """
    env = make("cabt", debug=True)
    env.run([invalid_selection_agent, "random"])
    env_map = env.toJSON()

    assert sorted(env_map["statuses"]) == ["DONE", "INVALID"]
    if env_map["statuses"][0] == "INVALID":
        assert env_map["rewards"] == [None, 1]
    else:
        assert env_map["rewards"] == [1, None]


def test_cabt_version_1_plays_bo1():
    """Version 1 uses the legacy engine and ignores bo."""
    env = make("cabt", configuration={"version": 1, "bo": 3}, debug=True)
    env.run(["random", "random"])
    assert env.toJSON()["statuses"] == ["DONE", "DONE"]
    assert sum(env.result) == 1
    assert not hasattr(Battle.lib, "BattleStartReverse")


def test_cabt_versions_alternate_in_process():
    """Both engines can run in the same process."""
    for version in [2, 1, 2]:
        env = make("cabt", configuration={"version": version}, debug=True)
        env.run(["random", "first"])
        assert env.toJSON()["statuses"] == ["DONE", "DONE"]
        assert Battle.lib is get_lib(version)
