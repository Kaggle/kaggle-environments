from kaggle_environments import make
from kaggle_environments.envs.cabt.cabt import random_agent, first_agent, deck


def test_cabt_inits():
    """
    Test that a game runs to completion with random agents.
    """
    env = make("cabt", debug=True)
    env.run(["random", "random"])
    json = env.toJSON()
    assert json["name"] == "cabt"
    assert json["statuses"] == ["DONE", "DONE"]
    # Rewards can be [1, -1], [-1, 1], or [0, 0].
    assert sorted(json["rewards"]) in [[-1, 1], [0, 0]]


def test_cabt_first_agent_run():
    """
    Test that a game runs to completion with first_agent.
    """
    env = make("cabt", debug=True)
    env.run(["first", "first"])
    json = env.toJSON()
    assert json["name"] == "cabt"
    assert json["statuses"] == ["DONE", "DONE"]
    assert sorted(json["rewards"]) in [[-1, 1], [0, 0]]


def test_cabt_random_vs_first():
    """
    Test a game between a random and a first agent.
    """
    env = make("cabt", debug=True)
    env.run(["random", "first"])
    json = env.toJSON()
    assert json["name"] == "cabt"
    assert json["statuses"] == ["DONE", "DONE"]
    assert sorted(json["rewards"]) in [[-1, 1], [0, 0]]


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
    json = env.toJSON()
    assert json["statuses"] == ["INVALID", "DONE"]
    assert json["rewards"] == [None, None]
    assert "deck does not have 60 cards" in json["steps"][0][0]["error"]


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
    json = env.toJSON()

    assert sorted(json["statuses"]) == ["DONE", "INVALID"]
    if json["statuses"][0] == "INVALID":
        assert json["rewards"] == [-1, 1]
    else:
        assert json["rewards"] == [1, -1]
