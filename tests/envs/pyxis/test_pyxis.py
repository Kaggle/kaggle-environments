from kaggle_environments import make


def _do_nothing(observation, configuration):
    """Isolated submission (standard Kaggle API): obs in, action out."""
    return {}


def _illegal_investor(observation, configuration):
    """Submission that ignores the action mask and over-invests everywhere."""
    inv_mask = observation["actionMask"].get("investments", [])
    return {"investments": [3 for _ in inv_mask]}


def test_pyxis_completes():
    env = make("pyxis", configuration={"seed": 1})
    env.run(["knapsack", "random"])
    j = env.toJSON()
    assert j["name"] == "pyxis"
    assert j["statuses"] == ["DONE", "DONE"]
    rewards = [s["reward"] for s in env.steps[-1]]
    assert set(rewards) <= {0.0, 0.5, 1.0}
    # Exactly one winner (or a draw), never two winners.
    assert sum(rewards) == 1.0


def test_pyxis_seed_is_deterministic_and_hidden():
    def outcome(seed):
        env = make("pyxis", configuration={"seed": seed})
        env.run(["knapsack", "random"])
        return [s["reward"] for s in env.steps[-1]]

    assert outcome(4242) == outcome(4242)

    # The seed is scrubbed from configuration so agents cannot read it, but is
    # persisted on env.info for the replay.
    env = make("pyxis", configuration={"seed": 4242})
    env.run(["do_nothing", "do_nothing"])
    assert env.configuration["seed"] is None
    assert env.info["seed"] == 4242
    assert "seed" not in env.steps[0][0]["observation"]


def test_pyxis_do_nothing_draws():
    env = make("pyxis", configuration={"seed": 7})
    env.run(["do_nothing", "do_nothing"])
    assert [s["reward"] for s in env.steps[-1]] == [0.5, 0.5]


def test_pyxis_isolated_submission_plays():
    env = make("pyxis", configuration={"seed": 9})
    env.run([_do_nothing, "knapsack"])
    assert env.steps[-1][0]["status"] == "DONE"
    assert env.steps[-1][1]["status"] == "DONE"


def test_pyxis_illegal_action_forfeits():
    env = make("pyxis", configuration={"seed": 3})
    env.run([_illegal_investor, "do_nothing"])
    last = env.steps[-1]
    assert last[0]["status"] == "INVALID"
    assert last[0]["reward"] is None
    assert last[1]["status"] == "DONE"
    assert last[1]["reward"] == 1.0


def test_pyxis_partial_action_is_filled_with_noops():
    """An agent may send only the heads it uses; the rest default to no-ops."""

    def only_investments(observation, configuration):
        return {"investments": [0 for _ in observation["actionMask"]["investments"]]}

    def null_heads(observation, configuration):
        return {"investments": None, "upgrade": None}

    for agent in (only_investments, null_heads):
        env = make("pyxis", configuration={"seed": 9})
        env.run([agent, "knapsack"])
        assert [s["status"] for s in env.steps[-1]] == ["DONE", "DONE"], agent.__name__


def test_pyxis_malformed_action_forfeits_rather_than_draws():
    """A malformed action must not be a cheap escape from a losing position."""
    for bad in ([1, 2, 3], {"bogus_head": 1}):
        env = make("pyxis", configuration={"seed": 3})
        env.run([lambda o, c, b=bad: b, "do_nothing"])
        last = env.steps[-1]
        assert last[0]["status"] == "INVALID", bad
        assert last[1]["reward"] == 1.0, bad


def test_pyxis_every_masked_head_is_validated():
    """Mask violations forfeit on all discrete heads, not just investments."""
    # Choices outside the head's range, which every mask forbids on step 1.
    violations = {
        "investments": 4,
        "ptrs_research": 10,
        "upgrade": 1,
    }
    for head, choice in violations.items():

        def agent(observation, configuration, h=head, c=choice):
            mask = observation["actionMask"][h]
            return {h: c if isinstance(mask[0], bool) else [c for _ in mask]}

        env = make("pyxis", configuration={"seed": 3})
        env.run([agent, "do_nothing"])
        last = env.steps[-1]
        assert last[0]["status"] == "INVALID", head
        assert last[1]["reward"] == 1.0, head


def test_pyxis_wrong_shape_action_forfeits():
    """Continuous heads are unmasked, so the action space is the validator."""
    for bad in ({"investments": [0] * 5}, {"bd_bids": [-5.0, -5.0, -5.0]}, {"site_bid": [1e12]}):
        env = make("pyxis", configuration={"seed": 3})
        env.run([lambda o, c, b=bad: b, "do_nothing"])
        assert env.steps[-1][0]["status"] == "INVALID", bad
