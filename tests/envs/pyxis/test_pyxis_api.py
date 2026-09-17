"""API / integration smoke tests for the pyxis environment.

These cover the ways users actually reach the environment (make/run/evaluate,
the train() gym loop, isolated submissions) and the JSON replay that downstream
tooling consumes. Gameplay-outcome tests live in test_pyxis.py.
"""

import json

from kaggle_environments import evaluate, make


def _noop(observation, configuration):
    """Isolated submission (standard Kaggle API) that always no-ops."""
    return {}


def test_pyxis_default_config_runs():
    # No seed supplied: the default (None) path must still produce a full match.
    env = make("pyxis")
    env.run(["do_nothing", "do_nothing"])
    assert env.done
    assert [s["status"] for s in env.steps[-1]] == ["DONE", "DONE"]


def test_pyxis_reset_populates_observation():
    env = make("pyxis")
    obs = env.state[0].observation
    assert isinstance(obs.obs, list) and len(obs.obs) > 0
    assert "investments" in obs.actionMask
    assert isinstance(obs.cash, (int, float))
    assert isinstance(obs.enpv, (int, float))
    assert isinstance(obs.bankrupt, bool)


def test_pyxis_agent_index_and_shared_env_id():
    env = make("pyxis")
    assert env.state[0].observation.agentIndex == 0
    assert env.state[1].observation.agentIndex == 1
    # kaggleEnvId is a shared field stored on seat 0 and equal to the env id.
    assert env.state[0].observation.kaggleEnvId == env.id
    assert env.id


def test_pyxis_builtin_agents_and_spec_registered():
    env = make("pyxis")
    assert set(env.agents) == {"knapsack", "random", "do_nothing"}
    assert env.specification.agents == [2]


def test_pyxis_evaluate_returns_rewards():
    rewards = evaluate("pyxis", ["knapsack", "do_nothing"], {"seed": 11})
    assert len(rewards) == 1
    (row,) = rewards
    assert len(row) == 2
    assert set(row) <= {0.0, 0.5, 1.0}
    assert sum(row) == 1.0


def test_pyxis_agent_observation_hides_internal_flags():
    seen = {}

    def capture(observation, configuration):
        seen.update(observation)
        return {}

    env = make("pyxis", configuration={"seed": 8})
    env.run([capture, "do_nothing"])
    # Submissions get the shared engine id but never the internal init flag.
    assert "kaggleEnvId" in seen
    assert "initialized" not in seen
    assert all(k in seen for k in ("obs", "actionMask", "agentIndex"))


def test_pyxis_two_isolated_submissions_play():
    # The common submission shape: both seats are isolated obs->action callables.
    env = make("pyxis", configuration={"seed": 12})
    env.run([_noop, _noop])
    assert [s["status"] for s in env.steps[-1]] == ["DONE", "DONE"]
    assert [s["reward"] for s in env.steps[-1]] == [0.5, 0.5]


def test_pyxis_toJSON_is_serializable():
    env = make("pyxis", configuration={"seed": 13})
    env.run(["do_nothing", "do_nothing"])
    # The replay must round-trip through JSON (no numpy leaking into the state).
    dumped = json.dumps(env.toJSON())
    assert json.loads(dumped)["name"] == "pyxis"


def test_pyxis_toJSON_replay_structure():
    env = make("pyxis", configuration={"seed": 14})
    env.run(["knapsack", "do_nothing"])
    j = env.toJSON()
    assert j["name"] == "pyxis"
    assert isinstance(j["steps"], list) and len(j["steps"]) >= 2
    # Every step records both seats with the fields a viewer renders.
    for step in j["steps"]:
        assert len(step) == 2
        for agent in step:
            for key in ("obs", "cash", "enpv", "bankrupt"):
                assert key in agent["observation"]
    assert j["rewards"] == [s["reward"] for s in env.steps[-1]]
    assert j["statuses"] == ["DONE", "DONE"]


def test_pyxis_train_api_steps_to_done():
    # The gym-style single-agent loop used for RL training.
    env = make("pyxis", configuration={"seed": 15})
    trainer = env.train([None, "do_nothing"])
    obs = trainer.reset()
    assert isinstance(obs.obs, list) and len(obs.obs) > 0
    done, steps = False, 0
    while not done and steps < env.configuration.episodeSteps:
        _, _, done, _ = trainer.step({})
        steps += 1
    assert done
    assert steps > 0
