"""Upstream-style tests (run once this env lives in kaggle_environments/envs/).

In this repo the env is not registered with the installed package, so these are
exercised via tests/test_contrib_env.py instead; here they document the expected
behaviour in the upstream test layout.
"""

from kaggle_environments import make


def test_runs_and_terminates():
    env = make("weeping_angel", configuration={"seed": 0})
    env.run(["inference_blue", "mixed_red"])
    assert env.state[0].status == "DONE"
    assert env.state[1].status == "DONE"


def test_reward_is_zero_sum():
    env = make("weeping_angel", configuration={"seed": 2})
    final = env.run(["sweep_blue", "rush_red"])[-1]
    assert final[0].reward == -final[1].reward


def test_hidden_information():
    env = make("weeping_angel", configuration={"seed": 3})
    final = env.run(["inference_blue", "mixed_red"])[-1]
    assert final[0].observation.angels == []  # blue never sees Angel placement
    assert final[1].observation.coverageLog == []  # red never sees coverage
