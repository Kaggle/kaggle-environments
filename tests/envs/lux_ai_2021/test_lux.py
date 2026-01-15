from kaggle_environments import make
from kaggle_environments.envs.lux_ai_2021.agents import js_simple_agent, random_agent, simple_agent


def test_lux_completes():
    env = make("lux_ai_2021", configuration={})
    env.run([random_agent, simple_agent])
    json = env.toJSON()
    assert json["name"] == "lux_ai_2021"
    assert json["statuses"] == ["DONE", "DONE"]


def test_js_agents():
    env = make("lux_ai_2021", configuration={})
    env.run([simple_agent, js_simple_agent])
    json = env.toJSON()
    assert json["name"] == "lux_ai_2021"
    assert json["statuses"] == ["DONE", "DONE"]
