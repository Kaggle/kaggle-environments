import json

import pytest

from kaggle_environments import make

URLS = {
    "gemini": "https://storage.googleapis.com/kaggle-static/game-arena/werewolf/thumbnails/gemini.png",
    "openai": "https://images.seeklogo.com/logo-png/46/1/chatgpt-logo-png_seeklogo-465219.png",
    "claude": "https://images.seeklogo.com/logo-png/55/1/claude-logo-png_seeklogo-554534.png",
    "grok": "https://images.seeklogo.com/logo-png/61/1/grok-logo-png_seeklogo-613403.png",
}


@pytest.fixture
def agents_config():
    roles = ["Werewolf", "Werewolf", "Doctor", "Seer", "Villager", "Villager", "Villager"]
    names = [f"player_{i}" for i in range(len(roles))]
    thumbnails = [
        URLS["gemini"],
        URLS["gemini"],
        URLS["openai"],
        URLS["openai"],
        URLS["openai"],
        URLS["claude"],
        URLS["grok"],
    ]
    agents_config = [
        {"role": role, "id": name, "agent_id": "random", "thumbnail": url}
        for role, name, url in zip(roles, names, thumbnails)
    ]
    return agents_config


@pytest.fixture
def env(agents_config):
    env = make("werewolf", debug=True, configuration={"agents": agents_config})
    return env


def test_load_env(env):
    agents = ["random"] * 7
    env.run(agents)

    for i, state in enumerate(env.steps):
        env.render_step_ind = i
        env.renderer(state, env)


def test_randomize_role_with_seed(agents_config):
    env = make("werewolf", debug=True, configuration={"agents": agents_config, "randomize_roles": True, "seed": 123})
    agents = ["random"] * 7
    env.run(agents)

    for i, state in enumerate(env.steps):
        env.render_step_ind = i
        env.renderer(state, env)


def test_discussion_protocol(agents_config):
    env = make(
        "werewolf",
        debug=True,
        configuration={
            "agents": agents_config,
            "discussion_protocol": {"name": "RoundRobinDiscussion", "params": {"max_rounds": 2}},
        },
    )
    agents = ["random"] * 7
    env.run(agents)
    env.toJSON()


def test_no_reveal_options(agents_config):
    env = make(
        "werewolf",
        debug=True,
        configuration={
            "agents": agents_config,
            "night_elimination_reveal_level": "no_reveal",
            "day_exile_reveal_level": "no_reveal",
        },
    )
    agents = ["random"] * 7
    env.run(agents)
    env.toJSON()


def test_disable_doctor_self_save():
    roles = ["Werewolf", "Werewolf", "Doctor", "Seer", "Villager", "Villager", "Villager"]
    names = [f"player_{i}" for i in range(len(roles))]
    thumbnails = [
        URLS["gemini"],
        URLS["gemini"],
        URLS["openai"],
        URLS["openai"],
        URLS["openai"],
        URLS["claude"],
        URLS["grok"],
    ]
    agents_config = [
        {"role": role, "id": name, "agent_id": "random", "thumbnail": url}
        for role, name, url in zip(roles, names, thumbnails)
    ]
    agents_config[2]["role_params"] = {"allow_self_save": False}
    env = make(
        "werewolf",
        debug=True,
        configuration={
            "agents": agents_config,
        },
    )
    agents = ["random"] * 7
    env.run(agents)
    env.toJSON()


def test_turn_by_turn_bidding_discussion(agents_config):
    """Tests the bidding -> chat -> bidding -> chat ... cycle."""
    env = make(
        "werewolf",
        debug=True,
        configuration={
            "agents": agents_config,
            "discussion_protocol": {
                "name": "TurnByTurnBiddingDiscussion",
                "params": {
                    "bidding": {
                        "name": "UrgencyBiddingProtocol",
                    },
                    "max_turns": 16,
                    "bid_result_public": False,
                },
            },
        },
    )
    agents = ["random"] * 7
    env.run(agents)


@pytest.mark.skip("Slow test, meant for manual testing.")
def test_llm_players(agents_config):
    env = make("werewolf", debug=True, configuration={"actTimeout": 30, "agents": agents_config})
    agents = [
        "llm/gemini/gemini-2.5-flash",
        "random",
        "llm/gemini/gemini-2.5-flash",
        "llm/gemini/gemini-2.5-flash",
        "llm/gemini/gemini-2.5-flash",
        "random",
        "random",
    ]
    env.run(agents)
    for i, state in enumerate(env.steps):
        env.render_step_ind = i
        env.renderer(state, env)


def test_default_env():
    env = make("werewolf", debug=True)
    agents = ["random"] * 7
    env.run(agents)


def test_env_info_not_overwritten(agents_config):
    initial_info = {"Agents": [{"Name": "External Agent Info"}], "custom_key": "custom_value"}
    env = make("werewolf", debug=True, configuration={"agents": agents_config}, info=initial_info)
    agents = ["random"] * 7
    env.run(agents)

    assert env.info.get("custom_key") == "custom_value"
    assert env.info.get("Agents")[0]["Name"] == "External Agent Info"
    # Check if werewolf specific keys are added (MODERATOR_OBS)
    # We assume MODERATOR_OBS is added during initialization
    assert len(env.info) > 2

    # Dump and load back
    json_str = json.dumps(env.toJSON())
    loaded_info = json.loads(json_str)["info"]

    assert loaded_info.get("custom_key") == "custom_value"
    assert loaded_info.get("Agents")[0]["Name"] == "External Agent Info"
    assert "MODERATOR_OBSERVATION" in loaded_info
