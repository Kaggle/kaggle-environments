import pytest

from kaggle_environments import make

URLS = {
    "gemini": "https://logos-world.net/wp-content/uploads/2025/01/Google-Gemini-Symbol.png",
    "openai": "https://images.seeklogo.com/logo-png/46/1/chatgpt-logo-png_seeklogo-465219.png",
    "claude": "https://images.seeklogo.com/logo-png/55/1/claude-logo-png_seeklogo-554534.png",
    "grok": "https://images.seeklogo.com/logo-png/61/1/grok-logo-png_seeklogo-613403.png"
}

@pytest.fixture
def agents_config():
    roles = ["Werewolf", "Werewolf", "Doctor", "Seer", "Villager", "Villager", "Villager"]
    names = [f"player_{i}" for i in range(len(roles))]
    thumbnails = [URLS['gemini'], URLS['gemini'], URLS['openai'], URLS['openai'], URLS['openai'], URLS['claude'],
                  URLS['grok']]
    agents_config = [{"role": role, "id": name, "agent_id": "random", "thumbnail": url} for role, name, url in
                     zip(roles, names, thumbnails)]
    return agents_config


@pytest.fixture
def env(agents_config):
    env = make(
        'werewolf',
        debug=True,
        configuration={
            "agents": agents_config
        }
    )
    return env


def test_load_env(env):
    agents = ['random'] * 7
    env.run(agents)

    for i, state in enumerate(env.steps):
        env.render_step_ind = i
        out = env.renderer(state, env)


def test_discussion_protocol(agents_config):
    env = make(
        'werewolf',
        debug=True,
        configuration={
            "agents": agents_config,
            "discussion_protocol": {
                "name": "RoundRobinDiscussion",
                "params": {
                    "max_rounds": 2
                }
            }
        }
    )
    agents = ['random'] * 7
    env.run(agents)
    out = env.toJSON()


def test_no_reveal_options(agents_config):
    env = make(
        'werewolf',
        debug=True,
        configuration={
            "agents": agents_config,
            "reveal_night_elimination_role": False,
            "reveal_day_exile_role": False
        }
    )
    agents = ['random'] * 7
    env.run(agents)
    out = env.toJSON()


def test_disable_doctor_self_save(agents_config):
    env = make(
        'werewolf',
        debug=True,
        configuration={
            "agents": agents_config,
            "allow_doctor_self_save": False
        }
    )
    agents = ['random'] * 7
    env.run(agents)
    out = env.toJSON()


@pytest.mark.skip('Slow test, meant for manual testing.')
def test_llm_players(agents_config):
    env = make(
        'werewolf',
        debug=True,
        configuration={
            "actTimeout": 30,
            "agents": agents_config
        }
    )
    agents = ['llm/gemini/gemini-2.5-flash', 'random', 'llm/gemini/gemini-2.5-flash', 'llm/gemini/gemini-2.5-flash',
              'llm/gemini/gemini-2.5-flash', 'random', 'random']
    env.run(agents)
    for i, state in enumerate(env.steps):
        env.render_step_ind = i
        out = env.renderer(state, env)


def test_default_env():
    env = make('werewolf', debug=True)
    agents = ['random'] * 7
    env.run(agents)


def test_html_render(env, tmp_path):
    agents = ['random'] * 7
    env.run(agents)
    content = env.render(mode='html', configuration={"allow_doctor_self_save": True})
    replay_file = tmp_path / "game_replay.html"
    with open(replay_file, 'w') as handle:
        handle.write(content)
    assert replay_file.exists()
