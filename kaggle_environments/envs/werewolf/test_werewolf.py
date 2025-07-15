
import pytest

from kaggle_environments import make

"""
{
  "roles": ["WEREWOLF", "VILLAGER", "VILLAGER", "SEER", "DOCTOR"],
  "names": ["gpt-4o", "claude-3", "gemini-pro", "player-4", "player-5"],
  "player_thumbnails": {
    "gpt-4o": "https://images.seeklogo.com/logo-png/46/1/chatgpt-logo-png_seeklogo-465219.png",
    "claude-3": "https://images.seeklogo.com/logo-png/55/1/claude-logo-png_seeklogo-554534.png",
    "gemini-pro": "https://logos-world.net/wp-content/uploads/2025/01/Google-Gemini-Symbol.png"
  }
}
"""


URLS = {
    "gemini": "https://logos-world.net/wp-content/uploads/2025/01/Google-Gemini-Symbol.png",
    "openai": "https://images.seeklogo.com/logo-png/46/1/chatgpt-logo-png_seeklogo-465219.png",
    "claude": "https://images.seeklogo.com/logo-png/55/1/claude-logo-png_seeklogo-554534.png",
    "grok": "https://images.seeklogo.com/logo-png/61/1/grok-logo-png_seeklogo-613403.png"
}


@pytest.fixture
def env():
    roles = ["Werewolf", "Werewolf", "Doctor", "Seer", "Villager", "Villager", "Villager"]
    env = make(
        'werewolf',
        debug=True,
        configuration={
            "roles": roles,
            "names": ["gemini-2.5-pro", "gemini-2.5-flash", "gpt-4.1", "o3", "o4-mini", "claude-4-sonnet", "grok-4"],
            "player_thumbnails": {
                "gemini-2.5-pro": URLS['gemini'],
                "gemini-2.5-flash": URLS['gemini'],
                "gpt-4.1": URLS['openai'],
                "o3": URLS['openai'],
                "o4-mini": URLS['openai'],
                "claude-4-sonnet": URLS['claude'],
                "grok-4": URLS['grok']
            }
        }
    )
    return env


def test_load_env(env):
    agents = ['random'] * 7
    env.run(agents)

    for i, state in enumerate(env.steps):
        env.render_step_ind = i
        out = env.renderer(state, env)
        print(out)


def test_default_env():
    env = make('werewolf', debug=True)
    agents = ['random'] * 8
    env.run(agents)



def test_run_dummy_llm():
    env = make('werewolf', debug=True)
    agents = ['dummy_llm'] * 7
    env.run(agents)


def test_html_render(env):
    agents = ['random'] * 7
    env.run(agents)
    env.render(mode='html')
