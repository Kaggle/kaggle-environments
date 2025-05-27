
from kaggle_environments import make


def test_load_env():
    env = make('werewolf', debug=True)
    agents = ['random'] * 7
    env.run(agents)

    for i, state in enumerate(env.steps):
        env.render_step_ind = i
        out = env.renderer(state, env)
        print(out)


def test_html_render():
    env = make('werewolf', debug=True)
    agents = ['random'] * 7
    env.run(agents)
    env.render(mode='html')
