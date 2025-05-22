
from kaggle_environments import make


def test_load_env():
    env = make('werewolf', debug=True)
    agents = ['random'] * 7
    env.run(agents)
    # env.render(mode=)