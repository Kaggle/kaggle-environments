import sys
from kaggle_environments import make
import open_spiel as open_spiel_env


def test_envs_load():
    envs = open_spiel_env._register_open_spiel_envs()
    print(len(envs))


def test_tic_tac_toe_playthrough():
    envs = open_spiel_env._register_open_spiel_envs(["tic_tac_toe"])
    print(envs)
    env = make("open_spiel_tic_tac_toe", debug=True)
    env.run(["random", "random", "game_master"])
    json = env.toJSON()
    assert json["name"] == "open_spiel_tic_tac_toe"
    assert all([status == "DONE" for status in json["statuses"]])