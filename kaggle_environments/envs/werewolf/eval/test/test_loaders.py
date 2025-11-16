import os
from pathlib import Path

from kaggle_environments.envs.werewolf.eval import loaders

DIR_PATH = Path(os.path.dirname(os.path.abspath(__file__)))



def test_load_games():
    input_dir = DIR_PATH / "data" / "w_replace"
    games = loaders.get_games(input_dir)

    game = loaders.GameResult(games[0])

    assert 0


