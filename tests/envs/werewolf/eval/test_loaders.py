import os
from pathlib import Path

import pytest

from kaggle_environments.envs.werewolf.eval import loaders

DIR_PATH = Path(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skip("need to have local test data")
def test_load_games():
    input_dir = DIR_PATH / "data" / "w_replace"
    games = loaders.get_games(input_dir)

    # Just check that we can create the object.
    game = loaders.GameResult(games[0])  # noqa F841
