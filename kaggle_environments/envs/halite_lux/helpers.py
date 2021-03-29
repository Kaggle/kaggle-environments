import json
from os import path
from kaggle_environments.helpers import *


class Observation(Observation):
    """
    Observation primarily used as a helper to construct the State from the raw observation.
    This provides bindings for the observation type described at https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/rps/rps.json
    """
    @property
    def last_opponent_action(self) -> int:
        """Move the opponent took on the last turn."""
        return self["halite"]


class Configuration(Configuration):
    """
    Configuration provides access to tunable parameters in the environment.
    This provides bindings for the configuration type described at https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/rps/rps.json
    """
    @property
    def signs(self) -> int:
        """Number of choices each step (3 for the normal rock, paper, scissors)"""
        return self["signs"]
