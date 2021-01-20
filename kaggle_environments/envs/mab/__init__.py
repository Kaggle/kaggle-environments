from kaggle_environments import register
from .mab import MabEnvironment

register("mab", MabEnvironment())