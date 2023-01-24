"""
Wrappers that allow users to insert heuristics into the environment reset and step functions
"""
from typing import Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces

import luxai_s2.env
from luxai_s2.env import LuxAI_S2
from luxai_s2.state import ObservationStateDict
from luxai_s2.utils import my_turn_to_place_factory
from luxai_s2.wrappers.controllers import (
    Controller,
    SimpleDiscreteController,
    SimpleSingleUnitDiscreteController,
)


class FactoryControlWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def step(self, action):
        return super().step(action)
