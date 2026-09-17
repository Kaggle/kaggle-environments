import pickle

import numpy as np
import upath
from stable_baselines3.common.base_class import BaseAlgorithm

from pyxis_portfolio_challenge.file_io import download_file


class InferenceNormalizer:
    """
    A lightweight observation normalizer for inference.

    Extracts the necessary statistics from a saved VecNormalize object
    to normalize observations during inference.
    """

    def __init__(self, path: str):
        """Load normalization statistics from a pickled VecNormalize object."""
        with open(path, "rb") as file_handler:
            vec_normalize_data = pickle.load(file_handler)

        self.obs_rms = vec_normalize_data.obs_rms
        self.epsilon = 1e-8
        # SB3 defaults to clipping between [-10, 10], critical to load this
        self.clip_obs = getattr(vec_normalize_data, "clip_obs", 10.0)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize and clip the observation using loaded statistics."""
        norm_obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        # Apply the same clipping as training
        return np.clip(norm_obs, -self.clip_obs, self.clip_obs)


class PyxieAgent:
    """An investment agent using a trained RL policy from Stable Baselines 3."""

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        model_path: upath.UPath,
        vecnorm_path: upath.UPath,
        deterministic: bool = True,
    ):
        """Initialize the PyxieAgent by loading the model and normalizer."""
        self.model = algorithm.load(download_file(model_path))
        self.deterministic = deterministic

        # Load the lightweight normalizer
        self.normalizer = InferenceNormalizer(download_file(vecnorm_path))

    def set_env(self, env):
        """Set the environment attribute for getting in the __call__ method."""
        self.env = env

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Sample from the trained policy network."""
        obs_normalised = self.normalizer.normalize(obs)

        mask = self.env.unwrapped.action_masks()

        action, _ = self.model.predict(
            obs_normalised, deterministic=self.deterministic, action_masks=mask
        )

        return action
