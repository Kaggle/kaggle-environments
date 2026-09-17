"""
Action masking wrapper for algorithms that don't support native action masking.

This wrapper applies action masks by zeroing out invalid actions before they
are executed in the environment. This is useful for RecurrentPPO and other
algorithms from sb3-contrib that don't have built-in action masking support.
"""

import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


class VecActionMaskWrapper(VecEnvWrapper):
    """
    Vectorized wrapper that enforces action masking by modifying actions.

    This wrapper intercepts actions before they are sent to the environment
    and zeros out any invalid actions according to the action mask.

    For MultiBinary action spaces, invalid actions (where mask=False/0) are
    set to 0, preventing the agent from taking those actions.

    This wrapper should be placed AFTER VecNormalize in the wrapper chain,
    as it needs to access the underlying environments for action masks.
    """

    def __init__(self, venv: VecEnv, verbose: bool = False):
        """
        Initialize the action mask wrapper.

        Parameters
        ----------
        venv : VecEnv
            The vectorized environment to wrap.
        verbose : bool
            Whether to log when actions are masked.

        """
        super().__init__(venv)
        self.verbose = verbose
        self._cached_masks = None
        # Get action space size for MultiBinary
        self._action_size = self.action_space.n

    def reset(self) -> np.ndarray:
        """Reset the environment and cache action masks."""
        obs = self.venv.reset()
        self._cache_action_masks()
        return obs

    def _cache_action_masks(self) -> None:
        """Cache action masks from all sub-environments."""
        try:
            # env_method traverses the wrapper chain to call on underlying envs
            masks = self.venv.env_method("action_masks_binary")
            self._cached_masks = np.array(masks)
        except Exception:
            # Fallback: all actions valid
            self._cached_masks = np.ones(
                (self.num_envs, self._action_size), dtype=np.int32
            )

    def step_async(self, actions: np.ndarray) -> None:
        """Apply action mask and send actions to environment."""
        if self._cached_masks is not None:
            # Apply mask: set invalid actions to 0
            masked_actions = (actions * self._cached_masks).astype(actions.dtype)
            if self.verbose and not np.array_equal(actions, masked_actions):
                num_masked = np.sum(actions != masked_actions)
                print(f"Masked {num_masked} invalid actions")
            actions = masked_actions
        self.venv.step_async(actions)

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Wait for step to complete and update cached masks."""
        obs, rewards, dones, infos = self.venv.step_wait()
        self._cache_action_masks()
        return obs, rewards, dones, infos

    def get_action_masks(self) -> np.ndarray:
        """Return the current action masks for all environments."""
        if self._cached_masks is None:
            self._cache_action_masks()
        return self._cached_masks

    def save(self, path: str) -> None:
        """Forward save to underlying VecNormalize if present."""
        self.venv.save(path)

    @property
    def obs_rms(self):
        """Forward obs_rms to underlying VecNormalize."""
        return self.venv.obs_rms

    @obs_rms.setter
    def obs_rms(self, value):
        """Forward obs_rms setter to underlying VecNormalize."""
        self.venv.obs_rms = value

    @property
    def ret_rms(self):
        """Forward ret_rms to underlying VecNormalize."""
        return self.venv.ret_rms

    @ret_rms.setter
    def ret_rms(self, value):
        """Forward ret_rms setter to underlying VecNormalize."""
        self.venv.ret_rms = value
