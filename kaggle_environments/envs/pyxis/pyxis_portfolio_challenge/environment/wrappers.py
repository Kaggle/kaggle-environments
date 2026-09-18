import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


class VecAutoCenterWrapper(VecEnvWrapper):
    """
    A Vectorized Wrapper that calculates a GLOBAL mean across all parallel environments.

    This wrapper maintains a running average of all rewards received across
    all environments and all steps taken. It subtracts this global mean from
    the rewards returned at each step to center the rewards.

    Running average is computed until the specified number of calibration steps
    has been reached, after which the mean is frozen.

    This is to account for the fact that a default random policy can yield a
    significant positive reward due to the structure of the investment game
    environment. By centering rewards around the global mean, we help stabilize
    learning.
    """

    def __init__(self, venv, calibration_steps=10_000):
        """Initialize the VecAutoCenterWrapper."""
        super().__init__(venv)
        self.calibration_steps = calibration_steps
        self.total_steps = 0
        self.running_sum = 0.0
        self.mean = 0.0
        self.frozen = False

    def reset(self):
        """Reset the environment within the wrapper."""
        return self.venv.reset()

    def step_wait(self):
        """Step the environment within the wrapper with all vectorized envs."""
        obs, rews, dones, infos = self.venv.step_wait()

        # 1. Update Global Statistics (if not frozen)
        if not self.frozen:
            # Add up rewards from ALL envs in this batch
            batch_sum = np.sum(rews)
            batch_count = len(rews)

            self.running_sum += batch_sum
            self.total_steps += batch_count
            self.mean = self.running_sum / self.total_steps

            if self.total_steps >= self.calibration_steps:
                self.frozen = True
                print(
                    f"GLOBAL AutoCenter: Baseline Frozen at {self.mean:.2f}"
                    f" after {self.total_steps} global steps."
                )

        # 2. Subtract the Global Mean from the entire batch of rewards
        # rews is a numpy array of shape (n_envs,), so this broadcasts correctly
        centered_rews = rews - self.mean

        return obs, centered_rews, dones, infos


class AutoCenterWrapper(gym.Wrapper):
    """
    A Wrapper that calculates a running mean of rewards for a single environment.

    This wrapper maintains a running average of all rewards received and
    all steps taken. It subtracts this mean from the reward returned at each
    step to center the rewards.

    Running average is computed until the specified number of calibration steps
    has been reached, after which the mean is frozen.

    This is to account for the fact that a default random policy can yield a
    significant positive reward due to the structure of the investment game
    environment. By centering rewards around the mean, we help stabilize
    learning.
    """

    def __init__(self, env, calibration_steps=10_000):
        """Initialize the AutoCenterWrapper."""
        super().__init__(env)
        self.calibration_steps = calibration_steps
        self.total_steps = 0
        self.running_sum = 0.0
        self.mean = 0.0
        self.frozen = False

    def step(self, action):
        """Step the environment within the wrapper."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 1. Update Statistics (if not frozen)
        if not self.frozen:
            self.running_sum += reward
            self.total_steps += 1
            self.mean = self.running_sum / self.total_steps

            if self.total_steps >= self.calibration_steps:
                self.frozen = True
                print(
                    f"AutoCenter: Baseline Frozen at {self.mean:.2f}"
                    f" after {self.total_steps} steps."
                )

        # 2. Subtract the Mean from the reward
        centered_reward = reward - self.mean

        return obs, centered_reward, terminated, truncated, info
