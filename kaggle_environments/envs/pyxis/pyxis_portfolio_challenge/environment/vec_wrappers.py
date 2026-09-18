"""Vectorized (stable-baselines3) wrappers, used only for RL training.

Kept apart from ``wrappers`` / ``warmup_wrapper`` so importing the game engine
does not pull in stable-baselines3 — and through it torch, matplotlib and
tensorboard. Import this module only from training code paths.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

from pyxis_portfolio_challenge.environment.warmup_wrapper import (
    _clear_warmup_history,
)


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


class VecWarmupOnResetWrapper(VecEnvWrapper):
    """
    Vectorized wrapper that warms up environments after reset.

    Note: This wraps a VecEnv and handles warmup for all parallel environments.
    Each environment warms up independently when it auto-resets.
    """

    def __init__(
        self,
        venv,
        warmup_steps: int,
        policy: str = "do_nothing",
        verbose: bool = False,
    ):
        """
        Initialize vectorized warmup wrapper.

        Args:
            venv: Vectorized environment (e.g., SubprocVecEnv, DummyVecEnv)
            warmup_steps: Number of warmup steps per environment
            policy: "do_nothing" or "random"
            verbose: If True, log warmup progress

        """
        super().__init__(venv)
        self.num_envs = getattr(venv, "num_envs", 1)

        # warmup_steps and horizon are independent, additive knobs: each env's
        # warmup pre-roll runs on a temporarily extended horizon and its clock
        # is then rebased to 0 (see reset()), so the agent always plays a full
        # horizon regardless of how large warmup_steps is. No
        # warmup_steps < horizon check.
        self.warmup_steps = warmup_steps
        self.policy = policy
        self.verbose = verbose

        if policy not in ["do_nothing", "random"]:
            raise ValueError(f"Unknown warmup policy: {policy}")

    def reset(self):
        """Reset all environments and warm them up."""
        obs = self.venv.reset()

        if self.warmup_steps <= 0:
            return obs

        # Extend every env's horizon for the pre-roll so warmup never trips
        # horizon-based termination, however long it runs; the configured
        # horizon is restored after each clock is rebased below. This makes
        # warmup_steps and horizon additive (warmup may exceed horizon).
        try:
            self.venv.env_method("extend_horizon_for_warmup", self.warmup_steps)
        except (AttributeError, TypeError):
            pass

        # Temporarily disable metrics during warmup for all environments
        original_metrics = None
        try:
            # Try to get metrics from all environments
            original_metrics = self.venv.get_attr("metrics")
            # Set warmup flag on all metrics in all environments
            for env_metrics in original_metrics:
                for metric in env_metrics:
                    metric._warmup_mode = True
        except (AttributeError, TypeError):
            # get_attr/set_attr not available or metrics don't exist
            pass

        if self.verbose:
            print(f"Warming up {self.num_envs} envs for {self.warmup_steps} steps...")

        # Track which environments need warming up and retry counts
        envs_need_warmup = np.ones(
            self.num_envs, dtype=bool
        )  # All start needing warmup
        warmup_steps_done = np.zeros(self.num_envs, dtype=int)
        retry_counts = np.zeros(self.num_envs, dtype=int)
        max_retries = 100

        # Keep warming up until all environments complete warmup without termination
        total_terminations = 0
        while np.any(envs_need_warmup):
            # Choose actions for all environments
            if self.policy == "do_nothing":
                actions = np.zeros(
                    (self.num_envs, self.action_space.shape[0]),
                    dtype=self.action_space.dtype,
                )
            elif self.policy == "random":
                # Get action masks from all environments
                action_masks = np.array(self.venv.env_method("action_masks_binary"))

                # For MultiDiscrete (investment levels), randomly choose to invest
                # at STANDARD level (2) rather than sampling all levels uniformly.
                # This prevents overspending during warmup from ACCELERATED level.
                from gymnasium.spaces import MultiDiscrete

                if isinstance(self.action_space, MultiDiscrete):
                    # Binary decision: invest (STANDARD=2) or not (NONE=0)
                    invest_decisions = np.random.randint(
                        0, 2, size=(self.num_envs, self.action_space.shape[0])
                    )
                    # Convert to STANDARD level (2) for investments
                    actions = invest_decisions * 2  # 0 stays 0, 1 becomes 2
                else:
                    # MultiBinary: standard random sampling
                    actions = np.array([
                        self.action_space.sample() for _ in range(self.num_envs)
                    ])

                # Apply masks
                actions = actions * action_masks
            else:
                raise ValueError(f"Unknown policy: {self.policy}")

            # Step all environments
            obs, rewards, dones, infos = self.venv.step(actions)

            # Update warmup progress for environments that still need it
            warmup_steps_done += envs_need_warmup.astype(int)

            # Check which environments terminated during warmup
            for i in range(self.num_envs):
                if envs_need_warmup[i] and dones[i]:
                    # This environment terminated during warmup - need to retry
                    total_terminations += 1
                    retry_counts[i] += 1

                    if retry_counts[i] <= max_retries:
                        if self.verbose:
                            print(
                                f"  Env {i}: terminated at step "
                                f"{warmup_steps_done[i]}/{self.warmup_steps}. "
                                f"Retrying (attempt {retry_counts[i]})..."
                            )

                        # Reset this specific environment and restart its warmup
                        # Note: VecEnv reset with indices resets specific environments.
                        # Re-extend the horizon: reset() rebuilt a fresh game with
                        # the configured (un-extended) horizon.
                        try:
                            self.venv.env_method("reset", indices=[i])
                            self.venv.env_method(
                                "extend_horizon_for_warmup",
                                self.warmup_steps,
                                indices=[i],
                            )
                        except Exception:
                            # env_method doesn't work, reset all and re-extend
                            obs = self.venv.reset()
                            try:
                                self.venv.env_method(
                                    "extend_horizon_for_warmup", self.warmup_steps
                                )
                            except (AttributeError, TypeError):
                                pass

                        warmup_steps_done[i] = 0  # Restart count for this env
                    else:
                        # Exceeded max retries - give up on this environment
                        if self.verbose:
                            print(
                                f"  Env {i}: exceeded {max_retries} retries. Giving up."
                            )
                        envs_need_warmup[i] = False

                # Check if this environment has completed warmup
                elif warmup_steps_done[i] >= self.warmup_steps:
                    envs_need_warmup[i] = False

            if (
                self.verbose
                and np.sum(warmup_steps_done * envs_need_warmup) % (100 * self.num_envs)
                < self.num_envs
            ):
                avg_progress = (
                    np.mean(warmup_steps_done[envs_need_warmup])
                    if np.any(envs_need_warmup)
                    else self.warmup_steps
                )
                print(
                    f"  Warmup progress: ~{int(avg_progress)}/{self.warmup_steps} "
                    f"(avg across {np.sum(envs_need_warmup)} envs warming up)"
                )

        # Warn if there were many terminations during warmup
        if total_terminations > 0:
            if self.verbose:
                print(
                    f"  Total warmup terminations: {total_terminations} "
                    f"across {self.num_envs} environments"
                )

            if (
                total_terminations > self.num_envs * 2
            ):  # More than 2 retries per env on average
                import warnings

                warnings.warn(
                    f"{total_terminations} episodes terminated during warmup. "
                    f"Consider reducing warmup_on_reset_steps ({self.warmup_steps}) "
                    f"or changing warmup_on_reset_policy ('{self.policy}').",
                    UserWarning,
                    stacklevel=2,
                )

        # Clear GameState history lists that accumulate during warmup.
        try:
            game_states = self.venv.get_attr("game_state")
            for gs in game_states:
                _clear_warmup_history(gs)
        except (AttributeError, TypeError):
            pass

        # Rebase each env's clock so warmup counts as a pre-roll: clocks reset
        # to 0, configured horizons are restored, and each agent plays a full
        # horizon. Recompute obs so they reflect the rebased (time 0) state.
        try:
            self.venv.env_method("rebase_clock_after_warmup")
            rebased_obs = self.venv.env_method("_get_obs")
            if all(isinstance(o, np.ndarray) for o in rebased_obs):
                obs = np.stack(rebased_obs)
        except (AttributeError, TypeError):
            pass

        # Disable warmup mode on all metrics after warmup completes
        if original_metrics is not None:
            try:
                for env_metrics in original_metrics:
                    for metric in env_metrics:
                        metric._warmup_mode = False
            except (AttributeError, TypeError):
                pass

        if self.verbose:
            print("Warmup complete!")

        return obs

    def step_async(self, actions):
        """Forward to wrapped env."""
        return self.venv.step_async(actions)

    def step_wait(self):
        """Forward to wrapped env."""
        return self.venv.step_wait()

    def close(self):
        """Forward to wrapped env."""
        return self.venv.close()

    def get_attr(self, attr_name, indices=None):
        """Forward to wrapped env."""
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        """Forward to wrapped env."""
        return self.venv.set_attr(attr_name, value, indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Forward to wrapped env."""
        return self.venv.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )
