"""Wrapper to warm up environment on every reset."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


def _clear_warmup_history(game_state) -> None:
    """Clear history lists that accumulate during warmup on a GameState."""
    for attr_name in [
        "realised_revenues",
        "realised_costs",
        "running_enpv",
        "running_eroi",
        "realised_net_cash_flow",
    ]:
        attr = getattr(game_state, attr_name, None)
        if isinstance(attr, list):
            attr.clear()
    # Reset TA experience — expertise should be built during the actual episode
    ta_exp = getattr(game_state, "ta_experience", None)
    if ta_exp is not None:
        for ta in ta_exp:
            ta_exp[ta] = 0.0


class WarmupOnResetWrapper(gym.Wrapper):
    """
    Wrapper that warms up the environment after every reset.

    After each reset() call, this wrapper runs the environment for
    a specified number of warmup steps before returning control to
    the agent. This ensures every episode starts with a "warmed up"
    environment state.

    Useful for:
    - Consistent environment initialization
    - Building up observation/reward statistics per episode
    - Ensuring environment is in a stable state before agent acts

    Example:
        env = InvestmentGameEnv(...)
        env = WarmupOnResetWrapper(env, warmup_steps=100, policy="do_nothing")

        # Each reset now includes 100 warmup steps
        obs, info = env.reset()  # Warms up automatically
        # ... agent acts ...
        obs, info = env.reset()  # Warms up again on next episode

    """

    def __init__(
        self,
        env: gym.Env,
        warmup_steps: int,
        policy: str = "do_nothing",
        verbose: bool = True,
    ):
        """
        Initialize the warmup wrapper.

        Args:
            env: The environment to wrap
            warmup_steps: Number of steps to warm up after each reset
            policy: Warmup policy - "do_nothing" or "random"
            verbose: If True, print warmup progress

        """
        super().__init__(env)

        # warmup_steps and horizon are independent, additive knobs: the warmup
        # pre-roll runs on a temporarily extended horizon and the clock is then
        # rebased to 0 (see reset()), so the agent always plays a full horizon
        # regardless of how large warmup_steps is. No warmup_steps < horizon
        # check.
        self.warmup_steps = warmup_steps
        self.policy = policy
        self.verbose = verbose

        if policy not in ["do_nothing", "random"]:
            raise ValueError(
                f"Unknown warmup policy: {policy}. Must be 'do_nothing' or 'random'"
            )

    def reset(self, **kwargs):
        """
        Reset environment and run warmup steps.

        Args:
            **kwargs: Arguments passed to env.reset()

        Returns:
            observation: Final observation after warmup
            info: Info dict from final warmup step

        """
        if self.warmup_steps <= 0:
            # No warmup - just reset normally
            return self.env.reset(**kwargs)

        # Keep trying until we get a successful warmup (no terminations during warmup)
        max_attempts = 100
        attempts = 0

        while attempts < max_attempts:
            attempts += 1

            # Temporarily disable metrics during warmup to avoid polluting
            # evaluation data. We need to do this BEFORE calling reset()
            # because on_episode_begin is called in reset()
            unwrapped_env = self.env.unwrapped
            original_metrics = None
            if hasattr(unwrapped_env, "metrics"):
                original_metrics = unwrapped_env.metrics
                # Set a warmup flag on each metric to disable collection
                for metric in original_metrics:
                    metric._warmup_mode = True

            # Reset the underlying environment (with metrics in warmup mode)
            obs, info = self.env.reset(**kwargs)

            # Extend the horizon for the pre-roll so warmup never trips
            # horizon-based termination, however long it runs; the configured
            # horizon is restored after the clock is rebased below. This makes
            # warmup_steps and horizon additive (warmup may exceed horizon).
            if hasattr(unwrapped_env, "extend_horizon_for_warmup"):
                unwrapped_env.extend_horizon_for_warmup(self.warmup_steps)

            # Run warmup steps
            terminated_during_warmup = False
            for step in range(self.warmup_steps):
                # Choose action based on policy
                if self.policy == "do_nothing":
                    action = np.zeros(
                        self.action_space.shape, dtype=self.action_space.dtype
                    )
                elif self.policy == "random":
                    # For MultiDiscrete (investment levels), randomly choose to invest
                    # at STANDARD level (2) rather than sampling all levels uniformly.
                    # This prevents overspending during warmup from ACCELERATED level.
                    from gymnasium.spaces import MultiDiscrete

                    if isinstance(self.action_space, MultiDiscrete):
                        # Binary decision: invest (STANDARD=2) or not (NONE=0)
                        invest_decisions = np.random.randint(
                            0, 2, size=self.action_space.shape
                        )
                        # Convert to STANDARD level (2) for investments
                        action = invest_decisions * 2  # 0 stays 0, 1 becomes 2
                    else:
                        # MultiBinary: standard random sampling
                        action = self.action_space.sample()

                    # Apply action masks if available
                    if hasattr(self.env, "action_masks"):
                        action_masks = self.env.action_masks_binary()
                        # For MultiDiscrete, mask zeros out invalid investments
                        action = action * action_masks
                else:
                    raise ValueError(f"Unknown policy: {self.policy}")

                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)

                # If episode ends during warmup, scrap this attempt and try again
                if terminated or truncated:
                    terminated_during_warmup = True
                    if self.verbose:
                        print(
                            f"  Episode terminated at step {step + 1}/"
                            f"{self.warmup_steps}. Retrying..."
                        )
                    break

                # Optional verbose logging
                if self.verbose and (step + 1) % 50 == 0:
                    print(f"  Warmup progress: {step + 1}/{self.warmup_steps}")

            # If we completed warmup without termination, we're done
            if not terminated_during_warmup:
                if self.verbose and attempts > 1:
                    print(f"  Warmup successful after {attempts} attempt(s)")
                break

        # Check if we exceeded max attempts
        if attempts >= max_attempts:
            if self.verbose:
                print(
                    f"  WARNING: Exceeded {max_attempts} warmup attempts. "
                    "Using last attempt."
                )
            import warnings

            warnings.warn(
                f"Episode terminated during warmup {max_attempts} times. "
                f"Consider reducing warmup_on_reset_steps ({self.warmup_steps}) "
                f"or changing warmup_on_reset_policy ('{self.policy}').",
                UserWarning,
                stacklevel=2,
            )

        # Clear GameState history lists that accumulate during warmup.
        unwrapped_env = self.env.unwrapped
        if hasattr(unwrapped_env, "game_state"):
            _clear_warmup_history(unwrapped_env.game_state)

        # Rebase the clock so warmup counts as a pre-roll: the agent's clock
        # resets to 0, the configured horizon is restored, and it plays a full
        # horizon. Recompute obs/info so they reflect the rebased (time 0)
        # state before returning to the agent.
        if hasattr(unwrapped_env, "rebase_clock_after_warmup"):
            unwrapped_env.rebase_clock_after_warmup()
            obs = unwrapped_env._get_obs()
            info = unwrapped_env._get_info()

        # Restore metrics after warmup
        if original_metrics is not None:
            # Disable warmup mode on all metrics
            for metric in original_metrics:
                metric._warmup_mode = False

            # Call on_episode_begin now that warmup is complete
            # This initializes the episode in the metrics system
            from pyxis_portfolio_challenge.environment.metrics import (
                MetricsContext,
                collect_metrics,
            )

            episode_id = getattr(unwrapped_env, "_episode_fingerprint", None)
            ctx = MetricsContext(
                unwrapped_env.game_state, reward=0.0, episode_id=episode_id
            )
            collect_metrics(
                collection_fn="on_episode_begin", context=ctx, metrics=original_metrics
            )

        return obs, info


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


class MultiAgentWarmupOnResetWrapper:
    """
    Warmup wrapper for PettingZoo ParallelEnv (multi-agent).

    After each reset(), runs the environment for a specified number of
    warmup steps using do_nothing actions, then clears accumulated history
    so the episode starts cleanly for the agent.

    Delegates all attribute access to the wrapped env for transparency.
    """

    def __init__(
        self,
        env,
        warmup_steps: int,
        policy: str = "do_nothing",
        verbose: bool = True,
    ):
        """Initialize multi-agent warmup wrapper."""
        self.env = env
        self.warmup_steps = warmup_steps
        self.policy = policy
        self.verbose = verbose

        if policy not in ["do_nothing"]:
            raise ValueError(
                f"Unknown warmup policy: {policy}. "
                "Multi-agent warmup only supports 'do_nothing'."
            )

        # warmup_steps and horizon are independent, additive knobs: the warmup
        # pre-roll runs on a temporarily extended horizon (see reset()) and the
        # clock is then rebased to 0, so the agent always plays a full horizon
        # no matter how large warmup_steps is. No warmup_steps < horizon check.

    def __getattr__(self, name):
        """Delegate attribute access to wrapped env."""
        return getattr(self.env, name)

    def reset(self, **kwargs):
        """Reset environment and run warmup steps."""
        if self.warmup_steps <= 0:
            return self.env.reset(**kwargs)

        configured_horizon = self.env.horizon

        max_attempts = 100
        for attempt in range(max_attempts):
            observations, infos = self.env.reset(**kwargs)

            # Extend the horizon for the pre-roll so warmup never trips
            # horizon-based termination, however long it runs; the configured
            # horizon is restored after the clock is rebased below. This makes
            # warmup_steps and horizon additive (warmup may exceed horizon).
            self.env.multi_agent_game = self.env.multi_agent_game.with_horizon(
                configured_horizon + self.warmup_steps
            )

            terminated_during_warmup = False
            for step in range(self.warmup_steps):
                if not self.env.agents:
                    terminated_during_warmup = True
                    break

                # Do-nothing warmup: emit the env's canonical no-op for every
                # enabled head (the strict parser requires all heads each step).
                actions = {
                    agent_id: self.env.noop_action() for agent_id in self.env.agents
                }

                observations, _, terminations, truncations, infos = self.env.step(
                    actions
                )

                if all(terminations.values()) or all(truncations.values()):
                    terminated_during_warmup = True
                    if self.verbose:
                        print(
                            f"  Episode terminated at warmup step {step + 1}/"
                            f"{self.warmup_steps}. Retrying..."
                        )
                    break

            if not terminated_during_warmup:
                break

        if attempt >= max_attempts - 1:
            import warnings

            warnings.warn(
                f"Episode terminated during warmup {max_attempts} times.",
                UserWarning,
                stacklevel=2,
            )

        # Clear warmup history on each agent's GameState
        for agent_state in self.env.multi_agent_game.agent_states.values():
            _clear_warmup_history(agent_state)

        # Rebase the game clock so warmup counts as a pre-roll: the agent's
        # clock resets to 0 and it plays a full horizon. Restore the configured
        # horizon (dropped from the extended pre-roll value) so the agent plays
        # exactly `horizon` steps. Recompute observations/infos so they reflect
        # the rebased (time 0) state.
        self.env.multi_agent_game = self.env.multi_agent_game.rebase_time_to_zero()
        self.env.multi_agent_game = self.env.multi_agent_game.with_horizon(
            configured_horizon
        )
        observations = {
            agent: self.env._get_observation(agent) for agent in self.env.agents
        }
        infos = {agent: self.env._get_info(agent) for agent in self.env.agents}

        return observations, infos

    def step(self, actions):
        """Forward step to wrapped env."""
        return self.env.step(actions)
