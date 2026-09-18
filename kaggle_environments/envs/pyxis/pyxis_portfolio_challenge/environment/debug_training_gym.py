"""
Debug script for validating InvestmentGameEnv Gym environment.

Validates both creation and functionality.

This main function serves as a debugging tool to:
1. Validate that the environment conforms to Gymnasium and Stable-Baselines3
standards
2. Test environment initialization, reset, and step operations
3. Verify observation space compatibility with FlattenObservation wrapper
4. Run a sample episode with random actions to ensure proper environment flow

Run this script directly to debug environment issues during development.
"""

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.env_checker import check_env as check_env_sb3


def debug_env(test_env):
    """Debug the Gym environment."""
    print(f"Testing environment {test_env.__class__.__name__}")
    try:
        check_env(test_env)
        check_env_sb3(test_env)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")

    # Test environment reset functionality
    obs, info = test_env.reset(seed=42)

    # Test observation space flattening for RL algorithms that require vector inputs
    wrapped_env = FlattenObservation(test_env)
    print(f"Flattened observation space shape: {wrapped_env.observation_space.shape}")

    # Run sample episode with random actions to test environment flow
    obs, info = test_env.reset()
    n_steps = 10
    for step in range(n_steps):
        # Sample random action from action space
        action = test_env.action_space.sample()
        obs, reward, terminated, truncated, info = test_env.step(action)
        print(f"Step {step + 1}: Reward = {reward:.2f}, Terminated = {terminated}")
        if terminated:
            print("Episode terminated, resetting environment")
            obs, info = test_env.reset()

    # Uncomment the line below to enable detailed observation debugging
    # debug_obs(test_env.observation_space, obs)


def debug_obs(space, obs, prefix=""):
    """
    Recursively debug observation space and observation compatibility.

    This development utility function validates that observations match their
    corresponding Gym spaces and provides detailed debugging output when
    mismatches occur. Useful for troubleshooting observation space issues
    during environment development.

    Parameters
    ----------
    space : gym.Space
        The Gym space to validate against
    obs : Any
        The observation to validate
    prefix : str, optional
        Prefix for debugging output to show hierarchy level

    Examples
    --------
    # Debug the full observation space
    debug_obs(env.observation_space, obs)

    # Debug a specific part of the observation
    debug_obs(env.observation_space['assets'], obs['assets'], prefix="assets.")

    """
    if isinstance(space, gym.spaces.Dict):
        if not isinstance(obs, dict):
            print(f"{prefix}INVALID: obs is not a dict for Dict space")
            return
        for key, subspace in space.spaces.items():
            if key not in obs:
                print(f"{prefix}{key}: MISSING in obs")
            else:
                debug_obs(subspace, obs[key], prefix=f"{prefix}{key}.")
    elif isinstance(space, gym.spaces.Tuple):
        if not isinstance(obs, (list, tuple)):
            print(f"{prefix}INVALID: obs is not a tuple or list for Tuple space")
            return
        if len(obs) != len(space.spaces):
            print(
                f"{prefix}INVALID: obs length {len(obs)} "
                f"!= space length {len(space.spaces)}"
            )
        for i, (subspace, subobs) in enumerate(zip(space.spaces, obs)):
            debug_obs(subspace, subobs, prefix=f"{prefix}[{i}].")
    else:
        valid = space.contains(obs)
        dtype = getattr(obs, "dtype", type(obs))
        shape = getattr(obs, "shape", "N/A")
        print(f"{prefix} value={obs}, type={dtype}, shape={shape}, valid={valid}")
        if not valid:
            print(f"      space: {space}")
