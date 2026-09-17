"""
Self-play training wrapper for multi-agent environments.

Converts a PettingZoo multi-agent environment into a single-agent Gymnasium
environment for training with MaskablePPO. One agent (the "primary") is
trained while opponents use frozen copies of the policy, periodically
synced from the training model.

Example usage::

    from pyxis_portfolio_challenge.environment import make_multi_agent_train_env
    from pyxis_portfolio_challenge.environment.self_play import (
        SelfPlayWrapper,
        OpponentSyncCallback,
    )
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    # Create wrapped envs for parallel training
    def make_env(seed):
        def _init():
            env = make_multi_agent_train_env()
            wrapped = SelfPlayWrapper(env, policy_kwargs={"net_arch": [256, 256]})
            wrapped.reset(seed=seed)
            return wrapped
        return _init

    train_env = SubprocVecEnv([make_env(i) for i in range(8)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    model = MaskablePPO("MlpPolicy", train_env, policy_kwargs={"net_arch": [256, 256]})
    model.learn(
        total_timesteps=1_000_000,
        callback=OpponentSyncCallback(sync_every_n_rollouts=1),
    )
"""

from __future__ import annotations

import logging

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class SelfPlayWrapper(gym.Env):
    """
    Converts multi-agent PettingZoo env to single-agent Gymnasium for self-play.

    Each subprocess holds its own opponent policy copy. Weights are synced
    from the main training model via update_opponent().

    Action space is a single MultiDiscrete combining investments + BD bids:
    - Investments: MultiDiscrete with 5 levels per asset
      (NONE/MINIMAL/STANDARD/ACCELERATED/STOP)
    - BD bids: MultiDiscrete with 5 levels per BD slot
      (0=no bid, 1-4=increasing bid amounts)

    Invalid actions are masked via action_masks() for MaskablePPO.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env,
        policy_kwargs: dict,
        bd_actions_masked: bool = False,
    ):
        """
        Initialise self-play wrapper.

        Parameters
        ----------
        env : MultiAgentInvestmentGameEnv
            The PettingZoo multi-agent environment to wrap.
        policy_kwargs : dict
            Policy configuration dict passed to MaskableActorCriticPolicy
            for creating opponent policy copies in each subprocess.
        bd_actions_masked : bool
            If True, BD bid actions are forced to pass-only (for curriculum).

        """
        super().__init__()
        self.env = env
        self.agents = env.possible_agents
        self.primary_agent = self.agents[0]

        self.n_investments = env.max_num_assets
        self.n_bd = env.bd_max_slots
        self.bd_actions_masked = bd_actions_masked

        # Observation space: same as any agent's
        self.observation_space = env.observation_space(self.primary_agent)

        # Combine investments + BD bids into a single MultiDiscrete
        underlying = env.action_space(self.primary_agent)
        inv_space = underlying["investments"]
        bd_space = underlying["bd_bids"]

        if isinstance(inv_space, gym.spaces.MultiDiscrete):
            inv_nvec = list(inv_space.nvec)
        else:
            # MultiBinary(n) -> MultiDiscrete([2]*n)
            inv_nvec = [2] * inv_space.n

        bd_nvec = list(bd_space.nvec)
        self.action_space = gym.spaces.MultiDiscrete(inv_nvec + bd_nvec)

        self._current_obs = {}
        self._policy_kwargs = policy_kwargs

        # Opponent policy (lazy init on first update_opponent call)
        self._opponent_policy = None
        self._obs_rms_mean = None
        self._obs_rms_var = None

    def update_opponent(self, policy_state_dict, obs_rms_mean, obs_rms_var):
        """
        Update opponent policy weights and obs normalization stats.

        Called from main process via SubprocVecEnv.env_method().
        """
        if self._opponent_policy is None:
            from sb3_contrib.common.maskable.policies import (
                MaskableActorCriticPolicy,
            )

            self._opponent_policy = MaskableActorCriticPolicy(
                self.observation_space,
                self.action_space,
                lr_schedule=lambda _: 0.0,
                **self._policy_kwargs,
            )
            self._opponent_policy.set_training_mode(False)

        # Load weights (convert numpy arrays back to tensors)
        torch_state_dict = {k: torch.tensor(v) for k, v in policy_state_dict.items()}
        self._opponent_policy.load_state_dict(torch_state_dict)
        self._obs_rms_mean = obs_rms_mean
        self._obs_rms_var = obs_rms_var

    def _normalize_obs(self, obs):
        """Normalize observation using VecNormalize running stats."""
        if self._obs_rms_mean is not None:
            return (obs - self._obs_rms_mean) / np.sqrt(self._obs_rms_var + 1e-8)
        return obs

    def reset(self, seed=None, options=None):
        """Reset and return primary agent's observation."""
        self._current_obs, infos = self.env.reset(seed=seed)
        obs = self._current_obs[self.primary_agent]
        info = infos.get(self.primary_agent, {})
        return obs, info

    def step(self, action):
        """Step: apply action for primary agent, use local policy for opponents."""
        actions = {}

        # Primary agent's action from PPO
        actions[self.primary_agent] = self._decode_action(action, self.primary_agent)

        # Opponent actions: use local policy copy if available, else do nothing
        for agent in self.env.agents:
            if agent == self.primary_agent:
                continue
            if agent not in self._current_obs:
                actions[agent] = self._do_nothing_action()
                continue
            if self._opponent_policy is not None:
                opponent_obs = self._normalize_obs(self._current_obs[agent])
                obs_tensor = torch.tensor(opponent_obs[np.newaxis], dtype=torch.float32)
                # Get flattened opponent action masks (inv + BD)
                opp_masks = self.env.action_masks(agent)
                opp_inv_mask = opp_masks["investments"]
                opp_bd_mask = opp_masks["bd_bids"]
                parts = []
                if isinstance(opp_inv_mask, list):
                    parts.extend(b for slot in opp_inv_mask for b in slot)
                else:
                    for m in opp_inv_mask:
                        parts.extend([True, bool(m)])
                for slot in opp_bd_mask:
                    parts.extend(slot)
                flat_mask = np.array(parts, dtype=bool)
                mask_tensor = torch.tensor(flat_mask[np.newaxis], dtype=torch.bool)
                with torch.no_grad():
                    dist = self._opponent_policy.get_distribution(
                        obs_tensor, action_masks=mask_tensor
                    )
                    opponent_action = dist.mode().cpu().numpy()[0]
                actions[agent] = self._decode_action(opponent_action, agent)
            else:
                actions[agent] = self._do_nothing_action()

        obs_dict, rewards, terms, truncs, infos = self.env.step(actions)
        self._current_obs = obs_dict

        done = terms.get(self.primary_agent, False)
        truncated = truncs.get(self.primary_agent, False)
        reward = rewards.get(self.primary_agent, 0.0)
        obs = obs_dict.get(self.primary_agent, np.zeros(self.observation_space.shape))
        info = infos.get(self.primary_agent, {})

        return obs, reward, done, truncated, info

    def action_masks(self) -> np.ndarray:
        """
        Return action masks for MaskablePPO.

        Returns a flattened 1D boolean array covering investments + BD bids.
        When bd_actions_masked=True, BD dimensions are forced to pass-only.
        """
        masks = self.env.action_masks(self.primary_agent)
        inv_mask = masks["investments"]
        bd_mask = masks["bd_bids"]

        parts = []
        if isinstance(inv_mask, list):
            parts.extend(b for slot in inv_mask for b in slot)
        else:
            for m in inv_mask:
                parts.extend([True, bool(m)])

        if self.bd_actions_masked:
            # Force pass-only on all BD slots
            for slot in bd_mask:
                parts.extend([True] + [False] * (len(slot) - 1))
        else:
            for slot in bd_mask:
                parts.extend(slot)

        return np.array(parts, dtype=bool)

    def _decode_action(self, action, agent: str) -> dict:
        """Split combined MultiDiscrete action into investments + BD bids."""
        action = np.asarray(action, dtype=np.int64)
        investments = action[: self.n_investments]
        bd_bids = action[self.n_investments : self.n_investments + self.n_bd]

        return {
            "investments": investments,
            "bd_bids": bd_bids,
        }

    def _do_nothing_action(self) -> dict:
        """Return a do-nothing action."""
        return {
            "investments": np.zeros(self.n_investments, dtype=np.int64),
            "bd_bids": np.zeros(self.n_bd, dtype=np.int64),
        }

    def set_bd_actions_masked(self, masked: bool):
        """Enable/disable BD action masking (for curriculum)."""
        self.bd_actions_masked = masked

    def set_reward_type(self, reward_type: str):
        """
        Change the reward type on the underlying multi-agent env.

        Called from main process via SubprocVecEnv.env_method().
        """
        from pyxis_portfolio_challenge.environment.multi_agent_training_gym import (
            MultiAgentInvestmentGameEnv,
        )

        base = self.env
        while hasattr(base, "env") and not isinstance(
            base, MultiAgentInvestmentGameEnv
        ):
            base = base.env
        base._reward_type = reward_type

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        return self.env.close()


class OpponentSyncCallback(BaseCallback):
    """
    Syncs opponent policy weights to all subprocess envs periodically.

    Syncs every ``sync_every_n_rollouts`` rollouts to amortize the cost of
    sending ~8MB of weights to each subprocess through pipes.

    Syncs to both train and eval envs so that eval episodes use a live
    opponent rather than a do-nothing placeholder.

    Parameters
    ----------
    sync_every_n_rollouts : int
        How often to sync weights. Default 5.
    eval_env : VecEnv | None
        Optional eval VecEnv to also sync weights to.

    """

    def __init__(self, sync_every_n_rollouts: int = 5, eval_env=None):
        """Initialize."""
        super().__init__()
        self.sync_every_n_rollouts = sync_every_n_rollouts
        self._rollout_count = 0
        self._eval_env = eval_env

    def _sync_to_vec_env(self, vec_env, numpy_state_dict, obs_rms_mean, obs_rms_var):
        """Send opponent weights to all subprocess envs in a VecEnv stack."""
        base_env = vec_env
        while hasattr(base_env, "venv"):
            base_env = base_env.venv
        base_env.env_method(
            "update_opponent", numpy_state_dict, obs_rms_mean, obs_rms_var
        )

    def _on_rollout_start(self):
        """Called at the start of each rollout — sync opponent weights periodically."""
        self._rollout_count += 1

        # Sync every N rollouts to amortize pipe overhead
        if (
            self._rollout_count > 1
            and self._rollout_count % self.sync_every_n_rollouts != 0
        ):
            return

        # Get current policy weights as numpy (for pickling through pipes)
        state_dict = self.model.policy.state_dict()
        numpy_state_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}

        # Get VecNormalize obs running stats from training env
        vec_env = self.model.get_env()
        if not hasattr(vec_env, "obs_rms"):
            obs_rms_mean = None
            obs_rms_var = None
        else:
            obs_rms_mean = vec_env.obs_rms.mean.copy()
            obs_rms_var = vec_env.obs_rms.var.copy()

        # Sync to training env
        self._sync_to_vec_env(vec_env, numpy_state_dict, obs_rms_mean, obs_rms_var)

        # Sync to eval env so eval opponents are not do-nothing placeholders
        if self._eval_env is not None:
            self._sync_to_vec_env(
                self._eval_env, numpy_state_dict, obs_rms_mean, obs_rms_var
            )

    def _on_step(self):
        return True
