"""PPO-based agent for multi-agent competitive environment."""

from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from pyxis_portfolio_challenge.agents.pyxie import InferenceNormalizer


def _noop_lr_schedule(_):
    return 0.0


class MultiAgentPyxieAgent:
    """
    RL agent using a trained MaskablePPO policy for multi-agent play.

    Loads a pre-trained model and observation normaliser, then produces
    actions in the multi-agent dict format (``investments`` + ``bd_bids``).
    """

    def __init__(
        self,
        agent_name: str,
        *,
        model_path: str | Path,
        vecnorm_path: str | Path,
        deterministic: bool = True,
        env=None,
    ):
        """
        Initialise multi-agent Pyxie agent.

        Parameters
        ----------
        agent_name : str
            The agent identifier in the multi-agent environment
            (e.g. ``"pharma_0"``).
        model_path : str | Path
            Path to the saved MaskablePPO model (``.zip``).
        vecnorm_path : str | Path
            Path to the saved VecNormalize statistics (``.pkl``).
        deterministic : bool
            Whether to use deterministic action selection.
        env : MultiAgentInvestmentGameEnv | None
            Environment reference. Can be ``None`` if ``set_env`` is
            called before the first ``__call__``.

        """
        self.agent_name = agent_name
        self.model = MaskablePPO.load(
            str(model_path),
            custom_objects={"lr_schedule": _noop_lr_schedule},
        )
        # Replace any remaining unpicklable lr closures saved with the model
        self.model.lr_schedule = _noop_lr_schedule
        self.model.learning_rate = _noop_lr_schedule
        self.normalizer = InferenceNormalizer(str(vecnorm_path))
        self.deterministic = deterministic
        self.env = env

    def set_env(self, env):
        """Set or update the environment reference."""
        self.env = env

    def __call__(self, obs: np.ndarray) -> dict:
        """
        Select actions using the trained policy.

        Parameters
        ----------
        obs : np.ndarray
            Flattened observation vector from the environment.

        Returns
        -------
        dict
            Action dict with ``"investments"`` and ``"bd_bids"`` arrays.

        """
        norm_obs = self.normalizer.normalize(obs)

        masks = self.env.action_masks(self.agent_name)
        inv_mask = masks["investments"]
        bd_mask = masks["bd_bids"]

        # Flatten mask dict into a single boolean array matching the
        # model's expected format (concatenated per-asset mask slots
        # followed by per-BD-slot mask slots).
        parts = []
        if (
            isinstance(inv_mask, list)
            and len(inv_mask) > 0
            and isinstance(inv_mask[0], list)
        ):
            # MultiDiscrete: each asset has a list of valid levels
            for slot in inv_mask:
                parts.extend(slot)
        else:
            # MultiBinary: each asset has a single 0/1 mask
            for m in inv_mask:
                parts.extend([True, bool(m)])
        for slot in bd_mask:
            parts.extend(slot)
        flat_mask = np.array(parts, dtype=bool)

        action, _ = self.model.predict(
            norm_obs[np.newaxis],
            deterministic=self.deterministic,
            action_masks=flat_mask[np.newaxis],
        )
        action = action.flatten()

        n_inv = self.env.max_num_assets
        n_bd = self.env.bd_max_slots
        # The trained policy only emits investments + bd_bids. Fill every other
        # enabled head with its no-op value so the strict env parser accepts the
        # action (all enabled heads are required each step).
        return {
            **self.env.noop_action(),
            "investments": action[:n_inv],
            "bd_bids": action[n_inv : n_inv + n_bd],
        }
