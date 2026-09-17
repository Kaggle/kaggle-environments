"""Random agent for multi-agent competitive environment."""

import numpy as np


class MultiAgentRandomAgent:
    """
    Random agent that respects action masks in the multi-agent environment.

    Selects uniformly random valid actions for both investments and BD bids.
    """

    def __init__(self, agent_name: str, *, env=None):
        """
        Initialise multi-agent random agent.

        Parameters
        ----------
        agent_name : str
            The agent identifier in the multi-agent environment
            (e.g. ``"pharma_0"``).
        env : MultiAgentInvestmentGameEnv | None
            Environment reference. Can be ``None`` if ``set_env`` is
            called before the first ``__call__``.

        """
        self.agent_name = agent_name
        self.env = env

    def set_env(self, env):
        """Set or update the environment reference."""
        self.env = env

    def __call__(self, obs) -> dict:
        """
        Select random valid actions.

        Parameters
        ----------
        obs
            Observation from the environment (unused — actions are
            mask-based).

        Returns
        -------
        dict
            Action dict with ``"investments"`` and ``"bd_bids"`` arrays.

        """
        masks = self.env.action_masks(self.agent_name)
        inv_mask = masks["investments"]

        investments = np.zeros(self.env.max_num_assets, dtype=np.int64)
        for i, m in enumerate(inv_mask):
            if isinstance(m, list):
                valid = [j for j, ok in enumerate(m) if ok]
                investments[i] = np.random.choice(valid) if valid else 0
            else:
                investments[i] = np.random.randint(0, 2) if m else 0

        # Continuous BD bids: a random cash amount (GBP millions) per slot that
        # holds a real BD asset. There is no affordability mask, so bids are
        # capped at the agent's current cash to keep the random baseline from
        # bankrupting itself every episode via overbids.
        bd_bids = np.zeros(self.env.bd_max_slots, dtype=np.float32)
        if getattr(self.env, "bd_enabled", False):
            shared = self.env.multi_agent_game.shared_market
            n_slots = min(len(shared.current_bd_assets), self.env.bd_max_slots)
            portfolio = self.env.agent_portfolios[self.agent_name]
            cash_millions = max(0.0, portfolio.cash / 1e6)
            cap = int(min(float(self.env.bd_max_bid), cash_millions))
            for i in range(n_slots):
                # ~half the time pass (bid 0), else a uniform random cash bid.
                if cap > 0 and np.random.randint(0, 2):
                    bd_bids[i] = np.random.randint(0, cap + 1)

        # Start from the canonical no-op so every enabled head is present
        # (ptrs_research, marketing, etc.), then overwrite the heads this agent
        # actively randomises. The strict env parser requires all heads.
        result = self.env.noop_action()
        result["investments"] = investments
        result["bd_bids"] = bd_bids

        # Clinical-site actions, mask-respecting and cash-capped so the random
        # baseline does not bankrupt itself every episode.
        sites_cfg = getattr(self.env, "clinical_sites_config", None)
        if sites_cfg is not None and sites_cfg.enabled:
            upgrade_mask = masks.get("upgrade", [True, False])
            can_buy = bool(upgrade_mask[1])
            # ~1-in-4 chance to buy a site when affordable.
            result["upgrade"] = int(
                can_buy and np.random.randint(0, 4) == 0
            )
            if sites_cfg.auction_enabled:
                portfolio = self.env.agent_portfolios[self.agent_name]
                cash_millions = max(0.0, portfolio.cash / 1e6)
                cap = int(min(float(sites_cfg.site_max_bid), cash_millions))
                bid = 0.0
                if cap > 0 and np.random.randint(0, 2):
                    bid = float(np.random.randint(0, cap + 1))
                result["site_bid"] = np.array([bid], dtype=np.float32)
            if sites_cfg.agent_priority:
                result["site_priority"] = np.random.random_sample(
                    self.env.max_num_assets
                ).astype(np.float32)

        return result
