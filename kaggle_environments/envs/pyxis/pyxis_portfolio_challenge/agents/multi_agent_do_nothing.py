"""Do-nothing agent for multi-agent competitive environment."""


class MultiAgentDoNothingAgent:
    """
    Agent that takes no actions — passes on all investments and BD bids.

    Useful as a baseline or placeholder opponent.
    """

    def __init__(self, agent_name: str, *, env=None):
        """
        Initialise multi-agent do-nothing agent.

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
        Return a do-nothing action.

        Parameters
        ----------
        obs
            Observation from the environment (unused).

        Returns
        -------
        dict
            The env's canonical no-op action -- the do-nothing value for every
            enabled action head (investments, bd_bids, and any of ptrs_research,
            upgrade, site_bid, site_priority, pricing, demand_creation,
            brand_equity that the current config enables).

        """
        return self.env.noop_action()
