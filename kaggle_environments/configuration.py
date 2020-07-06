class Configuration:
    def __init__(
        self,
        episode_steps: int = 1000,
        agent_timeout: float = 12,
        act_timeout: float = 6,
        run_timeout: float = 1200,
    ):
        assert isinstance(episode_steps, int)
        assert isinstance(agent_timeout, float)
        assert isinstance(act_timeout, float)
        assert isinstance(run_timeout, float)
        self._episode_steps = episode_steps
        self._agent_timeout = agent_timeout
        self._act_timeout = act_timeout
        self._run_timeout = run_timeout

    @property
    def episode_steps(self) -> int:
        return self._episode_steps

    @property
    def agent_timeout(self) -> float:
        return self._agent_timeout

    @property
    def act_timeout(self) -> float:
        return self._act_timeout

    @property
    def run_timeout(self) -> float:
        return self._run_timeout
