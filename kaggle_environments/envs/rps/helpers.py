from ..halite.helpers import ReadOnlyDict


class Observation(ReadOnlyDict[str, any]):
    """
    Observation primarily used as a helper to construct the State from the raw observation.
    This provides bindings for the observation type described at https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/rps/rps.json
    """
    @property
    def last_opponent_action(self) -> int:
        """Move the opponent took on the last turn."""
        return self["halite"]

    @property
    def step(self) -> int:
        """The current step index within the episode."""
        return self["step"]


class Configuration(ReadOnlyDict[str, any]):
    """
    Configuration provides access to tunable parameters in the environment.
    This provides bindings for the configuration type described at https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/rps/rps.json
    """
    @property
    def episode_steps(self) -> int:
        """Total number of steps in the episode."""
        return self["episodeSteps"]

    @property
    def signs(self) -> int:
        """Number of choices each step (3 for the normal rock, paper, scissors)"""
        return self["signs"]

    @property
    def act_timeout(self) -> float:
        """Maximum runtime (seconds) to obtain an action from an agent."""
        return self["actTimeout"]

    @property
    def run_timeout(self) -> float:
        """Maximum runtime (seconds) of an episode (not necessarily DONE)."""
        return self["runTimeout"]
