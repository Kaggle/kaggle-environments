from kaggle_environments.configuration import Configuration


class HaliteConfiguration(Configuration):
    def __init__(
        self,
        episode_steps: int = 400,
        agent_timeout: float = 30,
        act_timeout: float = 6,
        run_timeout: float = 1200,
        starting_halite: int = 24000,
        size: int = 21,
        spawn_cost: int = 500,
        convert_cost: int = 500,
        move_cost: float = 0,
        collect_rate: float = 0.25,
        regen_rate: float = 0.02,
        max_cell_halite: float = 500,
        player_starting_halite: int = 5000,
    ):
        Configuration.__init__(self, episode_steps, agent_timeout, act_timeout, run_timeout)
        self._starting_halite = starting_halite
        self._size = size
        self._spawn_cost = spawn_cost
        self._convert_cost = convert_cost
        self._move_cost = move_cost
        self._collect_rate = collect_rate
        self._regen_rate = regen_rate
        self._max_cell_halite = max_cell_halite
        self._player_starting_halite = player_starting_halite

    @property
    def starting_halite(self):
        return self._starting_halite

    @property
    def size(self):
        return self._size

    @property
    def spawn_cost(self):
        return self._spawn_cost

    @property
    def convert_cost(self):
        return self._convert_cost

    @property
    def move_cost(self):
        return self._move_cost

    @property
    def collect_rate(self):
        return self._collect_rate

    @property
    def regen_rate(self):
        return self._regen_rate

    @property
    def max_cell_halite(self):
        return self._max_cell_halite

    @property
    def player_starting_halite(self):
        return self._player_starting_halite
