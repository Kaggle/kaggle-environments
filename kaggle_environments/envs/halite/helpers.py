from typing import *
from string import Template


# See https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/halite/halite.json for schema


class Object:
    def __init__(self, data: Dict[str, any]):
        self._data = data

    def __getitem__(self, item):
        return self._data[item]


class Observation(Object):
    @property
    def halite(self) -> List[int]:
        """Serialized list of available halite per cell on the board."""
        return self["halite"]

    @property
    def players(self) -> List[List[int]]:
        """List of players and their assets."""
        return self["players"]

    @property
    def player(self) -> int:
        """The current agent's player index."""
        return self["player"]

    @property
    def step(self) -> int:
        """The current step index within the episode."""
        return self["step"]


# TODO: Convert this to a typed object
class Configuration(Object):
    @property
    def episode_steps(self):
        """Total number of steps/turns in the run."""
        return self["episodeSteps"]

    @property
    def agent_exec(self):
        """How the agent is executed alongside the running envionment ('LOCAL' or separate 'PROCESS')."""
        return self["agentExec"]

    @property
    def agent_timeout(self):
        """Maximum runtime (seconds) to initialize an agent."""
        return self["agentTimeout"]

    @property
    def act_timeout(self):
        """Maximum runtime (seconds) to obtain an action from an agent."""
        return self["actTimeout"]

    @property
    def run_timeout(self):
        """Maximum runtime (seconds) of an episode (not necessarily DONE)."""
        return self["runTimeout"]

    @property
    def halite(self):
        """The starting amount of halite available on the board."""
        return self["halite"]

    @property
    def size(self):
        """The number of cells vertically and horizontally on the board."""
        return self["size"]

    @property
    def spawn_cost(self):
        """The amount of halite to spawn a new ship."""
        return self["spawnCost"]

    @property
    def convert_cost(self):
        """The amount of halite to convert a ship into a shipyard."""
        return self["convertCost"]

    @property
    def move_cost(self):
        """The percent deducted from ship's current halite per move."""
        return self["moveCost"]

    @property
    def collect_rate(self):
        """The rate of halite collected by a ship from a cell by not moving."""
        return self["collectRate"]

    @property
    def regen_rate(self):
        """The rate halite regenerates on the board."""
        return self["regenRate"]

    @property
    def max_cell_halite(self):
        """The maximum halite that can be in any cell."""
        return self["maxCellHalite"]


ShipId = NewType('ShipId', str)
ShipyardId = NewType('ShipyardId', str)
PlayerId = NewType('PlayerId', int)


class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def map(self, f: Callable[[int], int]) -> 'Point':
        return Point(f(self.x), f(self.y))

    def __abs__(self) -> 'Point':
        return self.map(abs)

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __eq__(self, other: 'Point') -> bool:
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __neg__(self) -> 'Point':
        return Point(-self.x, -self.y)

    def __str__(self) -> str:
        return Template('($x, $y)').substitute(x=self.x, y=self.y)


class Cell:
    def __init__(self, position: Point, halite: float, ship_id: Optional[ShipId], shipyard_id: Optional[ShipyardId], board: 'Board') -> None:
        self.position = position
        self.halite = halite
        self.ship_id = ship_id
        self.shipyard_id = shipyard_id
        self.board = board


class Ship:
    def __init__(self, ship_id: ShipId, position: Point, halite: int, player_id: PlayerId, board: 'Board') -> None:
        self.ship_id = ship_id
        self.position = position
        self.halite = halite
        self.player_id = player_id
        self.board = board


class Shipyard:
    def __init__(self, shipyard_id: ShipyardId, position: Point, player_id: PlayerId, board: 'Board') -> None:
        self.shipyard_id = shipyard_id
        self.position = position
        self.player_id = player_id
        self.board = board


class Player:
    def __init__(self, player_id: PlayerId, halite: int, shipyard_ids: Set[ShipyardId], ship_ids: Set[ShipId], board: 'Board') -> None:
        self.player_id = player_id
        self.halite = halite
        self.shipyard_ids = shipyard_ids
        self.ship_ids = ship_ids
        self.board = board


class Board:
    def __init__(self, observation: Observation, configuration: Configuration) -> None:
        size = configuration.size
        self.players: Dict[PlayerId, Player] = {}
        self.ships: Dict[ShipId, Ship] = {}
        self.shipyards: Dict[ShipyardId, Shipyard] = {}
        self.cells: Dict[Point, Cell] = {}
        for (player_id, player) in observation.players.items():
            # We know the length of player is always 3 based on the schema -- this is a tuple in json
            [player_halite, shipyards, ships] = player
            self.players[player_id] = Player(player_id, player_halite, shipyards.keys(), ships.keys(), self)
            for (ship_id, [ship_position, ship_halite]) in ships.items():
                self.ships[ship_id] = Ship(ship_id, ship_position, ship_halite, player_id, self)
            for (shipyard_id, shipyard_position) in shipyards.items():
                self.shipyards[shipyard_id] = Shipyard(shipyard_id, shipyard_position, player_id, self)
        ships_by_position = {ship.position: ship.ship_id for ship in self.ships}
        shipyards_by_position = {shipyard.position: shipyard.shipyard_id for shipyard in self.shipyards}
        for x in range(size):
            for y in range(size):
                position = Point(x, y)
                index = size * x + y
                halite = observation.halite[index]
                self.cells[position] = Cell(position, halite, ships_by_position.get(position), shipyards_by_position.get(position), self)
