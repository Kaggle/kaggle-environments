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


class Configuration(Object):
    @property
    def episode_steps(self) -> int:
        """Total number of steps/turns in the run."""
        return self["episodeSteps"]

    @property
    def agent_exec(self) -> str:
        """How the agent is executed alongside the running envionment ('LOCAL' or separate 'PROCESS')."""
        return self["agentExec"]

    @property
    def agent_timeout(self) -> float:
        """Maximum runtime (seconds) to initialize an agent."""
        return self["agentTimeout"]

    @property
    def act_timeout(self) -> float:
        """Maximum runtime (seconds) to obtain an action from an agent."""
        return self["actTimeout"]

    @property
    def run_timeout(self) -> float:
        """Maximum runtime (seconds) of an episode (not necessarily DONE)."""
        return self["runTimeout"]

    @property
    def halite(self) -> int:
        """The starting amount of halite available on the board."""
        return self["halite"]

    @property
    def size(self) -> int:
        """The number of cells vertically and horizontally on the board."""
        return self["size"]

    @property
    def spawn_cost(self) -> int:
        """The amount of halite to spawn a new ship."""
        return self["spawnCost"]

    @property
    def convert_cost(self) -> int:
        """The amount of halite to convert a ship into a shipyard."""
        return self["convertCost"]

    @property
    def move_cost(self) -> float:
        """The percent deducted from ship's current halite per move."""
        return self["moveCost"]

    @property
    def collect_rate(self) -> float:
        """The rate of halite collected by a ship from a cell by not moving."""
        return self["collectRate"]

    @property
    def regen_rate(self) -> float:
        """The rate halite regenerates on the board."""
        return self["regenRate"]

    @property
    def max_cell_halite(self) -> int:
        """The maximum halite that can be in any cell."""
        return self["maxCellHalite"]


ShipId = NewType('ShipId', str)
ShipyardId = NewType('ShipyardId', str)
PlayerId = NewType('PlayerId', int)


class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __abs__(self) -> 'Point':
        return Point(abs(self.x), abs(self.y))

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

    def __mul__(self, other) -> 'Point':
        return Point(self.x * other, self.y * other)


class Direction:
    NORTH = Point(0, 1)
    SOUTH = Point(0, -1)
    EAST = Point(1, 0)
    WEST = Point(-1, 0)
    ZERO = Point(0, 0)
    ONE = Point(1, 1)


class Cell:
    def __init__(self, position: Point, halite: float, ship_id: Optional[ShipId], shipyard_id: Optional[ShipyardId], board: 'Board') -> None:
        self._position = position
        self._halite = halite
        self._ship_id = ship_id
        self._shipyard_id = shipyard_id
        self._board = board

    @property
    def position(self) -> Point:
        return self._position

    @property
    def halite(self) -> float:
        return self._halite

    @property
    def ship_id(self) -> Optional[ShipId]:
        return self._ship_id

    @property
    def shipyard_id(self) -> Optional[ShipyardId]:
        return self._shipyard_id

    @property
    def ship(self) -> Optional['Ship']:
        return (
            self._board.ships.get(self.ship_id)
            if self.ship_id is not None
            else None
        )

    @property
    def shipyard(self) -> Optional['Shipyard']:
        return (
            self._board.shipyards.get(self.shipyard_id)
            if self.shipyard_id is not None
            else None
        )

    def neighbor(self, direction: Point) -> 'Cell':
        return self._board.cells[self.position + direction]

    @property
    def north(self) -> 'Cell':
        return self.neighbor(Point.NORTH)

    @property
    def south(self) -> 'Cell':
        return self.neighbor(Point.SOUTH)

    @property
    def east(self) -> 'Cell':
        return self.neighbor(Point.EAST)

    @property
    def west(self) -> 'Cell':
        return self.neighbor(Point.WEST)


class Ship:
    def __init__(self, ship_id: ShipId, position: Point, halite: int, player_id: PlayerId, board: 'Board') -> None:
        self._ship_id = ship_id
        self._position = position
        self._halite = halite
        self._player_id = player_id
        self._board = board

    @property
    def ship_id(self) -> ShipId:
        return self._ship_id

    @property
    def position(self) -> Point:
        return self._position

    @property
    def halite(self) -> int:
        return self._halite

    @property
    def player_id(self) -> PlayerId:
        return self._player_id

    @property
    def cell(self) -> Cell:
        return self._board.cells[self.position]

    @property
    def player(self) -> 'Player':
        return self._board.players[self.player_id]


class Shipyard:
    def __init__(self, shipyard_id: ShipyardId, position: Point, player_id: PlayerId, board: 'Board') -> None:
        self._shipyard_id = shipyard_id
        self._position = position
        self._player_id = player_id
        self._board = board

    @property
    def shipyard_id(self) -> ShipyardId:
        return self._shipyard_id

    @property
    def position(self) -> Point:
        return self._position

    @property
    def player_id(self) -> PlayerId:
        return self._player_id

    @property
    def cell(self) -> Cell:
        return self._board.cells[self.position]

    @property
    def player(self) -> 'Player':
        return self._board.players[self.player_id]


class Player:
    def __init__(self, player_id: PlayerId, halite: int, shipyard_ids: Set[ShipyardId], ship_ids: Set[ShipId], board: 'Board') -> None:
        self._player_id = player_id
        self._halite = halite
        self._shipyard_ids = shipyard_ids
        self._ship_ids = ship_ids
        self._board = board

    @property
    def player_id(self) -> PlayerId:
        return self._player_id

    @property
    def halite(self) -> int:
        return self._halite

    @property
    def shipyard_ids(self) -> Set[ShipyardId]:
        return self._shipyard_ids

    @property
    def ship_ids(self) -> Set[ShipId]:
        return self._ship_ids

    @property
    def shipyards(self) -> Set[Shipyard]:
        return set([self._board.shipyards[shipyard_id] for shipyard_id in self.shipyard_ids])

    @property
    def ships(self) -> Set[Ship]:
        return set([self._board.ships[ship_id] for ship_id in self.ship_ids])


class Board:
    def __init__(self, observation: Observation, configuration: Configuration) -> None:
        self._configuration = configuration
        self._players: Dict[PlayerId, Player] = {}
        self._ships: Dict[ShipId, Ship] = {}
        self._shipyards: Dict[ShipyardId, Shipyard] = {}
        self._cells: Dict[Point, Cell] = {}
        # We know the length of player is always 3 based on the schema -- this is a tuple in json
        for (player_id, [player_halite, shipyards, ships]) in enumerate(observation.players):
            self._players[player_id] = Player(player_id, player_halite, shipyards.keys(), ships.keys(), self)
            for (ship_id, [ship_position, ship_halite]) in ships.items():
                self._ships[ship_id] = Ship(ship_id, ship_position, ship_halite, player_id, self)
            for (shipyard_id, shipyard_position) in shipyards.items():
                self._shipyards[shipyard_id] = Shipyard(shipyard_id, shipyard_position, player_id, self)
        ships_by_position = {ship.position: ship.ship_id for ship in self.ships.values()}
        shipyards_by_position = {shipyard.position: shipyard.shipyard_id for shipyard in self.shipyards.values()}
        size = self._configuration.size
        for x in range(size):
            for y in range(size):
                position = Point(x, y)
                index = size * x + y
                halite = observation.halite[index]
                self._cells[position] = Cell(position, halite, ships_by_position.get(position), shipyards_by_position.get(position), self)

    @property
    def configuration(self):
        return self._configuration

    @property
    def players(self):
        return self._players

    @property
    def ships(self):
        return self._ships

    @property
    def shipyards(self):
        return self._shipyards

    @property
    def cells(self):
        return self._cells

    def __str__(self):
        size = self.configuration.size
        result = ''
        for x in range(size):
            for y in range(size):
                position = Point(x, y)
                cell = self.cells[position]
                result += '.'
            result += '\n'
        return result




















