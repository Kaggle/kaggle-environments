from typing import *


# See https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/halite/halite.json for schema


TKey = TypeVar('TKey')
TValue = TypeVar('TValue')


class ReadOnlyDict(Generic[TKey, TValue]):
    def __init__(self, data: Dict[TKey, TValue]):
        self._data = data

    def __getitem__(self, item) -> Optional[TValue]:
        return self._data.get(item)

    def __iter__(self):
        return self._data.__iter__()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()


class Observation(ReadOnlyDict[str, any]):
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


class Configuration(ReadOnlyDict[str, any]):
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
Point = NewType('Point', Tuple[int, int])


class Cell:
    def __init__(self, position: Point, halite: float, shipyard_id: Optional[ShipyardId], ship_id: Optional[ShipId], board: 'Board') -> None:
        self._position = position
        self._halite = halite
        self._shipyard_id = shipyard_id
        self._ship_id = ship_id
        self._board = board

    @property
    def position(self) -> Point:
        return self._position

    @property
    def halite(self) -> float:
        return self._halite

    @property
    def shipyard_id(self) -> Optional[ShipyardId]:
        return self._shipyard_id

    @property
    def ship_id(self) -> Optional[ShipId]:
        return self._ship_id

    @property
    def ship(self) -> Optional['Ship']:
        return (
            self._board.ships[self.ship_id]
            if self.ship_id is not None
            else None
        )

    @property
    def shipyard(self) -> Optional['Shipyard']:
        return (
            self._board.shipyards[self.shipyard_id]
            if self.shipyard_id is not None
            else None
        )

    def _get_relative_cell(self, x_offset: int, y_offset: int) -> 'Cell':
        size = self._board.configuration.size
        (x, y) = self.position
        (x, y) = (x + x_offset, y + y_offset)
        (x, y) = (x % size, y % size)
        return self._board.cells[(x + x_offset, y + y_offset)]

    @property
    def north(self) -> 'Cell':
        return self._get_relative_cell(0, 1)

    @property
    def south(self) -> 'Cell':
        return self._get_relative_cell(0, -1)

    @property
    def east(self) -> 'Cell':
        return self._get_relative_cell(1, 0)

    @property
    def west(self) -> 'Cell':
        return self._get_relative_cell(-1, 0)


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
    def __init__(self, raw_observation: Dict[str, any], raw_configuration: Dict[str, any]) -> None:
        observation = Observation(raw_observation)
        self._configuration = Configuration(raw_configuration)
        size = self._configuration.size

        players: Dict[PlayerId, Player] = {}
        ships: Dict[ShipId, Ship] = {}
        shipyards: Dict[ShipyardId, Shipyard] = {}
        cells: Dict[Point, Cell] = {}

        # We know the length of player is always 3 based on the schema -- this is a tuple in json
        for (player_id, [player_halite, player_shipyards, player_ships]) in enumerate(observation.players):
            player_shipyard_ids = set(player_shipyards.keys())
            player_ship_ids = set(player_ships.keys())
            players[player_id] = Player(player_id, player_halite, player_shipyard_ids, player_ship_ids, self)
            for (ship_id, [ship_index, ship_halite]) in player_ships.items():
                ship_position = divmod(ship_index, size)
                ships[ship_id] = Ship(ship_id, ship_position, ship_halite, player_id, self)
            for (shipyard_id, shipyard_index) in player_shipyards.items():
                shipyard_position = divmod(shipyard_index, size)
                shipyards[shipyard_id] = Shipyard(shipyard_id, shipyard_position, player_id, self)

        ship_ids_by_position = {ship.position: ship.ship_id for ship in ships.values()}
        shipyard_ids_by_position = {shipyard.position: shipyard.shipyard_id for shipyard in shipyards.values()}
        for x in range(size):
            for y in range(size):
                position = (x, y)
                index = size * x + y
                halite = observation.halite[index]
                ship_id = ship_ids_by_position.get(position)
                shipyard_id = shipyard_ids_by_position.get(position)
                cells[position] = Cell(position, halite, shipyard_id, ship_id, self)

        self._players = ReadOnlyDict(players)
        self._ships = ReadOnlyDict(ships)
        self._shipyards = ReadOnlyDict(shipyards)
        self._cells = ReadOnlyDict(cells)

    @property
    def configuration(self) -> Configuration:
        return self._configuration

    @property
    def players(self) -> ReadOnlyDict[PlayerId, Player]:
        return self._players

    @property
    def ships(self) -> ReadOnlyDict[ShipId, Ship]:
        return self._ships

    @property
    def shipyards(self) -> ReadOnlyDict[ShipyardId, Shipyard]:
        return self._shipyards

    @property
    def cells(self) -> ReadOnlyDict[Point, Cell]:
        return self._cells

    def __str__(self):
        size = self.configuration.size
        horizontal_line = '-' * (self.configuration.size * 4 + 1) + '\n'
        result = horizontal_line
        for x in range(size):
            for y in range(size):
                position = (x, y)
                cell = self.cells[position]
                result += '|'
                result += (
                    chr(ord('a') + cell.ship.player_id)
                    if cell.ship is not None
                    else ' '
                )
                normalized_halite = int(9.0 * cell.halite / float(self.configuration.max_cell_halite))
                result += str(normalized_halite)
                result += (
                    chr(ord('A') + cell.shipyard.player_id)
                    if cell.shipyard is not None
                    else ' '
                )
            result += '|\n'
        return result + horizontal_line
