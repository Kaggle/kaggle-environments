from copy import deepcopy
from enum import Enum, auto
from typing import *
import json
import math


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

    def __str__(self):
        return self._data.__str__()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def json(self):
        return json.dumps(self._data)


class Observation(ReadOnlyDict[str, any]):
    @property
    def halite(self) -> List[float]:
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


def translate_point(point: Point, offset: Point):
    (x1, y1) = point
    (x2, y2) = offset
    return x1 + x2, y1 + y2


def mod_point(point: Point, mod: int):
    (x, y) = point
    return x % mod, y % mod


def point_to_index(point: Point, size: int):
    """This method translates a 2d point in the form (x, y) to an index in the observation.halite list  """
    (x, y) = point
    return x * size + y


TElement = TypeVar('TElement')
THash = TypeVar('TComparable')


def group_by(elements: Iterable[TElement], selector: Callable[[TElement], THash]):
    results = {}
    for element in elements:
        key = selector(element)
        if key not in results:
            results[key] = []
        results[key].append(element)
    return results


class ShipAction(Enum):
    NORTH = auto()
    SOUTH = auto()
    EAST = auto()
    WEST = auto()
    CONVERT = auto()

    def to_point(self) -> Optional[Point]:
        """
        This returns the position offset associated with a particular action or None if the action does not change the ship's position
        Note that the y axis is inverted so NORTH is downward and SOUTH is upward
        NORTH -> (0, -1)
        SOUTH -> (0, 1)
        EAST -> (1, 0)
        WEST -> (-1, 0)
        """
        return (
            (0, -1) if self == ShipAction.NORTH else
            (0, 1) if self == ShipAction.SOUTH else
            (1, 0) if self == ShipAction.EAST else
            (-1, 0) if self == ShipAction.WEST else
            None
        )

    def __str__(self) -> str:
        return self.name


class ShipyardAction(Enum):
    SPAWN = auto()

    def __str__(self) -> str:
        return self.name


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

    def neighbor(self, offset: Point) -> 'Cell':
        """Returns the cell at self.position + offset"""
        (x, y) = translate_point(self.position, offset)
        return self._board[x, y]

    @property
    def north(self) -> 'Cell':
        return self.neighbor(ShipAction.NORTH.to_point())

    @property
    def south(self) -> 'Cell':
        return self.neighbor(ShipAction.SOUTH.to_point())

    @property
    def east(self) -> 'Cell':
        return self.neighbor(ShipAction.EAST.to_point())

    @property
    def west(self) -> 'Cell':
        return self.neighbor(ShipAction.WEST.to_point())


class Ship:
    def __init__(self, ship_id: ShipId, position: Point, halite: int, player_id: PlayerId, board: 'Board') -> None:
        self._ship_id = ship_id
        self._position = position
        self._halite = halite
        self._player_id = player_id
        self._board = board
        self._pending_action = None

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
        return self._board[self.position]

    @property
    def player(self) -> 'Player':
        return self._board.players[self.player_id]

    @property
    def pending_action(self) -> Optional[ShipAction]:
        """The action that will be executed by this ship when the turn ends"""
        return self._pending_action

    @pending_action.setter
    def pending_action(self, value) -> None:
        self._pending_action = value


class Shipyard:
    def __init__(self, shipyard_id: ShipyardId, position: Point, player_id: PlayerId, board: 'Board') -> None:
        self._shipyard_id = shipyard_id
        self._position = position
        self._player_id = player_id
        self._board = board
        self._pending_spawn = False

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
        return self._board[self.position]

    @property
    def player(self) -> 'Player':
        return self._board.players[self.player_id]

    @property
    def pending_spawn(self) -> bool:
        """Returns True if this shipyard will attempt to spawn a ship when the turn ends"""
        return self._pending_spawn

    @pending_spawn.setter
    def pending_spawn(self, value) -> None:
        self._pending_spawn = value


class Player:
    def __init__(self, player_id: PlayerId, halite: int, shipyard_ids: List[ShipyardId], ship_ids: List[ShipId], board: 'Board') -> None:
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
    def shipyard_ids(self) -> List[ShipyardId]:
        return self._shipyard_ids

    @property
    def ship_ids(self) -> List[ShipId]:
        return self._ship_ids

    @property
    def shipyards(self) -> List[Shipyard]:
        return [self._board.shipyards[shipyard_id] for shipyard_id in self.shipyard_ids]

    @property
    def ships(self) -> List[Ship]:
        return [self._board.ships[ship_id] for ship_id in self.ship_ids]

    @property
    def is_current_player(self) -> bool:
        return self.player_id is self._board.current_player_id

    @property
    def agent_actions(self) -> Dict[str, str]:
        """Returns all pending ship and shipyard actions for this player formatted for the halite interpreter to receive as an agent response"""
        ship_actions = {
            ship.ship_id: str(ship.pending_action)
            for ship in self.ships
            if ship.pending_action is not None
        }
        shipyard_actions = {
            shipyard.shipyard_id: str(ShipyardAction.SPAWN)
            for shipyard in self.shipyards
            if shipyard.pending_spawn
        }
        return {**ship_actions, **shipyard_actions}


class Board:
    def __init__(self, observation: Union[Configuration, Dict[str, any]], configuration: Union[Configuration, Dict[str, any]], actions: Optional[Dict[str, str]]) -> None:
        if actions is None:
            actions = {}
        if isinstance(observation, dict):
            observation = Observation(observation)
        if isinstance(configuration, dict):
            configuration = Configuration(configuration)
        self._step = observation.step
        self._current_player_id = observation.player
        self._configuration: Configuration = configuration
        size = self._configuration.size

        players: Dict[PlayerId, Player] = {}
        ships: Dict[ShipId, Ship] = {}
        shipyards: Dict[ShipyardId, Shipyard] = {}
        cells: Dict[Point, Cell] = {}

        # We know the length of player is always 3 based on the schema -- this is a hack to have a tuple in json
        for (player_id, [player_halite, player_shipyards, player_ships]) in enumerate(observation.players):
            players[player_id] = Player(player_id, player_halite, list(player_shipyards.keys()), list(player_ships.keys()), self)
            for (ship_id, [ship_index, ship_halite]) in player_ships.items():
                ship_position = divmod(ship_index, size)
                ships[ship_id] = Ship(ship_id, ship_position, ship_halite, player_id, self)
                if ship_id in actions:
                    ships[ship_id].pending_action = ShipAction[actions[ship_id]]
            for (shipyard_id, shipyard_index) in player_shipyards.items():
                shipyard_position = divmod(shipyard_index, size)
                shipyards[shipyard_id] = Shipyard(shipyard_id, shipyard_position, player_id, self)
                if shipyard_id in actions:
                    shipyards[shipyard_id].pending_spawn = True

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

    @property
    def step(self) -> int:
        return self._step

    @property
    def current_player_id(self):
        return self._current_player_id

    @property
    def current_player(self):
        return self._players[self.current_player_id]

    @property
    def opponents(self) -> List[Player]:
        """
        Returns all players that aren't the current player
        You can get all opponent ships with [ship for ship in player.ships for player in board.opponents]
        """
        return [player for player in self.players if not player.is_current_player]

    def simulate_actions(self) -> 'Board':
        """
        Returns a new board with the current board's pending actions applied
        The current board is unmodified
        """
        convert_cost = self.configuration.convert_cost
        spawn_cost = self.configuration.spawn_cost

        # This is the stored halite total for each player after all actions have processed
        players: Dict[PlayerId, int] = {}
        ships: List[Ship] = []
        shipyards: List[Shipyard] = []

        uid_counter = 0

        def create_uid():
            nonlocal uid_counter
            uid_counter += 1
            return f"{self.step}-{uid_counter}"

        # Process actions and store the results in the ships and shipyards lists for collision checking
        for player in self.players.values():
            player_halite = player.halite
            leftover_convert_halite = 0

            for shipyard in player.shipyards:
                shipyards.append(shipyard)
                if shipyard.pending_spawn and player_halite > spawn_cost:
                    player_halite -= spawn_cost
                    ship_id = ShipId(create_uid())
                    ship = Ship(ship_id, shipyard.position, 0, player.player_id, self)
                    ships.append(ship)

            for ship in player.ships:
                if ship.pending_action is None:
                    ships.append(ship)
                elif ship.pending_action == ShipAction.CONVERT:
                    if (
                        ship.cell.shipyard_id is None and  # Can't convert on an existing shipyard
                        (ship.halite + player_halite) > convert_cost
                    ):
                        delta_halite = ship.halite - convert_cost
                        # Excess halite leftover from conversion is added to the player's total only after all conversions have completed
                        # This is to prevent the edge case of chaining halite from one convert to fund other converts
                        leftover_convert_halite += max(delta_halite, 0)
                        player_halite += min(delta_halite, 0)
                        shipyard_id = ShipyardId(create_uid())
                        shipyard = Shipyard(shipyard_id, ship.position, player.player_id, self)
                        shipyards.append(shipyard)
                else:
                    # If the action is not None and is not CONVERT it must be NORTH, SOUTH, EAST, or WEST
                    offset = ship.pending_action.to_point()
                    ship = Ship(ship.ship_id, translate_point(ship.position, offset), ship.halite, ship.player_id, self)
                    ships.append(ship)

            player_halite += leftover_convert_halite
            players[player.player_id] = player_halite

        def get_collision_winner(ships: List[Ship]) -> Optional[Ship]:
            """
            Accepts the list of ships at a particular position
            Returns the ship with the least halite or None in the case of a tie
            """
            if len(ships) == 1:
                return ships[0]
            ships_by_halite = group_by(ships, lambda ship: ship.halite)
            smallest_halite = min(ships_by_halite.keys())
            smallest_ships = ships_by_halite[smallest_halite]
            if len(smallest_ships) == 1:
                return smallest_ships[0]
            return None

        # Check for collisions
        ships_by_position = {
            position: ship
            for position, group in group_by(ships, lambda ship: ship.position).items()
            if (ship := get_collision_winner(group)) is not None
        }

        for shipyard in shipyards:
            if shipyard.position in ships_by_position:
                ship = ships_by_position[shipyard.position]
                if ship.player_id is not shipyard.player_id:
                    shipyards.remove(shipyard)
                    del ships_by_position[shipyard.position]

        board = deepcopy(self)
        board._ships = ReadOnlyDict({
            ship.ship_id: ship
            for ship in ships_by_position.values()
        })
        board._shipyards = ReadOnlyDict({
            shipyard.shipyard_id: shipyard
            for shipyard in shipyards
        })

        ships_by_player = group_by(board._ships.values(), lambda ship: ship.player_id)
        shipyards_by_player = group_by(board._shipyards.values(), lambda shipyard: shipyard.player_id)
        board._players = ReadOnlyDict({
            player_id: Player(
                player_id,
                player_halite,
                [shipyard.shipyard_id for shipyard in shipyards_by_player.get(player_id) or []],
                [ship.ship_id for ship in ships_by_player.get(player_id) or []],
                board
            )
            for player_id, player_halite in players.items()
        })
        shipyards_by_position = {
            shipyard.position: shipyard
            for shipyard in board._shipyards.values()
        }
        size = board.configuration.size
        board._cells = ReadOnlyDict({
            (position := (x, y)): Cell(
                position,
                min(board[position].halite * (1.0 + board.configuration.regen_rate), board.configuration.max_cell_halite),
                ship.ship_id
                if (ship := ships_by_position.get(position)) is not None
                else None,
                shipyard.shipyard_id
                if (shipyard := shipyards_by_position.get(position)) is not None
                else None,
                board
            )
            for x in range(size)
            for y in range(size)
        })

        board._step += 1
        return board

    def raw(self) -> Dict[str, Any]:
        size = self.configuration.size
        """This converts a Board back to the observation that constructed it."""
        def normalize_player(player: Player):
            shipyards = {
                shipyard.shipyard_id: point_to_index(shipyard.position, size)
                for shipyard in player.shipyards
            }
            ships = {
                ship.ship_id: [point_to_index(ship.position, size), ship.halite]
                for ship in player.ships
            }
            return [player.halite, shipyards, ships]

        halite = [
            self[(x, y)].halite
            for x in range(size)
            for y in range(size)
        ]
        players = [
            normalize_player(player)
            for player in self.players.values()
        ]

        return {
            "halite": halite,
            "players": players,
            "player": self.current_player_id,
            "step": self.step,
        }

    def __deepcopy__(self, _):
        actions = {}
        for player in self.players.values():
            actions = {**actions, **player.agent_actions}
        return Board(self.raw(), self.configuration, actions)

    def __getitem__(self, position: Point) -> Cell:
        """
        This method will wrap the supplied position to fit within the board size and return the cell at that location
        e.g. on a 3x3 board, board[(2, 1)] is the same as board[(5, 1)]
        """
        (x, y) = position
        size = self.configuration.size
        key = (x % size, y % size)
        return self._cells[key]

    def __str__(self):
        """
        The board is printed in a grid with the following rules:
        Capital letters are shipyards
        Lower case letters are ships
        Digits are a scale from 0-9 directly proportional to a value between 0 and self.configuration.max_cell_halite
        Player 1 is letter a
        Player 2 is letter b
        etc.
        """
        size = self.configuration.size
        horizontal_line = '-' * (self.configuration.size * 4 + 1) + '\n'
        result = horizontal_line
        for x in range(size):
            for y in range(size):
                cell = self[(x, y)]
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
