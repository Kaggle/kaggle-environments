from enum import Enum, Flag, auto
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

    def __str__(self):
        return self._data.__str__()

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


def add_points(left: Point, right: Point):
    (x1, y1) = left
    (x2, y2) = right
    return x1 + x2, y1 + y2


def mod_point(point: Point, mod: int):
    (x, y) = point
    return x % mod, y % mod


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
        """This will convert a ShipAction into an action string that's recognizable by the Halite interpreter"""
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
        """Returns the cell at position + offset"""
        (x, y) = add_points(self.position, offset)
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


class CollisionBehavior(Flag):
    """
    CollisionBehaviors are created and passed into Ship.try_move_* methods
    Ship.try_move_* returns False when the associated movement would cause a collision not allowed by the CollisionBehavior
    CollisionBehavior.DEFAULT allows collisions with weaker opponents and all shipyards
    e.g. If you have a weaker opponent ship north of you
        ship.try_move_north(CollisionBehavior.OPPONENT_SHIPS) == True
        ship.try_move_north(CollisionBehavior.ALLIES) == False
        ship.try_move_north(CollisionBehavior.SHIPYARD) == False
        ship.try_move_north(CollisionBehavior.WEAKER_OPPONENT_SHIPS | CollisionBehavior.OPPONENT_SHIPYARDS) == True
    """
    NONE = 0
    """Prevents your ship from colliding with any ship or shipyard"""
    ALLY_SHIPS = auto()
    """Allows your ship to collide with your ships, one ship will be destroyed"""
    ALLY_SHIPYARDS = auto()
    """Allows your ship to collide with your shipyards, the shipyard will be unable to SPAWN until the ship moves"""
    STRONGER_OPPONENT_SHIPS = auto()
    """Allows your ship to collide with stronger opponent ships, your ship will be destroyed"""
    EQUAL_OPPONENT_SHIPS = auto()
    """Allows your ship to collide with equal opponent ships, both ships will be destroyed"""
    WEAKER_OPPONENT_SHIPS = auto()
    """Allows your ship to collide with weaker opponent ships, the opponent ship will be destroyed"""
    OPPONENT_SHIPS = STRONGER_OPPONENT_SHIPS | EQUAL_OPPONENT_SHIPS | WEAKER_OPPONENT_SHIPS
    """Allows your ship to collide with opponent ships, the weaker ship will be destroyed"""
    OPPONENT_SHIPYARDS = auto()
    """Allows your ship to collide with opponent shipyards, both ship and shipyard will be destroyed"""
    SHIPS = ALLY_SHIPS | OPPONENT_SHIPS
    """Allows your ship to collide with any ship, the weaker ship will be destroyed"""
    SHIPYARDS = ALLY_SHIPYARDS | OPPONENT_SHIPYARDS
    """Allows your ship to collide with any shipyard, both ship and shipyard will be destroyed"""
    ALLIES = ALLY_SHIPS | ALLY_SHIPYARDS
    """Allows your ship to collide with your ships and shipyards"""
    OPPONENTS = OPPONENT_SHIPS | OPPONENT_SHIPYARDS
    """Allows your ship to collide with opponent ships and shipyards"""
    ALL = ALLIES | OPPONENTS
    """No collision guards will be applied to your ship, your ship can collide with any ship or shipyard"""
    DEFAULT = WEAKER_OPPONENT_SHIPS | EQUAL_OPPONENT_SHIPS | OPPONENT_SHIPYARDS | ALLY_SHIPYARDS
    """Allows your ship to collide with opponent ships or any shipyard"""


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
        """This is the action that will be executed by this ship when the current player ends their turn"""
        return self._pending_action

    def try_set_pending_action(self, action: Optional[ShipAction], collision_behavior: CollisionBehavior = CollisionBehavior.DEFAULT) -> bool:
        """
        This method does nothing and returns False if this ship is not owned by the current player
        This method also returns False when the action causes a collision that would violate the passed CollisionBehavior
        You can pass CollisionBehavior.ALL to skip collision checks and allow this ship to collide with any object
        The default is CollisionBehavior.DEFAULT which prevents crashing into stronger opponent ships or your own ships
        """
        if not self.player.is_current_player:
            # This ship is not able to move on this turn because it is owned by a different player
            return False
        if action is not None:
            offset = action.to_point()
            if offset is None:
                # action.to_point() returns None for CONVERT so we know this is a CONVERT action
                if self.player.halite < self._board.configuration.convert_cost:
                    # Not enough halite to build a shipyard
                    return False
            else:
                is_valid_move = True
                destination = self.cell.neighbor(offset)
                # Ensure the correct collision flags are set to allow us to collide with the ship / shipyard at the destination
                if destination.ship is not None:
                    if destination.ship.player.is_current_player:
                        is_valid_move &= bool(collision_behavior & CollisionBehavior.ALLY_SHIPS)
                    elif destination.ship.halite > self.halite:
                        is_valid_move &= bool(collision_behavior & CollisionBehavior.WEAKER_OPPONENT_SHIPS)
                    elif destination.ship.halite < self.halite:
                        is_valid_move &= bool(collision_behavior & CollisionBehavior.STRONGER_OPPONENT_SHIPS)
                    else:
                        is_valid_move &= bool(collision_behavior & CollisionBehavior.EQUAL_OPPONENT_SHIPS)
                if destination.shipyard is not None:
                    if destination.shipyard.player.is_current_player:
                        # If the shipyard is spawning a ship, we must also be able to collide with allied ships to move to it
                        if destination.shipyard.pending_spawn:
                            is_valid_move &= bool(collision_behavior & CollisionBehavior.ALLY_SHIPS)
                        is_valid_move &= bool(collision_behavior & CollisionBehavior.ALLY_SHIPYARDS)
                    else:
                        is_valid_move &= bool(collision_behavior & CollisionBehavior.OPPONENT_SHIPYARDS)
                if not is_valid_move:
                    # Collision flags prevent this ship from colliding with something at the destination
                    return False
        self._pending_action = action
        return True


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
        """This returns True if this shipyard will attempt to spawn a ship when the current player ends their turn"""
        return self._pending_spawn

    def try_set_pending_spawn(self, pending_spawn: bool, prevent_collision: bool = True):
        """This orders the shipyard to spawn a ship when the current player ends their turn"""
        if pending_spawn:
            if prevent_collision and self.cell.ship is not None:
                # Spawning would cause a collision and collisions are prevented
                return False
            if self.player.halite < self._board.configuration.spawn_cost:
                # Not enough halite to spawn a ship
                return False
        self._pending_spawn = pending_spawn
        return True


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
            players[player_id] = Player(player_id, player_halite, player_shipyards.keys(), player_ships.keys(), self)
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
        self._step = observation.step   
        self._current_player_id = observation.player

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
    def current_player_id(self):
        return self._current_player_id

    @property
    def current_player(self) -> Player:
        """
        Returns the player that's selecting actions for their ships and shipyards (generally this is you)
        This property can help you find your ships and shipyards -- try board.current_player.ships
        """
        return self.players[self.current_player_id]

    @property
    def opponents(self) -> List[Player]:
        """
        Returns all players that aren't the current player
        You can get all opponent ships with [ship for ship in player.ships for player in board.opponents]
        """
        return [player for player in self.players if not player.is_current_player]

    @property
    def pending_actions(self) -> Dict[str, str]:
        """Returns all pending ship and shipyard actions formatted for the halite interpreter to receive as an agent response"""
        ship_actions = {ship.ship_id: str(ship.pending_action) for ship in self.ships.values() if ship.pending_action is not None}
        shipyard_actions = {shipyard.shipyard_id: "CONVERT" for shipyard in self.shipyards.values() if shipyard.pending_spawn}
        return {**ship_actions, **shipyard_actions}

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
