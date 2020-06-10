from copy import deepcopy
from enum import Enum, auto
from typing import *

# region Helper Classes and Methods
Point = NewType('Point', Tuple[int, int])


def translate_point(point: Point, offset: Point) -> Point:
    """Returns (point.x + offset.x, point.y + offset.y)"""
    (x1, y1) = point
    (x2, y2) = offset
    return x1 + x2, y1 + y2


def mod_point(point: Point, mod: int) -> Point:
    """Returns (point.x % mod, point.y % mod)"""
    (x, y) = point
    return x % mod, y % mod


def position_to_index(point: Point, size: int) -> int:
    """
    Converts a 2d position in the form (x, y) to an index in the observation.halite list.
    See index_to_position for the inverse.
    """
    x, y = point
    return y * size + x


def index_to_position(index: int, size: int) -> Point:
    """
    Converts an index in the observation.halite list to a 2d position in the form (x, y).
    See position_to_index for the inverse.
    """
    y, x = divmod(index, size)
    return x, y


def range_2d(size: int) -> List[Point]:
    """Returns all points from (0, 0) to (size - 1, size - 1)."""
    return [
        (x, y)
        for x in range(size)
        for y in range(size)
    ]


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


TKey = TypeVar('TKey')
TValue = TypeVar('TValue')


class ReadOnlyDict(Generic[TKey, TValue]):
    """Offers Dict-like semantics while preventing modification of the underlying datastructure."""
    def __init__(self, data: Union['ReadOnlyDict[TKey, TValue]', Dict[TKey, TValue]]):
        if isinstance(data, dict):
            self._data = data
        else:
            # If it's not a Dict it must be a ReadOnlyDict based on our type
            # Unwrap inner ReadOnlyDict's data
            self._data = data._data

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
# endregion


# region Data Model Classes
class Observation(ReadOnlyDict[str, any]):
    """
    Observation primarily used as a helper to construct the Board from the raw observation.
    This provides bindings for the observation type described at https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/halite/halite.json
    """
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
    """
    Configuration provides access to tunable parameters in the environment.
    This provides bindings for the configuration type described at https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/halite/halite.json
    """
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


class ShipAction(Enum):
    NORTH = auto()
    SOUTH = auto()
    EAST = auto()
    WEST = auto()
    CONVERT = auto()

    def to_point(self) -> Optional[Point]:
        """
        This returns the position offset associated with a particular action or None if the action does not change the ship's position.
        Note that the y axis is inverted so NORTH is downward and SOUTH is upward.
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


ShipId = NewType('ShipId', str)
ShipyardId = NewType('ShipyardId', str)
PlayerId = NewType('PlayerId', int)


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
        """Returns the ship on this cell if it exists and None otherwise."""
        return (
            self._board.ships[self.ship_id]
            if self.ship_id is not None
            else None
        )

    @property
    def shipyard(self) -> Optional['Shipyard']:
        """Returns the shipyard on this cell if it exists and None otherwise."""
        return (
            self._board.shipyards[self.shipyard_id]
            if self.shipyard_id is not None
            else None
        )

    def neighbor(self, offset: Point) -> 'Cell':
        """Returns the cell at self.position + offset."""
        (x, y) = translate_point(self.position, offset)
        return self._board[x, y]

    @property
    def north(self) -> 'Cell':
        """Returns the cell north of this cell."""
        return self.neighbor(ShipAction.NORTH.to_point())

    @property
    def south(self) -> 'Cell':
        """Returns the cell south of this cell."""
        return self.neighbor(ShipAction.SOUTH.to_point())

    @property
    def east(self) -> 'Cell':
        """Returns the cell east of this cell."""
        return self.neighbor(ShipAction.EAST.to_point())

    @property
    def west(self) -> 'Cell':
        """Returns the cell west of this cell."""
        return self.neighbor(ShipAction.WEST.to_point())


class Ship:
    def __init__(self, ship_id: ShipId, position: Point, halite: int, player_id: PlayerId, board: 'Board', next_action: Optional[ShipAction] = None) -> None:
        self._id = ship_id
        self._position = position
        self._halite = halite
        self._player_id = player_id
        self._board = board
        self._next_action = next_action

    @property
    def id(self) -> ShipId:
        return self._id

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
        """Returns the cell this ship is on."""
        return self._board[self.position]

    @property
    def player(self) -> 'Player':
        """Returns the player that owns this ship."""
        return self._board.players[self.player_id]

    @property
    def next_action(self) -> Optional[ShipAction]:
        """Returns the action that will be executed by this ship when Board.next() is called (when the current turn ends)."""
        return self._next_action

    @next_action.setter
    def next_action(self, value) -> None:
        """Sets the action that will be executed by this ship when Board.next() is called (when the current turn ends)."""
        self._next_action = value

    @property
    def observation(self) -> List[int]:
        """Converts a ship back to the normalized observation subset that constructed it."""
        return [position_to_index(self.position, self._board.configuration.size), self.halite]


class Shipyard:
    def __init__(self, shipyard_id: ShipyardId, position: Point, player_id: PlayerId, board: 'Board', next_action: Optional[ShipyardAction] = None) -> None:
        self._id = shipyard_id
        self._position = position
        self._player_id = player_id
        self._board = board
        self._next_action = next_action

    @property
    def id(self) -> ShipyardId:
        return self._id

    @property
    def position(self) -> Point:
        return self._position

    @property
    def player_id(self) -> PlayerId:
        return self._player_id

    @property
    def cell(self) -> Cell:
        """Returns the cell this shipyard is on."""
        return self._board[self.position]

    @property
    def player(self) -> 'Player':
        return self._board.players[self.player_id]

    @property
    def next_action(self) -> ShipyardAction:
        """Returns the action that will be executed by this shipyard when Board.next() is called (when the current turn ends)."""
        return self._next_action

    @next_action.setter
    def next_action(self, value) -> None:
        """Sets the action that will be executed by this shipyard when Board.next() is called (when the current turn ends)."""
        self._next_action = value

    @property
    def observation(self) -> int:
        """Converts a shipyard back to the normalized observation subset that constructed it."""
        return position_to_index(self.position, self._board.configuration.size)


class Player:
    def __init__(self, player_id: PlayerId, halite: int, shipyard_ids: List[ShipyardId], ship_ids: List[ShipId], board: 'Board') -> None:
        self._id = player_id
        self._halite = halite
        self._shipyard_ids = shipyard_ids
        self._ship_ids = ship_ids
        self._board = board

    @property
    def id(self) -> PlayerId:
        return self._id

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
        """Returns all shipyards owned by this player."""
        return [
            self._board.shipyards[shipyard_id]
            for shipyard_id in self.shipyard_ids
        ]

    @property
    def ships(self) -> List[Ship]:
        """Returns all ships owned by this player."""
        return [
            self._board.ships[ship_id]
            for ship_id in self.ship_ids
        ]

    @property
    def is_current_player(self) -> bool:
        """Returns whether this player is the current player (generally if this returns True, this player is you)."""
        return self.id is self._board.current_player_id

    @property
    def next_actions(self) -> Dict[str, str]:
        """Returns all queued ship and shipyard actions for this player formatted for the halite interpreter to receive as an agent response."""
        ship_actions = {
            ship.id: ship.next_action.name
            for ship in self.ships
            if ship.next_action is not None
        }
        shipyard_actions = {
            shipyard.id: shipyard.next_action.name
            for shipyard in self.shipyards
            if shipyard.next_action is not None
        }
        return {**ship_actions, **shipyard_actions}

    @property
    def observation(self):
        """Converts a player back to the normalized observation subset that constructed it."""
        shipyards = {shipyard.id: shipyard.observation for shipyard in self.shipyards}
        ships = {ship.id: ship.observation for ship in self.ships}
        return [self.halite, shipyards, ships]
# endregion


class Board:
    def __init__(
        self,
        raw_observation: Dict[str, any],
        raw_configuration: Union[Configuration, Dict[str, any]],
        next_actions: Optional[List[Dict[str, str]]] = None
    ) -> None:
        """
        Creates a board from the provided observation, configuration, and next_actions.
        Board tracks players (by id), ships (by id), shipyards (by id), and cells (by position).
        Each entity contains both key values (e.g. ship.player_id) as well as entity references (e.g. ship.player).
        References are deep and chainable e.g.
            [ship.halite for player in board.players for ship in player.ships]
            ship.player.shipyards[0].cell.north.east.ship
        """
        observation = Observation(raw_observation)
        # Pending actions is effectively a Dict[Union[[ShipId, ShipAction], [ShipyardId, ShipyardAction]]]
        # but that type's not very expressible so we simplify it to Dict[str, str]
        # Later we'll iterate through it once for each ship and shipyard to pull all the actions out
        next_actions = next_actions or ([{}] * len(observation.players))

        self._step = observation.step
        self._configuration = Configuration(raw_configuration)
        self._current_player_id = observation.player
        self._players: Dict[PlayerId, Player] = {}
        self._ships: Dict[ShipId, Ship] = {}
        self._shipyards: Dict[ShipyardId, Shipyard] = {}
        self._cells: Dict[Point, Cell] = {}

        size = self.configuration.size
        for position in range_2d(size):
            halite = observation.halite[position_to_index(position, size)]
            # We'll populate the cell's ships and shipyards as we loop through them down below
            self.cells[position] = Cell(position, halite, None, None, self)

        for (player_id, player_observation) in enumerate(observation.players):
            # We know the len(player_observation) == 3 based on the schema -- this is a hack to have a tuple in json
            [player_halite, player_shipyards, player_ships] = player_observation
            # We'll populate the player's ships and shipyards as we loop through them down below
            self.players[player_id] = Player(player_id, player_halite, [], [], self)
            player_actions = next_actions[player_id] or {}

            for (ship_id, [ship_index, ship_halite]) in player_ships.items():
                # In the raw observation, halite is stored as a 1d list but we convert it to a 2d dict for convenience
                # Accordingly we also need to convert our list indices to dict keys / 2d positions
                ship_position = index_to_position(ship_index, size)
                action = (
                    ShipAction[player_actions[ship_id]]
                    if ship_id in player_actions
                    else None
                )
                self._add_ship(Ship(ship_id, ship_position, ship_halite, player_id, self, action))

            for (shipyard_id, shipyard_index) in player_shipyards.items():
                shipyard_position = index_to_position(shipyard_index, size)
                action = (
                    ShipyardAction[player_actions[shipyard_id]]
                    if shipyard_id in player_actions
                    else None
                )
                self._add_shipyard(Shipyard(shipyard_id, shipyard_position, player_id, self, action))

    @property
    def configuration(self) -> Configuration:
        return self._configuration

    @property
    def players(self) -> Dict[PlayerId, Player]:
        return self._players

    @property
    def ships(self) -> Dict[ShipId, Ship]:
        """Returns all ships on the current board."""
        return self._ships

    @property
    def shipyards(self) -> Dict[ShipyardId, Shipyard]:
        """Returns all shipyards on the current board."""
        return self._shipyards

    @property
    def cells(self) -> Dict[Point, Cell]:
        """Returns all cells on the current board."""
        return self._cells

    @property
    def step(self) -> int:
        return self._step

    @property
    def current_player_id(self):
        return self._current_player_id

    @property
    def current_player(self):
        """Returns the current player (generally this is you)."""
        return self._players[self.current_player_id]

    @property
    def opponents(self) -> List[Player]:
        """
        Returns all players that aren't the current player.
        You can get all opponent ships with [ship for ship in player.ships for player in board.opponents]
        """
        return [player for player in self.players if not player.is_current_player]

    @property
    def observation(self) -> Dict[str, Any]:
        """Converts a Board back to the normalized observation that constructed it."""
        size = self.configuration.size
        halite = [self[position].halite for position in range_2d(size)]
        players = [player.observation for player in self.players.values()]

        return {
            "halite": halite,
            "players": players,
            "player": self.current_player_id,
            "step": self.step
        }

    def __deepcopy__(self, _):
        actions = [player.next_actions for player in self.players.values()]
        return Board(self.observation, self.configuration, actions)

    def __getitem__(self, position: Point) -> Cell:
        """
        This method will wrap the supplied position to fit within the board size and return the cell at that location.
        e.g. on a 3x3 board, board[(2, 1)] is the same as board[(5, 1)]
        """
        return self._cells[mod_point(position, self.configuration.size)]

    def __str__(self):
        """
        The board is printed in a grid with the following rules:
        Capital letters are shipyards
        Lower case letters are ships
        Digits are cell halite and scale from 0-9 directly proportional to a value between 0 and self.configuration.max_cell_halite
        Player 1 is letter a/A
        Player 2 is letter b/B
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

    def _add_ship(self: 'Board', ship: Ship):
        ship.player.ship_ids.append(ship.id)
        ship.cell._ship_id = ship.id
        self.ships[ship.id] = ship

    def _add_shipyard(self: 'Board', shipyard: Shipyard):
        shipyard.player.shipyard_ids.append(shipyard.id)
        shipyard.cell._shipyard_id = shipyard.id
        self.shipyards[shipyard.id] = shipyard

    def _delete_ship(self: 'Board', ship: Ship):
        ship.player.ship_ids.remove(ship.id)
        if ship.cell.ship_id == ship.id:
            ship.cell._ship_id = None
        del self.ships[ship.id]

    def _delete_shipyard(self: 'Board', shipyard: Shipyard):
        shipyard.player.shipyard_ids.remove(shipyard.id)
        if shipyard.cell.shipyard_id == shipyard.id:
            shipyard.cell._shipyard_id = None
        del self.shipyards[shipyard.id]

    def next(self) -> 'Board':
        """
        Returns a new board with the current board's next actions applied.
        The current board is unmodified.
        This can form a halite interpreter, e.g.
            next_observation = Board(current_observation, configuration, actions).next.observation
        """
        board = deepcopy(self)
        configuration = board.configuration
        convert_cost = configuration.convert_cost
        spawn_cost = configuration.spawn_cost
        uid_counter = 0

        # This is a consistent way to generate unique strings to form ship and shipyard ids
        def create_uid():
            nonlocal uid_counter
            uid_counter += 1
            return f"{self.step}-{uid_counter}"

        # Process actions and store the results in the ships and shipyards lists for collision checking
        for player in board.players.values():
            leftover_convert_halite = 0

            for shipyard in player.shipyards:
                if shipyard.next_action == ShipyardAction.SPAWN and player.halite > spawn_cost:
                    # Handle SPAWN actions
                    player._halite -= spawn_cost
                    board._add_ship(Ship(ShipId(create_uid()), shipyard.position, 0, player.id, board))
                shipyard.next_spawn = False

            for ship in player.ships:
                if ship.next_action == ShipAction.CONVERT:
                    # Can't convert on an existing shipyard but you can use halite in a ship to fund conversion
                    if ship.cell.shipyard_id is None and (ship.halite + player.halite) > convert_cost:
                        # Handle CONVERT actions
                        delta_halite = ship.halite - convert_cost
                        # Excess halite leftover from conversion is added to the player's total only after all conversions have completed
                        # This is to prevent the edge case of chaining halite from one convert to fund other converts
                        leftover_convert_halite += max(delta_halite, 0)
                        player._halite += min(delta_halite, 0)
                        board._add_shipyard(Shipyard(ShipyardId(create_uid()), ship.position, player.id, board))
                        board._delete_ship(ship)
                elif ship.next_action is not None:
                    # If the action is not None and is not CONVERT it must be NORTH, SOUTH, EAST, or WEST
                    ship.cell._ship_id = None
                    ship._position = translate_point(ship.position, ship.next_action.to_point())
                    ship.cell._ship_id = ship.id
                ship.next_action = None

            player._halite += leftover_convert_halite

        def resolve_collision(ships: List[Ship]) -> Tuple[Optional[Ship], List[Ship]]:
            """
            Accepts the list of ships at a particular position (must not be empty).
            Returns the ship with the least halite or None in the case of a tie along with all other ships.
            """
            if len(ships) == 1:
                return ships[0], []
            ships_by_halite = group_by(ships, lambda ship: ship.halite)
            smallest_halite = min(ships_by_halite.keys())
            smallest_ships = ships_by_halite[smallest_halite]
            if len(smallest_ships) == 1:
                # There was a winner, return it
                return smallest_ships[0], smallest_ships[1:]
            # There was a tie for least halite, all are deleted
            return None, ships

        # Check for collisions
        ship_collision_groups = group_by(board.ships.values(), lambda ship: ship.position)
        for position, collided_ships in ship_collision_groups.items():
            winner, deleted = resolve_collision(collided_ships)
            for ship in deleted:
                board._delete_ship(ship)
                if winner is not None:
                    # Winner takes destroyed ships' halite
                    winner._halite += ship.halite
            if winner is not None:
                winner.cell._ship_id = winner.id

        for shipyard in list(board.shipyards.values()):
            ship = shipyard.cell.ship
            if ship is None:
                # No collision
                continue
            elif ship.player_id == shipyard.player_id:
                # Ship depositing halite
                shipyard.player._halite += ship.halite
                ship._halite = 0
            else:
                # Ship to shipyard collision
                board._delete_shipyard(shipyard)
                board._delete_ship(ship)

        for ship in board.ships.values():
            cell = ship.cell
            delta_halite = int(cell.halite * configuration.collect_rate)
            if cell.shipyard_id is None and delta_halite > 1:
                ship._halite += delta_halite
                cell._halite -= delta_halite

        board._step = board.step + 1
        return board


def helper_agent(obs, config):
    board = Board(obs, config)
    for ship in board.current_player.ships:
        if ship.cell.shipyard is None:
            ship.next_action = ShipAction.CONVERT
        else:
            ship.next_action = ShipAction.NORTH
    for shipyard in board.current_player.shipyards:
        shipyard.next_action = ShipyardAction.SPAWN
    return board.current_player.next_actions
