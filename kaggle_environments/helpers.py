import operator
import math
from enum import Enum, auto
import random
from typing import *


class Point(tuple):
    """
    This type wraps Tuple[int, int] to provide additional operators and convenience members.
    Point are expressed in the form (x, y) where x is the board column and y is the row.
    (0, 0) is the lower left corner of the board and (size - 1, size - 1) is the upper right corner of the board.
    Note that this differs from arrays where the top left is (0, 0) and the bottom right is (size - 1, size - 1).
    Note that operators in this class do not constrain points to the board.
    You can generally constrain a point to the board by calling point % board.configuration.size.
    """
    def __new__(cls: Type['Point'], x: int, y: int):
        return super(Point, cls).__new__(cls, tuple((x, y)))

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    def map(self, f: Callable[[int], int]) -> 'Point':
        return Point(f(self[0]), f(self[1]))

    def map2(self, other: Union[Tuple[int, int], 'Point'], f: Callable[[int, int], int]) -> 'Point':
        return Point(f(self[0], other[0]), f(self[1], other[1]))

    def translate(self, offset: 'Point', size: int):
        """Translates the current point by offset and wraps it around a board of width and height size"""
        return (self + offset) % size

    def distance_to(self, other: 'Point', size: int):
        """Translates the current point by offset and wraps it around a board of width and height size"""
        abs_x = abs(self.x - other.x)
        dist_x = abs_x if abs_x < size/2 else size - abs_x
        abs_y = abs(self.y - other.y)
        dist_y = abs_y if abs_y < size/2 else size - abs_y
        return dist_x + dist_y

    def to_index(self, size: int):
        """
        Converts a 2d position in the form (x, y) to an index in the observation.halite list.
        See index_to_position for the inverse.
        """
        return (size - self.y - 1) * size + self.x

    @staticmethod
    def from_index(index: int, size: int) -> 'Point':
        """
        Converts an index in the observation.halite list to a 2d position in the form (x, y).
        See position_to_index for the inverse.
        """
        y, x = divmod(index, size)
        return Point(x, (size - y - 1))

    def __abs__(self) -> 'Point':
        return self.map(operator.abs)

    def __add__(self, other: Union[Tuple[int, int], 'Point']) -> 'Point':
        return self.map2(other, operator.add)

    def __eq__(self, other: Union[Tuple[int, int], 'Point']) -> bool:
        try:
            return self[0] == other[0] and self[1] == other[1]
        except (TypeError, IndexError):
            return False

    def __floordiv__(self, denominator: int) -> 'Point':
        return self.map(lambda x: x // denominator)

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __mod__(self, mod: int) -> 'Point':
        return self.map(lambda x: x % mod)

    def __mul__(self, factor: int) -> 'Point':
        return self.map(lambda x: x * factor)

    def __neg__(self) -> 'Point':
        return self.map(operator.neg)

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __sub__(self, other: Union[Tuple[int, int], 'Point']) -> 'Point':
        return self.map2(other, operator.sub)


class Direction(Enum):
    NORTH = auto()
    EAST = auto()
    SOUTH = auto()
    WEST = auto()

    def to_point(self) -> Point:
        """
        This returns the position offset associated with a particular action 
        NORTH -> (0, 1)
        EAST -> (1, 0)
        SOUTH -> (0, -1)
        WEST -> (-1, 0)
        """
        return (
            Point(0, 1) if self == Direction.NORTH else
            Point(1, 0) if self == Direction.EAST else
            Point(0, -1) if self == Direction.SOUTH else
            Point(-1, 0) if self == Direction.WEST else
            None
        )

    def __str__(self) -> str:
        return self.name

    def to_index(self) -> int:
        return (
            0 if self == Direction.NORTH else
            1 if self == Direction.EAST else
            2 if self == Direction.SOUTH else
            3 if self == Direction.WEST else
            None
        )

    def to_char(self) -> str:
        return (
            "N" if self == Direction.NORTH else
            "E" if self == Direction.EAST else
            "S" if self == Direction.SOUTH else
            "W" if self == Direction.WEST else
            None
        )

    def opposite(self) -> 'Direction':
        return (
            Direction.SOUTH if self == Direction.NORTH else
            Direction.WEST if self == Direction.EAST else
            Direction.NORTH if self == Direction.SOUTH else
            Direction.EAST if self == Direction.WEST else
            None
        )

    def rotate_left(self) -> 'Direction':
        return (
            Direction.WEST if self == Direction.NORTH else
            Direction.NORTH if self == Direction.EAST else
            Direction.EAST if self == Direction.SOUTH else
            Direction.SOUTH if self == Direction.WEST else
            None
        )

    def rotate_right(self) -> 'Direction':
        return (
            Direction.EAST if self == Direction.NORTH else
            Direction.SOUTH if self == Direction.EAST else
            Direction.WEST if self == Direction.SOUTH else
            Direction.NORTH if self == Direction.WEST else
            None
        )

    @staticmethod
    def from_str(str_dir: str):
        return  (
            Direction.NORTH if str_dir == "NORTH" else
            Direction.EAST if str_dir == "EAST" else
            Direction.SOUTH if str_dir == "SOUTH" else
            Direction.WEST if str_dir == "WEST" else
            None
        )

    @staticmethod
    def from_char(str_char: str):
        return  (
            Direction.NORTH if str_char == "N" else
            Direction.EAST if str_char == "E" else
            Direction.SOUTH if str_char == "S" else
            Direction.WEST if str_char == "W" else
            None
        )

    @staticmethod
    def from_index(idx: int):
        return (
            Direction.NORTH if idx == 0 else
            Direction.EAST if idx == 1 else
            Direction.SOUTH if idx == 2 else
            Direction.WEST if idx == 3 else
            None
        )

    @staticmethod
    def random_direction() -> 'Direction':
        rand = random.random()
        if rand <= .25:
            return Direction.NORTH
        elif rand <= .5:
            return Direction.EAST
        elif rand <= .75:
            return Direction.SOUTH
        else:
            return Direction.WEST

    @staticmethod
    def list_directions() -> List['Direction']:
        return [
            Direction.NORTH,
            Direction.EAST,
            Direction.SOUTH,
            Direction.WEST,
        ]


TItem = TypeVar('TItem')
THash = TypeVar('THash')


def group_by(items: Iterable[TItem], selector: Callable[[TItem], THash]) -> Dict[THash, List[TItem]]:
    results = {}
    for item in items:
        key = selector(item)
        if key not in results:
            results[key] = []
        results[key].append(item)
    return results


def histogram(items: Iterable[TItem]) -> Dict[TItem, int]:
    """Accepts a list of hashable items and returns a dictionary where the keys are items and the values are counts of each item in the list."""
    results = {}
    for item in items:
        if item not in results:
            results[item] = 1
        else:
            results[item] += 1
    return results


def with_print(item: TItem) -> TItem:
    """Prints an item and returns it -- useful for debug printing in lambdas and chained functions."""
    print(item)
    return item


class Observation(Dict[str, any]):
    """
    Observation provides access to per-step parameters in the environment.
    """
    @property
    def step(self) -> int:
        """Current step within the episode."""
        return self["step"]

    @property
    def remaining_overage_time(self) -> int:
        """Total remaining banked time (seconds) that can be used in excess of per-step actTimeouts -- agent is disqualified with TIMEOUT status when this drops below 0."""
        return self["remainingOverageTime"]


class Configuration(Dict[str, any]):
    """
    Configuration provides access to tunable parameters in the environment.
    """
    @property
    def episode_steps(self) -> int:
        """Total number of steps/turns in the run."""
        return self["episodeSteps"]

    @property
    def act_timeout(self) -> float:
        """Maximum runtime (seconds) to obtain an action from an agent."""
        return self["actTimeout"]

    @property
    def run_timeout(self) -> float:
        """Maximum runtime (seconds) of an episode (not necessarily DONE)."""
        return self["runTimeout"]


TConfiguration = TypeVar('TConfiguration', bound=Configuration)
TObservation = TypeVar('TObservation', bound=Observation)
TAction = TypeVar('TAction')
Agent = Callable[[TObservation, TConfiguration], TAction]


class AgentStatus(Enum):
    UNKNOWN = 0
    ACTIVE = 1
    INACTIVE = 2
    DONE = 3
    INVALID = 4
    ERROR = 5


class AgentState(Generic[TObservation, TAction], Dict[str, any]):
    @property
    def observation(self) -> TObservation:
        return self["observation"]

    @property
    def action(self) -> TAction:
        return self["action"]

    @property
    def reward(self) -> int:
        return self["reward"]

    @property
    def status(self) -> AgentStatus:
        status = self["status"]
        if status in AgentStatus.__members__:
            return AgentStatus[status]
        return AgentStatus.UNKNOWN


class Environment(Generic[TConfiguration, TObservation, TAction]):
    @property
    def specification(self) -> Dict[str, any]:
        raise NotImplemented()

    def interpret(self, configuration: TConfiguration, state: List[AgentState[TObservation, TAction]]) -> List[AgentState[TObservation, TAction]]:
        raise NotImplemented()

    def render_html(self, configuration: TConfiguration, state: List[AgentState[TObservation, TAction]]) -> str:
        raise NotImplemented()

    def render_text(self, configuration: TConfiguration, state: List[AgentState[TObservation, TAction]]) -> str:
        raise NotImplemented()

    @property
    def builtin_agents(self) -> Dict[str, Agent]:
        """Override this property to provide default agents that can be referenced by name in this environment, e.g. `{"random": my_random_agent}`"""
        return {}
