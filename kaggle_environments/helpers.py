from __future__ import annotations

import operator
import random
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterable, Type, TypeVar


class Point(tuple):
    """
    This type wraps Tuple[int, int] to provide additional operators and convenience members.
    Point are expressed in the form (x, y) where x is the board column and y is the row.
    (0, 0) is the lower left corner of the board and (size - 1, size - 1) is the upper right corner of the board.
    Note that this differs from arrays where the top left is (0, 0) and the bottom right is (size - 1, size - 1).
    Note that operators in this class do not constrain points to the board.
    You can generally constrain a point to the board by calling point % board.configuration.size.
    """

    def __new__(cls: Type["Point"], x: int, y: int):
        return super(Point, cls).__new__(cls, tuple((x, y)))

    @property
    def x(self) -> int:
        return self[0]

    @property
    def y(self) -> int:
        return self[1]

    def map(self, f: Callable[[int], int]) -> "Point":
        return Point(f(self[0]), f(self[1]))

    def map2(self, other: tuple[int, int] | Point, f: Callable[[int, int], int]) -> Point:
        return Point(f(self[0], other[0]), f(self[1], other[1]))

    def translate(self, offset: "Point", size: int) -> "Point":
        """Translates the current point by offset and wraps it around a board of width and height size"""
        return (self + offset) % size

    def distance_to(self, other: "Point", size: int) -> float:
        """Computes total distance (manhattan) to travel to other Point"""
        abs_x = abs(self.x - other.x)
        dist_x = abs_x if abs_x < size / 2 else size - abs_x
        abs_y = abs(self.y - other.y)
        dist_y = abs_y if abs_y < size / 2 else size - abs_y
        return dist_x + dist_y

    def to_index(self, size: int) -> int:
        """
        Converts a 2d position in the form (x, y) to an index in the observation.halite list.
        See staticmethod from_index for the inverse.
        """
        return (size - self.y - 1) * size + self.x

    @staticmethod
    def from_index(index: int, size: int) -> "Point":
        """
        Converts an index in the observation.halite list to a 2d position in the form (x, y).
        See Point method to_index for the inverse.
        """
        y, x = divmod(index, size)
        return Point(x, (size - y - 1))

    def __abs__(self) -> "Point":
        return self.map(operator.abs)

    def __add__(self, other: tuple[int, int] | Point) -> Point:
        return self.map2(other, operator.add)

    def __eq__(self, other: tuple[int, int] | Point) -> bool:
        try:
            return self[0] == other[0] and self[1] == other[1]
        except (TypeError, IndexError):
            return False

    def __floordiv__(self, denominator: int) -> "Point":
        return self.map(lambda x: x // denominator)

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __mod__(self, mod: int) -> "Point":
        return self.map(lambda x: x % mod)

    def __mul__(self, factor: int) -> "Point":
        return self.map(lambda x: x * factor)

    def __neg__(self) -> "Point":
        return self.map(operator.neg)

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __sub__(self, other: tuple[int, int] | Point) -> Point:
        return self.map2(other, operator.sub)


class Direction(Enum):
    NORTH = auto()
    EAST = auto()
    SOUTH = auto()
    WEST = auto()

    def to_point(self) -> Point | None:
        """
        This returns the position offset associated with a particular action
        NORTH -> (0, 1)
        EAST -> (1, 0)
        SOUTH -> (0, -1)
        WEST -> (-1, 0)
        """
        return (
            Point(0, 1)
            if self == Direction.NORTH
            else Point(1, 0)
            if self == Direction.EAST
            else Point(0, -1)
            if self == Direction.SOUTH
            else Point(-1, 0)
            if self == Direction.WEST
            else None
        )

    def __str__(self) -> str:
        return self.name

    def to_index(self) -> int | None:
        return (
            0
            if self == Direction.NORTH
            else 1
            if self == Direction.EAST
            else 2
            if self == Direction.SOUTH
            else 3
            if self == Direction.WEST
            else None
        )

    def to_char(self) -> str | None:
        return (
            "N"
            if self == Direction.NORTH
            else "E"
            if self == Direction.EAST
            else "S"
            if self == Direction.SOUTH
            else "W"
            if self == Direction.WEST
            else None
        )

    def opposite(self) -> "Direction" | None:
        return (
            Direction.SOUTH
            if self == Direction.NORTH
            else Direction.WEST
            if self == Direction.EAST
            else Direction.NORTH
            if self == Direction.SOUTH
            else Direction.EAST
            if self == Direction.WEST
            else None
        )

    def rotate_left(self) -> "Direction" | None:
        return (
            Direction.WEST
            if self == Direction.NORTH
            else Direction.NORTH
            if self == Direction.EAST
            else Direction.EAST
            if self == Direction.SOUTH
            else Direction.SOUTH
            if self == Direction.WEST
            else None
        )

    def rotate_right(self) -> "Direction" | None:
        return (
            Direction.EAST
            if self == Direction.NORTH
            else Direction.SOUTH
            if self == Direction.EAST
            else Direction.WEST
            if self == Direction.SOUTH
            else Direction.NORTH
            if self == Direction.WEST
            else None
        )

    @staticmethod
    def from_str(str_dir: str) -> "Direction" | None:
        return (
            Direction.NORTH
            if str_dir == "NORTH"
            else Direction.EAST
            if str_dir == "EAST"
            else Direction.SOUTH
            if str_dir == "SOUTH"
            else Direction.WEST
            if str_dir == "WEST"
            else None
        )

    @staticmethod
    def from_char(str_char: str) -> "Direction" | None:
        return (
            Direction.NORTH
            if str_char == "N"
            else Direction.EAST
            if str_char == "E"
            else Direction.SOUTH
            if str_char == "S"
            else Direction.WEST
            if str_char == "W"
            else None
        )

    @staticmethod
    def from_index(idx: int) -> "Direction" | None:
        return (
            Direction.NORTH
            if idx == 0
            else Direction.EAST
            if idx == 1
            else Direction.SOUTH
            if idx == 2
            else Direction.WEST
            if idx == 3
            else None
        )

    @staticmethod
    def random_direction() -> "Direction":
        rand = random.random()
        if rand <= 0.25:
            return Direction.NORTH
        elif rand <= 0.5:
            return Direction.EAST
        elif rand <= 0.75:
            return Direction.SOUTH
        else:
            return Direction.WEST

    @staticmethod
    def list_directions() -> list["Direction"]:
        return [
            Direction.NORTH,
            Direction.EAST,
            Direction.SOUTH,
            Direction.WEST,
        ]


TItem = TypeVar("TItem")
THash = TypeVar("THash")


def group_by(items: Iterable[TItem], selector: Callable[[TItem], THash]) -> dict[THash, list[TItem]]:
    results = {}
    for item in items:
        key = selector(item)
        if key not in results:
            results[key] = []
        results[key].append(item)
    return results


def histogram(items: Iterable[TItem]) -> dict[TItem, int]:
    """Accepts a list of hashable items and returns a dictionary where the keys are items and the values are counts of each item in the list."""
    results = {}
    for item in items:
        if item not in results:
            results[item] = 1
        else:
            results[item] += 1
    return results


class Observation(Dict[str, any]):
    """
    Observation provides access to per-step parameters in the environment.
    """

    @property
    def step(self) -> int:
        """Current step within the episode."""
        return self["step"]

    @property
    def remaining_overage_time(self) -> float:
        """Total remaining banked time (seconds) that can be used in excess of per-step actTimeouts -- agent is disqualified with TIMEOUT status when this drops below 0."""
        return self["remainingOverageTime"]


class Configuration(dict[str, Any]):
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
