import copy
import json
import operator
from enum import Enum, auto
from typing import *

from .errors import InvalidArgument
from .utils import get, has, process_schema, schemas, structify


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
    results = {}
    for item in items:
        if item not in results:
            results[item] = 1
        else:
            results[item] += 1
    return results


def with_print(value: TItem) -> TItem:
    print(value)
    return value


class Observation(Dict[str, Any]):
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


class Configuration(Dict[str, Any]):
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

    @property
    def agent_count(self) -> int:
        """Total number of agents that will participate in the episode -- must be a value from environment.specification.agents."""
        return self["agentCount"]


class Log(Dict[str, Any]):
    @property
    def duration(self) -> int:
        return self["action"]

    @property
    def stdout(self) -> str:
        return self["action"]

    @property
    def stderr(self) -> str:
        return self["action"]


TConfiguration = TypeVar('TConfiguration', bound=Configuration)
TObservation = TypeVar('TObservation', bound=Observation)
TAction = TypeVar('TAction')
Agent = Callable[[TObservation, TConfiguration], TAction]


class AgentStatus(Enum):
    UNKNOWN = auto()
    INACTIVE = auto()
    ACTIVE = auto()
    DONE = auto()
    ERROR = auto()
    INVALID = auto()
    TIMEOUT = auto()


class State(Generic[TObservation, TAction], Dict[str, Any]):
    @property
    def action(self) -> TAction:
        return self["action"]

    @action.setter
    def action(self, action: TAction):
        self["action"] = action

    @property
    def reward(self) -> int:
        return self["reward"]

    @reward.setter
    def reward(self, reward: int):
        self["reward"] = reward

    @property
    def info(self) -> Dict[str, Any]:
        return self["info"]

    @info.setter
    def info(self, info: Dict[str, Any]):
        self["info"] = info

    @property
    def observation(self) -> TObservation:
        return self["observation"]

    @observation.setter
    def observation(self, observation: TObservation):
        self["observation"] = observation

    @property
    def status(self) -> AgentStatus:
        return AgentStatus.__members__.get(self["status"]) or AgentStatus.UNKNOWN

    @status.setter
    def status(self, status: AgentStatus):
        self["status"] = status.name


TState = TypeVar('TState', bound=State[TObservation, TAction])


class Environment(Generic[TState, TConfiguration]):
    """This class represents the base interface for an environment compatible with kaggle-environments."""
    @property
    def specification(self) -> Dict[str, Any]:
        raise NotImplemented()

    def reset(self, default_state: TState, configuration: TConfiguration) -> TState:
        raise NotImplemented()

    def step(self, state: TState, configuration: TConfiguration) -> TState:
        raise NotImplemented()

    def builtin_agents(self) -> Dict[str, Agent]:
        raise NotImplemented()

    def render_text(self, configuration: TConfiguration, state: List[TState]) -> str:
        raise NotImplemented()

    def render_html(self, configuration: TConfiguration) -> str:
        raise NotImplemented()


class BaseEnvironment(Environment[TState, TConfiguration]):
    """This class provides helpful implementations for part of the Environment interface."""
    def __init__(self, specification_path: str):
        self._specification = BaseEnvironment.__load_specification(specification_path)

    @property
    def specification(self) -> Dict[str, Any]:
        return self._specification

    @staticmethod
    def __load_specification(specification_file_path: str) -> Dict[str, Any]:
        """Create a default specification file from schema.json and merge in a json specification file on disk."""
        with open(specification_file_path) as json_file:
            specification = json.load(json_file)

        # Allow environments to extend various parts of the specification.
        def extend_specification(source, field_name):
            field = copy.deepcopy(source[field_name]["properties"])
            for key, value in get(specification, dict, {}, [field_name]).items():
                # The override is a literal value, use it as the default value in the specification.
                if not isinstance(value, dict):
                    field[key]["default"] = value
                # The override already exists in the specification, merge it in.
                elif key in field:
                    for inner_key, inner_value in value.items():
                        field[key][inner_key] = inner_value
                # The override is not yet in the specification, add it in.
                else:
                    field[key] = value

            specification[field_name] = field

        extend_specification(schemas, "configuration")
        extend_specification(schemas["state"]["properties"], "observation")
        return structify(process_schema(schemas.specification, specification))