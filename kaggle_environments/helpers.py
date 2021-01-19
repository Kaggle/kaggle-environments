import copy
import json
import operator
import traceback
from contextlib import redirect_stdout, redirect_stderr
from time import perf_counter

from StringIO import StringIO
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


def trim_end(string: str, to_trim: str):
    while string.endswith(to_trim):
        string = string[:-len(to_trim)]
    return string


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


TResult = TypeVar('TResult')


class Log(Dict[str, Any]):
    @property
    def duration(self) -> float:
        return self["duration"]

    @property
    def stdout(self) -> str:
        return self["stdout"]

    @property
    def stderr(self) -> str:
        return self["stderr"]

    @staticmethod
    def collect(function: Callable[[], TResult], print_std: bool = False) -> Tuple[Union[TResult, Exception], 'Log']:
        """This function aggregates stdout, stderr, duration, and exception stack trace (if applicable) from a function execution as a Log."""
        with StringIO() as out_buffer, StringIO() as err_buffer, redirect_stdout(out_buffer), redirect_stderr(err_buffer):
            duration = 0
            try:
                duration = perf_counter()
                result = function()
                duration = perf_counter() - duration
            except Exception as exception:
                # Print the exception stack trace to our log
                traceback.print_exc(file=err_buffer)
                result = exception

            # Allow up to 1k log characters per step which is ~1MB per 600 step episode
            max_log_length = 1024
            out = out_buffer.getvalue()
            err = err_buffer.getvalue()
            log = Log({
                "stdout": out[0:max_log_length],
                "stderr": err[0:max_log_length],
                "duration": duration,
            })

            if print_std:
                if out:
                    print(trim_end(out, '\n'))
                if err:
                    print(trim_end(err, '\n'))

            return result, log


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

    @property
    def is_terminal(self):
        return self != AgentStatus.INACTIVE and self != AgentStatus.ACTIVE

    @property
    def is_error(self):
        return self in {
            AgentStatus.UNKNOWN,
            AgentStatus.ERROR,
            AgentStatus.INVALID,
            AgentStatus.TIMEOUT,
        }


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
TField = TypeVar('TField')
TNumericField = TypeVar('TNumericField', int, float)


class Field(Generic[TField], Dict[str, Any]):
    @property
    def type(self) -> str:
        return self["type"]

    @property
    def default(self) -> TItem:
        return self["default"]

    @property
    def description(self) -> TItem:
        return self["description"]


class NumericField(Field[TNumericField]):
    @property
    def minimum(self) -> TItem:
        return self["minimum"]

    @property
    def maximum(self) -> TItem:
        return self["maximum"]


class ArrayField(Field[List[TField]]):
    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        kwargs["items"] = create_field(kwargs["items"])
        super().__init__(*args, **kwargs)

    @property
    def items(self) -> Field[TField]:
        return self["items"]


class ObjectField(Field[Dict[str, Any]]):
    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        properties = kwargs["properties"]
        default = kwargs["default"]
        for name, property in properties.items():
            property = properties[name] = create_field(property)
            if name not in default:
                default[name] = property.default
        super().__init__(*args, **kwargs)

    @property
    def properties(self) -> Dict[str, Field[Any]]:
        return self["properties"]


class ConfigurationField(Generic[TConfiguration], ObjectField):
    @property
    def episode_steps(self) -> NumericField[int]:
        # These casts should be safe unless an environment has inappropriately overridden specification.**.type
        # Environments cannot overwrite these types because the competition system depends on their existence
        return cast(NumericField[int], self.properties["episodeSteps"])

    @property
    def act_timeout(self) -> NumericField[float]:
        return cast(NumericField[float], self.properties["actTimeout"])

    @property
    def run_timeout(self) -> NumericField[float]:
        return cast(NumericField[float], self.properties["runTimeout"])

    @property
    def agent_count(self) -> NumericField[int]:
        return cast(NumericField[int], self.properties["agentCount"])


class ObservationField(ObjectField):
    @property
    def remaining_overage_time(self) -> NumericField[float]:
        return cast(NumericField[float], self.properties["remainingOverageTime"])

    @property
    def step(self) -> NumericField[int]:
        return cast(NumericField[int], self.properties["step"])


TActionField = TypeVar('TActionField', bound=Field[TAction])


class Specification(ObjectField):
    @property
    def name(self) -> Field[str]:
        return self.properties["name"]

    @property
    def title(self) -> Field[str]:
        return self.properties["title"]

    @property
    def version(self) -> Field[str]:
        return self.properties["version"]

    @property
    def description(self) -> Field[str]:
        return self.properties["description"]

    @property
    def configuration(self) -> ConfigurationField:
        return cast(ConfigurationField, self.properties["configuration"])

    @property
    def agents(self) -> ArrayField[int]:
        return cast(ArrayField[int], self.properties["agents"])

    @property
    def reward(self) -> NumericField[int]:
        return cast(NumericField[int], self.properties["reward"])

    @property
    def info(self) -> ObjectField:
        return cast(ObjectField, self.properties["info"])

    @property
    def observation(self) -> ObservationField:
        return cast(ObservationField, self.properties["observation"])

    @property
    def action(self) -> TActionField:
        return cast(TActionField, self.properties["action"]

    @property
    def status(self) -> Field[str]:
        return self.properties["status"]


field_type_registry = {
    "object": ObjectField,
    "array": ArrayField,
    "number": NumericField,
    "integer": NumericField,
    "Configuration": ConfigurationField,
    "Observation": ObservationField,
    "Specification": Specification
}


def create_field(json: Dict[str, Any]) -> Field[TField]:
    type = json["type"]
    if type in field_type_registry:
        return field_type_registry[type](**json)
    return Field(**json)


class Environment(Generic[TState, TConfiguration]):
    """This class represents the base interface for an environment compatible with kaggle-environments."""
    @property
    def specification(self) -> Specification:
        raise NotImplemented()

    def reset(self, configuration: TConfiguration) -> List[TState]:
        raise NotImplemented()

    def step(self, state: List[TState], configuration: TConfiguration) -> List[TState]:
        raise NotImplemented()

    def builtin_agents(self) -> Dict[str, Agent]:
        raise NotImplemented()

    def builtin_configurations(self) -> Dict[str, TConfiguration]:
        raise NotImplemented()

    def render_text(self, configuration: TConfiguration, state: List[List[TState]]) -> str:
        raise NotImplemented()

    def render_html(self, configuration: TConfiguration) -> str:
        raise NotImplemented()


class BaseEnvironment(Environment[TState, TConfiguration]):
    """This class provides helpful implementations for part of the Environment interface."""
    def __init__(self, specification_path: str):
        self._specification = BaseEnvironment.load_specification(specification_path)

    @property
    def specification(self) -> Specification:
        return self._specification

    @staticmethod
    def load_specification(specification_file_path: str) -> Specification:
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
        return Specification(process_schema(schemas.specification, specification))