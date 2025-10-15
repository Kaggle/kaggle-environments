"""OpenSpiel Game and State proxies.

Proxies that act as a pyspiel.State/Game by wrapping the original object and
forwarding calls. Subclassing allows to override specific methods or add
additional functionality, or payload to the State/Game object.

WARNING: Serialization of proxy games and states is not supported.
"""

from typing import Any

import pyspiel

from . import observation


class State(pyspiel.State):
    """Base class for a pyspiel.State proxy."""

    def __init__(self, wrapped: pyspiel.State, game: "Game"):
        super().__init__(game)
        self.__wrapped__ = wrapped

    def current_player(self) -> int:
        return self.__wrapped__.current_player()

    def _legal_actions(self, player: int) -> list[int]:
        return self.__wrapped__.legal_actions(player)

    def _apply_action(self, action: int) -> None:
        return self.__wrapped__.apply_action(action)

    def _action_to_string(self, player: int, action: int) -> str:
        return self.__wrapped__.action_to_string(player, action)

    def chance_outcomes(self) -> list[tuple[int, float]]:
        return self.__wrapped__.chance_outcomes()

    def is_terminal(self) -> bool:
        return self.__wrapped__.is_terminal()

    def returns(self) -> list[float]:
        return self.__wrapped__.returns()

    def rewards(self) -> list[float]:
        return self.__wrapped__.rewards()

    def __str__(self) -> str:
        return self.__wrapped__.__str__()

    def to_string(self) -> str:
        return self.__wrapped__.to_string()

    def __getattr__(self, name: str) -> Any:
        # Escape hatch when proxying Python implementations that have attributes
        # that need to be accessed, e.g. TicTacToeState.board from its observer.
        return object.__getattribute__(self.__wrapped__, name)


class Game(pyspiel.Game):
    """Base class for a pyspiel.Game proxy."""

    def __init__(self, wrapped: pyspiel.Game, **kwargs):
        # TODO(hennes): Add serialization.
        game_info = pyspiel.GameInfo(
            num_distinct_actions=wrapped.num_distinct_actions(),
            max_chance_outcomes=wrapped.max_chance_outcomes(),
            num_players=wrapped.num_players(),
            min_utility=wrapped.min_utility(),
            max_utility=wrapped.max_utility(),
            utility_sum=wrapped.utility_sum(),
            max_game_length=wrapped.max_game_length(),
        )
        super().__init__(
            _game_type(wrapped.get_type(), **kwargs),
            game_info,
            wrapped.get_parameters(),
        )
        self.__wrapped__ = wrapped

    def new_initial_state(self, from_string: str | None = None) -> State:
        args = () if from_string is None else (from_string)
        return State(wrapped=self.__wrapped__.new_initial_state(*args), game=self)

    def max_chance_nodes_in_history(self) -> int:
        return self.__wrapped__.max_chance_nodes_in_history()

    def make_py_observer(
        self,
        iig_obs_type: pyspiel.IIGObservationType | None = None,
        params: dict[str, Any] | None = None,
    ) -> pyspiel.Observer:
        return _Observation(observation.make_observation(self.__wrapped__, iig_obs_type, params))


class _Observation(observation._Observation):  # pylint: disable=protected-access
    """_Observation proxy that passes the wrapped state to the observation."""

    def __init__(self, wrapped: observation._Observation):
        self.__wrapped__ = wrapped
        self.dict = self.__wrapped__.dict
        self.tensor = self.__wrapped__.tensor

    def set_from(self, state: State, player: int):
        self.__wrapped__.set_from(state.__wrapped__, player)

    def string_from(self, state: State, player: int) -> str | None:
        return self.__wrapped__.string_from(state.__wrapped__, player)

    def compress(self) -> Any:
        return self.__wrapped__.compress()

    def decompress(self, compressed_observation: Any):
        self.__wrapped__.decompress(compressed_observation)


def _game_type(game_type: pyspiel.GameType, **overrides) -> pyspiel.GameType:
    """Returns a GameType with the given overrides."""
    kwargs = dict(
        short_name=game_type.short_name,
        long_name=game_type.long_name,
        dynamics=game_type.dynamics,
        chance_mode=game_type.chance_mode,
        information=game_type.information,
        utility=game_type.utility,
        reward_model=game_type.reward_model,
        max_num_players=game_type.max_num_players,
        min_num_players=game_type.min_num_players,
        provides_information_state_string=game_type.provides_information_state_string,
        provides_information_state_tensor=game_type.provides_information_state_tensor,
        provides_observation_string=game_type.provides_observation_string,
        provides_observation_tensor=game_type.provides_observation_tensor,
        parameter_specification=game_type.parameter_specification,
        default_loadable=game_type.default_loadable,
        provides_factored_observation_string=game_type.provides_factored_observation_string,
    )
    kwargs.update(**overrides)
    return pyspiel.GameType(**kwargs)
