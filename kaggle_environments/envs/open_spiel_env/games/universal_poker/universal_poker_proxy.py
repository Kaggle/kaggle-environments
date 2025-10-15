"""Change Universal Poker state and action string representations."""

import json
import re
from typing import Any

import pyspiel

from ... import proxy


class UniversalPokerState(proxy.State):
    """Universal Poker state proxy."""

    def _player_string(self, player: int) -> str:
        if player == pyspiel.PlayerId.CHANCE:
            return "chance"
        elif player == pyspiel.PlayerId.TERMINAL:
            return "terminal"
        if player < 0:
            return pyspiel.PlayerId(player).name.lower()
        else:
            return str(player)

    def _state_dict(self) -> dict[str, Any]:
        params = self.get_game().get_parameters()
        blinds = params["blind"].strip().split()
        blinds = [int(blind) for blind in blinds]
        assert len(blinds) == self.num_players()
        starting_stacks = params["stack"].strip().split()
        starting_stacks = [int(stack) for stack in starting_stacks]
        assert len(starting_stacks) == self.num_players()
        state_str = self.to_string()
        state_lines = state_str.split("\n")
        player_hands = []
        for i in range(self.num_players()):
            for line in state_lines:
                if line.startswith(f"P{i} Cards:"):
                    hand = line.split(":")[1].strip()
                    player_hands.append([hand[i : i + 2] for i in range(0, len(hand), 2)])
        assert len(player_hands) == self.num_players()
        board_cards = None
        for line in state_lines:
            if line.startswith("BoardCards"):
                board_cards_str = line.removeprefix("BoardCards").strip()
                board_cards = [board_cards_str[i : i + 2] for i in range(0, len(board_cards_str), 2)]
        assert board_cards is not None
        pattern = r"P\d+:\s*(\d+)"
        player_contributions = []
        for line in state_lines:
            if line.startswith("Spent:"):
                matches = re.findall(pattern, line)
                player_contributions = [int(match) for match in matches]
        assert len(player_contributions) == self.num_players()
        acpc_state = None
        betting_history = None
        for line in state_lines:
            if line.startswith("ACPC State:"):
                acpc_state = line.split("ACPC State:")[1].strip()
                betting_history = acpc_state.split(":")[2]
        assert acpc_state is not None
        assert betting_history is not None

        state_dict = {}
        state_dict["acpc_state"] = acpc_state
        state_dict["current_player"] = self._player_string(self.current_player())
        state_dict["blinds"] = blinds
        state_dict["betting_history"] = betting_history
        state_dict["player_contributions"] = player_contributions
        state_dict["pot_size"] = sum(player_contributions)
        state_dict["starting_stacks"] = starting_stacks
        state_dict["player_hands"] = player_hands
        state_dict["board_cards"] = board_cards
        return state_dict

    def to_json(self) -> str:
        return json.dumps(self._state_dict())

    def _action_to_string(self, player: int, action: int) -> str:
        if player == pyspiel.PlayerId.CHANCE:
            return f"deal {action}"  # TODO(jhtschultz): Add card.
        if action == 0:
            return "fold"
        elif action == 1:
            if 0 in self.legal_actions():
                return "call"
            else:
                return "check"
        else:
            return f"raise{action}"

    def action_to_json(self, action: int) -> str:
        action_str = self._action_to_string(self.current_player(), action)
        return json.dumps({"action": action_str})

    def observation_dict(self, player: int) -> dict[str, Any]:
        state_dict = self._state_dict()
        for i in range(self.num_players()):
            if i == player:
                continue
            state_dict["player_hands"][i] = ["??", "??"]
        del state_dict["acpc_state"]
        return state_dict

    def observation_json(self, player: int) -> str:
        return json.dumps(self.observation_dict(player))

    def observation_string(self, player: int) -> str:
        return self.observation_json(player)

    def __str__(self):
        return self.to_json()


def _strip_empty_kwargs(input_string):
    try:
        open_paren_index = input_string.index("(")
        close_paren_index = input_string.rindex(")")
    except ValueError:
        return input_string

    function_name = input_string[:open_paren_index]
    args_string = input_string[open_paren_index + 1 : close_paren_index]
    args_list = args_string.split(",")
    non_empty_args = []
    for arg in args_list:
        parts = arg.split("=", 1)
        if len(parts) > 1 and parts[1]:
            non_empty_args.append(arg)
        elif len(parts) == 1 and parts[0]:
            non_empty_args.append(arg)
    new_args_string = ",".join(non_empty_args)
    return f"{function_name}({new_args_string})"


class UniversalPokerGame(proxy.Game):
    """Universal Poker game proxy."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("universal_poker", params)
        super().__init__(
            wrapped,
            short_name="universal_poker_proxy",
            long_name="Universal Poker (proxy)",
        )

    def __str__(self):
        s = _strip_empty_kwargs(self.__wrapped__.__str__())
        return s.split("(")[0] + "_proxy(" + s.split("(")[1]

    def new_initial_state(self, *args) -> UniversalPokerState:
        return UniversalPokerState(
            self.__wrapped__.new_initial_state(*args),
            game=self,
        )


pyspiel.register_game(UniversalPokerGame().get_type(), UniversalPokerGame)
