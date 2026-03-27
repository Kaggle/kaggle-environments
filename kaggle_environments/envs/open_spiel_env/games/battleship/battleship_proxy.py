"""Structured JSON observations for Battleship (imperfect information)."""

import json
from typing import Any

import pyspiel

from ... import proxy


def _parse_board(section: str) -> list[list[str]]:
    """Parse an ASCII grid section into a 2D list of characters.

    Grid format:
        +-----+
        |abcde|
        ...
        +-----+
    """
    rows: list[list[str]] = []
    for line in section.split("\n"):
        if line.startswith("|") and line.endswith("|"):
            rows.append(list(line[1:-1]))
    return rows


def _detect_phase(shots: list[list[str]]) -> str:
    """Return 'placement' if the shots board is all spaces, else 'war'."""
    for row in shots:
        for cell in row:
            if cell != " ":
                return "war"
    return "placement"


class BattleshipState(proxy.State):
    """Wraps OpenSpiel Battleship state with JSON observations."""

    def _parse_player_observation(self, player: int) -> dict[str, Any] | None:
        """Parse a player's observation into ships and shots boards."""
        raw = self.__wrapped__.observation_string(player)
        sections = raw.split("\nPlayer's shot outcomes:\n")
        if len(sections) != 2:
            return None
        ships = _parse_board(sections[0])
        shots = _parse_board(sections[1])
        if not ships or not shots:
            return None
        return {
            "ships": ships,
            "shots": shots,
            "width": len(ships[0]) if ships else 0,
            "height": len(ships),
        }

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        if player is None:
            player = max(0, self.current_player())
        obs = self._parse_player_observation(player)
        phase = _detect_phase(obs["shots"]) if obs else "placement"
        winner = None
        if self.is_terminal():
            returns = self.returns()
            if returns[0] > returns[1]:
                winner = 0
            elif returns[1] > returns[0]:
                winner = 1
            else:
                winner = "draw"
        return {
            "ships": obs["ships"] if obs else [],
            "shots": obs["shots"] if obs else [],
            "width": obs["width"] if obs else 0,
            "height": obs["height"] if obs else 0,
            "phase": phase,
            "current_player": self.current_player(),
            "is_terminal": self.is_terminal(),
            "winner": winner,
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self):
        return self.to_json()


class BattleshipGame(proxy.Game):
    """Wraps the OpenSpiel Battleship game to use the proxy state."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("battleship", params)
        super().__init__(
            wrapped,
            short_name="battleship_proxy",
            long_name="Battleship (proxy)",
        )

    def new_initial_state(self, *args) -> BattleshipState:
        return BattleshipState(
            self.__wrapped__.new_initial_state(*args), game=self
        )


pyspiel.register_game(BattleshipGame().get_type(), BattleshipGame)
