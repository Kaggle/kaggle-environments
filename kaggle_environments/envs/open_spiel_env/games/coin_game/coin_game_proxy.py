"""Structured JSON observations for the OpenSpiel Coin Game.

The Coin Game (Lerer & Peysakhovich, 2018) places several players on an
8x8 grid alongside coins of multiple colours. Each player is privately
assigned a preferred coin colour. On each turn a player picks one of
{up, down, left, right, stand}; moving onto a coin collects it. At the
end of a fixed number of moves, each player's reward is

    self_pref^2 + other_pref^2 - bad_coins^2

where ``self_pref`` counts the player's own preference picked up across
the game, ``other_pref`` counts other players' preferences picked up by
*anyone*, and ``bad_coins`` are coins of unowned colours collected by
anyone. Imperfect information: each player only learns its own
preference until the game ends.

The OpenSpiel framework auto-resolves the chance-driven setup phases
(AssignPreferences, DeployPlayers, DeployCoins) via random outcomes, so
agents only see the play phase. Each ``observation_string(player)`` is
therefore a per-player JSON view that hides other players' preferences.
"""

import json
from typing import Any

import pyspiel

from ... import proxy

_COIN_RANGE = "abcdefghijklmnopqrstuvwxyz"


class CoinGameState(proxy.State):
    """Coin Game state proxy.

    The board is exposed as ``board[r][c]`` with ``r=0`` at the top-left
    of the OpenSpiel grid. Cells are one-character strings: ``"."`` for
    empty squares, ``"0"``..``"9"`` for players, and ``"a"``..``"z"`` for
    coins (one letter per colour).
    """

    def _params(self) -> dict[str, int]:
        # ``get_parameters`` returns wrapped values — extract ints.
        params = self.get_game().get_parameters()
        return {
            "rows": int(params.get("rows", 8)),
            "columns": int(params.get("columns", 8)),
            "episode_length": int(params.get("episode_length", 20)),
            "num_extra_coin_colors": int(params.get("num_extra_coin_colors", 1)),
            "num_coins_per_color": int(params.get("num_coins_per_color", 4)),
            "players": int(params.get("players", self.__wrapped__.num_players())),
        }

    def _coin_colors(self) -> list[str]:
        p = self._params()
        num_colors = p["players"] + p["num_extra_coin_colors"]
        return list(_COIN_RANGE[:num_colors])

    def _parse_full_state(self) -> dict[str, Any]:
        """Parse OpenSpiel's ToString for the full (cross-player) state.

        ToString output looks like::

            phase=Play
            preferences=0:a 1:b
            moves=3
                    a b c
            player0 1 0 0
            player1 0 0 0
            +--------+
            |01aaaabb|
            ...
            +--------+
        """
        raw = str(self.__wrapped__)
        lines = raw.split("\n")
        params = self._params()
        rows = params["rows"]
        cols = params["columns"]

        phase = ""
        preferences: dict[int, str] = {}
        moves = 0
        coins_collected: dict[int, dict[str, int]] = {}
        coin_colors = self._coin_colors()

        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("phase="):
                phase = line.split("=", 1)[1].strip()
            elif line.startswith("preferences="):
                pref_part = line.split("=", 1)[1].strip()
                for tok in pref_part.split():
                    pid_str, color = tok.split(":")
                    preferences[int(pid_str)] = color
            elif line.startswith("moves="):
                moves = int(line.split("=", 1)[1].strip())
            elif line.startswith("player") and ":" not in line:
                # "player0 1 0 0" — number of each coin colour collected.
                parts = line.split()
                pid = int(parts[0][len("player"):])
                counts = [int(x) for x in parts[1:]]
                coins_collected[pid] = {
                    color: counts[idx] if idx < len(counts) else 0
                    for idx, color in enumerate(coin_colors)
                }
            i += 1

        # Find the board between the two "+----+" delimiter lines.
        delim_indices = [idx for idx, ln in enumerate(lines) if ln.startswith("+") and ln.endswith("+")]
        board: list[list[str]] = []
        if len(delim_indices) >= 2:
            top, bottom = delim_indices[0], delim_indices[1]
            for ln in lines[top + 1:bottom]:
                # Strip the leading/trailing '|' frame; pad to expected width.
                inner = ln[1:1 + cols]
                row = [(c if c != " " else ".") for c in inner.ljust(cols)]
                board.append(row)
        if len(board) != rows:
            # Defensive: build an empty board if parsing failed.
            board = [["."] * cols for _ in range(rows)]

        # Locate each player on the board.
        player_positions: dict[int, list[int] | None] = {}
        for pid in range(params["players"]):
            symbol = str(pid)
            pos = None
            for r in range(rows):
                for c in range(cols):
                    if board[r][c] == symbol:
                        pos = [r, c]
                        break
                if pos is not None:
                    break
            player_positions[pid] = pos

        return {
            "phase": phase,
            "preferences": preferences,
            "moves": moves,
            "coins_collected": coins_collected,
            "board": board,
            "player_positions": player_positions,
        }

    def _last_action_str(self) -> str | None:
        """String form of the most recent (non-chance) action, if any."""
        history = self.__wrapped__.history()
        if not history:
            return None
        # Walk back through history; the last entry is the most recent action.
        # During play phase actions are 0..4 mapped to up/down/left/right/stand.
        last = history[-1]
        # Heuristic: only the play actions are 0..4. Earlier setup actions are
        # also small ints, so use phase to gate this.
        full = self._parse_full_state()
        if full["phase"] != "Play" and full["moves"] == 0:
            return None
        try:
            # action_to_string wants a non-chance player; use 0 as a stand-in.
            return self.__wrapped__.action_to_string(0, last)
        except Exception:  # noqa: BLE001
            return None

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        full = self._parse_full_state()
        params = self._params()

        winner: int | str | None = None
        if self.is_terminal():
            returns = list(self.returns())
            best = max(returns)
            top = [i for i, r in enumerate(returns) if r == best]
            winner = top[0] if len(top) == 1 else "draw"

        result: dict[str, Any] = {
            "phase": full["phase"].lower() if full["phase"] else "play",
            "board": full["board"],
            "num_rows": params["rows"],
            "num_columns": params["columns"],
            "coin_colors": self._coin_colors(),
            "player_positions": full["player_positions"],
            "coins_collected": full["coins_collected"],
            "current_player": self.current_player(),
            "move_number": full["moves"],
            "moves_remaining": max(0, params["episode_length"] - full["moves"]),
            "episode_length": params["episode_length"],
            "is_terminal": self.is_terminal(),
            "winner": winner,
            "last_action": self._last_action_str(),
        }

        # Imperfect info: only expose the requesting player's preference.
        if player is None:
            result["preferences"] = full["preferences"]
        elif player in full["preferences"]:
            result["your_preference"] = full["preferences"][player]
            result["your_player_id"] = player

        if self.is_terminal():
            result["returns"] = list(self.returns())
            result["preferences"] = full["preferences"]

        return result

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class CoinGameGame(proxy.Game):
    """Coin Game proxy."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("coin_game", params)
        super().__init__(
            wrapped,
            short_name="coin_game_proxy",
            long_name="Coin Game (proxy)",
        )

    def new_initial_state(self, *args) -> CoinGameState:
        return CoinGameState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(CoinGameGame().get_type(), CoinGameGame)
