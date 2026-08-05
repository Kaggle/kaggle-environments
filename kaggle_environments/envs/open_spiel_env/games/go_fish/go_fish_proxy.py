"""Structured JSON observations for Go Fish.

OpenSpiel's default observation_string for go_fish is multi-line text with a
phase header, the observing player's hand (rank letter + count tokens like
"a3 b1"), a per-player card/book summary, and a tail of recent opponent events
written in a terse, sometimes confusing form (e.g. "0 asked -3 for 9 recieved
1", where target -3 is OpenSpiel's kInvalidPlayer sentinel meaning the player
drew from the pool). This proxy parses that text into a clean JSON object.

Ranks are indexed 0..ranks-1. For the standard 13-rank deck they are mapped to
human-readable labels A,2-10,J,Q,K; otherwise the raw OpenSpiel letter
(a,b,c,...) is used. Each player only sees their own hand and the opponent
events that occurred since their previous turn -- this proxy preserves that
information structure exactly.

Alongside the parsed text, the proxy surfaces public state the text observation
omits, read straight off pyspiel's state accessors: ``pool_size``, ``booked``
(:meth:`GoFishState._public_counts`) and the game-long ``deductions`` table
(:meth:`GoFishState._deductions`). All three are common information available
to every player, so surfacing them reveals nothing private -- they just spare
the consumer from re-deriving public facts by arithmetic over full history.
"""

import json
import re
from collections.abc import Sequence
from typing import Any

import pyspiel

from ... import proxy

_STANDARD_RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

_PHASE_RE = re.compile(r"^Phase (\S+)$")
_YOUR_CARDS_RE = re.compile(r"^Your cards:\s*(.*)$")
_CARD_TOKEN_RE = re.compile(r"([a-z])(\d+)")
_PLAYER_RE = re.compile(r"^player (\d+) cards (\d+) books (\d+)$")
_EVENT_RE = re.compile(r"^(\d+) asked (-?\d+) for (\d+)(?: recieved (\d+))?(?: booked (\d+))?$")


def _rank_label(rank_index: int, num_ranks: int) -> str:
    if num_ranks == 13 and 0 <= rank_index < 13:
        return _STANDARD_RANKS[rank_index]
    return chr(ord("a") + rank_index)


def _parse_observation(text: str, num_ranks: int) -> dict[str, Any]:
    """Parse the OpenSpiel go_fish observation_string into a dict."""
    lines = text.split("\n")
    result: dict[str, Any] = {
        "phase": None,
        "hand": {},
        "players": [],
        "recent_events": [],
    }
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        m = _PHASE_RE.match(line)
        if m:
            result["phase"] = m.group(1)
            continue
        if line.startswith("Current Player"):
            continue  # redundant with state.current_player()
        m = _YOUR_CARDS_RE.match(line)
        if m:
            hand: dict[str, int] = {}
            for letter, count in _CARD_TOKEN_RE.findall(m.group(1)):
                rank_index = ord(letter) - ord("a")
                hand[_rank_label(rank_index, num_ranks)] = int(count)
            result["hand"] = hand
            continue
        m = _PLAYER_RE.match(line)
        if m:
            result["players"].append(
                {
                    "player": int(m.group(1)),
                    "cards": int(m.group(2)),
                    "books": int(m.group(3)),
                }
            )
            continue
        m = _EVENT_RE.match(line)
        if m:
            player = int(m.group(1))
            target = int(m.group(2))
            rank = int(m.group(3))
            received = int(m.group(4)) if m.group(4) is not None else 0
            booked = m.group(5) is not None
            if target < 0:
                # OpenSpiel sentinel target: the player drew a card from the pool.
                event: dict[str, Any] = {
                    "type": "draw",
                    "player": player,
                    "rank": rank,
                    "rank_label": _rank_label(rank, num_ranks),
                    "booked": booked,
                }
            else:
                event = {
                    "type": "ask",
                    "player": player,
                    "target": target,
                    "rank": rank,
                    "rank_label": _rank_label(rank, num_ranks),
                    "received": received,
                    "booked": booked,
                }
            result["recent_events"].append(event)
    return result


class GoFishState(proxy.State):
    """Go Fish state proxy with structured JSON observations."""

    def _num_ranks(self) -> int:
        return int(self.__wrapped__.get_game().get_parameters().get("ranks", 13))

    def _num_suits(self) -> int:
        return int(self.__wrapped__.get_game().get_parameters().get("suits", 4))

    def _public_counts(self, num_ranks: int) -> tuple[int, list[str]]:
        """Return ``(pool_size, booked_rank_labels)``.

        Both are fully public state that the text observation_string omits
        entirely, so without this the model can only recover them by doing
        deck arithmetic over the whole game history. Pool size is strategically
        central: it decides whether a miss draws a card at all (an ask into an
        empty pool just ends your turn) and how near the game is to ending.
        The booked list says which ranks are already scored and therefore dead.

        Both are common information -- pool size is a plain count with no rank
        attached, and a laid-down book is public by construction -- so surfacing
        them leaks nothing, exactly as with :meth:`_deductions`.
        """
        state = self.__wrapped__
        booked = [_rank_label(rank, num_ranks) for rank, is_booked in enumerate(state.booked()) if is_booked]
        return state.pool_size(), booked

    def _deductions(self, num_ranks: int, booked: Sequence[str] = ()) -> list[dict[str, Any]]:
        """Build the durable public deduction table.

        In Go Fish, asks are public and accumulate into common knowledge that
        every player is entitled to track: who is known to hold a rank (they
        asked for it, so hold >=1, or received cards of it) and who is known to
        have *none* of a rank (they were emptied by an ask and have not drawn
        since). OpenSpiel maintains exactly this as common info, and unlike the
        ``recent_events`` window it never truncates. We surface it here so the
        harness gets the full game-long signal, not just the last turn.

        The four fields come straight off the pybind-exposed state accessors
        (``player_min``, ``player_was_asked``, ``drawn_since_was_asked``,
        ``player_did_ask``), each a ``[player][rank]`` grid of plain ints/bools.
        These are the same members ``ObservationTensor`` encodes, read at the
        source: no tensor offsets to keep in sync and no float round-tripping.

        All four are public by construction. In particular ``drawn_since``
        counts cards drawn after being asked but NEVER their ranks, so it leaks
        no hidden information -- it only weakens a known-void into "possibly
        holds again".

        Two of these fields are cumulative and never expire on their own, so we
        retire stale entries here rather than emit facts that are no longer true:

        * **Booked ranks are dead.** Once a rank is booked nobody holds it and
          nobody can be asked for it, but ``player_did_ask`` and
          ``player_was_asked`` still carry its history. Reporting "known to
          have none of 9" after the 9s are booked is vacuous (nobody has any),
          and "has asked for 9" is unactionable. A booked rank is dropped from
          all three lists.
        * **An ask expires once we learn the asker was emptied.** ``wanted``
          derives from ``player_did_ask``, a counter that only ever increments,
          so a rank stayed listed as wanted even after the player was asked for
          it and turned out to hold none. That produced rows that contradicted
          themselves -- "known to have none of 9; has asked for 9". Where a rank
          is currently ``known_void`` for a player, its stale ask is dropped.

        Both filters only ever remove entries the public record has already
        invalidated; nothing private is consulted.
        """
        state = self.__wrapped__
        player_min = state.player_min()
        was_asked_grid = state.player_was_asked()
        drawn_since_grid = state.drawn_since_was_asked()
        did_ask_grid = state.player_did_ask()
        booked_set = set(booked)

        deductions: list[dict[str, Any]] = []
        for pid in range(state.get_game().num_players()):
            known_has: list[str] = []  # rank labels this player is known to hold
            known_void: list[str] = []  # rank labels this player is known to lack
            wanted: list[str] = []  # ranks this player has asked for and may still want
            for rank in range(num_ranks):
                label = _rank_label(rank, num_ranks)
                if label in booked_set:
                    # Rank is scored and out of play: no one holds it, no one
                    # can be asked for it, so every claim about it is stale.
                    continue
                minimum = player_min[pid][rank]
                is_void = was_asked_grid[pid][rank] and drawn_since_grid[pid][rank] == 0 and minimum == 0
                if minimum > 0:
                    known_has.append(f"{label}>={minimum}")
                if is_void:
                    known_void.append(label)
                # A past ask only still tells us they want the rank while they
                # might hold one. Once they are known void for it, the ask is
                # spent -- keeping it would contradict known_void on the same row.
                if did_ask_grid[pid][rank] > 0 and not is_void:
                    wanted.append(label)
            deductions.append(
                {
                    "player": pid,
                    "known_has": known_has,
                    "known_void": known_void,
                    "wanted": wanted,
                }
            )
        return deductions

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        observer = player if player is not None else 0
        num_ranks = self._num_ranks()
        num_suits = self._num_suits()
        parsed = _parse_observation(self.__wrapped__.observation_string(observer), num_ranks)
        pool_size, booked = self._public_counts(num_ranks)

        winner: int | str | None = None
        returns_list: list[float] = []
        if self.is_terminal():
            returns_list = list(self.returns())
            best = max(returns_list)
            leaders = [i for i, r in enumerate(returns_list) if r == best]
            if len(leaders) == 1:
                winner = leaders[0]
            else:
                winner = "draw"

        return {
            "phase": parsed["phase"],
            "current_player": self.current_player(),
            "observer": observer,
            "is_terminal": self.is_terminal(),
            "winner": winner,
            "returns": returns_list,
            "num_ranks": num_ranks,
            "num_suits": num_suits,
            "pool_size": pool_size,
            "booked": booked,
            "hand": parsed["hand"],
            "players": parsed["players"],
            "recent_events": parsed["recent_events"],
            "deductions": self._deductions(num_ranks, booked),
        }

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class GoFishGame(proxy.Game):
    """Go Fish game proxy."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("go_fish", params)
        super().__init__(
            wrapped,
            short_name="go_fish_proxy",
            long_name="Go Fish (proxy)",
        )

    def new_initial_state(self, *args) -> GoFishState:
        return GoFishState(self.__wrapped__.new_initial_state(*args), game=self)


pyspiel.register_game(GoFishGame().get_type(), GoFishGame)
