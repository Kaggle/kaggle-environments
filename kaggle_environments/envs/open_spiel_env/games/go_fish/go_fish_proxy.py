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
"""

import json
import re
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

    def _deductions(self, observer: int, num_ranks: int, num_suits: int) -> list[dict[str, Any]]:
        """Decode the durable public deduction table from the observation tensor.

        In Go Fish, asks are public and accumulate into common knowledge that
        every player is entitled to track: who is known to hold a rank (they
        asked for it, so hold >=1, or received cards of it) and who is known to
        have *none* of a rank (they were emptied by an ask and have not drawn
        since). OpenSpiel maintains exactly this as common info in the
        observation tensor (go_fish.cc:497-510) -- it never truncates, unlike
        the ``recent_events`` window. We reconstruct it here so the harness can
        surface the full game-long signal, not just events since the last turn.

        The tensor layout (see ``ObservationTensor``) is::

            [ranks]            observer's own card fractions
            [4]                phase one-hots
            [1]                pool size fraction
            [ranks]            booked flags
            per player:
              [1]              is-current-player flag
              [1]              books fraction (/ranks)
              [1]              total-cards fraction (/(ranks*suits))
              per rank: [4]    did_ask, was_asked, drawn_since, min

        ``drawn_since`` counts cards drawn after being asked but NEVER their
        ranks, so decoding it leaks no hidden information -- it only weakens a
        known-void into "possibly holds again".
        """
        # The tensor is computed from a valid player's perspective; the common
        # (public) fields we read are identical for every observer. current_player
        # is kInvalidPlayer at chance/terminal nodes, so key off the observer.
        tensor = list(self.__wrapped__.observation_tensor(observer))
        num_players = self.__wrapped__.get_game().num_players()
        per_player = 3 + num_ranks * 4
        base0 = num_ranks + 4 + 1 + num_ranks  # own cards + phase + pool + booked

        deductions: list[dict[str, Any]] = []
        for pid in range(num_players):
            base = base0 + pid * per_player
            known_has: list[str] = []  # rank labels this player is known to hold
            known_void: list[str] = []  # rank labels this player is known to lack
            wanted: list[str] = []  # ranks this player has asked for
            for rank in range(num_ranks):
                b = base + 3 + rank * 4
                did_ask = round(tensor[b] * (num_suits * num_ranks))
                was_asked = tensor[b + 1] > 0.5
                drawn_since = round(tensor[b + 2] * (num_ranks * num_suits))
                minimum = round(tensor[b + 3] * num_suits)
                label = _rank_label(rank, num_ranks)
                if minimum > 0:
                    known_has.append(f"{label}>={minimum}")
                if was_asked and drawn_since == 0 and minimum == 0:
                    known_void.append(label)
                if did_ask > 0:
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
            "hand": parsed["hand"],
            "players": parsed["players"],
            "recent_events": parsed["recent_events"],
            "deductions": self._deductions(observer, num_ranks, num_suits),
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
