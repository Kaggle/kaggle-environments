"""Bridge Arena: 2v2 team variant of OpenSpiel's Bridge.

Wraps OpenSpiel's Bridge so that the external player IDs are laid out
as AABB (players 0/1 = team A, players 2/3 = team B) while bridge's
internal seating remains NESW. Partners sit opposite at the table
(N-S and E-W) so the table layout reads ABAB. Each external player
maps to an internal bridge seat:

    external 0 (team A, seat 0) -> internal 0 (North)
    external 1 (team A, seat 1) -> internal 2 (South)
    external 2 (team B, seat 0) -> internal 1 (East)
    external 3 (team B, seat 1) -> internal 3 (West)

The mapping is its own inverse, so the same permutation translates in
both directions. A model fielding both seats of a team can only
"communicate" via in-game bidding -- there is no side channel.

All bridge mechanics (chance dealing, bidding, card play, scoring) come
from the wrapped OpenSpiel game; the wrapper only renames players,
permutes returns/rewards, and emits a structured JSON observation that
tells each agent which seat it occupies and who its partner is.
"""

from __future__ import annotations

import json
from typing import Any

import pyspiel

from ... import observation, proxy

# External AABB player id -> internal bridge seat (NESW = 0/1/2/3).
# Self-inverse: _PERM[_PERM[i]] == i, so it also maps internal -> external.
_PERM = (0, 2, 1, 3)
_TABLE_POSITIONS = ("N", "E", "S", "W")  # indexed by internal bridge seat
_NUM_PLAYERS = 4
_PLAYERS_PER_TEAM = 2

# Bridge-specific constants used by play attribution.
_CARDS_IN_DECK = 52
_BRIDGE_PASS = 52
_BRIDGE_DBL = 53
_BRIDGE_RDBL = 54
_BRIDGE_FIRST_BID = 55
_BRIDGE_DENOMS = ("♣", "♦", "♥", "♠", "NT")  # OpenSpiel denomination order
_RANK_ORDER = "23456789TJQKA"  # ascending


def _card_suit_rank(card_str: str) -> tuple[str, str]:
    """Parse OpenSpiel card label like '♣10' or '♥A' into (suit, rank).

    Normalizes rank "10" to "T" so the canonical form is one char and
    sortable via ``_RANK_ORDER``.
    """
    suit = card_str[0]
    rank = card_str[1:]
    if rank == "10":
        rank = "T"
    return suit, rank


def _trick_winner_idx(trick_cards: list[str], trump: str | None) -> int:
    """Return the 0..3 index of the winning card in a 4-card trick.

    Trump cards beat non-trump; among trump (or among led suit if no
    trump played) highest rank wins. ``trump`` is a suit glyph
    (♣/♦/♥/♠) or ``None`` for no-trump contracts.
    """
    suit0, rank0 = _card_suit_rank(trick_cards[0])
    led_suit = suit0
    best_idx = 0
    best_rank = _RANK_ORDER.index(rank0)
    best_is_trump = trump is not None and suit0 == trump
    for i in range(1, len(trick_cards)):
        suit, rank = _card_suit_rank(trick_cards[i])
        rank_val = _RANK_ORDER.index(rank)
        is_trump = trump is not None and suit == trump
        if is_trump and (not best_is_trump or rank_val > best_rank):
            best_idx, best_rank, best_is_trump = i, rank_val, True
        elif not is_trump and not best_is_trump and suit == led_suit and rank_val > best_rank:
            best_idx, best_rank = i, rank_val
    return best_idx


def _ext_to_int(ext: int) -> int:
    return _PERM[ext]


def _int_to_ext(internal: int) -> int:
    return _PERM[internal]


def _table_position(ext: int) -> str:
    return _TABLE_POSITIONS[_PERM[ext]]


def _team_of(ext: int) -> int:
    return ext // _PLAYERS_PER_TEAM


def _partner_of(ext: int) -> int:
    return _team_of(ext) * _PLAYERS_PER_TEAM + (1 - ext % _PLAYERS_PER_TEAM)


class BridgeArenaState(proxy.State):
    """Wraps a Bridge state, exposing AABB player IDs over NESW seats."""

    def current_player(self) -> int:
        cp = self.__wrapped__.current_player()
        # Pass through CHANCE (-1), TERMINAL (-4), SIMULTANEOUS (-2) etc.
        if cp < 0:
            return cp
        return _int_to_ext(cp)

    def _legal_actions(self, player: int) -> list[int]:
        if player < 0:
            return self.__wrapped__.legal_actions(player)
        return self.__wrapped__.legal_actions(_ext_to_int(player))

    def _action_to_string(self, player: int, action: int) -> str:
        if player < 0:
            return self.__wrapped__.action_to_string(player, action)
        return self.__wrapped__.action_to_string(_ext_to_int(player), action)

    def returns(self) -> list[float]:
        wrapped_returns = self.__wrapped__.returns()
        return [wrapped_returns[_ext_to_int(ext)] for ext in range(_NUM_PLAYERS)]

    def rewards(self) -> list[float]:
        wrapped_rewards = self.__wrapped__.rewards()
        return [wrapped_rewards[_ext_to_int(ext)] for ext in range(_NUM_PLAYERS)]

    # pyspiel.State exposes per-player accessors that bypass our
    # overridden returns()/rewards() and hit the wrapped state's C++
    # implementation directly -- without these overrides they'd return
    # the wrong seat's value when called with an external player id.
    def player_return(self, player: int) -> float:
        if player < 0:
            return self.__wrapped__.player_return(player)
        return self.__wrapped__.player_return(_ext_to_int(player))

    def player_reward(self, player: int) -> float:
        if player < 0:
            return self.__wrapped__.player_reward(player)
        return self.__wrapped__.player_reward(_ext_to_int(player))

    # All per-player view methods that pyspiel exposes must permute the
    # external player id to the internal bridge seat. Without these
    # overrides ``__getattr__`` (or the C++ pybind shim) hands the
    # wrapped state the raw external id, leaking the wrong seat's
    # private view -- e.g. external 1 (South) would receive East's hand.
    def information_state_string(self, player: int) -> str:
        if player < 0:
            return self.__wrapped__.information_state_string(player)
        return self.__wrapped__.information_state_string(_ext_to_int(player))

    def information_state_tensor(self, player: int):
        if player < 0:
            return self.__wrapped__.information_state_tensor(player)
        return self.__wrapped__.information_state_tensor(_ext_to_int(player))

    def observation_tensor(self, player: int):
        if player < 0:
            return self.__wrapped__.observation_tensor(player)
        return self.__wrapped__.observation_tensor(_ext_to_int(player))

    def _phase(self) -> str:
        wrapped = self.__wrapped__
        if wrapped.is_terminal():
            return "terminal"
        if wrapped.is_chance_node():
            return "deal"
        # OpenSpiel bridge has exactly 52 chance nodes (one per dealt
        # card) and no further chance during bidding or play. Verified
        # against ``game.max_chance_nodes_in_history()`` -- adding game
        # params like ``dealer``/``dealer_vul``/``use_double_dummy_result``
        # does not introduce more chance nodes.
        actions_after_deal = wrapped.history()[_CARDS_IN_DECK:]
        for act in actions_after_deal:
            if act < _CARDS_IN_DECK:
                return "play"
        # Bridge transitions to play after three consecutive passes
        # following any bid -- even before the opening lead is made.
        if len(actions_after_deal) >= 3:
            if all(a == _BRIDGE_PASS for a in actions_after_deal[-3:]) and any(
                a >= _BRIDGE_FIRST_BID for a in actions_after_deal
            ):
                return "play"
        return "auction"

    def _derive_contract(self) -> dict[str, Any] | None:
        """Reconstruct the final contract from the auction.

        Returns ``None`` for passed-out auctions or before any bid is
        made. Used by ``_plays()`` to attribute each card to a seat
        (opening lead comes from declarer's left).
        """
        wrapped = self.__wrapped__
        dealer = int(wrapped.get_game().get_parameters().get("dealer", 0))
        auction_actions = [a for a in wrapped.history()[_CARDS_IN_DECK:] if a >= _CARDS_IN_DECK]
        if not auction_actions:
            return None

        last_bid_idx = -1
        last_bid: int | None = None
        doubled: str | None = None
        for i, act in enumerate(auction_actions):
            if act == _BRIDGE_PASS:
                continue
            if act == _BRIDGE_DBL:
                doubled = "X"
                continue
            if act == _BRIDGE_RDBL:
                doubled = "XX"
                continue
            last_bid = act
            last_bid_idx = i
            doubled = None
        if last_bid is None:
            return None

        bid_idx = last_bid - _BRIDGE_FIRST_BID
        level = bid_idx // len(_BRIDGE_DENOMS) + 1
        denom_idx = bid_idx % len(_BRIDGE_DENOMS)
        denom = _BRIDGE_DENOMS[denom_idx]
        trump_suit = denom if denom != "NT" else None

        # Calls iterate seats clockwise starting from the dealer.
        last_bid_seat = (dealer + last_bid_idx) % _NUM_PLAYERS
        # Declarer: the first player on the winning side to name this
        # denomination (regardless of bid level).
        declarer_seat = None
        for i, act in enumerate(auction_actions):
            if act < _BRIDGE_FIRST_BID:
                continue
            seat = (dealer + i) % _NUM_PLAYERS
            if seat % 2 != last_bid_seat % 2:
                continue
            if (act - _BRIDGE_FIRST_BID) % len(_BRIDGE_DENOMS) == denom_idx:
                declarer_seat = seat
                break
        if declarer_seat is None:
            return None
        return {
            "level": level,
            "denom": denom,
            "trump_suit": trump_suit,
            "doubled": doubled,
            "declarer_internal_seat": declarer_seat,
            "declarer_player_id": _int_to_ext(declarer_seat),
            "declarer_table_position": _TABLE_POSITIONS[declarer_seat],
        }

    def _auction(self) -> list[dict[str, Any]]:
        """Parsed auction history (one entry per call)."""
        wrapped = self.__wrapped__
        dealer = int(wrapped.get_game().get_parameters().get("dealer", 0))
        bids: list[dict[str, Any]] = []
        seat = dealer
        for act in wrapped.history()[52:]:
            if act < 52:
                break  # play has started; remaining actions are cards
            bids.append(
                {
                    "internal_seat": seat,
                    "table_position": _TABLE_POSITIONS[seat],
                    "player_id": _int_to_ext(seat),
                    "team_id": _team_of(_int_to_ext(seat)),
                    "call": wrapped.action_to_string(seat, act),
                    "action": int(act),
                }
            )
            seat = (seat + 1) % _NUM_PLAYERS
        return bids

    def _plays(self, contract: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Card play history with seat attribution.

        Opening leader is to declarer's left; subsequent leaders are
        trick winners. Within a trick, seats cycle clockwise from the
        leader. Trick winners are computed locally from the trump suit
        (in the contract) and the cards' suits/ranks.
        """
        wrapped = self.__wrapped__
        play_actions = [a for a in wrapped.history()[_CARDS_IN_DECK:] if a < _CARDS_IN_DECK]
        if not play_actions:
            return []
        if contract is None:
            contract = self._derive_contract()
        if contract is None:
            return []

        leader = (contract["declarer_internal_seat"] + 1) % _NUM_PLAYERS
        trump = contract["trump_suit"]

        plays: list[dict[str, Any]] = []
        trick_cards: list[str] = []
        trick_seats: list[int] = []
        for act in play_actions:
            card_str = wrapped.action_to_string(0, act)
            seat = (leader + len(trick_cards)) % _NUM_PLAYERS
            plays.append(
                {
                    "card": card_str,
                    "action": int(act),
                    "internal_seat": seat,
                    "table_position": _TABLE_POSITIONS[seat],
                    "player_id": _int_to_ext(seat),
                    "trick_number": len(plays) // _NUM_PLAYERS + 1,
                }
            )
            trick_cards.append(card_str)
            trick_seats.append(seat)
            if len(trick_cards) == _NUM_PLAYERS:
                winner_idx = _trick_winner_idx(trick_cards, trump)
                leader = trick_seats[winner_idx]
                trick_cards = []
                trick_seats = []
        return plays

    def state_dict(self, player: int | None = None) -> dict[str, Any]:
        wrapped = self.__wrapped__
        dealer_internal = int(wrapped.get_game().get_parameters().get("dealer", 0))
        ext_cp = self.current_player()
        result: dict[str, Any] = {
            "phase": self._phase(),
            "is_terminal": wrapped.is_terminal(),
            "num_teams": 2,
            "players_per_team": _PLAYERS_PER_TEAM,
            "dealer_internal_seat": dealer_internal,
            "dealer_table_position": _TABLE_POSITIONS[dealer_internal],
            "dealer_player_id": _int_to_ext(dealer_internal),
            "table_seating": {
                # external_pid -> table position
                str(ext): _table_position(ext)
                for ext in range(_NUM_PLAYERS)
            },
            "teams": {
                "0": [0, 1],  # team A: external pids 0 and 1 (N, S)
                "1": [2, 3],  # team B: external pids 2 and 3 (E, W)
            },
        }
        if ext_cp >= 0:
            result["current_player_id"] = ext_cp
            result["current_table_position"] = _table_position(ext_cp)
            result["current_team_id"] = _team_of(ext_cp)
        else:
            result["current_player_id"] = None
            result["current_table_position"] = None
            result["current_team_id"] = None

        result["auction"] = self._auction()
        contract = self._derive_contract()
        result["contract"] = contract
        result["plays"] = self._plays(contract)

        if player is not None and 0 <= player < _NUM_PLAYERS:
            partner = _partner_of(player)
            result["your_player_id"] = player
            result["your_team_id"] = _team_of(player)
            result["your_table_position"] = _table_position(player)
            result["your_partner_player_id"] = partner
            result["partner_table_position"] = _table_position(partner)
            result["opponent_player_ids"] = [p for p in range(_NUM_PLAYERS) if _team_of(p) != _team_of(player)]
            result["your_turn"] = ext_cp == player
            # Bridge's per-seat raw observation: hand, vulnerability,
            # auction; reveals all hands once play begins (standard
            # bridge convention -- dummy is exposed after the opening
            # lead).
            if not wrapped.is_chance_node():
                result["raw_observation"] = wrapped.observation_string(_ext_to_int(player))
            if not wrapped.is_terminal() and ext_cp == player:
                result["legal_actions"] = [
                    {
                        "action": int(a),
                        "label": wrapped.action_to_string(_ext_to_int(player), a),
                    }
                    for a in wrapped.legal_actions(_ext_to_int(player))
                ]

        if wrapped.is_terminal():
            external_returns = self.returns()
            team_totals = [
                external_returns[0] + external_returns[1],  # team A (N+S)
                external_returns[2] + external_returns[3],  # team B (E+W)
            ]
            result["returns"] = external_returns
            result["team_totals"] = team_totals
            if team_totals[0] > team_totals[1]:
                result["winning_team"] = 0
            elif team_totals[1] > team_totals[0]:
                result["winning_team"] = 1
            else:
                result["winning_team"] = "draw"

        return result

    def to_json(self, player: int | None = None) -> str:
        return json.dumps(self.state_dict(player))

    def observation_string(self, player: int) -> str:
        return self.to_json(player)

    def __str__(self) -> str:
        return self.to_json()


class _PermutedObservation(proxy._Observation):
    """Standard OpenSpiel observer that permutes external -> internal seat.

    The shared ``proxy._Observation`` forwards ``player`` straight to the
    wrapped state's observer, which would hand external 1 (South) the
    raw ``information_state_tensor(1)`` -- i.e. East's view. Override
    the per-player methods to apply ``_PERM`` first, mirroring the
    overrides on ``BridgeArenaState``.
    """

    def set_from(self, state: BridgeArenaState, player: int) -> None:
        internal = _ext_to_int(player) if player >= 0 else player
        self.__wrapped__.set_from(state.__wrapped__, internal)

    def string_from(self, state: BridgeArenaState, player: int) -> str | None:
        internal = _ext_to_int(player) if player >= 0 else player
        return self.__wrapped__.string_from(state.__wrapped__, internal)


class BridgeArenaGame(proxy.Game):
    """Bridge Arena: AABB external players mapped onto NESW bridge seats."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("bridge", params)
        super().__init__(
            wrapped,
            short_name="bridge_arena",
            long_name="Bridge Arena (2v2 AABB over NESW)",
        )

    def new_initial_state(self, *args) -> BridgeArenaState:
        return BridgeArenaState(self.__wrapped__.new_initial_state(*args), game=self)

    def make_py_observer(
        self,
        iig_obs_type: pyspiel.IIGObservationType | None = None,
        params: dict[str, Any] | None = None,
    ) -> pyspiel.Observer:
        return _PermutedObservation(
            observation.make_observation(self.__wrapped__, iig_obs_type, params)
        )


pyspiel.register_game(BridgeArenaGame().get_type(), BridgeArenaGame)
