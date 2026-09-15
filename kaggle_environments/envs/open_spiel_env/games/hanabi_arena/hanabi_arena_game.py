"""Hanabi Arena: 2v2 team variant of OpenSpiel's Hanabi.

Hanabi is fully cooperative -- every seat shares one score and OpenSpiel
reports ``Utility.IDENTICAL``, so a single table has no winner to rank by.
This variant manufactures a head-to-head: two teams of two each play their
own private 2-player Hanabi table, and the higher final score wins.

Team A is players 0,1; team B is players 2,3. The two tables are dealt from
the *same shuffled deck* -- identical starting hands and identical draw
order -- so an AA-vs-BB matchup is the same puzzle solved twice, and the
score gap measures the teams rather than the shuffle.

Players take turns one-at-a-time, alternating tables so the two teams
interleave. Within a table the acting seat is whichever seat Hanabi itself
says is up. Tables can finish at different move counts (a team that bombs
out early ends sooner); once one table is terminal the other simply takes
every remaining turn. The episode ends when both tables are terminal.

Players only ever see their own table: their teammate's hand face-up, their
own as hint knowledge only, and nothing at all of the opposing table. At
terminal every hand on both tables is revealed for scoring and replay. The
per-table ``move_history`` carries the public facts each action revealed --
the face of a card that left a hand, whether a play advanced its firework,
and which slots a hint pointed at -- so an agent never has to reconstruct
them from a serialized state that would also hand it everything hidden.

Both seats on the higher-scoring table are paid +1 and both seats on the
lower-scoring table -1; equal scores are a draw and pay 0. The raw Hanabi
scores (0 to ``colors * ranks``) are reported in the observation as
``team_totals`` / ``winning_team``.

The deck is pre-shuffled from the ``seed`` parameter, so this game exposes no
chance nodes of its own -- ``seed`` alone determines the deal.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

import numpy as np
import pyspiel

from ..hanabi.hanabi_proxy import HanabiGame, HanabiState

_NUM_TEAMS = 2
_PLAYERS_PER_TEAM = 2
_NUM_PLAYERS = _NUM_TEAMS * _PLAYERS_PER_TEAM

_DEFAULT_COLORS = 5
_DEFAULT_RANKS = 5
_DEFAULT_HAND_SIZE = 5
_DEFAULT_MAX_LIFE_TOKENS = 3
_DEFAULT_MAX_INFORMATION_TOKENS = 8


_GAME_TYPE = pyspiel.GameType(
    short_name="hanabi_arena",
    long_name="Hanabi Arena (2v2)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    # The deck is pre-shuffled from `seed`, so the arena itself never stops
    # at a chance node -- it pumps each table's deals internally.
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    # Not IDENTICAL: unlike a single Hanabi table, the two teams here score
    # independently and are ranked against each other. The payout is the
    # head-to-head result (+1 / -1 / 0 all round), which sums to zero.
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=False,
    parameter_specification={
        "colors": _DEFAULT_COLORS,
        "ranks": _DEFAULT_RANKS,
        "hand_size": _DEFAULT_HAND_SIZE,
        "max_life_tokens": _DEFAULT_MAX_LIFE_TOKENS,
        "max_information_tokens": _DEFAULT_MAX_INFORMATION_TOKENS,
        "seed": 0,
    },
)


def _team_of(player_id: int) -> int:
    return player_id // _PLAYERS_PER_TEAM


def _seat_of(player_id: int) -> int:
    return player_id % _PLAYERS_PER_TEAM


def _team_player_ids(team: int) -> list[int]:
    base = team * _PLAYERS_PER_TEAM
    return [base + s for s in range(_PLAYERS_PER_TEAM)]


def _table_params(params: dict[str, Any]) -> dict[str, Any]:
    """The underlying per-table Hanabi parameters."""
    return {
        "players": _PLAYERS_PER_TEAM,
        "colors": int(params.get("colors", _DEFAULT_COLORS)),
        "ranks": int(params.get("ranks", _DEFAULT_RANKS)),
        "hand_size": int(params.get("hand_size", _DEFAULT_HAND_SIZE)),
        "max_life_tokens": int(params.get("max_life_tokens", _DEFAULT_MAX_LIFE_TOKENS)),
        "max_information_tokens": int(params.get("max_information_tokens", _DEFAULT_MAX_INFORMATION_TOKENS)),
    }


_DECK_SIZE_RE = re.compile(r"^Deck size:\s*(\d+)\s*$", re.MULTILINE)


def _shuffled_deck(table_game: pyspiel.Game, seed: int) -> list[int]:
    """A full deck as a shuffled list of card actions, dealt to both tables.

    The root chance distribution lists every distinct card with probability
    ``count / deck_total``, so multiplying back out recovers the exact card
    multiset -- no need to restate NumberCardInstances, whose
    bottom-rank-before-top-rank ordering is easy to get wrong at ranks=1.

    The total is READ from the undealt state rather than inferred from the
    smallest probability. Inferring it assumes some card is a singleton, which
    holds only while the top rank has one copy: at ranks=1 every card is a
    rank-1 triple, so the smallest probability is 3/total and the inferred deck
    comes out a third of its true size. That shortfall is invisible at deal
    time and surfaces many moves later as an IndexError on the first draw past
    the truncated deck.

    Dealing this permutation position-by-position is always legal: the i-th
    card of a permutation of the whole deck is necessarily still among the
    cards the engine has left after the first i draws.
    """
    initial = table_game.new_initial_state()
    match = _DECK_SIZE_RE.search(initial.observation_string(0))
    if not match:
        raise ValueError("Hanabi's undealt observation did not report a deck size.")
    total = int(match.group(1))
    deck: list[int] = []
    for action, probability in initial.chance_outcomes():
        deck.extend([action] * round(probability * total))
    if len(deck) != total:
        raise ValueError(f"Recovered {len(deck)} cards from a {total}-card Hanabi deck.")
    random.Random(seed).shuffle(deck)
    return deck


_PLAY_LABEL_RE = re.compile(r"^\(Play (\d+)\)$")
_DISCARD_LABEL_RE = re.compile(r"^\(Discard (\d+)\)$")
_REVEAL_LABEL_RE = re.compile(r"^\(Reveal player \+(\d+) (color|rank) (\w+)\)$")


def _hand_cards(table: dict[str, Any], seat: int) -> list[dict[str, Any]]:
    for hand in table.get("hands") or []:
        if hand.get("player") == seat:
            return hand.get("cards") or []
    return []


class _TableView(HanabiState):
    """Reads a table the arena owns, using the base Hanabi proxy's parser.

    ``proxy.State`` forwards most calls to the wrapped state but not
    ``full_history`` / ``move_number``, which stay on the proxy's own
    (empty) history. The base proxy never notices because it creates the
    state it wraps; here the table is created and advanced by the arena, so
    both must be forwarded explicitly -- otherwise the endgame countdown
    reads as though no turns had been taken.
    """

    def full_history(self):
        return self.__wrapped__.full_history()

    def move_number(self) -> int:
        return self.__wrapped__.move_number()


class HanabiArenaGame(pyspiel.Game):
    """OpenSpiel game: two parallel Hanabi tables, 2 LLMs per team."""

    def __init__(self, params: dict[str, Any] | None = None):
        params = params or {}
        self.seed = int(params.get("seed", 0))
        self.table_params = _table_params(params)
        self.max_score = self.table_params["colors"] * self.table_params["ranks"]

        # The tables are real Hanabi games; let Hanabi itself size the
        # action space and horizon rather than recomputing them here.
        table_game = pyspiel.load_game("hanabi", dict(self.table_params))
        self.max_moves = _NUM_TEAMS * table_game.max_game_length()

        game_info = pyspiel.GameInfo(
            num_distinct_actions=table_game.num_distinct_actions(),
            max_chance_outcomes=0,
            num_players=_NUM_PLAYERS,
            # Utilities are the head-to-head result, not the Hanabi score --
            # see HanabiArenaState.returns().
            min_utility=-1.0,
            max_utility=1.0,
            utility_sum=0.0,
            max_game_length=self.max_moves,
        )
        super().__init__(_GAME_TYPE, game_info, params)

    def new_initial_state(self):
        return HanabiArenaState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return HanabiArenaObserver()


class HanabiArenaState(pyspiel.State):
    """State for Hanabi Arena.

    Holds one real Hanabi state per team plus the shared deck both tables
    are dealt from. Sequential play: tables alternate, and within a table
    the engine picks the seat. ``current_player()`` returns the active
    external player id (or TERMINAL).
    """

    def __init__(self, game: HanabiArenaGame):
        super().__init__(game)
        # Copied onto the state rather than read back off `self._game` at
        # use time: a state rebuilt by pyspiel.deserialize_game_and_state
        # is not guaranteed to carry the Python game's attributes.
        self._table_params = dict(game.table_params)
        self._max_score = game.max_score
        self._max_moves = game.max_moves
        self._move_number = 0
        # Which table moves next; flips after every applied action.
        self._next_table = 0
        self._move_history: list[list[dict[str, Any]]] = [[] for _ in range(_NUM_TEAMS)]

        # One proxy game shared by every view -- it is stateless, and
        # constructing it per observation would be wasted work.
        self._view_game = HanabiGame(dict(self._table_params))
        table_game = pyspiel.load_game("hanabi", dict(self._table_params))
        self._deck = _shuffled_deck(table_game, game.seed)
        self._deck_index = [0, 0]
        self._tables = [table_game.new_initial_state() for _ in range(_NUM_TEAMS)]
        for team in range(_NUM_TEAMS):
            self._deal(team)

    # --- Table plumbing ----------------------------------------------------

    def _deal(self, team: int) -> None:
        """Resolve a table's pending chance nodes off the shared deck."""
        table = self._tables[team]
        while table.is_chance_node():
            table.apply_action(self._deck[self._deck_index[team]])
            self._deck_index[team] += 1

    def _live_tables(self) -> list[int]:
        return [team for team in range(_NUM_TEAMS) if not self._tables[team].is_terminal()]

    def _active_table(self) -> int | None:
        """The table to move next, skipping one that has already finished."""
        live = self._live_tables()
        if not live:
            return None
        return self._next_table if self._next_table in live else live[0]

    # --- OpenSpiel core ----------------------------------------------------

    def _current_player_id(self) -> int | None:
        team = self._active_table()
        if team is None:
            return None
        return team * _PLAYERS_PER_TEAM + int(self._tables[team].current_player())

    def current_player(self):
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        return self._current_player_id()

    def _legal_actions(self, player: int):
        if self.is_terminal() or player != self._current_player_id():
            return []
        team = _team_of(player)
        return self._tables[team].legal_actions(_seat_of(player))

    def _apply_action(self, action: int) -> None:
        if self.is_terminal():
            return
        team = self._active_table()
        table = self._tables[team]
        seat = int(table.current_player())
        label = table.action_to_string(seat, action)
        before = self._table_view(team, None)
        table.apply_action(action)
        # Refill the hand the move just emptied, from the shared deck.
        self._deal(team)

        entry: dict[str, Any] = {
            "seat": seat,
            "player_id": team * _PLAYERS_PER_TEAM + seat,
            "action": action,
            "label": label,
        }
        entry.update(self._public_facts(team, seat, label, before))
        self._move_history[team].append(entry)
        # After the entry is appended, so a play or discard never re-indexes
        # the slot it is itself reporting.
        if "removed_slot" in entry:
            self._reindex_hint_slots(team, seat, entry["removed_slot"])
        self._move_number += 1
        self._next_table = (team + 1) % _NUM_TEAMS

    def _public_facts(
        self,
        team: int,
        seat: int,
        label: str,
        before: dict[str, Any],
    ) -> dict[str, Any]:
        """What this action made public at its table, recorded as it resolved.

        Everything here became common knowledge the instant the move landed --
        the face of a card that left a hand, whether a play advanced its
        firework, and which slots a hint pointed at -- so recording it leaks
        nothing the seats at this table did not already see. Recording it HERE
        is what lets an agent read its own table's history without
        deserializing a state that would also hand it every hidden hand.
        """
        match = _PLAY_LABEL_RE.match(label) or _DISCARD_LABEL_RE.match(label)
        if match:
            slot = int(match.group(1))
            cards = _hand_cards(before, seat)
            facts: dict[str, Any] = {
                "removed_slot": slot,
                "card": cards[slot].get("card") if slot < len(cards) else None,
            }
            if label.startswith("(Play"):
                after = self._table_view(team, None)
                # Stack heights, not the banked score: a bomb-out zeroes the
                # score on the very play that caused it, which would read as
                # "no firework advanced" for a card that did advance one.
                facts["advanced"] = after.get("fireworks_total", 0) > before.get("fireworks_total", 0)
            return facts

        match = _REVEAL_LABEL_RE.match(label)
        if not match:
            return {}
        offset, kind, value = int(match.group(1)), match.group(2), match.group(3)
        if kind == "rank":
            value = int(value)
        target_seat = (seat + offset) % _PLAYERS_PER_TEAM
        cards = _hand_cards(before, target_seat)
        touched = [i for i, card in enumerate(cards) if (card.get("card") or {}).get(kind) == value]
        return {
            "target_seat": target_seat,
            "target_player_id": team * _PLAYERS_PER_TEAM + target_seat,
            "hint_kind": kind,
            "hint_value": value,
            # Two slot lists, because the two questions a reader asks have
            # different answers: `slots_when_given` is the public record of
            # what was pointed at, and `slots` is where those same cards sit
            # NOW, kept current by _reindex_hint_slots. Publishing only the
            # first would have every hint disagree with the live hand as soon
            # as a card left it -- and the holder, who cannot see the faces,
            # has no way to tell which reading is stale.
            "slots_when_given": list(touched),
            "slots": list(touched),
        }

    def _reindex_hint_slots(self, team: int, seat: int, removed_slot: int) -> None:
        """Follow a seat's hinted cards through the shift a removal causes.

        Hanabi slots are positions, not identities: play or discard slot j and
        every higher card slides down one, so a slot number recorded earlier
        now names a different card. Past hints aimed at this seat are walked
        forward -- the removed card drops out of the list, anything above it
        decrements -- so ``slots`` always means "where those cards are today".
        A draw needs no handling: it lands in the highest slot, above every
        card already in the hand.
        """
        for entry in self._move_history[team]:
            if entry.get("target_seat") != seat:
                continue
            entry["slots"] = [s - 1 if s > removed_slot else s for s in entry["slots"] if s != removed_slot]

    def _action_to_string(self, player: int, action: int) -> str:
        # Hanabi's labels are seat-relative in notation ("+1" is an offset
        # from the actor), so a sentinel player is not a seat whose label we
        # can guess. Route it to the table on the clock, which is the only
        # table the action can belong to, and to the seat actually taking it.
        if player < 0:
            team = self._active_table() or 0
            seat = 0 if self._tables[team].is_terminal() else int(self._tables[team].current_player())
        else:
            team, seat = _team_of(player), _seat_of(player)
        return self._tables[team].action_to_string(seat, action)

    def hides_state_from_agents(self) -> bool:
        """Withhold ``serializedGameAndState`` from agent observations.

        Hanabi is hidden information end to end: a seat's own hand, and both
        of the opposing table's hands. The serialized state reconstructs all
        of it, so shipping it alongside a carefully redacted per-player
        observation would make that redaction decorative. Everything an agent
        legitimately needs -- its table's public move history, with the card
        faces and hint slots each move revealed -- is published in the
        observation instead; see ``_public_facts``.
        """
        return True

    def team_of(self, player: int) -> int | None:
        """Which team a seat belongs to, for the interpreter's forfeit scoping.

        A forfeit ends the episode for the offender's teammate too, so the
        interpreter reads this to charge the loss to the whole team rather
        than paying the partner the winning reward.
        """
        if player < 0 or player >= _NUM_PLAYERS:
            return None
        return _team_of(player)

    def is_terminal(self) -> bool:
        return all(table.is_terminal() for table in self._tables)

    def _team_score(self, team: int) -> int:
        # Returns() is Score() for every Hanabi seat and is defined at every
        # state, so this reads the engine's own rule -- including that a
        # bomb-out scores 0 regardless of the stacks already built.
        return int(self._tables[team].returns()[0])

    def returns(self) -> list[float]:
        """+1 to the higher-scoring team, -1 to the lower, 0 to both on a draw.

        Deliberately the head-to-head result rather than the raw 0-to-max
        Hanabi score. The score is what a team earns at its own table, but the
        episode exists to rank the two teams against each other, and paying
        the score makes those two different things: a team losing 24-25 would
        out-earn a team winning 3-0. It would also put the natural payout on a
        different scale from the +/-1 the interpreter pays when an episode ends
        in a forfeit, so a forfeited win and a played-out win would not be
        worth the same. The raw scores stay in the observation as
        ``team_totals`` for the renderer and for analysis.
        """
        rewards = [0.0] * _NUM_PLAYERS
        if not self.is_terminal():
            return rewards
        totals = self._team_totals()
        best = max(totals)
        if totals.count(best) > 1:
            return rewards
        for team in range(_NUM_TEAMS):
            result = 1.0 if totals[team] == best else -1.0
            for pid in _team_player_ids(team):
                rewards[pid] = result
        return rewards

    # --- Rendering & observations -----------------------------------------

    def _table_view(self, team: int, seat: int | None) -> dict[str, Any]:
        """One table as seen by ``seat``, or fully revealed when None.

        A Hanabi observation hides the observer's own hand and nothing
        else, so a genuine full reveal needs exactly two views however
        many seats the table has: seat 0's view carries every hand but
        seat 0's, and any second seat's view carries that one.
        """
        view = _TableView(self._tables[team], game=self._view_game)
        table = view.state_dict(0 if seat is None else seat)
        if seat is None:
            # Seat 1 as the second view, not because teams are pairs but
            # because a Hanabi table is never smaller than two seats, so
            # seat 1 is the one seat besides 0 that always exists.
            others = _TableView(self._tables[team], game=self._view_game).state_dict(1)
            # Each seat's own hand is hidden in its own view; take it from
            # the view where that seat is not the observer.
            table["hands"] = [
                hand if not hand["is_observer"] else others["hands"][hand["player"]] for hand in table["hands"]
            ]
            for hand in table["hands"]:
                hand["is_observer"] = False
            table["observer"] = None
        pids = _team_player_ids(team)
        table["team_id"] = team
        # A table reports its own seat as "up" whenever that table is
        # mid-turn, but the arena clock may be on the other table. Only the
        # player actually on the clock gets a move list, matching what the
        # env hands the agent as `legalActions`.
        if seat is None or pids[seat] != self._current_player_id():
            table["legal_actions"] = []
        # Relabel seats with their arena-wide player ids so agents can talk
        # about "player 3" rather than "the other seat at my table".
        for hand in table["hands"]:
            hand["player_id"] = pids[hand["player"]]
        table["current_player_id"] = (
            None if self._tables[team].is_terminal() else pids[int(self._tables[team].current_player())]
        )
        table["move_history"] = [dict(move) for move in self._move_history[team]]
        # The engine counts deals in its move number; at the arena level
        # only the turns the team actually took are meaningful.
        table["move_number"] = len(self._move_history[team])
        # Hanabi alone has no winner; the arena's winner is across tables.
        table.pop("winner", None)
        table.pop("returns", None)
        return table

    def _team_totals(self) -> list[int]:
        return [self._team_score(team) for team in range(_NUM_TEAMS)]

    def observation_dict(self, player: int | None = None) -> dict[str, Any]:
        terminal = self.is_terminal()
        active_player_id = None if terminal else self._current_player_id()
        result: dict[str, Any] = {
            "phase": "terminal" if terminal else "play",
            "move_number": self._move_number,
            "active_player_id": active_player_id,
            "active_team_id": None if active_player_id is None else _team_of(active_player_id),
            "active_seat": None if active_player_id is None else _seat_of(active_player_id),
            "num_teams": _NUM_TEAMS,
            "players_per_team": _PLAYERS_PER_TEAM,
            "max_score": self._max_score,
            "is_terminal": terminal,
        }
        if player is None:
            # Full reveal -- used for the renderer and for debugging.
            result["tables"] = [self._table_view(team, None) for team in range(_NUM_TEAMS)]
        else:
            team = _team_of(player)
            seat = _seat_of(player)
            result["your_player_id"] = player
            result["your_team_id"] = team
            result["your_seat"] = seat
            result["teammate_player_id"] = [pid for pid in _team_player_ids(team) if pid != player][0]
            result["your_turn"] = not terminal and player == active_player_id
            result["your_table_finished"] = self._tables[team].is_terminal()
            result["table"] = self._table_view(team, seat)
        if terminal:
            team_totals = self._team_totals()
            result["returns"] = self.returns()
            result["team_totals"] = team_totals
            best = max(team_totals)
            winners = [team for team, total in enumerate(team_totals) if total == best]
            result["winning_team"] = winners[0] if len(winners) == 1 else "draw"
            # Reveal both tables at terminal regardless of viewer.
            result["tables"] = [self._table_view(team, None) for team in range(_NUM_TEAMS)]
        return result

    def __str__(self) -> str:
        return json.dumps(self.observation_dict(None))


class HanabiArenaObserver:
    """Per-player JSON observer.

    We expose only the string view (LLM agents read JSON, not tensors).
    OpenSpiel still pokes at ``.tensor`` and ``.dict``, so we provide
    empty placeholders.
    """

    def __init__(self):
        self.tensor = np.zeros(0, dtype=np.float32)
        self.dict = {}

    def set_from(self, state: HanabiArenaState, player: int) -> None:
        pass

    def string_from(self, state: HanabiArenaState, player: int) -> str:
        return json.dumps(state.observation_dict(player))


pyspiel.register_game(_GAME_TYPE, HanabiArenaGame)
