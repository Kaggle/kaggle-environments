"""Exact evaluation of strategies in small two-player zero-sum poker games.

Kuhn and Leduc poker are small enough to evaluate a strategy exactly: the
value of a strategy profile, the best response to a strategy, and from those
NashConv and exploitability. This module builds both games in sequence form
and computes those quantities with vectorised NumPy, in a few hundred
microseconds for Kuhn and a few milliseconds for Leduc, which makes bootstrap
intervals over thousands of resampled strategies practical.

It depends on NumPy only. The games follow the rules and default parameters
of OpenSpiel's ``kuhn_poker`` and ``leduc_poker``, so results are comparable
with ``open_spiel.python.algorithms.exploitability``.

Games
-----

``kuhn_poker()``
    Three cards (J < Q < K), one card each, ante 1, one bet of size 1.
    Player 0 checks or bets. After a check player 1 checks (showdown, +-1) or
    bets, and player 0 then folds (-1) or calls (showdown, +-2). After a bet
    player 1 folds (+1) or calls (+-2). Value for player 0: -1/18.

``leduc_poker()``
    Six cards (J, Q, K in two suits), one private card each, ante 1, two
    betting rounds with raise sizes 2 and 4, at most two raises per round,
    player 0 opens both rounds, one public card dealt before round two.
    Fold is legal only when facing a raise; call is a check when there is
    nothing to call. A fold loses the folder's contribution. At showdown a
    private card pairing the public card wins, otherwise the higher rank
    wins, and equal ranks split. Value for player 0: about -0.0856.

    Information sets are keyed by rank by default. Suits never affect the
    outcome, so this is lossless: 144 information sets per player.
    ``leduc_poker(suits=True)`` keys them by card identity as OpenSpiel does
    (468 per player). Both are the same game.

Conventions
-----------

* A behavioural strategy is a 2-D array per player: one row per information
  set, one column per action. Rows with fewer actions than the widest row are
  padded with zeros on the right. ``Treeplex.index`` maps an information-set
  key to its row and ``Treeplex.actions`` lists each row's action names.
* ``nash_conv`` is the SUM of both players' best-response gains.
  ``exploitability`` is ``nash_conv / 2``. These match OpenSpiel's names.
* Payoffs are in antes (chips), from player 0's point of view in ``payoff``.

Scope
-----

This module is the scoring core only. Nothing in the environment calls it
yet. Planned follow-ups, not part of this module:

* ``kuhn_poker`` and ``leduc_poker`` entries in ``GAMES_LIST``, with proxies
  and harnesses.
* An elicitation runner that asks a model for its action at every
  information set and scores the resulting strategy with a bootstrap
  interval.
* A parity test against ``open_spiel.python.algorithms.exploitability``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Hashable, Sequence

import numpy as np


class Treeplex:
    """One player's decision structure in sequence form.

    Information sets are sorted by depth (the number of the player's own
    earlier decisions), so every parent comes before its children. Sequence 0
    is the empty sequence; every (information set, action) pair is one further
    sequence.

    Attributes:
        keys: Information-set keys, in row order.
        actions: Action names of each information set, in column order.
        index: Maps an information-set key to its row.
        n_info: Number of information sets.
        n_act: Number of actions of each information set.
        n_seq: Number of sequences, including the empty one.
        shape: Shape of a behavioural strategy array, ``(n_info, width)``.
        mask: True where an (information set, action) entry exists.
        parent: Sequence leading to each information set.
        seq_of: Sequence of each (information set, action) entry. Padding
            entries point at a dummy slot ``n_seq``.
    """

    def __init__(
        self,
        keys: Sequence[Hashable],
        actions: Sequence[Sequence[str]],
        parent_seq: Sequence[int],
    ):
        n = len(keys)
        counts = np.array([len(a) for a in actions], dtype=int)
        first_old = 1 + np.concatenate([[0], np.cumsum(counts)])[:n]
        owner = {}
        for i in range(n):
            for a in range(counts[i]):
                owner[int(first_old[i]) + a] = i
        # Creation order is top-down, so a parent's depth is known already.
        depth = np.zeros(n, dtype=int)
        for i in range(n):
            if parent_seq[i] != 0:
                depth[i] = depth[owner[parent_seq[i]]] + 1
        order = np.argsort(depth, kind="stable")

        self.keys = [keys[i] for i in order]
        self.actions = [tuple(actions[i]) for i in order]
        self.index = {key: row for row, key in enumerate(self.keys)}
        self.n_info = n
        self.n_act = counts[order]
        self.width = int(self.n_act.max())
        self.shape = (n, self.width)
        self.n_seq = 1 + int(self.n_act.sum())

        first_new = 1 + np.concatenate([[0], np.cumsum(self.n_act)])[:n]
        # Maps a sequence id used while building to its id after sorting.
        self.renumber = np.zeros(self.n_seq, dtype=int)
        for new_row, old_row in enumerate(order):
            for a in range(self.n_act[new_row]):
                self.renumber[int(first_old[old_row]) + a] = first_new[new_row] + a
        self.parent = self.renumber[np.array([parent_seq[i] for i in order], dtype=int)]

        cols = np.arange(self.width)[None, :]
        self.mask = cols < self.n_act[:, None]
        self.seq_of = np.where(self.mask, first_new[:, None] + cols, self.n_seq)

        sorted_depth = depth[order]
        cuts = np.flatnonzero(np.diff(sorted_depth)) + 1
        starts = np.concatenate([[0], cuts])
        stops = np.concatenate([cuts, [n]])
        self.levels = [slice(int(a), int(b)) for a, b in zip(starts, stops)]
        self._uniform = self.mask / self.n_act[:, None]

    def uniform(self) -> np.ndarray:
        """Returns the strategy that plays every legal action equally often."""
        return self._uniform.copy()

    def regret_matching(self, regrets: np.ndarray) -> np.ndarray:
        """Returns the regret-matching strategy for the given regrets.

        Positive parts are normalised per information set; where they all
        vanish the strategy is uniform.
        """
        positive = np.maximum(regrets, 0.0)
        total = positive.sum(axis=1, keepdims=True)
        return np.where(total > 0, positive / np.where(total > 0, total, 1.0), self._uniform)

    def plan(self, strategy: np.ndarray) -> np.ndarray:
        """Converts a behavioural strategy to a realization plan.

        The plan has one entry per sequence: the product of the player's own
        action probabilities along that sequence. ``plan[0]`` is 1.
        """
        plan = np.zeros(self.n_seq + 1)
        plan[0] = 1.0
        for level in self.levels:
            plan[self.seq_of[level]] = plan[self.parent[level]][:, None] * strategy[level]
        return plan[: self.n_seq]

    def behavioural(self, plan: np.ndarray) -> np.ndarray:
        """Converts a realization plan to a behavioural strategy.

        Information sets the plan never reaches get the uniform strategy.
        """
        padded = np.concatenate([plan, [0.0]])
        numerator = padded[self.seq_of]
        denominator = plan[self.parent][:, None]
        reached = denominator > 1e-300
        ratio = numerator / np.where(reached, denominator, 1.0)
        return np.where(reached, ratio, self._uniform) * self.mask

    def regrets(self, gradient: np.ndarray, strategy: np.ndarray) -> tuple[np.ndarray, float]:
        """Returns instantaneous counterfactual regrets and the strategy's value.

        Args:
            gradient: Expected payoff collected at leaves, per own sequence,
                with the opponent's and chance's probabilities folded in.
            strategy: The player's behavioural strategy.

        Returns:
            ``(regrets, value)``; regrets are zero on padding entries.
        """
        acc = np.zeros(self.n_seq + 1)
        acc[: self.n_seq] = gradient
        action_values = np.empty(self.shape)
        for level in reversed(self.levels):
            q = acc[self.seq_of[level]]
            action_values[level] = q
            acc += np.bincount(
                self.parent[level],
                weights=(strategy[level] * q).sum(axis=1),
                minlength=self.n_seq + 1,
            )
        state_values = (strategy * action_values).sum(axis=1)
        return (action_values - state_values[:, None]) * self.mask, float(acc[0])

    def best_response(self, gradient: np.ndarray) -> tuple[float, np.ndarray]:
        """Returns the best-response value and one pure best response.

        Args:
            gradient: As in ``regrets``.

        Returns:
            ``(value, strategy)``; ties go to the first maximising action.
        """
        acc = np.zeros(self.n_seq + 1)
        acc[: self.n_seq] = gradient
        response = np.zeros(self.shape)
        for level in reversed(self.levels):
            q = np.where(self.mask[level], acc[self.seq_of[level]], -np.inf)
            best = q.argmax(axis=1)
            response[level][np.arange(q.shape[0]), best] = 1.0
            acc += np.bincount(self.parent[level], weights=q.max(axis=1), minlength=self.n_seq + 1)
        return float(acc[0]), response


@dataclasses.dataclass
class SequenceFormGame:
    """A two-player zero-sum game in sequence form.

    Attributes:
        name: Short name of the game.
        payoff: Payoff to player 0 per pair of sequences, with the chance
            probabilities folded in. Player 1 receives the negative.
        players: The two players' treeplexes.
    """

    name: str
    payoff: np.ndarray
    players: tuple[Treeplex, Treeplex]

    def __post_init__(self):
        self._payoff_t = np.ascontiguousarray(self.payoff.T)

    def uniform(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the uniform strategy of each player."""
        return self.players[0].uniform(), self.players[1].uniform()

    def gradient(self, player: int, opponent_strategy: np.ndarray) -> np.ndarray:
        """Returns ``player``'s payoff per own sequence against the opponent."""
        if player == 0:
            return self.payoff @ self.players[1].plan(opponent_strategy)
        return -(self._payoff_t @ self.players[0].plan(opponent_strategy))

    def value(self, strategy_0: np.ndarray, strategy_1: np.ndarray) -> float:
        """Returns the expected payoff to player 0 under the profile."""
        plan_0 = self.players[0].plan(strategy_0)
        plan_1 = self.players[1].plan(strategy_1)
        return float(plan_0 @ self.payoff @ plan_1)

    def best_response(self, player: int, opponent_strategy: np.ndarray) -> tuple[float, np.ndarray]:
        """Returns ``player``'s best-response value, in its own payoff, and a pure best response."""
        return self.players[player].best_response(self.gradient(player, opponent_strategy))

    def nash_conv(self, strategy_0: np.ndarray, strategy_1: np.ndarray) -> float:
        """Returns the sum of both players' best-response gains.

        The game is zero-sum, so the players' current values cancel and the
        sum equals the two best-response values added together. It is zero
        exactly at a Nash equilibrium.
        """
        return self.best_response(0, strategy_1)[0] + self.best_response(1, strategy_0)[0]

    def exploitability(self, strategy_0: np.ndarray, strategy_1: np.ndarray) -> float:
        """Returns ``nash_conv / 2``, as OpenSpiel defines exploitability."""
        return self.nash_conv(strategy_0, strategy_1) / 2.0


class _Builder:
    """Collects information sets and leaves while a game tree is walked."""

    def __init__(self):
        self._keys = ([], [])
        self._actions = ([], [])
        self._parent = ([], [])
        self._index = ({}, {})
        self._first = ([], [])
        self._n_seq = [1, 1]
        self._leaves = {}

    def info_set(self, player: int, key: Hashable, actions: Sequence[str], parent_seq: int) -> list[int]:
        """Returns the sequence ids of an information set's actions, creating it on first visit."""
        row = self._index[player].get(key)
        if row is None:
            row = len(self._keys[player])
            self._index[player][key] = row
            self._keys[player].append(key)
            self._actions[player].append(tuple(actions))
            self._parent[player].append(parent_seq)
            self._first[player].append(self._n_seq[player])
            self._n_seq[player] += len(actions)
        elif self._actions[player][row] != tuple(actions) or self._parent[player][row] != parent_seq:
            raise ValueError(f"Information set {key} of player {player} is inconsistent.")
        return [self._first[player][row] + a for a in range(len(actions))]

    def leaf(self, seq_0: int, seq_1: int, weighted_payoff: float) -> None:
        """Adds a terminal payoff to player 0, already weighted by its chance probability."""
        self._leaves[(seq_0, seq_1)] = self._leaves.get((seq_0, seq_1), 0.0) + weighted_payoff

    def build(self, name: str) -> SequenceFormGame:
        players = tuple(Treeplex(self._keys[p], self._actions[p], self._parent[p]) for p in (0, 1))
        payoff = np.zeros((players[0].n_seq, players[1].n_seq))
        for (seq_0, seq_1), value in self._leaves.items():
            payoff[players[0].renumber[seq_0], players[1].renumber[seq_1]] += value
        return SequenceFormGame(name, payoff, players)


def kuhn_poker() -> SequenceFormGame:
    """Builds Kuhn poker: 6 information sets per player.

    Information-set keys are ``(card, history)`` with card 0, 1, 2 for J, Q, K
    and history a string of the actions so far: ``"p"`` for pass (check or
    fold) and ``"b"`` for bet (bet or call), as in OpenSpiel's ``kuhn_poker``.
    Action names are ``("pass", "bet")`` at every information set.
    """
    builder = _Builder()
    actions = ("pass", "bet")
    for card_0 in range(3):
        for card_1 in range(3):
            if card_0 == card_1:
                continue
            chance = 1.0 / 6.0
            showdown = 1.0 if card_0 > card_1 else -1.0
            check_0, bet_0 = builder.info_set(0, (card_0, ""), actions, 0)
            check_1, bet_1 = builder.info_set(1, (card_1, "p"), actions, 0)
            builder.leaf(check_0, check_1, chance * showdown)
            fold_0, call_0 = builder.info_set(0, (card_0, "pb"), actions, check_0)
            builder.leaf(fold_0, bet_1, -chance)
            builder.leaf(call_0, bet_1, 2.0 * chance * showdown)
            fold_1, call_1 = builder.info_set(1, (card_1, "b"), actions, 0)
            builder.leaf(bet_0, fold_1, chance)
            builder.leaf(bet_0, call_1, 2.0 * chance * showdown)
    return builder.build("kuhn_poker")


_LEDUC_RAISE_SIZES = (2, 4)
_LEDUC_MAX_RAISES = 2


def leduc_poker(suits: bool = False) -> SequenceFormGame:
    """Builds Leduc poker with the default parameters of OpenSpiel's ``leduc_poker``.

    Information-set keys are ``(own card, public card, round-one betting,
    round-two betting)``. Betting is a string over ``f`` (fold), ``c`` (call
    or check) and ``r`` (raise). The public card is ``None`` in round one. A
    card is its rank 0, 1, 2 (J, Q, K) by default, or ``(rank, suit)`` with
    ``suits=True``.

    Args:
        suits: Key information sets by card identity, as OpenSpiel does,
            instead of by rank. The game is the same either way.
    """
    deck = [(rank, suit) for rank in range(3) for suit in range(2)]
    builder = _Builder()

    def identity(card):
        return card if suits else card[0]

    def showdown(card_0, card_1, public):
        if card_0[0] == public[0]:
            return 1.0
        if card_1[0] == public[0]:
            return -1.0
        return float(np.sign(card_0[0] - card_1[0]))

    def play(cards, public, round_index, round_one, betting, contributions, n_raises, seqs, chance):
        # Player 0 opens both rounds and the players alternate within a round.
        player = len(betting) % 2
        stakes = max(contributions)
        actions = []
        if contributions[player] < stakes:
            actions.append("f")
        actions.append("c")
        if n_raises < _LEDUC_MAX_RAISES:
            actions.append("r")
        key = (
            identity(cards[player]),
            None if public is None else identity(public),
            round_one,
            betting,
        )
        action_seqs = builder.info_set(player, key, actions, seqs[player])
        for action, seq in zip(actions, action_seqs):
            next_seqs = (seq, seqs[1]) if player == 0 else (seqs[0], seq)
            paid = list(contributions)
            if action == "f":
                loss = -contributions[0] if player == 0 else contributions[1]
                builder.leaf(next_seqs[0], next_seqs[1], chance * loss)
            elif action == "r":
                paid[player] = stakes + _LEDUC_RAISE_SIZES[round_index]
                play(cards, public, round_index, round_one, betting + "r", tuple(paid), n_raises + 1, next_seqs, chance)
            else:
                paid[player] = stakes
                if not betting:
                    # An opening check: the other player still has to act.
                    play(cards, public, round_index, round_one, "c", tuple(paid), n_raises, next_seqs, chance)
                elif round_index == 0:
                    remaining = [card for card in deck if card not in cards]
                    for card in remaining:
                        play(cards, card, 1, betting + "c", "", tuple(paid), 0, next_seqs, chance / len(remaining))
                else:
                    builder.leaf(next_seqs[0], next_seqs[1], chance * showdown(cards[0], cards[1], public) * paid[0])

    n_deals = len(deck) * (len(deck) - 1)
    for card_0 in deck:
        for card_1 in deck:
            if card_0 != card_1:
                play((card_0, card_1), None, 0, "", "", (1, 1), 0, (0, 0), 1.0 / n_deals)
    return builder.build("leduc_poker_suits" if suits else "leduc_poker")
