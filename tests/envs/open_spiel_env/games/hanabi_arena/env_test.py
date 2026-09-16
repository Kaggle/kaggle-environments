"""Tests for the Hanabi Arena env (4-player 2v2 OpenSpiel game)."""

import collections
import json
import random

import pyspiel
from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env
from kaggle_environments.envs.open_spiel_env.games.hanabi_arena import (  # noqa: F401
    hanabi_arena_game,
)


def _new_state(seed=0, **params):
    return pyspiel.load_game("hanabi_arena", {"seed": seed, **params}).new_initial_state()


def _obs(state, player):
    return json.loads(state.observation_string(player))


def _play_out(state, seed=0, policy=None):
    """Run a state to terminal with random (or supplied) play."""
    rng = random.Random(seed)
    while not state.is_terminal():
        legal = state.legal_actions()
        state.apply_action(policy(state, rng) if policy else rng.choice(legal))
    return state


def _safe_move(state, rng=None):
    """Never play a card, so the table runs its deck dry instead of bombing.

    Discarding is illegal while the info tokens are full, so hints come
    first and the discard resumes once a token has been spent.
    """
    player = state.current_player()
    labels = {a: state.action_to_string(player, a) for a in state.legal_actions()}
    for prefix in ("Discard", "Reveal"):
        matching = [a for a, label in labels.items() if label.lstrip("(").startswith(prefix)]
        if matching:
            return matching[0]
    return next(iter(labels))


def _action_id(state, label):
    player = state.current_player()
    for action in state.legal_actions():
        if state.action_to_string(player, action) == label:
            return action
    raise AssertionError(f"{label!r} is not legal here")


def _first_rank_hint(state):
    """First legal rank hint as ``(action_id, rank)``.

    Which ranks are hintable depends on what the deal put in the target's
    hand, so tests pick from the legal set rather than naming a rank.
    """
    player = state.current_player()
    for action in state.legal_actions():
        label = state.action_to_string(player, action)
        if "rank" in label:
            return action, int(label.rstrip(")").split()[-1])
    raise AssertionError("no rank hint is legal here")


# Sequential player order: tables alternate, seats alternate within a table.
_PLAYER_ORDER = [0, 2, 1, 3]

# OpenSpiel's ColorIndexToChar order (hanabi_lib/util.cc).
_COLOR_LETTERS = "RYGWB"


class StructureTest(absltest.TestCase):
    """Basic game shape: 4 players, 2 teams, sequential turns."""

    def test_four_players(self):
        self.assertEqual(pyspiel.load_game("hanabi_arena").num_players(), 4)

    def test_sequential_dynamics(self):
        game = pyspiel.load_game("hanabi_arena")
        self.assertEqual(game.get_type().dynamics, pyspiel.GameType.Dynamics.SEQUENTIAL)
        state = _new_state(seed=1)
        self.assertFalse(state.is_simultaneous_node())
        self.assertEqual(state.current_player(), 0)

    def test_no_chance_nodes_are_exposed(self):
        # The deck is pre-shuffled from `seed` and dealt internally, so the
        # env interpreter never has to resolve a chance node for this game.
        game = pyspiel.load_game("hanabi_arena")
        self.assertEqual(game.get_type().chance_mode, pyspiel.GameType.ChanceMode.DETERMINISTIC)
        state = _play_out(_new_state(seed=6), seed=6)
        self.assertNotIn(-1, [item.player for item in state.full_history()])

    def test_only_acting_player_has_legal_actions(self):
        state = _new_state(seed=1)
        counts = [len(state.legal_actions(p)) for p in range(4)]
        self.assertGreater(counts[0], 0)
        self.assertEqual(counts[1:], [0, 0, 0])

    def test_player_order_interleaves_teams(self):
        state = _new_state(seed=1)
        observed = []
        for _ in range(8):
            observed.append(state.current_player())
            state.apply_action(_safe_move(state))
        self.assertEqual(observed, _PLAYER_ORDER * 2)

    def test_cards_are_dealt_before_the_first_turn(self):
        obs = _obs(_new_state(seed=1), 0)
        hands = obs["table"]["hands"]
        self.assertLen(hands, 2)
        for hand in hands:
            self.assertLen(hand["cards"], 5)
        # Two seats of 5 out of a 50-card deck.
        self.assertEqual(obs["table"]["deck_size"], 40)
        self.assertEqual(obs["table"]["deck_total"], 50)


class MirroringTest(absltest.TestCase):
    """Both tables are dealt from one shared, identically-ordered deck."""

    def test_both_tables_get_the_same_deal(self):
        # The whole point of the variant: AA-vs-BB is the same puzzle twice,
        # so the score gap measures the teams and not the shuffle.
        state = _new_state(seed=4)
        team_a = _obs(state, 0)["table"]
        team_b = _obs(state, 2)["table"]
        # Seat 0 of each team sees its teammate's hand; they must match.
        self.assertEqual(team_a["hands"][1]["cards"], team_b["hands"][1]["cards"])
        # And the hidden hands match too, visible via the full reveal.
        full = json.loads(str(state))
        self.assertEqual(
            [[card["card"] for card in hand["cards"]] for hand in full["tables"][0]["hands"]],
            [[card["card"] for card in hand["cards"]] for hand in full["tables"][1]["hands"]],
        )

    def test_draw_order_stays_mirrored(self):
        # Identical deals are not enough -- the two tables must also draw
        # replacements in the same order, or they diverge after turn one.
        state = _new_state(seed=8)
        for _ in range(8):  # Two full rounds: each seat discards once.
            state.apply_action(_safe_move(state))
        full = json.loads(str(state))
        self.assertEqual(
            [[card["card"] for card in hand["cards"]] for hand in full["tables"][0]["hands"]],
            [[card["card"] for card in hand["cards"]] for hand in full["tables"][1]["hands"]],
        )
        self.assertEqual(full["tables"][0]["discards"], full["tables"][1]["discards"])

    def test_same_seed_identical(self):
        self.assertEqual(_obs(_new_state(seed=99), 0), _obs(_new_state(seed=99), 0))

    def test_different_seed_differs(self):
        self.assertNotEqual(_obs(_new_state(seed=99), 0), _obs(_new_state(seed=100), 0))


class DeckRecoveryTest(absltest.TestCase):
    """The pre-shuffled deck must be the engine's deck, at every board size.

    The arena deals both tables itself, so the deck it builds has to be the
    exact multiset Hanabi would have dealt. Recovering it from the root chance
    distribution alone is not enough: the probabilities are counts over a
    total the distribution does not state, and inferring that total from the
    smallest probability assumes a singleton card. At ranks=1 there is none --
    every card is a rank-1 triple -- so the deck comes out short and the table
    runs off the end of it mid-game.
    """

    # (colors, ranks, hand_size, deck total). ranks=1 is the case with no
    # singleton card; ranks=2 has no middle ranks; the rest bracket them.
    _BOARDS = [
        (5, 5, 5, 50),
        (5, 1, 2, 15),
        (3, 1, 2, 9),
        (5, 2, 2, 20),
        (2, 3, 2, 12),
        (1, 5, 2, 10),
    ]

    def test_deck_is_the_full_engine_deck(self):
        for colors, ranks, hand_size, total in self._BOARDS:
            with self.subTest(colors=colors, ranks=ranks):
                state = _new_state(seed=3, colors=colors, ranks=ranks, hand_size=hand_size)
                self.assertLen(state._deck, total)
                # Per color: 3 of rank 1, 1 of the top rank, 2 of each middle.
                expected = collections.Counter()
                for color in range(colors):
                    for rank in range(1, ranks + 1):
                        count = 3 if rank == 1 else 1 if rank == ranks else 2
                        expected[f"{_COLOR_LETTERS[color]}{rank}"] = count
                table = pyspiel.load_game(
                    "hanabi",
                    {"players": 2, "colors": colors, "ranks": ranks, "hand_size": hand_size},
                )
                labels = collections.Counter(
                    table.new_initial_state().action_to_string(pyspiel.PlayerId.CHANCE, action)[len("(Deal ") : -1]
                    for action in state._deck
                )
                self.assertEqual(labels, expected)

    def test_every_board_size_plays_to_terminal(self):
        # The truncated deck dealt fine and only failed on a draw past its
        # end, many moves in -- so the regression has to play the game out.
        for colors, ranks, hand_size, _ in self._BOARDS:
            with self.subTest(colors=colors, ranks=ranks):
                state = _play_out(
                    _new_state(seed=3, colors=colors, ranks=ranks, hand_size=hand_size),
                    seed=3,
                )
                self.assertTrue(state.is_terminal())
                self.assertEqual(sum(state.returns()), 0.0)

    def test_deck_survives_a_table_that_draws_every_card(self):
        # The truncated deck only failed on the draw past its end, so the
        # regression needs a table that actually exhausts it: never play a
        # card, and both tables discard their way through the whole deck.
        state = _play_out(
            _new_state(seed=3, colors=5, ranks=1, hand_size=2),
            seed=3,
            policy=_safe_move,
        )
        for team in range(2):
            # Every card drawn, and not one more than the deck holds.
            self.assertEqual(state._deck_index[team], len(state._deck))
            self.assertEqual(_obs(state, team * 2)["tables"][team]["deck_size"], 0)


class ObservationTest(absltest.TestCase):
    """Per-player views: own table only, own hand hidden."""

    def test_player_sees_only_their_own_table(self):
        state = _new_state(seed=4)
        obs0 = _obs(state, 0)
        self.assertEqual(obs0["your_team_id"], 0)
        self.assertEqual(obs0["your_seat"], 0)
        self.assertEqual(obs0["teammate_player_id"], 1)
        self.assertEqual(obs0["table"]["team_id"], 0)
        # Mid-game the cross-table reveal is withheld.
        self.assertNotIn("tables", obs0)
        obs2 = _obs(state, 2)
        self.assertEqual(obs2["your_team_id"], 1)
        self.assertEqual(obs2["teammate_player_id"], 3)
        self.assertEqual(obs2["table"]["team_id"], 1)

    def test_observation_hides_own_hand(self):
        # The core of Hanabi: you see your partner's cards, never your own.
        state = _new_state(seed=4)
        for player in range(4):
            obs = _obs(state, player)
            seat = obs["your_seat"]
            hands = obs["table"]["hands"]
            own = hands[seat]
            other = hands[1 - seat]
            self.assertTrue(own["is_observer"])
            self.assertFalse(other["is_observer"])
            self.assertTrue(all(card["card"] is None for card in own["cards"]))
            self.assertTrue(all(card["card"] is not None for card in other["cards"]))

    def test_hands_are_labelled_with_arena_player_ids(self):
        obs = _obs(_new_state(seed=4), 2)
        self.assertEqual([hand["player_id"] for hand in obs["table"]["hands"]], [2, 3])

    def test_teammates_see_the_same_table_from_opposite_sides(self):
        state = _new_state(seed=4)
        obs0, obs1 = _obs(state, 0), _obs(state, 1)
        self.assertEqual(obs0["table"]["fireworks"], obs1["table"]["fireworks"])
        self.assertEqual(obs0["table"]["deck_size"], obs1["table"]["deck_size"])
        # Each sees the other's cards, so the two views disagree on hands.
        self.assertNotEqual(obs0["table"]["hands"], obs1["table"]["hands"])

    def test_only_the_actor_gets_legal_actions(self):
        # Legal hints enumerate exactly the colors and ranks IN a hand, so
        # handing a non-actor the actor's move list would leak the hand.
        state = _new_state(seed=4)
        self.assertNotEmpty(_obs(state, 0)["table"]["legal_actions"])
        for player in (1, 2, 3):
            self.assertEmpty(_obs(state, player)["table"]["legal_actions"])

    def test_your_turn_flag_tracks_the_active_player(self):
        state = _new_state(seed=4)
        flags = [_obs(state, player)["your_turn"] for player in range(4)]
        self.assertEqual(flags, [True, False, False, False])
        self.assertEqual(_obs(state, 0)["active_player_id"], 0)


class TurnHistoryTest(absltest.TestCase):
    """Each table records both teammates' moves so partners can review them."""

    def test_history_is_per_table_and_names_both_seats(self):
        state = _new_state(seed=2)
        for _ in range(4):  # One move each, in order [0, 2, 1, 3].
            state.apply_action(_safe_move(state))
        history_a = _obs(state, 0)["table"]["move_history"]
        self.assertEqual([move["player_id"] for move in history_a], [0, 1])
        history_b = _obs(state, 2)["table"]["move_history"]
        self.assertEqual([move["player_id"] for move in history_b], [2, 3])
        # Labels are the engine's own move strings, and the two mirrored
        # tables were driven by the same policy, so they read alike.
        for move in history_a + history_b:
            self.assertRegex(move["label"], r"^\((Play|Discard|Reveal)")
        self.assertEqual(
            [move["label"] for move in history_a],
            [move["label"] for move in history_b],
        )

    def test_history_records_the_face_of_a_card_that_left_a_hand(self):
        # The face becomes common knowledge the instant the card leaves, so
        # recording it leaks nothing -- and it is the one fact a harness
        # cannot recover from a view that hides the hand it came from.
        state = _new_state(seed=0)
        cards = _obs(state, 1)["table"]["hands"][0]["cards"]
        slot = next(i for i, c in enumerate(cards) if c["card"]["rank"] == 1)
        expected = cards[slot]["card"]
        state.apply_action(_action_id(state, f"(Play {slot})"))
        entry = _obs(state, 0)["table"]["move_history"][-1]
        self.assertEqual(entry["removed_slot"], slot)
        self.assertEqual(entry["card"], expected)
        self.assertTrue(entry["advanced"], "a rank 1 on an empty board advances its firework")

    def test_history_marks_a_misplay_as_not_advancing(self):
        state = _new_state(seed=2)
        cards = _obs(state, 1)["table"]["hands"][0]["cards"]
        slot = next(i for i, c in enumerate(cards) if c["card"]["rank"] >= 3)
        state.apply_action(_action_id(state, f"(Play {slot})"))
        entry = _obs(state, 0)["table"]["move_history"][-1]
        self.assertFalse(entry["advanced"])
        self.assertEqual(_obs(state, 0)["table"]["life_tokens"], 2)

    def test_history_records_which_slots_a_hint_pointed_at(self):
        state = _new_state(seed=2)
        cards = _obs(state, 0)["table"]["hands"][1]["cards"]
        action, rank = _first_rank_hint(state)
        expected = [i for i, c in enumerate(cards) if c["card"]["rank"] == rank]
        state.apply_action(action)
        entry = _obs(state, 0)["table"]["move_history"][-1]
        self.assertEqual(entry["hint_kind"], "rank")
        self.assertEqual(entry["hint_value"], rank)
        self.assertEqual(entry["target_player_id"], 1)
        self.assertEqual(entry["slots_when_given"], expected)
        self.assertEqual(entry["slots"], expected)

    def test_hint_slots_follow_the_cards_when_a_lower_slot_leaves(self):
        # Slots are positions, not identities. Removing slot j slides every
        # higher card down one, so a hint recorded against slot 4 now points
        # at a card that was never hinted. `slots` is walked forward;
        # `slots_when_given` keeps the public record of what was pointed at.
        state = _new_state(seed=2)
        action, rank = _first_rank_hint(state)
        cards = _obs(state, 0)["table"]["hands"][1]["cards"]
        touched = [i for i, c in enumerate(cards) if c["card"]["rank"] == rank]
        self.assertTrue(any(s > 0 for s in touched), "need a hinted slot above slot 0")
        state.apply_action(action)
        state.apply_action(_safe_move(state))  # Team B's turn.
        state.apply_action(_action_id(state, "(Discard 0)"))  # P1 removes slot 0.

        entry = _obs(state, 0)["table"]["move_history"][0]
        self.assertEqual(entry["slots_when_given"], touched)
        self.assertEqual(entry["slots"], [s - 1 for s in touched if s > 0])

    def test_a_removal_does_not_reindex_the_entry_reporting_it(self):
        # The discard entry names the slot the card sat in when it left. If
        # the reindex ran before the entry was appended -- or over it -- the
        # log would report a slot the card never occupied.
        state = _new_state(seed=2)
        state.apply_action(_safe_move(state))  # Spend a token so discard is legal.
        state.apply_action(_safe_move(state))  # Team B's turn.
        state.apply_action(_action_id(state, "(Discard 2)"))
        entry = _obs(state, 0)["table"]["move_history"][-1]
        self.assertEqual(entry["removed_slot"], 2)
        self.assertTrue(entry["label"].startswith("(Discard 2)"))

    def test_a_draw_leaves_lower_hint_slots_alone(self):
        # Every removal is followed by a draw, and the drawn card lands in the
        # highest slot -- above everything already held. So discarding the top
        # slot shifts nothing below it, and the refill must not be mistaken
        # for one either.
        state = _new_state(seed=2)
        action, rank = _first_rank_hint(state)
        cards = _obs(state, 0)["table"]["hands"][1]["cards"]
        touched = [i for i, c in enumerate(cards) if c["card"]["rank"] == rank]
        top = len(cards) - 1
        self.assertNotIn(top, touched, "the top slot must not be one of the hinted ones")
        state.apply_action(action)
        state.apply_action(_safe_move(state))  # Team B's turn.
        state.apply_action(_action_id(state, f"(Discard {top})"))  # P1 sheds the top slot.

        entry = _obs(state, 0)["table"]["move_history"][0]
        self.assertEqual(entry["slots"], touched)
        self.assertLen(_obs(state, 0)["table"]["hands"][1]["cards"], len(cards), "the hand refilled")

    def test_the_opposing_seats_hints_are_never_reindexed_together(self):
        # Two tables, two independent hands. A removal at one table must not
        # touch the other's hint bookkeeping.
        state = _new_state(seed=2)
        state.apply_action(_first_rank_hint(state)[0])  # P0 hints P1.
        state.apply_action(_first_rank_hint(state)[0])  # P2 hints P3, mirrored.
        table_b_before = _obs(state, 2)["table"]["move_history"][0]["slots"]
        state.apply_action(_action_id(state, "(Discard 0)"))  # P1 discards.
        self.assertEqual(_obs(state, 2)["table"]["move_history"][0]["slots"], table_b_before)


class HiddenStateTest(absltest.TestCase):
    """The serialized blob rebuilds every hand, so agents must not get it."""

    def test_the_state_declares_itself_hidden(self):
        self.assertTrue(_new_state().hides_state_from_agents())

    def test_the_env_omits_the_blob_from_agent_observations(self):
        env = make("open_spiel_hanabi_arena", debug=True)
        env.reset()
        env.step([{"submission": -1}] * 4)
        for agent in env.state:
            self.assertNotIn("serializedGameAndState", agent["observation"])

    def test_the_blob_would_have_exposed_the_readers_own_hand(self):
        # What the omission is protecting: deserializing it and asking any
        # SEAT for its view reveals the hand the arena view hides.
        game = pyspiel.load_game("hanabi_arena", {"seed": 5})
        state = game.new_initial_state()
        hidden = _obs(state, 0)["table"]["hands"][0]["cards"]
        self.assertTrue(all(card["card"] is None for card in hidden))
        _, restored = pyspiel.deserialize_game_and_state(pyspiel.serialize_game_and_state(game, state))
        # The mirrored teammate's view of the same table shows seat 0's cards.
        revealed = _obs(restored, 1)["table"]["hands"][0]["cards"]
        self.assertTrue(all(card["card"] is not None for card in revealed))


class ActionLabelTest(absltest.TestCase):
    """Labels must render for every player argument the platform passes."""

    def test_labels_agree_across_every_seat_argument(self):
        # The action ids are the table's own, so a label must not depend on
        # which arena player id it is asked about.
        state = _new_state(seed=4)
        actor = state.current_player()
        for action in state.legal_actions():
            expected = state.action_to_string(actor, action)
            for player in range(4):
                self.assertEqual(state.action_to_string(player, action), expected, (player, action))

    def test_a_negative_player_id_still_renders_a_label(self):
        # OpenSpiel passes sentinel ids (TERMINAL is -4) and the single-arg
        # action_to_string resolves to one. Indexing a table with -4 would
        # raise and take the episode down with it.
        state = _new_state(seed=4)
        action = state.legal_actions()[0]
        self.assertEqual(state.action_to_string(action), state.action_to_string(state.current_player(), action))

        terminal = _play_out(_new_state(seed=6), seed=6)
        self.assertEqual(terminal.current_player(), pyspiel.PlayerId.TERMINAL)
        # Nothing is legal, but the renderer must answer rather than raise.
        self.assertRegex(terminal.action_to_string(0), r"^\((Play|Discard|Reveal)")


class TerminationTest(absltest.TestCase):
    """Both tables must finish; a finished one is skipped, not waited on."""

    def test_terminal_only_when_both_tables_finish(self):
        state = _play_out(_new_state(seed=6), seed=6)
        self.assertTrue(state.is_terminal())
        self.assertEqual(state.current_player(), pyspiel.PlayerId.TERMINAL)
        self.assertEmpty(state.legal_actions())

    def test_finished_table_is_skipped(self):
        # Tables bomb out at different times, so the surviving team must be
        # able to keep taking turns alone rather than deadlocking.
        state = _new_state(seed=0)
        rng = random.Random(0)
        tail_teams = []
        while not state.is_terminal():
            tail_teams.append(state.current_player() // 2)
            state.apply_action(rng.choice(state.legal_actions()))
        # Seed 0 finishes unevenly: the tail is a run by one team alone.
        self.assertEqual(tail_teams[-2:], [tail_teams[-1]] * 2)
        obs = _obs(state, 0)
        self.assertTrue(obs["tables"][0]["is_terminal"])
        self.assertTrue(obs["tables"][1]["is_terminal"])

    def test_your_table_finished_flag(self):
        state = _new_state(seed=0)
        rng = random.Random(0)
        while not state.is_terminal():
            finished = [_obs(state, pid * 2)["your_table_finished"] for pid in range(2)]
            active_team = state.current_player() // 2
            # Whoever is on the clock must be at a table still in play.
            self.assertFalse(finished[active_team])
            state.apply_action(rng.choice(state.legal_actions()))


class ScoringTest(absltest.TestCase):
    """Head-to-head result to both seats; raw scores kept as team_totals."""

    def test_both_seats_share_their_team_result(self):
        state = _play_out(_new_state(seed=6), seed=6)
        returns = state.returns()
        self.assertEqual(returns[0], returns[1])
        self.assertEqual(returns[2], returns[3])

    def test_returns_rank_the_teams_rather_than_paying_the_score(self):
        # The payout must be the head-to-head result: a team losing 24-25
        # must not out-earn a team winning 3-0, and a played-out win must be
        # worth the same as the +/-1 the interpreter pays for a forfeit.
        for seed in range(12):
            state = _play_out(_new_state(seed=seed), seed=seed)
            obs = _obs(state, 0)
            a, b = obs["team_totals"]
            expected = [0.0] * 4 if a == b else [1.0, 1.0, -1.0, -1.0] if a > b else [-1.0, -1.0, 1.0, 1.0]
            self.assertEqual(obs["returns"], expected)
            self.assertCountEqual(set(obs["returns"]), set(expected))

    def test_returns_sum_to_zero(self):
        for seed in range(12):
            state = _play_out(_new_state(seed=seed), seed=seed)
            self.assertEqual(sum(state.returns()), 0.0)

    def test_team_totals_agree_with_the_tables(self):
        state = _play_out(_new_state(seed=6), seed=6)
        obs = _obs(state, 0)
        for team, table in enumerate(obs["tables"]):
            self.assertEqual(table["score"], obs["team_totals"][team])

    def test_winning_team_matches_the_totals(self):
        for seed in range(12):
            state = _play_out(_new_state(seed=seed), seed=seed)
            obs = _obs(state, 0)
            a, b = obs["team_totals"]
            expected = 0 if a > b else 1 if b > a else "draw"
            self.assertEqual(obs["winning_team"], expected)

    def test_higher_scoring_team_wins(self):
        # Random play scores 0-0, which would let a broken head-to-head pass
        # as a draw forever. So drive a deliberate asymmetry: team A only
        # plays cards it can see are correct (cheating via the full reveal),
        # while team B plays blind and burns its lives.
        def lopsided(state, rng):
            player = state.current_player()
            team = player // 2
            labels = {a: state.action_to_string(player, a) for a in state.legal_actions()}
            plays = [a for a, label in labels.items() if label.startswith("(Play")]
            if team == 1:
                return plays[0] if plays else next(iter(labels))
            # Team A: consult the hidden hand and play only a card that
            # extends its colour's stack right now.
            table = json.loads(str(state))["tables"][team]
            hand = table["hands"][player % 2]["cards"]
            for action, label in labels.items():
                if not label.startswith("(Play"):
                    continue
                card = hand[int(label[len("(Play ") : -1])]["card"]
                if card and table["fireworks"][card["color"]] == card["rank"] - 1:
                    return action
            return _safe_move(state)

        state = _play_out(_new_state(seed=6), seed=6, policy=lopsided)
        obs = _obs(state, 0)
        team_a, team_b = obs["team_totals"]
        self.assertGreater(team_a, team_b)
        self.assertEqual(obs["winning_team"], 0)
        self.assertEqual(obs["returns"], [1.0, 1.0, -1.0, -1.0])

    def test_bombing_out_scores_zero(self):
        # HanabiState::Score() returns 0 outright at zero lives, so a table
        # that built stacks before dying still banks nothing.
        found = False
        for seed in range(12):
            obs = _obs(_play_out(_new_state(seed=seed), seed=seed), 0)
            for team, table in enumerate(obs["tables"]):
                if table["outcome"] != "lives_exhausted":
                    continue
                found = True
                self.assertEqual(table["life_tokens"], 0)
                self.assertEqual(table["score"], 0)
                self.assertEqual(obs["team_totals"][team], 0)
        self.assertTrue(found, "no table bombed out across the sampled seeds")

    def test_max_score_is_reported(self):
        self.assertEqual(_obs(_new_state(seed=1), 0)["max_score"], 25)

    def test_utility_bounds_are_the_head_to_head_scale(self):
        # Not the Hanabi score: the utilities are the +1/-1/0 result, so the
        # declared bounds and utility_sum have to match that, not max_score.
        game = pyspiel.load_game("hanabi_arena")
        self.assertEqual(game.min_utility(), -1.0)
        self.assertEqual(game.max_utility(), 1.0)
        self.assertEqual(game.utility_sum(), 0.0)


class EndgameCountdownTest(absltest.TestCase):
    """The last-turns countdown must reflect the table's own history.

    ``proxy.State`` does not forward ``full_history``, so a view wrapping a
    table it did not create would read the countdown off an empty history
    and report it frozen at its starting value.
    """

    def test_final_turns_remaining_counts_down(self):
        state = _new_state(seed=8, colors=2, ranks=2, hand_size=2)
        seen = []
        while not state.is_terminal():
            obs = _obs(state, state.current_player())
            table = obs["table"]
            if table["deck_size"] > 0:
                self.assertIsNone(table["final_turns_remaining"])
            else:
                seen.append(table["final_turns_remaining"])
            state.apply_action(_safe_move(state))
        self.assertTrue(seen, "the deck never ran dry")
        self.assertLessEqual(seen[0], 2)
        self.assertEqual(seen, sorted(seen, reverse=True))

    def test_move_number_advances(self):
        state = _new_state(seed=3)
        self.assertEqual(_obs(state, 0)["move_number"], 0)
        for _ in range(4):
            state.apply_action(_safe_move(state))
        self.assertEqual(_obs(state, 0)["move_number"], 4)
        # Each table has seen half of them.
        self.assertEqual(_obs(state, 0)["table"]["move_number"], 2)


class TerminalRevealTest(absltest.TestCase):
    """At terminal, every hand on both tables becomes visible."""

    def test_terminal_reveals_both_tables(self):
        state = _play_out(_new_state(seed=6), seed=6)
        obs = _obs(state, 0)
        self.assertEqual(obs["phase"], "terminal")
        self.assertLen(obs["tables"], 2)
        self.assertIn("returns", obs)
        self.assertIn("team_totals", obs)

    def test_terminal_reveal_unhides_every_hand(self):
        # A Hanabi view always hides its own observer's hand, so the reveal
        # has to be stitched from both seats -- otherwise the replay would
        # still be missing a hand per table.
        state = _play_out(_new_state(seed=6), seed=6)
        for table in _obs(state, 0)["tables"]:
            for hand in table["hands"]:
                self.assertFalse(hand["is_observer"])
                for card in hand["cards"]:
                    self.assertIsNotNone(card["card"])


class SerializationTest(absltest.TestCase):
    """States must survive the round-trip the env does on every step."""

    def test_serialize_and_deserialize_midgame(self):
        game = pyspiel.load_game("hanabi_arena", {"seed": 5})
        state = game.new_initial_state()
        rng = random.Random(3)
        for _ in range(9):
            state.apply_action(rng.choice(state.legal_actions()))
        game2, state2 = pyspiel.deserialize_game_and_state(pyspiel.serialize_game_and_state(game, state))
        self.assertEqual(state2.current_player(), state.current_player())
        self.assertEqual(state2.observation_string(0), state.observation_string(0))
        self.assertEqual(state2.legal_actions(), state.legal_actions())


class KaggleEnvIntegrationTest(absltest.TestCase):
    """Full episodes through the kaggle env wrapper."""

    def test_random_agents_run_to_completion(self):
        env = make(
            "open_spiel_hanabi_arena",
            configuration={"openSpielGameParameters": {"seed": 11}, "includeLegalActions": True},
            debug=True,
        )
        env.run(["random"] * 4)
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_hanabi_arena")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))
        # Teammates are paid identically; the two teams are scored apart.
        rewards = playthrough["rewards"]
        self.assertEqual(rewards[0], rewards[1])
        self.assertEqual(rewards[2], rewards[3])

    def test_legal_actions_match_the_observation(self):
        env = make(
            "open_spiel_hanabi_arena",
            configuration={"includeLegalActions": True, "seed": 0},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}] * 4)  # Initial setup step.
        obs = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(
            [a["action"] for a in obs["table"]["legal_actions"]],
            env.state[0]["observation"]["legalActions"],
        )

    def test_invalid_action(self):
        env = make("open_spiel_hanabi_arena", debug=True)
        env.reset()
        env.step([{"submission": -1}] * 4)  # Initial setup step.
        env.step([{"submission": 999}] + [{"submission": -1}] * 3)  # Invalid action.
        self.assertTrue(env.done)
        self.assertEqual(
            env.toJSON()["rewards"][0],
            open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
        )

    def test_a_forfeit_is_charged_to_the_whole_team(self):
        # The forfeiter's teammate cannot keep playing -- its table is over --
        # so paying it the same winning reward as the two opponents would hand
        # a losing team a 50% win rate on its own forfeits and corrupt the Elo
        # signal on three of the four seats.
        env = make("open_spiel_hanabi_arena", debug=True)
        env.reset()
        env.step([{"submission": -1}] * 4)  # Initial setup step.
        env.step([{"submission": 999}] + [{"submission": -1}] * 3)  # P0 forfeits.
        rewards = env.toJSON()["rewards"]
        loss = open_spiel_env.DEFAULT_INVALID_ACTION_REWARD
        self.assertEqual(rewards, [loss, loss, -loss, -loss])


if __name__ == "__main__":
    absltest.main()
