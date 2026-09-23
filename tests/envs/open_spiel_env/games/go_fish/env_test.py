"""Env-level tests for open_spiel_go_fish."""

import json
import random
import re

from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env
from kaggle_environments.envs.open_spiel_env.games.go_fish import go_fish_proxy


def _advance_chance(state, rng):
    while not state.is_terminal() and state.is_chance_node():
        outcomes = state.chance_outcomes()
        state.apply_action(rng.choices([o[0] for o in outcomes], weights=[o[1] for o in outcomes])[0])


class GoFishDeductionTest(absltest.TestCase):
    """The deduction table must retain the full public ask history, even after
    individual asks scroll out of the per-turn ``recent_events`` window."""

    def _play_to_ask(self, seed, plies):
        game = go_fish_proxy.GoFishGame()
        state = game.new_initial_state()
        rng = random.Random(seed)
        _advance_chance(state, rng)
        asks = []  # (asker, target_rank_label) for every ask actually applied
        for _ in range(plies):
            if state.is_terminal() or state.current_player() < 0:
                break
            cp = int(state.current_player())
            action = rng.choice(state.legal_actions())
            action_str = state.action_to_string(action)  # "<target><letter>"
            asks.append((cp, action_str))
            state.apply_action(action)
            _advance_chance(state, rng)
        return game, state, asks

    def test_deductions_present_in_state_dict(self):
        game, state, _ = self._play_to_ask(seed=7, plies=6)
        sd = state.state_dict(0)
        self.assertIn("deductions", sd)
        self.assertEqual(len(sd["deductions"]), 2)
        for d in sd["deductions"]:
            self.assertIn("player", d)
            self.assertIn("known_has", d)
            self.assertIn("known_void", d)
            self.assertIn("wanted", d)

    def test_asker_recorded_as_holding_and_wanting_rank(self):
        # An ask publicly reveals the asker holds >=1 of that rank. That fact
        # must persist in the deduction table regardless of the events window.
        game, state, asks = self._play_to_ask(seed=7, plies=8)
        sd = state.state_dict(0)
        # Every distinct rank that a still-in-game asker asked for is a rank we
        # know they wanted (unless the rank was later booked away). Verify at
        # least one ask survives as a standing "wanted" fact somewhere.
        wanted_any = any(d["wanted"] for d in sd["deductions"])
        self.assertTrue(asks)  # sanity: asks did happen
        self.assertTrue(wanted_any, "no ask survived into the deduction table")

    def test_stale_ask_outlives_events_window(self):
        # The core bug: an ask older than the observer's last turn is dropped
        # from recent_events but MUST remain in the deduction table.
        game, state, asks = self._play_to_ask(seed=11, plies=10)
        observer = int(state.current_player())
        sd = state.state_dict(observer)
        # Standing facts from the whole game are richer than the truncated
        # window: the union of all deduced signals should not be empty when
        # many asks have already happened.
        total_signal = sum(
            len(d["known_has"]) + len(d["known_void"]) + len(d["wanted"])
            for d in sd["deductions"]
            if d["player"] != observer
        )
        self.assertGreater(len(asks), 3)
        self.assertGreater(total_signal, 0, "deduction table lost all history")

    def test_deductions_do_not_leak_hidden_cards(self):
        # known_has must never overstate: it may only assert counts the engine
        # publicly guarantees (player_min_), never the opponent's true hand.
        game, state, _ = self._play_to_ask(seed=3, plies=12)
        observer = 0
        sd = state.state_dict(observer)
        # Opponent's real hand is not observer-visible; the deduction for them
        # may only be a lower bound. Cross-check against the public card total:
        # sum of known minimums can never exceed their public card count.
        players = {p["player"]: p for p in sd["players"]}
        for d in sd["deductions"]:
            pid = d["player"]
            if pid == observer:
                continue
            min_total = 0
            for token in d["known_has"]:  # tokens look like "K>=2"
                min_total += int(token.split(">=")[1])
            self.assertLessEqual(min_total, players[pid]["cards"])


class GoFishAccessorSourceTest(absltest.TestCase):
    """The deduction fields must come from pyspiel's state accessors.

    An earlier version decoded them out of the observation tensor by hardcoded
    offset, recovering ints via ``round(float32 * denom)``. That was exact, but
    it silently depended on the tensor layout: if OpenSpiel ever reorders
    ``ObservationTensor``, offset-based reads keep returning plausible-looking
    numbers and the deductions go quietly wrong. Reading the accessors removes
    that coupling; these tests pin it down.
    """

    def test_state_exposes_the_accessors_we_rely_on(self):
        # If a pyspiel upgrade drops or renames one of these, fail loudly here
        # rather than somewhere deep in the deduction logic.
        game = go_fish_proxy.GoFishGame()
        state = game.new_initial_state().__wrapped__
        for name in (
            "player_min",
            "player_was_asked",
            "drawn_since_was_asked",
            "player_did_ask",
            "pool_size",
            "booked",
        ):
            self.assertTrue(hasattr(state, name), f"pyspiel GoFishState lost .{name}()")
            self.assertTrue(callable(getattr(state, name)))

    def test_accessors_return_exact_ints_not_floats(self):
        # The point of the switch: no float round-tripping. player_min and the
        # ask counters must be plain ints, booked plain bools/ints.
        game = go_fish_proxy.GoFishGame()
        state = game.new_initial_state()
        _advance_chance(state, random.Random(7))
        raw = state.__wrapped__
        self.assertIsInstance(raw.pool_size(), int)
        for grid in (raw.player_min(), raw.drawn_since_was_asked(), raw.player_did_ask()):
            for row in grid:
                for value in row:
                    self.assertIsInstance(value, int)
        for row in raw.player_was_asked():
            for value in row:
                self.assertIsInstance(value, (bool, int))

    def test_matches_legacy_tensor_decode(self):
        # Equivalence with the old offset-based decode, over full games and
        # several deck shapes. Keeps the switch honest: same numbers, safer
        # source. If this ever diverges, one of the two readings is wrong.
        def tensor_counts(raw, num_ranks, num_suits):
            tensor = list(raw.observation_tensor(0))
            pool = round(tensor[num_ranks + 4] * (num_ranks * num_suits))
            booked_base = num_ranks + 4 + 1
            booked = [
                go_fish_proxy._rank_label(r, num_ranks) for r in range(num_ranks) if tensor[booked_base + r] > 0.5
            ]
            return pool, booked

        def tensor_deductions(raw, num_ranks, num_suits, booked):
            tensor = list(raw.observation_tensor(0))
            per_player = 3 + num_ranks * 4
            base0 = num_ranks + 4 + 1 + num_ranks
            booked_set = set(booked)
            out = []
            for pid in range(raw.get_game().num_players()):
                base = base0 + pid * per_player
                known_has, known_void, wanted = [], [], []
                for rank in range(num_ranks):
                    label = go_fish_proxy._rank_label(rank, num_ranks)
                    if label in booked_set:
                        continue
                    b = base + 3 + rank * 4
                    did_ask = round(tensor[b] * (num_suits * num_ranks))
                    was_asked = tensor[b + 1] > 0.5
                    drawn_since = round(tensor[b + 2] * (num_ranks * num_suits))
                    minimum = round(tensor[b + 3] * num_suits)
                    is_void = was_asked and drawn_since == 0 and minimum == 0
                    if minimum > 0:
                        known_has.append(f"{label}>={minimum}")
                    if is_void:
                        known_void.append(label)
                    if did_ask > 0 and not is_void:
                        wanted.append(label)
                out.append(
                    {
                        "player": pid,
                        "known_has": known_has,
                        "known_void": known_void,
                        "wanted": wanted,
                    }
                )
            return out

        turns = 0
        # ranks=30 is where float32 round-tripping is most strained (denominator
        # 120), and ranks=7/suits=6 varies both dimensions off the default.
        for params in ({"ranks": 13, "suits": 4}, {"ranks": 7, "suits": 6}, {"ranks": 30, "suits": 4}):
            num_ranks, num_suits = params["ranks"], params["suits"]
            game = go_fish_proxy.GoFishGame(params)
            state = game.new_initial_state()
            rng = random.Random(4)
            while not state.is_terminal():
                if state.is_chance_node():
                    _advance_chance(state, rng)
                    continue
                turns += 1
                sd = state.state_dict(0)
                raw = state.__wrapped__
                pool, booked = tensor_counts(raw, num_ranks, num_suits)
                self.assertEqual(sd["pool_size"], pool)
                self.assertEqual(sd["booked"], booked)
                self.assertEqual(
                    sd["deductions"],
                    tensor_deductions(raw, num_ranks, num_suits, booked),
                )
                state.apply_action(rng.choice(state.legal_actions()))
        self.assertGreater(turns, 100, "comparison did not cover enough states")


class GoFishStaleDeductionTest(absltest.TestCase):
    """Deductions must retire facts the public record has already invalidated.

    Two tensor fields are cumulative and never expire on their own: a booked
    rank keeps its ask/was-asked history even though nobody can hold it, and
    ``player_did_ask_`` only increments, so a rank stayed "wanted" after the
    asker was shown to hold none of it.
    """

    def _walk(self, seed, fn):
        game = go_fish_proxy.GoFishGame()
        state = game.new_initial_state()
        rng = random.Random(seed)
        while not state.is_terminal():
            if state.is_chance_node():
                _advance_chance(state, rng)
                continue
            fn(state.state_dict(0))
            state.apply_action(rng.choice(state.legal_actions()))

    def test_wanted_never_contradicts_known_void(self):
        # The sharp case: a row saying "known to have none of 9" while also
        # saying "has asked for 9" is self-contradictory advice.
        rows = []

        def check(sd):
            for d in sd["deductions"]:
                rows.append(d)
                overlap = set(d["wanted"]) & set(d["known_void"])
                self.assertEqual(overlap, set(), f"wanted contradicts known_void: {d}")

        self._walk(3, check)
        self.assertGreater(len(rows), 50)  # the walk actually covered a game

    def test_booked_ranks_dropped_from_every_deduction_list(self):
        # Once a rank is booked nobody holds it and nobody can be asked for it,
        # so claims about it are vacuous noise.
        saw_book = False

        def check(sd):
            nonlocal saw_book
            booked = set(sd["booked"])
            if booked:
                saw_book = True
            for d in sd["deductions"]:
                self.assertEqual(set(d["known_void"]) & booked, set())
                self.assertEqual(set(d["wanted"]) & booked, set())
                held = {token.split(">=")[0] for token in d["known_has"]}
                self.assertEqual(held & booked, set())

        self._walk(3, check)
        self.assertTrue(saw_book, "no book was completed, filter untested")

    def test_filtering_preserves_true_claims(self):
        # The filters must only remove invalidated facts, never real signal.
        # Every surviving claim is re-checked against the actual hands.
        game = go_fish_proxy.GoFishGame()
        state = game.new_initial_state()
        rng = random.Random(5)
        total_signal = 0
        while not state.is_terminal():
            if state.is_chance_node():
                _advance_chance(state, rng)
                continue
            sd = state.state_dict(0)
            for d in sd["deductions"]:
                pid = d["player"]
                match = re.search(r"Your cards:\s*(.*)", state.__wrapped__.observation_string(pid))
                held = {letter: int(n) for letter, n in re.findall(r"([a-z])(\d+)", match.group(1))}
                for label in d["known_void"]:
                    letter = chr(ord("a") + go_fish_proxy._STANDARD_RANKS.index(label))
                    self.assertEqual(held.get(letter, 0), 0, f"false void: p{pid} {label}")
                for token in d["known_has"]:
                    label, minimum = token.split(">=")
                    letter = chr(ord("a") + go_fish_proxy._STANDARD_RANKS.index(label))
                    self.assertGreaterEqual(held.get(letter, 0), int(minimum))
                total_signal += len(d["known_has"]) + len(d["known_void"]) + len(d["wanted"])
            state.apply_action(rng.choice(state.legal_actions()))
        # Filtering removed noise, not the signal itself.
        self.assertGreater(total_signal, 100)


class GoFishPublicCountsTest(absltest.TestCase):
    """``pool_size`` and ``booked`` are public state the text observation drops.

    Both come from the observation tensor rather than observation_string, so
    these tests pin the decode against independently-computed ground truth.
    """

    def test_pool_and_booked_present_in_state_dict(self):
        game = go_fish_proxy.GoFishGame()
        state = game.new_initial_state()
        _advance_chance(state, random.Random(7))
        sd = state.state_dict(0)
        # Standard deck, 2 players, 7 cards each: 52 - 14 = 38 left in the pool.
        self.assertEqual(sd["pool_size"], 38)
        self.assertEqual(sd["booked"], [])

    def test_pool_size_matches_deck_arithmetic_all_game(self):
        # Ground truth: pool = deck - cards in hands - cards locked into books.
        # Checked at every decision point of a full game, so it also covers the
        # endgame where the pool drains to 0.
        game = go_fish_proxy.GoFishGame()
        state = game.new_initial_state()
        rng = random.Random(3)
        saw_empty_pool = False
        while not state.is_terminal():
            if state.is_chance_node():
                _advance_chance(state, rng)
                continue
            sd = state.state_dict(0)
            in_hands = sum(p["cards"] for p in sd["players"])
            in_books = sum(p["books"] for p in sd["players"]) * sd["num_suits"]
            expected = sd["num_ranks"] * sd["num_suits"] - in_hands - in_books
            self.assertEqual(sd["pool_size"], expected)
            self.assertGreaterEqual(sd["pool_size"], 0)
            saw_empty_pool = saw_empty_pool or sd["pool_size"] == 0
            state.apply_action(rng.choice(state.legal_actions()))
        # The 0-pool branch is the strategically important one; ensure the loop
        # above actually exercised it rather than passing vacuously.
        self.assertTrue(saw_empty_pool, "never reached an empty pool")

    def test_booked_tracks_completed_books(self):
        # The booked list must always have exactly one entry per book scored,
        # and every label must be a real rank label.
        game = go_fish_proxy.GoFishGame()
        state = game.new_initial_state()
        rng = random.Random(3)
        saw_any_book = False
        while not state.is_terminal():
            if state.is_chance_node():
                _advance_chance(state, rng)
                continue
            sd = state.state_dict(0)
            total_books = sum(p["books"] for p in sd["players"])
            self.assertEqual(len(sd["booked"]), total_books)
            self.assertEqual(len(set(sd["booked"])), len(sd["booked"]))
            for label in sd["booked"]:
                self.assertIn(label, go_fish_proxy._STANDARD_RANKS)
            saw_any_book = saw_any_book or total_books > 0
            state.apply_action(rng.choice(state.legal_actions()))
        self.assertTrue(saw_any_book, "no book was ever completed")

    def test_public_counts_identical_for_every_observer(self):
        # Both fields are common information: each player must see the same
        # values. A per-observer difference would mean a private leak.
        game = go_fish_proxy.GoFishGame()
        state = game.new_initial_state()
        rng = random.Random(11)
        for _ in range(25):
            if state.is_terminal():
                break
            if state.is_chance_node():
                _advance_chance(state, rng)
                continue
            sd0, sd1 = state.state_dict(0), state.state_dict(1)
            self.assertEqual(sd0["pool_size"], sd1["pool_size"])
            self.assertEqual(sd0["booked"], sd1["booked"])
            state.apply_action(rng.choice(state.legal_actions()))

    def test_pool_size_with_non_standard_deck(self):
        # The decode divides by ranks*suits, so a non-default deck would expose
        # any hardcoded 13x4 assumption in the tensor offsets.
        game = go_fish_proxy.GoFishGame({"ranks": 7, "suits": 6})
        state = game.new_initial_state()
        rng = random.Random(2)
        _advance_chance(state, rng)
        sd = state.state_dict(0)
        in_hands = sum(p["cards"] for p in sd["players"])
        self.assertEqual(sd["pool_size"], 7 * 6 - in_hands)


class GoFishHandParityTest(absltest.TestCase):
    """The parsed hand must equal the engine's hand for every deck size.

    OpenSpiel writes a rank as ``chr('a' + rank)``, which leaves a-z once
    ranks > 26: at ranks=30 the last four ranks are ``{``, ``|``, ``}``, ``~``.
    The token regex used to be ``([a-z])(\\d+)``, so those cards were dropped
    from the parsed hand while ``booked``, ``deductions`` and the legal-action
    set still carried them -- the observer saw a hand that could not justify its
    own legal moves. ranks=13 alone never exercises this; the tensor-parity test
    already ran ranks=30 but only compared deductions, never the hand.
    """

    def _assert_hand_matches_engine(self, params, seeds=6):
        num_ranks = params["ranks"]
        num_players = params.get("players", 2)
        checked = 0
        for seed in range(seeds):
            game = go_fish_proxy.GoFishGame(params)
            state = game.new_initial_state()
            rng = random.Random(seed)
            _advance_chance(state, rng)
            while not state.is_terminal():
                for pid in range(num_players):
                    expected = {
                        go_fish_proxy._rank_label(rank, num_ranks): count
                        for rank, count in enumerate(state.__wrapped__.player_cards()[pid])
                        if count
                    }
                    self.assertEqual(state.state_dict(pid)["hand"], expected, f"{params} seed={seed} p{pid}")
                    checked += 1
                state.apply_action(rng.choice(state.legal_actions()))
                _advance_chance(state, rng)
        return checked

    def test_hand_matches_engine_at_default_deck(self):
        self.assertGreater(self._assert_hand_matches_engine({"ranks": 13, "suits": 4}), 100)

    def test_hand_matches_engine_past_the_alphabet(self):
        # 26 is the last all-alphabetic deck; 27 and 30 spill into '{|}~'.
        for ranks in (26, 27, 30):
            with self.subTest(ranks=ranks):
                self.assertGreater(self._assert_hand_matches_engine({"ranks": ranks, "suits": 4}), 100)

    def test_non_alphabetic_ranks_actually_occur(self):
        # Guards the tests above from passing vacuously: if no hand ever holds a
        # past-'z' rank, they prove nothing about the boundary they target.
        game = go_fish_proxy.GoFishGame({"ranks": 30, "suits": 4})
        state = game.new_initial_state()
        rng = random.Random(2)
        _advance_chance(state, rng)
        seen = set()
        while not state.is_terminal() and len(seen) < 1:
            for pid in range(2):
                seen |= {label for label in state.state_dict(pid)["hand"] if not label.isalpha()}
            state.apply_action(rng.choice(state.legal_actions()))
            _advance_chance(state, rng)
        self.assertTrue(seen, "no past-'z' rank ever reached a hand")

    def test_hand_can_justify_every_legal_ask(self):
        # The concrete failure the drop caused: a legal move whose rank is absent
        # from the rendered hand. Asks are only generated for ranks you hold, so
        # every legal action's letter must appear in the hand.
        game = go_fish_proxy.GoFishGame({"ranks": 30, "suits": 4})
        state = game.new_initial_state()
        rng = random.Random(2)
        _advance_chance(state, rng)
        letters = [go_fish_proxy._rank_label(i, 30) for i in range(30)]
        while not state.is_terminal():
            pid = int(state.current_player())
            hand = state.state_dict(pid)["hand"]
            for action in state.legal_actions():
                label = letters[action % 30]
                self.assertIn(label, hand, f"legal ask {state.action_to_string(action)!r} not in hand {hand}")
            state.apply_action(rng.choice(state.legal_actions()))
            _advance_chance(state, rng)


class GoFishEnvTest(absltest.TestCase):
    def test_go_fish_agent_playthrough(self):
        env = make(
            "open_spiel_go_fish",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_go_fish")
        self.assertTrue(all(status == "DONE" for status in playthrough["statuses"]))

    def test_go_fish_observation_is_json(self):
        env = make(
            "open_spiel_go_fish",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # After dealing, it is player 0's turn (Ask phase).
        obs_p0 = json.loads(env.state[0]["observation"]["observationString"])
        self.assertEqual(obs_p0["phase"], "Ask")
        self.assertEqual(obs_p0["current_player"], 0)
        self.assertFalse(obs_p0["is_terminal"])
        # Two-player Go Fish deals 7 cards each.
        self.assertEqual(sum(obs_p0["hand"].values()), 7)
        self.assertEqual(obs_p0["players"][0], {"player": 0, "cards": 7, "books": 0})
        self.assertEqual(obs_p0["players"][1], {"player": 1, "cards": 7, "books": 0})

    def test_go_fish_observation_hides_opponent(self):
        env = make("open_spiel_go_fish", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        # Each player's observation reports only their own hand contents.
        obs_p0 = json.loads(env.state[0]["observation"]["observationString"])
        obs_p1 = json.loads(env.state[1]["observation"]["observationString"])
        self.assertEqual(obs_p0["observer"], 0)
        self.assertEqual(obs_p1["observer"], 1)
        self.assertEqual(sum(obs_p0["hand"].values()), 7)
        self.assertEqual(sum(obs_p1["hand"].values()), 7)
        # The two hands are dealt independently; opponent card contents are not
        # exposed in either observation (only aggregate counts in "players").

    def test_go_fish_invalid_action(self):
        env = make("open_spiel_go_fish", debug=True)
        env.reset()
        env.step([{"submission": -1}, {"submission": -1}])  # Initial setup step.
        env.step([{"submission": 999}, {"submission": -1}])  # Invalid action.
        self.assertTrue(env.done)
        playthrough = env.toJSON()
        self.assertEqual(
            playthrough["rewards"][0],
            open_spiel_env.DEFAULT_INVALID_ACTION_REWARD,
        )


if __name__ == "__main__":
    absltest.main()
