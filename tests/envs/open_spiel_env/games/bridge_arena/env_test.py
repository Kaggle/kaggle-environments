"""Tests for open_spiel_bridge_arena (4-player 2v2 bridge variant).

Bridge Arena exposes external player IDs as AABB (0,1 = team A;
2,3 = team B) over the wrapped bridge's NESW seating, so the table
layout reads ABAB. These tests pin the seat mapping, the partner
relationships, the returns permutation, and end-to-end playthrough
behaviour against the framework.
"""

import json
import random

import pyspiel
from absl.testing import absltest

from kaggle_environments import make
from kaggle_environments.envs.open_spiel_env import open_spiel_env  # noqa: F401
from kaggle_environments.envs.open_spiel_env.games.bridge_arena import bridge_arena_game

# AABB external -> NESW internal: 0->N, 1->S, 2->E, 3->W.
_EXT_TO_TABLE = {0: "N", 1: "S", 2: "E", 3: "W"}
_EXT_TO_INTERNAL = {0: 0, 1: 2, 2: 1, 3: 3}


def _dealt_state(seed=0, **params):
    """Load bridge_arena and resolve the chance (deal) phase."""
    g = pyspiel.load_game("bridge_arena", params)
    s = g.new_initial_state()
    rng = random.Random(seed)
    while s.is_chance_node():
        outcomes, probs = zip(*s.chance_outcomes())
        s.apply_action(rng.choices(outcomes, weights=probs)[0])
    return g, s


class SeatMappingTest(absltest.TestCase):
    """The AABB -> ABAB seat permutation must round-trip and group partners."""

    def test_perm_is_self_inverse(self):
        perm = bridge_arena_game._PERM
        for i in range(4):
            self.assertEqual(perm[perm[i]], i)

    def test_table_seating_groups_partners_opposite(self):
        # Team A (0,1) sits N/S, team B (2,3) sits E/W -- partners face
        # each other so the table reads ABAB clockwise from North.
        self.assertEqual(_EXT_TO_TABLE[0], "N")
        self.assertEqual(_EXT_TO_TABLE[1], "S")
        self.assertEqual(_EXT_TO_TABLE[2], "E")
        self.assertEqual(_EXT_TO_TABLE[3], "W")
        self.assertEqual(bridge_arena_game._table_position(0), "N")
        self.assertEqual(bridge_arena_game._table_position(1), "S")
        self.assertEqual(bridge_arena_game._table_position(2), "E")
        self.assertEqual(bridge_arena_game._table_position(3), "W")

    def test_partner_pairings(self):
        self.assertEqual(bridge_arena_game._partner_of(0), 1)
        self.assertEqual(bridge_arena_game._partner_of(1), 0)
        self.assertEqual(bridge_arena_game._partner_of(2), 3)
        self.assertEqual(bridge_arena_game._partner_of(3), 2)

    def test_team_of(self):
        self.assertEqual(bridge_arena_game._team_of(0), 0)
        self.assertEqual(bridge_arena_game._team_of(1), 0)
        self.assertEqual(bridge_arena_game._team_of(2), 1)
        self.assertEqual(bridge_arena_game._team_of(3), 1)


class StateMappingTest(absltest.TestCase):
    """Current player, legal actions, and returns are permuted correctly."""

    def test_four_players(self):
        g = pyspiel.load_game("bridge_arena")
        self.assertEqual(g.num_players(), 4)

    def test_initial_is_chance(self):
        g = pyspiel.load_game("bridge_arena")
        s = g.new_initial_state()
        # Bridge starts with a 52-card deal as chance nodes.
        self.assertTrue(s.is_chance_node())
        self.assertEqual(s.current_player(), pyspiel.PlayerId.CHANCE)

    def test_current_player_after_deal_is_dealer_external(self):
        # Default dealer is North (internal 0); external maps via _PERM[0] = 0.
        _, s = _dealt_state(seed=1)
        self.assertEqual(s.current_player(), 0)

    def test_current_player_with_dealer_param(self):
        # Set dealer to internal seat 1 (East). External should be _PERM[1] = 2.
        _, s = _dealt_state(seed=1, dealer=1)
        self.assertEqual(s.current_player(), 2)
        # And South dealer (internal 2) -> external 1.
        _, s2 = _dealt_state(seed=1, dealer=2)
        self.assertEqual(s2.current_player(), 1)
        # And West dealer (internal 3) -> external 3.
        _, s3 = _dealt_state(seed=1, dealer=3)
        self.assertEqual(s3.current_player(), 3)

    def test_only_current_player_has_legal_actions(self):
        _, s = _dealt_state(seed=2)
        cp = s.current_player()
        for pid in range(4):
            if pid == cp:
                self.assertGreater(len(s.legal_actions(pid)), 0)
            else:
                self.assertEqual(s.legal_actions(pid), [])

    def test_legal_actions_match_wrapped(self):
        _, s = _dealt_state(seed=3)
        ext_cp = s.current_player()
        internal_cp = _EXT_TO_INTERNAL[ext_cp]
        self.assertEqual(
            list(s.legal_actions(ext_cp)),
            list(s.__wrapped__.legal_actions(internal_cp)),
        )

    def test_apply_action_advances_in_table_order(self):
        # After each non-chance bid, the next active seat moves clockwise
        # at the table: N -> E -> S -> W. In external pids that's
        # 0 -> 2 -> 1 -> 3.
        _, s = _dealt_state(seed=4)
        order = [s.current_player()]
        for _ in range(3):
            s.apply_action(s.legal_actions()[0])  # everyone passes
            order.append(s.current_player())
        self.assertEqual(order, [0, 2, 1, 3])


class ObservationTest(absltest.TestCase):
    """Per-player JSON observation includes correct identity and view."""

    def test_observation_is_json(self):
        _, s = _dealt_state(seed=5)
        obs = json.loads(s.observation_string(0))
        self.assertIn("phase", obs)
        self.assertIn("your_player_id", obs)
        self.assertEqual(obs["your_player_id"], 0)
        self.assertEqual(obs["your_table_position"], "N")
        self.assertEqual(obs["your_partner_player_id"], 1)
        self.assertEqual(obs["partner_table_position"], "S")
        self.assertEqual(obs["opponent_player_ids"], [2, 3])
        self.assertEqual(obs["your_team_id"], 0)
        self.assertEqual(obs["teams"], {"0": [0, 1], "1": [2, 3]})

    def test_observation_for_each_player_identifies_table_seat(self):
        _, s = _dealt_state(seed=5)
        seat_names = {"N": "North", "E": "East", "S": "South", "W": "West"}
        for ext, table in _EXT_TO_TABLE.items():
            obs = json.loads(s.observation_string(ext))
            self.assertEqual(obs["your_player_id"], ext)
            self.assertEqual(obs["your_table_position"], table)
            # Raw observation comes from the corresponding bridge seat.
            self.assertIn(f"You are {seat_names[table]}", obs["raw_observation"])

    def test_legal_actions_in_observation_only_for_active_player(self):
        _, s = _dealt_state(seed=6)
        cp = s.current_player()
        obs_active = json.loads(s.observation_string(cp))
        self.assertIn("legal_actions", obs_active)
        self.assertGreater(len(obs_active["legal_actions"]), 0)
        # Bid labels have human-readable strings like "Pass" or "1♣".
        labels = {entry["label"] for entry in obs_active["legal_actions"]}
        self.assertIn("Pass", labels)

        for other in range(4):
            if other == cp:
                continue
            obs_other = json.loads(s.observation_string(other))
            self.assertNotIn("legal_actions", obs_other)
            self.assertFalse(obs_other["your_turn"])

    def test_auction_history_uses_external_player_ids(self):
        _, s = _dealt_state(seed=7)
        # 4 passes from dealer (external 0): players act in order 0,2,1,3.
        expected_seats = [0, 2, 1, 3]
        for _ in range(4):
            s.apply_action(52)  # Pass
        obs = json.loads(s.observation_string(0))
        self.assertEqual(len(obs["auction"]), 4)
        for i, call in enumerate(obs["auction"]):
            self.assertEqual(call["player_id"], expected_seats[i])
            self.assertEqual(call["call"], "Pass")
        # Each call carries the table position too.
        self.assertEqual([c["table_position"] for c in obs["auction"]], ["N", "E", "S", "W"])


class PrivateViewIsolationTest(absltest.TestCase):
    """All per-player view methods must respect the AABB->NESW permutation.

    A bug here would leak a different seat's private hand. Verified
    against the wrapped state directly so the permutation is the only
    moving part.
    """

    _SEAT_NAME = {0: "North", 1: "East", 2: "South", 3: "West"}
    _EXT_TO_INTERNAL = {0: 0, 1: 2, 2: 1, 3: 3}

    def test_information_state_string_matches_external_seat(self):
        _, s = _dealt_state(seed=11)
        for ext in range(4):
            internal = self._EXT_TO_INTERNAL[ext]
            seat_name = self._SEAT_NAME[internal]
            info = s.information_state_string(ext)
            self.assertIn(f"You are {seat_name}", info)
            # Must NOT identify as any other seat's hand.
            for other_ext, other_internal in self._EXT_TO_INTERNAL.items():
                if other_ext == ext:
                    continue
                self.assertNotIn(f"You are {self._SEAT_NAME[other_internal]}", info)

    def test_information_state_string_passthrough_matches_wrapped(self):
        _, s = _dealt_state(seed=11)
        # External X's info_state must equal wrapped seat _PERM[X]'s info_state.
        for ext in range(4):
            internal = self._EXT_TO_INTERNAL[ext]
            self.assertEqual(
                s.information_state_string(ext),
                s.__wrapped__.information_state_string(internal),
            )

    def test_information_state_tensor_matches_wrapped(self):
        _, s = _dealt_state(seed=11)
        for ext in range(4):
            internal = self._EXT_TO_INTERNAL[ext]
            self.assertEqual(
                list(s.information_state_tensor(ext)),
                list(s.__wrapped__.information_state_tensor(internal)),
            )

    def test_observation_tensor_matches_wrapped(self):
        _, s = _dealt_state(seed=11)
        for ext in range(4):
            internal = self._EXT_TO_INTERNAL[ext]
            self.assertEqual(
                list(s.observation_tensor(ext)),
                list(s.__wrapped__.observation_tensor(internal)),
            )


def _bid(level, denom_idx):
    """Encode a bid: action = 55 + (level-1)*5 + denom_idx (denom 0..4 = ♣♦♥♠NT)."""
    return 55 + (level - 1) * 5 + denom_idx


_PASS, _DBL, _RDBL = 52, 53, 54


class PlaySeatAttributionTest(absltest.TestCase):
    """``_plays()`` attributes each card to a seat using contract +
    trick-winner logic. Tested against ``use_double_dummy_result=False``
    so actual card play happens."""

    def _play_to_terminal(self, seed=21):
        _, s = _dealt_state(seed=seed, use_double_dummy_result=False)
        # Force a deterministic auction: North opens 3NT, all pass.
        # Sequence in external pids: 0(N)->3NT, 2(E)->Pass, 1(S)->Pass, 3(W)->Pass.
        s.apply_action(_bid(3, 4))  # 3NT
        s.apply_action(_PASS)
        s.apply_action(_PASS)
        s.apply_action(_PASS)
        # Phase should now be "play".
        obs = json.loads(s.observation_string(0))
        self.assertEqual(obs["phase"], "play")
        # Drive cards to terminal with random legal plays.
        rng = random.Random(seed)
        while not s.is_terminal():
            legal = s.legal_actions()
            s.apply_action(rng.choice(legal))
        return s

    def test_plays_have_seat_attribution_for_every_card(self):
        s = self._play_to_terminal(seed=21)
        obs = json.loads(s.observation_string(0))
        plays = obs["plays"]
        self.assertEqual(len(plays), 52)  # full deal, 13 tricks, 52 cards
        for entry in plays:
            self.assertIn("internal_seat", entry)
            self.assertIn("table_position", entry)
            self.assertIn("player_id", entry)
            self.assertIn("trick_number", entry)
            self.assertIn(entry["table_position"], ("N", "E", "S", "W"))
            self.assertIn(entry["player_id"], (0, 1, 2, 3))

    def test_opening_lead_is_to_left_of_declarer(self):
        # Contract is 3NT by North (external 0) -- opening leader is
        # East (internal 1, external 2), the seat to declarer's left.
        s = self._play_to_terminal(seed=21)
        obs = json.loads(s.observation_string(0))
        self.assertEqual(obs["contract"]["declarer_table_position"], "N")
        self.assertEqual(obs["contract"]["declarer_player_id"], 0)
        self.assertEqual(obs["plays"][0]["table_position"], "E")
        self.assertEqual(obs["plays"][0]["player_id"], 2)

    def test_every_seat_plays_thirteen_cards(self):
        s = self._play_to_terminal(seed=22)
        obs = json.loads(s.observation_string(0))
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for entry in obs["plays"]:
            counts[entry["player_id"]] += 1
        self.assertEqual(counts, {0: 13, 1: 13, 2: 13, 3: 13})

    def test_within_trick_seats_rotate_clockwise(self):
        s = self._play_to_terminal(seed=21)
        obs = json.loads(s.observation_string(0))
        clockwise = {0: 1, 1: 2, 2: 3, 3: 0}  # internal seat order N->E->S->W->N
        for trick_idx in range(13):
            trick = obs["plays"][trick_idx * 4 : (trick_idx + 1) * 4]
            for i in range(1, 4):
                prev_seat = trick[i - 1]["internal_seat"]
                self.assertEqual(trick[i]["internal_seat"], clockwise[prev_seat])

    def test_play_phase_returns_share_within_team(self):
        # With real play, both partners still get the same final reward.
        s = self._play_to_terminal(seed=23)
        ret = s.returns()
        self.assertEqual(ret[0], ret[1])
        self.assertEqual(ret[2], ret[3])
        self.assertAlmostEqual(ret[0] + ret[2], 0.0)

    def test_contract_decoded_correctly(self):
        s = self._play_to_terminal(seed=21)
        obs = json.loads(s.observation_string(0))
        self.assertEqual(obs["contract"]["level"], 3)
        self.assertEqual(obs["contract"]["denom"], "NT")
        self.assertIsNone(obs["contract"]["trump_suit"])


class TrickWinnerTest(absltest.TestCase):
    """Unit tests for the trick-winning helper used by ``_plays()``."""

    def test_highest_of_led_suit_wins_when_no_trump(self):
        # Trump=None (NT). Led suit = ♠.
        trick = ["♠5", "♠K", "♠2", "♥A"]  # ♥A is not led suit; ♠K wins.
        self.assertEqual(bridge_arena_game._trick_winner_idx(trick, None), 1)

    def test_trump_beats_higher_card_of_led_suit(self):
        # Trump=♥. Led=♠. Lowly ♥2 still beats the ♠A.
        trick = ["♠A", "♥2", "♠K", "♠Q"]
        self.assertEqual(bridge_arena_game._trick_winner_idx(trick, "♥"), 1)

    def test_higher_trump_wins(self):
        trick = ["♠A", "♥2", "♥K", "♠3"]
        self.assertEqual(bridge_arena_game._trick_winner_idx(trick, "♥"), 2)

    def test_ten_handled_as_T(self):
        # OpenSpiel uses "10" not "T" for tens; the parser normalizes
        # so trick winner logic still treats "10" between 9 and J.
        trick = ["♠9", "♠10", "♠J", "♠2"]
        self.assertEqual(bridge_arena_game._trick_winner_idx(trick, None), 2)
        trick2 = ["♠9", "♠10", "♠2", "♠5"]
        self.assertEqual(bridge_arena_game._trick_winner_idx(trick2, None), 1)

    def test_discards_cannot_win(self):
        # Trump=♠, led=♥. ♣A and ♦K are discards; ♥3 (only led-suit card)
        # wins because no trumps were played.
        trick = ["♥3", "♣A", "♦K", "♥2"]
        self.assertEqual(bridge_arena_game._trick_winner_idx(trick, "♠"), 0)


class ReturnsTest(absltest.TestCase):
    """Bridge scoring: partners share returns; permutation maps correctly."""

    def test_passout_returns_are_zero(self):
        _, s = _dealt_state(seed=8)
        for _ in range(4):
            s.apply_action(52)  # Pass -- whole table passes, no contract
        self.assertTrue(s.is_terminal())
        self.assertEqual(s.returns(), [0.0, 0.0, 0.0, 0.0])

    def test_partners_share_returns_via_perm(self):
        # Play a full random game and verify our returns() is a
        # permutation of wrapped returns aligning external pids 0/1 with
        # wrapped seats N/S, and external 2/3 with E/W.
        _, s = _dealt_state(seed=9)
        rng = random.Random(9)
        while not s.is_terminal():
            legal = s.legal_actions()
            s.apply_action(rng.choice(legal))
        wrapped_returns = s.__wrapped__.returns()
        external_returns = s.returns()
        self.assertEqual(external_returns[0], wrapped_returns[0])  # ext 0 = N
        self.assertEqual(external_returns[1], wrapped_returns[2])  # ext 1 = S
        self.assertEqual(external_returns[2], wrapped_returns[1])  # ext 2 = E
        self.assertEqual(external_returns[3], wrapped_returns[3])  # ext 3 = W
        # Partners on each team always share the same bridge return.
        self.assertEqual(external_returns[0], external_returns[1])
        self.assertEqual(external_returns[2], external_returns[3])


class EnvTest(absltest.TestCase):
    """End-to-end via the kaggle_environments runner."""

    def test_bridge_arena_agent_playthrough(self):
        env = make(
            "open_spiel_bridge_arena",
            configuration={"includeLegalActions": True},
            debug=True,
        )
        env.run(["random", "random", "random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["name"], "open_spiel_bridge_arena")
        self.assertEqual(playthrough["statuses"], ["DONE"] * 4)
        # Partners must share the same final reward (bridge is partnership-scored).
        rewards = playthrough["rewards"]
        self.assertEqual(rewards[0], rewards[1])
        self.assertEqual(rewards[2], rewards[3])
        # Zero-sum across teams (within float rounding).
        self.assertAlmostEqual(rewards[0] + rewards[2], 0.0)

    def test_bridge_arena_invalid_action(self):
        env = make("open_spiel_bridge_arena", debug=True)
        env.reset()
        # The framework auto-resolves chance, so the first non-setup step
        # has a real bridge actor (external 0 = dealer = N by default).
        env.step([{"submission": -1}] * 4)  # initial setup
        env.step(
            [
                {"submission": 999},  # invalid for player 0
                {"submission": -1},
                {"submission": -1},
                {"submission": -1},
            ]
        )
        self.assertTrue(env.done)
        playthrough = env.toJSON()
        self.assertEqual(playthrough["rewards"][0], open_spiel_env.DEFAULT_INVALID_ACTION_REWARD)

    def test_bridge_arena_information_state_mode(self):
        # The framework uses ``information_state_string`` instead of
        # ``observation_string`` when ``observationType == "information_state"``.
        # This exercises the BridgeArenaState override end-to-end and
        # asserts no AABB->NESW leak through that code path.
        env = make(
            "open_spiel_bridge_arena",
            configuration={
                "includeLegalActions": True,
                "observationType": "information_state",
            },
            debug=True,
        )
        env.run(["random", "random", "random", "random"])
        playthrough = env.toJSON()
        self.assertEqual(playthrough["statuses"], ["DONE"] * 4)
        rewards = playthrough["rewards"]
        self.assertEqual(rewards[0], rewards[1])
        self.assertEqual(rewards[2], rewards[3])
        # Spot-check one step's observation strings: each player must see
        # the seat name that matches their external pid's table position.
        # (Skip step 0 -- the framework setup step has no observation.)
        seat_names = {0: "North", 1: "South", 2: "East", 3: "West"}
        first_real_step = None
        for step in playthrough["steps"][1:]:
            if step[0].get("observation", {}).get("observationString"):
                first_real_step = step
                break
        self.assertIsNotNone(first_real_step)
        for ext, seat_name in seat_names.items():
            obs_str = first_real_step[ext]["observation"]["observationString"]
            self.assertIn(f"You are {seat_name}", obs_str)


class ObserverPatternTest(absltest.TestCase):
    """The standard OpenSpiel observer (game.make_py_observer) must also
    apply the AABB->NESW permutation. Without the BridgeArenaGame
    override of ``make_py_observer``, the shared ``proxy._Observation``
    forwards external player ids straight to the wrapped state.
    """

    _SEAT_NAME = {0: "North", 1: "East", 2: "South", 3: "West"}
    _EXT_TO_INTERNAL = {0: 0, 1: 2, 2: 1, 3: 3}

    def test_string_from_uses_permuted_seat(self):
        g = pyspiel.load_game("bridge_arena")
        s = _dealt_state(seed=33)[1]
        observer = g.make_py_observer()
        for ext in range(4):
            internal = self._EXT_TO_INTERNAL[ext]
            view = observer.string_from(s, ext)
            self.assertIn(f"You are {self._SEAT_NAME[internal]}", view)
            for other_ext, other_internal in self._EXT_TO_INTERNAL.items():
                if other_ext == ext:
                    continue
                self.assertNotIn(f"You are {self._SEAT_NAME[other_internal]}", view)

    def test_set_from_matches_wrapped_internal_seat(self):
        from kaggle_environments.envs.open_spiel_env import observation as obs_module

        g = pyspiel.load_game("bridge_arena")
        s = _dealt_state(seed=33)[1]
        observer = g.make_py_observer()
        wrapped_observer = obs_module.make_observation(s.__wrapped__.get_game())
        for ext in range(4):
            internal = self._EXT_TO_INTERNAL[ext]
            observer.set_from(s, ext)
            wrapped_observer.set_from(s.__wrapped__, internal)
            self.assertEqual(list(observer.tensor), list(wrapped_observer.tensor))


class SuitContractPlayTest(absltest.TestCase):
    """Verify ``_plays()`` seat attribution against a SUIT contract,
    not just NT. With trump play, the trick winner is computed via the
    trump branch of ``_trick_winner_idx`` and the next leader follows
    the winning seat -- so this exercises code paths NT contracts skip.
    """

    def _play_to_terminal_with_contract(self, dealer_internal, bid_action, seed):
        # Construct a game with the given dealer and play out: dealer
        # opens with the chosen bid, then 3 passes complete the auction.
        _, s = _dealt_state(seed=seed, dealer=dealer_internal, use_double_dummy_result=False)
        s.apply_action(bid_action)
        s.apply_action(52)  # Pass
        s.apply_action(52)  # Pass
        s.apply_action(52)  # Pass
        rng = random.Random(seed)
        while not s.is_terminal():
            s.apply_action(rng.choice(s.legal_actions()))
        return s

    def test_4_spades_by_north_seat_attribution(self):
        # 4♠ by N: trump=♠. Opening leader = E (external 2).
        s = self._play_to_terminal_with_contract(
            dealer_internal=0, bid_action=_bid(4, 3), seed=41
        )
        obs = json.loads(s.observation_string(0))
        self.assertEqual(obs["contract"]["denom"], "♠")
        self.assertEqual(obs["contract"]["trump_suit"], "♠")
        self.assertEqual(obs["contract"]["declarer_table_position"], "N")
        # Opening lead from East (external 2).
        self.assertEqual(obs["plays"][0]["table_position"], "E")
        self.assertEqual(obs["plays"][0]["player_id"], 2)
        # Every seat plays 13 cards.
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for entry in obs["plays"]:
            counts[entry["player_id"]] += 1
        self.assertEqual(counts, {0: 13, 1: 13, 2: 13, 3: 13})

    def test_across_trick_winner_leads_next_for_trump_contract(self):
        # For each completed trick, verify the FIRST card of the next
        # trick was played by the seat we compute as that trick's winner.
        s = self._play_to_terminal_with_contract(
            dealer_internal=0, bid_action=_bid(4, 3), seed=42
        )
        obs = json.loads(s.observation_string(0))
        plays = obs["plays"]
        trump = obs["contract"]["trump_suit"]
        for trick_idx in range(12):  # 12 transitions between 13 tricks
            trick = plays[trick_idx * 4 : (trick_idx + 1) * 4]
            trick_cards = [entry["card"] for entry in trick]
            trick_seats = [entry["internal_seat"] for entry in trick]
            winner_idx = bridge_arena_game._trick_winner_idx(trick_cards, trump)
            expected_next_leader = trick_seats[winner_idx]
            next_first_play = plays[(trick_idx + 1) * 4]
            self.assertEqual(
                next_first_play["internal_seat"],
                expected_next_leader,
                msg=f"trick {trick_idx + 1}: winner seat {expected_next_leader} should lead trick {trick_idx + 2}",
            )


class OpeningLeadByDeclarerSeatTest(absltest.TestCase):
    """``(declarer + 1) % 4`` opening lead formula must work for every
    declarer position, not just North."""

    # (dealer_internal, bid_action, expected_declarer_pos, expected_opener_pos, expected_opener_ext_pid)
    _CASES = [
        # Dealer N opens 3NT, all pass -> declarer N, opener E (ext 2).
        (0, _bid(3, 4), "N", "E", 2),
        # Dealer E opens 3NT, all pass -> declarer E, opener S (ext 1).
        (1, _bid(3, 4), "E", "S", 1),
        # Dealer S opens 3NT, all pass -> declarer S, opener W (ext 3).
        (2, _bid(3, 4), "S", "W", 3),
        # Dealer W opens 3NT, all pass -> declarer W, opener N (ext 0).
        (3, _bid(3, 4), "W", "N", 0),
    ]

    def test_opener_for_every_declarer_seat(self):
        for dealer, bid_action, decl_pos, opener_pos, opener_ext in self._CASES:
            with self.subTest(dealer=dealer, declarer=decl_pos):
                _, s = _dealt_state(seed=51, dealer=dealer, use_double_dummy_result=False)
                s.apply_action(bid_action)
                s.apply_action(52)
                s.apply_action(52)
                s.apply_action(52)
                obs = json.loads(s.observation_string(0))
                self.assertEqual(obs["contract"]["declarer_table_position"], decl_pos)
                self.assertEqual(obs["plays"], [])  # phase=="play" but no card yet
                self.assertEqual(obs["phase"], "play")
                # Advance one card to verify the opening lead seat.
                s.apply_action(s.legal_actions()[0])
                obs = json.loads(s.observation_string(0))
                self.assertEqual(obs["plays"][0]["table_position"], opener_pos)
                self.assertEqual(obs["plays"][0]["player_id"], opener_ext)


class ContractDuringAuctionTest(absltest.TestCase):
    """``contract`` is None until the auction yields one."""

    def test_contract_is_none_before_any_bid(self):
        _, s = _dealt_state(seed=61)
        obs = json.loads(s.observation_string(0))
        self.assertIsNone(obs["contract"])

    def test_contract_is_none_for_passout(self):
        _, s = _dealt_state(seed=62)
        for _ in range(4):
            s.apply_action(52)
        self.assertTrue(s.is_terminal())
        obs = json.loads(s.observation_string(0))
        self.assertIsNone(obs["contract"])


if __name__ == "__main__":
    absltest.main()
