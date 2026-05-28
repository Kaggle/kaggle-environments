"""Tests for the Bridge Arena LLM harness."""

import json
import random
from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.bridge_arena import (
    bridge_arena_game,  # noqa: F401 -- registers "bridge_arena" with pyspiel
)
from kaggle_environments.envs.open_spiel_env.games.bridge_arena.harness import (
    _match_action_to_legal,
    _normalize_action_string,
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _dealt_state(seed=7, **params):
    g = pyspiel.load_game("bridge_arena", params)
    s = g.new_initial_state()
    rng = random.Random(seed)
    while s.is_chance_node():
        outcomes, probs = zip(*s.chance_outcomes())
        s.apply_action(rng.choices(outcomes, weights=probs)[0])
    return s


def _make_observation(state, player_id):
    legal = state.legal_actions(player_id)
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "currentPlayer": state.current_player(),
        "isTerminal": state.is_terminal(),
        "legalActions": legal,
        "legalActionStrings": [state.action_to_string(player_id, a) for a in legal],
    }


class NormalizeActionStringTest(absltest.TestCase):
    """The normalizer accepts unicode/letter suits and common synonyms."""

    def test_glyph_and_letter_equivalent(self):
        self.assertEqual(_normalize_action_string("1♣"), _normalize_action_string("1C"))
        self.assertEqual(_normalize_action_string("4♠"), _normalize_action_string("4S"))
        self.assertEqual(_normalize_action_string("♥A"), _normalize_action_string("HA"))

    def test_double_synonyms(self):
        for raw in ("Dbl", "dbl", "double", "Double", "X", "x"):
            self.assertEqual(_normalize_action_string(raw), "dbl")
        for raw in ("RDbl", "redouble", "XX", "xx"):
            self.assertEqual(_normalize_action_string(raw), "rdbl")

    def test_pass_synonyms(self):
        self.assertEqual(_normalize_action_string("Pass"), "pass")
        self.assertEqual(_normalize_action_string("PASS"), "pass")

    def test_notrump_synonyms(self):
        target = _normalize_action_string("3NT")
        self.assertEqual(_normalize_action_string("3nt"), target)
        self.assertEqual(_normalize_action_string("3N"), target)
        self.assertEqual(_normalize_action_string("3 notrump"), target)
        self.assertEqual(_normalize_action_string("3 no trump"), target)

    def test_strips_action_prefixes(self):
        self.assertEqual(_normalize_action_string("bid 1♣"), _normalize_action_string("1c"))
        self.assertEqual(_normalize_action_string("call Pass"), "pass")
        self.assertEqual(_normalize_action_string("play ♠A"), _normalize_action_string("sa"))


class ParseResponseTest(absltest.TestCase):
    """The parser handles JSON, fallback text, and unicode/letter suits."""

    _SAMPLE_CALLS = ["Pass", "Dbl", "1♣", "1♦", "1♥", "1♠", "1NT", "4♠", "7NT"]

    def test_parse_json_call(self):
        result = parse_response('```json\n{"bid": "1♣"}\n```', self._SAMPLE_CALLS)
        self.assertEqual(result.legal_action, "1♣")

    def test_parse_json_letter_suit(self):
        result = parse_response('```json\n{"bid": "1C"}\n```', self._SAMPLE_CALLS)
        self.assertEqual(result.legal_action, "1♣")

    def test_parse_accepts_move_key(self):
        result = parse_response('```json\n{"move": "Pass"}\n```', self._SAMPLE_CALLS)
        self.assertEqual(result.legal_action, "Pass")

    def test_parse_accepts_call_key(self):
        result = parse_response('```json\n{"call": "4♠"}\n```', self._SAMPLE_CALLS)
        self.assertEqual(result.legal_action, "4♠")

    def test_parse_card_letter_suit(self):
        cards = ["♠A", "♥T", "♦2", "♣K"]
        self.assertEqual(parse_response('```json\n{"bid": "SA"}\n```', cards).legal_action, "♠A")
        self.assertEqual(parse_response('```json\n{"bid": "HT"}\n```', cards).legal_action, "♥T")
        self.assertEqual(parse_response('```json\n{"bid": "♣K"}\n```', cards).legal_action, "♣K")

    def test_parse_ten_rank_synonyms(self):
        # OpenSpiel emits "♥10" for ten-of-hearts; a model writing "H10",
        # "HT", "♥T", or "♥10" must all resolve to the same legal action.
        cards = ["♥10", "♣A"]
        for raw in ("♥10", "♥T", "HT", "H10", "h10", "hT"):
            payload = json.dumps({"bid": raw})
            result = parse_response(f"```json\n{payload}\n```", cards)
            self.assertEqual(result.legal_action, "♥10", msg=f"raw={raw!r}")

    def test_parse_ten_rank_with_glyph_legal(self):
        # If the legal label happens to be "♥T" (some other variant),
        # a model writing "H10" must still match.
        cards = ["♥T", "♣A"]
        result = parse_response('```json\n{"bid": "H10"}\n```', cards)
        self.assertEqual(result.legal_action, "♥T")

    def test_parse_double_synonyms(self):
        result = parse_response('```json\n{"bid": "double"}\n```', self._SAMPLE_CALLS)
        self.assertEqual(result.legal_action, "Dbl")
        result = parse_response('```json\n{"bid": "X"}\n```', self._SAMPLE_CALLS)
        self.assertEqual(result.legal_action, "Dbl")

    def test_parse_no_trump_synonyms(self):
        legals = ["Pass", "3NT"]
        self.assertEqual(parse_response('```json\n{"bid": "3N"}\n```', legals).legal_action, "3NT")
        self.assertEqual(parse_response('```json\n{"bid": "3 NT"}\n```', legals).legal_action, "3NT")

    def test_parse_fallback_text_scan(self):
        response = "After thinking I'll bid 1♥ to show my hand."
        result = parse_response(response, self._SAMPLE_CALLS)
        self.assertEqual(result.legal_action, "1♥")

    def test_parse_fallback_text_scan_letter(self):
        response = "I'll bid 4S to push them up."
        result = parse_response(response, self._SAMPLE_CALLS)
        self.assertEqual(result.legal_action, "4♠")

    def test_fallback_scan_prefers_last_mention(self):
        # The model considers 1♣ in reasoning but commits to Pass at the
        # end. The fallback must pick the LAST legal token in the
        # response, not the first one mentioned.
        response = (
            "I thought about opening 1♣ to show my clubs, but partner is "
            "preempted and the opponents are vulnerable. Final answer: Pass."
        )
        result = parse_response(response, self._SAMPLE_CALLS)
        self.assertEqual(result.legal_action, "Pass")

    def test_parses_against_full_auction_legal_list(self):
        # An opening auction surfaces ~36 legal calls: Pass + every bid
        # from 1♣ through 7NT. Verify token-collision resistance by
        # asserting a few specific picks against this realistic list.
        # Build the full opening legal list using a fresh dealt state.
        s = _dealt_state(seed=71)
        legal_strings = [s.action_to_string(0, a) for a in s.legal_actions(0)]
        self.assertGreaterEqual(len(legal_strings), 36)
        self.assertIn("Pass", legal_strings)
        self.assertIn("1♣", legal_strings)
        self.assertIn("3NT", legal_strings)
        self.assertIn("7NT", legal_strings)
        # Each canonical pick should round-trip cleanly.
        for target in ("Pass", "1♣", "3NT", "7NT", "4♠"):
            payload = json.dumps({"bid": target})
            result = parse_response(f"```json\n{payload}\n```", legal_strings)
            self.assertEqual(result.legal_action, target)
        # Fallback scan with reasoning that mentions multiple bids:
        # the FINAL answer at the end wins.
        response = (
            "Considering 1♣ to show clubs, or maybe 1NT to show balanced. "
            "Partner is silent so I'll stretch a bit. My final bid: 3NT."
        )
        result = parse_response(response, legal_strings)
        self.assertEqual(result.legal_action, "3NT")

    def test_fallback_scan_resists_echoed_prior_response(self):
        # After a retry the prompt echoes the previous (illegal) attempt
        # near the top of the model's next response. The parser must not
        # latch onto that echo and instead use the genuine final answer.
        response = (
            "(My previous response was: 'I'll bid 1♣'.) On reflection, "
            "I should not open with such a weak hand. My answer: Pass."
        )
        result = parse_response(response, self._SAMPLE_CALLS)
        self.assertEqual(result.legal_action, "Pass")

    def test_parse_no_match_returns_none(self):
        result = parse_response("I have absolutely no idea.", self._SAMPLE_CALLS)
        self.assertIsNone(result.legal_action)

    def test_returns_parse_result(self):
        self.assertIsInstance(
            parse_response('```json\n{"bid": "Pass"}\n```', self._SAMPLE_CALLS),
            ParseResult,
        )


class MatchActionTest(absltest.TestCase):
    def test_exact_unicode_match(self):
        self.assertEqual(_match_action_to_legal("1♣", ["1♣", "1♦"]), "1♣")

    def test_letter_to_glyph_match(self):
        self.assertEqual(_match_action_to_legal("1C", ["1♣", "1♦"]), "1♣")

    def test_no_match_returns_none(self):
        self.assertIsNone(_match_action_to_legal("garbage", ["1♣", "1♦"]))


class GeneratePromptTest(absltest.TestCase):
    """The prompt orients the LLM to its seat, partner, and the table."""

    def test_prompt_includes_seat_and_partner_for_team_a(self):
        s = _dealt_state(seed=7)
        prompt = generate_prompt(_make_observation(s, 0), [])
        self.assertIn("player 0", prompt)
        self.assertIn("seated at N", prompt)
        self.assertIn("partner is player 1", prompt)
        self.assertIn("seated at S", prompt)
        self.assertIn("team A sits N/S", prompt)

    def test_prompt_includes_seat_and_partner_for_team_b(self):
        s = _dealt_state(seed=7)
        prompt = generate_prompt(_make_observation(s, 2), [])
        self.assertIn("player 2", prompt)
        self.assertIn("seated at E", prompt)
        self.assertIn("partner is player 3", prompt)
        self.assertIn("seated at W", prompt)
        self.assertIn("team B sits E/W", prompt)

    def test_prompt_explains_2v2_and_bidding_communication(self):
        s = _dealt_state(seed=7)
        prompt = generate_prompt(_make_observation(s, 0), [])
        flat = " ".join(prompt.split())
        self.assertIn("another instance of YOU", flat)
        self.assertIn("two instances of a single different agent", flat)
        self.assertIn("NO side channel", flat)
        # Communication is via bidding/play, which the prompt must say.
        self.assertIn("through the public bidding and card play", flat)

    def test_prompt_includes_raw_observation_with_hand(self):
        s = _dealt_state(seed=7)
        prompt = generate_prompt(_make_observation(s, 0), [])
        # Player 0 is North; raw observation announces "You are North".
        self.assertIn("You are North", prompt)

    def test_prompt_lists_dealer_identity(self):
        s = _dealt_state(seed=7)
        prompt = generate_prompt(_make_observation(s, 0), [])
        # Default dealer = N = external player 0.
        self.assertIn("Dealer: player 0 (N)", prompt)

    def test_prompt_shows_action_string_format(self):
        s = _dealt_state(seed=7)
        prompt = generate_prompt(_make_observation(s, 0), [])
        self.assertIn('"bid"', prompt)
        self.assertIn("```json", prompt)
        # Mentions both letters and glyphs are accepted.
        self.assertIn("C/D/H/S", prompt)

    def test_prompt_includes_auction_after_bids(self):
        s = _dealt_state(seed=7)
        # Three passes from the dealer.
        s.apply_action(52)
        s.apply_action(52)
        s.apply_action(52)
        # Now external player 3 (W) is up.
        prompt = generate_prompt(_make_observation(s, 3), [])
        # Auction listing references each of the three callers' player ids.
        self.assertIn("player 0", prompt)
        self.assertIn("player 2", prompt)
        self.assertIn("player 1", prompt)
        self.assertIn("Pass", prompt)

    def test_rethink_suffix(self):
        s = _dealt_state(seed=7)
        prompt = generate_prompt(
            _make_observation(s, 0),
            [],
            previous_response="I'll bid 8NT",
            previous_action="8NT",
        )
        self.assertIn("Your previous response was", prompt)
        self.assertIn("8NT", prompt)

    def test_no_rethink_on_first_attempt(self):
        s = _dealt_state(seed=7)
        prompt = generate_prompt(_make_observation(s, 0), [])
        self.assertNotIn("Your previous response was", prompt)


class GetLegalMovesTest(absltest.TestCase):
    def test_active_player_gets_legal_calls(self):
        s = _dealt_state(seed=7)
        result = get_legal_moves(_make_observation(s, 0))
        self.assertGreater(len(result), 0)
        self.assertIn("Pass", result.values())

    def test_inactive_player_gets_empty(self):
        s = _dealt_state(seed=7)
        # External 1 (South) is not the dealer; not their turn.
        result = get_legal_moves(_make_observation(s, 1))
        self.assertEqual(result, {})


# --- Integration test scaffolding -----------------------------------------


class _StreamDelta:
    def __init__(self, content):
        self.content = content


class _StreamChoice:
    def __init__(self, content, finish_reason=None):
        self.delta = _StreamDelta(content)
        self.finish_reason = finish_reason


class _StreamChunk:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage


def _make_mock_response(content):
    usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        completion_tokens_details=None,
    )
    return [
        _StreamChunk([_StreamChoice(content)]),
        _StreamChunk([_StreamChoice("", finish_reason="stop")]),
        _StreamChunk([], usage=usage),
    ]


class _BridgeArenaHarness:
    def get_legal_moves(self, observation):
        return get_legal_moves(observation)

    def make_prompt(
        self,
        observation,
        move_history,
        previous_response=None,
        previous_action=None,
    ):
        return generate_prompt(
            observation,
            move_history,
            previous_response=previous_response,
            previous_action=previous_action,
        )

    def parse_response(self, response, legal_action_strings):
        return parse_response(response, legal_action_strings)


_ENV = {
    "MODEL_NAME": "test-model",
    "MODEL_PROXY_KEY": "test-key",
    "MODEL_PROXY_URL": "dummy_url",
}


class AgentIntegrationTest(absltest.TestCase):
    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_setup_step_returns_inactive(self, mock_litellm):
        mock_litellm.drop_params = True
        agent = create_agent_fn(_BridgeArenaHarness())
        result = agent({"step": 0, "remainingOverageTime": 60}, {})
        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_call(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            '```json\n{"bid": "Pass"}\n```',
        )
        agent = create_agent_fn(_BridgeArenaHarness())

        s = _dealt_state(seed=7)
        observation = _make_observation(s, 0)
        result = agent(observation, {})

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["actionString"], "Pass")
        # 52 is the Pass action id in OpenSpiel bridge.
        self.assertEqual(result["submission"], 52)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_letter_suit_input_resolves(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            '```json\n{"bid": "1C"}\n```',
        )
        agent = create_agent_fn(_BridgeArenaHarness())

        s = _dealt_state(seed=7)
        observation = _make_observation(s, 0)
        result = agent(observation, {})

        self.assertEqual(result["status"], "OK")
        # 1♣ is the lowest bid: kBiddingActionBase (52) + kFirstBid (3) = 55.
        self.assertEqual(result["submission"], 55)
        self.assertEqual(result["actionString"], "1♣")

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_then_succeed(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.side_effect = [
            _make_mock_response("I have no good options here."),
            _make_mock_response('```json\n{"bid": "Pass"}\n```'),
        ]
        agent = create_agent_fn(_BridgeArenaHarness())

        s = _dealt_state(seed=7)
        observation = _make_observation(s, 0)
        result = agent(observation, {})

        # The framework's first call returns garbage -> retry; second succeeds.
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["submission"], 52)
        self.assertGreaterEqual(mock_litellm.completion.call_count, 2)


if __name__ == "__main__":
    absltest.main()
