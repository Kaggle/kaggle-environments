"""Tests for the Oshi-Zumo LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.oshi_zumo import (
    oshi_zumo_proxy,
)
from kaggle_environments.envs.open_spiel_env.games.oshi_zumo.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(
    state: oshi_zumo_proxy.OshiZumoState,
    game: oshi_zumo_proxy.OshiZumoGame,
    player_id: int = 0,
) -> dict:
    """Build a harness-style observation dict from a proxy state."""
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        # Sim-move games surface this as PlayerId.SIMULTANEOUS == -2.
        "currentPlayer": int(state.current_player()),
        "isTerminal": state.is_terminal(),
        "legalActions": list(state.legal_actions(player_id)),
        "legalActionStrings": [
            state.action_to_string(player_id, a)
            for a in state.legal_actions(player_id)
        ],
        "serializedGameAndState": pyspiel.serialize_game_and_state(
            game.__wrapped__, state.__wrapped__
        ),
    }


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class ParseResponseTest(absltest.TestCase):
    legal = ["[P0]Bid: 0", "[P0]Bid: 1", "[P0]Bid: 2", "[P0]Bid: 3"]

    def test_parse_json_block(self):
        result = parse_response('```json\n{"bid": 2}\n```', self.legal)
        self.assertEqual(result.legal_action, "[P0]Bid: 2")
        self.assertEqual(result.raw_action, "2")

    def test_parse_bare_json(self):
        result = parse_response('I think {"bid": 1} is best.', self.legal)
        self.assertEqual(result.legal_action, "[P0]Bid: 1")

    def test_parse_action_string_in_response(self):
        result = parse_response(
            "I will play [P0]Bid: 3 this round.", self.legal
        )
        self.assertEqual(result.legal_action, "[P0]Bid: 3")

    def test_parse_fallback_integer_in_text(self):
        result = parse_response("After thinking I bid 2.", self.legal)
        self.assertEqual(result.legal_action, "[P0]Bid: 2")

    def test_parse_illegal_bid_returns_raw(self):
        result = parse_response('```json\n{"bid": 99}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "99")

    def test_parse_no_match_returns_none(self):
        result = parse_response("I have no idea.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_returns_parse_result_type(self):
        result = parse_response('```json\n{"bid": 0}\n```', self.legal)
        self.assertIsInstance(result, ParseResult)

    def test_parse_prefers_largest_legal_match(self):
        """A response containing '12' should NOT pick '1' or '2' — but if
        only single-digit bids are legal, it must pick the largest match."""
        legal = ["[P0]Bid: 1", "[P0]Bid: 2"]
        result = parse_response("after deliberation I'll bid 2.", legal)
        self.assertEqual(result.legal_action, "[P0]Bid: 2")
        # The substring '12' inside a larger number should not match either bid.
        result2 = parse_response("the score was 121-99.", legal)
        self.assertIsNone(result2.legal_action)


# ---------------------------------------------------------------------------
# generate_prompt
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game = oshi_zumo_proxy.OshiZumoGame(
            {"coins": 12, "size": 3, "horizon": 50}
        )
        self.state = self.game.new_initial_state()

    def test_basic_prompt_contents(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Oshi-Zumo", prompt)
        self.assertIn("simultaneously", prompt.lower())
        self.assertIn("Player 0", prompt)
        self.assertIn("Your coins:", prompt)
        # Initial state is centered with all 12 coins each.
        self.assertIn("12", prompt)

    def test_legal_bids_listed(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Initial bids 0..12 are all legal.
        for n in range(13):
            self.assertIn(str(n), prompt)

    def test_player_label_swap(self):
        obs0 = _make_observation(self.state, self.game, player_id=0)
        obs1 = _make_observation(self.state, self.game, player_id=1)
        self.assertIn("Player 0", generate_prompt(obs0, []))
        self.assertIn("Player 1", generate_prompt(obs1, []))

    def test_coins_swap_for_player(self):
        # Apply asymmetric bids: P0 bids 4, P1 bids 2.
        self.state.apply_actions([4, 2])
        obs0 = _make_observation(self.state, self.game, player_id=0)
        obs1 = _make_observation(self.state, self.game, player_id=1)
        # Player 0 now has 8 coins, opponent has 10. (12 - bid each.)
        p0_prompt = generate_prompt(obs0, [])
        p1_prompt = generate_prompt(obs1, [])
        self.assertIn("Your coins:        8", p0_prompt)
        self.assertIn("Opponent coins:    10", p0_prompt)
        self.assertIn("Your coins:        10", p1_prompt)
        self.assertIn("Opponent coins:    8", p1_prompt)

    def test_my_history_rendered(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, ["[P0]Bid: 4", "[P0]Bid: 2"])
        self.assertIn("Your past bids:        4, 2", prompt)

    def test_no_history_fallback(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("(none yet)", prompt)

    def test_rethink_suffix(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(
            obs, [], previous_response="I'll bid 99", previous_action="99"
        )
        self.assertIn("Your previous response was", prompt)
        self.assertIn("99", prompt)
        self.assertIn("NOT in the legal bid list", prompt)


# ---------------------------------------------------------------------------
# get_legal_moves
# ---------------------------------------------------------------------------


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        obs = {
            "legalActions": [0, 1, 2],
            "legalActionStrings": ["[P0]Bid: 0", "[P0]Bid: 1", "[P0]Bid: 2"],
        }
        result = get_legal_moves(obs)
        self.assertEqual(
            result,
            {0: "[P0]Bid: 0", 1: "[P0]Bid: 1", 2: "[P0]Bid: 2"},
        )

    def test_from_serialized_state(self):
        game = oshi_zumo_proxy.OshiZumoGame(
            {"coins": 5, "size": 1, "horizon": 20}
        )
        state = game.new_initial_state()
        obs = {
            "playerId": 0,
            "serializedGameAndState": pyspiel.serialize_game_and_state(
                game.__wrapped__, state.__wrapped__
            ),
        }
        result = get_legal_moves(obs)
        # Legal bids 0..5 = 6 actions.
        self.assertEqual(len(result), 6)
        self.assertEqual(result[0], "[P0]Bid: 0")
        self.assertEqual(result[5], "[P0]Bid: 5")

    def test_empty_serialized(self):
        self.assertEqual(get_legal_moves({"serializedGameAndState": ""}), {})

    def test_returns_typed_dict(self):
        result = get_legal_moves({
            "legalActions": [0, 5],
            "legalActionStrings": ["[P0]Bid: 0", "[P0]Bid: 5"],
        })
        self.assertIsInstance(result, dict)
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, str)


# ---------------------------------------------------------------------------
# create_agent_fn integration
# ---------------------------------------------------------------------------


class _OshiZumoHarness:
    """Adapter wrapping module-level functions into the GameHarness protocol."""

    def get_legal_moves(self, observation):
        return get_legal_moves(observation)

    def make_prompt(
        self, observation, move_history,
        previous_response=None, previous_action=None,
    ):
        return generate_prompt(
            observation, move_history,
            previous_response=previous_response,
            previous_action=previous_action,
        )

    def parse_response(self, response, legal_action_strings):
        return parse_response(response, legal_action_strings)


def _make_mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
    resp.choices = [
        MagicMock(message=MagicMock(content=content), finish_reason="stop")
    ]
    return resp


_ENV = {
    "MODEL_NAME": "test-model",
    "MODEL_PROXY_KEY": "test-key",
    "MODEL_PROXY_URL": "dummy_url",
}


class AgentIntegrationTest(absltest.TestCase):
    """Run the harness through ``create_agent_fn`` from ``core_harness``."""

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_setup_step_returns_inactive(self, mock_litellm):
        mock_litellm.drop_params = True
        agent = create_agent_fn(_OshiZumoHarness())

        # Empty obs (no playerId / currentPlayer): treated as inactive probe.
        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_sim_move_player_treated_as_active(self, mock_litellm):
        """For sim-move games currentPlayer is -2 (SIMULTANEOUS) — both
        players' agents must run, not return INACTIVE."""
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            '```json\n{"bid": 3}\n```'
        )
        agent = create_agent_fn(_OshiZumoHarness())

        game = oshi_zumo_proxy.OshiZumoGame(
            {"coins": 5, "size": 1, "horizon": 10}
        )
        state = game.new_initial_state()
        # Verify the sim-move signal we depend on is what we expect.
        self.assertEqual(int(state.current_player()), -2)

        obs = _make_observation(state, game, player_id=1)
        result = agent(obs, {})

        self.assertEqual(result["submission"], 3)
        self.assertEqual(result["actionString"], "[P1]Bid: 3")
        self.assertEqual(result["status"], "OK")

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_bid(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            '```json\n{"bid": 2}\n```'
        )
        agent = create_agent_fn(_OshiZumoHarness())

        game = oshi_zumo_proxy.OshiZumoGame(
            {"coins": 5, "size": 1, "horizon": 10}
        )
        state = game.new_initial_state()
        obs = _make_observation(state, game, player_id=0)

        result = agent(obs, {})

        self.assertEqual(result["submission"], 2)
        self.assertEqual(result["actionString"], "[P0]Bid: 2")
        self.assertEqual(result["status"], "OK")
        self.assertIn("thoughts", result)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"bid": 999}\n```'),
            _make_mock_response('```json\n{"bid": 1}\n```'),
        ]
        agent = create_agent_fn(_OshiZumoHarness())

        game = oshi_zumo_proxy.OshiZumoGame(
            {"coins": 5, "size": 1, "horizon": 10}
        )
        state = game.new_initial_state()
        obs = _make_observation(state, game, player_id=0)

        result = agent(obs, {})

        self.assertEqual(result["submission"], 1)
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            "I cannot decide."
        )
        agent = create_agent_fn(_OshiZumoHarness())

        game = oshi_zumo_proxy.OshiZumoGame(
            {"coins": 5, "size": 1, "horizon": 10}
        )
        state = game.new_initial_state()
        obs = _make_observation(state, game, player_id=0)

        with self.assertRaises(ValueError):
            agent(obs, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_full_two_player_game_via_agent_fns(self, mock_litellm):
        """Drive a tiny game to terminal with two scripted LLM agents,
        verifying the harness round-trips through pyspiel cleanly."""
        mock_litellm.drop_params = True

        # P0 always bids high (4), P1 always bids low (1) -> P0 should win
        # by pushing the wrestler off P1's edge.
        def fake_completion(*, model, messages, **kwargs):
            del model, kwargs
            content = messages[0]["content"]
            if "Player 0" in content:
                return _make_mock_response('```json\n{"bid": 4}\n```')
            return _make_mock_response('```json\n{"bid": 1}\n```')

        mock_litellm.completion.side_effect = fake_completion
        agent_p0 = create_agent_fn(_OshiZumoHarness())
        agent_p1 = create_agent_fn(_OshiZumoHarness())

        game = oshi_zumo_proxy.OshiZumoGame(
            {"coins": 12, "size": 1, "horizon": 20}
        )
        state = game.new_initial_state()
        rounds = 0
        while not state.is_terminal() and rounds < 20:
            obs0 = _make_observation(state, game, player_id=0)
            obs1 = _make_observation(state, game, player_id=1)
            r0 = agent_p0(obs0, {})
            r1 = agent_p1(obs1, {})
            self.assertEqual(r0["status"], "OK")
            self.assertEqual(r1["status"], "OK")
            state.apply_actions([r0["submission"], r1["submission"]])
            rounds += 1

        self.assertTrue(state.is_terminal())
        returns = state.returns()
        # P0 always outbid P1, so P0 wins (zero-sum: +1 / -1).
        self.assertEqual(returns[0], 1.0)
        self.assertEqual(returns[1], -1.0)


if __name__ == "__main__":
    absltest.main()
