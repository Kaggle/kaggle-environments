"""Tests for the Crazyhouse LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.crazyhouse import crazyhouse_proxy
from kaggle_environments.envs.open_spiel_env.games.crazyhouse.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _play(state, san):
    for action in state.legal_actions():
        if state.action_to_string(state.current_player(), action) == san:
            state.apply_action(action)
            return
    raise AssertionError(f"move {san!r} not legal here")


def _make_observation(state, game, player_id=None):
    """Build a harness-style observation dict from a Crazyhouse state."""
    pid = state.current_player() if player_id is None else player_id
    return {
        "observationString": state.observation_string(0),
        "playerId": pid,
        "currentPlayer": state.current_player(),
        "serializedGameAndState": pyspiel.serialize_game_and_state(
            game.__wrapped__, state.__wrapped__
        ),
    }


class ParseResponseTest(absltest.TestCase):
    def test_parse_pawn_push_from_json(self):
        legal = ["e4", "d4", "Nf3", "Nc3"]
        result = parse_response('```json\n{"move": "e4"}\n```', legal)
        self.assertEqual(result.legal_action, "e4")
        self.assertEqual(result.raw_action, "e4")

    def test_parse_piece_move(self):
        legal = ["e4", "Nf3", "Nc3"]
        result = parse_response('```json\n{"move": "Nf3"}\n```', legal)
        self.assertEqual(result.legal_action, "Nf3")

    def test_parse_drop(self):
        legal = ["e4", "P@e4", "P@d4", "N@d5"]
        result = parse_response('```json\n{"move": "P@e4"}\n```', legal)
        self.assertEqual(result.legal_action, "P@e4")

    def test_parse_drop_lowercase_piece(self):
        """Models often emit ``p@e4`` — match it to the canonical ``P@e4``."""
        legal = ["P@e4", "P@d4"]
        result = parse_response('```json\n{"move": "p@e4"}\n```', legal)
        self.assertEqual(result.legal_action, "P@e4")

    def test_parse_castling(self):
        legal = ["O-O", "O-O-O", "Nf3"]
        result = parse_response('```json\n{"move": "O-O"}\n```', legal)
        self.assertEqual(result.legal_action, "O-O")

    def test_parse_castling_zero_form(self):
        """Some clients write 0-0 for castling — accept it."""
        legal = ["O-O", "Nf3"]
        result = parse_response('```json\n{"move": "0-0"}\n```', legal)
        self.assertEqual(result.legal_action, "O-O")

    def test_parse_capture_with_check(self):
        legal = ["e4", "Bb5+", "Nf3"]
        result = parse_response('```json\n{"move": "Bb5+"}\n```', legal)
        self.assertEqual(result.legal_action, "Bb5+")

    def test_parse_drops_check_suffix_when_unique(self):
        """If only one legal move strips to ``Bb5``, accept ``Bb5`` for it."""
        legal = ["e4", "Bb5+"]
        result = parse_response('```json\n{"move": "Bb5"}\n```', legal)
        self.assertEqual(result.legal_action, "Bb5+")

    def test_parse_case_insensitive_pawn(self):
        legal = ["e4", "Nf3"]
        result = parse_response('```json\n{"move": "E4"}\n```', legal)
        self.assertEqual(result.legal_action, "e4")

    def test_parse_bare_json(self):
        legal = ["e4", "Nf3"]
        result = parse_response('I think {"move": "Nf3"} is best.', legal)
        self.assertEqual(result.legal_action, "Nf3")
        self.assertEqual(result.raw_action, "Nf3")

    def test_parse_fallback_scan(self):
        """Falls back to scanning prose for any legal SAN token."""
        legal = ["e4", "d4", "Nc3"]
        result = parse_response("In this position I think Nc3 wins material.", legal)
        self.assertEqual(result.legal_action, "Nc3")

    def test_parse_malformed_json_falls_back(self):
        legal = ["e4", "Nf3"]
        result = parse_response("```json\n{bad json}\n```\nI play e4.", legal)
        self.assertEqual(result.legal_action, "e4")

    def test_parse_illegal_returns_none(self):
        legal = ["e4", "d4"]
        result = parse_response('```json\n{"move": "Ke2"}\n```', legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "Ke2")

    def test_parse_no_match_returns_none(self):
        legal = ["e4", "d4"]
        result = parse_response("I have no idea what to play.", legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_returns_parse_result(self):
        result = parse_response('```json\n{"move": "e4"}\n```', ["e4"])
        self.assertIsInstance(result, ParseResult)


class GeneratePromptTest(absltest.TestCase):
    def test_basic_prompt(self):
        observation = {
            "observationString": '{"pockets": {"white": {}, "black": {}}}',
            "playerId": 1,
        }
        prompt = generate_prompt(observation, [])
        self.assertIn("Crazyhouse", prompt)
        self.assertIn("White", prompt)
        self.assertIn("(W)", prompt)
        self.assertIn("pocket", prompt.lower())
        self.assertIn("drop", prompt.lower())

    def test_black_player(self):
        observation = {
            "observationString": '{"pockets": {"white": {}, "black": {}}}',
            "playerId": 0,
        }
        prompt = generate_prompt(observation, [])
        self.assertIn("Black", prompt)
        self.assertIn("(B)", prompt)

    def test_pocket_rendered(self):
        observation = {
            "observationString": (
                '{"pockets": {"white": {"P": 2, "N": 1}, "black": {"R": 1}}}'
            ),
            "playerId": 1,
        }
        prompt = generate_prompt(observation, [])
        self.assertIn("Your pocket: 1xN, 2xP", prompt)
        self.assertIn("Opponent pocket: 1xR", prompt)

    def test_empty_pocket_marker(self):
        observation = {
            "observationString": '{"pockets": {"white": {}, "black": {}}}',
            "playerId": 1,
        }
        prompt = generate_prompt(observation, [])
        self.assertIn("Your pocket: (empty)", prompt)
        self.assertIn("Opponent pocket: (empty)", prompt)

    def test_move_history_included(self):
        observation = {
            "observationString": "{}",
            "playerId": 1,
        }
        prompt = generate_prompt(observation, ["e4", "d5", "exd5"])
        self.assertIn("e4 d5 exd5", prompt)

    def test_rethink_suffix(self):
        observation = {
            "observationString": "{}",
            "playerId": 1,
        }
        prompt = generate_prompt(
            observation,
            [],
            previous_response="I want Ke2",
            previous_action="Ke2",
        )
        self.assertIn("Your previous response was", prompt)
        self.assertIn("Ke2", prompt)
        self.assertIn("not in the legal moves", prompt)

    def test_no_rethink_on_first_attempt(self):
        observation = {
            "observationString": "{}",
            "playerId": 1,
        }
        prompt = generate_prompt(observation, [])
        self.assertNotIn("Your previous response was", prompt)


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        observation = {
            "legalActions": [89, 656],
            "legalActionStrings": ["a3", "Nc3"],
        }
        result = get_legal_moves(observation)
        self.assertEqual(result, {89: "a3", 656: "Nc3"})

    def test_from_serialized_state(self):
        game = crazyhouse_proxy.CrazyhouseGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)
        result = get_legal_moves(observation)
        # Standard chess opening: 20 legal moves (16 pawn pushes + 4 knight).
        self.assertEqual(len(result), 20)
        self.assertIn("e4", result.values())
        self.assertIn("Nf3", result.values())

    def test_drops_appear_after_capture(self):
        """After a capture the side-to-move can drop the captured piece."""
        game = crazyhouse_proxy.CrazyhouseGame()
        state = game.new_initial_state()
        for san in ["e4", "d5", "exd5", "Qxd5"]:
            _play(state, san)
        observation = _make_observation(state, game)
        result = get_legal_moves(observation)
        drops = [s for s in result.values() if "@" in s]
        self.assertTrue(drops, "expected at least one drop available")
        self.assertIn("P@e4", drops)

    def test_empty_serialized(self):
        observation = {"serializedGameAndState": ""}
        result = get_legal_moves(observation)
        self.assertEqual(result, {})

    def test_returns_int_string_dict(self):
        observation = {
            "legalActions": [89],
            "legalActionStrings": ["a3"],
        }
        result = get_legal_moves(observation)
        self.assertIsInstance(result, dict)
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, str)


def _make_mock_response(content):
    resp = MagicMock()
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
    resp.choices = [
        MagicMock(message=MagicMock(content=content), finish_reason="stop")
    ]
    return resp


class _CrazyhouseHarness:
    """Adapter wrapping module-level functions into the GameHarness protocol."""

    def get_legal_moves(self, observation):
        return get_legal_moves(observation)

    def make_prompt(self, observation, move_history, previous_response=None, previous_action=None):
        return generate_prompt(observation, move_history, previous_response, previous_action)

    def parse_response(self, response, legal_action_strings):
        return parse_response(response, legal_action_strings)


class AgentIntegrationTest(absltest.TestCase):
    """Drive the harness through ``create_agent_fn`` from ``core_harness``."""

    @patch.dict(
        "os.environ",
        {
            "MODEL_NAME": "test-model",
            "MODEL_PROXY_KEY": "test-key",
            "MODEL_PROXY_URL": "dummy_url",
        },
    )
    @patch("kaggle_environments.core_harness.litellm")
    def test_setup_step_returns_inactive(self, mock_litellm):
        mock_litellm.drop_params = True
        agent = create_agent_fn(_CrazyhouseHarness())

        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict(
        "os.environ",
        {
            "MODEL_NAME": "test-model",
            "MODEL_PROXY_KEY": "test-key",
            "MODEL_PROXY_URL": "dummy_url",
        },
    )
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            '```json\n{"move": "e4"}\n```',
        )

        agent = create_agent_fn(_CrazyhouseHarness())

        game = crazyhouse_proxy.CrazyhouseGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {})

        # action id 92 == "e4" in OpenSpiel's chess action encoding.
        e4_action = next(
            a for a in state.legal_actions()
            if state.action_to_string(state.current_player(), a) == "e4"
        )
        self.assertEqual(result["submission"], e4_action)
        self.assertEqual(result["actionString"], "e4")
        self.assertEqual(result["status"], "OK")

    @patch.dict(
        "os.environ",
        {
            "MODEL_NAME": "test-model",
            "MODEL_PROXY_KEY": "test-key",
            "MODEL_PROXY_URL": "dummy_url",
        },
    )
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "Ke2"}\n```'),
            _make_mock_response('```json\n{"move": "e4"}\n```'),
        ]

        agent = create_agent_fn(_CrazyhouseHarness())

        game = crazyhouse_proxy.CrazyhouseGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {})

        e4_action = next(
            a for a in state.legal_actions()
            if state.action_to_string(state.current_player(), a) == "e4"
        )
        self.assertEqual(result["submission"], e4_action)
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict(
        "os.environ",
        {
            "MODEL_NAME": "test-model",
            "MODEL_PROXY_KEY": "test-key",
            "MODEL_PROXY_URL": "dummy_url",
        },
    )
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            "I don't know what to play",
        )

        agent = create_agent_fn(_CrazyhouseHarness())

        game = crazyhouse_proxy.CrazyhouseGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        with self.assertRaises(ValueError):
            agent(observation, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)


if __name__ == "__main__":
    absltest.main()
