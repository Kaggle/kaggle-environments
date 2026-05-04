"""Tests for Go LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.go import go_proxy
from kaggle_environments.envs.open_spiel_env.games.go.harness import (
    get_legal_moves,
    generate_prompt,
    parse_response,
)


def _gtp_to_action(gtp_vertex, board_size=9):
    col_map = {c: i for i, c in enumerate("ABCDEFGHJKLMNOPQRSTUVWXYZ"[:board_size])}
    if gtp_vertex.lower() == "pass":
        return board_size * board_size
    col = col_map[gtp_vertex[0].upper()]
    row = int(gtp_vertex[1:]) - 1
    return row * board_size + col


def _make_observation(state, game, player_id=0):
    """Build a harness-style observation dict from a Go state."""
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
    }


class ParseResponseTest(absltest.TestCase):
    def test_parse_json_move(self):
        legal = ["B a1", "B b2", "B e5", "B Pass"]
        response = '```json\n{"move": "e5"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "B e5")
        self.assertEqual(result.raw_action, "e5")

    def test_parse_pass(self):
        legal = ["B a1", "B Pass"]
        response = '```json\n{"move": "PASS"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "B Pass")
        self.assertEqual(result.raw_action, "PASS")

    def test_parse_case_insensitive(self):
        legal = ["B a1", "B E5"]
        response = '```json\n{"move": "e5"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "B E5")
        self.assertEqual(result.raw_action, "e5")

    def test_parse_fallback_coordinate(self):
        """Falls back to searching for coordinates in response text."""
        legal = ["B a1", "B b2", "B e5"]
        response = "I think e5 is the best move here."
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "B e5")

    def test_parse_no_match_returns_none(self):
        legal = ["B a1", "B b2"]
        response = '```json\n{"move": "z9"}\n```'
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "z9")

    def test_parse_malformed_json_falls_back(self):
        legal = ["B a1", "B e5"]
        response = "```json\n{bad json}\n```\nI play e5."
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "B e5")

    def test_parse_no_json_no_coord_returns_none(self):
        legal = ["B a1", "B b2"]
        response = "I have no idea what to play."
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_bare_json(self):
        """Bare JSON object (no fenced block) is also parsed."""
        legal = ["B a1", "B e5"]
        response = 'I think {"move": "e5"} is best.'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "B e5")
        self.assertEqual(result.raw_action, "e5")

    def test_parse_returns_parse_result(self):
        """Verify the return type is ParseResult."""
        legal = ["B a1"]
        result = parse_response('```json\n{"move": "a1"}\n```', legal)
        self.assertIsInstance(result, ParseResult)


class GeneratePromptTest(absltest.TestCase):
    def test_basic_prompt(self):
        observation = {
            "observationString": '{"board_size": 9, "komi": 7.5}',
            "playerId": 0,
        }
        prompt = generate_prompt(observation, [])
        self.assertIn("Black", prompt)
        self.assertIn("Tromp-Taylor", prompt)
        self.assertIn("suicide is illegal", prompt)
        self.assertIn("superko", prompt)

    def test_white_player(self):
        observation = {
            "observationString": '{"board_size": 9}',
            "playerId": 1,
        }
        prompt = generate_prompt(observation, [])
        self.assertIn("White", prompt)
        self.assertIn("(W)", prompt)

    def test_move_history_included(self):
        observation = {
            "observationString": "{}",
            "playerId": 0,
        }
        prompt = generate_prompt(observation, ["B e5", "W d4", "B c3"])
        self.assertIn("B e5 W d4 B c3", prompt)

    def test_rethink_suffix(self):
        observation = {
            "observationString": "{}",
            "playerId": 0,
        }
        prompt = generate_prompt(
            observation,
            [],
            previous_response="I play z9",
            previous_action="z9",
        )
        self.assertIn("Your previous response was", prompt)
        self.assertIn("z9", prompt)
        self.assertIn("not in the legal moves list", prompt)

    def test_no_rethink_on_first_attempt(self):
        observation = {
            "observationString": "{}",
            "playerId": 0,
        }
        prompt = generate_prompt(observation, [])
        self.assertNotIn("Your previous response was", prompt)


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        observation = {
            "legalActions": [0, 1, 81],
            "legalActionStrings": ["B a1", "B b1", "B Pass"],
        }
        result = get_legal_moves(observation)
        self.assertEqual(result, {0: "B a1", 1: "B b1", 81: "B Pass"})

    def test_from_serialized_state(self):
        game = go_proxy.GoGame({"board_size": 9, "komi": 7.5})
        state = game.new_initial_state()
        observation = _make_observation(state, game)
        result = get_legal_moves(observation)
        # Initial 9x9 board has 81 intersections + pass = 82 legal moves
        self.assertEqual(len(result), 82)

    def test_empty_serialized(self):
        observation = {"serializedGameAndState": ""}
        result = get_legal_moves(observation)
        self.assertEqual(result, {})

    def test_returns_dict(self):
        """Verify the return type is dict[int, str]."""
        observation = {
            "legalActions": [0, 5],
            "legalActionStrings": ["B a1", "B f1"],
        }
        result = get_legal_moves(observation)
        self.assertIsInstance(result, dict)
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, str)


def _make_mock_response(content):
    """Create a mock LiteLLM response."""
    resp = MagicMock()
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
    resp.choices = [
        MagicMock(
            message=MagicMock(content=content),
            finish_reason="stop",
        )
    ]
    return resp


class _GoHarness:
    """Adapter wrapping module-level functions into the GameHarness protocol."""

    def get_legal_moves(self, observation):
        return get_legal_moves(observation)

    def make_prompt(self, observation, move_history, previous_response=None, previous_action=None):
        return generate_prompt(observation, move_history, previous_response, previous_action)

    def parse_response(self, response, legal_action_strings):
        return parse_response(response, legal_action_strings)


class AgentIntegrationTest(absltest.TestCase):
    """Test the Go harness through ``create_agent_fn`` from ``core_harness``."""

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
        """On the initial setup step with no legal moves and no player info,
        core_harness returns INACTIVE without calling the LLM."""
        mock_litellm.drop_params = True
        agent = create_agent_fn(_GoHarness())

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
            '```json\n{"move": "e5"}\n```',
        )

        agent = create_agent_fn(_GoHarness())

        game = go_proxy.GoGame({"board_size": 9, "komi": 7.5})
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {})

        self.assertEqual(result["submission"], _gtp_to_action("E5"))
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
            _make_mock_response('```json\n{"move": "z9"}\n```'),
            _make_mock_response('```json\n{"move": "a1"}\n```'),
        ]

        agent = create_agent_fn(_GoHarness())

        game = go_proxy.GoGame({"board_size": 9, "komi": 7.5})
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {})

        self.assertEqual(result["submission"], _gtp_to_action("A1"))
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

        agent = create_agent_fn(_GoHarness())

        game = go_proxy.GoGame({"board_size": 9, "komi": 7.5})
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        with self.assertRaises(ValueError):
            agent(observation, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)


if __name__ == "__main__":
    absltest.main()
