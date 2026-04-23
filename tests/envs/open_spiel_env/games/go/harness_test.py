"""Tests for Go LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.envs.open_spiel_env.games.go import go_proxy
from kaggle_environments.envs.open_spiel_env.games.go.harness import (
    _get_legal_moves,
    _make_go_prompt,
    _parse_go_response,
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


class ParseGoResponseTest(absltest.TestCase):
    def test_parse_json_move(self):
        legal = ["B a1", "B b2", "B e5", "B Pass"]
        response = '```json\n{"move": "e5"}\n```'
        self.assertEqual(_parse_go_response(response, legal), "B e5")

    def test_parse_pass(self):
        legal = ["B a1", "B Pass"]
        response = '```json\n{"move": "PASS"}\n```'
        self.assertEqual(_parse_go_response(response, legal), "B Pass")

    def test_parse_case_insensitive(self):
        legal = ["B a1", "B E5"]
        response = '```json\n{"move": "e5"}\n```'
        self.assertEqual(_parse_go_response(response, legal), "B E5")

    def test_parse_fallback_coordinate(self):
        """Falls back to searching for coordinates in response text."""
        legal = ["B a1", "B b2", "B e5"]
        response = "I think e5 is the best move here."
        self.assertEqual(_parse_go_response(response, legal), "B e5")

    def test_parse_no_match_returns_none(self):
        legal = ["B a1", "B b2"]
        response = '```json\n{"move": "z9"}\n```'
        self.assertIsNone(_parse_go_response(response, legal))

    def test_parse_malformed_json_falls_back(self):
        legal = ["B a1", "B e5"]
        response = "```json\n{bad json}\n```\nI play e5."
        self.assertEqual(_parse_go_response(response, legal), "B e5")

    def test_parse_no_json_no_coord_returns_none(self):
        legal = ["B a1", "B b2"]
        response = "I have no idea what to play."
        self.assertIsNone(_parse_go_response(response, legal))


class MakeGoPromptTest(absltest.TestCase):
    def test_basic_prompt(self):
        observation = {
            "observationString": '{"board_size": 9, "komi": 7.5}',
            "playerId": 0,
        }
        prompt = _make_go_prompt(observation, [])
        self.assertIn("Black", prompt)
        self.assertIn("Tromp-Taylor", prompt)
        self.assertIn("suicide is illegal", prompt)
        self.assertIn("superko", prompt)

    def test_white_player(self):
        observation = {
            "observationString": '{"board_size": 9}',
            "playerId": 1,
        }
        prompt = _make_go_prompt(observation, [])
        self.assertIn("White", prompt)
        self.assertIn("(W)", prompt)

    def test_move_history_included(self):
        observation = {
            "observationString": "{}",
            "playerId": 0,
        }
        prompt = _make_go_prompt(observation, ["B e5", "W d4", "B c3"])
        self.assertIn("B e5 W d4 B c3", prompt)

    def test_rethink_suffix(self):
        observation = {
            "observationString": "{}",
            "playerId": 0,
        }
        prompt = _make_go_prompt(
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
        prompt = _make_go_prompt(observation, [])
        self.assertNotIn("Your previous response was", prompt)


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        observation = {
            "legalActions": [0, 1, 81],
            "legalActionStrings": ["B a1", "B b1", "B Pass"],
        }
        actions, strings = _get_legal_moves(observation)
        self.assertEqual(actions, [0, 1, 81])
        self.assertEqual(strings, ["B a1", "B b1", "B Pass"])

    def test_from_serialized_state(self):
        game = go_proxy.GoGame({"board_size": 9, "komi": 7.5})
        state = game.new_initial_state()
        observation = _make_observation(state, game)
        actions, strings = _get_legal_moves(observation)
        # Initial 9x9 board has 81 intersections + pass = 82 legal moves
        self.assertEqual(len(actions), 82)
        self.assertEqual(len(strings), 82)

    def test_empty_serialized(self):
        observation = {"serializedGameAndState": ""}
        actions, strings = _get_legal_moves(observation)
        self.assertEqual(actions, [])
        self.assertEqual(strings, [])


class AgentFnTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        # Reset module-level state before each test
        import kaggle_environments.envs.open_spiel_env.games.go.harness as h

        h._SETUP_COMPLETE = False
        h._MODEL_NAME = None
        h._LITELLM_KWARGS = {}
        h._MOVE_HISTORY = []

    @patch.dict(
        "os.environ",
        {
            "MODEL_NAME": "test-model",
            "MODEL_PROXY_KEY": "test-key",
            "MODEL_PROXY_URL": "dummy_url",
        },
    )
    @patch("kaggle_environments.envs.open_spiel_env.games.go.harness.litellm")
    def test_setup_step_returns_noop(self, mock_litellm):
        """On the initial setup step the observation has no legal moves;
        the harness must return submission=-1 without calling the LLM."""
        from kaggle_environments.envs.open_spiel_env.games.go.harness import agent_fn

        mock_litellm.drop_params = True

        result = agent_fn({"step": 0, "remainingOverageTime": 60}, {})

        self.assertEqual(result["submission"], -1)
        self.assertEqual(result["actionString"], None)
        self.assertEqual(result["thoughts"], None)
        self.assertEqual(result["status"], "OK; Setup step; model not called.")
        self.assertEqual(result["generate_returns"], [])
        mock_litellm.completion.assert_not_called()

    @patch.dict(
        "os.environ",
        {
            "MODEL_NAME": "test-model",
            "MODEL_PROXY_KEY": "test-key",
            "MODEL_PROXY_URL": "dummy_url",
        },
    )
    @patch("kaggle_environments.envs.open_spiel_env.games.go.harness.litellm")
    def test_successful_move(self, mock_litellm):
        from kaggle_environments.envs.open_spiel_env.games.go.harness import agent_fn

        mock_litellm.drop_params = True
        mock_response = MagicMock()
        mock_response.usage = {}
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content='```json\n{"move": "e5"}\n```'),
                finish_reason="stop",
            )
        ]
        mock_litellm.completion.return_value = mock_response

        game = go_proxy.GoGame({"board_size": 9, "komi": 7.5})
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent_fn(observation, {})

        self.assertIn("submission", result)
        # e5 = row 4, col 4 = 4*9+4 = 40
        self.assertEqual(result["submission"], _gtp_to_action("E5"))

    @patch.dict(
        "os.environ",
        {
            "MODEL_NAME": "test-model",
            "MODEL_PROXY_KEY": "test-key",
            "MODEL_PROXY_URL": "dummy_url",
        },
    )
    @patch("kaggle_environments.envs.open_spiel_env.games.go.harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        from kaggle_environments.envs.open_spiel_env.games.go.harness import agent_fn

        mock_litellm.drop_params = True

        # First response: bad move, second response: good move
        bad_response = MagicMock()
        bad_response.usage = {}
        bad_response.choices = [
            MagicMock(
                message=MagicMock(content='```json\n{"move": "z9"}\n```'),
                finish_reason="stop",
            )
        ]
        good_response = MagicMock()
        good_response.usage = {}
        good_response.choices = [
            MagicMock(
                message=MagicMock(content='```json\n{"move": "a1"}\n```'),
                finish_reason="stop",
            )
        ]
        mock_litellm.completion.side_effect = [bad_response, good_response]

        game = go_proxy.GoGame({"board_size": 9, "komi": 7.5})
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent_fn(observation, {})

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
    @patch("kaggle_environments.envs.open_spiel_env.games.go.harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        from kaggle_environments.envs.open_spiel_env.games.go.harness import agent_fn

        mock_litellm.drop_params = True

        bad_response = MagicMock()
        bad_response.choices = [MagicMock(message=MagicMock(content="I don't know what to play"))]
        mock_litellm.completion.return_value = bad_response

        game = go_proxy.GoGame({"board_size": 9, "komi": 7.5})
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        with self.assertRaises(ValueError):
            agent_fn(observation, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)


if __name__ == "__main__":
    absltest.main()
