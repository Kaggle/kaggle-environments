"""Tests for Snake LLM harness."""

import json
from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.snake import snake_proxy
from kaggle_environments.envs.open_spiel_env.games.snake.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)

_LEGAL = ["UP", "DOWN", "LEFT", "RIGHT"]


def _make_observation(state, game, player_id=0):
    """Build a harness-style observation dict from a snake state."""
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "currentPlayer": state.current_player(),
        "isTerminal": state.is_terminal(),
        "serializedGameAndState": pyspiel.serialize_game_and_state(
            game.__wrapped__,
            state.__wrapped__,
        ),
    }


class ParseResponseTest(absltest.TestCase):
    def test_parse_json_move(self):
        response = '```json\n{"move": "UP"}\n```'
        result = parse_response(response, _LEGAL)
        self.assertEqual(result.legal_action, "UP")
        self.assertEqual(result.raw_action, "UP")

    def test_parse_each_action(self):
        for action in _LEGAL:
            response = f'```json\n{{"move": "{action}"}}\n```'
            result = parse_response(response, _LEGAL)
            self.assertEqual(result.legal_action, action)

    def test_parse_case_insensitive(self):
        response = '```json\n{"move": "up"}\n```'
        result = parse_response(response, _LEGAL)
        self.assertEqual(result.legal_action, "UP")

    def test_parse_with_whitespace(self):
        response = '```json\n{"move": "  down  "}\n```'
        result = parse_response(response, _LEGAL)
        self.assertEqual(result.legal_action, "DOWN")

    def test_parse_fallback_keyword_in_text(self):
        response = "Food is to my right, so I will go right toward it."
        result = parse_response(response, _LEGAL)
        self.assertEqual(result.legal_action, "RIGHT")

    def test_parse_no_match_returns_none(self):
        response = '```json\n{"move": "diagonal"}\n```'
        result = parse_response(response, _LEGAL)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "diagonal")

    def test_parse_malformed_json_falls_back(self):
        response = "```json\n{bad json}\n```\nI'll go LEFT."
        result = parse_response(response, _LEGAL)
        self.assertEqual(result.legal_action, "LEFT")

    def test_parse_no_move_returns_none(self):
        response = "I have no idea what to do."
        result = parse_response(response, _LEGAL)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_bare_json(self):
        response = 'I think {"move": "DOWN"} is best.'
        result = parse_response(response, _LEGAL)
        self.assertEqual(result.legal_action, "DOWN")

    def test_parse_returns_parse_result(self):
        result = parse_response('```json\n{"move": "UP"}\n```', _LEGAL)
        self.assertIsInstance(result, ParseResult)


class GeneratePromptTest(absltest.TestCase):
    def _obs(self, player_id=0):
        state = json.dumps(
            {
                "num_rows": 10,
                "num_columns": 10,
                "num_players": 2,
                "food": [3, 4],
                "snakes": [
                    {"player": 0, "body": [[1, 1]], "alive": True, "score": 0},
                    {"player": 1, "body": [[8, 8]], "alive": True, "score": 0},
                ],
                "scores": [0, 0],
                "is_alive": [True, True],
                "current_player": player_id,
                "turn": 0,
                "is_terminal": False,
                "winner": None,
            }
        )
        return {"observationString": state, "playerId": player_id}

    def test_basic_prompt_contains_rules(self):
        prompt = generate_prompt(self._obs(), [])
        self.assertIn("Snake", prompt)
        self.assertIn("UP, DOWN, LEFT, RIGHT", prompt)
        self.assertIn("simultaneously", prompt)
        self.assertIn("food", prompt.lower())

    def test_prompt_includes_player_id(self):
        prompt = generate_prompt(self._obs(player_id=1), [])
        self.assertIn("player 1", prompt)

    def test_prompt_includes_dimensions(self):
        prompt = generate_prompt(self._obs(), [])
        self.assertIn("10x10", prompt)

    def test_prompt_includes_food_location(self):
        prompt = generate_prompt(self._obs(), [])
        self.assertIn("[3, 4]", prompt)

    def test_prompt_includes_own_snake_body(self):
        prompt = generate_prompt(self._obs(player_id=1), [])
        self.assertIn("[8, 8]", prompt)

    def test_move_history_included(self):
        prompt = generate_prompt(self._obs(), ["UP", "RIGHT", "DOWN"])
        self.assertIn("UP RIGHT DOWN", prompt)

    def test_rethink_suffix(self):
        prompt = generate_prompt(
            self._obs(),
            [],
            previous_response="I move diagonally",
            previous_action="diagonal",
        )
        self.assertIn("Your previous response was", prompt)
        self.assertIn("diagonal", prompt)
        self.assertIn("not in the legal", prompt)

    def test_no_rethink_on_first_attempt(self):
        prompt = generate_prompt(self._obs(), [])
        self.assertNotIn("Your previous response was", prompt)

    def test_handles_dead_snake(self):
        state = json.dumps(
            {
                "num_rows": 10,
                "num_columns": 10,
                "num_players": 2,
                "food": [3, 4],
                "snakes": [
                    {"player": 0, "body": [], "alive": False, "score": 1},
                    {"player": 1, "body": [[5, 5]], "alive": True, "score": 0},
                ],
                "is_terminal": False,
                "winner": None,
            }
        )
        prompt = generate_prompt({"observationString": state, "playerId": 0}, [])
        self.assertIn("DEAD", prompt)

    def test_handles_missing_observation(self):
        prompt = generate_prompt({"observationString": "", "playerId": 0}, [])
        # Should not crash; falls back to defaults.
        self.assertIn("Snake", prompt)


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        observation = {
            "legalActions": [0, 1, 2, 3],
            "legalActionStrings": ["UP", "DOWN", "LEFT", "RIGHT"],
        }
        result = get_legal_moves(observation)
        self.assertEqual(result, {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"})

    def test_from_serialized_state(self):
        game = snake_proxy.SnakeGame()
        state = game.new_initial_state()
        observation = {
            "serializedGameAndState": pyspiel.serialize_game_and_state(
                game.__wrapped__,
                state.__wrapped__,
            ),
        }
        result = get_legal_moves(observation)
        self.assertEqual(sorted(result.values()), ["DOWN", "LEFT", "RIGHT", "UP"])

    def test_empty_serialized(self):
        observation = {"serializedGameAndState": ""}
        result = get_legal_moves(observation)
        self.assertEqual(result, {})

    def test_returns_dict(self):
        observation = {
            "legalActions": [0, 3],
            "legalActionStrings": ["UP", "RIGHT"],
        }
        result = get_legal_moves(observation)
        self.assertIsInstance(result, dict)
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, str)


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
    """Build a streaming-style mock LLM response (a re-iterable chunk list)."""
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


class _SnakeHarness:
    """Adapter wrapping module-level functions in the GameHarness protocol."""

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
    """Test the snake harness through ``create_agent_fn``."""

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_setup_step_returns_inactive(self, mock_litellm):
        mock_litellm.drop_params = True
        agent = create_agent_fn(_SnakeHarness())
        result = agent({"step": 0, "remainingOverageTime": 60}, {})
        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            '```json\n{"move": "UP"}\n```',
        )
        agent = create_agent_fn(_SnakeHarness())

        game = snake_proxy.SnakeGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game, state.current_player())

        result = agent(observation, {})

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["actionString"], "UP")
        self.assertEqual(
            state.__wrapped__.action_to_string(
                state.current_player(),
                result["submission"],
            ),
            "UP",
        )

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "diagonal"}\n```'),
            _make_mock_response('```json\n{"move": "LEFT"}\n```'),
        ]
        agent = create_agent_fn(_SnakeHarness())

        game = snake_proxy.SnakeGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game, state.current_player())

        result = agent(observation, {})

        self.assertEqual(result["actionString"], "LEFT")
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            "I have no idea what to do",
        )
        agent = create_agent_fn(_SnakeHarness())

        game = snake_proxy.SnakeGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game, state.current_player())

        with self.assertRaises(ValueError):
            agent(observation, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)


if __name__ == "__main__":
    absltest.main()
