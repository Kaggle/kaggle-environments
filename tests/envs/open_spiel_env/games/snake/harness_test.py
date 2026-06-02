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

    def test_prose_only_response_triggers_rethink(self):
        # No structured JSON. The parser must NOT guess at intent from a
        # direction keyword in the prose -- return None and let the
        # rethink loop ask the model to use the required JSON format.
        response = "Food is to my right, so I will go right toward it."
        result = parse_response(response, _LEGAL)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_no_match_returns_none(self):
        response = '```json\n{"move": "diagonal"}\n```'
        result = parse_response(response, _LEGAL)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "diagonal")

    def test_malformed_json_triggers_rethink(self):
        # Bad JSON: stage-1 extracts nothing. The parser must NOT
        # silently rescue a direction from the prose -- return None so
        # the rethink loop can ask the model to fix its format.
        response = "```json\n{bad json}\n```\nI'll go LEFT."
        result = parse_response(response, _LEGAL)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

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

    def test_illegal_json_does_not_ghost_substitute_from_prose(self):
        # The JSON answer ("diagonal") is illegal. The parser must NOT
        # silently substitute "UP" mentioned in the prose (the ghost
        # antipattern). Surface raw_action so the rethink loop fires.
        response = (
            "I considered UP but ruled it out. I'll play diagonal.\n"
            '```json\n{"move": "diagonal"}\n```'
        )
        result = parse_response(response, _LEGAL)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "diagonal")


class GeneratePromptTest(absltest.TestCase):
    def _obs(self, player_id=0, round_history=None):
        # Minimal 10x10 board with P0 head at (1,1) and P1 head at (8,8).
        board = [["." for _ in range(10)] for _ in range(10)]
        board[1][1] = "A"
        board[8][8] = "B"
        board[3][4] = "*"
        board[6][5] = "*"
        state = json.dumps(
            {
                "board": board,
                "num_rows": 10,
                "num_columns": 10,
                "num_players": 2,
                "foods": [[3, 4], [6, 5]],
                "food": [3, 4],
                "food_respawn_interval": 10,
                "turns_until_respawn": 7,
                "snakes": [
                    {"player": 0, "body": [[1, 1]], "alive": True, "score": 0},
                    {"player": 1, "body": [[8, 8]], "alive": True, "score": 0},
                ],
                "scores": [0, 0],
                "is_alive": [True, True],
                "current_player": player_id,
                "round_history": round_history or [],
                "turn": 3,
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
        self.assertIn("[6, 5]", prompt)

    def test_prompt_includes_own_snake_body(self):
        prompt = generate_prompt(self._obs(player_id=1), [])
        self.assertIn("[8, 8]", prompt)

    def test_board_rendered_as_ascii_grid(self):
        # The board must be a multi-line ASCII grid (not a raw JSON blob),
        # so the model can read spatial structure directly.
        prompt = generate_prompt(self._obs(), [])
        # P0 head 'A' at (1,1) in our test obs.
        self.assertIn(".A........", prompt)
        # Multi-line: at least one bare row of '.' between markers.
        self.assertIn("\n..........\n", prompt)
        # And the raw JSON board array must NOT appear.
        self.assertNotIn('"board":', prompt)

    def test_round_history_renders_opponent_moves(self):
        # The proxy's round_history is a list of per-round action lists,
        # one entry per player. The prompt must surface BOTH players' moves
        # so each side can see what the opponent did.
        rounds = [["UP", "LEFT"], ["UP", "DOWN"]]
        prompt = generate_prompt(self._obs(round_history=rounds), [])
        self.assertIn("Round 1: P0=UP, P1=LEFT", prompt)
        self.assertIn("Round 2: P0=UP, P1=DOWN", prompt)

    def test_round_history_empty(self):
        prompt = generate_prompt(self._obs(), [])
        self.assertIn("(no moves yet)", prompt)

    def test_ignores_framework_move_history(self):
        # The harness reads round_history from the proxy, not the
        # framework-supplied per-agent move_history (which has no
        # opponent moves and would mislead in a simultaneous game).
        prompt = generate_prompt(self._obs(), ["UP", "RIGHT", "DOWN"])
        self.assertNotIn("UP RIGHT DOWN", prompt)

    def test_rethink_suffix(self):
        prompt = generate_prompt(
            self._obs(),
            [],
            previous_response="I move diagonally",
            previous_action="diagonal",
        )
        self.assertIn("You suggested", prompt)  # ILLEGAL leads with action
        self.assertIn("diagonal", prompt)
        self.assertIn("not a legal", prompt)

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
