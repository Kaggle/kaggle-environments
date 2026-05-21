"""Tests for Havannah LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.havannah import havannah_proxy
from kaggle_environments.envs.open_spiel_env.games.havannah.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(state, game, player_id=0):
    """Build a harness-style observation dict from a Havannah state."""
    legal_actions = list(state.legal_actions())
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "legalActions": legal_actions,
        "legalActionStrings": [state.action_to_string(player_id, a) for a in legal_actions],
        "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
    }


class ParseResponseTest(absltest.TestCase):
    def test_parse_json_move(self):
        legal = ["a1", "b2", "g4", "e5"]
        response = '```json\n{"move": "g4"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "g4")
        self.assertEqual(result.raw_action, "g4")

    def test_parse_case_insensitive(self):
        legal = ["a1", "b2", "e5"]
        response = '```json\n{"move": "E5"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "e5")

    def test_parse_strips_whitespace(self):
        legal = ["a1", "b2"]
        response = '```json\n{"move": "  b2 "}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "b2")

    def test_parse_fallback_coordinate(self):
        legal = ["a1", "b2", "g4"]
        response = "I think g4 is the strongest move to bridge the corners."
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "g4")

    def test_parse_no_match_returns_none(self):
        legal = ["a1", "b2"]
        response = '```json\n{"move": "z99"}\n```'
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "z99")

    def test_parse_malformed_json_falls_back(self):
        legal = ["a1", "g4"]
        response = "```json\n{bad json}\n```\nI play g4."
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "g4")

    def test_parse_bare_json(self):
        legal = ["a1", "g4"]
        response = 'I think {"move": "g4"} works.'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "g4")

    def test_parse_no_signal_returns_none(self):
        legal = ["a1", "b2"]
        response = "No idea what to play."
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_returns_parse_result(self):
        legal = ["a1"]
        result = parse_response('```json\n{"move": "a1"}\n```', legal)
        self.assertIsInstance(result, ParseResult)


class GeneratePromptTest(absltest.TestCase):
    _OBS_STR = (
        '{"board": [[null, null, null, null], [null, null, null, null, null],'
        " [null, null, null, null, null, null], [null, null, null, null, null, null, null],"
        " [null, null, null, null, null, null], [null, null, null, null, null],"
        ' [null, null, null, null]], "board_size": 4, "current_player": "x",'
        ' "is_terminal": false, "winner": null, "last_move": null, "move_number": 0}'
    )

    def test_basic_prompt_for_x(self):
        observation = {"observationString": self._OBS_STR, "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertIn("Havannah", prompt)
        self.assertIn("Ring", prompt)
        self.assertIn("Bridge", prompt)
        self.assertIn("Fork", prompt)
        self.assertIn("player x", prompt)
        self.assertIn("side length 4", prompt)

    def test_prompt_for_o(self):
        observation = {"observationString": self._OBS_STR, "playerId": 1}
        prompt = generate_prompt(observation, [])
        self.assertIn("player o", prompt)

    def test_move_history_included(self):
        observation = {"observationString": self._OBS_STR, "playerId": 0}
        prompt = generate_prompt(observation, ["a1", "g4", "b2"])
        self.assertIn("a1 g4 b2", prompt)

    def test_empty_move_history(self):
        observation = {"observationString": self._OBS_STR, "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertIn("None", prompt)

    def test_rethink_suffix(self):
        observation = {"observationString": self._OBS_STR, "playerId": 0}
        prompt = generate_prompt(
            observation,
            [],
            previous_response="I play z99",
            previous_action="z99",
        )
        self.assertIn("Your previous response was", prompt)
        self.assertIn("z99", prompt)

    def test_no_rethink_on_first_attempt(self):
        observation = {"observationString": self._OBS_STR, "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertNotIn("Your previous response was", prompt)

    def test_board_state_rendered(self):
        """An x stone at a1 and o stone at a3 should appear in the rendered board."""
        obs_str = (
            '{"board": [["x", null, null, null], [null, null, null, null, null],'
            ' ["o", null, null, null, null, null],'
            " [null, null, null, null, null, null, null],"
            " [null, null, null, null, null, null], [null, null, null, null, null],"
            ' [null, null, null, null]], "board_size": 4, "current_player": "o",'
            ' "is_terminal": false, "winner": null, "last_move": "a3", "move_number": 2}'
        )
        observation = {"observationString": obs_str, "playerId": 0}
        prompt = generate_prompt(observation, ["a1", "a3"])
        self.assertIn("X", prompt)
        self.assertIn("O", prompt)


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        observation = {
            "legalActions": [0, 1, 7],
            "legalActionStrings": ["a1", "b1", "a2"],
        }
        result = get_legal_moves(observation)
        self.assertEqual(result, {0: "a1", 1: "b1", 7: "a2"})

    def test_from_serialized_state(self):
        game = havannah_proxy.HavannahGame({"board_size": 4})
        state = game.new_initial_state()
        observation = {
            "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
        }
        result = get_legal_moves(observation)
        # board_size=4 has 37 playable cells.
        self.assertEqual(len(result), 37)

    def test_empty_serialized(self):
        observation = {"serializedGameAndState": ""}
        result = get_legal_moves(observation)
        self.assertEqual(result, {})

    def test_returns_dict(self):
        observation = {"legalActions": [0, 5], "legalActionStrings": ["a1", "f1"]}
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


class _HavannahHarness:
    """Adapter wrapping module-level functions into the GameHarness protocol."""

    def get_legal_moves(self, observation):
        return get_legal_moves(observation)

    def make_prompt(self, observation, move_history, previous_response=None, previous_action=None):
        return generate_prompt(observation, move_history, previous_response, previous_action)

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
        agent = create_agent_fn(_HavannahHarness())
        result = agent({"step": 0, "remainingOverageTime": 60}, {})
        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response('```json\n{"move": "a1"}\n```')
        agent = create_agent_fn(_HavannahHarness())

        game = havannah_proxy.HavannahGame({"board_size": 4})
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {})
        self.assertEqual(result["submission"], 0)  # action "a1" == id 0
        self.assertEqual(result["status"], "OK")

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "z99"}\n```'),
            _make_mock_response('```json\n{"move": "b1"}\n```'),
        ]
        agent = create_agent_fn(_HavannahHarness())

        game = havannah_proxy.HavannahGame({"board_size": 4})
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {})
        self.assertEqual(result["submission"], 1)  # action "b1" == id 1
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("no idea")
        agent = create_agent_fn(_HavannahHarness())

        game = havannah_proxy.HavannahGame({"board_size": 4})
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        with self.assertRaises(ValueError):
            agent(observation, {})
        self.assertEqual(mock_litellm.completion.call_count, 2)


if __name__ == "__main__":
    absltest.main()
