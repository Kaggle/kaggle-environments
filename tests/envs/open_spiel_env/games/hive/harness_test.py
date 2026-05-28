"""Tests for Hive LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.hive import hive_proxy
from kaggle_environments.envs.open_spiel_env.games.hive.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(state, game, player_id=0):
    """Build a harness-style observation dict from a Hive state."""
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
        legal = ["wQ", "wA1", "wG1", "wB1"]
        response = '```json\n{"move": "wA1"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "wA1")
        self.assertEqual(result.raw_action, "wA1")

    def test_parse_case_insensitive(self):
        legal = ["wA1 wQ-"]
        response = '```json\n{"move": "WA1 WQ-"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "wA1 wQ-")

    def test_parse_strips_whitespace(self):
        legal = ["wA1 wQ-"]
        response = '```json\n{"move": "  wA1   wQ-  "}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "wA1 wQ-")

    def test_parse_fallback_text_scan(self):
        legal = ["wQ", "wA1", "wG1"]
        response = "After some thought I'll play wG1 because grasshoppers."
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "wG1")

    def test_parse_prefers_longest_legal_match(self):
        legal = ["wA1", "wA1 wQ-"]
        response = "I'm going to play wA1 wQ- this turn."
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "wA1 wQ-")

    def test_parse_no_match_returns_none(self):
        legal = ["wQ", "wA1"]
        response = '```json\n{"move": "zZ9"}\n```'
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "zZ9")

    def test_parse_bare_json(self):
        legal = ["wQ", "wA1"]
        response = 'I think {"move": "wQ"} works.'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "wQ")

    def test_parse_pass(self):
        legal = ["pass"]
        response = '```json\n{"move": "pass"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "pass")

    def test_parse_no_signal_returns_none(self):
        legal = ["wQ", "wA1"]
        response = "Hmm I have no idea."
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_returns_parse_result(self):
        result = parse_response('```json\n{"move": "wQ"}\n```', ["wQ"])
        self.assertIsInstance(result, ParseResult)

    def test_json_with_illegal_move_does_not_fall_back_to_prose(self):
        # The model explicitly stated its intent via JSON. If the JSON move
        # isn't legal, we should NOT silently substitute a different move
        # found in the surrounding prose -- let the rethink loop handle it.
        legal = ["wQ", "wA1", "wG1"]
        response = 'I\'ll play wG1.\n```json\n{"move": "wZ9"}\n```'
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "wZ9")

    def test_fallback_requires_word_boundary(self):
        # Short tokens like "wA1" must not match inside larger identifiers.
        legal = ["wA1", "wQ"]
        response = "Looking at piece wA12 (made up) -- not sure what to do."
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)

    def test_fallback_skips_pass(self):
        # ``pass`` is a common English word, so the prose scanner must not
        # match it. The model has to use JSON for a pass.
        legal = ["pass", "wA1", "wQ"]
        response = "I'm going to pass on the spider for now and think more."
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)


class GeneratePromptTest(absltest.TestCase):
    _OBS_STR = (
        '{"game_type": "Base+MLP", "expansions": {"mosquito": true, "ladybug": true,'
        ' "pillbug": true}, "board_radius": 8, "status": "NotStarted",'
        ' "turn": "White[1]", "current_player": "white", "move_number": 0,'
        ' "moves": [], "last_move": null, "legal_moves": ["wA1", "wQ"],'
        ' "pieces": {}, "is_terminal": false, "winner": null,'
        ' "uhp": "Base+MLP;NotStarted;White[1];"}'
    )

    def test_basic_prompt_for_white(self):
        observation = {"observationString": self._OBS_STR, "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertIn("Hive", prompt)
        self.assertIn("Queen Bee", prompt)
        self.assertIn("White (w)", prompt)
        self.assertIn("UHP", prompt)

    def test_prompt_for_black(self):
        observation = {"observationString": self._OBS_STR, "playerId": 1}
        prompt = generate_prompt(observation, [])
        self.assertIn("Black (b)", prompt)

    def test_move_history_included(self):
        observation = {"observationString": self._OBS_STR, "playerId": 0}
        prompt = generate_prompt(observation, ["wA1", "bA1 wA1/"])
        self.assertIn("wA1", prompt)
        self.assertIn("bA1 wA1/", prompt)

    def test_empty_move_history(self):
        observation = {"observationString": self._OBS_STR, "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertIn("None", prompt)

    def test_pieces_section_rendered(self):
        obs_str = (
            '{"game_type": "Base+MLP", "status": "InProgress", "current_player":'
            ' "black", "move_number": 1, "moves": ["wA1"], "last_move": "wA1",'
            ' "legal_moves": ["bQ wA1-"], "pieces": {"wA1": [0, 0, 0]},'
            ' "is_terminal": false, "winner": null, "uhp": "..."}'
        )
        observation = {"observationString": obs_str, "playerId": 1}
        prompt = generate_prompt(observation, ["wA1"])
        self.assertIn("wA1: [0, 0, 0]", prompt)

    def test_rethink_suffix(self):
        observation = {"observationString": self._OBS_STR, "playerId": 0}
        prompt = generate_prompt(
            observation,
            [],
            previous_response="I play zZ9",
            previous_action="zZ9",
        )
        self.assertIn("Your previous response was", prompt)
        self.assertIn("zZ9", prompt)

    def test_no_rethink_on_first_attempt(self):
        observation = {"observationString": self._OBS_STR, "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertNotIn("Your previous response was", prompt)

    def test_does_not_enumerate_legal_moves(self):
        # The prompt should not dump the full legal_moves list -- the proxy may
        # carry hundreds of entries.
        observation = {"observationString": self._OBS_STR, "playerId": 0}
        prompt = generate_prompt(observation, [])
        # The empty array shouldn't appear, and neither should the literal key.
        self.assertNotIn('"legal_moves"', prompt)


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        observation = {
            "legalActions": [209, 412],
            "legalActionStrings": ["wA1", "wA2"],
        }
        result = get_legal_moves(observation)
        self.assertEqual(result, {209: "wA1", 412: "wA2"})

    def test_from_serialized_state(self):
        game = hive_proxy.HiveGame()
        state = game.new_initial_state()
        observation = {
            "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
        }
        result = get_legal_moves(observation)
        # First move: white can place any of its 13 bug types (Base+MLP).
        self.assertEqual(len(result), 13)
        self.assertIn("wA1", result.values())

    def test_empty_serialized(self):
        observation = {"serializedGameAndState": ""}
        result = get_legal_moves(observation)
        self.assertEqual(result, {})

    def test_returns_dict(self):
        observation = {
            "legalActions": [0, 5],
            "legalActionStrings": ["wQ", "wA1"],
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


class _HiveHarness:
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
        agent = create_agent_fn(_HiveHarness())
        result = agent({"step": 0, "remainingOverageTime": 60}, {})
        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response('```json\n{"move": "wA1"}\n```')
        agent = create_agent_fn(_HiveHarness())

        game = hive_proxy.HiveGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {})
        self.assertEqual(result["status"], "OK")
        # "wA1" is action id 209 for Base+MLP.
        self.assertEqual(result["submission"], 209)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "totally bogus"}\n```'),
            _make_mock_response('```json\n{"move": "wG1"}\n```'),
        ]
        agent = create_agent_fn(_HiveHarness())

        game = hive_proxy.HiveGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {})
        self.assertEqual(result["status"], "OK")
        # The Queen Bee is not a legal first move (Tournament rule); placing
        # the white Grasshopper "wG1" is.
        self.assertEqual(result["submission"], 818)
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("no idea")
        agent = create_agent_fn(_HiveHarness())

        game = hive_proxy.HiveGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        with self.assertRaises(ValueError):
            agent(observation, {})
        self.assertEqual(mock_litellm.completion.call_count, 2)


if __name__ == "__main__":
    absltest.main()
