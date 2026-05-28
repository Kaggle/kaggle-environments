"""Tests for Quoridor LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.quoridor import quoridor_proxy
from kaggle_environments.envs.open_spiel_env.games.quoridor.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(state, game, player_id=0):
    """Build a harness-style observation dict from a Quoridor state."""
    legal_actions = list(state.legal_actions())
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "legalActions": legal_actions,
        "legalActionStrings": [state.action_to_string(player_id, a) for a in legal_actions],
        "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
    }


# A representative observation string: 3x3 board, default walls, start state.
_OBS_STR_3X3 = (
    '{"board_size": 3, "num_players": 2,'
    ' "cells": [[null, 1, null], [null, null, null], [null, 0, null]],'
    ' "pawns": {"o": "b1", "x": "b3"},'
    ' "vertical_walls": [], "horizontal_walls": [],'
    ' "walls_remaining": {"x": 1, "o": 1},'
    ' "current_player": "x", "is_terminal": false, "winner": null,'
    ' "legal_actions": ["b2", "a3", "c3", "a1v", "b1v", "a2v", "b2v",'
    ' "a1h", "b1h", "a2h", "b2h"], "move_number": 0}'
)


class ParseResponseTest(absltest.TestCase):
    def test_parse_json_pawn_move(self):
        legal = ["e5", "d6", "f6", "a1v"]
        response = '```json\n{"move": "e5"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "e5")
        self.assertEqual(result.raw_action, "e5")

    def test_parse_json_wall(self):
        legal = ["e5", "a1v", "b2h"]
        response = '```json\n{"move": "b2h"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "b2h")

    def test_parse_case_insensitive(self):
        legal = ["e5", "a1v"]
        response = '```json\n{"move": "A1V"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "a1v")

    def test_parse_strips_whitespace(self):
        legal = ["e5", "b2"]
        response = '```json\n{"move": "  b2 "}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "b2")

    def test_parse_fallback_text_scan(self):
        legal = ["e5", "d6", "a1v"]
        response = "After thinking, I'll play e5 to advance toward the goal."
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "e5")

    def test_parse_fallback_wall(self):
        legal = ["e5", "a1v", "c3h"]
        response = "I will place wall c3h to block the opponent."
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "c3h")

    def test_parse_no_match_returns_none(self):
        legal = ["e5", "d6"]
        response = '```json\n{"move": "z99v"}\n```'
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "z99v")

    def test_parse_malformed_json_falls_back(self):
        legal = ["e5", "a1v"]
        response = "```json\n{bad json}\n```\nI choose a1v."
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "a1v")

    def test_parse_bare_json(self):
        legal = ["e5", "a1v"]
        response = 'My answer: {"move": "a1v"} done.'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "a1v")

    def test_parse_no_signal_returns_none(self):
        legal = ["e5", "a1v"]
        response = "No idea what to play."
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_returns_parse_result(self):
        legal = ["e5"]
        result = parse_response('```json\n{"move": "e5"}\n```', legal)
        self.assertIsInstance(result, ParseResult)

    def test_parse_json_with_extra_fields(self):
        """A bare JSON object with extra keys (no fenced block) still parses."""
        legal = ["e5", "a1v"]
        response = 'Final answer: {"move": "a1v", "confidence": 0.9, "alts": ["e5"]}'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "a1v")

    def test_parse_prefers_last_text_match(self):
        """Models brainstorm earlier candidates and state the final move at the end."""
        legal = ["a1v", "b2", "c3"]
        response = "First I considered b2, then c3, but actually I'll play a1v."
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "a1v")


class GeneratePromptTest(absltest.TestCase):
    def test_basic_prompt_for_x(self):
        obs = {"observationString": _OBS_STR_3X3, "playerId": 0}
        prompt = generate_prompt(obs, [])
        self.assertIn("Quoridor", prompt)
        self.assertIn("3x3", prompt)
        self.assertIn("player x", prompt)
        # Wall placement notation must be explained.
        self.assertIn("a1v", prompt)
        self.assertIn("a1h", prompt)

    def test_prompt_for_o(self):
        obs = {"observationString": _OBS_STR_3X3, "playerId": 1}
        prompt = generate_prompt(obs, [])
        self.assertIn("player o", prompt)

    def test_walls_remaining_for_player(self):
        obs = {"observationString": _OBS_STR_3X3, "playerId": 0}
        prompt = generate_prompt(obs, [])
        self.assertIn("Your remaining walls: 1", prompt)
        self.assertIn("Opponent's remaining walls: 1", prompt)

    def test_move_history_included(self):
        obs = {"observationString": _OBS_STR_3X3, "playerId": 0}
        prompt = generate_prompt(obs, ["b2", "a1", "b1"])
        self.assertIn("Your moves so far: b2 a1 b1", prompt)

    def test_empty_move_history(self):
        obs = {"observationString": _OBS_STR_3X3, "playerId": 0}
        prompt = generate_prompt(obs, [])
        self.assertIn("Your moves so far: None", prompt)

    def test_prompt_mentions_draw_rule(self):
        obs = {"observationString": _OBS_STR_3X3, "playerId": 0}
        prompt = generate_prompt(obs, [])
        # 4 * 3^2 = 36 max moves on a 3x3 board.
        self.assertIn("36", prompt)
        self.assertIn("draw", prompt.lower())

    def test_prompt_mentions_pass_action(self):
        obs = {"observationString": _OBS_STR_3X3, "playerId": 0}
        prompt = generate_prompt(obs, [])
        self.assertIn("boxed in", prompt)

    def test_empty_observation_raises(self):
        obs = {"observationString": "", "playerId": 0}
        with self.assertRaises(ValueError):
            generate_prompt(obs, [])

    def test_rethink_suffix(self):
        obs = {"observationString": _OBS_STR_3X3, "playerId": 0}
        prompt = generate_prompt(
            obs,
            [],
            previous_response="I play z99",
            previous_action="z99",
        )
        self.assertIn("Your previous response was", prompt)
        self.assertIn("z99", prompt)

    def test_no_rethink_on_first_attempt(self):
        obs = {"observationString": _OBS_STR_3X3, "playerId": 0}
        prompt = generate_prompt(obs, [])
        self.assertNotIn("Your previous response was", prompt)

    def test_board_renders_pawns(self):
        obs = {"observationString": _OBS_STR_3X3, "playerId": 0}
        prompt = generate_prompt(obs, [])
        # Pawn glyphs should appear in the rendered board.
        self.assertIn(" x ", prompt)
        self.assertIn(" o ", prompt)

    def test_board_renders_walls(self):
        obs_str = (
            '{"board_size": 3, "num_players": 2,'
            ' "cells": [[null, 1, null], [null, null, null], [null, 0, null]],'
            ' "pawns": {"o": "b1", "x": "b3"},'
            ' "vertical_walls": ["a1v"], "horizontal_walls": ["a2h"],'
            ' "walls_remaining": {"x": 0, "o": 1},'
            ' "current_player": "o", "is_terminal": false, "winner": null,'
            ' "legal_actions": [], "move_number": 2}'
        )
        obs = {"observationString": obs_str, "playerId": 1}
        prompt = generate_prompt(obs, ["a1v", "a2h"])
        # Vertical wall character and horizontal wall dashes should both appear.
        self.assertIn("|", prompt)
        self.assertIn("---", prompt)
        # And the wall lists should be reflected in the summary section.
        self.assertIn("Vertical:   a1v", prompt)
        self.assertIn("Horizontal: a2h", prompt)


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        obs = {
            "legalActions": [2, 17, 1],
            "legalActionStrings": ["b2", "a1h", "a1v"],
        }
        result = get_legal_moves(obs)
        self.assertEqual(result, {2: "b2", 17: "a1h", 1: "a1v"})

    def test_from_serialized_state(self):
        game = quoridor_proxy.QuoridorGame({"board_size": 3, "wall_count": 1})
        state = game.new_initial_state()
        obs = {
            "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
        }
        result = get_legal_moves(obs)
        # 3x3 with 1 wall each player: 3 pawn moves + 2*(3-1)*(3-1) = 11 walls.
        # Returned dict keys should match the live legal action count.
        self.assertEqual(len(result), len(state.legal_actions()))
        # Every value should look like a Quoridor move string.
        for move in result.values():
            self.assertRegex(move, r"^[a-c]\d[vh]?$")

    def test_empty_serialized(self):
        obs = {"serializedGameAndState": ""}
        self.assertEqual(get_legal_moves(obs), {})

    def test_returns_dict(self):
        obs = {"legalActions": [0, 2], "legalActionStrings": ["b1", "b2"]}
        result = get_legal_moves(obs)
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


class _QuoridorHarness:
    """Adapter wrapping module-level functions into the GameHarness protocol.

    The harness module itself ships free functions (not a class) because the
    notebook-style competition runner drops the file in as-is and expects
    those exact callables. This adapter exists only so the test suite can
    exercise the harness via ``create_agent_fn``.
    """

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
        agent = create_agent_fn(_QuoridorHarness())
        result = agent({"step": 0, "remainingOverageTime": 60}, {})
        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response('```json\n{"move": "b2"}\n```')
        agent = create_agent_fn(_QuoridorHarness())

        game = quoridor_proxy.QuoridorGame({"board_size": 3, "wall_count": 1})
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {})
        # Action "b2" -- the relative-encoded "step up" for player 0 -- has id 2.
        self.assertEqual(result["submission"], 2)
        self.assertEqual(result["status"], "OK")

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "z99v"}\n```'),
            _make_mock_response('```json\n{"move": "b2"}\n```'),
        ]
        agent = create_agent_fn(_QuoridorHarness())

        game = quoridor_proxy.QuoridorGame({"board_size": 3, "wall_count": 1})
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {})
        self.assertEqual(result["submission"], 2)
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("no idea")
        agent = create_agent_fn(_QuoridorHarness())

        game = quoridor_proxy.QuoridorGame({"board_size": 3, "wall_count": 1})
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        with self.assertRaises(ValueError):
            agent(observation, {})
        self.assertEqual(mock_litellm.completion.call_count, 2)


if __name__ == "__main__":
    absltest.main()
