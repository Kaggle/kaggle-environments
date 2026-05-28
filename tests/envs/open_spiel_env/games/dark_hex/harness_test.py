"""Tests for Dark Hex LLM harness.

Includes regression tests for three bugs found in the May 2026 review:

1. Prompt over-permitted nominating known-opponent cells.
2. Fallback coord-scan iterated forward, picking earlier-mentioned (often
   rejected) coordinates instead of the model's final stated move.
3. ``_COORD_RE`` used ``\\s*`` between the letter and digit groups, which
   crossed newlines and captured the rendered board's column-header line
   (``"... e f\\n 1"``) as a fake ``f1`` coordinate.
"""

import json

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.dark_hex import dark_hex_proxy
from kaggle_environments.envs.open_spiel_env.games.dark_hex.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(state, game, player_id=0):
    """Build a harness-style observation dict from a Dark Hex state."""
    legal_actions = list(state.legal_actions())
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "legalActions": legal_actions,
        "legalActionStrings": [state.action_to_string(a) for a in legal_actions],
        "serializedGameAndState": pyspiel.serialize_game_and_state(
            game.__wrapped__, state.__wrapped__,
        ),
    }


# Pre-built 3x3 obs JSON used in several prompt tests.
_OBS_3X3_EMPTY = (
    '{"current_player": "x", "is_terminal": false, "winner": null, '
    '"num_rows": 3, "num_cols": 3, '
    '"board": [[".", ".", "."], [".", ".", "."], [".", ".", "."]]}'
)


class ParseResponseTest(absltest.TestCase):
    def test_parse_json_move(self):
        legal = ["a1", "b2", "c3"]
        response = '```json\n{"move": "b2"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "b2")
        self.assertEqual(result.raw_action, "b2")

    def test_parse_case_insensitive(self):
        legal = ["a1", "b2"]
        response = '```json\n{"move": "B2"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "b2")

    def test_parse_strips_whitespace_and_punctuation(self):
        legal = ["a1", "b2"]
        response = '```json\n{"move": " (b, 2). "}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "b2")

    def test_parse_bare_json(self):
        legal = ["a1", "b2"]
        response = 'I think {"move": "b2"} works.'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "b2")

    def test_parse_fallback_prose_coord(self):
        legal = ["a1", "b2", "c3"]
        response = "After some thought I'll play c3 to control the diagonal."
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "c3")

    def test_parse_no_match_returns_none(self):
        legal = ["a1", "b2"]
        response = '```json\n{"move": "z99"}\n```'
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "z99")

    def test_parse_no_signal_returns_none(self):
        legal = ["a1", "b2"]
        response = "I am not sure what to play."
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_returns_parse_result(self):
        result = parse_response('```json\n{"move": "a1"}\n```', ["a1"])
        self.assertIsInstance(result, ParseResult)

    # --- Regression: reverse-iter fallback (Issue #2) ---

    def test_fallback_prefers_last_mentioned_coord(self):
        """When fallback fires, the LAST legal coord in the prose should win.

        Models typically enumerate rejected options before stating the final
        move. Forward iteration used to pick the first-mentioned (rejected)
        candidate; reverse iteration picks the actual stated move.
        """
        legal = ["a1", "b2", "c3"]
        response = (
            "I considered a1 (too edgy) and b2 (blocked by opponent), "
            "but I'll play c3 because it controls the diagonal."
        )
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "c3")

    def test_fallback_after_illegal_json_picks_last_prose_coord(self):
        """If the JSON move is illegal, fallback should still find the right
        coord in the prose -- and pick the last one (intent), not the first."""
        legal = ["a1", "c3"]
        # JSON move b2 is illegal (not in legal); a1 is mentioned then rejected;
        # c3 is the model's actual choice. The fallback must reach c3.
        response = (
            "I rejected a1 because it's edge-bound. I'll play c3.\n"
            '```json\n{"move": "b2"}\n```'
        )
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "c3")

    # --- Regression: header-artifact regex (cross-newline \s*) ---

    def test_no_header_artifact_capture_from_rendered_board(self):
        """The fallback regex must NOT match across newlines.

        Before the fix, ``_COORD_RE = r"\\b([a-z])\\s*([0-9]+)\\b"`` matched
        the column-header letter ``f`` followed by the row label ``1`` on
        the next line, producing a phantom ``f1`` capture from any echoed
        board rendering.
        """
        # A rendered board echoed in the response. No real coordinate is
        # mentioned in prose; only the board art is present. With the fix,
        # the parser should return None instead of grabbing "f1" from
        # the header line.
        response = (
            "Here is my view of the board:\n"
            "    a b c d e f\n"
            " 1  . . . . . .\n"
            "  2  . . . . . .\n"
            "   3  . . . . . ."
        )
        legal = ["a1", "b1", "c1", "d1", "e1", "f1", "a2", "b2", "c2"]
        result = parse_response(response, legal)
        # Pre-fix: legal_action == "f1" (header artifact). Post-fix: None.
        self.assertIsNone(result.legal_action)

    def test_intended_coord_wins_over_echoed_board(self):
        """When the model echoes the board AND states a real intended coord,
        the parser should pick the stated coord, not anything from the board."""
        response = (
            "Board:\n    a b c d e f\n 1  . . . . . .\n"
            "  2  . . . . . .\n"
            "I'll play e4."
        )
        legal = ["a1", "b1", "e4", "f1"]
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "e4")


class GeneratePromptTest(absltest.TestCase):
    def test_basic_prompt_for_x(self):
        observation = {"observationString": _OBS_3X3_EMPTY, "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertIn("Dark Hex", prompt)
        self.assertIn("Player X", prompt)
        self.assertIn("TOP edge", prompt)
        self.assertIn("BOTTOM edge", prompt)
        self.assertIn("3-row", prompt)
        self.assertIn("3-column", prompt)

    def test_prompt_for_o(self):
        observation = {"observationString": _OBS_3X3_EMPTY, "playerId": 1}
        prompt = generate_prompt(observation, [])
        self.assertIn("Player O", prompt)
        self.assertIn("LEFT edge", prompt)
        self.assertIn("RIGHT edge", prompt)

    def test_move_history_rendered(self):
        observation = {"observationString": _OBS_3X3_EMPTY, "playerId": 0}
        prompt = generate_prompt(observation, ["a1", "b2"])
        self.assertIn("a1, b2", prompt)

    def test_empty_move_history(self):
        observation = {"observationString": _OBS_3X3_EMPTY, "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertIn("no moves yet", prompt)
        self.assertIn("first move", prompt)

    def test_rethink_suffix(self):
        observation = {"observationString": _OBS_3X3_EMPTY, "playerId": 0}
        prompt = generate_prompt(
            observation,
            [],
            previous_response="I play z99",
            previous_action="z99",
        )
        self.assertIn("Your previous response was", prompt)
        self.assertIn("z99", prompt)

    def test_no_rethink_on_first_attempt(self):
        observation = {"observationString": _OBS_3X3_EMPTY, "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertNotIn("Your previous response was", prompt)

    def test_board_state_rendered(self):
        obs_str = (
            '{"current_player": "x", "is_terminal": false, "winner": null, '
            '"num_rows": 3, "num_cols": 3, '
            '"board": [[".", ".", "."], [".", "x", "."], [".", ".", "."]]}'
        )
        observation = {"observationString": obs_str, "playerId": 0}
        prompt = generate_prompt(observation, [])
        # The own-x stone should appear in the rendered board.
        self.assertIn("x", prompt)

    # --- Regression: prompt must not over-permit known-opp cells (Issue #1) ---

    def test_prompt_does_not_invite_known_opp_nominations(self):
        """The pre-fix prompt said the player could nominate "any cell that
        you do not already know to contain one of your own stones (including
        unknown cells -- a collision there reveals an opponent stone and
        keeps your turn)". That phrasing invited nominating revealed-opponent
        cells, which are illegal."""
        observation = {"observationString": _OBS_3X3_EMPTY, "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertNotIn(
            "any cell that you do not already know to contain one of your own stones",
            prompt,
        )
        # "If the cell is already occupied by an opponent stone, no stone is
        # placed" was the old wording that implied known-opp cells were
        # legal nominations. The fix replaced it with "If the cell turns
        # out to hold an opponent stone".
        self.assertNotIn("If the cell is already occupied by an opponent", prompt)

    def test_prompt_states_known_cells_are_illegal(self):
        """The new phrasing must affirmatively say that known cells (own or
        revealed opponent) are not legal nominations."""
        observation = {"observationString": _OBS_3X3_EMPTY, "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertIn("no longer a legal nomination", prompt)

    def test_prompt_describes_collision_as_unknown_cell_outcome(self):
        """A collision is what happens when an *unknown* cell turns out to
        hold an opponent stone -- it is not a feature you opt into by
        targeting a known opponent cell."""
        observation = {"observationString": _OBS_3X3_EMPTY, "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertIn("turns out to hold an opponent stone", prompt)


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        observation = {
            "legalActions": [0, 4, 8],
            "legalActionStrings": ["a1", "b2", "c3"],
        }
        result = get_legal_moves(observation)
        self.assertEqual(result, {0: "a1", 4: "b2", 8: "c3"})

    def test_from_serialized_state(self):
        game = dark_hex_proxy.DarkHexGame()
        state = game.new_initial_state()
        observation = {
            "serializedGameAndState": pyspiel.serialize_game_and_state(
                game.__wrapped__, state.__wrapped__,
            ),
        }
        result = get_legal_moves(observation)
        # Default 3x3 board has 9 cells.
        self.assertEqual(len(result), 9)

    def test_empty_serialized(self):
        observation = {"serializedGameAndState": ""}
        result = get_legal_moves(observation)
        self.assertEqual(result, {})

    def test_returns_dict(self):
        observation = {"legalActions": [0], "legalActionStrings": ["a1"]}
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


class _DarkHexHarness:
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
        agent = create_agent_fn(_DarkHexHarness())
        result = agent({"step": 0, "remainingOverageTime": 60}, {})
        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            '```json\n{"move": "a1"}\n```',
        )
        agent = create_agent_fn(_DarkHexHarness())

        game = dark_hex_proxy.DarkHexGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {})
        self.assertEqual(result["submission"], 0)  # "a1" == action id 0
        self.assertEqual(result["status"], "OK")

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "z99"}\n```'),
            _make_mock_response('```json\n{"move": "b1"}\n```'),
        ]
        agent = create_agent_fn(_DarkHexHarness())

        game = dark_hex_proxy.DarkHexGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {})
        self.assertEqual(result["submission"], 1)  # "b1" == action id 1
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("no idea")
        agent = create_agent_fn(_DarkHexHarness())

        game = dark_hex_proxy.DarkHexGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        with self.assertRaises(ValueError):
            agent(observation, {})
        self.assertEqual(mock_litellm.completion.call_count, 2)


if __name__ == "__main__":
    absltest.main()
