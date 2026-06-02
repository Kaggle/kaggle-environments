"""Tests for Game of the Amazons LLM harness.

Includes regression tests for the bugs found in the May 2026 review:

1. Fallback coord-scan iterated forward, picking earlier-mentioned (often
   rejected) cells instead of the model's final stated move.
2. ``_CELL_RE`` used ``\\s*`` between the letter and digit groups, which
   crossed newlines and captured the rendered board's column-header line
   (``"... i j\\n 1"``) as a fake ``j1`` cell.

Plus the June 2026 audit follow-ups:

3. TO and SHOOT prompts hid the in-progress queen's source square -- the
   model had to infer it from the bare move-history list, often wrongly.
4. Move history was a flat list of cells with no phase / player markers,
   making it ambiguous which cell was a "from" vs a "to" vs a barrier.
5. Prompt vocabulary tripped some model-API safety filters (amazons /
   shoot / arrow / burned) -- sterilized to queens / barriers / blocked.
"""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.amazons import amazons_proxy
from kaggle_environments.envs.open_spiel_env.games.amazons.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(state, game, player_id=0):
    """Build a harness-style observation dict from an Amazons state."""
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


# Minimal 10x10 obs JSON: empty board, "from" phase, X to move. Used as a
# stand-in for prompt-shape tests where the actual board contents don't
# matter.
_OBS_10X10_EMPTY = (
    '{"current_player": "x", "phase": "from", "is_terminal": false, '
    '"winner": null, "num_rows": 10, "num_cols": 10, "move_number": 0, '
    '"board": ' + str([["."] * 10 for _ in range(10)]).replace("'", '"') + "}"
)


class ParseResponseTest(absltest.TestCase):
    def test_parse_json_move(self):
        legal = ["a7", "j7", "d10", "g10"]
        response = '```json\n{"move": "a7"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "a7")
        self.assertEqual(result.raw_action, "a7")

    def test_parse_case_insensitive(self):
        legal = ["a7", "j7"]
        response = '```json\n{"move": "A7"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "a7")

    def test_parse_two_digit_row(self):
        """Amazons defaults to 10x10, so row 10 is a valid two-digit number."""
        legal = ["d10", "g10"]
        response = '```json\n{"move": "d10"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "d10")

    def test_parse_bare_json(self):
        legal = ["a7", "j7"]
        response = 'My choice is {"move": "j7"} for sure.'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "j7")

    def test_unfenced_prose_cell_triggers_rethink(self):
        # No JSON. Return None and let the rethink loop ask the model to
        # use the required JSON format.
        legal = ["a7", "j7", "d10", "g10"]
        response = "I'll start by moving the amazon at g10."
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_illegal_json_does_not_ghost_substitute_from_prose_v2(self):
        # Distinct from the v1 test: emphasises that even when the prose
        # contains a clearly-stated and legal "intent" cell, the parser
        # must trust the JSON answer (which is illegal here) and surface
        # raw_action -- never silently submit the prose cell.
        legal = ["a7", "d10", "g10"]
        response = (
            "I'll play g10 because it controls the centre.\n"
            '```json\n{"move": "j7"}\n```'
        )
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "j7")

    def test_parse_no_match_returns_none(self):
        legal = ["a7", "j7"]
        response = '```json\n{"move": "z99"}\n```'
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "z99")

    def test_parse_no_signal_returns_none(self):
        legal = ["a7", "j7"]
        response = "I am not sure what to play."
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_returns_parse_result(self):
        result = parse_response('```json\n{"move": "a7"}\n```', ["a7"])
        self.assertIsInstance(result, ParseResult)

    # --- Regression: reverse-iter fallback ---

    def test_prose_only_response_triggers_rethink(self):
        # No structured JSON. The parser must NOT guess at intent from a
        # cell mentioned in the prose -- return None and let rethink ask
        # the model to use the required JSON format.
        legal = ["a7", "j7", "g10"]
        response = (
            "I considered a7 (too cornered) and j7 (blocked by an arrow), "
            "but I'll move g10 to keep mobility."
        )
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_illegal_json_does_not_ghost_substitute_from_prose(self):
        # The model's JSON answer (j7) is illegal. The parser must NOT
        # silently substitute g10 or a7 from the prose -- that's the
        # ghost antipattern. Surface raw_action so the rethink loop fires.
        legal = ["a7", "g10"]
        response = (
            "I thought about a7 but it's exposed. My move is g10.\n"
            '```json\n{"move": "j7"}\n```'
        )
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "j7")

    # --- Regression: header-artifact regex (cross-newline \s*) ---

    def test_no_header_artifact_capture_from_rendered_board(self):
        """The fallback regex must NOT match across newlines.

        Before the fix, ``_CELL_RE = r"\\b([a-zA-Z])\\s*([1-9][0-9]?)\\b"``
        matched the column-header letter ``j`` followed by the row label
        ``1`` on the next line, producing a phantom ``j1`` capture from
        the echoed 10x10 board rendering.
        """
        response = (
            "Here is the board:\n"
            "   a b c d e f g h i j\n"
            " 1 . . . . . . . . . .\n"
            " 2 . . . . . . . . . .\n"
            " 3 . . . . . . . . . ."
        )
        # All header-edge cells are in the legal set, so a forward-iter +
        # cross-newline regex would pick j1. With the fix, no real
        # coordinate appears in prose, so the parser returns None.
        legal = ["a1", "j1", "a2", "j2", "a3", "j3"]
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)

    def test_echoed_board_plus_prose_intent_triggers_rethink(self):
        # The model echoes the board and states a prose intent ("g10") but
        # no JSON answer. The parser must NOT guess, even when the prose
        # intent is unambiguous -- return None so the rethink loop asks
        # the model to wrap its answer in JSON.
        response = (
            "Board:\n   a b c d e f g h i j\n 1 . . . . . . . . . .\n"
            " 2 . . . . . . . . . .\n"
            "I'll play g10."
        )
        legal = ["a1", "j1", "g10"]
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)


class GeneratePromptTest(absltest.TestCase):
    def test_basic_prompt_for_x(self):
        observation = {"observationString": _OBS_10X10_EMPTY, "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertIn("Player 1", prompt)  # was "Black" before sterilization
        self.assertIn("(X)", prompt)
        self.assertIn("10x10", prompt)
        self.assertIn("PICK QUEEN", prompt)  # FROM phase label
        self.assertIn("queens", prompt.lower())
        self.assertIn("barrier", prompt.lower())
        # Sterilization regression: weapon-adjacent vocabulary must NOT
        # appear -- some model APIs refuse the prompt otherwise.
        self.assertNotIn("amazon", prompt.lower())
        self.assertNotIn("shoot", prompt.lower())
        self.assertNotIn("arrow", prompt.lower())
        self.assertNotIn("burned", prompt.lower())

    def test_prompt_for_o(self):
        obs_str = _OBS_10X10_EMPTY.replace('"current_player": "x"', '"current_player": "o"')
        observation = {"observationString": obs_str, "playerId": 1}
        prompt = generate_prompt(observation, [])
        self.assertIn("Player 2", prompt)
        self.assertIn("(O)", prompt)

    def test_from_phase_legality_hint(self):
        """FROM-phase instruction must warn about trapped queens."""
        observation = {"observationString": _OBS_10X10_EMPTY, "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertIn("empty neighbouring square", prompt)

    def test_phase_to_instruction_names_source_square(self):
        """TO prompt must explicitly name the queen's source square.

        Before the fix it said "you picked up an amazon" without revealing
        which one; the model had to infer from move_history. Regression
        for audit finding #1.
        """
        obs_str = _OBS_10X10_EMPTY.replace(
            '"phase": "from"', '"phase": "to", "from_action": 60',
        )
        observation = {"observationString": obs_str, "playerId": 0}
        prompt = generate_prompt(observation, ["a7"])
        self.assertIn("MOVE QUEEN", prompt)
        self.assertIn("a7", prompt)  # source square is named in the body
        self.assertIn("queen-move", prompt.lower())

    def test_phase_shoot_instruction_names_both_squares(self):
        """SHOOT prompt must name both from and to squares this turn.

        Regression for audit finding #1: a player with 4 queens of the
        same colour couldn't tell which one had just moved.
        """
        obs_str = _OBS_10X10_EMPTY.replace(
            '"phase": "from"',
            '"phase": "shoot", "from_action": 60, "to_action": 63',
        )
        observation = {"observationString": obs_str, "playerId": 0}
        prompt = generate_prompt(observation, ["a7", "d7"])
        self.assertIn("PLACE BARRIER", prompt)
        self.assertIn("a7", prompt)
        self.assertIn("d7", prompt)
        self.assertIn("barrier", prompt.lower())

    def test_to_phase_falls_back_to_move_history(self):
        """When proxy didn't surface from_action, infer from history."""
        # Note: no "from_action" key in obs -- harness has to fall back to
        # move_history[-1].
        obs_str = _OBS_10X10_EMPTY.replace('"phase": "from"', '"phase": "to"')
        observation = {"observationString": obs_str, "playerId": 0}
        prompt = generate_prompt(observation, ["g10"])
        self.assertIn("g10", prompt)

    def test_move_history_grouped_per_turn(self):
        """Each completed turn renders as `X: a -> b, barrier c`.

        Regression for audit finding #2: bare flat list was ambiguous
        about which cells were from / to / barriers.
        """
        observation = {"observationString": _OBS_10X10_EMPTY, "playerId": 0}
        prompt = generate_prompt(
            observation,
            ["a7", "a8", "b8", "d1", "d2", "e2"],
        )
        self.assertIn("X: a7 -> a8, barrier b8", prompt)
        self.assertIn("O: d1 -> d2, barrier e2", prompt)

    def test_move_history_skips_partial_turn(self):
        """Partial turn at the tail is omitted (already surfaced via prompt)."""
        observation = {"observationString": _OBS_10X10_EMPTY, "playerId": 0}
        prompt = generate_prompt(
            observation,
            # X has a complete turn; O has started but not finished.
            ["a7", "a8", "b8", "d1"],
        )
        self.assertIn("X: a7 -> a8, barrier b8", prompt)
        # The partial "O: d1 (moving...)" line would duplicate info that
        # the phase instruction already conveys; skip it.
        self.assertNotIn("O:", prompt)

    def test_empty_move_history(self):
        observation = {"observationString": _OBS_10X10_EMPTY, "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertIn("no completed turns yet", prompt)

    def test_rethink_suffix(self):
        observation = {"observationString": _OBS_10X10_EMPTY, "playerId": 0}
        prompt = generate_prompt(
            observation,
            [],
            previous_response="I play z99",
            previous_action="z99",
        )
        self.assertIn("You suggested", prompt)  # ILLEGAL leads with action
        self.assertIn("z99", prompt)

    def test_no_rethink_on_first_attempt(self):
        observation = {"observationString": _OBS_10X10_EMPTY, "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertNotIn("Your previous response was", prompt)


class GetLegalMovesTest(absltest.TestCase):
    def test_from_serialized_state(self):
        game = amazons_proxy.AmazonsGame()
        state = game.new_initial_state()
        observation = {
            "observationString": state.observation_string(0),
            "serializedGameAndState": pyspiel.serialize_game_and_state(
                game.__wrapped__, state.__wrapped__,
            ),
        }
        result = get_legal_moves(observation)
        # Default 10x10 board: X has 4 amazons, so 4 legal "from" choices.
        self.assertEqual(result, {60: "a7", 69: "j7", 93: "d10", 96: "g10"})

    def test_empty_observation_returns_empty(self):
        observation = {"observationString": "", "serializedGameAndState": ""}
        result = get_legal_moves(observation)
        self.assertEqual(result, {})

    def test_returns_dict(self):
        game = amazons_proxy.AmazonsGame()
        state = game.new_initial_state()
        observation = {
            "observationString": state.observation_string(0),
            "legalActions": list(state.legal_actions()),
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


class _AmazonsHarness:
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
        agent = create_agent_fn(_AmazonsHarness())
        result = agent({"step": 0, "remainingOverageTime": 60}, {})
        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            '```json\n{"move": "a7"}\n```',
        )
        agent = create_agent_fn(_AmazonsHarness())

        game = amazons_proxy.AmazonsGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {})
        self.assertEqual(result["submission"], 60)  # "a7" == action id 60
        self.assertEqual(result["status"], "OK")

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "z99"}\n```'),
            _make_mock_response('```json\n{"move": "j7"}\n```'),
        ]
        agent = create_agent_fn(_AmazonsHarness())

        game = amazons_proxy.AmazonsGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {})
        self.assertEqual(result["submission"], 69)  # "j7" == action id 69
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("no idea")
        agent = create_agent_fn(_AmazonsHarness())

        game = amazons_proxy.AmazonsGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        with self.assertRaises(ValueError):
            agent(observation, {})
        self.assertEqual(mock_litellm.completion.call_count, 2)


if __name__ == "__main__":
    absltest.main()
