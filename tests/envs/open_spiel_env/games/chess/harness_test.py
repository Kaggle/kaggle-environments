"""Tests for Chess LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.chess.harness import (
    _build_pgn_movetext,
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_state(*san_moves: str) -> pyspiel.State:
    """Create a pyspiel chess state after applying the given SAN moves."""
    game = pyspiel.load_game("chess")
    state = game.new_initial_state()
    for san in san_moves:
        state.apply_action(state.string_to_action(san))
    return state


def _make_observation(state: pyspiel.State, player_id: int | None = None) -> dict:
    """Build a harness-style observation dict from a pyspiel chess state."""
    game = state.get_game()
    if player_id is None:
        player_id = state.current_player()
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "currentPlayer": state.current_player(),
        "serializedGameAndState": pyspiel.serialize_game_and_state(game, state),
    }


# ---------------------------------------------------------------------------
# PGN movetext
# ---------------------------------------------------------------------------


class BuildPgnMovetextTest(absltest.TestCase):
    def test_empty_game(self):
        """Empty game (White to play) → trailing move number only."""
        state = _make_state()
        self.assertEqual(_build_pgn_movetext(state), "1.")

    def test_one_move(self):
        state = _make_state("e4")
        self.assertEqual(_build_pgn_movetext(state), "1. e4")

    def test_two_moves(self):
        """Two moves → trailing move number for White."""
        state = _make_state("e4", "e5")
        self.assertEqual(_build_pgn_movetext(state), "1. e4 e5 2.")

    def test_three_moves(self):
        state = _make_state("e4", "e5", "Nf3")
        self.assertEqual(_build_pgn_movetext(state), "1. e4 e5 2. Nf3")

    def test_four_moves(self):
        state = _make_state("e4", "e5", "Nf3", "Nc6")
        self.assertEqual(_build_pgn_movetext(state), "1. e4 e5 2. Nf3 Nc6 3.")

    def test_check_notation_included(self):
        """Moves that give check should include '+' in the SAN."""
        state = _make_state("e4", "f5")
        # Find Qh5+ (check)
        for a in state.legal_actions():
            if state.action_to_string(state.current_player(), a) == "Qh5+":
                state.apply_action(a)
                break
        mt = _build_pgn_movetext(state)
        self.assertIn("Qh5+", mt)


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class ParseResponseTest(absltest.TestCase):
    LEGAL = ["e4", "d4", "Nf3", "Nc3", "a3", "a4", "b3", "b4", "c3", "c4"]

    def test_final_answer_exact_match(self):
        result = parse_response("Final Answer: e4", self.LEGAL)
        self.assertEqual(result.legal_action, "e4")
        self.assertEqual(result.raw_action, "e4")

    def test_final_answer_with_reasoning(self):
        response = "I think the best opening is the King's Pawn. Final Answer: e4"
        result = parse_response(response, self.LEGAL)
        self.assertEqual(result.legal_action, "e4")

    def test_latex_boxed_move(self):
        result = parse_response("Final Answer: \\boxed{Nf3}", self.LEGAL)
        self.assertEqual(result.legal_action, "Nf3")

    def test_backtick_wrapped_move(self):
        result = parse_response("Final Answer: `e4`", self.LEGAL)
        self.assertEqual(result.legal_action, "e4")

    def test_asterisk_wrapped_move(self):
        result = parse_response("Final Answer: **d4**", self.LEGAL)
        self.assertEqual(result.legal_action, "d4")

    def test_colon_fallback(self):
        """Falls back to last ':' when 'Final Answer:' is absent."""
        result = parse_response("My move is: e4", self.LEGAL)
        self.assertEqual(result.legal_action, "e4")

    def test_is_fallback(self):
        """Falls back to 'is' when other tags are absent or earlier."""
        result = parse_response("My chosen move is e4", self.LEGAL)
        self.assertEqual(result.legal_action, "e4")

    def test_check_symbol_stripped(self):
        """Matches 'Qh5' to legal move 'Qh5+'."""
        legal_with_check = ["Qh5+", "e4", "Nf3"]
        result = parse_response("Final Answer: Qh5", legal_with_check)
        self.assertEqual(result.legal_action, "Qh5+")
        self.assertEqual(result.raw_action, "Qh5")

    def test_move_number_prefix_stripped(self):
        """Strips leading move number like '1. e4' → 'e4'."""
        result = parse_response("Final Answer: 1. e4", self.LEGAL)
        self.assertEqual(result.legal_action, "e4")

    def test_move_number_black_prefix_stripped(self):
        """Strips '1... e5' style prefix."""
        legal = ["e5", "d5", "Nf6"]
        result = parse_response("Final Answer: 1...e5", legal)
        self.assertEqual(result.legal_action, "e5")

    def test_case_insensitive_match(self):
        result = parse_response("Final Answer: NF3", self.LEGAL)
        self.assertEqual(result.legal_action, "Nf3")

    def test_no_match_returns_raw(self):
        result = parse_response("Final Answer: Zz9", self.LEGAL)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "Zz9")

    def test_no_tag_returns_none(self):
        result = parse_response("I have no idea what to play", self.LEGAL)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_empty_after_tag_returns_none(self):
        result = parse_response("Final Answer: ", self.LEGAL)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_returns_parse_result(self):
        result = parse_response("Final Answer: e4", self.LEGAL)
        self.assertIsInstance(result, ParseResult)

    def test_en_passant_suffix_stripped(self):
        """Strips 'ep' en passant annotation."""
        legal = ["exd6", "e5"]
        result = parse_response("Final Answer: exd6ep", legal)
        self.assertEqual(result.legal_action, "exd6")

    def test_html_tags_stripped(self):
        result = parse_response("Final Answer: <b>e4</b>", self.LEGAL)
        self.assertEqual(result.legal_action, "e4")

    def test_castling(self):
        legal = ["O-O", "Kf1", "Ke2"]
        result = parse_response("Final Answer: O-O", legal)
        self.assertEqual(result.legal_action, "O-O")


# ---------------------------------------------------------------------------
# generate_prompt
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def test_initial_position_white(self):
        state = _make_state()
        obs = _make_observation(state)
        prompt = generate_prompt(obs, [])
        self.assertIn("Let's play chess", prompt)
        self.assertIn("Forsyth-Edwards Notation (FEN) notation", prompt)
        self.assertIn("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", prompt)
        self.assertIn("player White", prompt)
        self.assertIn("standard algebraic notation (SAN)", prompt)
        self.assertIn("Final Answer: X", prompt)

    def test_black_player(self):
        state = _make_state("e4")
        obs = _make_observation(state)
        prompt = generate_prompt(obs, [])
        self.assertIn("player Black", prompt)
        self.assertIn("1. e4", prompt)

    def test_move_history_in_prompt(self):
        state = _make_state("e4", "e5", "Nf3")
        obs = _make_observation(state)
        prompt = generate_prompt(obs, [])
        self.assertIn("1. e4 e5 2. Nf3", prompt)

    def test_no_rethink_on_first_attempt(self):
        state = _make_state("e4")
        obs = _make_observation(state)
        prompt = generate_prompt(obs, [])
        self.assertNotIn("previously suggested", prompt)
        self.assertNotIn("not parsable", prompt)

    def test_rethink_unparseable(self):
        state = _make_state("e4")
        obs = _make_observation(state)
        prompt = generate_prompt(
            obs,
            [],
            previous_response="I love chess so much!",
            previous_action=None,
        )
        self.assertIn("not parsable", prompt)
        self.assertIn("I love chess so much!", prompt)

    def test_rethink_illegal(self):
        state = _make_state("e4")
        obs = _make_observation(state)
        prompt = generate_prompt(
            obs,
            [],
            previous_response="Final Answer: Qd8",
            previous_action="Qd8",
        )
        self.assertIn("Qd8", prompt)
        self.assertIn("illegal move", prompt)
        self.assertNotIn("not parsable", prompt)

    def test_prompt_matches_game_arena_format(self):
        """Verify exact format: template matches GameArena NO_LEGAL_ACTIONS_RETHINK_APPENDED."""
        state = _make_state("e4", "e5")
        obs = _make_observation(state)
        prompt = generate_prompt(obs, [])

        # Should be a single trailing newline (from \n{rethink_prompt} where rethink_prompt="")
        self.assertTrue(prompt.endswith("standard algebraic notation (SAN).\n"))

    def test_rethink_illegal_exact_format(self):
        """Verify rethink text matches GameArena RETHINK_WITH_ENV_ILLEGAL."""
        state = _make_state("e4")
        obs = _make_observation(state)
        prompt = generate_prompt(
            obs,
            [],
            previous_response="Final Answer: Nf4",
            previous_action="Nf4",
        )
        expected_suffix = (
            "Your previously suggested move was: Nf4, which is an illegal move.\n"
            "Please think carefully and generate a new and legal move.\n"
        )
        self.assertTrue(prompt.endswith(expected_suffix))

    def test_rethink_unparseable_exact_format(self):
        """Verify rethink text matches GameArena RETHINK_WITH_ENV_UNPARSABLE."""
        state = _make_state("e4")
        obs = _make_observation(state)
        prompt = generate_prompt(
            obs,
            [],
            previous_response="gibberish text",
            previous_action=None,
        )
        expected_suffix = (
            "Your previously suggested move was not parsable.\n"
            "Please think carefully and generate a new and legal move. "
            "Your previous response was:\n"
            "gibberish text\n"
        )
        self.assertTrue(prompt.endswith(expected_suffix))


# ---------------------------------------------------------------------------
# get_legal_moves
# ---------------------------------------------------------------------------


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        obs = {
            "legalActions": [2426, 1842],
            "legalActionStrings": ["e4", "d4"],
        }
        result = get_legal_moves(obs)
        self.assertEqual(result, {2426: "e4", 1842: "d4"})

    def test_from_serialized_state(self):
        state = _make_state()
        obs = _make_observation(state)
        result = get_legal_moves(obs)
        # Initial position has 20 legal moves
        self.assertEqual(len(result), 20)
        # Verify a known action
        self.assertIn("e4", result.values())

    def test_empty_serialized(self):
        obs = {"serializedGameAndState": ""}
        result = get_legal_moves(obs)
        self.assertEqual(result, {})

    def test_returns_dict_int_str(self):
        state = _make_state()
        obs = _make_observation(state)
        result = get_legal_moves(obs)
        self.assertIsInstance(result, dict)
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, str)


# ---------------------------------------------------------------------------
# Agent integration (mocked LLM)
# ---------------------------------------------------------------------------


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


class _ChessHarness:
    def get_legal_moves(self, observation):
        return get_legal_moves(observation)

    def make_prompt(self, observation, move_history, previous_response=None, previous_action=None):
        return generate_prompt(observation, move_history, previous_response, previous_action)

    def parse_response(self, response, legal_action_strings, *, observation=None):
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
        agent = create_agent_fn(_ChessHarness())
        result = agent({"step": 0, "remainingOverageTime": 60}, {})
        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            "I'll play the King's Pawn opening. Final Answer: e4"
        )
        agent = create_agent_fn(_ChessHarness())
        state = _make_state()
        obs = _make_observation(state)
        result = agent(obs, {})
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["actionString"], "e4")
        # Verify submission is the correct action ID
        self.assertEqual(
            state.action_to_string(state.current_player(), result["submission"]),
            "e4",
        )

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_illegal_then_succeed(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.side_effect = [
            _make_mock_response("Final Answer: Qd8"),  # illegal
            _make_mock_response("Final Answer: e4"),  # legal
        ]
        agent = create_agent_fn(_ChessHarness())
        state = _make_state()
        obs = _make_observation(state)
        result = agent(obs, {})
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["actionString"], "e4")
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_unparseable_then_succeed(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.side_effect = [
            _make_mock_response("I love chess!"),  # no Final Answer tag
            _make_mock_response("Final Answer: d4"),  # legal
        ]
        agent = create_agent_fn(_ChessHarness())
        state = _make_state()
        obs = _make_observation(state)
        result = agent(obs, {})
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["actionString"], "d4")
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_all_retries_fail(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("I have no idea what to play.")
        agent = create_agent_fn(_ChessHarness())
        state = _make_state()
        obs = _make_observation(state)
        with self.assertRaises(ValueError):
            agent(obs, {})
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_call_details_present(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("Final Answer: e4")
        agent = create_agent_fn(_ChessHarness())
        state = _make_state()
        obs = _make_observation(state)
        result = agent(obs, {})
        self.assertIn("call_details", result)
        self.assertLen(result["call_details"], 1)
        cd = result["call_details"][0]
        self.assertEqual(cd["generation_tokens"], 20)
        self.assertEqual(cd["prompt_tokens"], 10)
        self.assertIn("prompt", cd)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_not_our_turn_returns_inactive(self, mock_litellm):
        """Agent returns INACTIVE when it's not our turn."""
        mock_litellm.drop_params = True
        agent = create_agent_fn(_ChessHarness())
        state = _make_state()  # White to play
        obs = _make_observation(state, player_id=0)  # We are Black
        obs["currentPlayer"] = 1  # But White is playing
        result = agent(obs, {})
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_terminal_returns_inactive(self, mock_litellm):
        mock_litellm.drop_params = True
        agent = create_agent_fn(_ChessHarness())
        obs = {"isTerminal": True, "playerId": 0, "currentPlayer": 0}
        result = agent(obs, {})
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_check_symbol_handled_in_integration(self, mock_litellm):
        """LLM omits '+' but harness matches to 'Qh5+'."""
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("Final Answer: Qh5")
        agent = create_agent_fn(_ChessHarness())
        state = _make_state("e4", "f5")
        obs = _make_observation(state)
        result = agent(obs, {})
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["actionString"], "Qh5+")


if __name__ == "__main__":
    absltest.main()
