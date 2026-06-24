"""Tests for the Ultimate Tic-Tac-Toe LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.ultimate_tic_tac_toe import (
    ultimate_tic_tac_toe_proxy,
)
from kaggle_environments.envs.open_spiel_env.games.ultimate_tic_tac_toe.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(
    state: ultimate_tic_tac_toe_proxy.UltimateTicTacToeState,
    game: ultimate_tic_tac_toe_proxy.UltimateTicTacToeGame,
    player_id: int = 0,
) -> dict:
    """Build a harness-style observation dict from a proxy state."""
    legal = list(state.legal_actions())
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "currentPlayer": int(state.current_player()),
        "isTerminal": state.is_terminal(),
        "legalActions": legal,
        "legalActionStrings": [state.action_to_string(int(state.current_player()), a) for a in legal],
        "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
    }


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class ParseResponseTest(absltest.TestCase):
    legal_subgrid = ["Choose local board 0", "Choose local board 1", "Choose local board 2"]
    legal_cell = ["Local board 4: o(0,0)", "Local board 4: o(0,1)", "Local board 4: o(1,1)", "Local board 4: o(2,2)"]

    def test_parse_json_block(self):
        result = parse_response('```json\n{"move": "1"}\n```', self.legal_subgrid)
        self.assertEqual(result.legal_action, "Choose local board 1")
        self.assertEqual(result.raw_action, "1")

    def test_parse_bare_json(self):
        result = parse_response('I think {"move": "1,1"} is best.', self.legal_cell)
        self.assertEqual(result.legal_action, "Local board 4: o(1,1)")

    def test_prose_only_response_triggers_rethink(self):
        result = parse_response("I will play 1,1 this turn.", self.legal_cell)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_case_insensitive(self):
        # Exact matching should work case-insensitively
        result = parse_response('```json\n{"move": "Local Board 4: O(1,1)"}\n```', self.legal_cell)
        self.assertEqual(result.legal_action, "Local board 4: o(1,1)")

    def test_parse_illegal_move_returns_raw(self):
        result = parse_response('```json\n{"move": "2,1"}\n```', self.legal_cell)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "2,1")

    def test_parse_no_match_returns_none(self):
        result = parse_response("I have no idea.", self.legal_cell)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_returns_parse_result_type(self):
        result = parse_response('```json\n{"move": "1,1"}\n```', self.legal_cell)
        self.assertIsInstance(result, ParseResult)

    def test_illegal_json_does_not_ghost_substitute_from_prose(self):
        response = 'I considered 1,1 but ruled it out.\n```json\n{"move": "9,9"}\n```'
        result = parse_response(response, self.legal_cell)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "9,9")

    def test_coordinate_matching_variants(self):
        # Coordinates in parentheses
        result = parse_response('```json\n{"move": "(1,1)"}\n```', self.legal_cell)
        self.assertEqual(result.legal_action, "Local board 4: o(1,1)")

        # Coordinates with spaces/dash
        result = parse_response('```json\n{"move": "1-1"}\n```', self.legal_cell)
        self.assertEqual(result.legal_action, "Local board 4: o(1,1)")

        # Coordinates with dot
        result = parse_response('```json\n{"move": "1.1"}\n```', self.legal_cell)
        self.assertEqual(result.legal_action, "Local board 4: o(1,1)")

    def test_cell_index_matching(self):
        # Cell 4 -> (1,1)
        result = parse_response('```json\n{"move": "4"}\n```', self.legal_cell)
        self.assertEqual(result.legal_action, "Local board 4: o(1,1)")

        # Cell 8 -> (2,2)
        result = parse_response('```json\n{"move": "8"}\n```', self.legal_cell)
        self.assertEqual(result.legal_action, "Local board 4: o(2,2)")

    def test_choose_subgrid_matching(self):
        # Just digit
        result = parse_response('```json\n{"move": "2"}\n```', self.legal_subgrid)
        self.assertEqual(result.legal_action, "Choose local board 2")

        # Subgrid text
        result = parse_response('```json\n{"move": "subgrid 1"}\n```', self.legal_subgrid)
        self.assertEqual(result.legal_action, "Choose local board 1")


# ---------------------------------------------------------------------------
# generate_prompt
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game = ultimate_tic_tac_toe_proxy.UltimateTicTacToeGame()
        self.state = self.game.new_initial_state()

    def test_basic_prompt_contents(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Ultimate Tic-Tac-Toe", prompt)
        self.assertIn("Player 0", prompt)
        self.assertIn("'x'", prompt)
        self.assertIn("Local Board 0", prompt)
        self.assertIn("Local Board Winners", prompt)

    def test_phase_instructions_choose_subgrid(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("choose ANY active local board", prompt)
        self.assertIn("<subgrid_index>", prompt)

    def test_phase_instructions_choose_cell(self):
        # Player chooses subgrid 0
        self.state.apply_action(0)
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("must play in Local Board 0", prompt)
        self.assertIn("<row>,<col>", prompt)

    def test_move_history_rendered(self):
        self.state.apply_action(0)
        self.state.apply_action(4)
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Reconstructed history covers all players and shows board choices and cell indices
        self.assertIn("Player 0 (x): chose board 0", prompt)
        self.assertIn("Player 0 (x): board 0 cell (1,1) [idx 4]", prompt)

    def test_rethink_suffix(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [], previous_response="I'll play subgrid 9", previous_action="9")
        self.assertIn("You suggested", prompt)
        self.assertIn("9", prompt)
        self.assertIn("not a legal move", prompt)


# ---------------------------------------------------------------------------
# get_legal_moves
# ---------------------------------------------------------------------------


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        obs = {
            "legalActions": [0, 1],
            "legalActionStrings": ["Choose local board 0", "Choose local board 1"],
        }
        result = get_legal_moves(obs)
        self.assertEqual(result, {0: "Choose local board 0", 1: "Choose local board 1"})

    def test_from_serialized_state(self):
        game = ultimate_tic_tac_toe_proxy.UltimateTicTacToeGame()
        state = game.new_initial_state()
        obs = {
            "playerId": 0,
            "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
        }
        result = get_legal_moves(obs)
        self.assertEqual(len(result), 9)
        self.assertEqual(result[0], "Choose local board 0")


# ---------------------------------------------------------------------------
# create_agent_fn integration
# ---------------------------------------------------------------------------


class _UltimateTicTacToeHarness:
    """Adapter wrapping module-level functions into the GameHarness protocol."""

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

    def parse_response(self, response, legal_action_strings, *, observation=None):
        return parse_response(response, legal_action_strings)


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


def _make_mock_response(content: str):
    """Build a streaming-style mock LLM response."""
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
        agent = create_agent_fn(_UltimateTicTacToeHarness())

        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        game = ultimate_tic_tac_toe_proxy.UltimateTicTacToeGame()
        state = game.new_initial_state()
        mock_litellm.completion.return_value = _make_mock_response('```json\n{"move": "0"}\n```')
        agent = create_agent_fn(_UltimateTicTacToeHarness())

        obs = _make_observation(state, game, player_id=0)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], "Choose local board 0")
        self.assertEqual(result["status"], "OK")

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        game = ultimate_tic_tac_toe_proxy.UltimateTicTacToeGame()
        state = game.new_initial_state()
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "9"}\n```'),
            _make_mock_response('```json\n{"move": "0"}\n```'),
        ]
        agent = create_agent_fn(_UltimateTicTacToeHarness())

        obs = _make_observation(state, game, player_id=0)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], "Choose local board 0")
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("I cannot decide.")
        agent = create_agent_fn(_UltimateTicTacToeHarness())

        game = ultimate_tic_tac_toe_proxy.UltimateTicTacToeGame()
        state = game.new_initial_state()
        obs = _make_observation(state, game, player_id=0)

        with self.assertRaises(ValueError):
            agent(obs, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_short_game_via_agent_fns(self, mock_litellm):
        mock_litellm.drop_params = True

        game = ultimate_tic_tac_toe_proxy.UltimateTicTacToeGame()
        state = game.new_initial_state()

        def fake_completion(*, model, messages, **kwargs):
            del model, kwargs
            legal_actions = state.legal_actions()
            first = state.action_to_string(state.current_player(), legal_actions[0])
            # If it starts with "Choose local board X", yield the subgrid index
            if first.startswith("Choose local board"):
                val = first.split()[-1]
            else:
                # E.g. "Local board 0: x(1,1)" -> extract "1,1"
                coords = first.split()[-1].split("(")[-1].rstrip(")")
                val = coords
            return _make_mock_response(f'```json\n{{"move": "{val}"}}\n```')

        mock_litellm.completion.side_effect = fake_completion
        agent_p0 = create_agent_fn(_UltimateTicTacToeHarness())
        agent_p1 = create_agent_fn(_UltimateTicTacToeHarness())

        for _ in range(15):
            if state.is_terminal():
                break
            cp = int(state.current_player())
            agent = agent_p0 if cp == 0 else agent_p1
            obs = _make_observation(state, game, player_id=cp)
            result = agent(obs, {})
            self.assertEqual(result["status"], "OK")
            state.apply_action(result["submission"])

        self.assertGreater(state.move_number(), 0)


if __name__ == "__main__":
    absltest.main()
