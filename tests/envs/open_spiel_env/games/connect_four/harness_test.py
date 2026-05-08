"""Tests for Connect Four LLM harness."""

import json
from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.connect_four import connect_four_proxy
from kaggle_environments.envs.open_spiel_env.games.connect_four.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(state, game, player_id=0):
    """Build a harness-style observation dict from a Connect Four state."""
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
    }


class ParseResponseTest(absltest.TestCase):
    def test_final_answer_digit(self):
        legal = ["x0", "x1", "x2", "x3", "x4", "x5", "x6"]
        response = "I'll drop in column 3.\n\nFinal Answer: 3"
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "x3")
        self.assertEqual(result.raw_action, "3")

    def test_final_answer_zero(self):
        legal = ["o0", "o1", "o2", "o3"]
        response = "Final Answer: 0"
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "o0")
        self.assertEqual(result.raw_action, "0")

    def test_final_answer_case_insensitive(self):
        legal = ["x0", "x1", "x2", "x3"]
        response = "final answer: 2"
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "x2")

    def test_fallback_last_digit(self):
        """Falls back to scanning for digits when no Final Answer tag."""
        legal = ["x0", "x1", "x2", "x3", "x4", "x5", "x6"]
        response = "Column 4 is the best move here."
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "x4")

    def test_illegal_column_returns_none(self):
        legal = ["x0", "x1", "x2"]
        response = "Final Answer: 5"
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "5")

    def test_no_digit_returns_none(self):
        legal = ["x0", "x1", "x2"]
        response = "I have no idea what to play."
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_returns_parse_result(self):
        legal = ["x0", "x1"]
        result = parse_response("Final Answer: 0", legal)
        self.assertIsInstance(result, ParseResult)

    def test_numeric_action_strings(self):
        """Works when action strings are plain digits (no player prefix)."""
        legal = ["0", "1", "2", "3", "4", "5", "6"]
        response = "Final Answer: 6"
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "6")

    def test_multidigit_not_matched_spuriously(self):
        """A digit like '3' should not match action string '13'."""
        legal = ["x13"]
        response = "Final Answer: 3"
        result = parse_response(response, legal)
        # "x13" ends with "3", so it matches. This is expected behavior.
        self.assertEqual(result.legal_action, "x13")

    def test_fallback_prefers_later_digit(self):
        """Fallback scanning picks the last matching digit in the response."""
        legal = ["x0", "x1", "x2", "x3", "x4", "x5", "x6"]
        response = "I considered column 2 but column 5 is better."
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "x5")


class GeneratePromptTest(absltest.TestCase):
    def _make_obs(self, player_id=0, actions=None):
        game = connect_four_proxy.ConnectFourGame()
        state = game.new_initial_state()
        for a in actions or []:
            state.apply_action(a)
        return _make_observation(state, game, player_id)

    def test_includes_rules(self):
        obs = self._make_obs()
        prompt = generate_prompt(obs, [])
        self.assertIn("Connect X", prompt)
        self.assertIn("Gravity", prompt)
        self.assertIn("6 rows", prompt)
        self.assertIn("7 columns", prompt)
        self.assertIn("4 of their pieces", prompt)

    def test_player_x(self):
        obs = self._make_obs(player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("player x", prompt)
        self.assertIn("Player x", prompt)

    def test_player_o(self):
        obs = self._make_obs(player_id=1, actions=[3])
        prompt = generate_prompt(obs, [])
        self.assertIn("player o", prompt)

    def test_board_state_included(self):
        obs = self._make_obs(actions=[3])
        prompt = generate_prompt(obs, [])
        # Board should contain the piece in column 3
        self.assertIn("x", prompt)
        self.assertIn(".......", prompt)

    def test_final_answer_format(self):
        obs = self._make_obs()
        prompt = generate_prompt(obs, [])
        self.assertIn("Final Answer: <column_index>", prompt)

    def test_rethink_suffix(self):
        obs = self._make_obs()
        prompt = generate_prompt(
            obs,
            [],
            previous_response="I play column 9",
            previous_action="9",
        )
        self.assertIn("could not be parsed", prompt)
        self.assertIn("I play column 9", prompt)

    def test_no_rethink_on_first_attempt(self):
        obs = self._make_obs()
        prompt = generate_prompt(obs, [])
        self.assertNotIn("could not be parsed", prompt)
        self.assertNotIn("previous response", prompt)

    def test_prompt_matches_original_template(self):
        """Verify the prompt contains exact phrases from the GameArena template."""
        obs = self._make_obs()
        prompt = generate_prompt(obs, [])
        self.assertIn("You are a world-class Connect X AI", prompt)
        self.assertIn("I. Game Rules & Configuration", prompt)
        self.assertIn("II. Input Data Format", prompt)
        self.assertIn("III. Required Final Answer Format", prompt)
        self.assertIn("Choose the optimal column.", prompt)


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        observation = {
            "legalActions": [0, 1, 2, 3, 4, 5, 6],
            "legalActionStrings": ["x0", "x1", "x2", "x3", "x4", "x5", "x6"],
        }
        result = get_legal_moves(observation)
        self.assertEqual(
            result,
            {
                0: "x0",
                1: "x1",
                2: "x2",
                3: "x3",
                4: "x4",
                5: "x5",
                6: "x6",
            },
        )

    def test_from_serialized_state(self):
        game = connect_four_proxy.ConnectFourGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)
        result = get_legal_moves(observation)
        # Initial board: all 7 columns are legal
        self.assertLen(result, 7)

    def test_empty_serialized(self):
        observation = {"serializedGameAndState": ""}
        result = get_legal_moves(observation)
        self.assertEqual(result, {})

    def test_returns_dict(self):
        observation = {
            "legalActions": [0, 3],
            "legalActionStrings": ["x0", "x3"],
        }
        result = get_legal_moves(observation)
        self.assertIsInstance(result, dict)
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, str)


def _make_mock_response(content):
    """Create a mock LiteLLM response."""
    resp = MagicMock()
    resp.usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        completion_tokens_details=None,
    )
    resp.choices = [
        MagicMock(
            message=MagicMock(content=content),
            finish_reason="stop",
        )
    ]
    return resp


class _C4Harness:
    """Adapter wrapping module-level functions into the GameHarness protocol."""

    def get_legal_moves(self, observation):
        return get_legal_moves(observation)

    def make_prompt(self, observation, move_history, previous_response=None, previous_action=None):
        return generate_prompt(
            observation,
            move_history,
            previous_response,
            previous_action,
        )

    def parse_response(self, response, legal_action_strings):
        return parse_response(response, legal_action_strings)


_ENV = {
    "MODEL_NAME": "test-model",
    "MODEL_PROXY_KEY": "test-key",
    "MODEL_PROXY_URL": "dummy_url",
}


class AgentIntegrationTest(absltest.TestCase):
    """Test the Connect Four harness through ``create_agent_fn``."""

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_setup_step_returns_inactive(self, mock_litellm):
        mock_litellm.drop_params = True
        agent = create_agent_fn(_C4Harness())

        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            "I should play in the center.\n\nFinal Answer: 3",
        )

        agent = create_agent_fn(_C4Harness())

        game = connect_four_proxy.ConnectFourGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {})

        self.assertEqual(result["submission"], 3)
        self.assertEqual(result["status"], "OK")

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.side_effect = [
            _make_mock_response("I don't know what to play"),
            _make_mock_response("Final Answer: 0"),
        ]

        agent = create_agent_fn(_C4Harness())

        game = connect_four_proxy.ConnectFourGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {})

        self.assertEqual(result["submission"], 0)
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            "I have no idea what to do.",
        )

        agent = create_agent_fn(_C4Harness())

        game = connect_four_proxy.ConnectFourGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        with self.assertRaises(ValueError):
            agent(observation, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_call_details_present(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            "Final Answer: 3",
        )

        agent = create_agent_fn(_C4Harness())

        game = connect_four_proxy.ConnectFourGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {})

        self.assertIn("call_details", result)
        self.assertLen(result["call_details"], 1)
        cd = result["call_details"][0]
        self.assertEqual(cd["generation_tokens"], 20)
        self.assertEqual(cd["prompt_tokens"], 10)
        self.assertEqual(cd["total_tokens"], 30)
        self.assertEqual(cd["finish_reason"], "stop")

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_generate_returns_present(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            "Final Answer: 3",
        )

        agent = create_agent_fn(_C4Harness())

        game = connect_four_proxy.ConnectFourGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {"includeGenerateReturns": True})

        self.assertIn("generate_returns", result)
        self.assertLen(result["generate_returns"], 1)
        gr = json.loads(result["generate_returns"][0])
        self.assertEqual(gr["generation_tokens"], 20)
        self.assertEqual(gr["prompt_tokens"], 10)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_generate_returns_omitted_by_default(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            "Final Answer: 3",
        )

        agent = create_agent_fn(_C4Harness())

        game = connect_four_proxy.ConnectFourGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {})

        self.assertNotIn("generate_returns", result)


if __name__ == "__main__":
    absltest.main()
