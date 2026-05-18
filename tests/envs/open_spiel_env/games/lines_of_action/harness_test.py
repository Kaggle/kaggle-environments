"""Tests for Lines of Action LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.lines_of_action import (
    lines_of_action_proxy,
)
from kaggle_environments.envs.open_spiel_env.games.lines_of_action.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(state, game, player_id=0):
    """Build a harness-style observation dict from a Lines of Action state."""
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "currentPlayer": state.current_player(),
        "isTerminal": state.is_terminal(),
        "serializedGameAndState": pyspiel.serialize_game_and_state(
            game.__wrapped__, state.__wrapped__,
        ),
    }


class ParseResponseTest(absltest.TestCase):

    def test_parse_json_move(self):
        legal = ["b1-h1", "b1-b3", "c1-c3"]
        response = '```json\n{"move": "b1-h1"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "b1-h1")
        self.assertEqual(result.raw_action, "b1-h1")

    def test_parse_capture(self):
        legal = ["c3xa3", "b1-b3"]
        response = '```json\n{"move": "c3xa3"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "c3xa3")

    def test_parse_case_insensitive(self):
        legal = ["b1-h1", "c1-c3"]
        response = '```json\n{"move": "B1-H1"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "b1-h1")

    def test_parse_separator_swap(self):
        """Model said '-' but the legal move is a capture 'x' (or vice versa).

        The harness should still match on coordinates so the agent isn't
        penalised for getting the cosmetic separator wrong.
        """
        legal = ["c3xa3", "b1-b3"]
        response = '```json\n{"move": "c3-a3"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "c3xa3")

    def test_parse_fallback_coordinate_in_text(self):
        legal = ["b1-h1", "c1-c3"]
        response = "After thinking it over, I'm going to play b1-h1."
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "b1-h1")

    def test_parse_no_match_returns_none(self):
        legal = ["b1-h1", "c1-c3"]
        response = '```json\n{"move": "z9-a1"}\n```'
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "z9-a1")

    def test_parse_malformed_json_falls_back(self):
        legal = ["b1-h1", "c1-c3"]
        response = "```json\n{bad json}\n```\nI play b1-h1."
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "b1-h1")

    def test_parse_no_move_returns_none(self):
        legal = ["b1-h1", "c1-c3"]
        response = "I have no idea what to play."
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_bare_json(self):
        legal = ["b1-h1", "c1-c3"]
        response = 'I think {"move": "b1-h1"} is best.'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "b1-h1")

    def test_parse_returns_parse_result(self):
        legal = ["b1-h1"]
        result = parse_response('```json\n{"move": "b1-h1"}\n```', legal)
        self.assertIsInstance(result, ParseResult)


class GeneratePromptTest(absltest.TestCase):

    def test_basic_prompt_black(self):
        observation = {"observationString": "{}", "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertIn("Lines of Action", prompt)
        self.assertIn("Black", prompt)
        self.assertIn("(X)", prompt)
        self.assertIn("connecting all of your remaining pieces", prompt)

    def test_basic_prompt_white(self):
        observation = {"observationString": "{}", "playerId": 1}
        prompt = generate_prompt(observation, [])
        self.assertIn("White", prompt)
        self.assertIn("(O)", prompt)

    def test_move_history_included(self):
        observation = {"observationString": "{}", "playerId": 0}
        prompt = generate_prompt(observation, ["b1-h1", "a3-c3", "c1-c3"])
        self.assertIn("b1-h1 a3-c3 c1-c3", prompt)

    def test_rethink_suffix(self):
        observation = {"observationString": "{}", "playerId": 0}
        prompt = generate_prompt(
            observation,
            [],
            previous_response="I play z9-a1",
            previous_action="z9-a1",
        )
        self.assertIn("Your previous response was", prompt)
        self.assertIn("z9-a1", prompt)
        self.assertIn("not in the legal moves list", prompt)

    def test_no_rethink_on_first_attempt(self):
        observation = {"observationString": "{}", "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertNotIn("Your previous response was", prompt)


class GetLegalMovesTest(absltest.TestCase):

    def test_from_provided_actions(self):
        observation = {
            "legalActions": [142, 100],
            "legalActionStrings": ["b1-h1", "b1-b3"],
        }
        result = get_legal_moves(observation)
        self.assertEqual(result, {142: "b1-h1", 100: "b1-b3"})

    def test_from_serialized_state(self):
        game = lines_of_action_proxy.LinesOfActionGame()
        state = game.new_initial_state()
        observation = {
            "serializedGameAndState": pyspiel.serialize_game_and_state(
                game.__wrapped__, state.__wrapped__,
            ),
        }
        result = get_legal_moves(observation)
        # Initial position has a known set of legal opening moves; just
        # assert the structure and that "b1-h1" appears (the canonical
        # horizontal-edge move along rank 1).
        self.assertGreater(len(result), 0)
        self.assertIn("b1-h1", result.values())

    def test_empty_serialized(self):
        observation = {"serializedGameAndState": ""}
        result = get_legal_moves(observation)
        self.assertEqual(result, {})

    def test_returns_dict(self):
        observation = {
            "legalActions": [0, 5],
            "legalActionStrings": ["b1-h1", "c1-c3"],
        }
        result = get_legal_moves(observation)
        self.assertIsInstance(result, dict)
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, str)


def _make_mock_response(content):
    """Create a mock LiteLLM response."""
    resp = MagicMock()
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
    resp.choices = [
        MagicMock(message=MagicMock(content=content), finish_reason="stop"),
    ]
    return resp


class _LoAHarness:
    """Adapter wrapping module-level functions in the GameHarness protocol.

    Mirrors the ``_NotebookGameHarness`` injected by Kaggle's
    StaticHarnessAgentScript so this test exercises the same wiring.
    """

    def get_legal_moves(self, observation):
        return get_legal_moves(observation)

    def make_prompt(
        self, observation, move_history,
        previous_response=None, previous_action=None,
    ):
        return generate_prompt(
            observation, move_history,
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
    """Test the LoA harness through ``create_agent_fn`` from ``core_harness``."""

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_setup_step_returns_inactive(self, mock_litellm):
        mock_litellm.drop_params = True
        agent = create_agent_fn(_LoAHarness())
        result = agent({"step": 0, "remainingOverageTime": 60}, {})
        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            '```json\n{"move": "b1-h1"}\n```',
        )
        agent = create_agent_fn(_LoAHarness())

        game = lines_of_action_proxy.LinesOfActionGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {})

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["actionString"], "b1-h1")
        # The submission must be the OpenSpiel action id corresponding to
        # the matched move string — verify by re-asking the engine.
        self.assertEqual(
            state.__wrapped__.action_to_string(0, result["submission"]),
            "b1-h1",
        )

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "z9-a1"}\n```'),
            _make_mock_response('```json\n{"move": "b1-h1"}\n```'),
        ]
        agent = create_agent_fn(_LoAHarness())

        game = lines_of_action_proxy.LinesOfActionGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        result = agent(observation, {})

        self.assertEqual(result["actionString"], "b1-h1")
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            "I don't know what to play",
        )
        agent = create_agent_fn(_LoAHarness())

        game = lines_of_action_proxy.LinesOfActionGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game)

        with self.assertRaises(ValueError):
            agent(observation, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_full_episode_via_static_main_wrapper(self, mock_litellm):
        """End-to-end: drive a real ``open_spiel_lines_of_action`` episode
        through the same notebook-wrapper shape Kaggle injects.

        The model always picks the lexicographically-first legal move so the
        episode terminates deterministically.
        """
        mock_litellm.drop_params = True

        def _completion(*_, **__):
            # Inspect the prompt to find the current state's legal moves so
            # we can pick a guaranteed-legal answer. We piggyback by parsing
            # the engine via a fresh state isn't possible here; instead we
            # ask the harness adapter to expose them via a shared box.
            move = _next_move_box["move"]
            return _make_mock_response(f'```json\n{{"move": "{move}"}}\n```')

        mock_litellm.completion.side_effect = _completion

        # Mirror StaticHarnessAgentScript's _NotebookGameHarness, but
        # capture the next legal move so the mocked LLM "knows" what to
        # answer. This keeps the wiring identical to production.
        _next_move_box: dict = {"move": "b1-h1"}
        wrapped_harness = _LoAHarness()

        class _CapturingHarness:
            def get_legal_moves(self, observation):
                moves = wrapped_harness.get_legal_moves(observation)
                if moves:
                    _next_move_box["move"] = sorted(moves.values())[0]
                return moves

            def make_prompt(self, *a, **k):
                return wrapped_harness.make_prompt(*a, **k)

            def parse_response(self, *a, **k):
                return wrapped_harness.parse_response(*a, **k)

        agent_fn = create_agent_fn(_CapturingHarness())

        from kaggle_environments import make
        env = make(
            "open_spiel_lines_of_action",
            configuration={"includeLegalActions": True, "actTimeout": 120},
            debug=True,
        )
        # Run a bounded number of steps; episodes can be long but for a
        # smoke test we only need to confirm the harness submits valid
        # actions for several plies without raising.
        env.reset()
        for _ in range(10):
            if env.done:
                break
            env.step([agent_fn(env.state[0].observation, {}),
                      agent_fn(env.state[1].observation, {})])

        # No error raised and the game state advanced past the setup step.
        self.assertGreater(len(env.steps), 1)


if __name__ == "__main__":
    absltest.main()
