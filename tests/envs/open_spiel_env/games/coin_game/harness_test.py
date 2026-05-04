"""Tests for Coin Game LLM harness."""

import json
from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.coin_game import (
    coin_game_proxy,
)
from kaggle_environments.envs.open_spiel_env.games.coin_game.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _advance_through_chance(state):
    """Auto-resolve chance nodes (preference assignment + deployments)."""
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        state.apply_action(outcomes[0][0])


def _make_observation(state, game, player_id=0):
    """Build a harness-style observation dict from a Coin Game state."""
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
        legal = ["up", "down", "left", "right", "stand"]
        response = '```json\n{"move": "up"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "up")
        self.assertEqual(result.raw_action, "up")

    def test_parse_each_action(self):
        legal = ["up", "down", "left", "right", "stand"]
        for action in legal:
            response = f'```json\n{{"move": "{action}"}}\n```'
            result = parse_response(response, legal)
            self.assertEqual(result.legal_action, action)

    def test_parse_case_insensitive(self):
        legal = ["up", "down", "left", "right", "stand"]
        response = '```json\n{"move": "UP"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "up")

    def test_parse_with_whitespace(self):
        legal = ["up", "down", "left", "right", "stand"]
        response = '```json\n{"move": "  down  "}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "down")

    def test_parse_fallback_keyword_in_text(self):
        legal = ["up", "down", "left", "right", "stand"]
        response = "After thinking it over, I'll move right toward the gold coin."
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "right")

    def test_parse_no_match_returns_none(self):
        legal = ["up", "down", "left", "right", "stand"]
        response = '```json\n{"move": "diagonal"}\n```'
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "diagonal")

    def test_parse_malformed_json_falls_back(self):
        legal = ["up", "down", "left", "right", "stand"]
        response = "```json\n{bad json}\n```\nI'll go left."
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "left")

    def test_parse_no_move_returns_none(self):
        legal = ["up", "down", "left", "right", "stand"]
        response = "I have no idea what to play."
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_bare_json(self):
        legal = ["up", "down", "left", "right", "stand"]
        response = 'I think {"move": "stand"} is best.'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "stand")

    def test_parse_returns_parse_result(self):
        legal = ["up", "down", "left", "right", "stand"]
        result = parse_response('```json\n{"move": "up"}\n```', legal)
        self.assertIsInstance(result, ParseResult)


class GeneratePromptTest(absltest.TestCase):

    def _obs(self, player_id=0, your_pref="a"):
        state = json.dumps({
            "phase": "play",
            "num_rows": 8,
            "num_columns": 8,
            "episode_length": 20,
            "your_preference": your_pref,
            "your_player_id": player_id,
        })
        return {"observationString": state, "playerId": player_id}

    def test_basic_prompt_contains_rules(self):
        prompt = generate_prompt(self._obs(), [])
        self.assertIn("Coin Game", prompt)
        self.assertIn("up, down, left, right, stand", prompt)
        self.assertIn("self_pref^2 + other_pref^2 - bad_coins^2", prompt)

    def test_prompt_includes_preference(self):
        prompt = generate_prompt(self._obs(your_pref="b"), [])
        self.assertIn('preferred coin colour is "b"', prompt)

    def test_prompt_includes_player_id(self):
        prompt = generate_prompt(self._obs(player_id=1, your_pref="c"), [])
        self.assertIn("player id is 1", prompt)

    def test_prompt_includes_dimensions(self):
        prompt = generate_prompt(self._obs(), [])
        self.assertIn("8x8", prompt)
        self.assertIn("20 moves", prompt)

    def test_move_history_included(self):
        prompt = generate_prompt(self._obs(), ["up", "right", "stand"])
        self.assertIn("up right stand", prompt)

    def test_rethink_suffix(self):
        prompt = generate_prompt(
            self._obs(),
            [],
            previous_response="I play diagonally",
            previous_action="diagonal",
        )
        self.assertIn("Your previous response was", prompt)
        self.assertIn("diagonal", prompt)
        self.assertIn("not in the legal", prompt)

    def test_no_rethink_on_first_attempt(self):
        prompt = generate_prompt(self._obs(), [])
        self.assertNotIn("Your previous response was", prompt)

    def test_handles_missing_preference(self):
        obs = {"observationString": json.dumps({}), "playerId": 0}
        prompt = generate_prompt(obs, [])
        self.assertIn('preferred coin colour is "?"', prompt)


class GetLegalMovesTest(absltest.TestCase):

    def test_from_provided_actions(self):
        observation = {
            "legalActions": [0, 4],
            "legalActionStrings": ["up", "stand"],
        }
        result = get_legal_moves(observation)
        self.assertEqual(result, {0: "up", 4: "stand"})

    def test_from_serialized_state(self):
        game = coin_game_proxy.CoinGameGame()
        state = game.new_initial_state()
        _advance_through_chance(state)
        observation = {
            "serializedGameAndState": pyspiel.serialize_game_and_state(
                game.__wrapped__, state.__wrapped__,
            ),
        }
        result = get_legal_moves(observation)
        # Coin Game always has all five actions available during play.
        self.assertEqual(
            sorted(result.values()),
            ["down", "left", "right", "stand", "up"],
        )

    def test_empty_serialized(self):
        observation = {"serializedGameAndState": ""}
        result = get_legal_moves(observation)
        self.assertEqual(result, {})

    def test_returns_dict(self):
        observation = {
            "legalActions": [0, 4],
            "legalActionStrings": ["up", "stand"],
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


class _CoinHarness:
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
    """Test the Coin Game harness through ``create_agent_fn``."""

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_setup_step_returns_inactive(self, mock_litellm):
        mock_litellm.drop_params = True
        agent = create_agent_fn(_CoinHarness())
        result = agent({"step": 0, "remainingOverageTime": 60}, {})
        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            '```json\n{"move": "up"}\n```',
        )
        agent = create_agent_fn(_CoinHarness())

        game = coin_game_proxy.CoinGameGame()
        state = game.new_initial_state()
        _advance_through_chance(state)
        observation = _make_observation(state, game, state.current_player())

        result = agent(observation, {})

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["actionString"], "up")
        # The submission must be the OpenSpiel action id for "up" (which is 0).
        self.assertEqual(
            state.__wrapped__.action_to_string(
                state.current_player(), result["submission"],
            ),
            "up",
        )

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "diagonal"}\n```'),
            _make_mock_response('```json\n{"move": "stand"}\n```'),
        ]
        agent = create_agent_fn(_CoinHarness())

        game = coin_game_proxy.CoinGameGame()
        state = game.new_initial_state()
        _advance_through_chance(state)
        observation = _make_observation(state, game, state.current_player())

        result = agent(observation, {})

        self.assertEqual(result["actionString"], "stand")
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            "I have no idea what to do",
        )
        agent = create_agent_fn(_CoinHarness())

        game = coin_game_proxy.CoinGameGame()
        state = game.new_initial_state()
        _advance_through_chance(state)
        observation = _make_observation(state, game, state.current_player())

        with self.assertRaises(ValueError):
            agent(observation, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_full_episode_via_static_main_wrapper(self, mock_litellm):
        """End-to-end: drive a real ``open_spiel_coin_game`` episode through
        the same notebook-wrapper shape Kaggle injects.

        The model always picks the lexicographically-first legal move so the
        episode terminates deterministically.
        """
        mock_litellm.drop_params = True

        _next_move_box: dict = {"move": "down"}

        def _completion(*_, **__):
            return _make_mock_response(
                f'```json\n{{"move": "{_next_move_box["move"]}"}}\n```'
            )

        mock_litellm.completion.side_effect = _completion

        wrapped_harness = _CoinHarness()

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
            "open_spiel_coin_game",
            configuration={"includeLegalActions": True, "actTimeout": 120},
            debug=True,
        )
        env.reset()
        for _ in range(25):
            if env.done:
                break
            env.step([agent_fn(env.state[0].observation, {}),
                      agent_fn(env.state[1].observation, {})])

        # No error raised and the game state advanced past the setup step.
        self.assertGreater(len(env.steps), 1)


if __name__ == "__main__":
    absltest.main()
