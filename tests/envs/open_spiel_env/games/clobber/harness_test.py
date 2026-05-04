"""Tests for the Clobber LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.clobber import (
    clobber_proxy,
)
from kaggle_environments.envs.open_spiel_env.games.clobber.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(
    state: clobber_proxy.ClobberState,
    game: clobber_proxy.ClobberGame,
    player_id: int = 0,
) -> dict:
    """Build a harness-style observation dict from a proxy state."""
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "currentPlayer": int(state.current_player()),
        "isTerminal": state.is_terminal(),
        "legalActions": list(state.legal_actions(player_id)),
        "legalActionStrings": [
            state.action_to_string(player_id, a)
            for a in state.legal_actions(player_id)
        ],
        "serializedGameAndState": pyspiel.serialize_game_and_state(
            game.__wrapped__, state.__wrapped__
        ),
    }


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class ParseResponseTest(absltest.TestCase):
    legal = ["a1b1", "b2a2", "c3d3", "d4c4"]

    def test_parse_json_block(self):
        result = parse_response('```json\n{"move": "a1b1"}\n```', self.legal)
        self.assertEqual(result.legal_action, "a1b1")
        self.assertEqual(result.raw_action, "a1b1")

    def test_parse_bare_json(self):
        result = parse_response('I think {"move": "b2a2"} is best.', self.legal)
        self.assertEqual(result.legal_action, "b2a2")

    def test_parse_action_string_in_response(self):
        result = parse_response("I will play c3d3 this turn.", self.legal)
        self.assertEqual(result.legal_action, "c3d3")

    def test_parse_illegal_move_returns_raw(self):
        result = parse_response('```json\n{"move": "z9z9"}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "z9z9")

    def test_parse_no_match_returns_none(self):
        result = parse_response("I have no idea.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_returns_parse_result_type(self):
        result = parse_response('```json\n{"move": "a1b1"}\n```', self.legal)
        self.assertIsInstance(result, ParseResult)

    def test_parse_does_not_pick_unrelated_token(self):
        # A bare-text 'a1b9' is not in the legal list and shouldn't match.
        result = parse_response("After thinking, I'll play a1b9.", self.legal)
        self.assertIsNone(result.legal_action)


# ---------------------------------------------------------------------------
# generate_prompt
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game = clobber_proxy.ClobberGame({"rows": 4, "columns": 4})
        self.state = self.game.new_initial_state()

    def test_basic_prompt_contents(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Clobber", prompt)
        self.assertIn("Player 0", prompt)
        self.assertIn("'o'", prompt)
        # Default initial 4x4 board has both pieces present.
        self.assertIn("a1b1", prompt)

    def test_player_label_swap(self):
        # Apply one move so it becomes player 1's turn.
        self.state.apply_action(self.state.legal_actions(0)[0])
        obs1 = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs1, [])
        self.assertIn("Player 1", prompt)
        self.assertIn("'x'", prompt)

    def test_legal_moves_listed(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        for legal in obs["legalActionStrings"]:
            self.assertIn(legal, prompt)

    def test_board_ascii_includes_files_and_ranks(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # File header + at least one rank.
        self.assertIn("a b c d", prompt)
        self.assertIn("4 ", prompt)
        self.assertIn("1 ", prompt)

    def test_last_move_rendered_after_play(self):
        first = self.state.legal_actions(0)[0]
        first_str = self.state.action_to_string(0, first)
        self.state.apply_action(first)
        obs1 = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs1, [])
        self.assertIn(f"Last move played: {first_str}", prompt)

    def test_rethink_suffix(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(
            obs, [], previous_response="I'll play z9z9", previous_action="z9z9"
        )
        self.assertIn("Your previous response was", prompt)
        self.assertIn("z9z9", prompt)
        self.assertIn("NOT in the legal move list", prompt)


# ---------------------------------------------------------------------------
# get_legal_moves
# ---------------------------------------------------------------------------


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        obs = {
            "legalActions": [1, 2, 9],
            "legalActionStrings": ["a5b5", "a5a4", "c5d5"],
        }
        result = get_legal_moves(obs)
        self.assertEqual(result, {1: "a5b5", 2: "a5a4", 9: "c5d5"})

    def test_from_serialized_state(self):
        game = clobber_proxy.ClobberGame({"rows": 4, "columns": 4})
        state = game.new_initial_state()
        obs = {
            "playerId": 0,
            "serializedGameAndState": pyspiel.serialize_game_and_state(
                game.__wrapped__, state.__wrapped__
            ),
        }
        result = get_legal_moves(obs)
        # 4x4 alternating board: each 'o' has up to 4 neighbors with 'x'.
        # All resulting strings must be 4-char "<from><to>".
        self.assertGreater(len(result), 0)
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, str)
            self.assertEqual(len(v), 4)

    def test_empty_serialized(self):
        self.assertEqual(get_legal_moves({"serializedGameAndState": ""}), {})


# ---------------------------------------------------------------------------
# create_agent_fn integration
# ---------------------------------------------------------------------------


class _ClobberHarness:
    """Adapter wrapping module-level functions into the GameHarness protocol."""

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


def _make_mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
    resp.choices = [
        MagicMock(message=MagicMock(content=content), finish_reason="stop")
    ]
    return resp


_ENV = {
    "MODEL_NAME": "test-model",
    "MODEL_PROXY_KEY": "test-key",
    "MODEL_PROXY_URL": "dummy_url",
}


class AgentIntegrationTest(absltest.TestCase):
    """Run the harness through ``create_agent_fn`` from ``core_harness``."""

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_setup_step_returns_inactive(self, mock_litellm):
        mock_litellm.drop_params = True
        agent = create_agent_fn(_ClobberHarness())

        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        game = clobber_proxy.ClobberGame({"rows": 4, "columns": 4})
        state = game.new_initial_state()
        # Pick the first legal move so we know it's valid.
        first_legal = state.action_to_string(0, state.legal_actions(0)[0])
        mock_litellm.completion.return_value = _make_mock_response(
            f'```json\n{{"move": "{first_legal}"}}\n```'
        )
        agent = create_agent_fn(_ClobberHarness())

        obs = _make_observation(state, game, player_id=0)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal)
        self.assertEqual(result["status"], "OK")
        self.assertIn("thoughts", result)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        game = clobber_proxy.ClobberGame({"rows": 4, "columns": 4})
        state = game.new_initial_state()
        first_legal = state.action_to_string(0, state.legal_actions(0)[0])
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "z9z9"}\n```'),
            _make_mock_response(f'```json\n{{"move": "{first_legal}"}}\n```'),
        ]
        agent = create_agent_fn(_ClobberHarness())

        obs = _make_observation(state, game, player_id=0)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal)
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            "I cannot decide."
        )
        agent = create_agent_fn(_ClobberHarness())

        game = clobber_proxy.ClobberGame({"rows": 4, "columns": 4})
        state = game.new_initial_state()
        obs = _make_observation(state, game, player_id=0)

        with self.assertRaises(ValueError):
            agent(obs, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_full_two_player_game_via_agent_fns(self, mock_litellm):
        """Drive a tiny game with two scripted LLM agents that always pick
        their first legal move, verifying the harness round-trips through
        pyspiel cleanly to a terminal state."""
        mock_litellm.drop_params = True

        game = clobber_proxy.ClobberGame({"rows": 3, "columns": 3})
        state = game.new_initial_state()

        def fake_completion(*, model, messages, **kwargs):
            del model, kwargs
            content = messages[0]["content"]
            player_id = 0 if "Player 0" in content else 1
            first = state.action_to_string(
                player_id, state.legal_actions(player_id)[0]
            )
            return _make_mock_response(f'```json\n{{"move": "{first}"}}\n```')

        mock_litellm.completion.side_effect = fake_completion
        agent_p0 = create_agent_fn(_ClobberHarness())
        agent_p1 = create_agent_fn(_ClobberHarness())

        turns = 0
        max_turns = 30
        while not state.is_terminal() and turns < max_turns:
            cp = state.current_player()
            agent = agent_p0 if cp == 0 else agent_p1
            obs = _make_observation(state, game, player_id=cp)
            result = agent(obs, {})
            self.assertEqual(result["status"], "OK")
            state.apply_action(result["submission"])
            turns += 1

        self.assertTrue(state.is_terminal())


if __name__ == "__main__":
    absltest.main()
