"""Tests for the Breakthrough LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.breakthrough import (
    breakthrough_proxy,
)
from kaggle_environments.envs.open_spiel_env.games.breakthrough.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(
    state: breakthrough_proxy.BreakthroughState,
    game: breakthrough_proxy.BreakthroughGame,
    player_id: int = 0,
) -> dict:
    """Build a harness-style observation dict from a proxy state."""
    legal = list(state.legal_actions())
    cp = int(state.current_player()) if not state.is_terminal() else 0
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "currentPlayer": int(state.current_player()),
        "isTerminal": state.is_terminal(),
        "legalActions": legal,
        "legalActionStrings": [state.action_to_string(cp, a) for a in legal],
        "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
    }


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class ParseResponseTest(absltest.TestCase):
    legal = ["a7a6", "b7a6", "b7b6", "b7c6", "c7b6"]

    def test_parse_json_block(self):
        result = parse_response('```json\n{"move": "a7a6"}\n```', self.legal)
        self.assertEqual(result.legal_action, "a7a6")
        self.assertEqual(result.raw_action, "a7a6")

    def test_parse_bare_json(self):
        result = parse_response('I think {"move": "b7c6"} is best.', self.legal)
        self.assertEqual(result.legal_action, "b7c6")

    def test_parse_case_insensitive(self):
        result = parse_response('```json\n{"move": "A7A6"}\n```', self.legal)
        self.assertEqual(result.legal_action, "a7a6")

    def test_parse_illegal_move_returns_raw(self):
        result = parse_response('```json\n{"move": "z9z9"}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "z9z9")

    def test_prose_only_response_triggers_rethink(self):
        # No structured JSON. The parser must NOT guess at intent from a
        # move-shaped token in the prose -- return None and let rethink
        # ask the model to use the required JSON format.
        result = parse_response("I will play a7a6 this turn.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_no_match_returns_none(self):
        result = parse_response("I have no idea.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_returns_parse_result_type(self):
        result = parse_response('```json\n{"move": "a7a6"}\n```', self.legal)
        self.assertIsInstance(result, ParseResult)

    def test_multiple_json_blocks_last_wins(self):
        response = (
            'First I considered ```json\n{"move": "a7a6"}\n``` but then committed to ```json\n{"move": "b7c6"}\n```.'
        )
        result = parse_response(response, self.legal)
        self.assertEqual(result.legal_action, "b7c6")

    def test_illegal_json_does_not_ghost_substitute_from_prose(self):
        # The model's JSON answer (z9z9) isn't legal. The parser must
        # NOT silently substitute a legal token from the prose -- return
        # None so the rethink loop asks the model to fix its answer.
        response = 'I considered a7a6 but ruled it out.\n```json\n{"move": "z9z9"}\n```'
        result = parse_response(response, self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "z9z9")

    def test_capture_marker_tolerated_when_added(self):
        # Model wrote "a7a6*" when no capture is happening; matcher should
        # strip the spurious '*' and return the legal slide.
        result = parse_response('```json\n{"move": "a7a6*"}\n```', self.legal)
        self.assertEqual(result.legal_action, "a7a6")

    def test_capture_marker_tolerated_when_missing(self):
        # Model forgot to mark a diagonal capture with '*'.
        legal_with_capture = ["a2b3*", "c2c3"]
        result = parse_response(
            '```json\n{"move": "a2b3"}\n```',
            legal_with_capture,
        )
        self.assertEqual(result.legal_action, "a2b3*")

    def test_dash_separator_tolerated(self):
        # Common drift: model writes "a7-a6" instead of "a7a6".
        result = parse_response('```json\n{"move": "a7-a6"}\n```', self.legal)
        self.assertEqual(result.legal_action, "a7a6")


# ---------------------------------------------------------------------------
# generate_prompt
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game = breakthrough_proxy.BreakthroughGame()
        self.state = self.game.new_initial_state()

    def test_basic_prompt_contents(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Breakthrough", prompt)
        self.assertIn("Player 0", prompt)
        self.assertIn("'b'", prompt)
        self.assertIn("a7a6", prompt)  # example slide in instructions

    def test_player_label_swap(self):
        first = self.state.legal_actions()[0]
        self.state.apply_action(first)
        obs1 = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs1, [])
        self.assertIn("Player 1", prompt)
        self.assertIn("'w'", prompt)

    def test_legal_moves_not_listed(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # The prompt deliberately omits the legal-move list so the model
        # has to reason about legality from the board alone. "a7a6" is
        # excluded because the action-notation example uses that token.
        for legal in obs["legalActionStrings"]:
            if legal == "a7a6":
                continue
            self.assertNotIn(legal, prompt)

    def test_board_ascii_includes_files_and_ranks(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("a b c d e f g h", prompt)
        self.assertIn(" 1 ", prompt)
        self.assertIn(" 8 ", prompt)

    def test_piece_counts_rendered(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Black ('b') = 16", prompt)
        self.assertIn("White ('w') = 16", prompt)

    def test_own_pieces_listed_for_player_0(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Black starts on ranks 7-8.
        for sq in ("a8", "h8", "a7", "h7"):
            self.assertIn(sq, prompt)

    def test_own_pieces_listed_for_player_1(self):
        obs = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs, [])
        # White starts on ranks 1-2.
        for sq in ("a1", "h1", "a2", "h2"):
            self.assertIn(sq, prompt)

    def test_forward_direction_explained_per_player(self):
        obs0 = _make_observation(self.state, self.game, player_id=0)
        self.assertIn("toward rank 1", generate_prompt(obs0, []))

        obs1 = _make_observation(self.state, self.game, player_id=1)
        self.assertIn("toward rank 8", generate_prompt(obs1, []))

    def test_player_text_differs_between_players(self):
        # The two rendered prompts MUST disagree on player-asymmetric text;
        # diff-identical prompts mean the orientation language was baked in.
        obs0 = _make_observation(self.state, self.game, player_id=0)
        obs1 = _make_observation(self.state, self.game, player_id=1)
        self.assertNotEqual(generate_prompt(obs0, []), generate_prompt(obs1, []))

    def test_last_move_rendered_after_play(self):
        first = self.state.legal_actions()[0]
        first_str = self.state.action_to_string(0, first)
        self.state.apply_action(first)
        obs1 = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs1, [])
        self.assertIn(f"Last move played: {first_str}", prompt)

    def test_move_history_includes_both_players(self):
        # Play one move for each side; the full-game history line must
        # show BOTH moves, not just this agent's per-agent history.
        a0 = self.state.legal_actions()[0]
        s0 = self.state.action_to_string(0, a0)
        self.state.apply_action(a0)
        a1 = self.state.legal_actions()[0]
        s1 = self.state.action_to_string(1, a1)
        self.state.apply_action(a1)

        obs = _make_observation(self.state, self.game, player_id=0)
        # Pass an empty per-agent history -- the prompt should ignore it
        # and reconstruct full history from the engine state.
        prompt = generate_prompt(obs, [])
        self.assertIn(f"{s0}, {s1}", prompt)
        self.assertIn("both players", prompt)

    def test_move_history_none_when_empty(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("oldest first): None", prompt)

    def test_rules_disclosed(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Core win conditions and movement rules.
        self.assertIn("back rank", prompt)
        self.assertIn("capture", prompt.lower())
        # No-draw guarantee.
        self.assertIn("no draws", prompt.lower())
        # Straight-forward moves cannot capture.
        self.assertIn("Straight-forward", prompt)
        # Capture marker convention is described.
        self.assertIn("*", prompt)

    def test_rethink_illegal_suffix_leads_with_action(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(
            obs,
            [],
            previous_response="I'll play z9z9",
            previous_action="z9z9",
        )
        self.assertIn("You suggested", prompt)
        self.assertIn("z9z9", prompt)
        self.assertIn("not a legal move", prompt)
        # ILLEGAL branch must NOT paste the full previous response.
        self.assertNotIn("I'll play z9z9", prompt)

    def test_rethink_unparseable_suffix_includes_previous_response(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(
            obs,
            [],
            previous_response="I dunno what to do",
            previous_action=None,
        )
        self.assertIn("Your previous response", prompt)
        self.assertIn("I dunno what to do", prompt)
        self.assertIn('"move":', prompt)

    def test_no_rethink_on_first_attempt(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertNotIn("You suggested", prompt)
        self.assertNotIn("Your previous response", prompt)


# ---------------------------------------------------------------------------
# get_legal_moves
# ---------------------------------------------------------------------------


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        obs = {
            "legalActions": [2, 14, 26],
            "legalActionStrings": ["a7a6", "b7b6", "c7c6"],
        }
        result = get_legal_moves(obs)
        self.assertEqual(result, {2: "a7a6", 14: "b7b6", 26: "c7c6"})

    def test_from_serialized_state(self):
        game = breakthrough_proxy.BreakthroughGame()
        state = game.new_initial_state()
        obs = {
            "playerId": 0,
            "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
        }
        result = get_legal_moves(obs)
        self.assertGreater(len(result), 0)
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, str)
            # All breakthrough action strings are 4 or 5 chars.
            self.assertIn(len(v), (4, 5))

    def test_empty_serialized(self):
        self.assertEqual(get_legal_moves({"serializedGameAndState": ""}), {})


# ---------------------------------------------------------------------------
# create_agent_fn integration
# ---------------------------------------------------------------------------


class _BreakthroughHarness:
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
        agent = create_agent_fn(_BreakthroughHarness())

        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        game = breakthrough_proxy.BreakthroughGame()
        state = game.new_initial_state()
        first_legal = state.action_to_string(0, state.legal_actions()[0])
        mock_litellm.completion.return_value = _make_mock_response(f'```json\n{{"move": "{first_legal}"}}\n```')
        agent = create_agent_fn(_BreakthroughHarness())

        obs = _make_observation(state, game, player_id=0)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal)
        self.assertEqual(result["status"], "OK")
        self.assertIn("thoughts", result)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        game = breakthrough_proxy.BreakthroughGame()
        state = game.new_initial_state()
        first_legal = state.action_to_string(0, state.legal_actions()[0])
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "z9z9"}\n```'),
            _make_mock_response(f'```json\n{{"move": "{first_legal}"}}\n```'),
        ]
        agent = create_agent_fn(_BreakthroughHarness())

        obs = _make_observation(state, game, player_id=0)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal)
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("I cannot decide.")
        agent = create_agent_fn(_BreakthroughHarness())

        game = breakthrough_proxy.BreakthroughGame()
        state = game.new_initial_state()
        obs = _make_observation(state, game, player_id=0)

        with self.assertRaises(ValueError):
            agent(obs, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_short_game_via_agent_fns(self, mock_litellm):
        """Drive a short Breakthrough game with scripted LLM agents that
        always pick their first legal move, verifying the harness
        round-trips through pyspiel cleanly."""
        mock_litellm.drop_params = True

        game = breakthrough_proxy.BreakthroughGame()
        state = game.new_initial_state()

        def fake_completion(*, model, messages, **kwargs):
            del model, kwargs
            content = messages[0]["content"]
            player_id = 0 if "Player 0" in content else 1
            first = state.action_to_string(player_id, state.legal_actions()[0])
            return _make_mock_response(f'```json\n{{"move": "{first}"}}\n```')

        mock_litellm.completion.side_effect = fake_completion
        agent_p0 = create_agent_fn(_BreakthroughHarness())
        agent_p1 = create_agent_fn(_BreakthroughHarness())

        for _ in range(20):
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
