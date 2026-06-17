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

    def test_prose_only_response_triggers_rethink(self):
        # No structured JSON. The parser must NOT guess at intent from a
        # move-shaped token in the prose -- return None and let the
        # rethink loop ask the model to use the required JSON format.
        legal = ["b1-h1", "c1-c3"]
        response = "After thinking it over, I'm going to play b1-h1."
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_no_match_returns_none(self):
        legal = ["b1-h1", "c1-c3"]
        response = '```json\n{"move": "z9-a1"}\n```'
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "z9-a1")

    def test_malformed_json_triggers_rethink(self):
        # Bad JSON: stage-1 extracts nothing. The parser must NOT
        # silently rescue a move from the prose -- return None so the
        # rethink loop can ask the model to fix its format.
        legal = ["b1-h1", "c1-c3"]
        response = "```json\n{bad json}\n```\nI play b1-h1."
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

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

    def test_parse_multiple_json_blocks_picks_last(self):
        """When the model self-corrects mid-response (writes one JSON answer,
        then reconsiders and writes another), the *last* block is the model's
        actual intent. Picking the first one submits the rejected option."""
        legal = ["a6-a7", "g4-f4"]
        response = (
            "First attempt:\n```json\n{\"move\": \"a6-a7\"}\n```\n"
            "On second thought, that's bad. Here's my real move:\n"
            "```json\n{\"move\": \"g4-f4\"}\n```"
        )
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "g4-f4")

    def test_parse_move_re_does_not_cross_newlines(self):
        """``_MOVE_RE`` must not stitch a move-shaped token across newlines
        via the rank-sep-file gap. ``"b1\\n-\\nh1"`` should NOT parse as
        ``b1-h1`` (it would if the regex used ``\\s*`` instead of ``[ \\t]*``)."""
        legal = ["b1-h1", "c1-c3"]
        # A single move-shaped token cannot legitimately straddle newlines.
        # The only token-on-each-line shape here is "b1" / "-" / "h1" — none
        # of which is a valid move on its own, so the parser must return None.
        response = "Move:\nb1\n-\nh1"
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)

    def test_illegal_json_does_not_ghost_substitute_from_prose(self):
        # The JSON move "a8-a1" isn't in legal. The parser must NOT
        # silently substitute "b1-h1" from the prose (ghost antipattern).
        # Surface raw_action so the rethink loop can ask for a legal move.
        legal = ["b1-h1", "c1-c3"]
        response = (
            "I considered b1-h1 but ruled it out. Playing a8-a1.\n"
            '```json\n{"move": "a8-a1"}\n```'
        )
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "a8-a1")


class GeneratePromptTest(absltest.TestCase):

    def test_basic_prompt_black(self):
        observation = {"observationString": "{}", "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertIn("Lines of Action", prompt)
        self.assertIn("Black", prompt)
        self.assertIn("(X)", prompt)
        self.assertIn("connecting all of your remaining pieces", prompt)

    def test_prompt_states_draw_conditions(self):
        """Regression: prompt must disclose that OpenSpiel's lines_of_action
        draws on twofold position repetition and at the 1000-move cap. The
        old prompt claimed 'no draws under normal play', which contradicted
        the engine and stranded models in 5.5% of episodes that drew."""
        observation = {"observationString": "{}", "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertIn("drawn", prompt)
        self.assertIn("second time", prompt)
        self.assertIn("1000 moves", prompt)
        self.assertNotIn("no draws", prompt.lower())

    def test_prompt_double_connect_awards_mover(self):
        """Regression: OpenSpiel awards the win to the moving player when a
        move connects both groups (the my_piece check fires first in
        CheckTerminalState). The old prompt said the OPPONENT won, which is
        the standard-LoA rule but not what this engine implements."""
        observation = {"observationString": "{}", "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertIn("moving player", prompt)
        # The old phrasing said "your opponent wins" in the double-connect
        # case; make sure we did not regress to it.
        self.assertNotIn("your opponent wins", prompt)

    def test_prompt_states_no_moves_loses(self):
        """Regression: 'a player with no legal moves loses' is rule #9 in
        the engine. The old prompt omitted it, so models could blunder into
        a stalemate-loss without knowing it was a win condition."""
        observation = {"observationString": "{}", "playerId": 0}
        prompt = generate_prompt(observation, [])
        self.assertIn("no legal moves", prompt)
        self.assertIn("loses", prompt)

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
        self.assertIn("You suggested", prompt)  # ILLEGAL leads with action
        self.assertIn("z9-a1", prompt)
        self.assertIn("not a legal", prompt)

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

    def parse_response(self, response, legal_action_strings, *, observation=None):
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
