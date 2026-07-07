"""Tests for the Nine Men's Morris LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.nine_mens_morris import (
    nine_mens_morris_proxy,
)
from kaggle_environments.envs.open_spiel_env.games.nine_mens_morris.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(
    state: nine_mens_morris_proxy.NineMensMorrisState,
    game: nine_mens_morris_proxy.NineMensMorrisGame,
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


def _move_action(src: int, dst: int) -> int:
    return 24 + src * 24 + dst


def _concise(action_id: int) -> str:
    """Concise-notation form of an action id (mirrors the prompt spec)."""
    if action_id < 24:
        return str(action_id)
    src, dst = divmod(action_id - 24, 24)
    return f"{src}-{dst}"


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class ParseResponseTest(absltest.TestCase):
    legal = [
        "Place at point 0",
        "Place at point 5",
        "Place at point 14",
        "Move point 3 -> point 4",
        "Capture opponent piece at point 9",
    ]

    def test_concise_placement(self):
        result = parse_response('```json\n{"move": "5"}\n```', self.legal)
        self.assertEqual(result.legal_action, "Place at point 5")
        self.assertEqual(result.raw_action, "5")

    def test_concise_capture_resolves_when_capture_is_the_legal_form(self):
        # Placement and capture share the single-int shape; only one is
        # ever legal at a given moment. Here point 9 is only legal as a
        # capture, so "9" resolves unambiguously.
        result = parse_response('```json\n{"move": "9"}\n```', self.legal)
        self.assertEqual(result.legal_action, "Capture opponent piece at point 9")

    def test_concise_movement_dash(self):
        result = parse_response('```json\n{"move": "3-4"}\n```', self.legal)
        self.assertEqual(result.legal_action, "Move point 3 -> point 4")

    def test_concise_movement_arrow(self):
        result = parse_response('```json\n{"move": "3->4"}\n```', self.legal)
        self.assertEqual(result.legal_action, "Move point 3 -> point 4")

    def test_trailing_period_tolerated(self):
        result = parse_response('```json\n{"move": "5."}\n```', self.legal)
        self.assertEqual(result.legal_action, "Place at point 5")

    def test_verbose_form_no_longer_matches(self):
        # The prompt asks for the concise notation only; verbose forms
        # trigger a rethink so the model realigns.
        result = parse_response('```json\n{"move": "Place at point 5"}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "Place at point 5")

    def test_parse_illegal_move_returns_raw(self):
        result = parse_response('```json\n{"move": "99"}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "99")

    def test_prose_only_response_triggers_rethink(self):
        # No structured JSON; parser must NOT guess intent from a
        # move-shaped token in the prose.
        result = parse_response("I will place at point 5 this turn.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_no_match_returns_none(self):
        result = parse_response("I have no idea.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_returns_parse_result_type(self):
        result = parse_response('```json\n{"move": "0"}\n```', self.legal)
        self.assertIsInstance(result, ParseResult)

    def test_multiple_json_last_wins(self):
        response = (
            'First I considered {"move": "0"}, then reconsidered.\n'
            '```json\n{"move": "14"}\n```'
        )
        result = parse_response(response, self.legal)
        self.assertEqual(result.legal_action, "Place at point 14")

    def test_illegal_json_does_not_ghost_substitute_from_prose(self):
        # Model discusses a legal token in prose then commits to an illegal
        # one in JSON. Parser must return the illegal token as raw and NOT
        # silently swap in the prose token.
        response = 'I considered "5" but ruled it out.\n```json\n{"move": "99"}\n```'
        result = parse_response(response, self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "99")


# ---------------------------------------------------------------------------
# generate_prompt
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game = nine_mens_morris_proxy.NineMensMorrisGame()
        self.state = self.game.new_initial_state()

    def test_basic_prompt_contents(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Nine Men's Morris", prompt)
        self.assertIn("Player 0", prompt)
        self.assertIn("WHITE", prompt)
        # Both concise-notation examples appear in the action-format spec.
        self.assertIn('"move": "4"', prompt)
        self.assertIn('"move": "3-10"', prompt)

    def test_player_asymmetric_labels(self):
        prompt0 = generate_prompt(_make_observation(self.state, self.game, player_id=0), [])
        # Advance one placement so player 1 has a legitimate turn.
        self.state.apply_action(0)
        prompt1 = generate_prompt(_make_observation(self.state, self.game, player_id=1), [])
        self.assertIn("You are playing WHITE ('W')", prompt0)
        self.assertIn("You are playing BLACK ('B')", prompt1)
        # The White-first summary line is present in both prompts; the
        # asymmetric text is the "You are playing" / "Opponent" line.
        self.assertIn("Opponent (BLACK, 'B')", prompt0)
        self.assertIn("Opponent (WHITE, 'W')", prompt1)

    def test_legal_moves_not_enumerated(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # The prompt deliberately omits the legal-move list so the model
        # must reason from the state and rules. The concise-notation
        # examples use bare integers, so no full "Place at point N" legal
        # string should ever appear in the prompt.
        for legal in obs["legalActionStrings"]:
            self.assertNotIn(legal, prompt, f"legal move leaked into prompt: {legal!r}")

    def test_board_layout_includes_all_point_indices(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # The two-digit ASCII layout renders every point 00..23.
        for p in range(24):
            self.assertIn(f"{p:02d}", prompt)

    def test_adjacency_table_present(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Corners have exactly 2 neighbors; centres have 3 or 4.
        self.assertIn("0: 1, 9", prompt)
        self.assertIn("4: 1, 3, 5, 7", prompt)
        self.assertIn("19: 16, 18, 20, 22", prompt)

    def test_mill_list_present(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Sample from each of the three mill categories.
        self.assertIn("(0,1,2)", prompt)
        self.assertIn("(0,9,21)", prompt)
        self.assertIn("(1,4,7)", prompt)

    def test_loss_conditions_disclosed(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("fewer than 3 pieces", prompt)
        self.assertIn("no legal move", prompt)

    def test_placement_phase_at_start(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Current phase: placement", prompt)
        # Reserve counts.
        self.assertIn("9 still to place", prompt)

    def test_capture_phase_after_mill(self):
        # W plays 0, 1, 2 to form top-row mill; B parks on 9, 10.
        for a in [0, 9, 1, 10, 2]:
            self.state.apply_action(a)
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Current phase: capture", prompt)
        # Active mill for W should be reported.
        self.assertIn("yours: (0,1,2)", prompt)
        self.assertIn("REMOVE one opponent piece", prompt)

    def test_move_history_covers_both_players(self):
        for a in [0, 9, 1, 10]:
            self.state.apply_action(a)
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Both W (from actions 0, 1) and B (9, 10) plies should appear.
        self.assertIn("W: place point 0", prompt)
        self.assertIn("B: place point 9", prompt)
        self.assertIn("W: place point 1", prompt)
        self.assertIn("B: place point 10", prompt)

    def test_move_history_labels_capture_and_mill(self):
        for a in [0, 9, 1, 10, 2, 9]:  # W mills top row, captures B at 9
            self.state.apply_action(a)
        obs = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs, [])
        self.assertIn("W: place point 2 (mill!)", prompt)
        self.assertIn("W: capture point 9", prompt)

    def test_move_history_none_when_empty(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("(none yet)", prompt)

    def test_ignores_per_agent_move_history_argument(self):
        # The harness deliberately reconstructs both-player history from
        # the serialized state, so the per-agent list passed in is unused.
        # Passing garbage strings must not surface anywhere.
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, ["SHOULD_NOT_APPEAR"])
        self.assertNotIn("SHOULD_NOT_APPEAR", prompt)

    def test_flying_phase_language(self):
        # Fabricate a state_dict directly to hit the flying branch, since
        # a genuine flying-phase state requires a very long game.
        obs = {
            "observationString": (
                '{"board": ["W","W","W",".",".",".",".",".",".",'
                '".",".",".",".",".",".",".",".",".",".",".",".",".",".","."],'
                '"current_player": "W", "phase": "flying",'
                '"men_to_deploy": {"W": 0, "B": 0},'
                '"num_men": {"W": 3, "B": 3}, "turn_number": 60,'
                '"is_terminal": false, "winner": null, "last_action": null}'
            ),
            "playerId": 0,
            "serializedGameAndState": "",
        }
        prompt = generate_prompt(obs, [])
        self.assertIn("Current phase: flying", prompt)
        self.assertIn("ANY empty point", prompt)

    def test_rethink_illegal_suffix(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(
            obs,
            [],
            previous_response="I'll play garbage",
            previous_action="Place at point 999",
        )
        self.assertIn("You suggested", prompt)
        self.assertIn("Place at point 999", prompt)
        self.assertIn("not a legal move", prompt)

    def test_rethink_unparseable_suffix(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(
            obs,
            [],
            previous_response="totally missing json",
            previous_action=None,
        )
        self.assertIn("totally missing json", prompt)
        self.assertIn("No JSON answer could be parsed", prompt)

    def test_no_rethink_on_first_attempt(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertNotIn("You suggested", prompt)
        self.assertNotIn("No JSON answer could be parsed", prompt)


# ---------------------------------------------------------------------------
# get_legal_moves
# ---------------------------------------------------------------------------


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        obs = {
            "legalActions": [0, 5, _move_action(3, 4)],
            "legalActionStrings": [
                "Place at point 0",
                "Place at point 5",
                "Move point 3 -> point 4",
            ],
        }
        result = get_legal_moves(obs)
        self.assertEqual(
            result,
            {
                0: "Place at point 0",
                5: "Place at point 5",
                _move_action(3, 4): "Move point 3 -> point 4",
            },
        )

    def test_from_serialized_state(self):
        game = nine_mens_morris_proxy.NineMensMorrisGame()
        state = game.new_initial_state()
        obs = {
            "playerId": 0,
            "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
        }
        result = get_legal_moves(obs)
        # 24 legal placements at the initial state.
        self.assertEqual(len(result), 24)
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, str)

    def test_empty_serialized(self):
        self.assertEqual(get_legal_moves({"serializedGameAndState": ""}), {})


# ---------------------------------------------------------------------------
# create_agent_fn integration
# ---------------------------------------------------------------------------


class _NineMensMorrisHarness:
    """Test-local GameHarness adapter; mirrors the prod wrapper shape."""

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
        agent = create_agent_fn(_NineMensMorrisHarness())

        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        game = nine_mens_morris_proxy.NineMensMorrisGame()
        state = game.new_initial_state()
        first_id = state.legal_actions()[0]
        first_legal = state.action_to_string(0, first_id)
        mock_litellm.completion.return_value = _make_mock_response(f'```json\n{{"move": "{_concise(first_id)}"}}\n```')
        agent = create_agent_fn(_NineMensMorrisHarness())

        obs = _make_observation(state, game, player_id=0)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal)
        self.assertEqual(result["status"], "OK")

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        game = nine_mens_morris_proxy.NineMensMorrisGame()
        state = game.new_initial_state()
        first_id = state.legal_actions()[0]
        first_legal = state.action_to_string(0, first_id)
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "999"}\n```'),
            _make_mock_response(f'```json\n{{"move": "{_concise(first_id)}"}}\n```'),
        ]
        agent = create_agent_fn(_NineMensMorrisHarness())

        obs = _make_observation(state, game, player_id=0)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal)
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("I cannot decide.")
        agent = create_agent_fn(_NineMensMorrisHarness())

        game = nine_mens_morris_proxy.NineMensMorrisGame()
        state = game.new_initial_state()
        obs = _make_observation(state, game, player_id=0)

        with self.assertRaises(ValueError):
            agent(obs, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_short_game_via_agent_fns(self, mock_litellm):
        """Drive a short game with two scripted LLM agents (always pick
        their first legal move), verifying the harness round-trips through
        pyspiel cleanly across multiple phases including captures."""
        mock_litellm.drop_params = True

        game = nine_mens_morris_proxy.NineMensMorrisGame()
        state = game.new_initial_state()

        def fake_completion(*, model, messages, **kwargs):
            del model, kwargs
            first_id = state.legal_actions()[0]
            return _make_mock_response(f'```json\n{{"move": "{_concise(first_id)}"}}\n```')

        mock_litellm.completion.side_effect = fake_completion
        agent_p0 = create_agent_fn(_NineMensMorrisHarness())
        agent_p1 = create_agent_fn(_NineMensMorrisHarness())

        for _ in range(30):
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
