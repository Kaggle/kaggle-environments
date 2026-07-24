"""Tests for the Capture the Flag LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.capture_the_flag import (
    capture_the_flag_proxy,
)
from kaggle_environments.envs.open_spiel_env.games.capture_the_flag.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(
    state: capture_the_flag_proxy.CaptureTheFlagState,
    game: capture_the_flag_proxy.CaptureTheFlagGame,
    player_id: int = 0,
) -> dict:
    """Build a harness-style observation dict from a proxy state."""
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        # Sim-move games surface this as PlayerId.SIMULTANEOUS == -2.
        "currentPlayer": int(state.current_player()),
        "isTerminal": state.is_terminal(),
        "legalActions": list(state.legal_actions(player_id)),
        "legalActionStrings": [state.action_to_string(player_id, a) for a in state.legal_actions(player_id)],
        "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
    }


def _resolve_chance(state, initiative: int = 0) -> None:
    """Advance past the post-move chance node, forcing a chosen initiative."""
    while state.is_chance_node():
        state.apply_action(initiative)


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class ParseResponseTest(absltest.TestCase):
    legal = ["North", "East", "South", "West", "Stay"]

    def test_parse_json_block(self):
        result = parse_response('```json\n{"move": "North"}\n```', self.legal)
        self.assertEqual(result.legal_action, "North")
        self.assertEqual(result.raw_action, "North")

    def test_parse_bare_json(self):
        result = parse_response('I think {"move": "East"} is best.', self.legal)
        self.assertEqual(result.legal_action, "East")

    def test_parse_case_insensitive(self):
        result = parse_response('```json\n{"move": "south"}\n```', self.legal)
        self.assertEqual(result.legal_action, "South")

    def test_parse_whitespace_tolerated(self):
        result = parse_response('```json\n{"move": "  West  "}\n```', self.legal)
        self.assertEqual(result.legal_action, "West")

    def test_prose_move_word_triggers_rethink(self):
        # A direction word in the prose is not a structured answer. The
        # parser must NOT silently substitute it (ghost antipattern).
        result = parse_response("I will move North this round.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_illegal_move_returns_raw(self):
        result = parse_response('```json\n{"move": "diagonal"}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "diagonal")

    def test_parse_no_match_returns_none(self):
        result = parse_response("I cannot decide.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_returns_parse_result_type(self):
        result = parse_response('```json\n{"move": "Stay"}\n```', self.legal)
        self.assertIsInstance(result, ParseResult)

    def test_parse_rethink_takes_last_json_block(self):
        response = '```json\n{"move": "North"}\n```\nWait, actually:\n```json\n{"move": "South"}\n```'
        result = parse_response(response, self.legal)
        self.assertEqual(result.legal_action, "South")

    def test_parse_rethink_takes_last_bare_json(self):
        response = '{"move": "North"} ... reconsidering ... {"move": "East"}'
        result = parse_response(response, self.legal)
        self.assertEqual(result.legal_action, "East")

    def test_multiple_move_words_in_prose_trigger_rethink(self):
        response = "I could go North, South, or East, but maybe West is best."
        result = parse_response(response, self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_illegal_json_does_not_ghost_substitute_from_prose(self):
        # Prose mentions a legal direction; JSON commits to an illegal one.
        # Parser must surface the illegal raw answer, NOT the prose token.
        response = 'I considered North but going diagonal.\n```json\n{"move": "diagonal"}\n```'
        result = parse_response(response, self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "diagonal")


# ---------------------------------------------------------------------------
# generate_prompt
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game = capture_the_flag_proxy.CaptureTheFlagGame()
        self.state = self.game.new_initial_state()

    def test_basic_prompt_contents(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Capture the Flag", prompt)
        self.assertIn("SIMULTANEOUSLY", prompt)
        self.assertIn("5 x 7 grid", prompt)
        for action in ("North", "East", "South", "West", "Stay"):
            self.assertIn(action, prompt)

    def test_critical_capture_precondition_present(self):
        # The non-obvious mechanic: arriving at your own base with the
        # opponent's flag only scores when your OWN flag is still home.
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("your own flag is still sitting at your home base", prompt)

    def test_scoring_requires_move_not_stay(self):
        # Engine detail: kStay short-circuits ResolveMove, so scoring is
        # never checked on Stay. If you're already standing at your base
        # you must leave and step back in to score. The prompt must warn.
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Standing still (Stay) never triggers a score", prompt)

    def test_tag_rule_present(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Manhattan-adjacent", prompt)
        self.assertIn("respawn", prompt)

    def test_tag_uses_final_positions_not_per_initiative(self):
        # Real strategic subtlety: ResolveTags fires ONCE after both moves
        # applied, based on final positions. A defender who moves adjacent
        # to a stationary/escaping carrier still tags. Grouping tagging
        # with the per-initiative mechanics would mislead the model.
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Post-turn resolution", prompt)
        self.assertIn("final positions", prompt)

    def test_hidden_initiative_disclosed(self):
        # Coin flip picks order after both moves are revealed; the model
        # must know it can't predict initiative AND what that means for
        # contested cells (only whoever resolves first lands there).
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("hidden coin flip", prompt)
        self.assertIn("cannot know the order in advance", prompt)
        self.assertIn("if you both target the same empty cell", prompt)

    def test_territory_split_rendered(self):
        # Default 7-wide grid: A owns 0-2, B owns 4-6, col 3 neutral.
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("A owns columns 0..2", prompt)
        self.assertIn("B owns columns 4..6", prompt)
        self.assertIn("Column 3 is neutral", prompt)

    def test_player_label_swap(self):
        obs0 = _make_observation(self.state, self.game, player_id=0)
        obs1 = _make_observation(self.state, self.game, player_id=1)
        self.assertIn("You are Player A", generate_prompt(obs0, []))
        self.assertIn("You are Player B", generate_prompt(obs1, []))

    def test_per_player_goal_target(self):
        obs0 = _make_observation(self.state, self.game, player_id=0)
        obs1 = _make_observation(self.state, self.game, player_id=1)
        # A hunts Flag B and returns to (2, 0); B hunts Flag A and returns to (2, 6).
        self.assertIn("carry Flag B to your base at (row 2, col 0)", generate_prompt(obs0, []))
        self.assertIn("carry Flag A to your base at (row 2, col 6)", generate_prompt(obs1, []))

    def test_flag_at_home_status_rendered(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Flag A (belongs to A): at home base", prompt)
        self.assertIn("Flag B (belongs to B): at home base", prompt)

    def test_carrier_status_rendered(self):
        # Force B to pick up A's flag. First walk A far away from its base so
        # that B doesn't get tagged the moment it steps onto A's flag (a
        # carrier Manhattan-adjacent to the defender inside the defender's
        # home is instantly respawned).
        for _ in range(2):
            self.state.apply_actions([0, 4])  # A North, B Stay
            _resolve_chance(self.state, 0)
        # A is now at (0, 0); A's flag is loose at (2, 0). Walk B west to it.
        for _ in range(6):
            self.state.apply_actions([4, 3])  # A Stay, B West
            _resolve_chance(self.state, 1)

        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Flag A (belongs to A): carried by Player B", prompt)

    def test_no_history_fallback(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("No rounds have been played yet.", prompt)

    def test_full_history_rendered_for_both_players(self):
        # Two completed rounds; both players' prompts see identical history.
        self.state.apply_actions([0, 4])  # R1: A North, B Stay
        self.state.apply_action(0)  # A first
        self.state.apply_actions([1, 3])  # R2: A East, B West
        self.state.apply_action(1)  # B first

        prompt_a = generate_prompt(_make_observation(self.state, self.game, 0), [])
        prompt_b = generate_prompt(_make_observation(self.state, self.game, 1), [])
        for prompt in (prompt_a, prompt_b):
            self.assertIn("Move history so far (both players, oldest first):", prompt)
            self.assertIn("Round 1: A=North, B=Stay (A's move resolved first)", prompt)
            self.assertIn("Round 2: A=East, B=West (B's move resolved first)", prompt)

    def test_full_history_ignores_per_agent_history_argument(self):
        self.state.apply_actions([0, 4])
        self.state.apply_action(0)
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, ["nonsense_token_xyz"])
        self.assertNotIn("nonsense_token_xyz", prompt)
        self.assertIn("Round 1: A=North, B=Stay", prompt)

    def test_score_and_horizon_disclosed(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Score: A=0, B=0", prompt)
        self.assertIn("first to 1 wins", prompt)
        self.assertIn("draw at 1000 rounds", prompt)

    def test_round_number_advances(self):
        obs0 = _make_observation(self.state, self.game, player_id=0)
        self.assertIn("Round 1 of at most 1000", generate_prompt(obs0, []))

        self.state.apply_actions([0, 0])
        self.state.apply_action(0)
        obs1 = _make_observation(self.state, self.game, player_id=0)
        self.assertIn("Round 2 of at most 1000", generate_prompt(obs1, []))

    def test_rethink_suffix_illegal(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [], previous_response="I'll go Northwest", previous_action="Northwest")
        self.assertIn("You suggested", prompt)
        self.assertIn("Northwest", prompt)
        self.assertIn("not a legal move", prompt)

    def test_rethink_suffix_unparsable(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [], previous_response="lots of reasoning here", previous_action=None)
        self.assertIn("No JSON answer could be parsed", prompt)
        self.assertIn("lots of reasoning here", prompt)


# ---------------------------------------------------------------------------
# get_legal_moves
# ---------------------------------------------------------------------------


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        obs = {
            "legalActions": [0, 1, 2, 3, 4],
            "legalActionStrings": ["North", "East", "South", "West", "Stay"],
        }
        result = get_legal_moves(obs)
        self.assertEqual(
            result,
            {0: "North", 1: "East", 2: "South", 3: "West", 4: "Stay"},
        )

    def test_from_serialized_state(self):
        game = capture_the_flag_proxy.CaptureTheFlagGame()
        state = game.new_initial_state()
        obs = {
            "playerId": 0,
            "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
        }
        result = get_legal_moves(obs)
        self.assertEqual(
            result,
            {0: "North", 1: "East", 2: "South", 3: "West", 4: "Stay"},
        )

    def test_empty_serialized(self):
        self.assertEqual(get_legal_moves({"serializedGameAndState": ""}), {})

    def test_returns_typed_dict(self):
        result = get_legal_moves(
            {
                "legalActions": [0, 4],
                "legalActionStrings": ["North", "Stay"],
            }
        )
        self.assertIsInstance(result, dict)
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, str)


# ---------------------------------------------------------------------------
# create_agent_fn integration
# ---------------------------------------------------------------------------


class _CaptureTheFlagHarness:
    """Test-local GameHarness adapter."""

    def get_legal_moves(self, observation):
        return get_legal_moves(observation)

    def make_prompt(self, observation, move_history, previous_response=None, previous_action=None):
        return generate_prompt(
            observation,
            move_history,
            previous_response=previous_response,
            previous_action=previous_action,
        )

    def parse_response(self, response, legal_action_strings, *, observation):
        del observation
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
        agent = create_agent_fn(_CaptureTheFlagHarness())

        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_sim_move_player_treated_as_active(self, mock_litellm):
        """currentPlayer is -2 (SIMULTANEOUS); both agents must run."""
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response('```json\n{"move": "East"}\n```')
        agent = create_agent_fn(_CaptureTheFlagHarness())

        game = capture_the_flag_proxy.CaptureTheFlagGame()
        state = game.new_initial_state()
        self.assertEqual(int(state.current_player()), -2)

        obs = _make_observation(state, game, player_id=1)
        result = agent(obs, {})

        self.assertEqual(result["submission"], 1)  # East == 1
        self.assertEqual(result["actionString"], "East")
        self.assertEqual(result["status"], "OK")

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response('```json\n{"move": "North"}\n```')
        agent = create_agent_fn(_CaptureTheFlagHarness())

        game = capture_the_flag_proxy.CaptureTheFlagGame()
        state = game.new_initial_state()
        obs = _make_observation(state, game, player_id=0)

        result = agent(obs, {})

        self.assertEqual(result["submission"], 0)  # North == 0
        self.assertEqual(result["actionString"], "North")
        self.assertEqual(result["status"], "OK")
        self.assertIn("thoughts", result)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "diagonal"}\n```'),
            _make_mock_response('```json\n{"move": "Stay"}\n```'),
        ]
        agent = create_agent_fn(_CaptureTheFlagHarness())

        game = capture_the_flag_proxy.CaptureTheFlagGame()
        state = game.new_initial_state()
        obs = _make_observation(state, game, player_id=0)

        result = agent(obs, {})

        self.assertEqual(result["submission"], 4)  # Stay == 4
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("I cannot decide.")
        agent = create_agent_fn(_CaptureTheFlagHarness())

        game = capture_the_flag_proxy.CaptureTheFlagGame()
        state = game.new_initial_state()
        obs = _make_observation(state, game, player_id=0)

        with self.assertRaises(ValueError):
            agent(obs, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_terminal_step_returns_inactive(self, mock_litellm):
        mock_litellm.drop_params = True
        agent = create_agent_fn(_CaptureTheFlagHarness())

        obs = {"isTerminal": True, "playerId": 0, "currentPlayer": -4}
        result = agent(obs, {})
        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_scripted_game_round_trips_through_pyspiel(self, mock_litellm):
        """Drive a few rounds with scripted LLM agents. P0 always Stays,
        P1 walks West. Neither wins — this just proves the harness rounds-
        trip cleanly against pyspiel without raising."""
        mock_litellm.drop_params = True

        def fake_completion(*, model, messages, **kwargs):
            del model, kwargs
            content = messages[0]["content"]
            if "You are Player A" in content:
                return _make_mock_response('```json\n{"move": "Stay"}\n```')
            return _make_mock_response('```json\n{"move": "West"}\n```')

        mock_litellm.completion.side_effect = fake_completion
        agent_p0 = create_agent_fn(_CaptureTheFlagHarness())
        agent_p1 = create_agent_fn(_CaptureTheFlagHarness())

        game = capture_the_flag_proxy.CaptureTheFlagGame()
        state = game.new_initial_state()

        rounds = 0
        while not state.is_terminal() and rounds < 8:
            if state.is_chance_node():
                state.apply_action(0)
                continue
            obs0 = _make_observation(state, game, player_id=0)
            obs1 = _make_observation(state, game, player_id=1)
            r0 = agent_p0(obs0, {})
            r1 = agent_p1(obs1, {})
            self.assertEqual(r0["status"], "OK")
            self.assertEqual(r1["status"], "OK")
            state.apply_actions([r0["submission"], r1["submission"]])
            rounds += 1

        # Game is far from over at 8 rounds; check it advanced cleanly.
        self.assertGreater(rounds, 0)


if __name__ == "__main__":
    absltest.main()
