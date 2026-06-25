"""Tests for the Markov Soccer LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.markov_soccer import (
    markov_soccer_proxy,
)
from kaggle_environments.envs.open_spiel_env.games.markov_soccer.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(
    state: markov_soccer_proxy.MarkovSoccerState,
    game: markov_soccer_proxy.MarkovSoccerGame,
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


def _initial_sim_state(
    game: markov_soccer_proxy.MarkovSoccerGame,
    ball_spawn_action: int = 2,
) -> markov_soccer_proxy.MarkovSoccerState:
    """Advance past the initial chance node to the first sim-move node."""
    state = game.new_initial_state()
    state.apply_action(ball_spawn_action)
    return state


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class ParseResponseTest(absltest.TestCase):
    legal = ["up", "down", "left", "right", "stand"]

    def test_parse_json_block(self):
        result = parse_response('```json\n{"move": "up"}\n```', self.legal)
        self.assertEqual(result.legal_action, "up")
        self.assertEqual(result.raw_action, "up")

    def test_parse_bare_json(self):
        result = parse_response('I think {"move": "right"} is best.', self.legal)
        self.assertEqual(result.legal_action, "right")

    def test_parse_case_insensitive(self):
        result = parse_response('```json\n{"move": "RIGHT"}\n```', self.legal)
        self.assertEqual(result.legal_action, "right")

    def test_parse_whitespace_tolerated(self):
        result = parse_response('```json\n{"move": "  left  "}\n```', self.legal)
        self.assertEqual(result.legal_action, "left")

    def test_prose_move_word_triggers_rethink(self):
        # The word "right" appears in the prose but no JSON answer was given.
        # The parser must NOT silently substitute the prose mention (ghost
        # antipattern) -- return None so the rethink loop fires.
        result = parse_response("I will move right this round.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_illegal_move_returns_raw(self):
        result = parse_response('```json\n{"move": "jump"}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "jump")

    def test_parse_no_match_returns_none(self):
        result = parse_response("I have no idea.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_returns_parse_result_type(self):
        result = parse_response('```json\n{"move": "stand"}\n```', self.legal)
        self.assertIsInstance(result, ParseResult)

    def test_parse_rethink_takes_last_json_block(self):
        # On rethink the model writes one answer, reconsiders, writes another.
        # Take the last block, not the first.
        response = '```json\n{"move": "up"}\n```\nWait, actually:\n```json\n{"move": "down"}\n```'
        result = parse_response(response, self.legal)
        self.assertEqual(result.legal_action, "down")

    def test_parse_rethink_takes_last_bare_json(self):
        response = '{"move": "left"} ... reconsidering ... {"move": "right"}'
        result = parse_response(response, self.legal)
        self.assertEqual(result.legal_action, "right")

    def test_multiple_move_words_in_prose_trigger_rethink(self):
        # Multiple legal-move words in the prose are NOT a structured answer.
        # The parser must NOT pick one (the ghost antipattern); return None
        # so the rethink loop asks the model for JSON.
        response = "I could go up, down, or left, but maybe right is best."
        result = parse_response(response, self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_illegal_json_does_not_ghost_substitute_from_prose(self):
        # The model's JSON answer ("jump") isn't legal. A legal "left" is
        # mentioned in the prose. The parser must NOT silently substitute
        # it (ghost antipattern). Surface raw_action so the rethink fires.
        response = 'I considered left but went bigger.\n```json\n{"move": "jump"}\n```'
        result = parse_response(response, self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "jump")


# ---------------------------------------------------------------------------
# generate_prompt
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game = markov_soccer_proxy.MarkovSoccerGame()
        self.state = _initial_sim_state(self.game, ball_spawn_action=2)

    def test_basic_prompt_contents(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Markov Soccer", prompt)
        self.assertIn("SIMULTANEOUSLY", prompt)
        self.assertIn("Player A", prompt)
        self.assertIn("4 row x 5 col", prompt)
        # All five actions named.
        for action in ("up", "down", "left", "right", "stand"):
            self.assertIn(action, prompt)

    def test_critical_steal_rule_present(self):
        # The non-obvious mechanic: defender walking into ball-holder
        # does NOT steal. The prompt MUST disclose this -- it's the rule
        # most likely to surprise a model that "knows soccer" generically.
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("does NOT steal", prompt)
        self.assertIn("wait for the ball-holder to walk", prompt)

    def test_goal_row_restriction_present(self):
        # Only rows 1 and 2 can score -- the prompt must say so.
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("row 1 or row 2", prompt)
        self.assertIn("Only rows 1 and 2", prompt)

    def test_player_label_swap(self):
        obs0 = _make_observation(self.state, self.game, player_id=0)
        obs1 = _make_observation(self.state, self.game, player_id=1)
        self.assertIn("You are Player A", generate_prompt(obs0, []))
        self.assertIn("You are Player B", generate_prompt(obs1, []))

    def test_per_player_goal_direction(self):
        # Engine: A scores on right edge (col == num_cols == 5);
        # B scores on left edge (col == -1). Verify both players see their
        # own goal as RIGHT/LEFT correctly.
        obs0 = _make_observation(self.state, self.game, player_id=0)
        obs1 = _make_observation(self.state, self.game, player_id=1)
        p0_prompt = generate_prompt(obs0, [])
        p1_prompt = generate_prompt(obs1, [])
        # Player A: own goal = RIGHT off col 4, opponent goal = LEFT off col 0
        self.assertIn("Your goal: walk RIGHT off column 4", p0_prompt)
        self.assertIn("Opponent's goal: walk LEFT off column 0", p0_prompt)
        # Player B: own goal = LEFT off col 0, opponent goal = RIGHT off col 4
        self.assertIn("Your goal: walk LEFT off column 0", p1_prompt)
        self.assertIn("Opponent's goal: walk RIGHT off column 4", p1_prompt)

    def test_loose_ball_status_rendered(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Default 4x5 grid + ball_spawn_action=2 => ball at (1, 2).
        self.assertIn("loose at (row 1, col 2)", prompt)
        self.assertIn("Neither player holds the ball yet", prompt)

    def test_held_ball_status_rendered(self):
        # Walk A onto the loose ball at (1, 2).
        # From (2, 1): up to (1, 1), then right to (1, 2) picks up the ball.
        # B stays put so it can't interfere.
        self.state.apply_actions([0, 4])  # A up, B stand
        # Auto-resolve initiative chance node.
        while self.state.is_chance_node():
            outcomes, _ = zip(*self.state.chance_outcomes())
            self.state.apply_action(outcomes[0])
        self.state.apply_actions([3, 4])  # A right (picks up ball), B stand
        while self.state.is_chance_node():
            outcomes, _ = zip(*self.state.chance_outcomes())
            self.state.apply_action(outcomes[0])

        obs_a = _make_observation(self.state, self.game, player_id=0)
        obs_b = _make_observation(self.state, self.game, player_id=1)
        prompt_a = generate_prompt(obs_a, [])
        prompt_b = generate_prompt(obs_b, [])
        self.assertIn("held by Player A", prompt_a)
        self.assertIn("YOU currently hold the ball", prompt_a)
        self.assertIn("held by Player A", prompt_b)
        self.assertIn("opponent currently holds the ball", prompt_b)

    def test_my_history_rendered(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, ["up", "right"])
        self.assertIn("up, right", prompt)

    def test_no_history_fallback(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("(none yet)", prompt)

    def test_rethink_suffix_illegal(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [], previous_response="I'll go jump", previous_action="jump")
        self.assertIn("You suggested", prompt)
        self.assertIn("jump", prompt)
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
            "legalActionStrings": ["up", "down", "left", "right", "stand"],
        }
        result = get_legal_moves(obs)
        self.assertEqual(
            result,
            {0: "up", 1: "down", 2: "left", 3: "right", 4: "stand"},
        )

    def test_from_serialized_state(self):
        game = markov_soccer_proxy.MarkovSoccerGame()
        state = _initial_sim_state(game)
        obs = {
            "playerId": 0,
            "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
        }
        result = get_legal_moves(obs)
        self.assertEqual(
            result,
            {0: "up", 1: "down", 2: "left", 3: "right", 4: "stand"},
        )

    def test_empty_serialized(self):
        self.assertEqual(get_legal_moves({"serializedGameAndState": ""}), {})

    def test_returns_typed_dict(self):
        result = get_legal_moves(
            {
                "legalActions": [0, 4],
                "legalActionStrings": ["up", "stand"],
            }
        )
        self.assertIsInstance(result, dict)
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, str)


# ---------------------------------------------------------------------------
# create_agent_fn integration
# ---------------------------------------------------------------------------


class _MarkovSoccerHarness:
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
        agent = create_agent_fn(_MarkovSoccerHarness())

        # Empty obs (no playerId / currentPlayer): treated as inactive probe.
        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_sim_move_player_treated_as_active(self, mock_litellm):
        """For sim-move games currentPlayer is -2 (SIMULTANEOUS) -- both
        players' agents must run, not return INACTIVE."""
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response('```json\n{"move": "right"}\n```')
        agent = create_agent_fn(_MarkovSoccerHarness())

        game = markov_soccer_proxy.MarkovSoccerGame()
        state = _initial_sim_state(game)
        # Verify the sim-move signal we depend on is what we expect.
        self.assertEqual(int(state.current_player()), -2)

        obs = _make_observation(state, game, player_id=1)
        result = agent(obs, {})

        self.assertEqual(result["submission"], 3)  # right == 3
        self.assertEqual(result["actionString"], "right")
        self.assertEqual(result["status"], "OK")

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response('```json\n{"move": "up"}\n```')
        agent = create_agent_fn(_MarkovSoccerHarness())

        game = markov_soccer_proxy.MarkovSoccerGame()
        state = _initial_sim_state(game)
        obs = _make_observation(state, game, player_id=0)

        result = agent(obs, {})

        self.assertEqual(result["submission"], 0)  # up == 0
        self.assertEqual(result["actionString"], "up")
        self.assertEqual(result["status"], "OK")
        self.assertIn("thoughts", result)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "jump"}\n```'),
            _make_mock_response('```json\n{"move": "stand"}\n```'),
        ]
        agent = create_agent_fn(_MarkovSoccerHarness())

        game = markov_soccer_proxy.MarkovSoccerGame()
        state = _initial_sim_state(game)
        obs = _make_observation(state, game, player_id=0)

        result = agent(obs, {})

        self.assertEqual(result["submission"], 4)  # stand == 4
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("I cannot decide.")
        agent = create_agent_fn(_MarkovSoccerHarness())

        game = markov_soccer_proxy.MarkovSoccerGame()
        state = _initial_sim_state(game)
        obs = _make_observation(state, game, player_id=0)

        with self.assertRaises(ValueError):
            agent(obs, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_terminal_step_returns_inactive(self, mock_litellm):
        mock_litellm.drop_params = True
        agent = create_agent_fn(_MarkovSoccerHarness())

        obs = {"isTerminal": True, "playerId": 0, "currentPlayer": -4}
        result = agent(obs, {})
        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_full_scripted_game_via_agent_fns(self, mock_litellm):
        """Drive a short game with two scripted LLM agents to confirm the
        harness round-trips through pyspiel cleanly. P0 always goes 'right'
        and grabs the ball + walks to its goal edge; P1 stays out of the way.
        """
        mock_litellm.drop_params = True

        def fake_completion(*, model, messages, **kwargs):
            del model, kwargs
            content = messages[0]["content"]
            # Player A walks right (toward its goal); Player B stands.
            if "You are Player A" in content:
                return _make_mock_response('```json\n{"move": "right"}\n```')
            return _make_mock_response('```json\n{"move": "stand"}\n```')

        mock_litellm.completion.side_effect = fake_completion
        agent_p0 = create_agent_fn(_MarkovSoccerHarness())
        agent_p1 = create_agent_fn(_MarkovSoccerHarness())

        game = markov_soccer_proxy.MarkovSoccerGame()
        state = game.new_initial_state()
        # Resolve initial chance node deterministically: ball at (2, 2).
        # (Chance action 3 spawns at the second listed ball start point.)
        state.apply_action(3)

        rounds = 0
        while not state.is_terminal() and rounds < 20:
            if state.is_chance_node():
                # Auto-resolve initiative chance node (prefer A-first).
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

        # With A walking right from (2, 1) onto the ball at (2, 2) and then
        # off the right edge (rows 1-2 are valid goal rows; row 2 qualifies),
        # A wins in a small number of rounds.
        self.assertTrue(state.is_terminal())
        returns = state.returns()
        self.assertEqual(returns[0], 1.0)
        self.assertEqual(returns[1], -1.0)


if __name__ == "__main__":
    absltest.main()
