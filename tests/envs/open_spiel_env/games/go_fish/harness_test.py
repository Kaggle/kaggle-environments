"""Tests for the Go Fish LLM harness."""

import random
from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.go_fish import (
    go_fish_proxy,
)
from kaggle_environments.envs.open_spiel_env.games.go_fish.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _advance_chance(state, rng: random.Random) -> None:
    """Resolve chance nodes (deal/fish/draw) with a seeded RNG."""
    while not state.is_terminal() and state.is_chance_node():
        outcomes = state.chance_outcomes()
        action = rng.choices([o[0] for o in outcomes], weights=[o[1] for o in outcomes])[0]
        state.apply_action(action)


def _make_ask_state(seed: int = 7, plies: int = 0):
    """Return (game, state) advanced to an Ask node (a player's decision).

    ``plies`` additional random Ask moves are played first (each followed by
    chance resolution) so tests can exercise mid-game states with events.
    """
    game = go_fish_proxy.GoFishGame()
    state = game.new_initial_state()
    rng = random.Random(seed)
    _advance_chance(state, rng)
    for _ in range(plies):
        if state.is_terminal() or state.current_player() < 0:
            break
        state.apply_action(rng.choice(state.legal_actions()))
        _advance_chance(state, rng)
    return game, state


def _make_observation(state, game, player_id: int | None = None) -> dict:
    """Build a harness-style observation dict from a proxy state."""
    if player_id is None:
        player_id = int(state.current_player())
    legal = list(state.legal_actions())
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "currentPlayer": int(state.current_player()),
        "isTerminal": state.is_terminal(),
        "legalActions": legal,
        "legalActionStrings": [state.action_to_string(a) for a in legal],
        "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
    }


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class ParseResponseTest(absltest.TestCase):
    legal = ["1a", "1b", "1e", "1j", "1m"]

    def test_parse_json_block(self):
        result = parse_response('```json\n{"move": "1a"}\n```', self.legal)
        self.assertEqual(result.legal_action, "1a")
        self.assertEqual(result.raw_action, "1a")

    def test_parse_bare_json(self):
        result = parse_response('I think {"move": "1b"} is best.', self.legal)
        self.assertEqual(result.legal_action, "1b")

    def test_parse_case_insensitive(self):
        result = parse_response('```json\n{"move": "1A"}\n```', self.legal)
        self.assertEqual(result.legal_action, "1a")

    def test_parse_illegal_move_returns_raw(self):
        result = parse_response('```json\n{"move": "1c"}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "1c")

    def test_prose_only_response_triggers_rethink(self):
        result = parse_response("I will ask Player 1 for aces (1a).", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_multiple_json_last_wins(self):
        response = 'First draft: {"move": "1a"}\nOn reflection: {"move": "1j"}'
        result = parse_response(response, self.legal)
        self.assertEqual(result.legal_action, "1j")

    def test_parse_returns_parse_result_type(self):
        result = parse_response('```json\n{"move": "1a"}\n```', self.legal)
        self.assertIsInstance(result, ParseResult)

    def test_human_rank_label_tolerated(self):
        # Model names the rank by its human label instead of the action
        # letter: "1K" -> ask Player 1 for Kings -> letter 'm'.
        result = parse_response('```json\n{"move": "1K"}\n```', self.legal)
        self.assertEqual(result.legal_action, "1m")

    def test_numeric_rank_label_tolerated(self):
        # "1, 10" -> ask Player 1 for tens -> letter 'j'.
        result = parse_response('```json\n{"move": "1, 10"}\n```', self.legal)
        self.assertEqual(result.legal_action, "1j")

    def test_separators_tolerated(self):
        result = parse_response('```json\n{"move": "1-a"}\n```', self.legal)
        self.assertEqual(result.legal_action, "1a")

    def test_illegal_json_does_not_ghost_substitute_from_prose(self):
        # The model discusses a legal move in prose, then commits to an
        # illegal one in JSON. The parser must NOT substitute the prose token.
        response = 'I considered 1a but ruled it out.\n```json\n{"move": "9z"}\n```'
        result = parse_response(response, self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "9z")


# ---------------------------------------------------------------------------
# generate_prompt
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game, self.state = _make_ask_state(seed=7)

    def test_basic_prompt_contents(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        self.assertIn("Go Fish", prompt)
        self.assertIn("book", prompt.lower())
        self.assertIn("go fish", prompt.lower())

    def test_go_fish_hit_continues_turn(self):
        # Engine (go_fish.cc:256-266): drawing the exact rank you asked for on
        # a "go fish" does NOT end your turn -- you ask again. The prompt must
        # not claim the turn simply ends.
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        self.assertIn("very rank you asked for, you take another turn", prompt)

    def test_rank_legend_present(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        # Standard 13-rank deck: a=A ... m=K.
        self.assertIn("a=A", prompt)
        self.assertIn("j=10", prompt)
        self.assertIn("m=K", prompt)

    def test_hand_annotated_with_ask_letters(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        self.assertIn("ask-letter", prompt)
        # Player 0's opening hand (seed 7) contains an Ace.
        self.assertIn("A: ", prompt)

    def test_player_identity_rendered(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("You are Player 0", prompt)

    def test_does_not_leak_opponent_hand(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Opponent info is a card/book count only, never their cards.
        self.assertIn("You do NOT see other players' hands", prompt)
        self.assertIn("Player 1:", prompt)

    def test_legal_moves_not_listed(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        # The prompt derives askable ranks from the hand; it should not dump
        # the raw legal-action list. Allow the concrete example token through.
        legal = obs["legalActionStrings"]
        example_ok = {legal[0]} if legal else set()
        listed = [s for s in legal if s not in example_ok and f'"{s}"' in prompt]
        # No verbatim legal-list block: at most the single example may appear.
        self.assertLessEqual(len(listed), 0)

    def test_events_rendered_after_play(self):
        # A mid-game state produces opponent events since the last turn.
        game, state = _make_ask_state(seed=7, plies=3)
        obs = _make_observation(state, game)
        prompt = generate_prompt(obs, [])
        self.assertIn("Events since your last turn", prompt)

    def test_move_history_rendered(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, ["1a", "1b"])
        self.assertIn("1a, 1b", prompt)

    def test_move_history_none_when_empty(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        self.assertIn("Moves you have played so far: None", prompt)

    def test_example_move_is_legal(self):
        # The concrete example in the format section must itself be a legal
        # move so the prompt never advises an illegal action.
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        legal = set(obs["legalActionStrings"])
        # Example rendered as `{"move": "<tok>"}`.
        import re as _re

        tokens = _re.findall(r'\{"move": "([^"]+)"\}', prompt)
        # First is the template placeholder; the concrete example is the last.
        concrete = [t for t in tokens if not t.startswith("<")]
        self.assertTrue(concrete)
        self.assertIn(concrete[-1], legal)

    def test_rethink_illegal_suffix(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [], previous_response="I'll play 9z", previous_action="9z")
        self.assertIn("You suggested", prompt)
        self.assertIn("9z", prompt)
        self.assertIn("not a legal move", prompt)

    def test_rethink_unparsable_suffix(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [], previous_response="hmm no json here", previous_action=None)
        self.assertIn("No JSON answer could be parsed", prompt)
        self.assertIn("hmm no json here", prompt)

    def test_no_rethink_on_first_attempt(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        self.assertNotIn("not a legal move", prompt)
        self.assertNotIn("No JSON answer could be parsed", prompt)


# ---------------------------------------------------------------------------
# get_legal_moves
# ---------------------------------------------------------------------------


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        obs = {
            "legalActions": [13, 14, 17],
            "legalActionStrings": ["1a", "1b", "1e"],
        }
        result = get_legal_moves(obs)
        self.assertEqual(result, {13: "1a", 14: "1b", 17: "1e"})

    def test_from_serialized_state(self):
        game, state = _make_ask_state(seed=7)
        obs = {
            "playerId": int(state.current_player()),
            "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
        }
        result = get_legal_moves(obs)
        self.assertGreater(len(result), 0)
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, str)
            # Go Fish ask strings are two chars: <target><letter>.
            self.assertEqual(len(v), 2)

    def test_empty_serialized(self):
        self.assertEqual(get_legal_moves({"serializedGameAndState": ""}), {})


# ---------------------------------------------------------------------------
# create_agent_fn integration
# ---------------------------------------------------------------------------


class _GoFishHarness:
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
        agent = create_agent_fn(_GoFishHarness())

        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        game, state = _make_ask_state(seed=7)
        first_legal = state.action_to_string(state.legal_actions()[0])
        mock_litellm.completion.return_value = _make_mock_response(f'```json\n{{"move": "{first_legal}"}}\n```')
        agent = create_agent_fn(_GoFishHarness())

        obs = _make_observation(state, game)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal)
        self.assertEqual(result["status"], "OK")
        self.assertIn("thoughts", result)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        game, state = _make_ask_state(seed=7)
        first_legal = state.action_to_string(state.legal_actions()[0])
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "9z"}\n```'),
            _make_mock_response(f'```json\n{{"move": "{first_legal}"}}\n```'),
        ]
        agent = create_agent_fn(_GoFishHarness())

        obs = _make_observation(state, game)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal)
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("I cannot decide.")
        agent = create_agent_fn(_GoFishHarness())

        game, state = _make_ask_state(seed=7)
        obs = _make_observation(state, game)

        with self.assertRaises(ValueError):
            agent(obs, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_short_game_via_agent_fns(self, mock_litellm):
        """Drive a Go Fish game with scripted agents that always pick their
        first legal ask, verifying the harness round-trips through pyspiel."""
        mock_litellm.drop_params = True

        game, state = _make_ask_state(seed=7)
        rng = random.Random(123)

        def fake_completion(*, model, messages, **kwargs):
            del model, kwargs
            first = state.action_to_string(state.legal_actions()[0])
            return _make_mock_response(f'```json\n{{"move": "{first}"}}\n```')

        mock_litellm.completion.side_effect = fake_completion
        agent_p0 = create_agent_fn(_GoFishHarness())
        agent_p1 = create_agent_fn(_GoFishHarness())

        for _ in range(30):
            _advance_chance(state, rng)
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
