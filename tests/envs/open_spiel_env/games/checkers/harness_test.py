"""Tests for the Checkers LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.checkers import (
    checkers_proxy,
)
from kaggle_environments.envs.open_spiel_env.games.checkers.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(
    state: checkers_proxy.CheckersState,
    game: checkers_proxy.CheckersGame,
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


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class ParseResponseTest(absltest.TestCase):
    legal = ["a3b4", "c3d4", "e3f4", "g3h4"]

    def test_parse_json_block(self):
        result = parse_response('```json\n{"move": "a3b4"}\n```', self.legal)
        self.assertEqual(result.legal_action, "a3b4")
        self.assertEqual(result.raw_action, "a3b4")

    def test_parse_bare_json(self):
        result = parse_response('I think {"move": "c3d4"} is best.', self.legal)
        self.assertEqual(result.legal_action, "c3d4")

    def test_prose_only_response_triggers_rethink(self):
        # No structured JSON. The parser must NOT guess at intent from a
        # move-shaped token in the prose -- return None and let rethink
        # ask the model to use the required JSON format.
        result = parse_response("I will play e3f4 this turn.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_case_insensitive(self):
        result = parse_response('```json\n{"move": "A3B4"}\n```', self.legal)
        self.assertEqual(result.legal_action, "a3b4")

    def test_parse_illegal_move_returns_raw(self):
        result = parse_response('```json\n{"move": "a1b2"}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "a1b2")

    def test_parse_no_match_returns_none(self):
        result = parse_response("I have no idea.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_returns_parse_result_type(self):
        result = parse_response('```json\n{"move": "a3b4"}\n```', self.legal)
        self.assertIsInstance(result, ParseResult)

    def test_parse_does_not_pick_unrelated_token(self):
        result = parse_response("I'm thinking about a1b2.", self.legal)
        self.assertIsNone(result.legal_action)

    def test_illegal_json_does_not_ghost_substitute_from_prose(self):
        # The model's JSON answer (z9z9) isn't legal. The parser must
        # NOT silently substitute a legal token from the prose -- return
        # None so the rethink loop asks the model to fix its answer.
        legal_example = self.legal[0]
        response = (
            f"I considered {legal_example} but ruled it out.\n"
            '```json\n{"move": "z9z9"}\n```'
        )
        result = parse_response(response, self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "z9z9")


# ---------------------------------------------------------------------------
# generate_prompt
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game = checkers_proxy.CheckersGame()
        self.state = self.game.new_initial_state()

    def test_basic_prompt_contents(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Checkers", prompt)
        self.assertIn("Player 0", prompt)
        self.assertIn("'o'", prompt)
        # The action-notation example uses "a3b4".
        self.assertIn("a3b4", prompt)

    def test_player_label_swap(self):
        first = self.state.legal_actions()[0]
        self.state.apply_action(first)
        obs1 = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs1, [])
        self.assertIn("Player 1", prompt)
        self.assertIn("'+'", prompt)

    def test_legal_moves_not_listed(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # The prompt deliberately omits the legal-move list so the model has
        # to reason about legality from the board alone. "a3b4" is excluded
        # because the action-notation example uses that token.
        for legal in obs["legalActionStrings"]:
            if legal == "a3b4":
                continue
            self.assertNotIn(legal, prompt)

    def test_board_ascii_includes_files_and_ranks(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("a b c d e f g h", prompt)
        # Both bottom and top rank labels should appear.
        self.assertIn("1 ", prompt)
        self.assertIn("8 ", prompt)

    def test_last_move_rendered_after_play(self):
        first = self.state.legal_actions()[0]
        first_str = self.state.action_to_string(0, first)
        self.state.apply_action(first)
        obs1 = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs1, [])
        self.assertIn(f"Last move played: {first_str}", prompt)

    def test_move_history_rendered(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, ["a3b4", "f6e5"])
        self.assertIn("a3b4, f6e5", prompt)

    def test_move_history_none_when_empty(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Moves you have played so far: None", prompt)

    def test_piece_counts_rendered(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Initial position: 12 men per side, no kings.
        self.assertIn("Player 0 men=12", prompt)
        self.assertIn("Player 1 men=12", prompt)

    def test_own_pieces_listed_for_player_0(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Initial position: player 0 men on ranks 1-3 dark squares.
        for sq in ("a1", "c1", "e1", "g1", "b2", "d2", "f2", "h2",
                   "a3", "c3", "e3", "g3"):
            self.assertIn(sq, prompt)
        self.assertIn("Your kings ('O') are at: (none)", prompt)

    def test_own_pieces_listed_for_player_1(self):
        first = self.state.legal_actions()[0]
        self.state.apply_action(first)
        obs = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs, [])
        for sq in ("b6", "d6", "f6", "h6", "a7", "c7", "e7", "g7",
                   "b8", "d8", "f8", "h8"):
            self.assertIn(sq, prompt)
        self.assertIn("Your kings ('*') are at: (none)", prompt)

    def test_opponent_pieces_listed(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Player 0's opponent is Player 1; opponent men ('+') start on
        # ranks 6-8 dark squares; no opponent kings yet.
        self.assertIn("Opponent men ('+') are at:", prompt)
        for sq in ("b6", "d6", "f6", "h6", "a7", "c7", "e7", "g7",
                   "b8", "d8", "f8", "h8"):
            self.assertIn(sq, prompt)
        self.assertIn("Opponent kings ('*') are at: (none)", prompt)

    def test_opponent_pieces_listed_for_player_1(self):
        first = self.state.legal_actions()[0]
        self.state.apply_action(first)
        obs = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs, [])
        # Player 1's opponent is Player 0; opponent men ('o') character.
        self.assertIn("Opponent men ('o') are at:", prompt)
        self.assertIn("Opponent kings ('O') are at: (none)", prompt)

    def test_draw_rule_disclosed(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # The 40-ply no-capture draw rule must be in the prompt -- it's a
        # real terminal path that decided ~9% of recorded games.
        self.assertIn("40", prompt)
        self.assertIn("draw", prompt.lower())
        self.assertIn("capture", prompt.lower())

    def test_captures_flag_no_at_start(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Captures available this turn: no", prompt)
        self.assertNotIn("MUST take a capture this turn", prompt)

    def test_captures_flag_yes_when_capture_available(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        # Synthesize a capture by overriding legal moves with a 2-rank jump.
        obs["legalActions"] = [0]
        obs["legalActionStrings"] = ["c3e5"]
        prompt = generate_prompt(obs, [])
        self.assertIn("Captures available this turn: yes", prompt)
        self.assertIn("MUST take a capture this turn", prompt)

    def test_forward_direction_explained_per_player(self):
        obs0 = _make_observation(self.state, self.game, player_id=0)
        self.assertIn("toward rank 8", generate_prompt(obs0, []))

        first = self.state.legal_actions()[0]
        self.state.apply_action(first)
        obs1 = _make_observation(self.state, self.game, player_id=1)
        self.assertIn("toward rank 1", generate_prompt(obs1, []))

    def test_rethink_suffix(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [], previous_response="I'll play z9z9", previous_action="z9z9")
        self.assertIn("Your previous response was", prompt)
        self.assertIn("z9z9", prompt)
        self.assertIn("not a legal move", prompt)

    def test_no_rethink_on_first_attempt(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertNotIn("Your previous response was", prompt)


# ---------------------------------------------------------------------------
# get_legal_moves
# ---------------------------------------------------------------------------


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        obs = {
            "legalActions": [322, 336, 338],
            "legalActionStrings": ["a3b4", "c3b4", "c3d4"],
        }
        result = get_legal_moves(obs)
        self.assertEqual(result, {322: "a3b4", 336: "c3b4", 338: "c3d4"})

    def test_from_serialized_state(self):
        game = checkers_proxy.CheckersGame()
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
            # All checkers action strings are 4 characters: <from><to>.
            self.assertEqual(len(v), 4)

    def test_empty_serialized(self):
        self.assertEqual(get_legal_moves({"serializedGameAndState": ""}), {})


# ---------------------------------------------------------------------------
# create_agent_fn integration
# ---------------------------------------------------------------------------


class _CheckersHarness:
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

    def parse_response(self, response, legal_action_strings):
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
        agent = create_agent_fn(_CheckersHarness())

        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        game = checkers_proxy.CheckersGame()
        state = game.new_initial_state()
        first_legal = state.action_to_string(0, state.legal_actions()[0])
        mock_litellm.completion.return_value = _make_mock_response(f'```json\n{{"move": "{first_legal}"}}\n```')
        agent = create_agent_fn(_CheckersHarness())

        obs = _make_observation(state, game, player_id=0)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal)
        self.assertEqual(result["status"], "OK")
        self.assertIn("thoughts", result)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        game = checkers_proxy.CheckersGame()
        state = game.new_initial_state()
        first_legal = state.action_to_string(0, state.legal_actions()[0])
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "z9z9"}\n```'),
            _make_mock_response(f'```json\n{{"move": "{first_legal}"}}\n```'),
        ]
        agent = create_agent_fn(_CheckersHarness())

        obs = _make_observation(state, game, player_id=0)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal)
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("I cannot decide.")
        agent = create_agent_fn(_CheckersHarness())

        game = checkers_proxy.CheckersGame()
        state = game.new_initial_state()
        obs = _make_observation(state, game, player_id=0)

        with self.assertRaises(ValueError):
            agent(obs, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_short_game_via_agent_fns(self, mock_litellm):
        """Drive a short Checkers game with two scripted LLM agents that
        always pick their first legal move, verifying the harness
        round-trips through pyspiel cleanly."""
        mock_litellm.drop_params = True

        game = checkers_proxy.CheckersGame()
        state = game.new_initial_state()

        def fake_completion(*, model, messages, **kwargs):
            del model, kwargs
            content = messages[0]["content"]
            player_id = 0 if "Player 0" in content else 1
            first = state.action_to_string(player_id, state.legal_actions()[0])
            return _make_mock_response(f'```json\n{{"move": "{first}"}}\n```')

        mock_litellm.completion.side_effect = fake_completion
        agent_p0 = create_agent_fn(_CheckersHarness())
        agent_p1 = create_agent_fn(_CheckersHarness())

        for _ in range(20):
            if state.is_terminal():
                break
            cp = int(state.current_player())
            agent = agent_p0 if cp == 0 else agent_p1
            obs = _make_observation(state, game, player_id=cp)
            result = agent(obs, {})
            self.assertEqual(result["status"], "OK")
            state.apply_action(result["submission"])

        # Game may not terminate in 20 moves; just confirm we played without
        # raising and the state is still consistent.
        self.assertGreater(state.move_number(), 0)


if __name__ == "__main__":
    absltest.main()
