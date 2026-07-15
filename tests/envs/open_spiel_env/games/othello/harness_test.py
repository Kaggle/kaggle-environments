"""Tests for the Reversi (open_spiel_othello) LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.othello import (
    othello_proxy,
)
from kaggle_environments.envs.open_spiel_env.games.othello.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(
    state: othello_proxy.OthelloState,
    game: othello_proxy.OthelloGame,
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
    legal = ["d3", "c4", "f5", "e6"]

    def test_parse_json_block(self):
        result = parse_response('```json\n{"move": "d3"}\n```', self.legal)
        self.assertEqual(result.legal_action, "d3")
        self.assertEqual(result.raw_action, "d3")

    def test_parse_bare_json(self):
        result = parse_response('I think {"move": "c4"} is best.', self.legal)
        self.assertEqual(result.legal_action, "c4")

    def test_parse_case_insensitive(self):
        result = parse_response('```json\n{"move": "D3"}\n```', self.legal)
        self.assertEqual(result.legal_action, "d3")

    def test_parse_pass_action(self):
        result = parse_response('```json\n{"move": "pass"}\n```', ["pass"])
        self.assertEqual(result.legal_action, "pass")

    def test_parse_illegal_move_returns_raw(self):
        result = parse_response('```json\n{"move": "z9"}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "z9")

    def test_parse_no_match_returns_none(self):
        result = parse_response("I have no idea.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_prose_only_response_triggers_rethink(self):
        # No structured JSON. The parser must NOT guess intent from a
        # move-shaped token in the prose -- return None so the rethink
        # loop asks the model to use the required JSON format.
        result = parse_response("I will play f5 this turn.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_multiple_json_blocks_last_wins(self):
        # Model draft-then-revises; last block is the intent.
        response = '```json\n{"move": "d3"}\n```\nActually, wait, better:\n```json\n{"move": "e6"}\n```'
        result = parse_response(response, self.legal)
        self.assertEqual(result.legal_action, "e6")

    def test_parse_returns_parse_result_type(self):
        result = parse_response('```json\n{"move": "d3"}\n```', self.legal)
        self.assertIsInstance(result, ParseResult)

    def test_illegal_json_does_not_ghost_substitute_from_prose(self):
        # The model's JSON answer (z9) isn't legal. The parser must NOT
        # silently substitute a legal token from the prose -- return None
        # so the rethink loop asks the model to fix its answer.
        response = f'I considered {self.legal[0]} but ruled it out.\n```json\n{{"move": "z9"}}\n```'
        result = parse_response(response, self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "z9")


# ---------------------------------------------------------------------------
# generate_prompt
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game = othello_proxy.OthelloGame()
        self.state = self.game.new_initial_state()

    def test_basic_prompt_contents(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Reversi", prompt)
        self.assertIn("Player 0", prompt)
        self.assertIn("Black", prompt)
        self.assertIn("'x'", prompt)
        # The action-notation example uses "d3".
        self.assertIn("d3", prompt)

    def test_player_label_swap(self):
        first = self.state.legal_actions()[0]
        self.state.apply_action(first)
        obs1 = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs1, [])
        self.assertIn("Player 1", prompt)
        self.assertIn("White", prompt)
        self.assertIn("'o'", prompt)

    def test_pass_rule_disclosed(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # "pass" as a legal move must be described somewhere; it decides
        # whether the model tries to invent a placement it doesn't have.
        self.assertIn("pass", prompt.lower())

    def test_pass_note_only_when_must_pass(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Initial state: Black has 4 legal placements, must_pass is false.
        self.assertNotIn("only legal move is `pass`", prompt)

    def test_legal_moves_not_listed(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # The prompt deliberately omits the legal-move list so the model
        # has to derive legality from the board. Exclude "d3" because the
        # notation example uses it.
        for legal in obs["legalActionStrings"]:
            if legal == "d3":
                continue
            self.assertNotIn(f'"{legal}"', prompt)

    def test_board_ascii_includes_files_and_ranks(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("a b c d e f g h", prompt)
        # Both extreme rank labels should appear.
        self.assertIn("1 ", prompt)
        self.assertIn("8 ", prompt)

    def test_initial_disks_rendered_in_center(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Row 4 (rank 4) has White at d4 and Black at e4: ". . . o x . . ."
        self.assertIn(". . . o x . . .", prompt)
        self.assertIn(". . . x o . . .", prompt)

    def test_disk_counts_rendered(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Initial position: 2 disks each.
        self.assertIn("Black ('x') = 2", prompt)
        self.assertIn("White ('o') = 2", prompt)

    def test_last_move_rendered_after_play(self):
        first = self.state.legal_actions()[0]
        first_str = self.state.action_to_string(0, first)
        self.state.apply_action(first)
        obs1 = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs1, [])
        self.assertIn(f"Last move played: {first_str}", prompt)

    def test_full_game_move_history_rendered(self):
        # Play d3 then c3 and confirm the prompt shows BOTH moves (the
        # per-agent framework `move_history` arg would show only Black's).
        self.state.apply_action(19)  # d3 (Black)
        self.state.apply_action(18)  # c3 (White)
        obs = _make_observation(self.state, self.game, player_id=0)
        # Pass a stale per-agent history to prove the harness sources from
        # the proxy, not from the argument.
        prompt = generate_prompt(obs, ["deliberately-stale"])
        self.assertIn("d3, c3", prompt)
        self.assertNotIn("deliberately-stale", prompt)

    def test_move_history_none_when_empty(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Moves played so far this game (both players, oldest first): None", prompt)

    def test_direction_language_symmetric(self):
        # Reversi is symmetric; the prompt should NOT bake in a
        # player-specific direction. Diff P0 and P1 prompts and confirm
        # only the {player_id, my_piece, opp_piece, colour} substitutions
        # differ, not any directional / orientation claim.
        obs0 = _make_observation(self.state, self.game, player_id=0)
        p0 = generate_prompt(obs0, [])
        # After a move so P1 becomes the current player.
        self.state.apply_action(self.state.legal_actions()[0])
        obs1 = _make_observation(self.state, self.game, player_id=1)
        p1 = generate_prompt(obs1, [])
        # Both prompts should describe rank 1 as the TOP row identically.
        self.assertIn("rank 1 is the top row", p0)
        self.assertIn("rank 1 is the top row", p1)

    def test_rethink_illegal_leads_with_previous_action(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [], previous_response="I'll play z9", previous_action="z9")
        self.assertIn("You suggested", prompt)
        self.assertIn("z9", prompt)
        self.assertIn("not a legal move", prompt)

    def test_rethink_unparsable_shows_previous_response(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [], previous_response="I forgot the JSON.", previous_action=None)
        self.assertIn("Your previous response ended with", prompt)
        self.assertIn("No JSON answer could be parsed", prompt)

    def test_no_rethink_on_first_attempt(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertNotIn("Your previous response", prompt)
        self.assertNotIn("You suggested", prompt)


# ---------------------------------------------------------------------------
# get_legal_moves
# ---------------------------------------------------------------------------


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        obs = {
            "legalActions": [19, 26, 37, 44],
            "legalActionStrings": ["d3", "c4", "f5", "e6"],
        }
        result = get_legal_moves(obs)
        self.assertEqual(result, {19: "d3", 26: "c4", 37: "f5", 44: "e6"})

    def test_from_serialized_state(self):
        game = othello_proxy.OthelloGame()
        state = game.new_initial_state()
        obs = {
            "playerId": 0,
            "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
        }
        result = get_legal_moves(obs)
        # Initial state: 4 legal opening moves for Black.
        self.assertEqual(len(result), 4)
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, str)
        # All 4 openings are 2-character algebraic squares.
        self.assertEqual({v for v in result.values()}, {"d3", "c4", "f5", "e6"})

    def test_empty_serialized(self):
        self.assertEqual(get_legal_moves({"serializedGameAndState": ""}), {})


# ---------------------------------------------------------------------------
# create_agent_fn integration
# ---------------------------------------------------------------------------


class _ReversiHarness:
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
        agent = create_agent_fn(_ReversiHarness())

        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        game = othello_proxy.OthelloGame()
        state = game.new_initial_state()
        first_legal = state.action_to_string(0, state.legal_actions()[0])
        mock_litellm.completion.return_value = _make_mock_response(f'```json\n{{"move": "{first_legal}"}}\n```')
        agent = create_agent_fn(_ReversiHarness())

        obs = _make_observation(state, game, player_id=0)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal)
        self.assertEqual(result["status"], "OK")
        self.assertIn("thoughts", result)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        game = othello_proxy.OthelloGame()
        state = game.new_initial_state()
        first_legal = state.action_to_string(0, state.legal_actions()[0])
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "z9"}\n```'),
            _make_mock_response(f'```json\n{{"move": "{first_legal}"}}\n```'),
        ]
        agent = create_agent_fn(_ReversiHarness())

        obs = _make_observation(state, game, player_id=0)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal)
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("I cannot decide.")
        agent = create_agent_fn(_ReversiHarness())

        game = othello_proxy.OthelloGame()
        state = game.new_initial_state()
        obs = _make_observation(state, game, player_id=0)

        with self.assertRaises(ValueError):
            agent(obs, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_short_game_via_agent_fns(self, mock_litellm):
        """Drive a short Reversi game with two scripted LLM agents that
        always pick their first legal move, verifying the harness
        round-trips through pyspiel cleanly."""
        mock_litellm.drop_params = True

        game = othello_proxy.OthelloGame()
        state = game.new_initial_state()

        def fake_completion(*, model, messages, **kwargs):
            del model, kwargs
            content = messages[0]["content"]
            player_id = 0 if "You are Player 0" in content else 1
            first = state.action_to_string(player_id, state.legal_actions()[0])
            return _make_mock_response(f'```json\n{{"move": "{first}"}}\n```')

        mock_litellm.completion.side_effect = fake_completion
        agent_p0 = create_agent_fn(_ReversiHarness())
        agent_p1 = create_agent_fn(_ReversiHarness())

        for _ in range(20):
            if state.is_terminal():
                break
            cp = int(state.current_player())
            agent = agent_p0 if cp == 0 else agent_p1
            obs = _make_observation(state, game, player_id=cp)
            result = agent(obs, {})
            self.assertEqual(result["status"], "OK")
            state.apply_action(result["submission"])

        # Game may not terminate in 20 moves; confirm we played without
        # raising and the state advanced.
        self.assertGreater(state.move_number(), 0)


if __name__ == "__main__":
    absltest.main()
