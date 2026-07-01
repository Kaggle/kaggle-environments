"""Tests for the Shogi LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.shogi import shogi_proxy
from kaggle_environments.envs.open_spiel_env.games.shogi.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(
    state: shogi_proxy.ShogiState,
    game: shogi_proxy.ShogiGame,
    player_id: int = 0,
) -> dict:
    """Build a harness-style observation dict from a proxy state."""
    legal = list(state.legal_actions())
    current = int(state.current_player())
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "currentPlayer": current,
        "isTerminal": state.is_terminal(),
        "legalActions": legal,
        "legalActionStrings": [state.action_to_string(current, a) for a in legal],
        "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
    }


def _apply_sequence(state: shogi_proxy.ShogiState, moves: list[str]) -> None:
    """Apply a list of USI move strings, raising if any is not legal."""
    for m in moves:
        for a in state.legal_actions():
            if state.action_to_string(int(state.current_player()), a) == m:
                state.apply_action(a)
                break
        else:
            raise AssertionError(f"Move {m!r} not legal from current state")


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class ParseResponseTest(absltest.TestCase):
    legal = ["7g7f", "2g2f", "8h2b+", "P*5e"]

    def test_parse_json_block(self):
        result = parse_response('```json\n{"move": "7g7f"}\n```', self.legal)
        self.assertEqual(result.legal_action, "7g7f")
        self.assertEqual(result.raw_action, "7g7f")

    def test_parse_bare_json(self):
        result = parse_response('I think {"move": "2g2f"} is best.', self.legal)
        self.assertEqual(result.legal_action, "2g2f")

    def test_parse_promotion(self):
        result = parse_response('```json\n{"move": "8h2b+"}\n```', self.legal)
        self.assertEqual(result.legal_action, "8h2b+")

    def test_parse_drop(self):
        result = parse_response('```json\n{"move": "P*5e"}\n```', self.legal)
        self.assertEqual(result.legal_action, "P*5e")

    def test_parse_case_insensitive(self):
        result = parse_response('```json\n{"move": "7G7F"}\n```', self.legal)
        self.assertEqual(result.legal_action, "7g7f")

    def test_prose_only_response_triggers_rethink(self):
        # No structured JSON -- parser must NOT guess an intent from a
        # move-shaped token in prose; return None and let rethink ask
        # the model to use the required JSON format.
        result = parse_response("I will play 7g7f this turn.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_illegal_move_returns_raw(self):
        result = parse_response('```json\n{"move": "9a9b"}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "9a9b")

    def test_parse_no_match_returns_none(self):
        result = parse_response("I have no idea.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_multiple_json_blocks_last_wins(self):
        response = '```json\n{"move": "9a9b"}\n```\nOn reflection, I\'ll play:\n```json\n{"move": "7g7f"}\n```'
        result = parse_response(response, self.legal)
        self.assertEqual(result.legal_action, "7g7f")

    def test_parse_returns_parse_result_type(self):
        result = parse_response('```json\n{"move": "7g7f"}\n```', self.legal)
        self.assertIsInstance(result, ParseResult)

    def test_illegal_json_does_not_ghost_substitute_from_prose(self):
        # The model discussed 7g7f in prose but committed to an illegal
        # move in the JSON answer. The parser must NOT silently
        # substitute the prose token.
        response = 'I considered 7g7f but ruled it out.\n```json\n{"move": "z9z9"}\n```'
        result = parse_response(response, self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "z9z9")


# ---------------------------------------------------------------------------
# generate_prompt
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game = shogi_proxy.ShogiGame()
        self.state = self.game.new_initial_state()

    def test_basic_prompt_contents(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Shogi", prompt)
        self.assertIn("Player 0", prompt)
        self.assertIn("Sente", prompt)
        # The action-notation example uses 7g7f.
        self.assertIn("7g7f", prompt)

    def test_player_label_swap(self):
        _apply_sequence(self.state, ["7g7f"])
        obs1 = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs1, [])
        self.assertIn("Player 1", prompt)
        self.assertIn("Gote", prompt)
        self.assertIn("lowercase", prompt)

    def test_player_asymmetric_text_differs(self):
        # The "You are Player N (...)" line must differ for the two
        # players -- otherwise the harness has silently baked one
        # player's identity into both prompts.
        obs0 = _make_observation(self.state, self.game, player_id=0)
        _apply_sequence(self.state, ["7g7f"])
        obs1 = _make_observation(self.state, self.game, player_id=1)
        prompt0 = generate_prompt(obs0, [])
        prompt1 = generate_prompt(obs1, [])
        self.assertNotEqual(prompt0, prompt1)
        self.assertIn("You are Player 0", prompt0)
        self.assertIn("You are Player 1", prompt1)

    def test_forward_direction_explained_symmetrically(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # A single sentence explains both players' forward directions,
        # so it must appear regardless of which side we are asking about.
        self.assertIn("Sente is toward rank a", prompt)
        self.assertIn("Gote it is", prompt)
        self.assertIn("toward rank i", prompt)

    def test_legal_moves_not_listed(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Deliberately omit the legal-move list so the model has to
        # reason about legality from the board. "7g7f" is excluded --
        # the action-notation example uses that token.
        for legal in obs["legalActionStrings"]:
            if legal == "7g7f":
                continue
            self.assertNotIn(legal, prompt)

    def test_board_ascii_includes_files_and_ranks(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # File header uses right-to-left numbering (9 first, 1 last).
        self.assertIn("9  8  7  6  5  4  3  2  1", prompt)
        # Both top and bottom rank labels should appear.
        self.assertIn(" a ", prompt)
        self.assertIn(" i ", prompt)

    def test_board_shows_initial_position(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Gote's back rank (rank a) should show the lowercase pieces.
        self.assertIn(" a  l  n  s  g  k  g  s  n  l", prompt)
        # Sente's back rank (rank i) should show uppercase pieces.
        self.assertIn(" i  L  N  S  G  K  G  S  N  L", prompt)

    def test_hands_initially_empty(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Sente: (empty)", prompt)
        self.assertIn("Gote: (empty)", prompt)

    def test_hands_populated_after_capture(self):
        # 7g7f, 3c3d, 8h2b+, 3a2b -> Sente has bishop in hand, Gote
        # has bishop in hand. Both hand renderings use uppercase piece
        # letters (USI drop notation is always uppercase).
        _apply_sequence(self.state, ["7g7f", "3c3d", "8h2b+", "3a2b"])
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Sente: B", prompt)
        self.assertIn("Gote: B", prompt)

    def test_sfen_rendered_initial_position(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("SFEN", prompt)
        self.assertIn(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            prompt,
        )

    def test_sfen_updates_after_moves(self):
        # After 7g7f, Sente's pawn on 7g slides to 7f. The SFEN board
        # field must reflect that (rank g now has a gap where the pawn
        # was), and the side-to-move must flip to 'w'.
        _apply_sequence(self.state, ["7g7f"])
        obs = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs, [])
        # Rank g had 9 pawns; after 7g7f the middle pawn is gone, so
        # SFEN encodes it as "PP1PPPPPP" (two, gap, six).
        self.assertIn("PP1PPPPPP", prompt)
        # Side to move is now Gote.
        self.assertRegex(prompt, r"SFEN[^\n]*\bw\b")

    def test_last_move_rendered_after_play(self):
        _apply_sequence(self.state, ["7g7f"])
        obs1 = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs1, [])
        self.assertIn("Last move played: 7g7f", prompt)

    def test_full_move_history_includes_both_sides(self):
        # The framework's per-agent move_history is Sente's moves only,
        # but the prompt must render the FULL game history (both sides)
        # sourced from the proxy state_dict.
        _apply_sequence(self.state, ["7g7f", "3c3d", "8h2b+"])
        obs = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs, ["7g7f", "8h2b+"])
        self.assertIn("7g7f, 3c3d, 8h2b+", prompt)

    def test_move_history_none_when_empty(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Moves played so far this game", prompt)
        self.assertIn("None", prompt)

    def test_drop_rules_present(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("nifu", prompt)
        self.assertIn("uchifuzume", prompt)
        # Drop notation example uses the ``<PIECE>*<square>`` form.
        self.assertIn("P*5e", prompt)

    def test_promotion_rules_present(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("promotion zone", prompt)
        # Promotion-zone ranks for both sides must be spelled out.
        self.assertIn("a, b, c", prompt)
        self.assertIn("g, h, i", prompt)

    def test_compulsory_promotion_language_unambiguous(self):
        # "The last rank" alone is ambiguous. The prompt must anchor
        # compulsory promotion to the opponent's back ranks, matching
        # the drop-restriction language a few lines away.
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("opponent's back rank", prompt)
        self.assertIn("opponent's last two ranks", prompt)

    def test_all_five_terminal_conditions_disclosed(self):
        # Engine (shogi.cc:334-373 MaybeFinalReturns) implements five
        # terminal paths. Every one must appear in the prompt or the
        # model plays blind to them.
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # 1. No-legal-move loss (covers both checkmate and stalemate).
        self.assertRegex(prompt, r"no legal move")
        self.assertIn("stalemate", prompt.lower())
        # 2. Perpetual check LOSS for the checker (not a draw!).
        self.assertIn("PERPETUAL CHECK", prompt)
        self.assertIn("LOSS", prompt)
        self.assertIn("6", prompt)  # 6-check threshold
        # 3. Fourfold repetition DRAW.
        self.assertIn("FOURFOLD REPETITION", prompt)
        # 4. Entering king declaration WIN.
        self.assertIn("ENTERING KING", prompt)
        self.assertIn("28", prompt)  # material threshold
        # 5. Mutual entering kings DRAW.
        self.assertIn("MUTUAL ENTERING KINGS", prompt)

    def test_perpetual_check_is_consecutive_and_repetition_includes_hands(self):
        # Engine (shogi.cc:132-135) resets the per-side check counter to
        # zero the moment a run of checks is broken, so the perpetual
        # check rule is about CONSECUTIVE checks, not lifetime.
        # And the repetition hash (shogi_board.cc:1104-1180) includes
        # both pockets, so pieces-in-hand count toward the position
        # fingerprint for sennichite. Both nuances must be spelled out
        # so the model doesn't misplan around them.
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("in a row", prompt)
        self.assertIn("consecutive", prompt)
        self.assertIn("hands", prompt)

    def test_king_capture_framing_avoided(self):
        # Old prompt said "capture the opponent's king" and referenced
        # an "illegal-move-forced sequence" -- the engine filters
        # self-check moves, so king capture never actually happens.
        # The prompt must not promise that framing.
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertNotIn("capture the opponent's king", prompt)
        self.assertNotIn("illegal-move-forced", prompt)
        # And it must positively state that self-check moves are
        # filtered out of the legal-move list.
        self.assertRegex(prompt, r"filtered|never actually captured")

    def test_json_example_unambiguous(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Placeholder and concrete example are clearly separated so a
        # model doesn't literally emit "<your_move>".
        self.assertIn('"move": "<your_move>"', prompt)
        self.assertIn('{"move": "7g7f"}', prompt)

    def test_rethink_illegal_suffix(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [], previous_response="I'll play z9z9", previous_action="z9z9")
        # ILLEGAL leads with the action string, not the previous response.
        self.assertIn("You suggested", prompt)
        self.assertIn("z9z9", prompt)
        self.assertIn("not a legal move", prompt)

    def test_rethink_unparsable_suffix(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(
            obs,
            [],
            previous_response="I forgot to add the JSON block.",
            previous_action=None,
        )
        # UNPARSABLE leads with the previous response and restates
        # the JSON format.
        self.assertIn("Your previous response ended with", prompt)
        self.assertIn("forgot to add the JSON block", prompt)
        self.assertIn("```json", prompt)

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
            "legalActions": [18, 346, 348],
            "legalActionStrings": ["9i9h", "7i7h", "7i6h"],
        }
        result = get_legal_moves(obs)
        self.assertEqual(result, {18: "9i9h", 346: "7i7h", 348: "7i6h"})

    def test_from_serialized_state(self):
        game = shogi_proxy.ShogiGame()
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
            # Every shogi action string is either a board move (4 chars,
            # optional trailing '+') or a drop (4 chars: 'X*<file><rank>').
            self.assertGreaterEqual(len(v), 4)
            self.assertLessEqual(len(v), 5)

    def test_empty_serialized(self):
        self.assertEqual(get_legal_moves({"serializedGameAndState": ""}), {})


# ---------------------------------------------------------------------------
# create_agent_fn integration
# ---------------------------------------------------------------------------


class _ShogiHarness:
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
        agent = create_agent_fn(_ShogiHarness())

        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        game = shogi_proxy.ShogiGame()
        state = game.new_initial_state()
        first_legal = state.action_to_string(0, state.legal_actions()[0])
        mock_litellm.completion.return_value = _make_mock_response(f'```json\n{{"move": "{first_legal}"}}\n```')
        agent = create_agent_fn(_ShogiHarness())

        obs = _make_observation(state, game, player_id=0)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal)
        self.assertEqual(result["status"], "OK")
        self.assertIn("thoughts", result)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        game = shogi_proxy.ShogiGame()
        state = game.new_initial_state()
        first_legal = state.action_to_string(0, state.legal_actions()[0])
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "z9z9"}\n```'),
            _make_mock_response(f'```json\n{{"move": "{first_legal}"}}\n```'),
        ]
        agent = create_agent_fn(_ShogiHarness())

        obs = _make_observation(state, game, player_id=0)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal)
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("I cannot decide.")
        agent = create_agent_fn(_ShogiHarness())

        game = shogi_proxy.ShogiGame()
        state = game.new_initial_state()
        obs = _make_observation(state, game, player_id=0)

        with self.assertRaises(ValueError):
            agent(obs, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_short_game_via_agent_fns(self, mock_litellm):
        """Drive a short shogi game with two scripted LLM agents that
        always pick their first legal move, verifying the harness
        round-trips through pyspiel cleanly."""
        mock_litellm.drop_params = True

        game = shogi_proxy.ShogiGame()
        state = game.new_initial_state()

        def fake_completion(*, model, messages, **kwargs):
            del model, kwargs
            content = messages[0]["content"]
            player_id = 0 if "You are Player 0" in content else 1
            first = state.action_to_string(player_id, state.legal_actions()[0])
            return _make_mock_response(f'```json\n{{"move": "{first}"}}\n```')

        mock_litellm.completion.side_effect = fake_completion
        agent_p0 = create_agent_fn(_ShogiHarness())
        agent_p1 = create_agent_fn(_ShogiHarness())

        for _ in range(10):
            if state.is_terminal():
                break
            cp = int(state.current_player())
            agent = agent_p0 if cp == 0 else agent_p1
            obs = _make_observation(state, game, player_id=cp)
            result = agent(obs, {})
            self.assertEqual(result["status"], "OK")
            state.apply_action(result["submission"])

        # Shogi rarely terminates in 10 plies; just confirm we round-tripped.
        self.assertGreater(state.move_number(), 0)


if __name__ == "__main__":
    absltest.main()
