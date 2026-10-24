"""Tests for the Shogi LLM harness."""

import json
import random
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


class NotationToleranceTest(absltest.TestCase):
    """Shogi promotion is OPTIONAL, so a from-to pair that touches the
    promotion zone is usually in the legal set TWICE (``7c7b`` and
    ``7c7b+``) as two different moves. The matcher must therefore never
    let tolerance override an exactly-legal form -- but where the
    requested form is not legal at all, the alternative is a forfeit.

    Across the 1391-episode shogi-v2 archive (103,714 turns) this matcher
    changed 0 of 103,374 already-correct parses and recovered 229 of 3,584
    failures, 23 of them on turns that actually ended in a forfeit.
    """

    # Both forms of 7c7b are legal, as they are in most real positions.
    both = ["7c7b", "7c7b+", "7g7f", "P*5e", "8h2b+"]

    def test_exact_form_always_wins_when_both_legal(self):
        # The load-bearing case: when both forms are legal each must map
        # to ITSELF. If tolerance ran first it could silently promote a
        # deliberate non-promotion (or vice versa) -- a real strategic
        # difference, not a notation detail.
        for move in ("7c7b", "7c7b+"):
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.both)
            self.assertEqual(result.legal_action, move)

    def test_stray_promotion_suffix_recovered(self):
        # 160 archive turns appended '+' to a move touching no promotion
        # zone. Only "7g7f" is legal, so "7g7f+" is unambiguous.
        result = parse_response('```json\n{"move": "7g7f+"}\n```', self.both)
        self.assertEqual(result.legal_action, "7g7f")

    def test_promoted_piece_prefix_echoed_as_suffix_recovered(self):
        # 15 archive turns moved an ALREADY-promoted piece (rendered
        # "+R" on the board) and copied that '+' onto the move. The
        # board's '+' is a prefix meaning "already promoted"; the move's
        # '+' is a suffix meaning "promote now".
        legal = ["8b8a", "8b7b"]
        result = parse_response('```json\n{"move": "8b8a+"}\n```', legal)
        self.assertEqual(result.legal_action, "8b8a")

    def test_omitted_compulsory_promotion_recovered(self):
        # Compulsory promotion: only the '+' form is legal (a pawn on the
        # opponent's back rank would otherwise have no move).
        legal = ["1c1b+", "7g7f"]
        result = parse_response('```json\n{"move": "1c1b"}\n```', legal)
        self.assertEqual(result.legal_action, "1c1b+")

    def test_san_style_piece_prefix_recovered(self):
        # 47 archive turns wrote chess-style "S3i2h" instead of "3i2h".
        legal = ["3i2h", "7g7f"]
        result = parse_response('```json\n{"move": "S3i2h"}\n```', legal)
        self.assertEqual(result.legal_action, "3i2h")

    def test_decorative_punctuation_stripped(self):
        result = parse_response('```json\n{"move": "7g-7f"}\n```', self.both)
        self.assertEqual(result.legal_action, "7g7f")

    def test_drop_is_not_mangled_by_piece_prefix_stripping(self):
        # "P*5e" starts with a piece letter but must NOT have it stripped.
        result = parse_response('```json\n{"move": "P*5e"}\n```', self.both)
        self.assertEqual(result.legal_action, "P*5e")

    def test_genuinely_illegal_move_still_rejected(self):
        # Tolerance must not manufacture a move out of nothing: the
        # rethink loop needs to see this failure.
        result = parse_response('```json\n{"move": "9a9b"}\n```', self.both)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "9a9b")

    def test_tolerance_never_overrides_an_exactly_legal_move(self):
        # Property check over every legal move of a real position: each
        # must round-trip to itself.
        game = shogi_proxy.ShogiGame()
        state = game.new_initial_state()
        # This line opens the long diagonal, so Gote's bishop on 2b can
        # reach 8h/7g either promoting or not -- both forms legal, which
        # is the case that makes matcher ordering observable.
        _apply_sequence(state, ["7g7f", "3c3d", "7f7e", "3d3e", "7e7d"])
        pid = int(state.current_player())
        legal = [state.action_to_string(pid, a) for a in state.legal_actions()]
        promo_pairs = 0
        for move in legal:
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', legal)
            self.assertEqual(result.legal_action, move)
            if not move.endswith("+") and "*" not in move and move + "+" in legal:
                promo_pairs += 1
        # Guard the guard: this position must actually contain both-form
        # pairs, or the assertion above proves nothing about ordering.
        self.assertGreater(promo_pairs, 0)


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

    def test_prompt_renders_from_serialized_state_only(self):
        # Defense-in-depth: if the harness is invoked with only
        # `serializedGameAndState` (no `observationString`), the
        # fallback path must still deserialize into a proxy ShogiState
        # whose observation_string emits JSON, not bare SFEN. In
        # production, `open_spiel_env.py:707` serializes the proxy
        # game/state directly, producing a header that references
        # `shogi_proxy(...)` -- and the deserializer needs that type
        # registered (via the module-level shogi_proxy import at the
        # top of harness.py). Without that import the deserialize
        # would raise; with the raw-pyspiel type registered but the
        # proxy missing, we'd silently get bare SFEN and the prompt
        # would render "(unavailable)" for board, SFEN, and hands.
        obs = {
            "playerId": 0,
            "currentPlayer": int(self.state.current_player()),
            "isTerminal": False,
            "serializedGameAndState": pyspiel.serialize_game_and_state(self.game, self.state),
        }
        prompt = generate_prompt(obs, [])
        self.assertNotIn("(unavailable)", prompt)
        self.assertIn(" i  L  N  S  G  K  G  S  N  L", prompt)
        self.assertIn(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            prompt,
        )

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


class OwnRosterTest(absltest.TestCase):
    """The prompt's per-player piece roster is the model's authoritative
    source-square list; wrong-piece and empty-square moves accounted for
    ~740 of ~1270 illegal-move rethinks in the July 2026 replay archive.
    """

    def setUp(self):
        super().setUp()
        self.game = shogi_proxy.ShogiGame()
        self.state = self.game.new_initial_state()

    def test_initial_roster_covers_sente_pieces(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Kings, back-rank golds, rook, bishop, and pawn rank should all
        # appear as source-square lists for Sente.
        self.assertIn("K:5i", prompt)
        self.assertIn("R:2h", prompt)
        self.assertIn("B:8h", prompt)
        # Sente's pawn rank g contains nine files.
        self.assertIn("P:9g,8g,7g,6g,5g,4g,3g,2g,1g", prompt)
        # Gote pieces (lowercase back rank) must NOT appear in Sente's roster.
        self.assertNotIn("K:5a", prompt)

    def test_roster_swaps_for_gote(self):
        obs = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs, [])
        self.assertIn("K:5a", prompt)
        self.assertIn("P:9c,8c,7c,6c,5c,4c,3c,2c,1c", prompt)
        self.assertNotIn("K:5i", prompt)


class DiagnoseIllegalMoveTest(absltest.TestCase):
    """The rethink prompt should explain WHY a move was rejected. Blind
    "not a legal move" retries make up the largest slice of rethink loops
    in the archive; targeting the specific failure mode gives models a
    concrete correction signal.
    """

    def setUp(self):
        super().setUp()
        self.game = shogi_proxy.ShogiGame()
        self.state = self.game.new_initial_state()

    def _prompt_after(self, moves, player_id, previous_action):
        _apply_sequence(self.state, moves)
        obs = _make_observation(self.state, self.game, player_id=player_id)
        return generate_prompt(
            obs,
            [],
            previous_response=f"I'll play {previous_action}",
            previous_action=previous_action,
        )

    def test_reason_empty_source(self):
        # 5e is empty in the initial position; moving from there is nonsense.
        prompt = self._prompt_after([], 0, "5e5d")
        self.assertIn("5e is empty", prompt)

    def test_reason_opponent_source(self):
        # Sente tries to move a Gote pawn (rank c).
        prompt = self._prompt_after([], 0, "7c7d")
        self.assertIn("Gote's", prompt)
        # And still identifies which side the calling model is.
        self.assertIn("you are Sente", prompt)

    def test_reason_own_capture(self):
        # 5i is Sente's king; 4i is Sente's gold. King cannot capture own gold.
        prompt = self._prompt_after([], 0, "5i4i")
        self.assertIn("your own", prompt)

    def test_reason_bad_promotion_geometry(self):
        # 7g7f is a pawn push that never touches the promotion zone
        # (ranks a-c for Sente), so appending + must be diagnosed.
        prompt = self._prompt_after([], 0, "7g7f+")
        self.assertIn("promotion", prompt)
        self.assertIn("a, b, c", prompt)

    def test_reason_promote_already_promoted(self):
        # After 7g7f 3c3d 8h2b+ 3a2b, Sente has captured Gote's bishop
        # and Gote's silver moved to 2b. Now Sente tries to move rook
        # onto 8h and re-promote a promoted piece (contrived) — instead
        # test via the diagnose helper directly for a promoted source.
        from kaggle_environments.envs.open_spiel_env.games.shogi.harness import (
            _diagnose_illegal_move,
        )
        board = [
            ["."] * 9,
            ["."] * 9,
            ["."] * 9,
            ["."] * 9,
            ["."] * 9,
            ["."] * 9,
            ["."] * 9,
            ["+P", ".", ".", ".", ".", ".", ".", ".", "."],
            ["."] * 9,
        ]
        # +P at row 7 col 0 = square 9h. Try "9h9g+" (already promoted).
        msg = _diagnose_illegal_move("9h9g+", board, {"b": {}, "w": {}}, 0)
        self.assertIn("already promoted", msg)

    def test_reason_drop_without_piece_in_hand(self):
        # Initial position: neither hand holds a pawn, but Sente tries P*5e.
        prompt = self._prompt_after([], 0, "P*5e")
        self.assertIn("no P in hand", prompt)

    def test_reason_drop_onto_occupied_square(self):
        # Give Sente a pawn in hand (7g7f 3c3d 8h2b+ 3a2b: Sente has B).
        # Then try B*5i (occupied by Sente's own king).
        from kaggle_environments.envs.open_spiel_env.games.shogi.harness import (
            _diagnose_illegal_move,
        )
        board = [["."] * 9 for _ in range(9)]
        board[8][4] = "K"  # 5i
        captured = {"b": {"B": 1}, "w": {}}
        msg = _diagnose_illegal_move("B*5i", board, captured, 0)
        self.assertIn("occupied", msg)

    def test_reason_nifu(self):
        # Sente already has an unpromoted pawn on file 5 (rank g). Given
        # a P in hand, dropping P*5e is nifu.
        from kaggle_environments.envs.open_spiel_env.games.shogi.harness import (
            _diagnose_illegal_move,
        )
        board = [["."] * 9 for _ in range(9)]
        board[6][4] = "P"  # 5g
        captured = {"b": {"P": 1}, "w": {}}
        msg = _diagnose_illegal_move("P*5e", board, captured, 0)
        self.assertIn("nifu", msg)

    def test_reason_drop_pawn_on_back_rank(self):
        # Sente drops P on rank a (opponent's back rank). Even without a
        # nifu conflict, this is illegal because the pawn has no forward
        # square.
        from kaggle_environments.envs.open_spiel_env.games.shogi.harness import (
            _diagnose_illegal_move,
        )
        board = [["."] * 9 for _ in range(9)]
        captured = {"b": {"P": 1}, "w": {}}
        msg = _diagnose_illegal_move("P*5a", board, captured, 0)
        self.assertIn("back rank", msg)

    def test_diagnosis_survives_braces_in_previous_action(self):
        # The diagnosis for "not a valid square" echoes the raw model
        # input (parts[1]) before it has been validated -- a hallucinated
        # move like "P*5{" or "P*{a}" would splice a stray '{' into the
        # illegal-rethink template, and render_rethink_suffix runs
        # .format() on the template afterwards, so any unescaped brace
        # crashes generate_prompt mid-turn. This must NOT raise.
        obs = _make_observation(self.state, self.game, player_id=0)
        for bad in ("P*5{", "P*{a}", "7g7{"):
            prompt = generate_prompt(
                obs,
                [],
                previous_response=f"I'll play {bad}",
                previous_action=bad,
            )
            # The literal (unformatted) model input must appear once via
            # the {previous_action} substitution; the diagnosis echo of
            # the same braces must not have been re-interpreted.
            self.assertIn(bad, prompt)

    def test_no_diagnosis_on_first_attempt(self):
        # Fresh prompt (no previous_action) should not include a diagnosis
        # sentence.
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertNotIn("Reason:", prompt)


def _reach_check(max_plies: int = 120) -> tuple[shogi_proxy.ShogiGame, shogi_proxy.ShogiState]:
    """Play a deterministic line until the side to move is in check.

    Uses a fixed seed and prefers checking moves so the search terminates
    quickly; asserts rather than returning a non-check position, so a test
    can never silently pass on a position it did not intend.
    """
    rng = random.Random(23)
    game = shogi_proxy.ShogiGame()
    for _ in range(500):
        state = game.new_initial_state()
        for _ in range(max_plies):
            if state.is_terminal():
                break
            legal = state.legal_actions()
            if not legal:
                break
            checking = []
            for action in legal:
                nxt = state.clone()
                nxt.apply_action(action)
                if not nxt.is_terminal() and nxt.__wrapped__.in_check():
                    checking.append(action)
            state.apply_action(rng.choice(checking or legal))
            if not state.is_terminal() and state.__wrapped__.in_check():
                return game, state
    raise AssertionError("could not reach an in-check position")


class InCheckDisclosureTest(absltest.TestCase):
    """The prompt must tell the model when its king is under attack.

    In the shogi-v2 archive, 822 of 3,402 illegal moves were geometrically
    valid but rejected purely on king safety (603 while in check, 219 with a
    pinned piece). In-check turns were 16.8% of all turns but produced 48.5%
    of all forfeits, failing ~3x as often as safe turns.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.game, cls.state = _reach_check()

    def _obs(self, player_id):
        return _make_observation(self.state, self.game, player_id=player_id)

    def test_state_dict_exposes_in_check(self):
        parsed = json.loads(self.state.observation_string(0))
        self.assertTrue(parsed["in_check"])
        # And it must track the engine, not a re-derivation.
        self.assertEqual(parsed["in_check"], self.state.__wrapped__.in_check())

    def test_initial_position_is_not_in_check(self):
        state = shogi_proxy.ShogiGame().new_initial_state()
        self.assertFalse(json.loads(state.observation_string(0))["in_check"])

    def test_prompt_warns_when_in_check(self):
        prompt = generate_prompt(self._obs(int(self.state.current_player())), [])
        self.assertIn("YOU ARE IN CHECK", prompt)
        self.assertIn("MUST leave your king", prompt)

    def test_warning_names_the_king_square(self):
        pid = int(self.state.current_player())
        prompt = generate_prompt(self._obs(pid), [])
        board = json.loads(self.state.observation_string(0))["board"]
        want = "K" if pid == 0 else "k"
        square = next(
            f"{'987654321'[c]}{'abcdefghi'[r]}"
            for r, row in enumerate(board)
            for c, cell in enumerate(row)
            if cell == want
        )
        self.assertIn(f"Your king is on {square}", prompt)

    def test_no_warning_when_king_is_safe(self):
        state = shogi_proxy.ShogiGame().new_initial_state()
        obs = _make_observation(state, shogi_proxy.ShogiGame(), player_id=0)
        self.assertNotIn("IN CHECK", generate_prompt(obs, []))

    def test_no_warning_for_the_player_not_to_move(self):
        # `in_check` describes the side to move. Rendering the warning for
        # the opponent would tell the wrong player their king is attacked.
        other = 1 - int(self.state.current_player())
        self.assertNotIn("YOU ARE IN CHECK", generate_prompt(self._obs(other), []))

    def test_king_safety_rule_always_stated(self):
        # The pin case needs a standing rule, not a per-turn warning:
        # 219 archive failures moved a pinned piece while NOT in check.
        state = shogi_proxy.ShogiGame().new_initial_state()
        obs = _make_observation(state, shogi_proxy.ShogiGame(), player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("pinned piece", prompt)
        self.assertIn("leaves your own king attacked", prompt)


class PromotionOptionalityTest(absltest.TestCase):
    """182 archive turns were rejected over a '+'-only difference, the
    single largest notation failure. The prompt must say promotion is
    optional and disambiguate the two meanings of '+'.
    """

    def setUp(self):
        super().setUp()
        self.game = shogi_proxy.ShogiGame()
        self.state = self.game.new_initial_state()
        self.prompt = generate_prompt(
            _make_observation(self.state, self.game, player_id=0), []
        )

    def test_promotion_stated_as_optional(self):
        self.assertIn("Promotion is OPTIONAL", self.prompt)

    def test_both_forms_shown_as_distinct_moves(self):
        self.assertIn("``7c7b``", self.prompt)
        self.assertIn("``7c7b+``", self.prompt)

    def test_prefix_versus_suffix_disambiguated(self):
        self.assertIn("prefix", self.prompt)
        self.assertIn("suffix", self.prompt)
        self.assertIn("ALREADY promoted", self.prompt)

    def test_drops_never_promote(self):
        self.assertIn("Drops never promote", self.prompt)


class DiagnoseKingSafetyAndGeometryTest(absltest.TestCase):
    """_diagnose_illegal_move returned "" on 1,465 of 3,723 failed
    attempts in the archive -- exactly the king-safety and bad-geometry
    cases. With only one rethink before forfeit, a contentless retry is
    close to a wasted one.
    """

    def test_unreachable_destination_lists_real_destinations(self):
        game = shogi_proxy.ShogiGame()
        state = game.new_initial_state()
        obs = _make_observation(state, game, player_id=0)
        # The rook on 2h cannot reach 5e, but it does have legal moves.
        prompt = generate_prompt(
            obs, [], previous_response="x", previous_action="2h5e"
        )
        self.assertIn("cannot legally move to 5e", prompt)
        self.assertIn("legal destinations", prompt)

    def test_in_check_move_that_ignores_the_check_is_explained(self):
        game, state = _reach_check()
        pid = int(state.current_player())
        obs = _make_observation(state, game, player_id=pid)
        legal = set(obs["legalActionStrings"])
        board = json.loads(state.observation_string(0))["board"]
        # Find an own-piece move that is NOT legal and whose source piece
        # also has no legal move -- i.e. rejected for king safety only.
        own_upper = pid == 0
        candidate = None
        for r, row in enumerate(board):
            for c, cell in enumerate(row):
                if cell == "." or cell[-1].isupper() != own_upper:
                    continue
                src = f"{'987654321'[c]}{'abcdefghi'[r]}"
                if any(m[:2] == src for m in legal if "*" not in m):
                    continue
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < 9 and 0 <= cc < 9):
                        continue
                    dest = board[rr][cc]
                    if dest != "." and dest[-1].isupper() == own_upper:
                        continue
                    move = f"{src}{'987654321'[cc]}{'abcdefghi'[rr]}"
                    if move not in legal:
                        candidate = move
                        break
                if candidate:
                    break
            if candidate:
                break
        self.assertIsNotNone(candidate, "no king-safety-only candidate found")
        prompt = generate_prompt(
            obs, [], previous_response="x", previous_action=candidate
        )
        self.assertIn("Reason:", prompt)
        self.assertIn("check", prompt.lower())

    def test_diagnosis_degrades_gracefully_without_legal_moves(self):
        # Called with no legal-move list (the older 4-arg form), the
        # geometry/king-safety branch must stay silent rather than guess.
        from kaggle_environments.envs.open_spiel_env.games.shogi.harness import (
            _diagnose_illegal_move,
        )
        board = [["."] * 9 for _ in range(9)]
        board[8][4] = "K"
        self.assertEqual(
            _diagnose_illegal_move("5i5a", board, {"b": {}, "w": {}}, 0), ""
        )


class GoteRookPromotionRegressionTest(absltest.TestCase):
    """Regression test for open_spiel commit 415f1d92: PieceTypeToString
    used to return "+R" for both Sente and Gote promoted rooks, so a
    White promoted rook would serialize as uppercase "+R" and confuse
    the proxy's board parser (which uses letter case to assign
    ownership). The pinned open_spiel 2.0.1 includes the fix; this test
    guards against a future version regression.
    """

    def test_gote_promoted_rook_is_lowercase(self):
        game = shogi_proxy.ShogiGame()
        state = game.new_initial_state()
        # A short forced line that ends with Gote's rook reaching 2g and
        # promoting there (into Sente's promotion zone).
        moves = [
            "7g7f", "8c8d", "7f7e", "8d8e", "7e7d", "7c7d",
            "2g2f", "8e8f", "8g8f", "8b8f", "2f2e", "8f7f",
            "2e2d", "7f2f", "5i6h", "2f2g+",
        ]
        _apply_sequence(state, moves)

        parsed = json.loads(state.observation_string(0))
        board = parsed["board"]
        sfen_board_field = parsed["sfen"].split(" ")[0]

        # SFEN board field must use lowercase +r for Gote's promoted rook.
        self.assertIn("+r", sfen_board_field)
        self.assertNotIn("+R", sfen_board_field)

        # And the parsed board grid must place a Gote (lowercase) promoted
        # rook at 2g -- row 6, col 7 (files are right-to-left).
        self.assertEqual(board[6][7], "+r")


if __name__ == "__main__":
    absltest.main()
