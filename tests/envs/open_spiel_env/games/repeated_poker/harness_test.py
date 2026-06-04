"""Tests for the repeated_poker LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.repeated_poker.harness import (
    _extract_move_from_response,
    _PokerHarness,
    generate_prompt,
    get_legal_moves,
    parse_response,
)

_GAME_STRING = (
    "repeated_poker(max_num_hands=100,reset_stacks=True,rotate_dealer=True,"
    "universal_poker_game_string=universal_poker(betting=nolimit,"
    "bettingAbstraction=fullgame,blind=2 1,calcOddsNumSims=1000000,"
    "firstPlayer=2 1 1 1,numBoardCards=0 3 1 1,numHoleCards=2,numPlayers=2,"
    "numRanks=13,numRounds=4,numSuits=4,stack=200 200))"
)


def _new_state_at_preflop_first_action():
    """Build a state at the start of hand 0 with cards dealt and Player1 (SB) to act."""
    game = pyspiel.load_game(_GAME_STRING)
    state = game.new_initial_state()
    # Hand 0 preset: 2 hole cards each, then flop/turn/river. Just deal hole cards.
    for card in [50, 2, 24, 25]:
        state.apply_action(card)
    return game, state


def _make_observation(game, state):
    """Build a harness-style observation dict from a poker state."""
    player = state.current_player()
    legal_actions = state.legal_actions()
    return {
        "serializedGameAndState": pyspiel.serialize_game_and_state(game, state),
        "currentPlayer": player,
        "playerId": player,
        "isTerminal": state.is_terminal(),
        "legalActions": legal_actions,
        "legalActionStrings": [state.action_to_string(player, a) for a in legal_actions],
    }


# ---------------------------------------------------------------------------
# Move extraction (RuleBasedMoveParser port)
# ---------------------------------------------------------------------------


class ExtractMoveTest(absltest.TestCase):
    def test_simple_final_answer(self):
        self.assertEqual(
            _extract_move_from_response("foo\nFinal Answer: fold"),
            "fold",
        )

    def test_strips_punctuation_and_spaces(self):
        self.assertEqual(
            _extract_move_from_response("Final Answer: raise 100."),
            "raise100",
        )

    def test_strips_markdown_bold(self):
        self.assertEqual(
            _extract_move_from_response("Final Answer: **call**"),
            "call",
        )

    def test_strips_latex_boxed(self):
        self.assertEqual(
            _extract_move_from_response("Final Answer: \\boxed{check}"),
            "check",
        )

    def test_uses_last_occurrence(self):
        self.assertEqual(
            _extract_move_from_response("Maybe Final Answer: fold\nActually Final Answer: call"),
            "call",
        )

    def test_no_tag_returns_none(self):
        self.assertIsNone(_extract_move_from_response("I will fold."))

    def test_empty_suffix_returns_none(self):
        self.assertIsNone(_extract_move_from_response("Final Answer: "))


# ---------------------------------------------------------------------------
# Soft parser integration (parse_response)
# ---------------------------------------------------------------------------


class ParseResponseTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game, self.state = _new_state_at_preflop_first_action()
        self.observation = _make_observation(self.game, self.state)
        self.legal_strings = self.observation["legalActionStrings"]

    def test_fold(self):
        result = parse_response(
            "Final Answer: fold",
            self.legal_strings,
            observation=self.observation,
        )
        self.assertEqual(result.legal_action, "player=1 move=Fold")
        self.assertEqual(result.raw_action, "fold")

    def test_call(self):
        result = parse_response(
            "Final Answer: call",
            self.legal_strings,
            observation=self.observation,
        )
        self.assertEqual(result.legal_action, "player=1 move=Call")

    def test_check_maps_to_call(self):
        result = parse_response(
            "Final Answer: check",
            self.legal_strings,
            observation=self.observation,
        )
        # Upstream parser maps "check" -> "Call" for ACPC compatibility.
        self.assertEqual(result.legal_action, "player=1 move=Call")

    def test_raise_exact_size(self):
        # Player1 (SB, posted 1) is first to act preflop. To raise to 6 total
        # ACPC (street raise of "raise 6" since contrib_prev=0), parser will
        # output Bet6 if it's a legal raise size.
        # Smallest legal raise: Bet4 (min-raise = 2*BB = 4 ACPC total).
        result = parse_response(
            "Final Answer: raise 4",
            self.legal_strings,
            observation=self.observation,
        )
        self.assertEqual(result.legal_action, "player=1 move=Bet4")

    def test_raise_under_minimum_maps_up(self):
        # "raise 3" is below min raise (Bet4); should map to smallest legal.
        result = parse_response(
            "Final Answer: raise 3",
            self.legal_strings,
            observation=self.observation,
        )
        self.assertEqual(result.legal_action, "player=1 move=Bet4")

    def test_raise_over_max_maps_down(self):
        # Largest legal is Bet200 (all-in).
        result = parse_response(
            "Final Answer: raise 9999",
            self.legal_strings,
            observation=self.observation,
        )
        self.assertEqual(result.legal_action, "player=1 move=Bet200")

    def test_all_in_uses_last_legal_move(self):
        # Upstream's _extract_move_from_response strips spaces, so "all in"
        # collapses to "allin" and doesn't trigger the all-in branch. The
        # hyphenated "all-in" form survives the stripping and does match.
        result = parse_response(
            "Final Answer: all-in",
            self.legal_strings,
            observation=self.observation,
        )
        self.assertEqual(result.legal_action, self.legal_strings[-1])

    def test_unparseable_returns_none(self):
        result = parse_response("I have no idea.", self.legal_strings, observation=self.observation)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_returns_parse_result(self):
        result = parse_response(
            "Final Answer: fold",
            self.legal_strings,
            observation=self.observation,
        )
        self.assertIsInstance(result, ParseResult)


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game, self.state = _new_state_at_preflop_first_action()
        self.observation = _make_observation(self.game, self.state)

    def test_includes_template_headers(self):
        prompt = generate_prompt(self.observation, [])
        self.assertIn(
            "world-class Heads-Up No-Limit Texas Hold'em (HU NLHE) poker AI",
            prompt,
        )
        self.assertIn("I. Guiding Principles", prompt)
        self.assertIn("II. Decision-Making Framework", prompt)
        self.assertIn("III. Required Final Answer Format", prompt)
        self.assertIn("Below is the hand history you are tasked with analyzing.", prompt)
        self.assertIn(
            "Action is on you. Remember to format your response correctly. Good luck!",
            prompt,
        )

    def test_includes_final_answer_format(self):
        prompt = generate_prompt(self.observation, [])
        self.assertIn("Final Answer: <action> <size-if-bet-or-raise>", prompt)

    def test_first_prompt_has_empty_past_hands(self):
        prompt = generate_prompt(self.observation, [])
        self.assertIn("Previously played hands this session:\n\n\n\n", prompt)
        self.assertIn("Hand #0:", prompt)

    def test_player_identity_appears_twice(self):
        # Upstream renders "You are PlayerN." once at the top and again right
        # before the current-hand history.
        prompt = generate_prompt(self.observation, [])
        self.assertEqual(prompt.count("You are Player1."), 2)

    def test_opponent_hole_cards_hidden(self):
        prompt = generate_prompt(self.observation, [])
        # Player1 is the current player; Player0's cards are masked.
        self.assertIn("Dealt to Player0 [?? ??]", prompt)
        self.assertNotIn("Dealt to Player1 [?? ??]", prompt)

    def test_rethink_suffix_appended(self):
        prompt = generate_prompt(
            self.observation,
            [],
            previous_response="line1\nline2\nline3\nline4\nline5\nFinal Answer: bogus",
        )
        self.assertIn(
            "A legal action could not be parsed from your previous response.",
            prompt,
        )
        # Upstream uses the last 5 lines of the prior generation.
        self.assertIn("Final Answer: bogus", prompt)
        self.assertIn("line2", prompt)
        self.assertNotIn("line1", prompt)

    def test_rethink_empty_response_uses_sentinel(self):
        prompt = generate_prompt(self.observation, [], previous_response="")
        self.assertIn("NO RESPONSE RECEIVED", prompt)

    def test_no_rethink_on_first_attempt(self):
        prompt = generate_prompt(self.observation, [])
        self.assertNotIn("A legal action could not be parsed", prompt)

    def test_trailing_newlines_preserved(self):
        # Empty rethink_prompt slot leaves a trailing \n\n -- prompt parity
        # depends on preserving this exactly.
        prompt = generate_prompt(self.observation, [])
        self.assertTrue(prompt.endswith("Good luck!\n\n"))


# ---------------------------------------------------------------------------
# Legal moves
# ---------------------------------------------------------------------------


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        observation = {
            "legalActions": [0, 1, 5],
            "legalActionStrings": [
                "player=0 move=Fold",
                "player=0 move=Call",
                "player=0 move=Bet5",
            ],
        }
        result = get_legal_moves(observation)
        self.assertEqual(
            result,
            {
                0: "player=0 move=Fold",
                1: "player=0 move=Call",
                5: "player=0 move=Bet5",
            },
        )

    def test_from_serialized_state(self):
        game, state = _new_state_at_preflop_first_action()
        observation = {
            "serializedGameAndState": pyspiel.serialize_game_and_state(game, state),
        }
        result = get_legal_moves(observation)
        self.assertGreater(len(result), 0)
        self.assertIn("player=1 move=Fold", result.values())
        self.assertIn("player=1 move=Call", result.values())

    def test_empty_serialized(self):
        result = get_legal_moves({"serializedGameAndState": ""})
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# Streaming mock + agent integration
# ---------------------------------------------------------------------------


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


_ENV = {
    "MODEL_NAME": "test-model",
    "MODEL_PROXY_KEY": "test-key",
    "MODEL_PROXY_URL": "dummy_url",
}


class AgentIntegrationTest(absltest.TestCase):
    """Test the harness through ``create_agent_fn`` with a mocked LLM."""

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_setup_step_returns_inactive(self, mock_litellm):
        mock_litellm.drop_params = True
        agent = create_agent_fn(_PokerHarness())

        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_fold(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            "I should fold.\n\nFinal Answer: fold",
        )

        agent = create_agent_fn(_PokerHarness())
        game, state = _new_state_at_preflop_first_action()
        observation = _make_observation(game, state)

        result = agent(observation, {})

        self.assertEqual(result["actionString"], "player=1 move=Fold")
        self.assertEqual(result["status"], "OK")

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.side_effect = [
            _make_mock_response("gibberish without final answer"),
            _make_mock_response("Final Answer: call"),
        ]

        agent = create_agent_fn(_PokerHarness())
        game, state = _new_state_at_preflop_first_action()
        observation = _make_observation(game, state)

        result = agent(observation, {})

        self.assertEqual(result["actionString"], "player=1 move=Call")
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_call_details_present(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("Final Answer: call")

        agent = create_agent_fn(_PokerHarness())
        game, state = _new_state_at_preflop_first_action()
        observation = _make_observation(game, state)

        result = agent(observation, {})

        self.assertIn("call_details", result)
        self.assertLen(result["call_details"], 1)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_forfeit_on_exhausted_retries(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            "I have no idea what to do.",
        )

        agent = create_agent_fn(_PokerHarness())
        game, state = _new_state_at_preflop_first_action()
        observation = _make_observation(game, state)

        result = agent(observation, {"illegalMoveForfeit": True})

        self.assertEqual(result["submission"], -1)
        self.assertIn("forfeiting", result["status"])


if __name__ == "__main__":
    absltest.main()
