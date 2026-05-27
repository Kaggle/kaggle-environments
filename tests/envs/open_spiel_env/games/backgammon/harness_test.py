"""Tests for the Backgammon LLM harness."""

import random
from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.backgammon import (
    backgammon_proxy,
)
from kaggle_environments.envs.open_spiel_env.games.backgammon.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _initial_player_state() -> tuple[backgammon_proxy.BackgammonGame, backgammon_proxy.BackgammonState]:
    """Build a game/state past the initial chance node so a player is to move."""
    game = backgammon_proxy.BackgammonGame()
    state = game.new_initial_state()
    # First chance outcome decides who goes first and the opening roll.
    state.apply_action(state.legal_actions()[0])
    return game, state


def _make_observation(
    state: backgammon_proxy.BackgammonState,
    game: backgammon_proxy.BackgammonGame,
    player_id: int | None = None,
) -> dict:
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
        "legalActionStrings": [state.action_to_string(player_id, a) for a in legal],
        "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
    }


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class ParseResponseTest(absltest.TestCase):
    # Legal-move strings always have the ``<id> - <notation>`` form.
    legal = [
        "0 - 24/23 24/22",
        "11 - 24/23 13/11",
        "648 - Bar/21 Bar/20",
        "1352 - Pass",
        "300 - 6/Off 5/Off",
    ]

    def test_parse_json_block(self):
        result = parse_response('Some thinking.\n```json\n{"move": "24/23 24/22"}\n```', self.legal)
        self.assertEqual(result.legal_action, "0 - 24/23 24/22")
        self.assertEqual(result.raw_action, "24/23 24/22")

    def test_parse_bare_json(self):
        result = parse_response('I will play {"move": "24/23 13/11"} next.', self.legal)
        self.assertEqual(result.legal_action, "11 - 24/23 13/11")

    def test_parse_case_insensitive(self):
        result = parse_response('```json\n{"move": "BAR/21 BAR/20"}\n```', self.legal)
        self.assertEqual(result.legal_action, "648 - Bar/21 Bar/20")

    def test_parse_whitespace_tolerant(self):
        result = parse_response('```json\n{"move": "24/23    24/22"}\n```', self.legal)
        self.assertEqual(result.legal_action, "0 - 24/23 24/22")

    def test_parse_pass(self):
        result = parse_response('```json\n{"move": "Pass"}\n```', self.legal)
        self.assertEqual(result.legal_action, "1352 - Pass")

    def test_parse_bear_off(self):
        result = parse_response('I will bear off: ```json\n{"move": "6/Off 5/Off"}\n```', self.legal)
        self.assertEqual(result.legal_action, "300 - 6/Off 5/Off")

    def test_parse_text_scan_fallback(self):
        result = parse_response("I'll play 24/23 24/22 because it builds the bar point.", self.legal)
        self.assertEqual(result.legal_action, "0 - 24/23 24/22")

    def test_parse_illegal_move_returns_raw(self):
        result = parse_response('```json\n{"move": "99/100"}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "99/100")

    def test_parse_no_match_returns_none(self):
        result = parse_response("I have no idea what to do.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_returns_parse_result_type(self):
        result = parse_response('```json\n{"move": "24/23 24/22"}\n```', self.legal)
        self.assertIsInstance(result, ParseResult)


# ---------------------------------------------------------------------------
# generate_prompt
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game, self.state = _initial_player_state()

    def test_basic_prompt_contents(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        self.assertIn("Backgammon", prompt)
        self.assertIn("Dice rolled this turn", prompt)
        self.assertIn("Your checkers", prompt)
        self.assertIn("Opponent's checkers", prompt)
        self.assertIn("Action notation", prompt)
        self.assertIn("```json", prompt)

    def test_includes_player_identity(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Player 0", prompt)
        self.assertIn("'x'", prompt)

        obs = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs, [])
        self.assertIn("Player 1", prompt)
        self.assertIn("'o'", prompt)

    def test_player_relative_point_numbering(self):
        # Initial board: X has 2 checkers at OpenSpiel pos 0 = X's point 24.
        # O has 2 checkers at OpenSpiel pos 23 = O's point 24.  Either way,
        # the starting checkers should show up on each player's point 24.
        obs_x = _make_observation(self.state, self.game, player_id=0)
        self.assertIn("point 24: 2", generate_prompt(obs_x, []))

        obs_o = _make_observation(self.state, self.game, player_id=1)
        self.assertIn("point 24: 2", generate_prompt(obs_o, []))

    def test_legal_moves_not_listed(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        # The prompt deliberately omits the legal-move list.
        for legal in obs["legalActionStrings"]:
            self.assertNotIn(legal, prompt)

    def test_move_history_rendered(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, ["24/23 24/22", "13/11 13/8"])
        self.assertIn("24/23 24/22, 13/11 13/8", prompt)

    def test_move_history_none_when_empty(self):
        obs = _make_observation(self.state, self.game)
        self.assertIn("Moves you have played so far: None", generate_prompt(obs, []))

    def test_dice_rendered(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        # Dice values are 1..6, comma-separated, on the "Dice rolled" line.
        import re

        match = re.search(r"Dice rolled this turn:\s*([0-9, ]+)", prompt)
        self.assertIsNotNone(match)

    def test_bar_and_off_lines(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        self.assertIn("Bar -- yours: 0, opponent's: 0", prompt)
        self.assertIn("Borne off -- yours: 0, opponent's: 0", prompt)

    def test_rethink_suffix(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(
            obs,
            [],
            previous_response="I tried bad/move",
            previous_action="bad/move",
        )
        self.assertIn("Your previous response was", prompt)
        self.assertIn("bad/move", prompt)
        self.assertIn("not a legal move", prompt)

    def test_no_rethink_on_first_attempt(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        self.assertNotIn("Your previous response was", prompt)


# ---------------------------------------------------------------------------
# get_legal_moves
# ---------------------------------------------------------------------------


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        obs = {
            "legalActions": [0, 11, 1352],
            "legalActionStrings": ["0 - 24/23 24/22", "11 - 24/23 13/11", "1352 - Pass"],
        }
        result = get_legal_moves(obs)
        self.assertEqual(
            result,
            {0: "0 - 24/23 24/22", 11: "11 - 24/23 13/11", 1352: "1352 - Pass"},
        )

    def test_from_serialized_state(self):
        game, state = _initial_player_state()
        obs = {
            "playerId": int(state.current_player()),
            "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
        }
        result = get_legal_moves(obs)
        self.assertGreater(len(result), 0)
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, str)
            self.assertIn(" - ", v)

    def test_empty_serialized(self):
        self.assertEqual(get_legal_moves({"serializedGameAndState": ""}), {})


# ---------------------------------------------------------------------------
# create_agent_fn integration
# ---------------------------------------------------------------------------


class _BackgammonHarness:
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
        agent = create_agent_fn(_BackgammonHarness())

        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        game, state = _initial_player_state()
        player_id = int(state.current_player())
        first_legal = state.action_to_string(player_id, state.legal_actions()[0])
        notation = first_legal.split(" - ", 1)[1]
        mock_litellm.completion.return_value = _make_mock_response(f'```json\n{{"move": "{notation}"}}\n```')
        agent = create_agent_fn(_BackgammonHarness())

        obs = _make_observation(state, game, player_id=player_id)
        result = agent(obs, {})

        # The harness may pick any action_id whose notation matches (backgammon
        # has equivalent action_ids for high-die-first / low-die-first), so
        # check the move notation and that the submission is legal.
        self.assertEqual(result["actionString"].split(" - ", 1)[1], notation)
        self.assertIn(result["submission"], state.legal_actions())
        self.assertEqual(result["status"], "OK")
        self.assertIn("thoughts", result)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        game, state = _initial_player_state()
        player_id = int(state.current_player())
        first_legal = state.action_to_string(player_id, state.legal_actions()[0])
        notation = first_legal.split(" - ", 1)[1]
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "99/100"}\n```'),
            _make_mock_response(f'```json\n{{"move": "{notation}"}}\n```'),
        ]
        agent = create_agent_fn(_BackgammonHarness())

        obs = _make_observation(state, game, player_id=player_id)
        result = agent(obs, {})

        self.assertEqual(result["actionString"].split(" - ", 1)[1], notation)
        self.assertIn(result["submission"], state.legal_actions())
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("I cannot decide.")
        agent = create_agent_fn(_BackgammonHarness())

        game, state = _initial_player_state()
        obs = _make_observation(state, game)

        with self.assertRaises(ValueError):
            agent(obs, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_short_game_via_agent_fns(self, mock_litellm):
        """Drive a short Backgammon game with two scripted LLM agents that
        always pick their first legal move, verifying the harness
        round-trips through pyspiel cleanly."""
        mock_litellm.drop_params = True

        game = backgammon_proxy.BackgammonGame()
        state = game.new_initial_state()
        rng = random.Random(0)

        def fake_completion(*, model, messages, **kwargs):
            del model, kwargs
            cp = int(state.current_player())
            first = state.action_to_string(cp, state.legal_actions()[0])
            notation = first.split(" - ", 1)[1]
            return _make_mock_response(f'```json\n{{"move": "{notation}"}}\n```')

        mock_litellm.completion.side_effect = fake_completion
        agent_p0 = create_agent_fn(_BackgammonHarness())
        agent_p1 = create_agent_fn(_BackgammonHarness())

        for _ in range(30):
            if state.is_terminal():
                break
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                state.apply_action(rng.choices(outcomes, probs)[0])
                continue
            cp = int(state.current_player())
            agent = agent_p0 if cp == 0 else agent_p1
            obs = _make_observation(state, game, player_id=cp)
            result = agent(obs, {})
            self.assertEqual(result["status"], "OK")
            state.apply_action(result["submission"])

        # Game won't terminate in 30 moves; just confirm we played without
        # raising and the state stayed consistent.
        self.assertGreater(state.move_number(), 0)


if __name__ == "__main__":
    absltest.main()
