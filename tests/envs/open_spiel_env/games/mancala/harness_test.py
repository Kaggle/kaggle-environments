"""Tests for the Mancala LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.mancala import (
    mancala_proxy,
)
from kaggle_environments.envs.open_spiel_env.games.mancala.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(
    state: mancala_proxy.MancalaState,
    game: mancala_proxy.MancalaGame,
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
    legal = ["1", "2", "3", "4", "5", "6"]

    def test_parse_json_block(self):
        result = parse_response('```json\n{"move": "3"}\n```', self.legal)
        self.assertEqual(result.legal_action, "3")
        self.assertEqual(result.raw_action, "3")

    def test_parse_json_block_numeric_value(self):
        # LLMs often emit a bare integer instead of a quoted string.
        result = parse_response('```json\n{"move": 3}\n```', self.legal)
        self.assertEqual(result.legal_action, "3")

    def test_parse_bare_json(self):
        result = parse_response('I think {"move": "5"} is best.', self.legal)
        self.assertEqual(result.legal_action, "5")

    def test_prose_only_response_triggers_rethink(self):
        # No structured JSON. The parser must NOT guess at intent from a
        # numeric token in the prose -- return None and let rethink ask
        # the model to use the required JSON format.
        result = parse_response("I'll sow from pit 4 this turn.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_illegal_pit_returns_raw(self):
        result = parse_response('```json\n{"move": "9"}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "9")

    def test_parse_no_match_returns_none(self):
        result = parse_response("I have no idea.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_returns_parse_result_type(self):
        result = parse_response('```json\n{"move": "1"}\n```', self.legal)
        self.assertIsInstance(result, ParseResult)

    def test_prose_only_response_with_mixed_tokens_triggers_rethink(self):
        # No structured JSON. The parser must NOT scan the prose for a
        # legal pit index -- the only signal is the model's JSON answer.
        # Return None and let the rethink loop ask for one.
        result = parse_response("Avoid pit 9, I'll choose pit 3.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_double_digit_pit(self):
        legal = ["8", "9", "10", "11", "12", "13"]
        result = parse_response('```json\n{"move": "11"}\n```', legal)
        self.assertEqual(result.legal_action, "11")

    def test_illegal_json_does_not_ghost_substitute_from_prose(self):
        # The model's JSON answer (99) isn't legal. The parser must NOT
        # silently substitute a legal pit index from the prose (the ghost
        # antipattern) -- return None so the rethink loop asks the model
        # to fix its answer.
        legal_example = self.legal[0]
        response = (
            f"I considered pit {legal_example} but went bigger.\n"
            '```json\n{"move": "99"}\n```'
        )
        result = parse_response(response, self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "99")


# ---------------------------------------------------------------------------
# generate_prompt
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game = mancala_proxy.MancalaGame()
        self.state = self.game.new_initial_state()

    def test_basic_prompt_contents(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Mancala", prompt)
        self.assertIn("Kalah", prompt)
        self.assertIn("Player 0", prompt)

    def test_legal_moves_not_listed(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # The prompt deliberately omits the legal-move list so the model has
        # to derive legality from the board. Individual pit digits still
        # appear in rule text and labels, so assert the comma-joined list
        # and the directive phrase are absent.
        self.assertNotIn(", ".join(obs["legalActionStrings"]), prompt)
        self.assertNotIn("legal pit indices:", prompt)

    def test_board_rows_rendered(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Initial board: every pit has 4 seeds, stores empty.
        self.assertIn("Player 0 pits", prompt)
        self.assertIn("Player 1 pits", prompt)
        self.assertIn("store [0] = 0", prompt)
        self.assertIn("store [7] = 0", prompt)

    def test_last_action_rendered_after_play(self):
        # P0's pit 1 has 4 seeds: sows to 2,3,4,5 (no bonus, no capture),
        # turn flips to P1. P1's prompt should attribute the move to P0.
        self.state.apply_action(1)
        obs1 = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs1, [])
        self.assertIn("Opponent (Player 0) played pit 1", prompt)

    def test_bonus_turn_rendered(self):
        # P0's pit 3 has 4 seeds: sows to 4,5,6,7 — lands in own store →
        # bonus turn. P0's next prompt should call out the bonus turn.
        self.state.apply_action(3)
        obs0 = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs0, ["3"])
        self.assertIn("BONUS TURN", prompt)
        self.assertIn("pit 3", prompt)

    def test_last_action_none_at_start(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Last action played: (none yet)", prompt)

    def test_endgame_sweep_rule_disclosed(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # The engine sweeps remaining seeds on each side into that side's
        # score at terminal -- prompt must say so, not the opposite.
        self.assertIn("PLUS any seeds remaining in their 6 pits", prompt)
        self.assertNotIn("no end-of-game sweep", prompt)

    def test_move_history_rendered(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, ["2", "11"])
        self.assertIn("2, 11", prompt)

    def test_move_history_none_when_empty(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Your move history: None", prompt)

    def test_rethink_suffix(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [], previous_response="I'll play 99", previous_action="99")
        self.assertIn("Your previous response was", prompt)
        self.assertIn("99", prompt)
        self.assertIn("not a legal move", prompt)

    def test_no_rethink_on_first_attempt(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertNotIn("Your previous response was", prompt)

    def test_prompt_renders_without_observation_string(self):
        # If only serializedGameAndState is provided, prompt should still build.
        obs = _make_observation(self.state, self.game, player_id=0)
        obs.pop("observationString")
        prompt = generate_prompt(obs, [])
        self.assertIn("Player 0 pits", prompt)


# ---------------------------------------------------------------------------
# get_legal_moves
# ---------------------------------------------------------------------------


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        obs = {
            "legalActions": [1, 2, 3],
            "legalActionStrings": ["1", "2", "3"],
        }
        result = get_legal_moves(obs)
        self.assertEqual(result, {1: "1", 2: "2", 3: "3"})

    def test_from_serialized_state(self):
        game = mancala_proxy.MancalaGame()
        state = game.new_initial_state()
        obs = {
            "playerId": 0,
            "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
        }
        result = get_legal_moves(obs)
        # All 6 initial pits for player 0 are legal.
        self.assertEqual(set(result.keys()), {1, 2, 3, 4, 5, 6})
        self.assertEqual(set(result.values()), {"1", "2", "3", "4", "5", "6"})

    def test_empty_serialized(self):
        self.assertEqual(get_legal_moves({"serializedGameAndState": ""}), {})


# ---------------------------------------------------------------------------
# create_agent_fn integration
# ---------------------------------------------------------------------------


class _MancalaHarness:
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
        agent = create_agent_fn(_MancalaHarness())

        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        game = mancala_proxy.MancalaGame()
        state = game.new_initial_state()
        first_legal = state.action_to_string(0, state.legal_actions()[0])
        mock_litellm.completion.return_value = _make_mock_response(f'```json\n{{"move": "{first_legal}"}}\n```')
        agent = create_agent_fn(_MancalaHarness())

        obs = _make_observation(state, game, player_id=0)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal)
        self.assertEqual(result["status"], "OK")
        self.assertIn("thoughts", result)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        game = mancala_proxy.MancalaGame()
        state = game.new_initial_state()
        first_legal = state.action_to_string(0, state.legal_actions()[0])
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "99"}\n```'),
            _make_mock_response(f'```json\n{{"move": "{first_legal}"}}\n```'),
        ]
        agent = create_agent_fn(_MancalaHarness())

        obs = _make_observation(state, game, player_id=0)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal)
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("I cannot decide.")
        agent = create_agent_fn(_MancalaHarness())

        game = mancala_proxy.MancalaGame()
        state = game.new_initial_state()
        obs = _make_observation(state, game, player_id=0)

        with self.assertRaises(ValueError):
            agent(obs, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_short_game_via_agent_fns(self, mock_litellm):
        """Drive a short Mancala game with two scripted LLM agents that
        always pick their first legal move, verifying the harness
        round-trips through pyspiel cleanly."""
        mock_litellm.drop_params = True

        game = mancala_proxy.MancalaGame()
        state = game.new_initial_state()

        def fake_completion(*, model, messages, **kwargs):
            del model, kwargs
            content = messages[0]["content"]
            player_id = 1 if "You are Player 1" in content else 0
            first = state.action_to_string(player_id, state.legal_actions()[0])
            return _make_mock_response(f'```json\n{{"move": "{first}"}}\n```')

        mock_litellm.completion.side_effect = fake_completion
        agent_p0 = create_agent_fn(_MancalaHarness())
        agent_p1 = create_agent_fn(_MancalaHarness())

        for _ in range(30):
            if state.is_terminal():
                break
            cp = int(state.current_player())
            agent = agent_p0 if cp == 0 else agent_p1
            obs = _make_observation(state, game, player_id=cp)
            result = agent(obs, {})
            self.assertEqual(result["status"], "OK")
            state.apply_action(result["submission"])

        # Just confirm we played without raising and the state advanced.
        self.assertGreater(state.move_number(), 0)


if __name__ == "__main__":
    absltest.main()
