"""Tests for the Dots and Boxes LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.dots_and_boxes import (
    dots_and_boxes_proxy,
)
from kaggle_environments.envs.open_spiel_env.games.dots_and_boxes.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(
    state: dots_and_boxes_proxy.DotsAndBoxesState,
    game: dots_and_boxes_proxy.DotsAndBoxesGame,
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
    legal = ["P1(h,0,0)", "P1(h,0,1)", "P1(v,0,0)", "P1(v,1,2)"]

    def test_parse_json_block_shorthand(self):
        result = parse_response('Thinking...\n```json\n{"move": "h 0 0"}\n```', self.legal)
        self.assertEqual(result.legal_action, "P1(h,0,0)")
        self.assertEqual(result.raw_action, "h 0 0")

    def test_parse_rejects_openspiel_form(self):
        result = parse_response('```json\n{"move": "P1(v,1,2)"}\n```', self.legal)
        self.assertIsNone(result.legal_action)

    def test_parse_bare_json(self):
        result = parse_response('I think {"move": "v 0 0"} works.', self.legal)
        self.assertEqual(result.legal_action, "P1(v,0,0)")

    def test_prose_only_response_triggers_rethink(self):
        # No structured JSON. The parser must NOT guess at intent from
        # an h/v token in the prose -- return None and let the rethink
        # loop ask the model to use the required JSON format.
        result = parse_response("I will draw h 0 1 this turn.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_case_insensitive(self):
        result = parse_response('```json\n{"move": "H 0 0"}\n```', self.legal)
        self.assertEqual(result.legal_action, "P1(h,0,0)")

    def test_parse_comma_separators(self):
        result = parse_response('```json\n{"move": "h, 0, 1"}\n```', self.legal)
        self.assertEqual(result.legal_action, "P1(h,0,1)")

    def test_parse_illegal_move_returns_raw(self):
        result = parse_response('```json\n{"move": "h 9 9"}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "h 9 9")

    def test_parse_no_match_returns_none(self):
        result = parse_response("I have no idea.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_returns_parse_result_type(self):
        result = parse_response('```json\n{"move": "h 0 0"}\n```', self.legal)
        self.assertIsInstance(result, ParseResult)

    def test_illegal_json_does_not_ghost_substitute_from_prose(self):
        # The model's JSON answer "v 9 9" is not legal. The parser must
        # NOT silently substitute "h 0 0" from the prose (the ghost
        # antipattern). Surface raw_action so the rethink loop fires.
        response = (
            'I considered "h 0 0" but think bigger is better.\n'
            '```json\n{"move": "v 9 9"}\n```'
        )
        result = parse_response(response, self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "v 9 9")


# ---------------------------------------------------------------------------
# generate_prompt
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game = dots_and_boxes_proxy.DotsAndBoxesGame()
        self.state = self.game.new_initial_state()

    def test_basic_prompt_contents(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Dots and Boxes", prompt)
        self.assertIn("Player 1", prompt)
        self.assertIn("Action notation", prompt)

    def test_player_label_swap(self):
        self.state.apply_action(self.state.legal_actions()[0])
        obs1 = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs1, [])
        self.assertIn("You are Player 2", prompt)

    def test_board_ascii_includes_dots(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Dot character should appear in the ASCII board.
        self.assertIn("+", prompt)

    def test_scores_rendered(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Player 1 = 0", prompt)
        self.assertIn("Player 2 = 0", prompt)

    def test_last_move_rendered_after_opponent_play(self):
        self.state.apply_action(0)  # P1(h,0,0)
        obs1 = _make_observation(self.state, self.game, player_id=1)
        prompt = generate_prompt(obs1, [])
        self.assertIn("Opponent's last move: h 0 0", prompt)

    def test_last_move_labeled_as_own_after_box_completion(self):
        # Drive the game so the same player completes the fourth edge of
        # box (0,0), triggering a bonus turn — last_action.player should
        # equal the current player.
        def play(orient: str, row: int, col: int) -> None:
            for action in self.state.legal_actions():
                m = self.state.action_to_string(self.state.current_player(), action)
                if f"({orient},{row},{col})" in m:
                    self.state.apply_action(action)
                    return
            raise AssertionError(f"No legal action for {orient} {row} {col}")

        play("h", 0, 0)  # P1
        play("h", 0, 1)  # P2
        play("v", 0, 0)  # P1
        play("v", 0, 1)  # P2
        play("h", 1, 0)  # P1 closes box (0,0); bonus turn
        obs = _make_observation(self.state, self.game, player_id=int(self.state.current_player()))
        prompt = generate_prompt(obs, [])
        self.assertIn("Your previous move", prompt)

    def test_last_move_none_at_start(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Previous move: (none yet)", prompt)

    def test_move_history_rendered(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, ["h 0 0", "v 0 0"])
        self.assertIn("h 0 0, v 0 0", prompt)

    def test_move_history_none_when_empty(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Moves you have played so far: None", prompt)

    def test_rethink_suffix(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [], previous_response="I'll play x 9 9", previous_action="x 9 9")
        self.assertIn("Your previous response was", prompt)
        self.assertIn("x 9 9", prompt)
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
            "legalActions": [0, 6],
            "legalActionStrings": ["P1(h,0,0)", "P1(v,0,0)"],
        }
        result = get_legal_moves(obs)
        self.assertEqual(result, {0: "P1(h,0,0)", 6: "P1(v,0,0)"})

    def test_from_serialized_state(self):
        game = dots_and_boxes_proxy.DotsAndBoxesGame()
        state = game.new_initial_state()
        obs = {
            "playerId": 0,
            "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
        }
        result = get_legal_moves(obs)
        # 2x2 board has 12 distinct edges.
        self.assertEqual(len(result), 12)
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertRegex(v, r"^P\d\([hv],\d,\d\)$")

    def test_empty_serialized(self):
        self.assertEqual(get_legal_moves({"serializedGameAndState": ""}), {})


# ---------------------------------------------------------------------------
# create_agent_fn integration
# ---------------------------------------------------------------------------


class _DotsAndBoxesHarness:
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


def _shorthand(action_string: str) -> str:
    """Convert OpenSpiel ``P1(h,0,1)`` to the documented ``h 0 1`` shorthand."""
    inner = action_string.split("(", 1)[1].rstrip(")")
    orient, r, c = inner.split(",")
    return f"{orient} {int(r)} {int(c)}"


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
        agent = create_agent_fn(_DotsAndBoxesHarness())

        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        game = dots_and_boxes_proxy.DotsAndBoxesGame()
        state = game.new_initial_state()
        first_legal = state.action_to_string(0, state.legal_actions()[0])
        mock_litellm.completion.return_value = _make_mock_response(
            f'```json\n{{"move": "{_shorthand(first_legal)}"}}\n```'
        )
        agent = create_agent_fn(_DotsAndBoxesHarness())

        obs = _make_observation(state, game, player_id=0)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal)
        self.assertEqual(result["status"], "OK")
        self.assertIn("thoughts", result)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        game = dots_and_boxes_proxy.DotsAndBoxesGame()
        state = game.new_initial_state()
        first_legal = state.action_to_string(0, state.legal_actions()[0])
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "h 9 9"}\n```'),
            _make_mock_response(f'```json\n{{"move": "{_shorthand(first_legal)}"}}\n```'),
        ]
        agent = create_agent_fn(_DotsAndBoxesHarness())

        obs = _make_observation(state, game, player_id=0)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal)
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("I cannot decide.")
        agent = create_agent_fn(_DotsAndBoxesHarness())

        game = dots_and_boxes_proxy.DotsAndBoxesGame()
        state = game.new_initial_state()
        obs = _make_observation(state, game, player_id=0)

        with self.assertRaises(ValueError):
            agent(obs, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_short_game_via_agent_fns(self, mock_litellm):
        """Drive a short Dots and Boxes game with scripted LLM agents that
        always pick their first legal move, verifying the harness round-trips
        through pyspiel cleanly."""
        mock_litellm.drop_params = True

        game = dots_and_boxes_proxy.DotsAndBoxesGame()
        state = game.new_initial_state()

        def fake_completion(*, model, messages, **kwargs):
            del model, kwargs
            content = messages[0]["content"]
            player_id = 0 if "You are Player 1" in content else 1
            first = state.action_to_string(player_id, state.legal_actions()[0])
            return _make_mock_response(f'```json\n{{"move": "{_shorthand(first)}"}}\n```')

        mock_litellm.completion.side_effect = fake_completion
        agent_p0 = create_agent_fn(_DotsAndBoxesHarness())
        agent_p1 = create_agent_fn(_DotsAndBoxesHarness())

        # Default 2x2 board has 12 edges; allow plenty of moves.
        for _ in range(20):
            if state.is_terminal():
                break
            cp = int(state.current_player())
            agent = agent_p0 if cp == 0 else agent_p1
            obs = _make_observation(state, game, player_id=cp)
            result = agent(obs, {})
            self.assertEqual(result["status"], "OK")
            state.apply_action(result["submission"])

        # Should have made some progress.
        self.assertGreater(state.move_number(), 0)


if __name__ == "__main__":
    absltest.main()
