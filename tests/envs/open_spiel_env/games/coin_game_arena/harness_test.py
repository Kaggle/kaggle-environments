"""Tests for Coin Game Arena LLM harness."""

import json
from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.coin_game_arena import (
    coin_game_arena_game,
)
from kaggle_environments.envs.open_spiel_env.games.coin_game_arena.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(state, player_id):
    legal = state.legal_actions(player_id)
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "currentPlayer": state.current_player(),
        "isTerminal": state.is_terminal(),
        "legalActions": legal,
        "legalActionStrings": [state.action_to_string(player_id, a) for a in legal],
    }


def _state(seed=7, episode_length=20):
    g = pyspiel.load_game(
        "coin_game_arena", {"seed": seed, "episode_length": episode_length}
    )
    return g.new_initial_state()


class ParseResponseTest(absltest.TestCase):

    def test_parse_json_move(self):
        legal = ["up", "down", "left", "right", "stand"]
        result = parse_response('```json\n{"move": "up"}\n```', legal)
        self.assertEqual(result.legal_action, "up")

    def test_parse_each_action(self):
        legal = ["up", "down", "left", "right", "stand"]
        for action in legal:
            result = parse_response(f'```json\n{{"move": "{action}"}}\n```', legal)
            self.assertEqual(result.legal_action, action)

    def test_parse_case_insensitive(self):
        legal = ["up", "down", "left", "right", "stand"]
        result = parse_response('```json\n{"move": "DOWN"}\n```', legal)
        self.assertEqual(result.legal_action, "down")

    def test_parse_fallback_keyword(self):
        legal = ["up", "down", "left", "right", "stand"]
        result = parse_response("I'll go right toward the b coin.", legal)
        self.assertEqual(result.legal_action, "right")

    def test_parse_no_match(self):
        legal = ["up", "down", "left", "right", "stand"]
        result = parse_response('```json\n{"move": "diagonal"}\n```', legal)
        self.assertIsNone(result.legal_action)

    def test_returns_parse_result(self):
        legal = ["up", "down", "left", "right", "stand"]
        self.assertIsInstance(
            parse_response('```json\n{"move": "up"}\n```', legal),
            ParseResult,
        )


class GeneratePromptTest(absltest.TestCase):

    def test_prompt_includes_team_and_seat(self):
        s = _state(seed=7)
        prompt = generate_prompt(_make_observation(s, 0), [])
        self.assertIn("team id is 0", prompt)
        self.assertIn("seat 0", prompt)
        self.assertIn("teammate is player 1", prompt)

    def test_prompt_for_team_b(self):
        s = _state(seed=7)
        prompt = generate_prompt(_make_observation(s, 3), [])
        self.assertIn("team id is 1", prompt)
        self.assertIn("player 3", prompt)
        self.assertIn("teammate is player 2", prompt)

    def test_prompt_explains_2v2(self):
        s = _state(seed=7)
        prompt = generate_prompt(_make_observation(s, 0), [])
        self.assertIn("2v2", prompt)
        self.assertIn("HIDDEN from", prompt)
        self.assertIn("self_pref^2 + other_pref^2 - bad_coins^2", prompt)
        self.assertIn("up, down, left, right, stand", prompt)

    def test_prompt_says_teammate_is_a_copy(self):
        s = _state(seed=7)
        prompt = generate_prompt(_make_observation(s, 0), [])
        # The model needs to know it's playing alongside another instance
        # of itself against another agent's two copies — that's the whole
        # point of the arena setup. Collapse whitespace so wrapped lines
        # in the template don't break the substring check.
        flat = " ".join(prompt.split())
        self.assertIn("another instance of YOU", flat)
        self.assertIn("two instances of a single different agent", flat)
        self.assertIn("NO in-game communication", flat)

    def test_prompt_only_contains_own_team_board(self):
        # Drive the game forward so each board has a distinguishable
        # state, then ensure player 0's prompt does not leak any of
        # team B's board info (and vice versa).
        s = _state(seed=7)
        # Sequential order [0, 2, 1, 3]: seat 0 then seat 1, on each board.
        s.apply_action(3)  # player 0
        s.apply_action(2)  # player 2
        s.apply_action(1)  # player 1
        s.apply_action(0)  # player 3

        obs_a = _make_observation(s, 0)
        obs_b = _make_observation(s, 2)
        parsed_a = json.loads(obs_a["observationString"])
        parsed_b = json.loads(obs_b["observationString"])

        # Sanity: env emits only the viewer's board, not "boards".
        self.assertIn("board", parsed_a)
        self.assertNotIn("boards", parsed_a)
        self.assertEqual(parsed_a["board"]["team_id"], 0)
        self.assertEqual(parsed_b["board"]["team_id"], 1)

        prompt_a = generate_prompt(obs_a, [])
        prompt_b = generate_prompt(obs_b, [])

        # Team A's prompt must reference its own players (ids 0 and 1)
        # and never the opponents (ids 2 and 3) — and vice versa. Player
        # ids appear both in the player_positions/coins_collected JSON
        # and in the rendered board (as digit cells).
        self.assertIn('"0"', prompt_a)
        self.assertIn('"1"', prompt_a)
        self.assertNotIn('"2"', prompt_a)
        self.assertNotIn('"3"', prompt_a)
        self.assertIn('"2"', prompt_b)
        self.assertIn('"3"', prompt_b)
        self.assertNotIn('"0"', prompt_b)
        self.assertNotIn('"1"', prompt_b)

        # The opposing team's preferences must not leak either.
        opp_pref_a = parsed_b["your_preference"]
        opp_pref_b = parsed_a["your_preference"]
        # If by chance the two teams happen to share a preferred colour,
        # skip the leakage check (the colour itself isn't secret then).
        if opp_pref_a != parsed_a["your_preference"]:
            self.assertNotIn(
                f'preferred coin colour is "{opp_pref_a}"', prompt_a,
            )
        if opp_pref_b != parsed_b["your_preference"]:
            self.assertNotIn(
                f'preferred coin colour is "{opp_pref_b}"', prompt_b,
            )

    def test_prompt_includes_player_preference(self):
        s = _state(seed=7)
        # Pull the player's actual preference from the obs and check it
        # appears in the prompt.
        obs = _make_observation(s, 0)
        pref = json.loads(obs["observationString"])["your_preference"]
        prompt = generate_prompt(obs, [])
        self.assertIn(f'preferred coin colour is "{pref}"', prompt)

    def test_history_shows_partner_moves(self):
        s = _state(seed=7)
        # Sequential order [0, 2, 1, 3]: seat 0 right, then seat 1 left
        # on each board.
        s.apply_action(3)  # player 0: right
        s.apply_action(3)  # player 2: right
        s.apply_action(2)  # player 1: left
        s.apply_action(2)  # player 3: left
        # Now seat 0 (player 0) is up again; their prompt should include both.
        prompt = generate_prompt(_make_observation(s, 0), [])
        self.assertIn("seat 0", prompt)
        self.assertIn("right", prompt)
        # Partner's most recent move must appear.
        self.assertIn("left", prompt)

    def test_rethink_suffix(self):
        s = _state(seed=7)
        prompt = generate_prompt(
            _make_observation(s, 0), [],
            previous_response="I play diagonally",
            previous_action="diagonal",
        )
        self.assertIn("Your previous response was", prompt)
        self.assertIn("diagonal", prompt)


class GetLegalMovesTest(absltest.TestCase):

    def test_active_player_gets_five_moves(self):
        s = _state(seed=7)
        result = get_legal_moves(_make_observation(s, 0))
        self.assertEqual(
            sorted(result.values()),
            ["down", "left", "right", "stand", "up"],
        )

    def test_inactive_player_gets_empty(self):
        s = _state(seed=7)
        # Player 1 (seat 1 on team A) is inactive on the first step.
        result = get_legal_moves(_make_observation(s, 1))
        self.assertEqual(result, {})


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


class _ArenaHarness:
    def get_legal_moves(self, observation):
        return get_legal_moves(observation)

    def make_prompt(
        self, observation, move_history,
        previous_response=None, previous_action=None,
    ):
        return generate_prompt(
            observation, move_history,
            previous_response=previous_response,
            previous_action=previous_action,
        )

    def parse_response(self, response, legal_action_strings):
        return parse_response(response, legal_action_strings)


_ENV = {
    "MODEL_NAME": "test-model",
    "MODEL_PROXY_KEY": "test-key",
    "MODEL_PROXY_URL": "dummy_url",
}


class AgentIntegrationTest(absltest.TestCase):

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_setup_step_returns_inactive(self, mock_litellm):
        mock_litellm.drop_params = True
        agent = create_agent_fn(_ArenaHarness())
        result = agent({"step": 0, "remainingOverageTime": 60}, {})
        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            '```json\n{"move": "up"}\n```',
        )
        agent = create_agent_fn(_ArenaHarness())

        s = _state(seed=7)
        observation = _make_observation(s, 0)
        result = agent(observation, {})

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["actionString"], "up")
        # Action id 0 is "up" by our enum.
        self.assertEqual(result["submission"], 0)


if __name__ == "__main__":
    absltest.main()
