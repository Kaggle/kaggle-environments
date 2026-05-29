"""Tests for the Python Ant Foraging LLM harness."""

import json
from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.python_ant_foraging import (
    python_ant_foraging_proxy,
)
from kaggle_environments.envs.open_spiel_env.games.python_ant_foraging.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(state, game, player_id=0):
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "currentPlayer": state.current_player(),
        "isTerminal": state.is_terminal(),
        "serializedGameAndState": pyspiel.serialize_game_and_state(
            game.__wrapped__,
            state.__wrapped__,
        ),
    }


def _legal_for_player(player_id):
    return [
        f"ant{player_id}:stay",
        f"ant{player_id}:up",
        f"ant{player_id}:down",
        f"ant{player_id}:left",
        f"ant{player_id}:right",
    ]


class ParseResponseTest(absltest.TestCase):
    def test_parse_json_bare_direction(self):
        legal = _legal_for_player(0)
        response = '```json\n{"move": "up"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "ant0:up")
        self.assertEqual(result.raw_action, "up")

    def test_parse_json_qualified_action(self):
        legal = _legal_for_player(1)
        response = '```json\n{"move": "ant1:down"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "ant1:down")

    def test_parse_case_insensitive(self):
        legal = _legal_for_player(0)
        response = '```json\n{"move": "RIGHT"}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "ant0:right")

    def test_parse_each_direction(self):
        legal = _legal_for_player(0)
        for direction in ("stay", "up", "down", "left", "right"):
            response = f'```json\n{{"move": "{direction}"}}\n```'
            result = parse_response(response, legal)
            self.assertEqual(result.legal_action, f"ant0:{direction}")

    def test_prose_only_response_triggers_rethink(self):
        # No structured JSON. The parser must NOT guess at intent from a
        # direction keyword in the prose -- return None and let the
        # rethink loop ask the model to use the required JSON format.
        legal = _legal_for_player(0)
        response = "I'll head right toward the food I can see."
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_bare_json(self):
        legal = _legal_for_player(0)
        response = 'After thinking, {"move": "stay"} is best.'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "ant0:stay")

    def test_malformed_json_triggers_rethink(self):
        # Bad JSON: stage-1 extracts nothing. The parser must NOT
        # silently rescue a direction from the prose -- return None so
        # the rethink loop can ask the model to fix its format.
        legal = _legal_for_player(0)
        response = "```json\n{bad}\n```\nI'll go up."
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_no_match_returns_none(self):
        legal = _legal_for_player(0)
        response = '```json\n{"move": "diagonal"}\n```'
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "diagonal")

    def test_parse_no_move_returns_none(self):
        legal = _legal_for_player(0)
        response = "I have no idea what to play."
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_returns_parse_result(self):
        legal = _legal_for_player(0)
        result = parse_response('```json\n{"move": "up"}\n```', legal)
        self.assertIsInstance(result, ParseResult)

    def test_parse_skips_directions_not_legal(self):
        # If only "stay" is legal, "up" in the text shouldn't match.
        legal = ["ant0:stay"]
        response = "I think I should go up."
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)

    def test_illegal_json_does_not_ghost_substitute_from_prose(self):
        # The JSON answer ("diagonal") is illegal. The parser must NOT
        # silently substitute "up" mentioned in the prose (the ghost
        # antipattern). Surface raw_action so the rethink loop fires.
        legal = ["ant0:stay", "ant0:up", "ant0:down", "ant0:left", "ant0:right"]
        response = (
            "I considered up but ruled it out. I'll play diagonal.\n"
            '```json\n{"move": "diagonal"}\n```'
        )
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "diagonal")


class GeneratePromptTest(absltest.TestCase):
    def _obs(self, player_id=0, carrying=False, position=(4, 4), turn=0, score=0):
        ant_positions = [[4, 4], [4, 4]]
        ant_positions[player_id] = list(position)
        carrying_food = [False, False]
        carrying_food[player_id] = carrying
        state = json.dumps(
            {
                "grid_size": 8,
                "num_ants": 2,
                "num_food": 3,
                "max_turns": 50,
                "turn": turn,
                "food_collected": score,
                "score": score,
                "ant_positions": ant_positions,
                "carrying_food": carrying_food,
                "nest_position": [4, 4],
                "food_positions": [[3, 5], [4, 5], [5, 2]],
                "pheromone_to_food": [[0.0] * 8 for _ in range(8)],
                "pheromone_to_nest": [[0.0] * 8 for _ in range(8)],
                "current_player": player_id,
            }
        )
        return {"observationString": state, "playerId": player_id}

    def test_includes_rules(self):
        prompt = generate_prompt(self._obs(), [])
        self.assertIn("Ant Foraging", prompt)
        self.assertIn("cooperative", prompt)
        self.assertIn("stay, up, down, left, right", prompt)

    def test_includes_grid_size_and_food_count(self):
        prompt = generate_prompt(self._obs(), [])
        self.assertIn("8x8", prompt)
        self.assertIn("3 food", prompt)

    def test_includes_player_id_and_position(self):
        prompt = generate_prompt(self._obs(player_id=1, position=(2, 3)), [])
        self.assertIn("(ant id) is 1", prompt)
        self.assertIn("[2, 3]", prompt)

    def test_carrying_status_reflected(self):
        prompt_carrying = generate_prompt(self._obs(carrying=True), [])
        self.assertIn("carrying food", prompt_carrying)
        prompt_searching = generate_prompt(self._obs(carrying=False), [])
        self.assertIn("searching for food", prompt_searching)

    def test_move_history_included(self):
        prompt = generate_prompt(self._obs(), ["ant0:up", "ant1:down", "ant0:right"])
        self.assertIn("ant0:up, ant1:down, ant0:right", prompt)

    def test_empty_move_history_shows_none(self):
        prompt = generate_prompt(self._obs(), [])
        self.assertIn("Moves taken so far this game: None", prompt)

    def test_rethink_suffix(self):
        prompt = generate_prompt(
            self._obs(),
            [],
            previous_response="I tried diagonal",
            previous_action="diagonal",
        )
        self.assertIn("Your previous response was", prompt)
        self.assertIn("diagonal", prompt)
        self.assertIn("not a legal move", prompt)

    def test_no_rethink_on_first_attempt(self):
        prompt = generate_prompt(self._obs(), [])
        self.assertNotIn("Your previous response was", prompt)

    def test_does_not_enumerate_legal_actions(self):
        # The prompt should describe rules, not list each legal move.
        prompt = generate_prompt(self._obs(), [])
        # The literal phrase the framework uses for the legal list
        # shouldn't appear (we only describe the action set verbally).
        self.assertNotIn("Legal actions:", prompt)


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        observation = {
            "legalActions": [0, 1, 2],
            "legalActionStrings": ["ant0:stay", "ant0:up", "ant0:down"],
        }
        result = get_legal_moves(observation)
        self.assertEqual(result, {0: "ant0:stay", 1: "ant0:up", 2: "ant0:down"})

    def test_from_serialized_state(self):
        game = python_ant_foraging_proxy.PythonAntForagingGame()
        state = game.new_initial_state()
        observation = {
            "serializedGameAndState": pyspiel.serialize_game_and_state(
                game.__wrapped__,
                state.__wrapped__,
            ),
        }
        result = get_legal_moves(observation)
        # At the nest centre the ant has all 5 actions available.
        self.assertEqual(
            sorted(result.values()),
            ["ant0:down", "ant0:left", "ant0:right", "ant0:stay", "ant0:up"],
        )

    def test_empty_serialized(self):
        observation = {"serializedGameAndState": ""}
        self.assertEqual(get_legal_moves(observation), {})


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


class _AntHarness:
    """Adapter wrapping module-level functions in the GameHarness protocol."""

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
        agent = create_agent_fn(_AntHarness())
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
        agent = create_agent_fn(_AntHarness())

        game = python_ant_foraging_proxy.PythonAntForagingGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game, state.current_player())

        result = agent(observation, {})

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["actionString"], "ant0:up")
        self.assertEqual(
            state.__wrapped__.action_to_string(
                state.current_player(),
                result["submission"],
            ),
            "ant0:up",
        )

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "diagonal"}\n```'),
            _make_mock_response('```json\n{"move": "stay"}\n```'),
        ]
        agent = create_agent_fn(_AntHarness())

        game = python_ant_foraging_proxy.PythonAntForagingGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game, state.current_player())

        result = agent(observation, {})

        self.assertEqual(result["actionString"], "ant0:stay")
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            "I have no idea what to do",
        )
        agent = create_agent_fn(_AntHarness())

        game = python_ant_foraging_proxy.PythonAntForagingGame()
        state = game.new_initial_state()
        observation = _make_observation(state, game, state.current_player())

        with self.assertRaises(ValueError):
            agent(observation, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)


if __name__ == "__main__":
    absltest.main()
