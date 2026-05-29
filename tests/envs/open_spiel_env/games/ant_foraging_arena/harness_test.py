"""Tests for the Ant Foraging Arena LLM harness."""

import json

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult
from kaggle_environments.envs.open_spiel_env.games.ant_foraging_arena import (
    ant_foraging_arena_game,  # noqa: F401  (registers the game)
)
from kaggle_environments.envs.open_spiel_env.games.ant_foraging_arena.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


_DIRECTIONS = ("stay", "up", "down", "left", "right")


def _make_arena_observation(player_id=0, seed=1):
    game = pyspiel.load_game("ant_foraging_arena", {"seed": seed})
    state = game.new_initial_state()
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "currentPlayer": state.current_player(),
        "isTerminal": state.is_terminal(),
        "legalActions": list(state.legal_actions(player_id)),
        "legalActionStrings": [
            state.action_to_string(player_id, a) for a in state.legal_actions(player_id)
        ],
        "serializedGameAndState": pyspiel.serialize_game_and_state(game, state),
    }


class ParseResponseTest(absltest.TestCase):
    def test_parse_json_direction(self):
        result = parse_response('```json\n{"move": "up"}\n```', list(_DIRECTIONS))
        self.assertEqual(result.legal_action, "up")
        self.assertEqual(result.raw_action, "up")

    def test_parse_case_insensitive(self):
        result = parse_response('```json\n{"move": "LEFT"}\n```', list(_DIRECTIONS))
        self.assertEqual(result.legal_action, "left")

    def test_parse_each_direction(self):
        for direction in _DIRECTIONS:
            result = parse_response(
                f'```json\n{{"move": "{direction}"}}\n```', list(_DIRECTIONS)
            )
            self.assertEqual(result.legal_action, direction)

    def test_parse_bare_json(self):
        result = parse_response(
            'Thinking… {"move": "down"} that is my choice.', list(_DIRECTIONS)
        )
        self.assertEqual(result.legal_action, "down")

    def test_prose_only_response_triggers_rethink(self):
        # No structured JSON answer. The parser must NOT guess at intent
        # from a direction word in the prose; that's the ghost-fallback
        # antipattern. Return None so the rethink loop asks the model
        # for a structured answer.
        response = "I considered going up but ended up choosing down"
        result = parse_response(response, list(_DIRECTIONS))
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_prefers_last_json_block(self):
        # Two JSON blocks: a draft in the reasoning and a final answer.
        response = (
            'First draft: ```json\n{"move": "up"}\n```\n'
            'Reconsidering, final: ```json\n{"move": "right"}\n```'
        )
        result = parse_response(response, list(_DIRECTIONS))
        self.assertEqual(result.legal_action, "right")

    def test_parse_no_match_returns_none(self):
        result = parse_response('```json\n{"move": "diagonal"}\n```', list(_DIRECTIONS))
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "diagonal")

    def test_parse_no_direction_returns_none(self):
        result = parse_response("I have no idea what to do.", list(_DIRECTIONS))
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_skips_directions_not_legal(self):
        # If only "stay" is legal, mentioning "up" should not match.
        result = parse_response("Maybe up?", ["stay"])
        self.assertIsNone(result.legal_action)

    def test_malformed_json_triggers_rethink(self):
        # Bad JSON block means stage-1 extracts nothing. The parser must
        # NOT silently rescue an action from the prose -- the model gets
        # a chance to fix its format via the rethink loop instead.
        response = "```json\n{bad}\n```\nFinal answer: stay"
        result = parse_response(response, list(_DIRECTIONS))
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_returns_parse_result(self):
        result = parse_response('```json\n{"move": "up"}\n```', list(_DIRECTIONS))
        self.assertIsInstance(result, ParseResult)

    def test_illegal_json_does_not_ghost_substitute_from_prose(self):
        # The model gave an explicit JSON answer ("diagonal") that isn't
        # legal. The parser must NOT silently substitute a direction word
        # mentioned elsewhere in the prose -- that's the ghost-fallback
        # antipattern. Surface raw_action so the rethink loop fires.
        response = (
            "I considered up but ruled it out. I'll play diagonal.\n"
            '```json\n{"move": "diagonal"}\n```'
        )
        result = parse_response(response, list(_DIRECTIONS))
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "diagonal")


class GeneratePromptTest(absltest.TestCase):
    def test_includes_core_rules(self):
        obs = _make_arena_observation(player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Ant Foraging Arena", prompt)
        self.assertIn("2v2", prompt)
        self.assertIn("stay, up, down, left, right", prompt)

    def test_includes_player_team_and_seat(self):
        obs0 = _make_arena_observation(player_id=0)
        prompt0 = generate_prompt(obs0, [])
        self.assertIn("team id is 0", prompt0)
        self.assertIn("player 0", prompt0)
        self.assertIn("seat 0", prompt0)

        obs2 = _make_arena_observation(player_id=2)
        prompt2 = generate_prompt(obs2, [])
        self.assertIn("team id is 1", prompt2)
        self.assertIn("player 2", prompt2)

    def test_includes_grid_and_food_count(self):
        prompt = generate_prompt(_make_arena_observation(player_id=0), [])
        self.assertIn("8x8", prompt)
        self.assertIn("3 food", prompt)

    def test_other_team_board_hidden(self):
        # The per-player observation must not leak the opposing board.
        obs2 = _make_arena_observation(player_id=2)
        prompt = generate_prompt(obs2, [])
        parsed = json.loads(obs2["observationString"])
        # Team B's view exposes board team_id == 1 only.
        self.assertEqual(parsed["board"]["team_id"], 1)
        self.assertNotIn('"team_id": 0', prompt)

    def test_rethink_suffix_appended(self):
        prompt = generate_prompt(
            _make_arena_observation(player_id=0),
            [],
            previous_response="I'll go diagonal",
            previous_action="diagonal",
        )
        self.assertIn("Your previous response was", prompt)
        self.assertIn("diagonal", prompt)
        self.assertIn("not in the legal", prompt)

    def test_no_rethink_on_first_attempt(self):
        prompt = generate_prompt(_make_arena_observation(player_id=0), [])
        self.assertNotIn("Your previous response was", prompt)

    def test_empty_move_history(self):
        prompt = generate_prompt(_make_arena_observation(player_id=0), [])
        self.assertIn("(no moves yet)", prompt)


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        observation = {
            "legalActions": [0, 1, 4],
            "legalActionStrings": ["stay", "up", "right"],
        }
        result = get_legal_moves(observation)
        self.assertEqual(result, {0: "stay", 1: "up", 4: "right"})

    def test_empty_when_off_turn(self):
        # Inactive players get no actions.
        observation = {"legalActions": [], "legalActionStrings": []}
        self.assertEqual(get_legal_moves(observation), {})

    def test_from_serialized_state(self):
        # Player 0 starts at the nest centre, all 5 actions legal.
        obs = _make_arena_observation(player_id=0)
        # Drop the helpful pre-computed fields to force the fallback path.
        obs.pop("legalActions")
        obs.pop("legalActionStrings")
        result = get_legal_moves(obs)
        self.assertEqual(
            sorted(result.values()),
            ["down", "left", "right", "stay", "up"],
        )

    def test_off_turn_player_via_serialized_state(self):
        # Step 0 belongs to player 0; player 2 has no legal actions.
        obs = _make_arena_observation(player_id=2)
        obs.pop("legalActions")
        obs.pop("legalActionStrings")
        self.assertEqual(get_legal_moves(obs), {})

    def test_empty_serialized(self):
        self.assertEqual(get_legal_moves({"serializedGameAndState": ""}), {})


if __name__ == "__main__":
    absltest.main()
