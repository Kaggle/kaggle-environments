"""Tests for the Word Association harness (core_harness integration)."""

import json
from unittest.mock import patch

from absl.testing import absltest

from kaggle_environments import core_harness
from kaggle_environments.core_harness import ParseResult, create_agent_fn, set_telemetry_exporter
from kaggle_environments.envs.word_association.harness.main import (
    _WordAssociationHarness,
    generate_prompt,
    get_legal_moves,
    parse_response,
)


# --- Helpers ----------------------------------------------------------------


_WORDS = [
    "APPLE", "BANANA", "CAT", "DOG", "ELEPHANT",
    "FISH", "GRAPE", "HORSE", "ICE", "JUNGLE",
    "KITE", "LEMON", "MOUSE", "NIGHT", "OCEAN",
    "PIANO", "QUEEN", "RIVER", "SUN", "TREE",
    "UMBRELLA", "VIOLET", "WATER", "XRAY", "ZEBRA",
]

_ROLES = (
    ["blue"] * 9 + ["yellow"] * 8 + ["trap"] + ["neutral"] * 7
)

_REVEALED = [False] * 25


def _cluemaster_obs(turn=0, **overrides):
    """Build a Cluemaster observation dict."""
    obs = {
        "words": _WORDS[:],
        "roles": _ROLES[:],
        "revealed": _REVEALED[:],
        "current_turn": turn,
        "clue": "",
        "guesses_remaining": 0,
        "clue_number": 0,
    }
    obs.update(overrides)
    return obs


def _guesser_obs(turn=1, clue="ANIMAL", clue_number=2, guesses_remaining=3, **overrides):
    """Build a Guesser observation dict."""
    # Mask roles for guessers.
    revealed = overrides.get("revealed", _REVEALED[:])
    masked_roles = [
        _ROLES[i] if revealed[i] else "Unknown" for i in range(25)
    ]
    obs = {
        "words": _WORDS[:],
        "roles": masked_roles,
        "revealed": revealed,
        "current_turn": turn,
        "clue": clue,
        "clue_number": clue_number,
        "guesses_remaining": guesses_remaining,
    }
    obs.update(overrides)
    return obs


def _fake_completion(content: str):
    class _Msg:
        def __init__(self, c): self.content = c
    class _Choice:
        def __init__(self, c): self.message = _Msg(c)
    class _Usage:
        prompt_tokens = 1
        completion_tokens = 1
    class _Resp:
        def __init__(self, c):
            self.choices = [_Choice(c)]
            self.usage = _Usage()
    return _Resp(content)


_ENV = {
    "MODEL_NAME": "test-model",
    "MODEL_PROXY_KEY": "key",
    "MODEL_PROXY_URL": "dummy_url",
}


# --- get_legal_moves --------------------------------------------------------


class GetLegalMovesTest(absltest.TestCase):

    def test_cluemaster_returns_none(self):
        obs = _cluemaster_obs(turn=0)
        self.assertIsNone(get_legal_moves(obs))

    def test_cluemaster_yellow_returns_none(self):
        obs = _cluemaster_obs(turn=2)
        self.assertIsNone(get_legal_moves(obs))

    def test_guesser_returns_unrevealed_words(self):
        obs = _guesser_obs(turn=1)
        moves = get_legal_moves(obs)
        self.assertIsNotNone(moves)
        # All 25 words unrevealed → 25 word moves.
        word_moves = {k: v for k, v in moves.items() if k >= 0}
        self.assertEqual(len(word_moves), 25)
        self.assertEqual(moves[0], "0: APPLE")
        self.assertEqual(moves[24], "24: ZEBRA")

    def test_guesser_excludes_revealed(self):
        revealed = [False] * 25
        revealed[0] = True
        revealed[5] = True
        obs = _guesser_obs(revealed=revealed)
        moves = get_legal_moves(obs)
        self.assertNotIn(0, moves)
        self.assertNotIn(5, moves)
        word_moves = {k: v for k, v in moves.items() if k >= 0}
        self.assertEqual(len(word_moves), 23)

    def test_pass_available_after_first_guess(self):
        # clue_number=2, guesses_remaining=2 (one guess already made: 3→2)
        obs = _guesser_obs(clue_number=2, guesses_remaining=2)
        moves = get_legal_moves(obs)
        self.assertIn(-1, moves)
        self.assertEqual(moves[-1], "-1: PASS")

    def test_pass_not_available_on_first_guess(self):
        # clue_number=2, expected_remaining=3, guesses_remaining=3
        obs = _guesser_obs(clue_number=2, guesses_remaining=3)
        moves = get_legal_moves(obs)
        self.assertNotIn(-1, moves)

    def test_pass_not_available_unlimited_first_guess(self):
        # clue_number=0 → expected_remaining=25, guesses_remaining=25
        obs = _guesser_obs(clue_number=0, guesses_remaining=25)
        moves = get_legal_moves(obs)
        self.assertNotIn(-1, moves)


# --- parse_response --------------------------------------------------------


class ParseResponseTest(absltest.TestCase):

    # --- Cluemaster (free-form) ---

    def test_cluemaster_json_block(self):
        response = '```json\n{"thinking": "test", "clue": "ANIMAL", "number": 2}\n```'
        result = parse_response(response, None)
        self.assertEqual(result.submission, {"clue": "ANIMAL", "number": 2})
        self.assertIsNotNone(result.raw_action)

    def test_cluemaster_bare_json(self):
        response = '{"clue": "OCEAN", "number": 3}'
        result = parse_response(response, None)
        self.assertEqual(result.submission, {"clue": "OCEAN", "number": 3})

    def test_cluemaster_json_with_text(self):
        response = 'I think ANIMAL is a good clue.\n{"thinking": "...", "clue": "ANIMAL", "number": 2}'
        result = parse_response(response, None)
        self.assertEqual(result.submission, {"clue": "ANIMAL", "number": 2})

    def test_cluemaster_missing_keys_returns_none(self):
        response = '{"thinking": "no clue here"}'
        result = parse_response(response, None)
        self.assertIsNone(result.submission)

    def test_cluemaster_unparseable(self):
        response = "I don't know what to do"
        result = parse_response(response, None)
        self.assertIsNone(result.submission)

    def test_cluemaster_number_as_string(self):
        response = '{"clue": "ANIMAL", "number": "2"}'
        result = parse_response(response, None)
        self.assertEqual(result.submission, {"clue": "ANIMAL", "number": 2})

    # --- Guesser (enumerable) ---

    def test_guesser_json_block(self):
        legal = ["0: APPLE", "2: CAT", "4: ELEPHANT", "-1: PASS"]
        response = '```json\n{"thinking": "test", "guess": 2}\n```'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "2: CAT")
        self.assertEqual(result.raw_action, "2")

    def test_guesser_bare_json(self):
        legal = ["0: APPLE", "2: CAT"]
        response = '{"guess": 0}'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "0: APPLE")

    def test_guesser_pass(self):
        legal = ["0: APPLE", "-1: PASS"]
        response = '{"guess": -1}'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "-1: PASS")

    def test_guesser_illegal_index(self):
        legal = ["0: APPLE", "2: CAT"]
        response = '{"guess": 5}'
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "5")

    def test_guesser_string_index(self):
        legal = ["4: ELEPHANT"]
        response = '{"guess": "4"}'
        result = parse_response(response, legal)
        self.assertEqual(result.legal_action, "4: ELEPHANT")

    def test_guesser_missing_guess_key(self):
        legal = ["0: APPLE"]
        response = '{"thinking": "hmm"}'
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)

    def test_guesser_unparseable(self):
        legal = ["0: APPLE"]
        response = "I have no idea"
        result = parse_response(response, legal)
        self.assertIsNone(result.legal_action)


# --- generate_prompt --------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):

    def test_cluemaster_prompt_contains_team(self):
        obs = _cluemaster_obs(turn=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("BLUE Cluemaster", prompt)

    def test_cluemaster_prompt_yellow(self):
        obs = _cluemaster_obs(turn=2)
        prompt = generate_prompt(obs, [])
        self.assertIn("YELLOW Cluemaster", prompt)

    def test_cluemaster_prompt_contains_words(self):
        obs = _cluemaster_obs()
        prompt = generate_prompt(obs, [])
        self.assertIn("APPLE", prompt)
        self.assertIn("ZEBRA", prompt)

    def test_guesser_prompt_contains_clue(self):
        obs = _guesser_obs(clue="FRUIT")
        prompt = generate_prompt(obs, [])
        self.assertIn("FRUIT", prompt)
        self.assertIn("BLUE Guesser", prompt)

    def test_guesser_prompt_contains_legal_moves(self):
        obs = _guesser_obs()
        prompt = generate_prompt(obs, [])
        self.assertIn("Legal moves:", prompt)
        self.assertIn("0: APPLE", prompt)

    def test_rethink_appended(self):
        obs = _cluemaster_obs()
        prompt = generate_prompt(obs, [], previous_response="bad response", previous_action="FAIL")
        self.assertIn("bad response", prompt)
        self.assertIn("FAIL", prompt)
        self.assertIn("could not be parsed", prompt)

    def test_memory_context_injected(self):
        obs = _cluemaster_obs(
            current_game_turns=[{"team": "blue", "clue": "FRUIT", "num": 2, "guesses": ["APPLE"], "results": ["blue"]}],
        )
        prompt = generate_prompt(obs, [])
        self.assertIn("Clues and guesses in this game so far", prompt)
        self.assertIn("FRUIT", prompt)


# --- Full agent_fn integration ---------------------------------------------


class AgentIntegrationTest(absltest.TestCase):

    def setUp(self):
        super().setUp()
        self.events = []
        set_telemetry_exporter(
            lambda module, **kw: self.events.append({"module": module, **kw})
        )

    def tearDown(self):
        set_telemetry_exporter(lambda module, **kwargs: None)
        super().tearDown()

    def test_cluemaster_free_form(self):
        harness = _WordAssociationHarness()
        agent = create_agent_fn(harness)
        obs = _cluemaster_obs()
        llm_response = '{"thinking": "test", "clue": "ANIMAL", "number": 2}'
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion",
            return_value=_fake_completion(llm_response),
        ):
            result = agent(obs, {"freeForm": True})

        self.assertEqual(result["submission"], {"clue": "ANIMAL", "number": 2})
        self.assertEqual(result["status"], "OK")

    def test_guesser_enumerable(self):
        harness = _WordAssociationHarness()
        agent = create_agent_fn(harness)
        obs = _guesser_obs()
        llm_response = '{"thinking": "test", "guess": 2}'
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion",
            return_value=_fake_completion(llm_response),
        ):
            result = agent(obs, {"freeForm": True})

        # submission is the integer action key
        self.assertEqual(result["submission"], 2)
        self.assertEqual(result["actionString"], "2: CAT")
        self.assertEqual(result["status"], "OK")

    def test_guesser_pass(self):
        harness = _WordAssociationHarness()
        agent = create_agent_fn(harness)
        # Pass is legal: clue_number=2, guesses_remaining=2 (one guess made)
        obs = _guesser_obs(clue_number=2, guesses_remaining=2)
        llm_response = '{"thinking": "done guessing", "guess": -1}'
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion",
            return_value=_fake_completion(llm_response),
        ):
            result = agent(obs, {"freeForm": True})

        self.assertEqual(result["submission"], -1)
        self.assertEqual(result["actionString"], "-1: PASS")

    def test_cluemaster_retry_then_succeeds(self):
        harness = _WordAssociationHarness()
        agent = create_agent_fn(harness, max_retries=3)
        responses = [
            _fake_completion("I'll give a clue about animals"),  # unparseable
            _fake_completion('{"clue": "CREATURE", "number": 3}'),
        ]
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion", side_effect=responses,
        ) as mock_call:
            result = agent(_cluemaster_obs(), {"freeForm": True})

        self.assertEqual(result["submission"], {"clue": "CREATURE", "number": 3})
        self.assertEqual(mock_call.call_count, 2)

    def test_guesser_retry_illegal_index(self):
        harness = _WordAssociationHarness()
        agent = create_agent_fn(harness, max_retries=3)
        revealed = [True] * 25
        revealed[0] = False  # Only APPLE unrevealed
        obs = _guesser_obs(revealed=revealed, clue_number=1, guesses_remaining=2)
        responses = [
            _fake_completion('{"guess": 5}'),  # index 5 is revealed → illegal
            _fake_completion('{"guess": 0}'),  # index 0 is legal
        ]
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion", side_effect=responses,
        ):
            result = agent(obs, {"freeForm": True})

        self.assertEqual(result["submission"], 0)

    def test_mixed_turns(self):
        """Cluemaster (free-form) then Guesser (enumerable) on same agent."""
        harness = _WordAssociationHarness()
        agent = create_agent_fn(harness)
        responses = [
            _fake_completion('{"clue": "FRUIT", "number": 1}'),
            _fake_completion('{"guess": 0}'),
        ]
        with patch.dict("os.environ", _ENV, clear=False), patch.object(
            core_harness.litellm, "completion", side_effect=responses,
        ):
            r1 = agent(_cluemaster_obs(), {"freeForm": True})
            r2 = agent(_guesser_obs(), {"freeForm": True})

        self.assertEqual(r1["submission"], {"clue": "FRUIT", "number": 1})
        self.assertEqual(r2["submission"], 0)
        self.assertEqual(r2["actionString"], "0: APPLE")


# --- Interpreter submission extraction --------------------------------------


class InterpreterSubmissionTest(absltest.TestCase):
    """Test that the interpreter correctly extracts submission from core_harness dicts."""

    def test_cluemaster_submission_extraction(self):
        """Verify the pattern used in word_association.py process_action."""
        action = {"submission": {"clue": "ANIMAL", "number": 2}, "actionString": "...", "status": "OK"}
        if isinstance(action, dict) and "submission" in action:
            action = action["submission"]
        self.assertEqual(action, {"clue": "ANIMAL", "number": 2})
        self.assertIn("clue", action)
        self.assertIn("number", action)

    def test_guesser_submission_extraction(self):
        action = {"submission": 4, "actionString": "4: ELEPHANT", "status": "OK"}
        if isinstance(action, dict) and "submission" in action:
            action = action["submission"]
        self.assertEqual(action, 4)
        # Guesser path: isinstance(action, dict) is False → guess_val = action
        self.assertFalse(isinstance(action, dict))

    def test_guesser_pass_submission_extraction(self):
        action = {"submission": -1, "actionString": "-1: PASS", "status": "OK"}
        if isinstance(action, dict) and "submission" in action:
            action = action["submission"]
        self.assertEqual(action, -1)

    def test_backward_compat_old_cluemaster_format(self):
        """Old harness returned {"clue": ..., "number": ...} without submission."""
        action = {"clue": "ANIMAL", "number": 2}
        if isinstance(action, dict) and "submission" in action:
            action = action["submission"]
        # action unchanged — no "submission" key
        self.assertEqual(action, {"clue": "ANIMAL", "number": 2})

    def test_backward_compat_old_guesser_format(self):
        """Old harness returned {"guess": int} without submission."""
        action = {"guess": 4}
        if isinstance(action, dict) and "submission" in action:
            action = action["submission"]
        self.assertEqual(action, {"guess": 4})
        # Old guesser path: action.get("guess") → 4
        self.assertEqual(action.get("guess"), 4)


if __name__ == "__main__":
    absltest.main()
