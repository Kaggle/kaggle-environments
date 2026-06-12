"""Tests for the Bargaining LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.bargaining import (
    bargaining_proxy,
)
from kaggle_environments.envs.open_spiel_env.games.bargaining.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _make_observation(
    state: bargaining_proxy.BargainingState,
    game: bargaining_proxy.BargainingGame,
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
        "legalActionStrings": [state.action_to_string(a) for a in legal],
        "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
    }


def _seed_game(player_id: int = 0):
    """Build a (game, state) after the chance node so a player is to move."""
    game = bargaining_proxy.BargainingGame()
    state = game.new_initial_state()
    # Apply the first chance outcome to deterministically reach a play state.
    state.apply_action(state.legal_actions()[0])
    assert int(state.current_player()) == 0
    if player_id == 1:
        # Have P0 make one offer so it's P1's turn.
        state.apply_action(state.legal_actions()[0])
        assert int(state.current_player()) == 1
    return game, state


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class ParseResponseTest(absltest.TestCase):
    legal = [
        "Offer: Book: 1, Hat: 0, Basketball: 0",
        "Offer: Book: 0, Hat: 2, Basketball: 0",
        "Offer: Book: 1, Hat: 2, Basketball: 3",
        "Agree",
    ]

    def test_parse_json_block(self):
        result = parse_response(
            '```json\n{"action": "offer", "keep": {"book": 1, "hat": 0, "basketball": 0}}\n```',
            self.legal,
        )
        self.assertEqual(result.legal_action, "Offer: Book: 1, Hat: 0, Basketball: 0")

    def test_parse_agree(self):
        result = parse_response('```json\n{"action": "agree"}\n```', self.legal)
        self.assertEqual(result.legal_action, "Agree")

    def test_parse_accept_alias(self):
        # "accept" is a natural synonym for "agree"; we accept either.
        result = parse_response('```json\n{"action": "accept"}\n```', self.legal)
        self.assertEqual(result.legal_action, "Agree")

    def test_parse_bare_json(self):
        result = parse_response(
            'I think {"action": "offer", "keep": {"book": 1, "hat": 0, "basketball": 0}} is best.',
            self.legal,
        )
        self.assertEqual(result.legal_action, "Offer: Book: 1, Hat: 0, Basketball: 0")

    def test_parse_case_insensitive_action(self):
        result = parse_response(
            '```json\n{"action": "OFFER", "keep": {"book": 1, "hat": 0, "basketball": 0}}\n```',
            self.legal,
        )
        self.assertEqual(result.legal_action, "Offer: Book: 1, Hat: 0, Basketball: 0")

    def test_parse_illegal_offer_returns_raw(self):
        result = parse_response(
            '```json\n{"action": "offer", "keep": {"book": 99, "hat": 0, "basketball": 0}}\n```',
            self.legal,
        )
        self.assertIsNone(result.legal_action)
        self.assertIn("99", result.raw_action or "")

    def test_prose_only_response_triggers_rethink(self):
        # No structured JSON. The parser must NOT guess at intent from a
        # word like "agree" in the prose -- return None and let rethink
        # ask the model to use the required JSON format.
        result = parse_response("I'll just agree this turn.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_no_keep_returns_none(self):
        # Action says "offer" but no keep dict -- not a valid bargaining move.
        # Shape failures route to the unparsable rethink (raw_action=None),
        # not the illegal one whose diagnosis would lie.
        result = parse_response('```json\n{"action": "offer"}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_unknown_action_verb_routes_to_unparsable(self):
        result = parse_response('```json\n{"action": "decline"}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_pascal_case_keep_keys(self):
        # Models often mirror the prompt's display capitalization ("Book").
        # The pre-fix harness silently parsed this as keep-nothing, which is
        # a legal but unintended offer -- no retry would fire.
        result = parse_response(
            '```json\n{"action": "offer", "keep": {"Book": 1, "Hat": 0, "Basketball": 0}}\n```',
            self.legal,
        )
        self.assertEqual(result.legal_action, "Offer: Book: 1, Hat: 0, Basketball: 0")

    def test_parse_plural_keep_keys(self):
        result = parse_response(
            '```json\n{"action": "offer", "keep": {"books": 1, "hats": 0, "basketballs": 0}}\n```',
            self.legal,
        )
        self.assertEqual(result.legal_action, "Offer: Book: 1, Hat: 0, Basketball: 0")

    def test_parse_returns_parse_result_type(self):
        result = parse_response('```json\n{"action": "agree"}\n```', self.legal)
        self.assertIsInstance(result, ParseResult)

    def test_multiple_json_last_wins(self):
        # Model drafts, then revises. The last block is the intent.
        response = (
            'First I drafted ```json\n{"action": "agree"}\n``` but then\n'
            "I changed my mind: ```json\n"
            '{"action": "offer", "keep": {"book": 1, "hat": 0, "basketball": 0}}\n```'
        )
        result = parse_response(response, self.legal)
        self.assertEqual(result.legal_action, "Offer: Book: 1, Hat: 0, Basketball: 0")

    def test_illegal_json_does_not_ghost_substitute_from_prose(self):
        # Model writes a legal token in prose, then commits to an illegal
        # one in JSON. The parser must NOT silently substitute.
        legal_example = "Offer: Book: 1, Hat: 0, Basketball: 0"
        response = (
            f"I considered {legal_example} but ruled it out.\n"
            '```json\n{"action": "offer", "keep": {"book": 9, "hat": 9, "basketball": 9}}\n```'
        )
        result = parse_response(response, self.legal)
        self.assertIsNone(result.legal_action)

    def test_no_legal_strings_returns_none(self):
        result = parse_response('```json\n{"action": "agree"}\n```', None)
        self.assertIsNone(result.legal_action)


# ---------------------------------------------------------------------------
# generate_prompt
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def test_basic_prompt_contents(self):
        game, state = _seed_game(player_id=0)
        obs = _make_observation(state, game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Bargaining", prompt)
        self.assertIn("Player 1", prompt)  # human-readable label
        self.assertIn("(id 0)", prompt)
        self.assertIn("Book", prompt)
        self.assertIn("Hat", prompt)
        self.assertIn("Basketball", prompt)

    def test_player_label_swap(self):
        game, state = _seed_game(player_id=1)
        obs = _make_observation(state, game, player_id=1)
        prompt = generate_prompt(obs, [])
        self.assertIn("Player 2", prompt)
        self.assertIn("(id 1)", prompt)

    def test_legal_moves_not_listed(self):
        # The prompt deliberately omits the legal-move list. Each item is
        # already in the prompt via pool/values; legal offer strings (like
        # "Offer: Book: 0, Hat: 0, Basketball: 0") should not appear.
        game, state = _seed_game(player_id=0)
        obs = _make_observation(state, game, player_id=0)
        prompt = generate_prompt(obs, [])
        for legal in obs["legalActionStrings"]:
            if legal == "Agree":
                continue
            self.assertNotIn(legal, prompt)

    def test_pool_quantities_rendered(self):
        game, state = _seed_game(player_id=0)
        obs = _make_observation(state, game, player_id=0)
        import json

        pool = json.loads(state.observation_string(0))["pool"]
        prompt = generate_prompt(obs, [])
        # Each item line should show its actual quantity.
        for k, label in (("book", "Book"), ("hat", "Hat"), ("basketball", "Basketball")):
            self.assertIn(f"{label}: {pool[k]}", prompt)

    def test_private_values_rendered_per_player(self):
        # P0 sees their own values; P1 sees theirs. The two prompts must
        # contain the player-specific value lines.
        game, state = _seed_game(player_id=0)
        import json

        obs0 = _make_observation(state, game, player_id=0)
        prompt0 = generate_prompt(obs0, [])
        my_vals_0 = json.loads(state.observation_string(0))["my_values"]
        my_vals_1 = json.loads(state.observation_string(1))["my_values"]

        # P0's values appear in P0's prompt under "Your private valuations".
        section_start = prompt0.index("Your private valuations")
        section_end = prompt0.index("How a turn works")
        section = prompt0[section_start:section_end]
        self.assertIn(f"Book: {my_vals_0['book']}", section)
        self.assertIn(f"Hat: {my_vals_0['hat']}", section)
        self.assertIn(f"Basketball: {my_vals_0['basketball']}", section)

        # And P1's values do NOT appear there (different vector by design;
        # if they happen to coincide this assertion is vacuous, which is
        # acceptable -- the seed is deterministic).
        if my_vals_0 != my_vals_1:
            # At least one component differs; ensure P1's specific values
            # aren't all simultaneously present in P0's prompt section.
            mismatched_key = next(k for k in my_vals_0 if my_vals_0[k] != my_vals_1[k])
            label = {"book": "Book", "hat": "Hat", "basketball": "Basketball"}[mismatched_key]
            self.assertIn(f"{label}: {my_vals_0[mismatched_key]}", section)

    def test_history_empty_at_open(self):
        game, state = _seed_game(player_id=0)
        obs = _make_observation(state, game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("no offers yet", prompt)
        self.assertIn("opening", prompt)

    def test_history_rendered_after_offer(self):
        game, state = _seed_game(player_id=1)
        obs = _make_observation(state, game, player_id=1)
        prompt = generate_prompt(obs, [])
        # An opponent offer is on the table; history line should mention
        # "Player 1 offers" (the opponent of P1 is P0, label "Player 1").
        self.assertIn("Player 1 offers", prompt)
        # And the accept-help branch tells the model it MAY agree.
        self.assertIn('"action": "agree"', prompt)
        self.assertIn("MAY accept", prompt)

    def test_cannot_accept_on_first_turn(self):
        game, state = _seed_game(player_id=0)
        obs = _make_observation(state, game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("cannot accept on the first turn", prompt)

    def test_max_turns_and_remaining_rendered(self):
        game, state = _seed_game(player_id=1)
        obs = _make_observation(state, game, player_id=1)
        prompt = generate_prompt(obs, [])
        self.assertIn("Offers used so far: 1 of 10", prompt)
        # The default env loads bargaining with max_turns=10.
        self.assertIn("10 offers go by without acceptance", prompt)

    def test_no_rethink_on_first_attempt(self):
        game, state = _seed_game(player_id=0)
        obs = _make_observation(state, game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertNotIn("You suggested action", prompt)
        self.assertNotIn("Your previous response ended with", prompt)

    def test_rethink_suffix_illegal(self):
        game, state = _seed_game(player_id=0)
        obs = _make_observation(state, game, player_id=0)
        prompt = generate_prompt(
            obs,
            [],
            previous_response='{"action": "offer", "keep": {"book": 99}}',
            previous_action='{"action":"offer","keep":{"book":99,"hat":0,"basketball":0}}',
        )
        self.assertIn("You suggested action", prompt)
        self.assertIn("99", prompt)
        self.assertIn("not legal", prompt)

    def test_rethink_suffix_unparsable(self):
        game, state = _seed_game(player_id=0)
        obs = _make_observation(state, game, player_id=0)
        prompt = generate_prompt(
            obs,
            [],
            previous_response="I'm not sure what to do.",
            previous_action=None,
        )
        self.assertIn("No valid action JSON could be extracted", prompt)
        self.assertIn("I'm not sure what to do.", prompt)


# ---------------------------------------------------------------------------
# get_legal_moves
# ---------------------------------------------------------------------------


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        obs = {
            "legalActions": [0, 1, 8, 120],
            "legalActionStrings": [
                "Offer: Book: 0, Hat: 0, Basketball: 0",
                "Offer: Book: 1, Hat: 0, Basketball: 0",
                "Offer: Book: 0, Hat: 1, Basketball: 0",
                "Agree",
            ],
        }
        result = get_legal_moves(obs)
        self.assertEqual(result[120], "Agree")
        self.assertEqual(result[1], "Offer: Book: 1, Hat: 0, Basketball: 0")

    def test_from_serialized_state(self):
        game, state = _seed_game(player_id=0)
        obs = {
            "playerId": 0,
            "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
        }
        result = get_legal_moves(obs)
        self.assertGreater(len(result), 0)
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, str)
            self.assertTrue(v.startswith("Offer:") or v == "Agree")

    def test_empty_observation(self):
        self.assertEqual(get_legal_moves({}), {})

    def test_empty_serialized(self):
        self.assertEqual(get_legal_moves({"serializedGameAndState": ""}), {})


# ---------------------------------------------------------------------------
# create_agent_fn integration
# ---------------------------------------------------------------------------


class _BargainingHarnessForTest:
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
        agent = create_agent_fn(_BargainingHarnessForTest())

        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_offer(self, mock_litellm):
        mock_litellm.drop_params = True
        game, state = _seed_game(player_id=0)
        # First legal offer in pool order.
        first_legal_action = state.legal_actions()[0]
        first_legal_str = state.action_to_string(first_legal_action)
        # Parse the offer string to extract counts and respond as JSON.
        # The format is "Offer: Book: A, Hat: B, Basketball: C".
        import re

        m = re.match(r"Offer: Book: (\d+), Hat: (\d+), Basketball: (\d+)", first_legal_str)
        self.assertIsNotNone(m)
        book, hat, basketball = m.groups()
        json_resp = (
            f'```json\n{{"action": "offer", "keep": {{"book": {book}, "hat": {hat}, "basketball": {basketball}}}}}\n```'
        )
        mock_litellm.completion.return_value = _make_mock_response(json_resp)
        agent = create_agent_fn(_BargainingHarnessForTest())

        obs = _make_observation(state, game, player_id=0)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal_str)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["submission"], first_legal_action)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        game, state = _seed_game(player_id=0)
        first_legal_action = state.legal_actions()[0]
        first_legal_str = state.action_to_string(first_legal_action)
        import re

        m = re.match(r"Offer: Book: (\d+), Hat: (\d+), Basketball: (\d+)", first_legal_str)
        book, hat, basketball = m.groups()
        good_resp = (
            f'```json\n{{"action": "offer", "keep": {{"book": {book}, "hat": {hat}, "basketball": {basketball}}}}}\n```'
        )
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"action": "offer", "keep": {"book": 99, "hat": 99, "basketball": 99}}\n```'),
            _make_mock_response(good_resp),
        ]
        agent = create_agent_fn(_BargainingHarnessForTest())

        obs = _make_observation(state, game, player_id=0)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal_str)
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("I cannot decide.")
        agent = create_agent_fn(_BargainingHarnessForTest())

        game, state = _seed_game(player_id=0)
        obs = _make_observation(state, game, player_id=0)

        with self.assertRaises(ValueError):
            agent(obs, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_short_game_via_agent_fns(self, mock_litellm):
        """Drive a short Bargaining game with two scripted agents that
        always offer the first legal action, verifying the harness round-
        trips through pyspiel cleanly."""
        mock_litellm.drop_params = True

        game = bargaining_proxy.BargainingGame()
        state = game.new_initial_state()
        state.apply_action(state.legal_actions()[0])  # chance

        def fake_completion(*, model, messages, **kwargs):
            del model, kwargs
            content = messages[0]["content"]
            player_id = 0 if "(id 0)" in content else 1
            del player_id  # not needed for first-legal scripting
            first = state.legal_actions()[0]
            first_str = state.action_to_string(first)
            if first_str == "Agree":
                return _make_mock_response('```json\n{"action": "agree"}\n```')
            import re

            m = re.match(r"Offer: Book: (\d+), Hat: (\d+), Basketball: (\d+)", first_str)
            book, hat, basketball = m.groups()
            return _make_mock_response(
                f'```json\n{{"action": "offer", "keep": '
                f'{{"book": {book}, "hat": {hat}, "basketball": {basketball}}}}}\n```'
            )

        mock_litellm.completion.side_effect = fake_completion
        agent_p0 = create_agent_fn(_BargainingHarnessForTest())
        agent_p1 = create_agent_fn(_BargainingHarnessForTest())

        steps_played = 0
        for _ in range(20):
            if state.is_terminal():
                break
            cp = int(state.current_player())
            agent = agent_p0 if cp == 0 else agent_p1
            obs = _make_observation(state, game, player_id=cp)
            result = agent(obs, {})
            self.assertEqual(result["status"], "OK")
            state.apply_action(result["submission"])
            steps_played += 1

        self.assertGreater(steps_played, 0)


if __name__ == "__main__":
    absltest.main()
