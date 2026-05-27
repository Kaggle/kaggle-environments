"""Tests for the Negotiation LLM harness."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.negotiation.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)

_PROPOSAL_STATE = {
    "current_player": 1,
    "turn_type": "proposal",
    "max_steps": 7,
    "item_pool": [5, 5, 5],
    "my_utilities": [10, 10, 10],
    "proposals": [{"player": 0, "items": [3, 2, 1], "accept": False}],
    "utterances": [{"player": 0, "symbols": [1, 2, 0]}],
    "most_recent_proposal": [3, 2, 1],
    "most_recent_utterance": [1, 2, 0],
    "agreement_reached": False,
    "is_terminal": False,
    "winner": None,
    "rewards": None,
    "params": {
        "num_items": 3,
        "num_symbols": 5,
        "utterance_dim": 3,
        "enable_proposals": True,
        "enable_utterances": True,
        "max_quantity": 5,
        "num_distinct_proposals": 217,
        "accept_action": 216,
    },
}

_UTTERANCE_STATE = {
    **_PROPOSAL_STATE,
    "turn_type": "utterance",
    "current_player": 1,
    "proposals": [
        {"player": 0, "items": [3, 2, 1], "accept": False},
        {"player": 1, "items": [2, 2, 2], "accept": False},
    ],
    "utterances": [{"player": 0, "symbols": [1, 2, 0]}],
    "most_recent_proposal": [2, 2, 2],
}


def _observation(state: dict, player_id: int = 1, with_legal: bool = True) -> dict:
    obs = {
        "observationString": json.dumps(state),
        "playerId": player_id,
        "currentPlayer": state["current_player"],
        "isTerminal": state["is_terminal"],
    }
    if with_legal:
        moves = get_legal_moves({**obs, "legalActions": None})
        obs["legalActions"] = list(moves.keys())
        obs["legalActionStrings"] = list(moves.values())
    return obs


# ---------------------------------------------------------------------------


class ParseResponseTest(absltest.TestCase):
    def test_parse_accept(self):
        obs = _observation(_PROPOSAL_STATE)
        result = parse_response(
            'Looks good.\n```json\n{"action": "accept"}\n```',
            obs["legalActionStrings"],
        )
        self.assertEqual(result.legal_action, "Proposal: Agreement reached!")

    def test_parse_propose_keep(self):
        obs = _observation(_PROPOSAL_STATE)
        result = parse_response(
            '```json\n{"action": "propose", "keep": [0, 1, 1]}\n```',
            obs["legalActionStrings"],
        )
        self.assertEqual(result.legal_action, "Proposal: [0, 1, 1]")

    def test_parse_propose_items_alias(self):
        obs = _observation(_PROPOSAL_STATE)
        result = parse_response(
            '```json\n{"items": [4, 0, 5]}\n```',
            obs["legalActionStrings"],
        )
        self.assertEqual(result.legal_action, "Proposal: [4, 0, 5]")

    def test_parse_utterance(self):
        obs = _observation(_UTTERANCE_STATE)
        result = parse_response(
            '```json\n{"symbols": [4, 0, 2]}\n```',
            obs["legalActionStrings"],
        )
        self.assertEqual(result.legal_action, ", Utterance: [4, 0, 2]")

    def test_parse_bare_json(self):
        obs = _observation(_PROPOSAL_STATE)
        result = parse_response(
            'I will go with {"action": "propose", "keep": [2, 3, 1]} this round.',
            obs["legalActionStrings"],
        )
        self.assertEqual(result.legal_action, "Proposal: [2, 3, 1]")

    def test_parse_fallback_text_list(self):
        obs = _observation(_PROPOSAL_STATE)
        result = parse_response(
            "After thinking it over my final answer is propose [1, 1, 1].",
            obs["legalActionStrings"],
        )
        self.assertEqual(result.legal_action, "Proposal: [1, 1, 1]")

    def test_parse_fallback_accept_keyword(self):
        obs = _observation(_PROPOSAL_STATE)
        result = parse_response(
            "I accept the previous offer; it works for me.",
            obs["legalActionStrings"],
        )
        self.assertEqual(result.legal_action, "Proposal: Agreement reached!")

    def test_parse_illegal_returns_none(self):
        obs = _observation(_PROPOSAL_STATE)
        # 9 exceeds pool[1]=5; not a legal proposal.
        result = parse_response(
            '```json\n{"action": "propose", "keep": [0, 9, 0]}\n```',
            obs["legalActionStrings"],
        )
        self.assertIsNone(result.legal_action)
        self.assertIsNotNone(result.raw_action)

    def test_parse_returns_parse_result(self):
        result = parse_response('```json\n{"action": "accept"}\n```', ["Proposal: Agreement reached!"])
        self.assertIsInstance(result, ParseResult)


class GeneratePromptTest(absltest.TestCase):
    def test_proposal_prompt_contains_pool_and_utilities(self):
        obs = _observation(_PROPOSAL_STATE)
        prompt = generate_prompt(obs, [])
        self.assertIn("Negotiation", prompt)
        self.assertIn("item 0: 5 units in pool", prompt)
        self.assertIn("item 0: 10 per unit", prompt)
        self.assertIn("PROPOSAL turn", prompt)

    def test_proposal_prompt_includes_history(self):
        obs = _observation(_PROPOSAL_STATE)
        prompt = generate_prompt(obs, [])
        self.assertIn("Player 1: proposes keep=[3, 2, 1]", prompt)
        self.assertIn("utters [1, 2, 0]", prompt)

    def test_proposal_prompt_offers_accept_when_open(self):
        obs = _observation(_PROPOSAL_STATE)
        prompt = generate_prompt(obs, [])
        self.assertIn('"action": "accept"', prompt)

    def test_proposal_prompt_no_accept_when_no_open_offer(self):
        empty_state = {
            **_PROPOSAL_STATE,
            "proposals": [],
            "utterances": [],
            "most_recent_proposal": None,
            "most_recent_utterance": None,
            "current_player": 0,
        }
        obs = _observation(empty_state, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("you must propose", prompt)

    def test_utterance_prompt_used_for_utterance_turn(self):
        obs = _observation(_UTTERANCE_STATE)
        prompt = generate_prompt(obs, [])
        self.assertIn("utterance turn", prompt)
        self.assertIn('"symbols"', prompt)

    def test_rethink_suffix(self):
        obs = _observation(_PROPOSAL_STATE)
        prompt = generate_prompt(obs, [], previous_response="I tried [9,9,9]", previous_action="[9,9,9]")
        self.assertIn("Your previous response was", prompt)
        self.assertIn("[9,9,9]", prompt)

    def test_no_rethink_on_first_attempt(self):
        prompt = generate_prompt(_observation(_PROPOSAL_STATE), [])
        self.assertNotIn("Your previous response was", prompt)


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        observation = {
            "legalActions": [0, 1, 216],
            "legalActionStrings": ["Proposal: [0, 0, 0]", "Proposal: [0, 0, 1]", "Proposal: Agreement reached!"],
        }
        result = get_legal_moves(observation)
        self.assertEqual(
            result,
            {
                0: "Proposal: [0, 0, 0]",
                1: "Proposal: [0, 0, 1]",
                216: "Proposal: Agreement reached!",
            },
        )

    def test_fallback_proposal_enumeration(self):
        # No legalActions provided; harness enumerates from the proxy JSON.
        moves = get_legal_moves(_observation(_PROPOSAL_STATE, with_legal=False))
        # Pool is [5,5,5], so there are 6*6*6 = 216 proposal splits + accept.
        self.assertEqual(len(moves), 6 * 6 * 6 + 1)
        self.assertIn(216, moves)
        self.assertEqual(moves[216], "Proposal: Agreement reached!")
        self.assertEqual(moves[0], "Proposal: [0, 0, 0]")

    def test_fallback_utterance_enumeration(self):
        moves = get_legal_moves(_observation(_UTTERANCE_STATE, with_legal=False))
        # 5^3 = 125 utterances.
        self.assertEqual(len(moves), 125)
        self.assertEqual(moves[217], ", Utterance: [0, 0, 0]")


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


class _Harness:
    def get_legal_moves(self, observation):
        return get_legal_moves(observation)

    def make_prompt(self, observation, move_history, previous_response=None, previous_action=None):
        return generate_prompt(observation, move_history, previous_response, previous_action)

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
    def test_successful_propose(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response(
            '```json\n{"action": "propose", "keep": [0, 1, 1]}\n```'
        )
        agent = create_agent_fn(_Harness())
        obs = _observation(_PROPOSAL_STATE)
        result = agent(obs, {})
        # Proposal [0, 1, 1] encodes as 0*36 + 1*6 + 1 = 7
        self.assertEqual(result["submission"], 7)
        self.assertEqual(result["status"], "OK")

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_accept(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response('```json\n{"action": "accept"}\n```')
        agent = create_agent_fn(_Harness())
        obs = _observation(_PROPOSAL_STATE)
        result = agent(obs, {})
        self.assertEqual(result["submission"], 216)
        self.assertEqual(result["status"], "OK")

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_utterance(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response('```json\n{"symbols": [0, 0, 1]}\n```')
        agent = create_agent_fn(_Harness())
        obs = _observation(_UTTERANCE_STATE)
        result = agent(obs, {})
        # Utterance [0, 0, 1] -> offset 217 + (0*25 + 0*5 + 1) = 218
        self.assertEqual(result["submission"], 218)
        self.assertEqual(result["status"], "OK")

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.side_effect = [
            _make_mock_response("not parseable at all"),
            _make_mock_response('```json\n{"action": "accept"}\n```'),
        ]
        agent = create_agent_fn(_Harness())
        obs = _observation(_PROPOSAL_STATE)
        result = agent(obs, {})
        self.assertEqual(result["submission"], 216)
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_repeated_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("no chance")
        agent = create_agent_fn(_Harness())
        with self.assertRaises(ValueError):
            agent(_observation(_PROPOSAL_STATE), {})
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_terminal_returns_inactive(self, mock_litellm):
        mock_litellm.drop_params = True
        agent = create_agent_fn(_Harness())
        result = agent({"step": 0, "remainingOverageTime": 60}, {})
        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()


if __name__ == "__main__":
    absltest.main()
