"""Tests for the Gin Rummy LLM harness."""

import random

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult
from kaggle_environments.envs.open_spiel_env.games.gin_rummy import (
    gin_rummy_proxy,
)
from kaggle_environments.envs.open_spiel_env.games.gin_rummy.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)

# OpenSpiel action constants (mirrors gin_rummy.h).
_DRAW_UPCARD = 52
_DRAW_STOCK = 53
_PASS = 54
_KNOCK = 55


def _make_observation(state, game, player_id: int) -> dict:
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "currentPlayer": state.current_player(),
        "isTerminal": state.is_terminal(),
        "legalActions": list(state.legal_actions()),
        "legalActionStrings": [
            state.action_to_string(a) for a in state.legal_actions()
        ],
        "serializedGameAndState": pyspiel.serialize_game_and_state(
            game.__wrapped__, state.__wrapped__
        ),
    }


def _deal_state(seed: int = 0) -> tuple[gin_rummy_proxy.GinRummyState, gin_rummy_proxy.GinRummyGame]:
    """Play through chance nodes to the first non-chance state (FirstUpcard)."""
    rng = random.Random(seed)
    g = gin_rummy_proxy.GinRummyGame()
    s = g.new_initial_state()
    while s.is_chance_node():
        outcomes = s.chance_outcomes()
        s.apply_action(rng.choices([a for a, _ in outcomes], [p for _, p in outcomes])[0])
    return s, g


def _play_to_layoff(seed: int) -> tuple[gin_rummy_proxy.GinRummyState, gin_rummy_proxy.GinRummyGame]:
    """Drive a random game forward until the knocker has laid all melds and
    the non-knocker is in the Layoff phase. Returns ``(state, game)`` or
    raises if no knock occurs before max_moves.
    """
    rng = random.Random(seed)
    g = gin_rummy_proxy.GinRummyGame()
    s = g.new_initial_state()
    for _ in range(400):
        if s.is_terminal():
            break
        if s.is_chance_node():
            outcomes = s.chance_outcomes()
            s.apply_action(rng.choices([a for a, _ in outcomes], [p for _, p in outcomes])[0])
            continue
        legal = s.legal_actions()
        if _KNOCK in legal:
            knocker = s.current_player()
            s.apply_action(_KNOCK)
            # Walk the knocker through any required discard + meld declarations
            # by always picking the last (longest meld id) until they pass.
            while not s.is_terminal() and s.current_player() == knocker:
                next_legal = s.legal_actions()
                s.apply_action(next_legal[-1])
            return s, g
        s.apply_action(rng.choice(legal))
    raise RuntimeError("no knock reached")


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class ParseResponseTest(absltest.TestCase):
    """Discard-phase legals: a few cards plus 'Knock'."""

    legal = ["Knock", "7h", "As", "2c", "Td", "Jh", "9s", "Qd", "3c", "5s", "8h"]

    def test_parse_json_block(self):
        result = parse_response('```json\n{"move": "7h"}\n```', self.legal)
        self.assertEqual(result.legal_action, "7h")
        self.assertEqual(result.raw_action, "7h")

    def test_parse_bare_json(self):
        result = parse_response('I think {"move": "Knock"} is right.', self.legal)
        self.assertEqual(result.legal_action, "Knock")

    def test_parse_strips_action_prefix(self):
        result = parse_response('```json\n{"move":"Player: 0 Action: 7h"}\n```', self.legal)
        self.assertEqual(result.legal_action, "7h")

    def test_parse_case_insensitive(self):
        result = parse_response('```json\n{"move":"7H"}\n```', self.legal)
        self.assertEqual(result.legal_action, "7h")

    def test_parse_rethink_takes_last_json(self):
        # Model wrote one move, reconsidered, wrote another. Pick the last.
        response = (
            '```json\n{"move":"Knock"}\n```\n'
            'Wait, on reflection:\n'
            '```json\n{"move":"7h"}\n```'
        )
        self.assertEqual(parse_response(response, self.legal).legal_action, "7h")

    def test_parse_no_json_returns_none(self):
        # The harness deliberately does NOT scan prose for legal tokens --
        # JSON only, let the rethink loop handle anything else (the rethink
        # suffix prints the previous response back to the model).
        response = "I considered 7h then As, but I will discard 8h."
        self.assertIsNone(parse_response(response, self.legal).legal_action)

    def test_parse_illegal_returns_raw(self):
        result = parse_response('```json\n{"move":"Kc"}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "Kc")

    def test_parse_refusal_returns_none(self):
        result = parse_response("I cannot decide.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_returns_parse_result_type(self):
        result = parse_response('```json\n{"move":"Knock"}\n```', self.legal)
        self.assertIsInstance(result, ParseResult)


class ParseMeldTest(absltest.TestCase):
    """Knock/Layoff legals contain concatenated meld strings like 'As2s3s'."""

    legal = ["Pass", "As2s3s", "AsAcAd", "AsAcAdAh"]

    def test_meld_with_commas_in_json(self):
        # Models often write '"As, 2s, 3s"' instead of the contiguous form.
        result = parse_response('```json\n{"move":"As, 2s, 3s"}\n```', self.legal)
        self.assertEqual(result.legal_action, "As2s3s")

    def test_meld_with_spaces_in_json(self):
        result = parse_response('```json\n{"move":"As 2s 3s"}\n```', self.legal)
        self.assertEqual(result.legal_action, "As2s3s")

    def test_rethink_with_two_json_blocks_takes_last(self):
        # Model writes one meld, reconsiders, writes another. The shared
        # extract_last_json_object helper takes the last block.
        response = (
            '```json\n{"move":"AsAcAdAh"}\n```\n'
            'actually no:\n'
            '```json\n{"move":"AsAcAd"}\n```'
        )
        self.assertEqual(parse_response(response, self.legal).legal_action, "AsAcAd")


# ---------------------------------------------------------------------------
# get_legal_moves
# ---------------------------------------------------------------------------


class GetLegalMovesTest(absltest.TestCase):
    def test_strips_player_action_prefix(self):
        obs = {
            "legalActions": [52, 54],
            "legalActionStrings": ["Player: 0 Action: Draw upcard", "Player: 0 Action: Pass"],
        }
        result = get_legal_moves(obs)
        self.assertEqual(result, {52: "Draw upcard", 54: "Pass"})

    def test_from_serialized_state(self):
        s, g = _deal_state(seed=0)
        obs = {"serializedGameAndState": pyspiel.serialize_game_and_state(g.__wrapped__, s.__wrapped__)}
        result = get_legal_moves(obs)
        # FirstUpcard: legal are {Draw upcard, Pass}.
        self.assertIn(_DRAW_UPCARD, result)
        self.assertIn(_PASS, result)
        self.assertEqual(result[_DRAW_UPCARD], "Draw upcard")
        self.assertEqual(result[_PASS], "Pass")

    def test_empty_obs(self):
        self.assertEqual(get_legal_moves({}), {})


# ---------------------------------------------------------------------------
# generate_prompt
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def test_first_upcard_prompt_basics(self):
        s, g = _deal_state(seed=0)
        obs = _make_observation(s, g, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Gin Rummy", prompt)
        self.assertIn("Player 0", prompt)
        self.assertIn("Phase: FirstUpcard", prompt)
        # Default knock card.
        self.assertIn("knock card = 10", prompt)
        self.assertIn("Knock card: 10", prompt)
        # Bonus values must be shown so the model can value gin/undercut.
        self.assertIn("+25", prompt)

    def test_prompt_does_not_claim_oklahoma_when_default(self):
        # The env loads default (non-Oklahoma) rules. The old prompt asserted
        # the knock card was set from the upcard's rank value -- that's wrong
        # for the default ruleset and confused models.
        s, g = _deal_state(seed=0)
        prompt = generate_prompt(_make_observation(s, g, player_id=0), [])
        self.assertNotIn("Oklahoma", prompt)
        self.assertIn("default rules", prompt)

    def test_prompt_describes_oklahoma_when_enabled(self):
        # When the env is configured with oklahoma=True the prompt should
        # explain the upcard-rank knock card rule instead.
        rng = random.Random(1)
        g = gin_rummy_proxy.GinRummyGame({"oklahoma": True})
        s = g.new_initial_state()
        while s.is_chance_node():
            outcomes = s.chance_outcomes()
            s.apply_action(rng.choices([a for a, _ in outcomes], [p for _, p in outcomes])[0])
        prompt = generate_prompt(_make_observation(s, g, player_id=0), [])
        self.assertIn("Oklahoma variant", prompt)
        self.assertIn("upcard", prompt.lower())
        self.assertIn("rank value", prompt)
        # And must NOT claim a fixed knock card of 10 as the rule.
        self.assertNotIn("default rules", prompt)

    def test_prompt_labels_history_as_yours(self):
        # core_harness only feeds the agent its own past actions. The label
        # must make that clear.
        s, g = _deal_state(seed=0)
        prompt = generate_prompt(
            _make_observation(s, g, player_id=0), ["Draw upcard", "7h"]
        )
        self.assertIn("Your previous actions", prompt)
        self.assertIn("Draw upcard, 7h", prompt)

    def test_prompt_first_upcard_hides_opponent(self):
        s, g = _deal_state(seed=0)
        prompt = generate_prompt(_make_observation(s, g, player_id=0), [])
        self.assertIn("Opponent hand: (hidden)", prompt)

    def test_layoff_prompt_shows_knocker_melds_and_hand(self):
        # Reproduces the previously-fatal Layoff bug: the layoff player
        # could not see the knocker's melds or hand.
        s, g = _play_to_layoff(seed=4)
        # Find the player about to lay off.
        cur = s.current_player()
        prompt = generate_prompt(_make_observation(s, g, cur), [])
        self.assertIn("Phase: Layoff", prompt)
        self.assertIn("Opponent has KNOCKED", prompt)
        # At least one laid meld must be shown (knocker laid melds in the
        # walk-through; the formatter joins cards with spaces).
        self.assertIn("Opponent laid melds:", prompt)
        self.assertNotIn("Opponent laid melds: (none yet)", prompt)
        # Opponent's remaining hand and deadwood must be visible.
        self.assertIn("Opponent remaining hand", prompt)
        self.assertIn("Opponent deadwood:", prompt)

    def test_wall_phase_has_correct_instruction(self):
        # Build a state in the Wall phase synthetically. Easiest path: drive
        # a game where players keep drawing/discarding until stock_size == 2
        # without knocking. With the random seed sweep below we just look for
        # a state with phase == 'Wall'.
        from kaggle_environments.envs.open_spiel_env.games.gin_rummy import harness
        self.assertIn("Wall", harness._PHASE_INSTRUCTION)
        wall_text = harness._instruction_for_phase("Wall", ["Pass", "Knock"])
        # The Wall instruction must NOT misroute to Layoff.
        self.assertNotIn("Your opponent knocked", wall_text)
        self.assertIn("wall", wall_text.lower())
        self.assertIn("Pass", wall_text)

    def test_rethink_suffix(self):
        s, g = _deal_state(seed=0)
        obs = _make_observation(s, g, player_id=0)
        prompt = generate_prompt(
            obs, [], previous_response="I'll play 99c", previous_action="99c"
        )
        self.assertIn("Your previous response was", prompt)
        self.assertIn("99c", prompt)


# ---------------------------------------------------------------------------
# Proxy regression: layoff exposes layed_melds + opponent state
# ---------------------------------------------------------------------------


class ProxyLayoffExposureTest(absltest.TestCase):
    def test_proxy_exposes_layed_melds_at_layoff(self):
        s, _ = _play_to_layoff(seed=4)
        cur = s.current_player()
        opp = 1 - cur
        d = s.state_dict(cur)
        self.assertIn("layed_melds", d)
        self.assertTrue(d["layed_melds"][str(opp)], "knocker should have laid melds")
        # Each meld is a list of card tokens.
        first_meld = d["layed_melds"][str(opp)][0]
        self.assertIsInstance(first_meld, list)
        self.assertTrue(all(isinstance(c, str) for c in first_meld))

    def test_proxy_reveals_opponent_hand_and_deadwood_after_knock(self):
        s, _ = _play_to_layoff(seed=4)
        cur = s.current_player()
        opp = 1 - cur
        d = s.state_dict(cur)
        # After knock the opponent's deadwood is public per gin rummy rules.
        self.assertIsNotNone(d["deadwood"][str(opp)])
        # Opponent's remaining hand is also public after knock.
        self.assertFalse(d["hand_hidden"][str(opp)])

    def test_proxy_hides_opponent_hand_before_knock(self):
        s, _ = _deal_state(seed=0)
        d = s.state_dict(0)
        # Pre-knock, opponent's hand should be hidden and deadwood None.
        self.assertEqual(d["hands"]["1"], [])
        self.assertIsNone(d["deadwood"]["1"])


if __name__ == "__main__":
    absltest.main()
