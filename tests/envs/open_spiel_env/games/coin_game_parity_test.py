"""Parity checks for shared content across coin_game and coin_game_arena.

Both harnesses ship as separately-versioned, manually-deployed files
(see memory: harness deploy isolation), so they cannot import shared
fragments from a sibling module without bypassing the deploy gate.
That leaves us with duplicated content -- this test guards against
drift by importing both harness modules in the test process (which DOES
run from repo HEAD) and asserting the fragments that should be
byte-identical actually are.

When a string intentionally diverges between the variants (e.g. the
no-op rule mentions "another player's cell" in base and "your
teammate's cell" in arena), update both sides AND the comparison
predicate here so future drift on the other axes still gets caught.
"""

from absl.testing import absltest

from kaggle_environments.envs.open_spiel_env.games.coin_game import (
    harness as base,
)
from kaggle_environments.envs.open_spiel_env.games.coin_game_arena import (
    harness as arena,
)


class RethinkTemplateParityTest(absltest.TestCase):
    """Rethink templates should be word-for-word identical."""

    def test_rethink_illegal_is_byte_identical(self):
        self.assertEqual(base.RETHINK_ILLEGAL, arena.RETHINK_ILLEGAL)

    def test_rethink_unparsable_is_byte_identical(self):
        self.assertEqual(base.RETHINK_UNPARSABLE, arena.RETHINK_UNPARSABLE)


class PromptFragmentParityTest(absltest.TestCase):
    """Recurring prompt fragments that should match across both variants.

    Each check works against the rendered prompt rather than internal
    constants, because the two harnesses construct prompts differently
    (base uses one template, arena another) -- the shared *meaning* is
    what matters.
    """

    def _base_prompt(self):
        import json
        # Minimal viable obs for the base harness.
        state = json.dumps({
            "phase": "play",
            "num_rows": 8,
            "num_columns": 8,
            "episode_length": 20,
            "your_preference": "a",
            "your_player_id": 0,
            "board": [["."] * 8 for _ in range(8)],
            "player_positions": {"0": [0, 0], "1": [7, 7]},
            "coins_collected": {"0": {}, "1": {}},
            "coin_colors": ["a", "b", "c"],
            "move_history": [],
            "move_number": 0,
            "moves_remaining": 20,
        })
        return base.generate_prompt(
            {"observationString": state, "playerId": 0}, [],
        )

    def _arena_prompt(self):
        import pyspiel
        g = pyspiel.load_game("coin_game_arena", {"seed": 7})
        s = g.new_initial_state()
        obs = {
            "observationString": s.observation_string(0),
            "playerId": 0,
            "legalActions": s.legal_actions(0),
            "legalActionStrings": [s.action_to_string(0, a) for a in s.legal_actions(0)],
        }
        return arena.generate_prompt(obs, [])

    def test_action_set_phrased_identically(self):
        token = "up, down, left, right, stand"
        self.assertIn(token, self._base_prompt())
        self.assertIn(token, self._arena_prompt())

    def test_reward_formula_phrased_identically(self):
        formula = "self_pref^2 + other_pref^2 - bad_coins^2"
        self.assertIn(formula, self._base_prompt())
        self.assertIn(formula, self._arena_prompt())

    def test_coord_convention_phrased_identically(self):
        token = "``[row, column]`` with ``row=0`` at the top"
        self.assertIn(token, self._base_prompt())
        self.assertIn(token, self._arena_prompt())

    def test_output_format_spec_phrased_identically(self):
        # The fenced JSON spec is paste-shared.
        spec = '{\n  "move": "<move>"\n}'
        self.assertIn(spec, self._base_prompt())
        self.assertIn(spec, self._arena_prompt())

    def test_no_op_rule_present_in_both(self):
        # Wording diverges by one phrase ("another player" vs
        # "teammate"), so we check the shared shell.
        for prompt in (self._base_prompt(), self._arena_prompt()):
            flat = " ".join(prompt.split())
            self.assertIn("leave the board", flat)
            self.assertIn("no-ops", flat)
            self.assertIn("stay in place", flat)


if __name__ == "__main__":
    absltest.main()
