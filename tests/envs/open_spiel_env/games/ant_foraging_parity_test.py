"""Parity checks for shared content across python_ant_foraging and ant_foraging_arena.

Both harnesses ship as separately-versioned, manually-deployed files
(see memory: harness deploy isolation), so they cannot import shared
fragments from a sibling module without bypassing the deploy gate.
That leaves us with duplicated content -- this test guards against
drift by importing both harness modules in the test process (which DOES
run from repo HEAD) and asserting the fragments that should be
byte-identical actually are.

When a string intentionally diverges between the variants (e.g. the
arena prompt mentions teammates and team boards, while the base prompt
talks about the single shared board), update both sides AND the
comparison predicate here so future drift on the other axes still gets
caught.
"""

import json

import pyspiel
from absl.testing import absltest

from kaggle_environments.envs.open_spiel_env.games.ant_foraging_arena import (
    ant_foraging_arena_game,  # noqa: F401  (registers the game)
    harness as arena,
)
from kaggle_environments.envs.open_spiel_env.games.python_ant_foraging import (
    harness as base,
    python_ant_foraging_proxy,
)


class RethinkTemplateParityTest(absltest.TestCase):
    """Rethink templates should be word-for-word identical."""

    def test_rethink_illegal_is_byte_identical(self):
        self.assertEqual(base.RETHINK_ILLEGAL, arena.RETHINK_ILLEGAL)

    def test_rethink_unparsable_is_byte_identical(self):
        self.assertEqual(base.RETHINK_UNPARSABLE, arena.RETHINK_UNPARSABLE)


class HelperConstantParityTest(absltest.TestCase):
    """Tuning constants that should track each other."""

    def test_pheromone_threshold_matches(self):
        # Both harnesses sparsify pheromones with the same cutoff so that
        # what one variant considers "visible" the other does too.
        self.assertEqual(base._PHEROMONE_THRESHOLD, arena._PHEROMONE_THRESHOLD)

    def test_move_history_tail_matches(self):
        # Both harnesses display the same number of recent moves.
        self.assertEqual(base._MOVE_HISTORY_TAIL, arena._MOVE_HISTORY_TAIL)


class PromptFragmentParityTest(absltest.TestCase):
    """Recurring prompt fragments that should match across both variants.

    Each check works against the rendered prompt rather than internal
    constants, because the two harnesses construct prompts differently
    (base uses one template, arena another) -- the shared *meaning* is
    what matters.
    """

    def _base_prompt(self):
        game = python_ant_foraging_proxy.PythonAntForagingGame()
        state = game.new_initial_state()
        obs = {
            "observationString": state.observation_string(0),
            "playerId": 0,
        }
        return base.generate_prompt(obs, [])

    def _arena_prompt(self):
        game = pyspiel.load_game("ant_foraging_arena", {"seed": 7})
        state = game.new_initial_state()
        obs = {
            "observationString": state.observation_string(0),
            "playerId": 0,
            "legalActions": list(state.legal_actions(0)),
            "legalActionStrings": [state.action_to_string(0, a) for a in state.legal_actions(0)],
        }
        return arena.generate_prompt(obs, [])

    def test_coord_convention_phrased_identically(self):
        # Both prompts must teach coordinates the same way; the arena's
        # docstring promises "same coordinate system as base".
        token = (
            "positions are ``[row, column]`` with ``row=0`` at the top\n"
            "and ``column=0`` on the left. ``up`` decreases row, ``down`` increases\n"
            "row, ``left`` decreases column, ``right`` increases column."
        )
        self.assertIn(token, self._base_prompt())
        self.assertIn(token, self._arena_prompt())

    def test_world_legend_present_in_both(self):
        # The three terrain glyphs are part of the world's definition;
        # both variants must explain them.
        for prompt in (self._base_prompt(), self._arena_prompt()):
            flat = " ".join(prompt.split())
            self.assertIn("``N``", flat)
            self.assertIn("``F`` marks each remaining food source", flat)
            self.assertIn("``.``", flat)

    def test_pickup_and_delivery_rules_present_in_both(self):
        # Both prompts must spell out the pickup-on-F and
        # deliver-on-N-while-carrying mechanics: the model can't infer
        # them from the board view.
        for prompt in (self._base_prompt(), self._arena_prompt()):
            flat = " ".join(prompt.split())
            self.assertIn("Stepping onto an ``F`` cell automatically picks up that food", flat)
            self.assertIn("one food at a time per", flat)
            self.assertIn("Returning to ``N`` while carrying drops it off", flat)

    def test_off_board_illegality_phrased_in_both(self):
        # The original arena prompt said off-board moves were "silently
        # blocked" (wrong: they're illegal). Both prompts must say
        # off-board is illegal so neither variant regresses.
        for prompt in (self._base_prompt(), self._arena_prompt()):
            flat = " ".join(prompt.split())
            self.assertIn("off-board moves are not legal", flat.lower())

    def test_pheromone_decay_phrased_identically(self):
        # The "fresh > faint" guidance is the only directly actionable
        # claim about pheromones in either prompt -- keep it identical.
        # Compare against a whitespace-flattened prompt so line wraps
        # in either template don't mask drift in the underlying words.
        token = "fresh trails are more reliable than faint ones."
        for prompt in (self._base_prompt(), self._arena_prompt()):
            flat = " ".join(prompt.split())
            self.assertIn(token, flat)

    def test_pheromone_channels_named_identically(self):
        # The two trail-type names are referenced both in the prose and
        # in the sparse-view labels; drift would mismatch the rules
        # against the data display.
        for prompt in (self._base_prompt(), self._arena_prompt()):
            self.assertIn("``to_food``", prompt)
            self.assertIn("``to_nest``", prompt)
            self.assertIn("to_food:", prompt)  # sparse-view label
            self.assertIn("to_nest:", prompt)

    def test_action_set_phrased_identically(self):
        # The literal action-set token appears in the rules text in both
        # variants; keep the spelling/order in sync.
        token = "{stay, up, down, left, right}"
        self.assertIn(token, self._base_prompt())
        self.assertIn(token, self._arena_prompt())

    def test_output_format_spec_phrased_identically(self):
        # The fenced JSON spec is the contract with the model -- a
        # mismatched spec between variants would mean models trained on
        # one tier output the wrong shape for the other.
        spec = '{\n  "move": "<'
        self.assertIn(spec, self._base_prompt())
        self.assertIn(spec, self._arena_prompt())

    def test_ascii_grid_header_present_in_both(self):
        # Same renderer shape (zero-indexed column header) in both
        # variants so models see the grid the same way.
        header_8x8 = "    0 1 2 3 4 5 6 7"
        self.assertIn(header_8x8, self._base_prompt())
        self.assertIn(header_8x8, self._arena_prompt())

    def test_carrying_glyph_legend_present_in_both(self):
        # Both render carrying ants as a capital letter overlay; both
        # prompts must say so.
        for prompt in (self._base_prompt(), self._arena_prompt()):
            flat = " ".join(prompt.split())
            self.assertIn("capital letter", flat)
            self.assertIn("carrying food", flat)


if __name__ == "__main__":
    absltest.main()
