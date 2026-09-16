"""Parity checks for shared content across hanabi and hanabi_arena.

Both harnesses ship as separately-versioned, manually-deployed files
(see memory: harness deploy isolation), so they cannot import shared
fragments from a sibling module without bypassing the deploy gate. That
leaves us with duplicated content -- the rules block, the action
notation, the rethink templates, and the whole ~200-line loose-notation
matcher. This test guards against drift by importing both harness
modules in the test process (which DOES run from repo HEAD) and
asserting the parts that should agree actually do.

The matcher parity check matters most: the arena and base harnesses feed
the same Elo leaderboard, so if one accepts "hint player +1 red" and the
other makes the model spend a retry on it, the two environments are
measuring subtly different things.

When a string intentionally diverges between the variants (e.g. base
says "Every hint is public" and the arena says "public at your table"),
update both sides AND the comparison predicate here so future drift on
the other axes still gets caught.
"""

import pyspiel
from absl.testing import absltest

from kaggle_environments.envs.open_spiel_env.games.hanabi import (
    hanabi_proxy,  # noqa: F401  (registers "hanabi_proxy")
)
from kaggle_environments.envs.open_spiel_env.games.hanabi import (
    harness as base,
)
from kaggle_environments.envs.open_spiel_env.games.hanabi_arena import (
    hanabi_arena_game,  # noqa: F401  (registers "hanabi_arena")
)
from kaggle_environments.envs.open_spiel_env.games.hanabi_arena import (
    harness as arena,
)

_BASE_PARAMS = {
    "players": 2,
    "colors": 5,
    "ranks": 5,
    "hand_size": 5,
    "max_life_tokens": 3,
    "max_information_tokens": 8,
}


def _base_prompt() -> str:
    game = pyspiel.load_game("hanabi_proxy", _BASE_PARAMS)
    state = game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    return base.generate_prompt(
        {
            "observationString": state.observation_string(0),
            "playerId": 0,
            "serializedGameAndState": pyspiel.serialize_game_and_state(game, state),
        },
        [],
    )


def _arena_prompt() -> str:
    game = pyspiel.load_game("hanabi_arena", {"seed": 7})
    state = game.new_initial_state()
    return arena.generate_prompt(
        {
            "observationString": state.observation_string(0),
            "playerId": 0,
            "serializedGameAndState": pyspiel.serialize_game_and_state(game, state),
        },
        [],
    )


class RethinkTemplateParityTest(absltest.TestCase):
    """Rethink templates should be word-for-word identical."""

    def test_rethink_illegal_is_byte_identical(self):
        self.assertEqual(base.RETHINK_ILLEGAL, arena.RETHINK_ILLEGAL)

    def test_rethink_unparsable_is_byte_identical(self):
        self.assertEqual(base.RETHINK_UNPARSABLE, arena.RETHINK_UNPARSABLE)


class PromptFragmentParityTest(absltest.TestCase):
    """Recurring prompt fragments that should match across both variants.

    Checked against the rendered prompt rather than the template
    constants, because the arena interpolates different values into the
    same sentences -- the shared *rendered* text is what the model sees.
    """

    def setUp(self):
        super().setUp()
        self.base = _base_prompt()
        self.arena = _arena_prompt()

    def _assert_in_both(self, fragment: str):
        self.assertIn(fragment, self.base)
        self.assertIn(fragment, self.arena)

    def test_deck_composition_phrased_identically(self):
        self._assert_in_both(
            "- The deck has 5 colors (R/Y/G/W/B) and ranks 1-5, 50 cards in total: "
            "per color, three 1s, two each of 2-4, and one 5."
        )

    def test_firework_and_scoring_rule_phrased_identically(self):
        self._assert_in_both(
            "- The team builds one firework stack per color in ascending order 1 to 5. "
            "The score is the sum of the stack heights, at most 25."
        )

    def test_play_and_discard_rules_phrased_identically(self):
        self._assert_in_both(
            "  * Play a card from one of your slots. If its rank is exactly one above "
            "its color's current stack height it joins that stack; otherwise it is "
            "discarded and the team loses a life token."
        )
        self._assert_in_both(
            "  * Discard a card from one of your slots. It is lost and the team regains "
            "one info token. Illegal while info tokens are at the maximum of 8."
        )

    def test_slot_shift_rule_phrased_identically(self):
        # The slot-numbering convention is what every "(Play N)" answer
        # depends on; a divergence here would mean the two variants are
        # teaching different action semantics.
        self._assert_in_both(
            "- After you play or discard, your remaining cards shift down one slot and, "
            "if the deck is not empty, a replacement is drawn into your highest slot. "
            "Slot 0 is therefore always your oldest card."
        )

    def test_stack_completion_rule_phrased_identically(self):
        self._assert_in_both(
            "- Completing a color's stack at rank 5 regains one info token. Info tokens never exceed 8."
        )

    def test_hint_rule_shares_its_mechanics(self):
        # Wording diverges by one phrase (base hints "another player",
        # the arena "another player at your table"), so check the shared
        # mechanics rather than the whole sentence.
        for prompt in (self.base, self.arena):
            flat = " ".join(prompt.split())
            self.assertIn("one color or one rank", flat)
            self.assertIn("the rest are thereby excluded from it", flat)
            self.assertIn("illegal at 0 info tokens", flat)
            self.assertIn("must match at least one card in that player's hand", flat)

    def test_endgame_rule_shares_its_mechanics(self):
        # Base says "The game ends", the arena "Your table ends" -- the
        # three terminal conditions must still be stated the same way.
        for prompt in (self.base, self.arena):
            flat = " ".join(prompt.split())
            self.assertIn("loses its last life token (the score becomes 0", flat)
            self.assertIn("when every stack is complete (25 points)", flat)
            self.assertIn("after the deck empties each player takes exactly one more turn", flat)

    def test_hint_publicity_stated_in_both(self):
        # Base: "public: all players hear". Arena: "public at your
        # table: all your team hears". Both must say hints are public.
        for prompt in (self.base, self.arena):
            self.assertIn("- Every hint is public", prompt)

    def test_token_and_deck_lines_phrased_identically(self):
        self._assert_in_both("Tokens: 3/3 life, 8/8 info")
        self._assert_in_both("Deck: 40 card(s) left")
        self._assert_in_both("Discarded: (none)")

    def test_own_hand_knowledge_rendered_identically(self):
        self._assert_in_both("told nothing; possible colors RYGWB, ranks 12345")
        for prompt in (self.base, self.arena):
            flat = " ".join(prompt.split())
            self.assertIn('"told" is what hints have said about the card out loud', flat)
            self.assertIn('"possible" is what is still consistent with every hint', flat)

    def test_action_notation_block_phrased_identically(self):
        self._assert_in_both(
            "Action notation -- your answer must be exactly one of these forms:\n"
            "  (Play N)                     play the card in your slot N\n"
            "  (Discard N)                  discard the card in your slot N\n"
            "  (Reveal player +K color C)   hint color C to the player K seats after you\n"
            "  (Reveal player +K rank R)    hint rank R to the player K seats after you\n"
        )

    def test_offset_convention_phrased_identically(self):
        self._assert_in_both(
            "Slots are numbered from 0. Colors are the single letters R/Y/G/W/B. "
            "Hint targets are written as an offset from your own seat:"
        )

    def test_output_format_spec_phrased_identically(self):
        self._assert_in_both('```json\n{\n  "move": "<action>"\n}\n```\n\nFor example: `{"move": "(Play 0)"}`')

    def test_reasoning_requested_before_json_in_both(self):
        # Non-negotiable across every prompt variant: the model must be
        # told to reason BEFORE it emits the JSON answer.
        for prompt in (self.base, self.arena):
            self.assertIn("Reason step by step", prompt)
            self.assertLess(prompt.index("Reason step by step"), prompt.index('"move"'))

    def test_failure_warning_phrased_identically(self):
        self._assert_in_both(
            "Failure to output your final answer in the specified format, or choosing "
            "an illegal action, will result in a loss."
        )


class ParseResponseParityTest(absltest.TestCase):
    """The loose-notation matcher is duplicated line-for-line; it must behave so.

    Every case below is run through BOTH parsers with the same legal set
    and the results compared. A divergence means one variant's models
    burn a retry where the other's do not -- which shows up as an Elo
    difference that has nothing to do with play strength.
    """

    legal = [
        "(Play 0)",
        "(Play 1)",
        "(Discard 2)",
        "(Reveal player +1 color R)",
        "(Reveal player +1 rank 3)",
    ]

    accepted = [
        "(Play 0)",
        "Play 1",
        "discard 2",
        "hint player +1 color R",
        "tell player +1 color red",
        "clue player +1 rank 3",
        "hint player +1 3",
        "Reveal player +1 colour R",
        "Reveal player +1 rank three",
        "Play 1 (likely W1)",
        "Play 0 (safest play)",
        "Play 0 (2/3 chance it is W1)",
        "Discard 2 (dead card, R/Y are both finished)",
        "Hint Player 1 about red",
        "Clue rank 3 to player +1",
        "Reveal color R to player +1",
        # Decided, just decorated. Both must take these without a rethink.
        "Play 0!",
        "**Play 0**",
        "`Play 0`",
        "**(Play 0)**",
        "Play 0 ✓",
        "Discard 2 - slot-2 is dead",
    ]

    refused = [
        "(Play 4)",
        "Discard 0",
        "Play R1",
        "Play the G2",
        "Play 0 or Play 1",
        "Play 1 then discard 2",
        "Play 0, 1",
        "Play 0 -- Or maybe Discard 1",
        "Hint player +3 rank 3",
        # A negative slot index; normalizing the sign away would silently
        # submit slot 1, the opposite end of the hand from what was meant.
        "Play -1",
    ]

    def test_accepted_notations_resolve_identically(self):
        for move in self.accepted:
            response = f'```json\n{{"move": "{move}"}}\n```'
            b = base.parse_response(response, self.legal)
            a = arena.parse_response(response, self.legal)
            self.assertEqual(b.legal_action, a.legal_action, move)
            self.assertIsNotNone(a.legal_action, move)

    def test_refused_notations_are_refused_identically(self):
        for move in self.refused:
            response = f'```json\n{{"move": "{move}"}}\n```'
            b = base.parse_response(response, self.legal)
            a = arena.parse_response(response, self.legal)
            self.assertEqual(b.legal_action, a.legal_action, move)
            self.assertIsNone(a.legal_action, move)
            self.assertEqual(b.raw_action, a.raw_action, move)

    def test_prose_only_response_refused_by_both(self):
        # Neither variant may ghost-substitute a move from the prose.
        response = "I will (Play 0) this turn."
        for parser in (base.parse_response, arena.parse_response):
            result = parser(response, self.legal)
            self.assertIsNone(result.legal_action)
            self.assertIsNone(result.raw_action)

    def test_last_json_block_wins_in_both(self):
        response = '```json\n{"move": "(Play 0)"}\n```\nOn reflection:\n```json\n{"move": "(Play 1)"}\n```'
        for parser in (base.parse_response, arena.parse_response):
            self.assertEqual(parser(response, self.legal).legal_action, "(Play 1)")


if __name__ == "__main__":
    absltest.main()
