"""Tests for the Hanabi LLM harness."""

from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.hanabi import hanabi_proxy
from kaggle_environments.envs.open_spiel_env.games.hanabi.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)

# Hanabi's underlying game validates its parameters as a set, so the proxy has
# to be loaded with all of them rather than the players count alone.
_PARAMS = {
    "players": 2,
    "colors": 5,
    "ranks": 5,
    "hand_size": 5,
    "max_life_tokens": 3,
    "max_information_tokens": 8,
}


def _load_game(**overrides) -> hanabi_proxy.HanabiGame:
    return pyspiel.load_game("hanabi_proxy", {**_PARAMS, **overrides})


def _deal(game: hanabi_proxy.HanabiGame) -> hanabi_proxy.HanabiState:
    """Play out the opening deal deterministically, stopping at the first turn."""
    state = game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    return state


def _advance(state: hanabi_proxy.HanabiState) -> None:
    """Resolve any chance nodes (the replacement draw) after an action."""
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])


def _make_observation(
    state: hanabi_proxy.HanabiState,
    game: hanabi_proxy.HanabiGame,
    player_id: int = 0,
    *,
    include_legal_actions: bool = False,
) -> dict:
    """Build a harness-style observation dict from a proxy state.

    ``includeLegalActions`` defaults to false in ``open_spiel_env``, so the
    default here matches production: no ``legalActions`` key.
    """
    observation = {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "currentPlayer": int(state.current_player()),
        "isTerminal": state.is_terminal(),
        "serializedGameAndState": pyspiel.serialize_game_and_state(game, state),
    }
    if include_legal_actions:
        legal = list(state.legal_actions())
        current = int(state.current_player())
        observation["legalActions"] = legal
        observation["legalActionStrings"] = [state.action_to_string(current, a) for a in legal]
    return observation


def _action_id(state: hanabi_proxy.HanabiState, label: str) -> int:
    player = int(state.current_player())
    for action in state.legal_actions():
        if state.action_to_string(player, action) == label:
            return action
    raise AssertionError(f"{label!r} is not legal here")


def _first_rank_hint(state: hanabi_proxy.HanabiState) -> tuple[int, str, int]:
    """First legal rank hint as ``(action_id, label, rank)``.

    Which ranks are hintable depends on what the deal put in the target's
    hand, so tests pick from the legal set rather than naming a rank.
    """
    player = int(state.current_player())
    for action in state.legal_actions():
        label = state.action_to_string(player, action)
        if "rank" in label:
            return action, label, int(label.rstrip(")").split()[-1])
    raise AssertionError("no rank hint is legal here")


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class ParseResponseTest(absltest.TestCase):
    legal = [
        "(Play 0)",
        "(Play 1)",
        "(Discard 2)",
        "(Reveal player +1 color R)",
        "(Reveal player +1 rank 3)",
    ]

    def test_parse_json_block(self):
        result = parse_response('```json\n{"move": "(Play 0)"}\n```', self.legal)
        self.assertEqual(result.legal_action, "(Play 0)")
        self.assertEqual(result.raw_action, "(Play 0)")

    def test_parse_bare_json(self):
        result = parse_response('I think {"move": "(Discard 2)"} is best.', self.legal)
        self.assertEqual(result.legal_action, "(Discard 2)")

    def test_parse_returns_parse_result_type(self):
        result = parse_response('```json\n{"move": "(Play 0)"}\n```', self.legal)
        self.assertIsInstance(result, ParseResult)

    def test_parse_without_parentheses(self):
        result = parse_response('```json\n{"move": "Play 1"}\n```', self.legal)
        self.assertEqual(result.legal_action, "(Play 1)")

    def test_parse_lowercase(self):
        result = parse_response('```json\n{"move": "discard 2"}\n```', self.legal)
        self.assertEqual(result.legal_action, "(Discard 2)")

    def test_parse_hint_synonym(self):
        # Models say "hint"/"tell"/"clue" far more naturally than "reveal".
        for verb in ("hint", "tell", "clue"):
            result = parse_response(f'```json\n{{"move": "{verb} player +1 color R"}}\n```', self.legal)
            self.assertEqual(result.legal_action, "(Reveal player +1 color R)", verb)

    def test_parse_color_word(self):
        result = parse_response('```json\n{"move": "Reveal player +1 color red"}\n```', self.legal)
        self.assertEqual(result.legal_action, "(Reveal player +1 color R)")

    def test_parse_hint_without_keyword(self):
        result = parse_response('```json\n{"move": "hint player +1 3"}\n```', self.legal)
        self.assertEqual(result.legal_action, "(Reveal player +1 rank 3)")

    def test_prose_only_response_triggers_rethink(self):
        # No structured JSON. The parser must NOT guess at intent from a
        # move-shaped token in the prose -- return None and let rethink
        # ask the model to use the required JSON format.
        result = parse_response("I will (Play 0) this turn.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_parse_illegal_move_returns_raw(self):
        result = parse_response('```json\n{"move": "(Play 4)"}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "(Play 4)")

    def test_parse_no_match_returns_none(self):
        result = parse_response("I have no idea.", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_illegal_json_does_not_ghost_substitute_from_prose(self):
        # The model's JSON answer isn't legal. The parser must NOT silently
        # substitute a legal token from the prose -- return None so the
        # rethink loop asks the model to fix its answer.
        response = 'I considered (Play 0) but ruled it out.\n```json\n{"move": "(Play 9)"}\n```'
        result = parse_response(response, self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "(Play 9)")

    def test_discard_not_confused_with_play(self):
        # Slot numbers are shared across verbs, so a verb mismatch must fail
        # rather than fall through to the same slot's other action.
        result = parse_response('```json\n{"move": "Discard 0"}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "Discard 0")

    def test_card_named_instead_of_slot_is_refused(self):
        # The prompt renders teammates' cards as "R1"/"B3" tokens, so naming
        # the card instead of the slot is the likeliest notation slip. The
        # card's RANK must never be read as a slot index: "Play R1" would
        # silently become slot 1, an action the model never chose. Rethink is
        # the only correct response.
        for move in ("Play R1", "Play the G2", "Play my B3", "play G1", "Discard R2"):
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertIsNone(result.legal_action, move)
            self.assertEqual(result.raw_action, move)

    def test_slot_annotated_with_its_card_is_accepted(self):
        # Naming the slot AND guessing its card is normal Hanabi shorthand.
        # The slot is unambiguous, so this must not cost a rethink.
        for move in ("Play slot 1 (likely W1)", "Play 1 (W1)", "(Play 1) - my W1", "Play 1 -- probably W1"):
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertEqual(result.legal_action, "(Play 1)", move)

    def test_trailing_prose_and_spelling_variants_accepted(self):
        cases = {
            "Discard 2 to regain an info token": "(Discard 2)",
            "Reveal player +1 rank 3 (slots 0 and 2)": "(Reveal player +1 rank 3)",
            "Hint player +1 colour R": "(Reveal player +1 color R)",
            "Reveal player +1 rank three": "(Reveal player +1 rank 3)",
        }
        for move, expected in cases.items():
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertEqual(result.legal_action, expected, move)

    def test_strategic_rationale_is_not_mistaken_for_a_second_action(self):
        # Hanabi's strategic vocabulary IS its verb list, so a guard that
        # rejects any trailing "play"/"hint"/"clue"/"discard" rejects most
        # rationales a model writes -- and each rejection burns one of the two
        # retries before the agent forfeits with submission=-1. A second
        # action needs an OPERAND, not just a verb.
        cases = {
            "Play 0 (safest play)": "(Play 0)",
            "Play 0 (better than a hint right now)": "(Play 0)",
            "Play 0 (no useful clue available)": "(Play 0)",
            "Play 0 (I would rather not discard)": "(Play 0)",
            "Discard 2 (to regain a token for a future hint)": "(Discard 2)",
            "Reveal player +1 rank 3 (sets up two plays)": "(Reveal player +1 rank 3)",
            "Reveal player +1 rank 3 (this clue is the best available)": "(Reveal player +1 rank 3)",
        }
        for move, expected in cases.items():
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertEqual(result.legal_action, expected, move)

    def test_probabilistic_rationale_is_accepted(self):
        # Hanabi reasoning is inherently probabilistic, so a model annotates
        # its move with odds. A bare "/" in the undecided check rejected every
        # fraction while letting "0.66" and "75%" through -- an arbitrary
        # split that cost a rethink for saying the same thing in the more
        # natural notation. "R/Y" is the same slash in a different disguise.
        cases = {
            "Play 0 (2/3 chance it is W1)": "(Play 0)",
            "Play 0 -- 60/40 it is W1": "(Play 0)",
            "Play 0 (~2/3 safe)": "(Play 0)",
            "Play 0 (probability 0.66)": "(Play 0)",
            "Play 0 (75% safe)": "(Play 0)",
            "Discard 2 (dead card, R/Y are both finished)": "(Discard 2)",
            "Play 0 (it is W1 or B1 -- both playable)": "(Play 0)",
        }
        for move, expected in cases.items():
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertEqual(result.legal_action, expected, move)

    def test_decorated_answer_costs_no_rethink(self):
        # A decided move dressed in punctuation. The undecided check used to
        # refuse any all-punctuation tail, so an exclamation mark or a pair of
        # markdown asterisks bought a retry for an answer that named exactly
        # one move -- pure measurement noise in a model-vs-model tournament.
        for move in (
            "Play 0!",
            "Play 0?",
            "Play 0 !!!",
            "**Play 0**",
            "*Play 0*",
            "`Play 0`",
            "**(Play 0)**",
            "Play 0 \u2713",
            "Play 0 \u2192",
            "Play 0\u2026",
        ):
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertEqual(result.legal_action, "(Play 0)", move)

    def test_negative_slot_is_refused(self):
        # _normalize drops the hyphen -- which is what makes "Discard slot-2"
        # work -- so "Play -1" would silently become slot 1. A model writing
        # -1 most plausibly means the LAST card, the opposite end of the hand,
        # so guessing submits a move it did not choose and no rethink fires.
        for move in ("Play -1", "Discard -0"):
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertIsNone(result.legal_action, move)
            self.assertEqual(result.raw_action, move)

    def test_a_negative_slot_behind_a_filler_word_is_refused(self):
        # The guard has to walk the same filler run the slot matcher does.
        # _RAW_SLOT_RE absorbs "slot"/"the"/"my" between verb and operand, so a
        # guard demanding the sign sit flush against the verb caught "Play -1"
        # and waved "Play slot -1" through to slot 1 -- the opposite card, by
        # the exact route the guard exists to close, with no rethink.
        for move in (
            "Play slot -1",
            "Play the -1",
            "Play card -1",
            "Play my -1",
            "Play number -1",
            "Play from -1",
            "Play in -1",
            "Discard slot -1",
            "Discard the -2",
            "Play slot - 1",
        ):
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertIsNone(result.legal_action, move)
            self.assertEqual(result.raw_action, move)

    def test_a_hyphen_that_is_not_a_sign_still_parses(self):
        # The negative-slot guard is anchored at the verb's operand, so a
        # hyphen anywhere else keeps its old meaning.
        cases = {
            "Discard slot-2": "(Discard 2)",
            "Play 0 - my best guess": "(Play 0)",
            "Play slot 1": "(Play 1)",
            "Discard the 2": "(Discard 2)",
            "Play 0 - 2 lives left": "(Play 0)",
        }
        for move, expected in cases.items():
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertEqual(result.legal_action, expected, move)

    def test_a_second_action_on_the_next_line_is_refused(self):
        # The annotation splitter spans newlines; the undecided check has to
        # agree with it or the same indecision gets two verdicts depending on
        # whether the model wrote a space or a line break. A numbered list of
        # candidate moves is the shape a model reaches for when it has not
        # committed, and it is multi-line by construction.
        for move in ("Play 0,\n1", "Play 0\n1", "Play 0 or\nDiscard 2", "Play 0\nor 1"):
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertIsNone(result.legal_action, move)
            self.assertEqual(result.raw_action, move)

    def test_a_multi_line_annotation_still_parses(self):
        # Making the undecided check newline-aware must not start refusing a
        # settled answer whose rationale happens to wrap.
        cases = {
            "Play 0\n(likely W1)": "(Play 0)",
            "Play 0 -- my reasoning:\nR is dead": "(Play 0)",
            "Play 0\n-- safest play": "(Play 0)",
        }
        for move, expected in cases.items():
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertEqual(result.legal_action, expected, move)

    def test_undecided_answer_is_refused(self):
        # Two actions named and none chosen. Picking either would submit a
        # move the model did not settle on.
        for move in (
            "Play 0 or Play 1",
            "Play 1 then discard 2",
            "Play 0 or hint player +1 rank 3",
            "Play 0, otherwise Reveal player +1 color R",
        ):
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertIsNone(result.legal_action, move)

    def test_second_slot_offered_as_a_bare_digit_is_refused(self):
        # "Play 0 (W1)" names the card the model thinks is in slot 0, but
        # "Play 0, 1" and "Play 0-1" read at least as naturally as two slots.
        # This is the one shape that fails CONFIDENTLY: the engine accepts the
        # first slot, no rethink fires, and the model never learns it was
        # misread. Refuse and let the rethink ask for one slot.
        for move in ("Play 0-1", "Play 0, 1", "Play 0; 1", "Play 0 -- 1", "Play 0 / 1", "Play 0 or 1"):
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertIsNone(result.legal_action, move)
            self.assertEqual(result.raw_action, move)

    def test_undecided_answer_is_refused_whatever_the_capitalization(self):
        # The undecided/second-verb check runs on the RAW trailing text, which
        # _normalize has not lowercased yet, so a capitalized "Or" or "Discard"
        # must be caught just as readily. Sentence-initial capitals after a
        # dash or inside a parenthetical are exactly where models put them.
        for move in (
            "Play 0 -- Or maybe Discard 1",
            "Play 0, Else Discard 1",
            "Play 0 (Discard 1 is also fine)",
            "Play 0 - Instead, Discard 1",
            "Play 0; alternatively Play 1",
            "Play 0 -- Play 1 also works",
        ):
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertIsNone(result.legal_action, move)
            self.assertEqual(result.raw_action, move)

    def test_a_rejected_alternative_is_not_an_undecided_answer(self):
        # Naming the move you did NOT take is how Hanabi reasoning is written
        # down, and the answer is fully settled. Refusing these burns one of
        # the two retries on exactly the models that reason best, and (since
        # the move parses) the rethink they get calls their legal move
        # illegal -- pointing them at the board, which was never the problem.
        #
        # "instead OF" belongs here rather than with the refusals above: the
        # preposition inverts the word, naming the alternative being rejected
        # rather than offering one.
        cases = {
            "Play 0 rather than Discard 2": "(Play 0)",
            "Play 0 (better than Reveal player +1 rank 3)": "(Play 0)",
            "Play 0 -- Discard 2 is worse here": "(Play 0)",
            "Play 0 (not Play 1)": "(Play 0)",
            "Play 0 - Instead of Discard 2": "(Play 0)",
            "Play 0 -- I considered Discard 2 and rejected it": "(Play 0)",
        }
        for move, expected in cases.items():
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertEqual(result.legal_action, expected, move)

    def test_a_predicted_teammate_move_is_not_a_second_action(self):
        # The most common thing a Hanabi player says about their move is what
        # it lets their PARTNER do next. That is one move this turn plus a
        # forecast of somebody else's, not two instructions.
        cases = {
            "Reveal player +1 rank 3 (they will then play slot 3)": "(Reveal player +1 rank 3)",
            "Play 0 -- my teammate can then play slot 1": "(Play 0)",
            "Reveal player +1 color R (P1 should then play 2)": "(Reveal player +1 color R)",
        }
        for move, expected in cases.items():
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertEqual(result.legal_action, expected, move)

    def test_hint_phrasings_with_filler_and_reversed_order(self):
        # "reveal <target> <value>" is the engine's word order, but models
        # narrate hints in English: filler between the target and the value,
        # or the value first with the target trailing after "to".
        cases = {
            "Hint Player 1 about red": "(Reveal player +1 color R)",
            "Tell player +1 that they have a 3": "(Reveal player +1 rank 3)",
            "Hint player +1 about their red cards": "(Reveal player +1 color R)",
            "Clue rank 3 to player +1": "(Reveal player +1 rank 3)",
            "Reveal color R to player +1": "(Reveal player +1 color R)",
            "Tell 3 to player +1": "(Reveal player +1 rank 3)",
            # A rank hint is naturally said in the plural, and the seat
            # number is as often written closed up as spaced.
            "hint player +1 their 3s": "(Reveal player +1 rank 3)",
            "Tell player +1 about the 3s": "(Reveal player +1 rank 3)",
            "Reveal player1 rank 3": "(Reveal player +1 rank 3)",
            "Clue 3s to player +1": "(Reveal player +1 rank 3)",
        }
        for move, expected in cases.items():
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertEqual(result.legal_action, expected, move)

    def test_undecided_colors_are_refused(self):
        # A guard anchored on digits catches "rank 3 or 4" but waves through
        # the colour spelling of the same indecision, and the first colour
        # named would be submitted as if the model had chosen it.
        for move in (
            "Reveal player +1 color R or Y",
            "Hint player +1 red or blue",
            "Reveal player +1 color R / W",
            "Reveal player +1 color R, or maybe W",
            "Reveal player +1 color R or nothing",
            "Hint player +1 rank 3 or color R",
        ):
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertIsNone(result.legal_action, move)
            self.assertEqual(result.raw_action, move)

    def test_a_second_hint_named_with_a_bare_colour_letter_is_refused(self):
        # The same indecision as above, with the hint's colour abbreviated to
        # the single letter the prompt itself teaches ("Colors are the single
        # letters R/Y/G/W/B"). A guard whose hint operand accepts only the
        # spelled-out word never sees the second action, so the first one is
        # submitted as though the model had chosen it -- and because that move
        # parses, no rethink fires to tell it otherwise.
        for move in (
            "Play 0 or reveal R",
            "Play 0 or hint R",
            "Play 0 or tell R",
            "Play 0 or clue Y",
            "Play 0 then reveal R",
            "Reveal player +1 rank 3 or hint Y",
            "Reveal player +1 color R (hint Y also fine)",
        ):
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertIsNone(result.legal_action, move)
            self.assertEqual(result.raw_action, move)

    def test_a_refused_tail_stays_refused_after_normalization(self):
        # The annotation stripper judges these tails on the RAW text and
        # refuses them. Canonicalization then works from normalized text,
        # where "red" has become "r" -- so re-deriving the refusal there would
        # let a tail that was already rejected read as harmless commentary,
        # and the answer would be blocked by one code path and submitted by
        # the next.
        for move in (
            "Play 0 (or reveal red)",
            "Play 0 -- or clue yellow",
            "Play 0, or reveal blue",
            "Reveal player +1 color R -- or reveal yellow",
        ):
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertIsNone(result.legal_action, move)
            self.assertEqual(result.raw_action, move)

    def test_a_colour_letter_in_ordinary_commentary_still_parses(self):
        # The widened operand must not start refusing settled answers. Hanabi
        # commentary is full of bare colour letters -- naming the card behind
        # a play, the stacks already finished, the alternative it rejected --
        # and none of those name a second move for this turn.
        cases = {
            "Play 0 (likely R1)": "(Play 0)",
            "Play 0 -- my R1 is dead, R/Y both done": "(Play 0)",
            "Play 0 rather than reveal R": "(Play 0)",
            "Play 0 (better than hint R right now)": "(Play 0)",
            "Reveal player +1 color R (P1 should then play 0)": "(Reveal player +1 color R)",
            "Reveal player +1 color R -- touches their R2 only": "(Reveal player +1 color R)",
        }
        for move, expected in cases.items():
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertEqual(result.legal_action, expected, move)

    def test_rank_word_naming_a_card_is_refused(self):
        # "one"/"two" are ranks in the hint-value position but CARDS after a
        # play/discard verb. Expanding them to digits everywhere would turn
        # "play the one" into slot 1 -- the card-name hole, reopened in words.
        for move in ("Play the one", "Play my two", "Discard the three", "Play card two"):
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertIsNone(result.legal_action, move)
            self.assertEqual(result.raw_action, move)

    def test_rank_words_still_work_in_the_hint_value_position(self):
        # Refusing rank words as card names must not cost the natural hint
        # phrasings, which are a rethink each if they stop parsing.
        for move in ("Reveal player +1 rank three", "Clue rank three to player +1"):
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertEqual(result.legal_action, "(Reveal player +1 rank 3)", move)

    def test_reversed_hint_still_refuses_an_ambiguous_bare_seat(self):
        # The "value to target" order must not become a back door around the
        # offset-vs-seat ambiguity check that the normal order enforces.
        legal = [
            "(Play 0)",
            "(Reveal player +1 rank 3)",
            "(Reveal player +2 rank 3)",
        ]
        result = parse_response('```json\n{"move": "Clue rank 3 to player 1"}\n```', legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "Clue rank 3 to player 1")

    def test_bare_seat_number_resolves_when_only_one_target(self):
        # At two players "+1" and "player 1" name the same seat, so a bare
        # number is safe to accept.
        result = parse_response('```json\n{"move": "Hint P1 rank 3"}\n```', self.legal)
        self.assertEqual(result.legal_action, "(Reveal player +1 rank 3)")

    def test_bare_seat_number_refused_when_targets_are_ambiguous(self):
        # With two hintable targets, "player 1" could mean offset +1 or seat
        # 1 -- different teammates. Guessing wrong hints the wrong player, so
        # the matcher must refuse and let the rethink restate the notation.
        legal = [
            "(Play 0)",
            "(Reveal player +1 rank 3)",
            "(Reveal player +2 rank 3)",
        ]
        result = parse_response('```json\n{"move": "Reveal player 1 rank 3"}\n```', legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "Reveal player 1 rank 3")
        # The explicit offset form still works.
        explicit = parse_response('```json\n{"move": "Reveal player +1 rank 3"}\n```', legal)
        self.assertEqual(explicit.legal_action, "(Reveal player +1 rank 3)")

    def test_single_hintable_target_never_steals_a_legal_seat(self):
        # _resolve_offset reads a bare "K" as an offset when K is the only
        # hintable offset. From a non-zero seat that is NOT the same player
        # the absolute reading names -- so the docstring's guarantee has to be
        # the weaker one it now states: whenever the readings diverge, the
        # absolute reading needs an offset outside the legal set, so no legal
        # alternative is ever stolen. Assert that exhaustively rather than
        # trusting the prose, which was wrong before.
        for num_players in range(2, 6):
            for seat in range(num_players):
                for offset in range(1, num_players):
                    if (seat + offset) % num_players == offset:
                        continue  # readings agree; nothing to guarantee
                    # The absolute reading wants seat `offset`, which is
                    # `(offset - seat) % num_players` seats away.
                    needed = (offset - seat) % num_players
                    self.assertNotEqual(
                        needed,
                        offset,
                        f"n={num_players} seat={seat} offset={offset}: both readings legal",
                    )


# ---------------------------------------------------------------------------
# generate_prompt
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game = _load_game()
        self.state = _deal(self.game)

    def test_basic_prompt_contents(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Hanabi", prompt)
        self.assertIn("You are Player 0", prompt)
        self.assertIn("3/3 life, 8/8 info", prompt)
        self.assertIn("(Play N)", prompt)
        self.assertIn("(Reveal player +K color C)", prompt)

    def test_own_hand_is_hidden_but_teammate_hand_is_visible(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        own = prompt.split("Player 1's hand")[0]
        # The observer's own section reports knowledge only, never a card face.
        self.assertIn("slot 0: told nothing; possible", own)
        # The teammate's section shows faces.
        teammate = prompt.split("Player 1's hand")[1]
        cards = [c["card"] for c in self.state.state_dict(0)["hands"][1]["cards"]]
        for card in cards:
            self.assertIn(f"{card['color']}{card['rank']}", teammate)

    def test_perspective_swaps_between_players(self):
        obs0 = _make_observation(self.state, self.game, player_id=0)
        obs1 = _make_observation(self.state, self.game, player_id=1)
        prompt0 = generate_prompt(obs0, [])
        prompt1 = generate_prompt(obs1, [])

        self.assertNotEqual(prompt0, prompt1)
        self.assertIn("You are Player 0", prompt0)
        self.assertIn("You are Player 1", prompt1)
        self.assertIn("Player 1's hand (hint target +1)", prompt0)
        self.assertIn("Player 0's hand (hint target +1)", prompt1)
        # Player 0 must not be shown their own cards under any framing.
        self.assertNotIn("Player 0's hand", prompt0)
        self.assertNotIn("Player 1's hand", prompt1)

    def test_offset_map_resolves_seats_for_three_players(self):
        game = _load_game(players=3)
        state = _deal(game)
        prompt = generate_prompt(_make_observation(state, game, player_id=2), [])
        # From seat 2 of 3, +1 wraps to seat 0 and +2 to seat 1.
        self.assertIn("+1 = Player 0, +2 = Player 1", prompt)
        self.assertIn("Player 0's hand (hint target +1)", prompt)
        self.assertIn("Player 1's hand (hint target +2)", prompt)

    def test_parameters_read_from_observation_not_hardcoded(self):
        game = _load_game(colors=3, ranks=4, hand_size=4, max_life_tokens=2, max_information_tokens=5)
        state = _deal(game)
        prompt = generate_prompt(_make_observation(state, game, player_id=0), [])
        self.assertIn("3 colors", prompt)
        self.assertIn("ranks 1-4", prompt)
        self.assertIn("Each player holds up to 4 cards", prompt)
        self.assertIn("at most 12", prompt)  # 3 colors * 4 ranks
        self.assertIn("2/2 life, 5/5 info", prompt)
        self.assertNotIn("25", prompt)
        self.assertNotIn("50 cards", prompt)

    def test_deck_composition_scales_with_ranks(self):
        # OpenSpiel deals three 1s, one top rank, and two of everything
        # between -- so the total is not a constant.
        game = _load_game(colors=2, ranks=3, hand_size=2)
        state = _deal(game)
        prompt = generate_prompt(_make_observation(state, game, player_id=0), [])
        self.assertIn("12 cards in total", prompt)  # 2 * (3 + 2 + 1)
        self.assertIn("three 1s, two 2s, and one 3", prompt)

    def test_deck_composition_matches_engine_at_every_size(self):
        # NumberCardInstances tests the bottom rank BEFORE the top one, so at
        # ranks=1 -- where the single rank is both -- there are three copies,
        # not one. Checked against the engine's own pre-deal deck size.
        for colors in (2, 3, 5):
            for ranks in (1, 2, 3, 4, 5):
                game = _load_game(colors=colors, ranks=ranks, hand_size=1)
                engine_deck = game.new_initial_state().state_dict(0)["deck_size"]
                state = _deal(game)
                prompt = generate_prompt(_make_observation(state, game, player_id=0), [])
                self.assertIn(f"{engine_deck} cards in total", prompt, f"colors={colors} ranks={ranks}")

    def test_legal_moves_not_listed(self):
        # The prompt teaches the rules instead of enumerating the action set,
        # so the model has to reason about legality itself. "(Play 0)" is
        # excluded because the output-format example uses that token.
        obs = _make_observation(self.state, self.game, player_id=0, include_legal_actions=True)
        prompt = generate_prompt(obs, [])
        listed = [label for label in obs["legalActionStrings"] if label in prompt]
        self.assertEqual(listed, ["(Play 0)"])

    def test_prompt_requests_reasoning_before_json(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("Reason step by step", prompt)
        self.assertLess(prompt.index("Reason step by step"), prompt.index('"move"'))

    def test_fireworks_report_next_playable_rank(self):
        play = _action_id(self.state, "(Play 0)")
        self.state.apply_action(play)
        _advance(self.state)
        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=1), [])
        # Whatever happened, every color line names the rank that plays next.
        self.assertIn("next playable", prompt)

    def test_move_history_includes_all_players(self):
        hint, _, rank = _first_rank_hint(self.state)
        self.state.apply_action(hint)
        _advance(self.state)
        self.state.apply_action(self.state.legal_actions()[0])
        _advance(self.state)

        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=0), [])
        history = prompt.split("Moves played so far")[1]
        self.assertIn(f"1. P0 hinted P1 rank {rank}", history)
        self.assertIn("2. P1 ", history)

    def test_move_history_annotates_hint_targets_and_cards(self):
        hint, _, rank = _first_rank_hint(self.state)
        touched = [
            str(i) for i, c in enumerate(self.state.state_dict(0)["hands"][1]["cards"]) if c["card"]["rank"] == rank
        ]
        self.assertTrue(touched, "a legal hint must touch at least one card")
        self.state.apply_action(hint)
        _advance(self.state)
        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=1), [])
        noun = "slot" if len(touched) == 1 else "slots"
        self.assertIn(f"P0 hinted P1 rank {rank} -- {noun} {', '.join(touched)}", prompt)

    def test_move_history_distinguishes_advance_from_misplay(self):
        cards = self.state.state_dict(1)["hands"][0]["cards"]
        # On an empty board a rank-1 card advances its firework and anything
        # higher is a misplay -- the two outcomes render differently.
        good = next(i for i, c in enumerate(cards) if c["card"]["rank"] == 1)
        self.state.apply_action(_action_id(self.state, f"(Play {good})"))
        _advance(self.state)
        played = cards[good]["card"]
        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=1), [])
        self.assertIn(
            f"P0 played slot {good} ({played['color']}{played['rank']}) -- firework advanced",
            prompt,
        )

        cards1 = self.state.state_dict(0)["hands"][1]["cards"]
        bad = next(i for i, c in enumerate(cards1) if c["card"]["rank"] >= 3)
        self.state.apply_action(_action_id(self.state, f"(Play {bad})"))
        _advance(self.state)
        misplayed = cards1[bad]["card"]
        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=0), [])
        self.assertIn(
            f"P1 played slot {bad} ({misplayed['color']}{misplayed['rank']}) -- misplay, life lost",
            prompt,
        )
        self.assertIn("2/3 life", prompt)

    def test_empty_history_on_first_turn(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        self.assertIn("(none yet -- this is the first turn)", generate_prompt(obs, []))

    def test_unreplayable_history_is_reported_not_rendered_as_empty(self):
        # The history is rebuilt by deserializing, which can fail -- the blob
        # names "hanabi_proxy", so anywhere that game is not registered raises.
        # Rendering that as "this is the first turn" would tell a model deep
        # into a game that nothing had happened, and it has no way to tell.
        hint, _, _ = _first_rank_hint(self.state)
        self.state.apply_action(hint)
        _advance(self.state)
        for broken in ("not a serialized state", ""):
            obs = _make_observation(self.state, self.game, player_id=0)
            obs["serializedGameAndState"] = broken
            prompt = generate_prompt(obs, [])
            self.assertIn("could not be reconstructed", prompt)
            self.assertNotIn("this is the first turn", prompt)

    def test_own_hand_is_shown_face_up_under_seer_observations(self):
        # observation_type=seer reveals the observer's own cards. Rendering
        # them as unknown would hide information the engine handed over, and
        # the "you cannot see your own cards" rule line would be false.
        game = _load_game(observation_type="seer")
        state = _deal(game)
        prompt = generate_prompt(_make_observation(state, game, player_id=0), [])
        own = prompt.split("Player 1's hand")[0]
        for card in state.state_dict(0)["hands"][0]["cards"]:
            self.assertIn(f"{card['card']['color']}{card['card']['rank']}", own)
        self.assertIn("can see every player's cards, including their own", prompt)
        self.assertNotIn("but not their own", prompt)

    def test_default_observations_still_hide_the_observers_own_cards(self):
        # The seer branch must not weaken the default: the rule line still
        # says the hand is hidden, and no face appears in the own-hand block.
        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=0), [])
        self.assertIn("can see every other player's cards but not their own", prompt)
        self.assertNotIn("including their own", prompt)
        # Just the slot lines -- "R1" legitimately appears in the rules above
        # them ("next playable R1").
        own = prompt.split("You are Player 0")[1].split("Player 1's hand")[0]
        for card in self.state.state_dict(1)["hands"][0]["cards"]:
            self.assertNotIn(f"{card['card']['color']}{card['card']['rank']}", own)

    def test_no_endgame_countdown_while_the_deck_holds_cards(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        self.assertNotIn("final round", generate_prompt(obs, []))

    def test_endgame_countdown_tracks_the_engine(self):
        # The engine seeds turns_to_play_ at num_players and decrements it on
        # every decision taken once the deck is empty, so the model must be
        # told how many final turns are left -- the prompt promises the rule
        # but the proxy exposes no counter.
        # A tiny deck so discard-only play drains it before the lives run out.
        game = _load_game(colors=2, ranks=2, hand_size=2)
        state = _deal(game)
        seen = []
        while not state.is_terminal():
            _advance(state)
            if state.is_terminal():
                break
            player = int(state.current_player())
            observation = _make_observation(state, game, player_id=player)
            if state.state_dict(player)["deck_size"] == 0:
                seen.append(generate_prompt(observation, []))
            labels = {state.action_to_string(player, a): a for a in state.legal_actions()}
            # Discard by preference: it empties the deck without burning lives.
            discard = next((a for label, a in labels.items() if label.startswith("(Discard")), None)
            state.apply_action(discard if discard is not None else next(iter(labels.values())))

        self.assertTrue(seen, "the game never reached deck exhaustion")
        # Two players, so the countdown opens at 2 and ends on the last turn.
        self.assertIn("final round: 2 turns remain, including yours", seen[0])
        self.assertIn("final round: this is the last turn of the game", seen[-1])
        self.assertLessEqual(len(seen), 2)

    def test_discards_listed_with_counts(self):
        # Discarding needs an info token spent first: the opening state sits at
        # the cap, where discards are illegal.
        self.state.apply_action(_first_rank_hint(self.state)[0])
        _advance(self.state)
        card = self.state.state_dict(0)["hands"][1]["cards"][0]["card"]
        self.state.apply_action(_action_id(self.state, "(Discard 0)"))
        _advance(self.state)
        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=0), [])
        self.assertIn(f"Discarded: {card['color']}{card['rank']}", prompt)

    def test_rethink_suffix_illegal(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [], previous_response="I'll play slot 9", previous_action="(Play 9)")
        self.assertIn("You suggested", prompt)
        self.assertIn("(Play 9)", prompt)
        self.assertIn("not a legal move", prompt)

    def test_rethink_suffix_unparsable(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [], previous_response="Hmm, let me think about it.")
        self.assertIn("Your previous response ended with", prompt)
        self.assertIn("Hmm, let me think about it.", prompt)

    def test_rethink_for_an_undecided_answer_names_the_phrasing_not_the_board(self):
        # An answer refused for naming two moves usually named a LEGAL one
        # first, so the illegal-move text sends the model to re-examine the
        # board -- the one thing that was not wrong. render_rethink_suffix
        # only splits parsed-vs-unparsed, so this third case needs its own.
        obs = _make_observation(self.state, self.game, player_id=0)
        for answer in ("Play 0 or Play 1", "Play 1 then discard 2", "Play 0 (Discard 1 is also fine)"):
            prompt = generate_prompt(obs, [], previous_response=f'{{"move": "{answer}"}}', previous_action=answer)
            self.assertIn("names more than one move", prompt, answer)
            self.assertIn(answer, prompt)
            self.assertNotIn("not a legal move", prompt, answer)

    def test_a_genuinely_illegal_move_still_gets_the_illegal_rethink(self):
        # The undecided route must not swallow the ordinary case: a single,
        # settled, illegal move is a game-state problem and the model should
        # be sent back to the board.
        obs = _make_observation(self.state, self.game, player_id=0)
        for answer in ("(Play 9)", "Play 9 (my only playable card)", "Discard 4 rather than Play 1"):
            prompt = generate_prompt(obs, [], previous_response=f'{{"move": "{answer}"}}', previous_action=answer)
            self.assertIn("not a legal move", prompt, answer)
            self.assertNotIn("names more than one move", prompt, answer)

    def test_no_rethink_on_first_attempt(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertNotIn("Your previous response", prompt)
        self.assertNotIn("You suggested", prompt)
        self.assertNotIn("names more than one move", prompt)


# ---------------------------------------------------------------------------
# get_legal_moves
# ---------------------------------------------------------------------------


class GetLegalMovesTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game = _load_game()
        self.state = _deal(self.game)

    def test_from_provided_actions(self):
        obs = {
            "legalActions": [0, 5, 15],
            "legalActionStrings": ["(Discard 0)", "(Play 0)", "(Reveal player +1 rank 1)"],
        }
        self.assertEqual(
            get_legal_moves(obs),
            {0: "(Discard 0)", 5: "(Play 0)", 15: "(Reveal player +1 rank 1)"},
        )

    def test_from_serialized_state_matches_engine(self):
        # The production config omits legalActions, so this is the real path.
        obs = _make_observation(self.state, self.game, player_id=0)
        self.assertNotIn("legalActions", obs)
        expected = {a: self.state.action_to_string(0, a) for a in self.state.legal_actions()}
        self.assertEqual(get_legal_moves(obs), expected)
        self.assertGreater(len(expected), 0)

    def test_proxy_labels_match_serialized_path(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        from_proxy = get_legal_moves(obs)
        stripped = {k: v for k, v in obs.items() if k != "observationString"}
        self.assertEqual(from_proxy, get_legal_moves(stripped))

    def test_discard_illegal_at_max_info_tokens(self):
        moves = get_legal_moves(_make_observation(self.state, self.game, player_id=0)).values()
        self.assertFalse(any(m.startswith("(Discard") for m in moves))

    def test_undeserializable_state_returns_no_moves_instead_of_raising(self):
        # The serialized blob names "hanabi_proxy", so the deserialize raises
        # wherever that game is not registered. An escaping SpielError would
        # void the whole episode; an empty dict costs only this turn.
        self.assertEqual(get_legal_moves({"serializedGameAndState": "not a state", "playerId": 0}), {})

    def test_serialized_fallback_gives_a_non_actor_nothing(self):
        # The legal hints against a hand are exactly the colors and ranks IN
        # it, so the actor's move list handed to a non-actor would spell out
        # that non-actor's own cards -- the same leak the proxy guards against.
        actor = int(self.state.current_player())
        other = (actor + 1) % 2
        obs = _make_observation(self.state, self.game, player_id=other)
        del obs["observationString"]  # Force the serialized-state tier.
        self.assertEqual(get_legal_moves(obs), {})

    def test_empty_serialized(self):
        self.assertEqual(get_legal_moves({"serializedGameAndState": ""}), {})

    def test_terminal_state_has_no_moves(self):
        # Burn all three lives on misplays to reach a terminal state.
        while not self.state.is_terminal():
            _advance(self.state)
            if self.state.is_terminal():
                break
            labels = {
                self.state.action_to_string(int(self.state.current_player()), a): a for a in self.state.legal_actions()
            }
            self.state.apply_action(labels.get("(Play 0)", next(iter(labels.values()))))
        self.assertEqual(get_legal_moves(_make_observation(self.state, self.game, player_id=0)), {})


# ---------------------------------------------------------------------------
# create_agent_fn integration
# ---------------------------------------------------------------------------


class _HanabiHarness:
    """Adapter wrapping module-level functions into the GameHarness protocol."""

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

    def parse_response(self, response, legal_action_strings, *, observation=None):
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

    def setUp(self):
        super().setUp()
        self.game = _load_game()
        self.state = _deal(self.game)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_setup_step_returns_inactive(self, mock_litellm):
        mock_litellm.drop_params = True
        agent = create_agent_fn(_HanabiHarness())

        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response('```json\n{"move": "(Play 0)"}\n```')
        agent = create_agent_fn(_HanabiHarness())

        result = agent(_make_observation(self.state, self.game, player_id=0), {})

        self.assertEqual(result["actionString"], "(Play 0)")
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["submission"], _action_id(self.state, "(Play 0)"))
        self.assertIn("thoughts", result)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_loose_notation_accepted_without_retry(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response('```json\n{"move": "play 0"}\n```')
        agent = create_agent_fn(_HanabiHarness())

        result = agent(_make_observation(self.state, self.game, player_id=0), {})

        self.assertEqual(result["actionString"], "(Play 0)")
        self.assertEqual(mock_litellm.completion.call_count, 1)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_illegal_move(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.side_effect = [
            # Discarding is illegal at the info-token cap.
            _make_mock_response('```json\n{"move": "(Discard 0)"}\n```'),
            _make_mock_response('```json\n{"move": "(Play 0)"}\n```'),
        ]
        agent = create_agent_fn(_HanabiHarness())

        result = agent(_make_observation(self.state, self.game, player_id=0), {})

        self.assertEqual(result["actionString"], "(Play 0)")
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("I cannot decide.")
        agent = create_agent_fn(_HanabiHarness())

        with self.assertRaises(ValueError):
            agent(_make_observation(self.state, self.game, player_id=0), {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_short_game_via_agent_fns(self, mock_litellm):
        """Drive a short Hanabi game with scripted LLM agents that always pick
        their first legal move, verifying the harness round-trips through
        pyspiel cleanly."""
        mock_litellm.drop_params = True
        state = self.state

        def fake_completion(*, model, messages, **kwargs):
            del model, kwargs
            content = messages[0]["content"]
            player_id = 0 if "You are Player 0" in content else 1
            first = state.action_to_string(player_id, state.legal_actions()[0])
            return _make_mock_response(f'```json\n{{"move": "{first}"}}\n```')

        mock_litellm.completion.side_effect = fake_completion
        agents = [create_agent_fn(_HanabiHarness()), create_agent_fn(_HanabiHarness())]

        for _ in range(20):
            _advance(state)
            if state.is_terminal():
                break
            cp = int(state.current_player())
            result = agents[cp](_make_observation(state, self.game, player_id=cp), {})
            self.assertEqual(result["status"], "OK")
            state.apply_action(result["submission"])

        self.assertGreater(state.move_number(), 0)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_three_player_game_via_agent_fns(self, mock_litellm):
        """Same drive with three seats, so hint offsets past +1 are exercised
        end to end."""
        mock_litellm.drop_params = True
        game = _load_game(players=3)
        state = _deal(game)

        def fake_completion(*, model, messages, **kwargs):
            del model, kwargs
            content = messages[0]["content"]
            player_id = next(i for i in range(3) if f"You are Player {i}" in content)
            first = state.action_to_string(player_id, state.legal_actions()[0])
            return _make_mock_response(f'```json\n{{"move": "{first}"}}\n```')

        mock_litellm.completion.side_effect = fake_completion
        agents = [create_agent_fn(_HanabiHarness()) for _ in range(3)]

        for _ in range(15):
            _advance(state)
            if state.is_terminal():
                break
            cp = int(state.current_player())
            result = agents[cp](_make_observation(state, game, player_id=cp), {})
            self.assertEqual(result["status"], "OK")
            state.apply_action(result["submission"])

        self.assertGreater(state.move_number(), 0)


if __name__ == "__main__":
    absltest.main()
