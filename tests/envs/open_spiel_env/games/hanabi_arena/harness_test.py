"""Tests for the Hanabi Arena LLM harness."""

import json
from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.hanabi_arena import hanabi_arena_game  # noqa: F401
from kaggle_environments.envs.open_spiel_env.games.hanabi_arena.harness import (
    generate_prompt,
    get_legal_moves,
    parse_response,
)

_PLAYERS_PER_TEAM = 2


def _load_game(**overrides) -> pyspiel.Game:
    return pyspiel.load_game("hanabi_arena", {"seed": 7, **overrides})


def _safe_action(state) -> int:
    """Never play a card, so a table runs its deck dry instead of bombing.

    Discarding is illegal while the info tokens are full, so hints come first
    and the discard resumes once a token has been spent.
    """
    player = int(state.current_player())
    labels = {a: state.action_to_string(player, a) for a in state.legal_actions(player)}
    for prefix in ("(Discard", "(Reveal"):
        matching = [a for a, label in labels.items() if label.startswith(prefix)]
        if matching:
            return matching[0]
    return next(iter(labels))


def _advance(state, steps: int) -> None:
    for _ in range(steps):
        if state.is_terminal():
            return
        state.apply_action(_safe_action(state))


def _make_observation(
    state,
    game: pyspiel.Game,
    player_id: int = 0,
    *,
    include_legal_actions: bool = False,
) -> dict:
    """Build a harness-style observation dict from an arena state.

    ``includeLegalActions`` defaults to false in ``open_spiel_env``, so the
    default here matches production: no ``legalActions`` key. So does the
    absent ``serializedGameAndState`` -- ``hanabi_arena`` declares
    ``hides_state_from_agents()``, because that blob rebuilds every hidden
    hand including the reader's own. Tests that want to prove the harness
    ignores it put it back themselves.
    """
    del game  # Only the state is needed now that the blob is withheld.
    observation = {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "currentPlayer": int(state.current_player()),
        "isTerminal": state.is_terminal(),
    }
    if include_legal_actions:
        legal = list(state.legal_actions(player_id))
        observation["legalActions"] = legal
        observation["legalActionStrings"] = [state.action_to_string(player_id, a) for a in legal]
    return observation


def _action_id(state, label: str) -> int:
    player = int(state.current_player())
    for action in state.legal_actions(player):
        if state.action_to_string(player, action) == label:
            return action
    raise AssertionError(f"{label!r} is not legal here")


def _first_rank_hint(state) -> tuple[int, int]:
    """First legal rank hint as ``(action_id, rank)``.

    Which ranks are hintable depends on what the deal put in the target's
    hand, so tests pick from the legal set rather than naming a rank.
    """
    player = int(state.current_player())
    for action in state.legal_actions(player):
        label = state.action_to_string(player, action)
        if "rank" in label:
            return action, int(label.rstrip(")").split()[-1])
    raise AssertionError("no rank hint is legal here")


def _table_of(state, player_id: int) -> dict:
    return json.loads(state.observation_string(player_id))["table"]


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

    def test_last_json_block_wins(self):
        # Models draft an answer, reconsider, then commit. The last block is
        # the intent; taking the first silently submits the rejected draft.
        response = '```json\n{"move": "(Play 0)"}\n```\nOn reflection:\n```json\n{"move": "(Play 1)"}\n```'
        self.assertEqual(parse_response(response, self.legal).legal_action, "(Play 1)")

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
        # move-shaped token in the prose -- return None and let rethink ask
        # the model to use the required JSON format.
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
        # The prompt renders the teammate's cards as "R1"/"B3" tokens, so
        # naming the card instead of the slot is the likeliest notation slip.
        # The card's RANK must never be read as a slot index: "Play R1" would
        # silently become slot 1, an action the model never chose.
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
        # retries before the agent forfeits. A second action needs an OPERAND.
        cases = {
            "Play 0 (safest play)": "(Play 0)",
            "Play 0 (better than a hint right now)": "(Play 0)",
            "Discard 2 (to regain a token for a future hint)": "(Discard 2)",
            "Reveal player +1 rank 3 (sets up two plays)": "(Reveal player +1 rank 3)",
        }
        for move, expected in cases.items():
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertEqual(result.legal_action, expected, move)

    def test_probabilistic_rationale_is_accepted(self):
        # Hanabi reasoning is inherently probabilistic, so a model annotates
        # its move with odds. A bare "/" in the undecided check would reject
        # every fraction while letting "0.66" through -- an arbitrary split
        # that costs a rethink for saying the same thing more naturally.
        cases = {
            "Play 0 (2/3 chance it is W1)": "(Play 0)",
            "Play 0 -- 60/40 it is W1": "(Play 0)",
            "Play 0 (probability 0.66)": "(Play 0)",
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
        # misread.
        for move in ("Play 0-1", "Play 0, 1", "Play 0; 1", "Play 0 -- 1", "Play 0 / 1", "Play 0 or 1"):
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertIsNone(result.legal_action, move)
            self.assertEqual(result.raw_action, move)

    def test_undecided_answer_is_refused_whatever_the_capitalization(self):
        # The undecided/second-verb check runs on the RAW trailing text, which
        # _normalize has not lowercased yet, so a capitalized "Or" or "Discard"
        # must be caught just as readily.
        for move in (
            "Play 0 -- Or maybe Discard 1",
            "Play 0, Else Discard 1",
            "Play 0 (Discard 1 is also fine)",
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
        cases = {
            "Play 0 rather than Discard 2": "(Play 0)",
            "Play 0 (better than Reveal player +1 rank 3)": "(Play 0)",
            "Play 0 -- Discard 2 is worse here": "(Play 0)",
            "Play 0 (not Play 1)": "(Play 0)",
            "Play 0 (instead of burning a token on Reveal player +1 color R)": "(Play 0)",
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
            "Clue rank 3 to player +1": "(Reveal player +1 rank 3)",
            "Reveal color R to player +1": "(Reveal player +1 color R)",
            "hint player +1 their 3s": "(Reveal player +1 rank 3)",
            "Reveal player1 rank 3": "(Reveal player +1 rank 3)",
        }
        for move, expected in cases.items():
            result = parse_response(f'```json\n{{"move": "{move}"}}\n```', self.legal)
            self.assertEqual(result.legal_action, expected, move)

    def test_bare_seat_number_resolves_when_only_one_target(self):
        # A table seats two, so "+1" and "player 1" name the same seat and a
        # bare number is safe to accept.
        result = parse_response('```json\n{"move": "Hint P1 rank 3"}\n```', self.legal)
        self.assertEqual(result.legal_action, "(Reveal player +1 rank 3)")

    def test_arena_player_id_is_not_read_as_a_hint_target(self):
        # The prompt names the teammate by ARENA id ("Player 3"), while the
        # engine's hint targets are table-relative offsets. A model that
        # writes the arena id must not have it silently read as an offset --
        # "+3" is not a legal offset at a two-seat table, so the matcher has
        # to refuse rather than coerce it to the one legal target.
        result = parse_response('```json\n{"move": "Hint player +3 rank 3"}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "Hint player +3 rank 3")

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


class ArenaPlayerIdHintTest(absltest.TestCase):
    """A hint addressed by arena player id resolves at every seat.

    Table-relative offsets only coincide with arena ids for team 0 seat 0. The
    matcher therefore needs the observation to map ids to offsets; without it
    a model that names its teammate by the id the PROMPT gave it is refused on
    three of the four seats, which is a team-asymmetric handicap rather than a
    parser quirk.
    """

    # Colour names as the prompt writes them, for the "Hint player 3 green"
    # phrasing that spells the colour out.
    _COLOR_NAMES = {"R": "red", "Y": "yellow", "G": "green", "W": "white", "B": "blue"}

    def setUp(self):
        super().setUp()
        self.game = _load_game()

    def _turn_of(self, player_id: int):
        """A state where ``player_id`` is the actor, with its legal labels.

        Only the acting seat has legal actions, so each seat needs the game
        walked forward to its own turn before its notation can be exercised.
        """
        state = _load_game().new_initial_state()
        for _ in range(16):
            if int(state.current_player()) == player_id:
                break
            state.apply_action(_safe_action(state))
        else:
            raise AssertionError(f"player {player_id} never got a turn")
        legal = [state.action_to_string(player_id, a) for a in state.legal_actions(player_id)]
        return state, legal

    def _a_legal_color_hint(self, legal: list[str]) -> tuple[str, str]:
        """``(label, color letter)`` for some colour hint that is legal here.

        Which colours are hintable depends on the deal, so the test picks from
        the legal set rather than naming one.
        """
        for label in legal:
            if label.startswith("(Reveal player +1 color "):
                return label, label.rstrip(")").split()[-1]
        raise AssertionError("no colour hint is legal here")

    def test_every_seat_resolves_its_teammates_arena_id(self):
        teammate_of = {0: 1, 1: 0, 2: 3, 3: 2}
        for player_id, teammate in teammate_of.items():
            state, legal = self._turn_of(player_id)
            observation = _make_observation(state, self.game, player_id=player_id)
            expected, letter = self._a_legal_color_hint(legal)
            for move in (
                f"Reveal player {teammate} color {letter}",
                f"hint P{teammate} color {letter}",
                f"Hint player {teammate} {self._COLOR_NAMES[letter]}",
            ):
                result = parse_response(
                    f'```json\n{{"move": "{move}"}}\n```',
                    legal,
                    observation=observation,
                )
                self.assertEqual(result.legal_action, expected, (player_id, move))

    def test_an_opponents_arena_id_is_still_refused(self):
        # Hints cross seats at a table, never tables. Reading an opposing
        # player's id as "my partner" would submit a hint the model never
        # intended at the only seat it could legally land on.
        #
        # Every seat against every id that is not its teammate, because the
        # failure is seat-specific: with two players per team the only legal
        # offset is always +1, so the bare number 1 reads as a valid offset at
        # the same time as it names a real seat at the other table. Checking
        # from seat 0 alone (whose opponents are 2 and 3) never presents that
        # collision, and the leak -- P3 writing "player 1" and hitting P2 --
        # hides behind a passing test.
        teammate_of = {0: 1, 1: 0, 2: 3, 3: 2}
        for player_id, teammate in teammate_of.items():
            state, legal = self._turn_of(player_id)
            observation = _make_observation(state, self.game, player_id=player_id)
            _, letter = self._a_legal_color_hint(legal)
            for other in (pid for pid in range(4) if pid != teammate):
                result = parse_response(
                    f'```json\n{{"move": "Reveal player {other} color {letter}"}}\n```',
                    legal,
                    observation=observation,
                )
                self.assertIsNone(result.legal_action, (player_id, other))

    def test_the_observation_is_optional(self):
        # ``parse_response`` is called without the kwarg by any caller that
        # predates it, and must keep working on explicit offsets.
        _, legal = self._turn_of(0)
        expected, letter = self._a_legal_color_hint(legal)
        result = parse_response(f'```json\n{{"move": "Reveal player +1 color {letter}"}}\n```', legal)
        self.assertEqual(result.legal_action, expected)

    def test_without_an_observation_a_bare_number_is_still_an_offset(self):
        # The cross-table refusal is driven by the id map, which only exists
        # when an observation was passed. With no map there is no way to know
        # a number names another table, and the old offset-only reading -- a
        # bare number when exactly one target is hintable -- has to survive.
        _, legal = self._turn_of(0)
        expected, letter = self._a_legal_color_hint(legal)
        result = parse_response(f'```json\n{{"move": "Reveal player 1 color {letter}"}}\n```', legal)
        self.assertEqual(result.legal_action, expected)

    def test_an_explicit_offset_resolves_at_every_seat(self):
        # The id map must not shadow "+1", which is offset notation by
        # construction and never an arena id.
        for player_id in range(4):
            state, legal = self._turn_of(player_id)
            observation = _make_observation(state, self.game, player_id=player_id)
            expected, letter = self._a_legal_color_hint(legal)
            result = parse_response(
                f'```json\n{{"move": "Reveal player +1 color {letter}"}}\n```',
                legal,
                observation=observation,
            )
            self.assertEqual(result.legal_action, expected, player_id)


# ---------------------------------------------------------------------------
# generate_prompt
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game = _load_game()
        self.state = self.game.new_initial_state()

    def test_basic_prompt_contents(self):
        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=0), [])
        self.assertIn("Hanabi Arena", prompt)
        self.assertIn("You are Player 0 on Team 0, seat 0", prompt)
        self.assertIn("Your teammate is Player 1", prompt)
        self.assertIn("3/3 life, 8/8 info", prompt)
        self.assertIn("(Play N)", prompt)
        self.assertIn("(Reveal player +K color C)", prompt)

    def test_arena_framing_is_explained(self):
        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=0), [])
        self.assertIn("same shuffled deck", prompt)
        self.assertIn("higher final score wins", prompt)
        self.assertIn("never see the opposing table", prompt)

    def test_own_hand_is_hidden_but_teammate_hand_is_visible(self):
        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=0), [])
        own = prompt.split("Player 1's hand")[0]
        # The observer's own section reports knowledge only, never a card face.
        self.assertIn("slot 0: told nothing; possible", own)
        teammate = prompt.split("Player 1's hand")[1]
        for card in _table_of(self.state, 0)["hands"][1]["cards"]:
            self.assertIn(f"{card['card']['color']}{card['card']['rank']}", teammate)

    def test_opposing_table_is_never_rendered(self):
        # The whole point of the arena is that the two tables are the same
        # puzzle. Leaking team B's hands to team A would hand the second
        # mover the answer key.
        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=0), [])
        for pid in (2, 3):
            self.assertNotIn(f"Player {pid}'s hand", prompt)
        self.assertNotIn("Team 1", prompt)

    def test_perspective_swaps_between_seats_at_a_table(self):
        prompt0 = generate_prompt(_make_observation(self.state, self.game, player_id=0), [])
        prompt1 = generate_prompt(_make_observation(self.state, self.game, player_id=1), [])
        self.assertNotEqual(prompt0, prompt1)
        self.assertIn("You are Player 0 on Team 0, seat 0", prompt0)
        self.assertIn("You are Player 1 on Team 0, seat 1", prompt1)
        self.assertIn("Player 1's hand (hint target +1)", prompt0)
        self.assertIn("Player 0's hand (hint target +1)", prompt1)
        # Neither seat may be shown its own cards under any framing.
        self.assertNotIn("Player 0's hand", prompt0)
        self.assertNotIn("Player 1's hand", prompt1)

    def test_perspective_swaps_between_teams(self):
        prompt0 = generate_prompt(_make_observation(self.state, self.game, player_id=0), [])
        prompt2 = generate_prompt(_make_observation(self.state, self.game, player_id=2), [])
        self.assertIn("You are Player 2 on Team 1, seat 0", prompt2)
        self.assertIn("Your teammate is Player 3", prompt2)
        self.assertIn("Player 3's hand (hint target +1)", prompt2)
        self.assertIn("+1 = Player 3", prompt2)
        self.assertIn("+1 = Player 1", prompt0)
        self.assertIn("wrapping from Player 3 back to Player 2", prompt2)
        self.assertIn("wrapping from Player 1 back to Player 0", prompt0)

    def test_mirrored_tables_render_identical_hands_for_mirrored_seats(self):
        # Both tables are dealt from one shuffled deck, so at move 0 seat 1 of
        # each table holds the same cards. If this ever diverges the arena is
        # no longer a fair head-to-head, and the prompt is where it shows.
        prompt0 = generate_prompt(_make_observation(self.state, self.game, player_id=0), [])
        prompt2 = generate_prompt(_make_observation(self.state, self.game, player_id=2), [])
        hand0 = prompt0.split("Player 1's hand (hint target +1):")[1].split("Moves played")[0]
        hand2 = prompt2.split("Player 3's hand (hint target +1):")[1].split("Moves played")[0]
        self.assertEqual(hand0, hand2)

    def test_parameters_read_from_observation_not_hardcoded(self):
        game = _load_game(colors=3, ranks=4, hand_size=4, max_life_tokens=2, max_information_tokens=5)
        state = game.new_initial_state()
        prompt = generate_prompt(_make_observation(state, game, player_id=0), [])
        self.assertIn("3 colors", prompt)
        self.assertIn("ranks 1-4", prompt)
        self.assertIn("Each player holds up to 4 cards", prompt)
        self.assertIn("at most 12", prompt)  # 3 colors * 4 ranks
        self.assertIn("2/2 life, 5/5 info", prompt)
        self.assertNotIn("25", prompt)
        self.assertNotIn("50 cards", prompt)

    def test_fallback_constants_describe_a_real_default_game(self):
        # With no observationString the prompt still renders, and every rule
        # line it states must be true of the default game. A wrong constant is
        # worse than a missing one -- the model cannot tell it was misinformed,
        # so "25 cards" or "up to 0 cards" would corrupt its whole plan.
        prompt = generate_prompt({"playerId": 0}, [])
        self.assertIn("50 cards in total", prompt)  # NOT colors * ranks
        self.assertIn("Each player holds up to 5 cards", prompt)
        self.assertIn("5 colors (R/Y/G/W/B)", prompt)
        self.assertIn("ranks 1-5", prompt)
        self.assertIn("at most 25", prompt)

        # The same numbers the engine's defaults actually produce.
        table = _table_of(self.state, 0)
        self.assertEqual(table["deck_total"], 50)
        self.assertEqual(table["hand_size"], 5)
        self.assertEqual(table["max_info_tokens"], 8)
        self.assertEqual(table["max_life_tokens"], 3)

    def test_deck_composition_matches_engine_at_every_size(self):
        # NumberCardInstances tests the bottom rank BEFORE the top one, so at
        # ranks=1 -- where the single rank is both -- there are three copies,
        # not one. Checked against the engine's own reported deck total.
        for colors in (2, 3, 5):
            for ranks in (1, 2, 3, 4, 5):
                game = _load_game(colors=colors, ranks=ranks, hand_size=1)
                state = game.new_initial_state()
                total = _table_of(state, 0)["deck_total"]
                prompt = generate_prompt(_make_observation(state, game, player_id=0), [])
                self.assertIn(f"{total} cards in total", prompt, f"colors={colors} ranks={ranks}")

    def test_legal_moves_not_listed(self):
        # A Hanabi player cannot see their own hand, so the play/discard slots
        # are not derivable from the visible state -- but the prompt still
        # teaches legality from the rules rather than pasting the action set.
        # "(Play 0)" is excluded because the output-format example uses it.
        obs = _make_observation(self.state, self.game, player_id=0, include_legal_actions=True)
        prompt = generate_prompt(obs, [])
        listed = [label for label in obs["legalActionStrings"] if label in prompt]
        self.assertEqual(listed, ["(Play 0)"])

    def test_prompt_requests_reasoning_before_json(self):
        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=0), [])
        self.assertIn("Reason step by step", prompt)
        self.assertLess(prompt.index("Reason step by step"), prompt.index('"move"'))

    def test_move_history_covers_both_seats_at_the_table(self):
        hint, rank = _first_rank_hint(self.state)
        self.state.apply_action(hint)  # P0
        _advance(self.state, 2)  # P2 (other table), then P1
        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=0), [])
        history = prompt.split("Moves played at your table")[1]
        self.assertIn(f"1. P0 hinted P1 rank {rank}", history)
        self.assertIn("2. P1 ", history)

    def test_move_history_excludes_the_opposing_table(self):
        _advance(self.state, 4)
        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=0), [])
        history = prompt.split("Moves played at your table")[1]
        for pid in ("P2", "P3"):
            self.assertNotIn(pid, history)

    def test_move_history_annotates_hint_targets(self):
        hint, rank = _first_rank_hint(self.state)
        touched = [
            str(i) for i, c in enumerate(_table_of(self.state, 0)["hands"][1]["cards"]) if c["card"]["rank"] == rank
        ]
        self.assertTrue(touched, "a legal hint must touch at least one card")
        self.state.apply_action(hint)
        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=1), [])
        noun = "slot" if len(touched) == 1 else "slots"
        self.assertIn(f"P0 hinted P1 rank {rank} -- {noun} {', '.join(touched)}", prompt)

    def test_move_history_distinguishes_advance_from_misplay(self):
        cards = _table_of(self.state, 1)["hands"][0]["cards"]
        # On an empty board a rank-1 card advances its firework and anything
        # higher is a misplay -- the two outcomes render differently.
        good = next(i for i, c in enumerate(cards) if c["card"]["rank"] == 1)
        played = cards[good]["card"]
        self.state.apply_action(_action_id(self.state, f"(Play {good})"))
        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=1), [])
        self.assertIn(
            f"P0 played slot {good} ({played['color']}{played['rank']}) -- firework advanced",
            prompt,
        )

        self.state.apply_action(_safe_action(self.state))  # Team B's turn.
        cards1 = _table_of(self.state, 0)["hands"][1]["cards"]
        bad = next(i for i, c in enumerate(cards1) if c["card"]["rank"] >= 3)
        misplayed = cards1[bad]["card"]
        self.state.apply_action(_action_id(self.state, f"(Play {bad})"))
        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=0), [])
        self.assertIn(
            f"P1 played slot {bad} ({misplayed['color']}{misplayed['rank']}) -- misplay, life lost",
            prompt,
        )
        self.assertIn("2/3 life", prompt)

    def test_empty_history_on_first_turn(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        self.assertIn("(none yet -- this is the first turn at your table)", generate_prompt(obs, []))

    def test_history_needs_no_serialized_state(self):
        # The env withholds serializedGameAndState from this game, so the
        # annotated history has to come entirely out of the table view. A
        # blob that is absent, empty, or unreadable must change nothing.
        hint, rank = _first_rank_hint(self.state)
        self.state.apply_action(hint)
        _advance(self.state, 2)
        reference = generate_prompt(_make_observation(self.state, self.game, player_id=0), [])
        self.assertIn(f"1. P0 hinted P1 rank {rank}", reference)
        self.assertIn("2. P1 ", reference)
        self.assertNotIn("this is the first turn", reference)
        for blob in ("not a serialized state", "", pyspiel.serialize_game_and_state(self.game, self.state)):
            obs = _make_observation(self.state, self.game, player_id=0)
            self.assertNotIn("serializedGameAndState", obs)
            obs["serializedGameAndState"] = blob
            self.assertEqual(generate_prompt(obs, []), reference, blob[:24])

    def test_hint_slots_are_walked_forward_as_cards_leave_the_hand(self):
        # Slots are positions, not identities: discarding slot 0 slides every
        # card above it down one. A hint line still naming the slot the cards
        # sat in when it was given points at different cards -- and the holder
        # cannot see the faces to notice. Both readings have to be stated.
        hint, rank = _first_rank_hint(self.state)
        cards = _table_of(self.state, 0)["hands"][1]["cards"]
        touched = [i for i, c in enumerate(cards) if c["card"]["rank"] == rank]
        self.assertTrue(any(s > 0 for s in touched), "need a touched slot above slot 0 to shift")
        self.state.apply_action(hint)
        self.state.apply_action(_safe_action(self.state))  # Team B's turn.

        # P1 discards slot 0, shifting everything above it down one.
        self.state.apply_action(_action_id(self.state, "(Discard 0)"))
        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=0), [])

        expected_now = [s - 1 for s in touched if s > 0]
        given_noun = "slot" if len(touched) == 1 else "slots"
        line = f"P0 hinted P1 rank {rank} -- {given_noun} {', '.join(str(s) for s in touched)} at the time, "
        if expected_now:
            now_noun = "slot" if len(expected_now) == 1 else "slots"
            line += f"now {now_noun} {', '.join(str(s) for s in expected_now)}"
            if len(expected_now) != len(touched):
                line += " (the rest since played or discarded)"
        else:
            line += "all since played or discarded"
        self.assertIn(line, prompt)

    def test_unmoved_hint_slots_render_without_the_qualifier(self):
        # The forward-walk must stay invisible until something actually moves.
        hint, rank = _first_rank_hint(self.state)
        self.state.apply_action(hint)
        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=1), [])
        self.assertNotIn("at the time", prompt)
        self.assertNotIn("since played or discarded", prompt)

    def test_history_unavailable_is_reported_not_rendered_as_empty(self):
        # With no move log in the table view, the prompt must say so.
        # Rendering that as "this is the first turn" would tell a model deep
        # into a game that nothing had happened, and it has no way to tell.
        self.state.apply_action(_safe_action(self.state))
        obs = _make_observation(self.state, self.game, player_id=0)
        view = json.loads(obs["observationString"])
        del view["table"]["move_history"]
        obs["observationString"] = json.dumps(view)
        prompt = generate_prompt(obs, [])
        self.assertIn("could not be reconstructed", prompt)
        self.assertNotIn("this is the first turn", prompt)

    def test_no_endgame_countdown_while_the_deck_holds_cards(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        self.assertNotIn("final round", generate_prompt(obs, []))

    def test_endgame_countdown_tracks_the_engine(self):
        # The engine seeds turns_to_play_ at the table's seat count and
        # decrements it on every decision taken once the deck is empty. The
        # prompt promises the rule, so it has to show the counter.
        game = _load_game(colors=2, ranks=2, hand_size=2)
        state = game.new_initial_state()
        seen = []
        while not state.is_terminal():
            player = int(state.current_player())
            if player // _PLAYERS_PER_TEAM == 0 and _table_of(state, 0)["deck_size"] == 0:
                seen.append(generate_prompt(_make_observation(state, game, player_id=player), []))
            state.apply_action(_safe_action(state))

        self.assertTrue(seen, "team A's table never reached deck exhaustion")
        self.assertIn("final round: 2 turns remain at your table, including yours", seen[0])
        self.assertIn("final round: this is the last turn at your table", seen[-1])
        self.assertLessEqual(len(seen), 2)

    def test_discards_listed_with_counts(self):
        # Discarding needs an info token spent first: the opening state sits
        # at the cap, where discards are illegal.
        self.state.apply_action(_first_rank_hint(self.state)[0])  # P0 hint.
        self.state.apply_action(_safe_action(self.state))  # Team B.
        card = _table_of(self.state, 0)["hands"][1]["cards"][0]["card"]
        self.state.apply_action(_action_id(self.state, "(Discard 0)"))  # P1.
        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=0), [])
        self.assertIn(f"Discarded: {card['color']}{card['rank']}", prompt)

    def test_score_reflects_the_players_own_table(self):
        cards = _table_of(self.state, 1)["hands"][0]["cards"]
        good = next(i for i, c in enumerate(cards) if c["card"]["rank"] == 1)
        self.state.apply_action(_action_id(self.state, f"(Play {good})"))  # P0 scores.
        self.state.apply_action(_safe_action(self.state))  # Team B hints instead.
        self.assertIn("Score: 1/25", generate_prompt(_make_observation(self.state, self.game, player_id=1), []))
        self.assertIn("Score: 0/25", generate_prompt(_make_observation(self.state, self.game, player_id=3), []))

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
        prompt = generate_prompt(_make_observation(self.state, self.game, player_id=0), [])
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
        self.state = self.game.new_initial_state()

    def test_from_provided_actions(self):
        obs = {
            "legalActions": [0, 5, 15],
            "legalActionStrings": ["(Discard 0)", "(Play 0)", "(Reveal player +1 rank 1)"],
        }
        self.assertEqual(
            get_legal_moves(obs),
            {0: "(Discard 0)", 5: "(Play 0)", 15: "(Reveal player +1 rank 1)"},
        )

    def test_from_observation_matches_engine(self):
        # The production config omits legalActions, so this is the real path.
        obs = _make_observation(self.state, self.game, player_id=0)
        self.assertNotIn("legalActions", obs)
        expected = {a: self.state.action_to_string(0, a) for a in self.state.legal_actions(0)}
        self.assertEqual(get_legal_moves(obs), expected)
        self.assertGreater(len(expected), 0)

    def test_there_is_no_serialized_state_fallback(self):
        # Deserializing the blob would rebuild every hidden hand, including
        # the reader's own -- which is why the env withholds it for this game
        # (hides_state_from_agents). A harness tier that reached for it anyway
        # would re-open the leak the moment the view went missing, so with no
        # readable view the answer is {}: this turn is lost, not the secret.
        obs = _make_observation(self.state, self.game, player_id=0)
        self.assertTrue(get_legal_moves(obs), "the view path must work to make this test meaningful")
        stripped = {k: v for k, v in obs.items() if k != "observationString"}
        stripped["serializedGameAndState"] = pyspiel.serialize_game_and_state(self.game, self.state)
        self.assertEqual(get_legal_moves(stripped), {})

    def test_discard_illegal_at_max_info_tokens(self):
        moves = get_legal_moves(_make_observation(self.state, self.game, player_id=0)).values()
        self.assertFalse(any(m.startswith("(Discard") for m in moves))

    def test_non_actor_gets_nothing(self):
        # Only one player is on the clock across both tables. The legal hints
        # against a hand are exactly the colors and ranks IN it, so handing
        # the actor's list to anyone else spells out their own cards.
        actor = int(self.state.current_player())
        for player in range(4):
            if player == actor:
                continue
            self.assertEqual(get_legal_moves(_make_observation(self.state, self.game, player_id=player)), {}, player)

    def test_malformed_observations_return_no_moves_instead_of_raising(self):
        # An escaping exception would void the whole episode; an empty dict
        # costs only this turn.
        for obs in (
            {},
            {"playerId": 0},
            {"observationString": "not json", "playerId": 0},
            {"observationString": "", "playerId": 0},
            {"serializedGameAndState": "not a state", "playerId": 0},
            {"legalActions": [], "legalActionStrings": [], "playerId": 0},
        ):
            with self.subTest(obs=obs):
                self.assertEqual(get_legal_moves(obs), {})

    def test_terminal_state_has_no_moves(self):
        while not self.state.is_terminal():
            self.state.apply_action(_safe_action(self.state))
        self.assertEqual(get_legal_moves(_make_observation(self.state, self.game, player_id=0)), {})


# ---------------------------------------------------------------------------
# create_agent_fn integration
# ---------------------------------------------------------------------------


class _HanabiArenaHarness:
    """Test-local GameHarness adapter; mirrors the prod wrapper shape."""

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
        # Forwarded, not dropped: core_harness always passes the observation,
        # and it is what lets the parser read a hint target written as the
        # arena player id the prompt printed. Swallowing it here would leave
        # that whole path untested through the wiring production uses.
        return parse_response(response, legal_action_strings, observation=observation)


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
        self.state = self.game.new_initial_state()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_setup_step_returns_inactive(self, mock_litellm):
        mock_litellm.drop_params = True
        agent = create_agent_fn(_HanabiArenaHarness())

        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response('```json\n{"move": "(Play 0)"}\n```')
        agent = create_agent_fn(_HanabiArenaHarness())

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
        agent = create_agent_fn(_HanabiArenaHarness())

        result = agent(_make_observation(self.state, self.game, player_id=0), {})

        self.assertEqual(result["actionString"], "(Play 0)")
        self.assertEqual(mock_litellm.completion.call_count, 1)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_arena_player_id_hint_resolves_through_the_agent(self, mock_litellm):
        """A hint written as the arena player id must land on the first try.

        The id-to-offset map comes from the observation, which reaches the
        parser only because ``core_harness`` forwards it. Exercised at seat 3,
        where the id (2) and the offset (+1) differ -- at team 0 seat 0 they
        coincide, so that seat passes even with the map missing entirely.
        """
        mock_litellm.drop_params = True
        state = self.state
        for _ in range(16):
            if int(state.current_player()) == 3:
                break
            state.apply_action(_safe_action(state))
        legal = [state.action_to_string(3, a) for a in state.legal_actions(3)]
        expected = next(label for label in legal if label.startswith("(Reveal player +1 color "))
        letter = expected.rstrip(")").split()[-1]

        mock_litellm.completion.return_value = _make_mock_response(
            f'```json\n{{"move": "Hint player 2 color {letter}"}}\n```'
        )
        agent = create_agent_fn(_HanabiArenaHarness())

        result = agent(_make_observation(state, self.game, player_id=3), {})

        self.assertEqual(result["actionString"], expected)
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
        agent = create_agent_fn(_HanabiArenaHarness())

        result = agent(_make_observation(self.state, self.game, player_id=0), {})

        self.assertEqual(result["actionString"], "(Play 0)")
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("I cannot decide.")
        agent = create_agent_fn(_HanabiArenaHarness())

        with self.assertRaises(ValueError):
            agent(_make_observation(self.state, self.game, player_id=0), {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_short_game_via_agent_fns(self, mock_litellm):
        """Drive a short arena game with four scripted LLM agents that always
        pick their first legal move, verifying the harness round-trips through
        pyspiel cleanly and every seat's prompt identifies the right player."""
        mock_litellm.drop_params = True
        state = self.state

        def fake_completion(*, model, messages, **kwargs):
            del model, kwargs
            content = messages[0]["content"]
            player_id = next(i for i in range(4) if f"You are Player {i} on" in content)
            first = state.action_to_string(player_id, state.legal_actions(player_id)[0])
            return _make_mock_response(f'```json\n{{"move": "{first}"}}\n```')

        mock_litellm.completion.side_effect = fake_completion
        agents = [create_agent_fn(_HanabiArenaHarness()) for _ in range(4)]

        seen_players = set()
        for _ in range(20):
            if state.is_terminal():
                break
            cp = int(state.current_player())
            seen_players.add(cp)
            result = agents[cp](_make_observation(state, self.game, player_id=cp), {})
            self.assertEqual(result["status"], "OK")
            state.apply_action(result["submission"])

        self.assertGreater(state.move_number(), 0)
        # Both tables and both seats actually took turns.
        self.assertEqual(seen_players, {0, 1, 2, 3})


if __name__ == "__main__":
    absltest.main()
