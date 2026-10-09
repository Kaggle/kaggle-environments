"""Tests for the Go Fish LLM harness."""

import json
import random
import re
from unittest.mock import MagicMock, patch

import pyspiel
from absl.testing import absltest

from kaggle_environments.core_harness import ParseResult, create_agent_fn
from kaggle_environments.envs.open_spiel_env.games.go_fish import (
    go_fish_proxy,
)
from kaggle_environments.envs.open_spiel_env.games.go_fish.harness import (
    _annotate_move_history,
    _format_booked,
    _format_deductions,
    _format_events,
    _format_pool,
    _own_counts,
    generate_prompt,
    get_legal_moves,
    parse_response,
)


def _advance_chance(state, rng: random.Random) -> None:
    """Resolve chance nodes (deal/fish/draw) with a seeded RNG."""
    while not state.is_terminal() and state.is_chance_node():
        outcomes = state.chance_outcomes()
        action = rng.choices([o[0] for o in outcomes], weights=[o[1] for o in outcomes])[0]
        state.apply_action(action)


def _make_ask_state(seed: int = 7, plies: int = 0):
    """Return (game, state) advanced to an Ask node (a player's decision).

    ``plies`` additional random Ask moves are played first (each followed by
    chance resolution) so tests can exercise mid-game states with events.
    """
    game = go_fish_proxy.GoFishGame()
    state = game.new_initial_state()
    rng = random.Random(seed)
    _advance_chance(state, rng)
    for _ in range(plies):
        if state.is_terminal() or state.current_player() < 0:
            break
        state.apply_action(rng.choice(state.legal_actions()))
        _advance_chance(state, rng)
    return game, state


def _example_tokens(prompt: str) -> list[str]:
    """Concrete move tokens from the prompt's `{"move": "..."}` examples.

    Skips the ``<target><letter>`` placeholders in the output-format blocks,
    leaving only the real illustrative moves.
    """
    tokens = re.findall(r'\{"move": "([^"]+)"\}', prompt)
    return [t for t in tokens if not t.startswith("<")]


def _play_own_asks(seed: int, plies: int, player_id: int = 0):
    """Play ``plies`` random moves, returning (game, state, player_id's own asks).

    The returned move list is what a harness accumulates across an episode: only
    the observer's own asks, in order, ending at the most recent one.
    """
    game = go_fish_proxy.GoFishGame()
    state = game.new_initial_state()
    rng = random.Random(seed)
    _advance_chance(state, rng)
    own_asks: list[str] = []
    num_ranks = 13
    for _ in range(plies):
        if state.is_terminal():
            break
        mover = int(state.current_player())
        action = rng.choice(state.legal_actions())
        if mover == player_id:
            target, rank = action // num_ranks, action % num_ranks
            own_asks.append(f"{target}{chr(ord('a') + rank)}")
        state.apply_action(action)
        _advance_chance(state, rng)
    return game, state, own_asks


def _make_observation(state, game, player_id: int | None = None) -> dict:
    """Build a harness-style observation dict from a proxy state.

    ``serializedGameAndState`` serializes the *proxy* game and state, matching
    what production does: ``open_spiel_env`` substitutes ``go_fish_proxy`` for
    ``go_fish`` before serializing (open_spiel_env.py ``interpreter``). Passing
    the unwrapped objects here would round-trip to a raw ``go_fish`` state whose
    ``observation_string`` is OpenSpiel text rather than the proxy's JSON --
    a divergence that previously hid a harness bug from every test in this file.
    """
    if player_id is None:
        player_id = int(state.current_player())
    legal = list(state.legal_actions())
    return {
        "observationString": state.observation_string(player_id),
        "playerId": player_id,
        "currentPlayer": int(state.current_player()),
        "isTerminal": state.is_terminal(),
        "legalActions": legal,
        "legalActionStrings": [state.action_to_string(a) for a in legal],
        "serializedGameAndState": pyspiel.serialize_game_and_state(game, state),
    }


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class ParseResponseTest(absltest.TestCase):
    legal = ["1a", "1b", "1e", "1j", "1m"]

    def test_parse_json_block(self):
        result = parse_response('```json\n{"move": "1a"}\n```', self.legal)
        self.assertEqual(result.legal_action, "1a")
        self.assertEqual(result.raw_action, "1a")

    def test_parse_bare_json(self):
        result = parse_response('I think {"move": "1b"} is best.', self.legal)
        self.assertEqual(result.legal_action, "1b")

    def test_parse_case_insensitive(self):
        result = parse_response('```json\n{"move": "1A"}\n```', self.legal)
        self.assertEqual(result.legal_action, "1a")

    def test_parse_illegal_move_returns_raw(self):
        result = parse_response('```json\n{"move": "1c"}\n```', self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "1c")

    def test_prose_only_response_triggers_rethink(self):
        result = parse_response("I will ask Player 1 for aces (1a).", self.legal)
        self.assertIsNone(result.legal_action)
        self.assertIsNone(result.raw_action)

    def test_multiple_json_last_wins(self):
        response = 'First draft: {"move": "1a"}\nOn reflection: {"move": "1j"}'
        result = parse_response(response, self.legal)
        self.assertEqual(result.legal_action, "1j")

    def test_parse_returns_parse_result_type(self):
        result = parse_response('```json\n{"move": "1a"}\n```', self.legal)
        self.assertIsInstance(result, ParseResult)

    def test_uppercase_action_letter_matches_that_letter(self):
        # Matching is case-insensitive over the single action-letter namespace,
        # so an uppercased action letter resolves to that action letter.
        #   "1J" -> action letter j     "1K" -> action letter k
        # NOTE: this fixture deliberately includes BOTH 1j and 1k so the
        # colliding case is actually exercised (an earlier test dodged it by
        # omitting 1k). It documents the accepted residual edge: a model that
        # wrote "1K" meaning the card King (ask-letter 'm') is read as action
        # letter k (=Jack) here, because 1k is legal. The prompt only ever
        # teaches action letters, so this is a known trade-off, not a guarantee
        # of no collision.
        legal = ["1j", "1k", "1m"]
        self.assertEqual(parse_response('{"move": "1J"}', legal).legal_action, "1j")
        self.assertEqual(parse_response('{"move": "1K"}', legal).legal_action, "1k")

    def test_human_rank_label_not_accepted(self):
        # The parser accepts ONLY the action-letter namespace shown in the hand
        # lines. A human rank label that is not also an action letter must NOT
        # be silently reinterpreted -- it falls through to the rethink loop.
        #   "1K" with only 1m legal: 'k' is not legal, and King is NOT relabeled.
        result = parse_response('```json\n{"move": "1K"}\n```', ["1a", "1m"])
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "1K")
        # "1Q" -> 'q' is never a legal action letter; no label fallback either.
        result = parse_response('```json\n{"move": "1Q"}\n```', ["1a", "1l"])
        self.assertIsNone(result.legal_action)

    def test_numeric_rank_label_not_accepted(self):
        # "1, 10" is a human label ("10"), not an action letter -> no match,
        # deferred to the rethink loop rather than guessed at.
        result = parse_response('```json\n{"move": "1, 10"}\n```', self.legal)
        self.assertIsNone(result.legal_action)

    def test_separators_tolerated(self):
        result = parse_response('```json\n{"move": "1-a"}\n```', self.legal)
        self.assertEqual(result.legal_action, "1a")

    def test_illegal_json_does_not_ghost_substitute_from_prose(self):
        # The model discusses a legal move in prose, then commits to an
        # illegal one in JSON. The parser must NOT substitute the prose token.
        response = 'I considered 1a but ruled it out.\n```json\n{"move": "9z"}\n```'
        result = parse_response(response, self.legal)
        self.assertIsNone(result.legal_action)
        self.assertEqual(result.raw_action, "9z")


# ---------------------------------------------------------------------------
# generate_prompt
# ---------------------------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.game, self.state = _make_ask_state(seed=7)

    def test_basic_prompt_contents(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        self.assertIn("Go Fish", prompt)
        self.assertIn("book", prompt.lower())
        self.assertIn("go fish", prompt.lower())

    def test_go_fish_hit_continues_turn(self):
        # Engine (go_fish.cc:256-266): drawing the exact rank you asked for on
        # a "go fish" does NOT end your turn -- you ask again. The prompt must
        # not claim the turn simply ends.
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        self.assertIn("rank you asked for, you take another turn", prompt)

    def test_hit_does_not_always_grant_another_turn(self):
        # Engine (go_fish.cc CheckEmptyAsk :296-307): CheckEmptyAsk runs after
        # EVERY ask. On the !askee branch -- the asker still holds cards but the
        # hit stripped the last cards off every other player -- there is nobody
        # left to ask, so the engine sets kEmptyDraw and calls AdvancePlayer.
        # The turn passes on despite the hit. Distinct from the
        # PlayerCounts(current_player_) == 0 branch (the asker's own hand is
        # empty), which is already covered by the empty-hand rules.
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        flat = " ".join(prompt.split())
        self.assertIn("unless that leaves every other player with no cards", flat)

    def test_hit_turn_passes_matches_engine(self):
        # Drive the claim from the engine rather than trusting the sentence:
        # find a real hit that hands the turn away and assert the preconditions
        # the prompt describes (asker still holds cards, everyone else is empty).
        found = None
        for seed in range(80):
            game = go_fish_proxy.GoFishGame()
            state = game.new_initial_state()
            rng = random.Random(seed)
            while not state.is_terminal() and found is None:
                _advance_chance(state, rng)
                if state.is_terminal():
                    break
                asker = int(state.current_player())
                action = rng.choice(state.legal_actions())
                num_ranks = 13
                target, rank = action // num_ranks, action % num_ranks
                received = state.__wrapped__.player_cards()[target][rank]
                state.apply_action(action)
                if received <= 0 or state.is_terminal():
                    continue
                cards = [sum(row) for row in state.__wrapped__.player_cards()]
                probe = state.clone()
                _advance_chance(probe, random.Random(1))
                if probe.is_terminal() or int(probe.current_player()) == asker:
                    continue
                found = (asker, cards)
                break
            if found is not None:
                break
        self.assertIsNotNone(found, "no hit that passed the turn found in 80 games")
        asker, cards = found
        # The prompt's stated precondition: asker keeps cards, everyone else is
        # emptied. If this ever fails, the sentence describes the wrong branch.
        self.assertGreater(cards[asker], 0)
        self.assertTrue(all(n == 0 for p, n in enumerate(cards) if p != asker))

    def test_rank_legend_present(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        # Standard 13-rank deck: a=A ... m=K.
        self.assertIn("a=A", prompt)
        self.assertIn("j=10", prompt)
        self.assertIn("m=K", prompt)

    def test_hand_annotated_with_ask_letters(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        self.assertIn("ask-letter", prompt)
        # Player 0's opening hand (seed 7) contains an Ace.
        self.assertIn("A: ", prompt)

    def test_player_identity_rendered(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        self.assertIn("You are Player 0", prompt)

    def test_does_not_leak_opponent_hand(self):
        obs = _make_observation(self.state, self.game, player_id=0)
        prompt = generate_prompt(obs, [])
        # Opponent info is a card/book count only, never their cards. Match on
        # collapsed whitespace so the assertion survives template rewrapping.
        flat = " ".join(prompt.split())
        self.assertIn("You do NOT see other players' hands", flat)
        self.assertIn("you do NOT see which card anyone draws from the pool", flat)
        self.assertIn("Player 1:", prompt)

    def test_legal_moves_not_listed(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        # The prompt derives askable ranks from the hand; it should not dump
        # the raw legal-action list. The static "1a" format example is allowed
        # through even when it happens to also be legal this turn.
        legal = obs["legalActionStrings"]
        listed = [s for s in legal if s != "1a" and f'"{s}"' in prompt]
        self.assertEqual(listed, [])

    def test_pool_size_rendered(self):
        # Pool size is public and strategically central (it decides whether a
        # miss draws at all), but the raw text observation omits it entirely.
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        # Standard deck, 7 cards dealt to each of 2 players: 52 - 14 = 38.
        self.assertIn("Pool: 38 card(s) left to draw", prompt)

    def test_empty_pool_spells_out_the_consequence(self):
        # With an empty pool a miss draws nothing -- state the consequence
        # rather than leaving the model to infer it from a bare "0".
        self.assertIn("empty", _format_pool(0))
        self.assertIn("ends your turn", _format_pool(0))

    def test_pool_unknown_when_absent_from_payload(self):
        # A payload without pool_size (hand-built, or predating the field) must
        # render "(unknown)", never "empty" -- claiming an empty pool when the
        # pool is merely unreported would be an actively wrong game fact.
        self.assertEqual(_format_pool(None), "(unknown)")
        obs = _make_observation(self.state, self.game)
        stripped = dict(json.loads(obs["observationString"]))
        stripped.pop("pool_size", None)
        obs["observationString"] = json.dumps(stripped)
        prompt = generate_prompt(obs, [])
        self.assertIn("Pool: (unknown)", prompt)

    def test_booked_ranks_rendered(self):
        # Booked ranks are dead -- nobody can be asked for them.
        self.assertIn("(none yet)", _format_booked([]))
        self.assertIn("A, 9", _format_booked(["A", "9"]))
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        self.assertIn("Ranks already booked", prompt)

    def test_pool_size_tracks_the_game(self):
        # End-to-end: the rendered pool line must match the payload's pool_size
        # at every turn, including once the pool empties.
        rng = random.Random(3)
        game, state = _make_ask_state(seed=3)
        saw_empty = False
        for _ in range(80):
            _advance_chance(state, rng)
            if state.is_terminal():
                break
            cp = int(state.current_player())
            obs = _make_observation(state, game, player_id=cp)
            pool = json.loads(obs["observationString"])["pool_size"]
            prompt = generate_prompt(obs, [])
            if pool == 0:
                saw_empty = True
                self.assertIn("Pool: empty", prompt)
            else:
                self.assertIn(f"Pool: {pool} card(s) left to draw", prompt)
            state.apply_action(rng.choice(state.legal_actions()))
        self.assertTrue(saw_empty, "never reached an empty pool")

    def test_deduction_section_rendered(self):
        # The prompt must surface the game-long public deduction table so the
        # model retains ask history that has scrolled out of recent_events.
        game, state = _make_ask_state(seed=7, plies=8)
        obs = _make_observation(state, game)
        prompt = generate_prompt(obs, [])
        self.assertIn("What you know about opponents' cards", prompt)

    def test_deduction_signal_survives_events_truncation(self):
        # After many plies, recent_events only spans the last turn, but the
        # deduction block should still carry standing facts about the opponent.
        game, state = _make_ask_state(seed=11, plies=10)
        observer = int(state.current_player())
        obs = _make_observation(state, game, player_id=observer)
        prompt = generate_prompt(obs, [])
        section = prompt.split("What you know about opponents' cards")[1]
        section = section.split("Events since your last turn")[0]
        # Some concrete deduction was rendered (not the empty "(none)" body and
        # not merely "nothing deduced yet" for every opponent).
        self.assertTrue(
            "known to hold" in section or "known to have none of" in section or "has asked for" in section,
            f"no standing deduction rendered:\n{section}",
        )

    def test_deduction_header_covers_hit_created_voids(self):
        # go_fish.cc:234 sets player_min_[target][rank] = 0 on EVERY ask, so a
        # HIT empties the target of that rank just as a miss reveals they were
        # already empty. The header used to teach only the miss case ("a miss
        # reveals the target holds none of it"), leaving 84% of rendered
        # known_void entries unexplained -- a model reading it literally would
        # conclude its own successful asks create no void and re-ask the rank.
        obs = _make_observation(self.state, self.game)
        flat = " ".join(generate_prompt(obs, []).split())
        self.assertIn("after any ask the target holds none of it", flat)
        self.assertIn("whether they handed cards over or had none to give", flat)

    def test_hit_creates_known_void_matches_prompt_claim(self):
        # Drive the claim from the engine: find an ask that RECEIVED cards and
        # confirm the target is then rendered "known to have none of" that rank.
        # If this stops holding, the header sentence is the thing that is wrong.
        num_ranks = 13
        labels = [go_fish_proxy._rank_label(i, num_ranks) for i in range(num_ranks)]
        found = 0
        for seed in range(40):
            game = go_fish_proxy.GoFishGame()
            state = game.new_initial_state()
            rng = random.Random(seed)
            _advance_chance(state, rng)
            while not state.is_terminal() and found < 3:
                action = rng.choice(state.legal_actions())
                target, rank = action // num_ranks, action % num_ranks
                received = state.__wrapped__.player_cards()[target][rank]
                state.apply_action(action)
                _advance_chance(state, rng)
                if state.is_terminal() or received <= 0:
                    continue
                observer = int(state.current_player())
                if observer == target:
                    continue  # a player never gets a deduction row for themselves
                row = [d for d in state.state_dict(observer)["deductions"] if d["player"] == target][0]
                if labels[rank] in row["known_void"]:
                    found += 1
            if found >= 3:
                break
        self.assertGreaterEqual(found, 3, "no hit ever produced a known_void entry")

    def test_wanted_filtered_against_known_has(self):
        # "known to hold 9>=3; has asked for 9" repeats the rank the first
        # clause already gave a count for. ~93% of emitted wanted entries are
        # this, on the densest line in the prompt.
        rendered = _format_deductions(
            [
                {
                    "player": 1,
                    "known_has": ["4>=1", "7>=2", "9>=3", "J>=2", "Q>=2"],
                    "known_void": [],
                    "wanted": ["A", "4", "5", "7", "9", "J", "Q"],
                }
            ],
            player_id=0,
        )
        # Only the two ranks that known_has does not already cover survive.
        self.assertIn("has asked for A, 5", rendered)
        self.assertIn("known to hold 4>=1, 7>=2, 9>=3, J>=2, Q>=2", rendered)

    def test_wanted_clause_dropped_when_fully_redundant(self):
        # If every wanted rank is already in known_has the clause carries no
        # information at all and must not be rendered.
        rendered = _format_deductions(
            [{"player": 1, "known_has": ["9>=3"], "known_void": [], "wanted": ["9"]}],
            player_id=0,
        )
        self.assertNotIn("has asked for", rendered)
        self.assertIn("known to hold 9>=3", rendered)

    def test_surviving_wanted_entries_mean_emptied_but_drew_since(self):
        # The rendered gloss asserts a specific meaning, so pin it to the
        # engine: a wanted rank absent from known_has can only be one the player
        # was asked for and emptied of, then drew after. Asking sets player_min
        # >= 1, so a zero floor alongside a past ask implies they were emptied;
        # the proxy drops the entry as known_void unless drawn_since > 0.
        checked = 0
        for params in ({"ranks": 13, "suits": 4}, {"ranks": 13, "suits": 4, "players": 3}):
            num_ranks = params["ranks"]
            labels = [go_fish_proxy._rank_label(i, num_ranks) for i in range(num_ranks)]
            game = go_fish_proxy.GoFishGame(params)
            state = game.new_initial_state()
            rng = random.Random(4)
            while not state.is_terminal():
                _advance_chance(state, rng)
                if state.is_terminal():
                    break
                raw = state.__wrapped__
                was_asked, drawn_since = raw.player_was_asked(), raw.drawn_since_was_asked()
                for d in state.state_dict(0)["deductions"]:
                    pid = d["player"]
                    has = {h.split(">=", 1)[0] for h in d["known_has"]}
                    for label in (w for w in d["wanted"] if w not in has):
                        rank = labels.index(label)
                        self.assertTrue(was_asked[pid][rank], f"P{pid} {label}: never asked back")
                        self.assertGreater(drawn_since[pid][rank], 0, f"P{pid} {label}: no draw since")
                        checked += 1
                state.apply_action(rng.choice(state.legal_actions()))
        self.assertGreater(checked, 50, "too few residual entries to be meaningful")

    def test_events_rendered_after_play(self):
        # A mid-game state produces opponent events since the last turn.
        game, state = _make_ask_state(seed=7, plies=3)
        obs = _make_observation(state, game)
        prompt = generate_prompt(obs, [])
        self.assertIn("Events since your last turn", prompt)

    def test_draw_event_does_not_leak_drawn_rank(self):
        # A card drawn from the pool is HIDDEN information: naming its rank
        # would leak an opponent's hand, contradicting the prompt's "you do NOT
        # see other players' hands" and mooting the deduction block. The draw
        # line must report only that a draw happened, never which rank.
        events = [{"type": "draw", "player": 1, "rank_label": "K", "booked": False}]
        rendered = _format_events(events)
        self.assertIn("Player 1 drew a card from the pool", rendered)
        self.assertNotIn("K", rendered)

    def test_draw_that_completes_book_names_only_the_book(self):
        # A laid-down book IS public, so the completed-book clause may name the
        # rank -- that is the one case where the rank is legitimately revealed.
        events = [{"type": "draw", "player": 1, "rank_label": "9", "booked": True}]
        rendered = _format_events(events)
        self.assertIn("completed a book of 9", rendered)

    def test_full_game_prompts_never_leak_a_drawn_rank(self):
        # End-to-end guard: across a played-out game, no draw event in any
        # prompt should ever spell out the drawn rank. This is the assertion
        # the original leak silently failed -- 41% of rendered events named it.
        rng = random.Random(3)
        game, state = _make_ask_state(seed=3)
        for _ in range(40):
            _advance_chance(state, rng)
            if state.is_terminal():
                break
            cp = int(state.current_player())
            obs = _make_observation(state, game, player_id=cp)
            prompt = generate_prompt(obs, [])
            for line in prompt.splitlines():
                if "drew" in line and "book" not in line:
                    # A pure draw line: must not carry a rank token.
                    self.assertEqual(
                        line.strip(),
                        f"Player {line.split()[1]} drew a card from the pool",
                        f"draw line leaked a rank: {line!r}",
                    )
            state.apply_action(rng.choice(state.legal_actions()))

    def test_move_history_rendered(self):
        # With a fresh state there are no own-asks to reconstruct outcomes for,
        # so the moves render bare (the annotator falls back gracefully).
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, ["1a", "1b"])
        self.assertIn("1a, 1b", prompt)

    def test_move_history_none_when_empty(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        # The move-history section ends in "None" when no moves have been played.
        self.assertRegex(prompt, r"they had none\): None")

    def test_own_ask_outcomes_annotated(self):
        # The model is never shown the result of its OWN asks (OpenSpiel stops
        # the event walk at the observer's last action). generate_prompt must
        # reconstruct each own-ask outcome from the replayed history and attach
        # it to the move string: "1x (received N)" on a hit, "1x (go fish)" on
        # a miss.
        rng = random.Random(7)
        game, state = _make_ask_state(seed=7)
        observer = 0
        own_moves: list[str] = []
        prompt = ""
        for _ in range(30):
            _advance_chance(state, rng)
            if state.is_terminal():
                break
            cp = int(state.current_player())
            if cp == observer:
                obs = _make_observation(state, game, player_id=observer)
                prompt = generate_prompt(obs, list(own_moves))
                action = state.legal_actions()[0]
                own_moves.append(state.action_to_string(action))
                state.apply_action(action)
            else:
                state.apply_action(state.legal_actions()[0])
        # After several own asks, the latest prompt must annotate them with a
        # concrete outcome -- at least one hit or one "go fish".
        self.assertTrue(own_moves, "no own moves were played")
        self.assertTrue(
            "(received " in prompt or "(go fish)" in prompt,
            f"own-ask outcomes not annotated in move history:\n{prompt}",
        )

    def test_own_ask_hit_reports_received_count(self):
        # A hit annotates with the exact number of cards received. Seed 7,
        # observer 0's opening ask "1a" is a hit for 1 card (verified against
        # the engine), so the second prompt must show "1a (received 1)".
        rng = random.Random(7)
        game, state = _make_ask_state(seed=7)
        observer = 0
        # First own turn: play 1a (Ace ask -> hit for 1 under seed 7).
        _advance_chance(state, rng)
        self.assertEqual(int(state.current_player()), observer)
        first = state.action_to_string(state.legal_actions()[0])
        self.assertEqual(first, "1a")
        state.apply_action(state.legal_actions()[0])
        _advance_chance(state, rng)
        # Advance to the observer's next decision point.
        while not state.is_terminal() and int(state.current_player()) != observer:
            state.apply_action(state.legal_actions()[0])
            _advance_chance(state, rng)
        obs = _make_observation(state, game, player_id=observer)
        prompt = generate_prompt(obs, [first])
        self.assertIn("1a (received 1)", prompt)

    def test_own_ask_outcomes_survive_proxy_serialization(self):
        # Regression: the outcome reconstruction replays the state recovered
        # from serializedGameAndState, and production serializes the *proxy*
        # game -- so the replayed observation_string is the proxy's JSON, not
        # OpenSpiel's raw text. A text-only reader returned (0, 0) for every
        # card/book count, making every delta zero and mislabelling every hit
        # as "go fish". Pin both encodings to the same annotation.
        rng = random.Random(7)
        game, state = _make_ask_state(seed=7)
        observer = 0
        _advance_chance(state, rng)
        first = state.action_to_string(state.legal_actions()[0])
        state.apply_action(state.legal_actions()[0])
        _advance_chance(state, rng)
        while not state.is_terminal() and int(state.current_player()) != observer:
            state.apply_action(state.legal_actions()[0])
            _advance_chance(state, rng)

        proxy_ser = pyspiel.serialize_game_and_state(game, state)
        raw_ser = pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__)
        # Precondition: the two really do round-trip to different observation
        # encodings, so this test can't pass vacuously.
        self.assertIn("go_fish_proxy", proxy_ser)
        self.assertNotIn("go_fish_proxy", raw_ser)

        annotated = {}
        for name, serialized in (("proxy", proxy_ser), ("raw", raw_ser)):
            obs = {"serializedGameAndState": serialized, "playerId": observer}
            annotated[name] = _annotate_move_history(obs, [first], observer, 4)
        self.assertEqual(annotated["proxy"], [f"{first} (received 1)"])
        self.assertEqual(annotated["proxy"], annotated["raw"])

    def test_own_counts_reads_both_observation_encodings(self):
        game, state = _make_ask_state(seed=7)
        proxy_text = state.observation_string(0)
        raw_text = state.__wrapped__.observation_string(0)
        self.assertNotEqual(proxy_text, raw_text)
        self.assertEqual(_own_counts(proxy_text, 0), _own_counts(raw_text, 0))
        # Non-zero, so a (0, 0) default can't masquerade as a correct read.
        self.assertGreater(_own_counts(proxy_text, 0)[0], 0)

    def test_annotations_anchor_from_the_end(self):
        # move_history is a SUFFIX of the observer's asks, not necessarily the
        # whole run: create_agent_fn's move_history closure starts empty, so an
        # agent process that restarts mid-episode passes only its own asks while
        # the engine state is mid-game. A positional zip from index 0 would pair
        # move_history[0] with the observer's FIRST ask of the game, inverting
        # every annotation with no signal that anything is off.
        game, state, own_asks = _play_own_asks(seed=3, plies=40)
        self.assertGreaterEqual(len(own_asks), 4)
        obs = _make_observation(state, game)

        full = _annotate_move_history(obs, own_asks, 0, 4)
        self.assertIsNotNone(full)
        for tail_len in (1, 2, 3):
            suffix = _annotate_move_history(obs, own_asks[-tail_len:], 0, 4)
            self.assertEqual(suffix, full[-tail_len:], f"tail of {tail_len} misaligned")

        # Guard against passing vacuously: seed 3 must actually be a case where
        # the naive index-0 zip disagrees, otherwise this test proves nothing.
        naive = [f"{mv} ({full[i].split(' (', 1)[1][:-1]})" for i, mv in enumerate(own_asks[-2:])]
        self.assertNotEqual(naive, full[-2:])

    def test_unalignable_history_falls_back_to_bare(self):
        # More moves than the replay accounts for (e.g. a rethink appended a
        # move the engine never applied): the sequences can't be anchored at
        # either end, so return None and let the caller show the bare history
        # rather than emit a guess.
        game, state, own_asks = _play_own_asks(seed=3, plies=40)
        obs = _make_observation(state, game)
        too_many = list(own_asks) + ["1a", "1b"]
        self.assertIsNone(_annotate_move_history(obs, too_many, 0, 4))
        # And generate_prompt degrades to the bare moves, not a wrong annotation.
        prompt = generate_prompt(obs, too_many)
        self.assertIn(", ".join(too_many), prompt)

    def test_example_move_is_static(self):
        # The format example uses a fixed rank letter "a". It must NOT be
        # derived from the legal-action list: a per-turn example is read as a
        # per-turn *suggestion*, and deriving it from legalActions always
        # surfaced the lowest-index rank in hand, nudging the model toward that
        # ask every turn.
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        concrete = _example_tokens(prompt)
        self.assertEqual(concrete, ["1a"])
        self.assertIn("not a suggestion", prompt)

    def test_example_move_never_a_self_ask(self):
        # Only the example's TARGET varies, and it must never be the observer:
        # GenerateAsks skips target == player_id (go_fish.cc), so a self-ask can
        # never be legal. A hardcoded "1a" was structurally impossible for
        # Player 1 -- roughly half of all turns in a 2-player game.
        for players in (2, 3, 4):
            game = go_fish_proxy.GoFishGame({"players": players})
            state = game.new_initial_state()
            _advance_chance(state, random.Random(4))
            for pid in range(players):
                obs = _make_observation(state, game, player_id=pid)
                for prompt in (
                    generate_prompt(obs, []),
                    # The rethink template carries its own copy of the example.
                    generate_prompt(obs, [], previous_response="junk", previous_action=None),
                ):
                    for token in _example_tokens(prompt):
                        self.assertNotEqual(
                            token[0],
                            str(pid),
                            f"{players}p player {pid}: example {token!r} is a self-ask",
                        )
                        self.assertEqual(token[1:], "a", f"example letter drifted: {token!r}")

    def test_example_move_static_across_turns(self):
        # Same example on every turn for a given player, regardless of hand or
        # legal actions -- including turns where the example is not legal.
        rng = random.Random(5)
        game, state = _make_ask_state(seed=5)
        saw_example_illegal = False
        for _ in range(40):
            _advance_chance(state, rng)
            if state.is_terminal():
                break
            cp = int(state.current_player())
            obs = _make_observation(state, game, player_id=cp)
            prompt = generate_prompt(obs, [])
            expected = "1a" if cp != 1 else "0a"
            self.assertEqual(_example_tokens(prompt), [expected])
            if expected not in obs["legalActionStrings"]:
                saw_example_illegal = True
            state.apply_action(rng.choice(state.legal_actions()))
        # The invariant is only meaningful if some turn had the example illegal.
        self.assertTrue(saw_example_illegal, "never hit a turn where the example was illegal")

    def test_rethink_illegal_suffix(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [], previous_response="I'll play 9z", previous_action="9z")
        self.assertIn("You suggested", prompt)
        self.assertIn("9z", prompt)
        self.assertIn("not a legal move", prompt)

    def test_rethink_unparsable_suffix(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [], previous_response="hmm no json here", previous_action=None)
        self.assertIn("No JSON answer could be parsed", prompt)
        self.assertIn("hmm no json here", prompt)

    def test_no_rethink_on_first_attempt(self):
        obs = _make_observation(self.state, self.game)
        prompt = generate_prompt(obs, [])
        self.assertNotIn("not a legal move", prompt)
        self.assertNotIn("No JSON answer could be parsed", prompt)


# ---------------------------------------------------------------------------
# get_legal_moves
# ---------------------------------------------------------------------------


class GetLegalMovesTest(absltest.TestCase):
    def test_from_provided_actions(self):
        obs = {
            "legalActions": [13, 14, 17],
            "legalActionStrings": ["1a", "1b", "1e"],
        }
        result = get_legal_moves(obs)
        self.assertEqual(result, {13: "1a", 14: "1b", 17: "1e"})

    def test_from_serialized_state(self):
        game, state = _make_ask_state(seed=7)
        obs = {
            "playerId": int(state.current_player()),
            "serializedGameAndState": pyspiel.serialize_game_and_state(game.__wrapped__, state.__wrapped__),
        }
        result = get_legal_moves(obs)
        self.assertGreater(len(result), 0)
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, str)
            # Go Fish ask strings are two chars: <target><letter>.
            self.assertEqual(len(v), 2)

    def test_empty_serialized(self):
        self.assertEqual(get_legal_moves({"serializedGameAndState": ""}), {})


# ---------------------------------------------------------------------------
# create_agent_fn integration
# ---------------------------------------------------------------------------


class _GoFishHarness:
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
        agent = create_agent_fn(_GoFishHarness())

        result = agent({"step": 0, "remainingOverageTime": 60}, {})

        self.assertIsNone(result["submission"])
        self.assertEqual(result["status"], "INACTIVE")
        mock_litellm.completion.assert_not_called()

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_successful_move(self, mock_litellm):
        mock_litellm.drop_params = True
        game, state = _make_ask_state(seed=7)
        first_legal = state.action_to_string(state.legal_actions()[0])
        mock_litellm.completion.return_value = _make_mock_response(f'```json\n{{"move": "{first_legal}"}}\n```')
        agent = create_agent_fn(_GoFishHarness())

        obs = _make_observation(state, game)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal)
        self.assertEqual(result["status"], "OK")
        self.assertIn("thoughts", result)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_retry_on_bad_parse(self, mock_litellm):
        mock_litellm.drop_params = True
        game, state = _make_ask_state(seed=7)
        first_legal = state.action_to_string(state.legal_actions()[0])
        mock_litellm.completion.side_effect = [
            _make_mock_response('```json\n{"move": "9z"}\n```'),
            _make_mock_response(f'```json\n{{"move": "{first_legal}"}}\n```'),
        ]
        agent = create_agent_fn(_GoFishHarness())

        obs = _make_observation(state, game)
        result = agent(obs, {})

        self.assertEqual(result["actionString"], first_legal)
        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_raises_after_two_failures(self, mock_litellm):
        mock_litellm.drop_params = True
        mock_litellm.completion.return_value = _make_mock_response("I cannot decide.")
        agent = create_agent_fn(_GoFishHarness())

        game, state = _make_ask_state(seed=7)
        obs = _make_observation(state, game)

        with self.assertRaises(ValueError):
            agent(obs, {})

        self.assertEqual(mock_litellm.completion.call_count, 2)

    @patch.dict("os.environ", _ENV)
    @patch("kaggle_environments.core_harness.litellm")
    def test_short_game_via_agent_fns(self, mock_litellm):
        """Drive a Go Fish game with scripted agents that always pick their
        first legal ask, verifying the harness round-trips through pyspiel."""
        mock_litellm.drop_params = True

        game, state = _make_ask_state(seed=7)
        rng = random.Random(123)

        def fake_completion(*, model, messages, **kwargs):
            del model, kwargs
            first = state.action_to_string(state.legal_actions()[0])
            return _make_mock_response(f'```json\n{{"move": "{first}"}}\n```')

        mock_litellm.completion.side_effect = fake_completion
        agent_p0 = create_agent_fn(_GoFishHarness())
        agent_p1 = create_agent_fn(_GoFishHarness())

        for _ in range(30):
            _advance_chance(state, rng)
            if state.is_terminal():
                break
            cp = int(state.current_player())
            agent = agent_p0 if cp == 0 else agent_p1
            obs = _make_observation(state, game, player_id=cp)
            result = agent(obs, {})
            self.assertEqual(result["status"], "OK")
            state.apply_action(result["submission"])

        self.assertGreater(state.move_number(), 0)


if __name__ == "__main__":
    absltest.main()
