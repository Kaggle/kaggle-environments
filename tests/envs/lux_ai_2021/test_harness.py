"""Tests for the Lux AI 2021 harness."""

from __future__ import annotations

import json

from absl.testing import absltest

from kaggle_environments.envs.lux_ai_2021.harness import (
    _INVALID_COMMAND_THRESHOLD,
    _normalize_command,
    _phase,
    _rebuild_game,
    _render_ascii_map,
    _validate_command,
    generate_prompt,
    get_legal_moves,
    parse_response,
)

# --- Fixtures ---------------------------------------------------------------


def _turn_updates(
    width: int = 12,
    *,
    p0_units: list[tuple[str, int, int]] | None = None,
    p1_units: list[tuple[str, int, int]] | None = None,
    p0_cities: list[tuple[str, list[tuple[int, int]]]] | None = None,
    resources: list[tuple[str, int, int, int]] | None = None,
    p0_rp: int = 0,
    p1_rp: int = 0,
) -> list[str]:
    """Build a minimal turn-frame update list.

    Unit tuples: ``(unit_id, x, y)`` -- always workers, cooldown 0, empty cargo.
    City tuples: ``(city_id, [(x, y), ...])`` -- fuel 100, upkeep 5.
    Resource tuples: ``(type, x, y, amount)``.
    """
    lines = [f"rp 0 {p0_rp}", f"rp 1 {p1_rp}"]
    for r_type, x, y, amt in resources or []:
        lines.append(f"r {r_type} {x} {y} {amt}")
    for uid, x, y in p0_units or []:
        # unit type 0 = worker; team 0
        lines.append(f"u 0 0 {uid} {x} {y} 0 0 0 0")
    for uid, x, y in p1_units or []:
        lines.append(f"u 0 1 {uid} {x} {y} 0 0 0 0")
    for cid, tiles in p0_cities or []:
        lines.append(f"c 0 {cid} 100 5")
        for x, y in tiles:
            lines.append(f"ct 0 {cid} {x} {y} 0")
    return lines


def _observation(
    step: int = 2,
    player: int = 0,
    width: int = 12,
    updates: list[str] | None = None,
) -> dict:
    """Build a turn-shaped observation. ``step`` defaults to 2 so
    ``_rebuild_game`` computes ``game.turn == 0`` (the first playable turn).
    Pass ``step=0`` with header-prefixed ``updates`` to exercise the
    step-0 stripping branch.
    """
    if updates is None:
        updates = _turn_updates(width=width, p0_units=[("u_1", 3, 4)])
    return {
        "player": player,
        "step": step,
        "width": width,
        "height": width,
        "updates": updates,
        "isTerminal": False,
    }


# --- get_legal_moves --------------------------------------------------------


class GetLegalMovesTest(absltest.TestCase):
    def test_always_none(self):
        self.assertIsNone(get_legal_moves(_observation()))
        self.assertIsNone(get_legal_moves({}))


# --- _validate_command ------------------------------------------------------


class ValidateCommandTest(absltest.TestCase):
    def test_move_directions(self):
        for d in ("n", "s", "e", "w", "c"):
            self.assertTrue(_validate_command(f"m u_1 {d}"))

    def test_move_bad_direction(self):
        self.assertFalse(_validate_command("m u_1 north"))
        self.assertFalse(_validate_command("m u_1 x"))

    def test_move_bad_unit_id(self):
        self.assertFalse(_validate_command("m unit_1 n"))
        self.assertFalse(_validate_command("m u1 n"))
        self.assertFalse(_validate_command("m u_ n"))

    def test_bcity(self):
        self.assertTrue(_validate_command("bcity u_2"))
        self.assertFalse(_validate_command("bcity"))
        self.assertFalse(_validate_command("bcity 5 5"))

    def test_pillage(self):
        self.assertTrue(_validate_command("p u_1"))
        self.assertFalse(_validate_command("p 3 5"))

    def test_transfer(self):
        self.assertTrue(_validate_command("t u_1 u_2 wood 100"))
        self.assertTrue(_validate_command("t u_1 u_2 coal 0"))
        self.assertTrue(_validate_command("t u_1 u_2 uranium 500"))
        self.assertFalse(_validate_command("t u_1 u_2 gold 100"))
        self.assertFalse(_validate_command("t u_1 u_2 wood -1"))
        self.assertFalse(_validate_command("t u_1 wood 100"))

    def test_city_tile_commands(self):
        self.assertTrue(_validate_command("r 3 5"))
        self.assertTrue(_validate_command("bw 0 0"))
        self.assertTrue(_validate_command("bc 11 11"))
        self.assertFalse(_validate_command("r 3"))
        self.assertFalse(_validate_command("r 3 5 7"))

    def test_empty_and_junk(self):
        self.assertFalse(_validate_command(""))
        self.assertFalse(_validate_command("do stuff"))
        self.assertFalse(_validate_command("annotate 1 2"))  # dc/annotations off

    def test_trailing_whitespace_rejected(self):
        # Fullmatch: no leading/trailing whitespace allowed.
        self.assertFalse(_validate_command("m u_1 n "))
        self.assertFalse(_validate_command(" m u_1 n"))


# --- _normalize_command -----------------------------------------------------


class NormalizeCommandTest(absltest.TestCase):
    def test_passthrough(self):
        self.assertEqual(_normalize_command("m u_1 n"), "m u_1 n")
        self.assertEqual(_normalize_command("bcity u_2"), "bcity u_2")

    def test_uppercase_direction(self):
        self.assertEqual(_normalize_command("m u_1 N"), "m u_1 n")
        self.assertEqual(_normalize_command("M u_1 South"), "m u_1 s")

    def test_full_direction_names(self):
        for full, short in [("north", "n"), ("south", "s"), ("east", "e"), ("west", "w"), ("center", "c")]:
            self.assertEqual(_normalize_command(f"m u_1 {full}"), f"m u_1 {short}")

    def test_extra_internal_whitespace_collapsed(self):
        self.assertEqual(_normalize_command("m  u_1  n"), "m u_1 n")
        self.assertEqual(_normalize_command("t  u_1  u_2  wood  100"), "t u_1 u_2 wood 100")

    def test_leading_bullets_stripped(self):
        self.assertEqual(_normalize_command("- m u_1 n"), "m u_1 n")
        self.assertEqual(_normalize_command("* bcity u_2"), "bcity u_2")
        self.assertEqual(_normalize_command("1. m u_1 n"), "m u_1 n")
        self.assertEqual(_normalize_command("2) bcity u_2"), "bcity u_2")

    def test_trailing_punctuation_stripped(self):
        self.assertEqual(_normalize_command("m u_1 n."), "m u_1 n")
        self.assertEqual(_normalize_command("bcity u_2,"), "bcity u_2")
        self.assertEqual(_normalize_command("r 5 7;"), "r 5 7")

    def test_normalization_composes_with_validator(self):
        # After normalization every one of these must be a valid command.
        for raw in [
            "M u_1 N",
            "  m u_1 n  ",
            "- m u_1 north",
            "1. bcity u_2.",
            "t  u_1  u_2  wood  50",
        ]:
            normalized = _normalize_command(raw)
            self.assertTrue(_validate_command(normalized), f"{raw!r} -> {normalized!r}")

    def test_unit_id_case_folded(self):
        # The engine's unit ids are exclusively lowercase (u_N / c_N); models
        # frequently uppercase them to match the ASCII map letters. Fold so
        # the validator accepts.
        self.assertEqual(_normalize_command("m U_1 n"), "m u_1 n")
        self.assertTrue(_validate_command("m u_1 n"))
        self.assertEqual(_normalize_command("bcity U_2"), "bcity u_2")
        self.assertEqual(_normalize_command("t U_1 U_2 wood 40"), "t u_1 u_2 wood 40")
        # Non-id tokens are untouched (e.g. resource keyword case is
        # already handled elsewhere; the fold is scoped to [uUcC]_\\d+).
        self.assertEqual(_normalize_command("m U_1 N"), "m u_1 n")

    def test_missing_underscore_in_id_inserted(self):
        # Models routinely drop the underscore in unit/city ids (``u9`` for
        # ``u_9``), often to match the bare numbers on the ASCII map. The
        # engine's wire form requires the underscore, so normalization must
        # insert it or the command is silently dropped.
        self.assertEqual(_normalize_command("m u9 w"), "m u_9 w")
        self.assertEqual(_normalize_command("bcity u12"), "bcity u_12")
        self.assertEqual(_normalize_command("p u3"), "p u_3")
        self.assertEqual(_normalize_command("t u12 u1 wood 45"), "t u_12 u_1 wood 45")
        # Case + missing underscore together fold in one pass.
        self.assertEqual(_normalize_command("M U9 N"), "m u_9 n")
        # After folding, each is a valid command.
        for raw in ["m u9 w", "bcity u12", "p u3", "t u12 u1 wood 45", "M U9 N"]:
            self.assertTrue(_validate_command(_normalize_command(raw)), raw)

    def test_missing_underscore_does_not_touch_directions_or_coords(self):
        # The ``c`` (stay) direction is a bare letter with no digits, and city
        # commands take coordinate integers -- neither is an id, so neither
        # must be rewritten into ``c_N``/``u_N``.
        self.assertEqual(_normalize_command("m u_1 c"), "m u_1 c")
        self.assertEqual(_normalize_command("m u1 center"), "m u_1 c")
        self.assertEqual(_normalize_command("r 5 7"), "r 5 7")
        self.assertEqual(_normalize_command("bw 0 6"), "bw 0 6")

    def test_strips_bracket_wrappers(self):
        for raw in ["m [u_1] n", "bcity <u_2>", "t (u_1) (u_2) wood 40"]:
            normalized = _normalize_command(raw)
            self.assertTrue(_validate_command(normalized), f"{raw!r} -> {normalized!r}")

    def test_strips_trailing_line_comments(self):
        self.assertEqual(_normalize_command("m u_1 n # go north"), "m u_1 n")
        self.assertEqual(_normalize_command("bcity u_2 // build"), "bcity u_2")
        self.assertEqual(_normalize_command("r 5 7  #  research"), "r 5 7")

    def test_crystal_rewritten_to_uranium_in_transfer(self):
        # The prompt uses "crystal" for the third resource type; the engine's
        # wire keyword is "uranium". Normalization must rewrite it.
        self.assertEqual(
            _normalize_command("t u_1 u_2 crystal 100"),
            "t u_1 u_2 uranium 100",
        )
        # Uppercase variant folds too.
        self.assertEqual(
            _normalize_command("T u_1 u_2 Crystal 100"),
            "t u_1 u_2 uranium 100",
        )

    def test_uppercase_command_head_folded(self):
        # All command heads are lowercase in the engine grammar. Models
        # frequently emit uppercase or title-case variants; normalization
        # should fold the head and let the validator pass.
        for raw, want in [
            ("BCITY u_1", "bcity u_1"),
            ("Bcity u_2", "bcity u_2"),
            ("R 5 7", "r 5 7"),
            ("BW 0 0", "bw 0 0"),
            ("BC 3 4", "bc 3 4"),
            ("P u_1", "p u_1"),
        ]:
            normalized = _normalize_command(raw)
            self.assertEqual(normalized, want, f"{raw!r} -> {normalized!r}")
            self.assertTrue(_validate_command(normalized))

    def test_resource_case_folded_in_transfer(self):
        # Resource names are lowercase in the engine grammar. Fold uppercase
        # and title-case variants so the validator accepts them.
        for raw, want in [
            ("t u_1 u_2 Wood 50", "t u_1 u_2 wood 50"),
            ("t u_1 u_2 COAL 5", "t u_1 u_2 coal 5"),
            ("t u_1 u_2 URANIUM 50", "t u_1 u_2 uranium 50"),
            ("t u_1 u_2 CRYSTAL 50", "t u_1 u_2 uranium 50"),
        ]:
            normalized = _normalize_command(raw)
            self.assertEqual(normalized, want, f"{raw!r} -> {normalized!r}")
            self.assertTrue(_validate_command(normalized))

    def test_uranium_still_accepted(self):
        # If the model uses the wire keyword directly, don't break it.
        self.assertEqual(
            _normalize_command("t u_1 u_2 uranium 100"),
            "t u_1 u_2 uranium 100",
        )


# --- parse_response ---------------------------------------------------------


class ParseResponseTest(absltest.TestCase):
    def _parse(self, response: str):
        return parse_response(response, None)

    def test_all_valid(self):
        payload = {"actions": ["m u_1 n", "bcity u_2", "r 5 7"]}
        result = self._parse(f"```json\n{json.dumps(payload)}\n```")
        self.assertEqual(result.submission, ["m u_1 n", "bcity u_2", "r 5 7"])

    def test_empty_list_is_legal(self):
        result = self._parse('```json\n{"actions": []}\n```')
        self.assertEqual(result.submission, [])
        self.assertEqual(result.raw_action, "[]")

    def test_mixed_below_threshold_drops_invalid(self):
        # 3 valid, 1 invalid = 25% invalid, below 50% threshold → submit valid.
        payload = {
            "actions": ["m u_1 n", "m u_2 s", "bcity u_3", "junk"],
        }
        result = self._parse(f"```json\n{json.dumps(payload)}\n```")
        self.assertEqual(result.submission, ["m u_1 n", "m u_2 s", "bcity u_3"])

    def test_mixed_above_threshold_triggers_rethink(self):
        # 1 valid, 3 invalid = 75% invalid → whole turn rejected.
        payload = {"actions": ["m u_1 n", "junk1", "junk2", "junk3"]}
        result = self._parse(f"```json\n{json.dumps(payload)}\n```")
        self.assertIsNone(result.submission)
        self.assertIsNotNone(result.raw_action)

    def test_exactly_at_threshold_submits(self):
        # 2 valid, 2 invalid = 50% invalid, at threshold (>0.5 rejects) → submit.
        payload = {"actions": ["m u_1 n", "m u_2 s", "junk1", "junk2"]}
        result = self._parse(f"```json\n{json.dumps(payload)}\n```")
        self.assertEqual(result.submission, ["m u_1 n", "m u_2 s"])
        # Sanity-check the constant so this test stays meaningful.
        self.assertEqual(_INVALID_COMMAND_THRESHOLD, 0.5)

    def test_all_invalid_triggers_rethink(self):
        payload = {"actions": ["junk1", "junk2"]}
        result = self._parse(f"```json\n{json.dumps(payload)}\n```")
        self.assertIsNone(result.submission)

    def test_no_json_returns_unparsable(self):
        result = self._parse("I have no idea what to do here.")
        self.assertIsNone(result.submission)
        self.assertIsNone(result.raw_action)

    def test_json_without_actions_key_returns_unparsable(self):
        result = self._parse('```json\n{"move": "north"}\n```')
        self.assertIsNone(result.submission)
        self.assertIsNone(result.raw_action)

    def test_actions_not_a_list_routes_to_unparseable(self):
        # Scalar / null actions is a structural failure ("send an array"),
        # not a command-level failure ("fix your commands"). Route to the
        # UNPARSABLE rethink by returning raw_action=None so core_harness
        # picks that template.
        for payload in ['{"actions": "m u_1 n"}', '{"actions": null}', '{"actions": {"m": "u_1"}}']:
            result = self._parse(f"```json\n{payload}\n```")
            self.assertIsNone(result.submission, payload)
            self.assertIsNone(result.raw_action, payload)

    def test_normalizes_uppercase_directions(self):
        # Model uses full-word directions; parser should still accept them.
        payload = {"actions": ["m u_1 North", "m u_2 SOUTH"]}
        result = self._parse(f"```json\n{json.dumps(payload)}\n```")
        self.assertEqual(result.submission, ["m u_1 n", "m u_2 s"])

    def test_normalizes_extra_whitespace(self):
        payload = {"actions": ["m  u_1  n", "bcity  u_2"]}
        result = self._parse(f"```json\n{json.dumps(payload)}\n```")
        self.assertEqual(result.submission, ["m u_1 n", "bcity u_2"])

    def test_crystal_in_transfer_rewritten_to_uranium(self):
        # End-to-end: the model uses "crystal" (matching the prompt), the
        # submission must go to the engine as "uranium" (the wire keyword).
        payload = {"actions": ["t u_1 u_2 crystal 40"]}
        result = self._parse(f"```json\n{json.dumps(payload)}\n```")
        self.assertEqual(result.submission, ["t u_1 u_2 uranium 40"])

    def test_last_json_wins(self):
        # Model writes a draft then revises. Last one wins.
        response = 'Draft: ```json\n{"actions": ["junk"]}\n```\nFinal: ```json\n{"actions": ["m u_1 n"]}\n```'
        result = self._parse(response)
        self.assertEqual(result.submission, ["m u_1 n"])


# --- _rebuild_game & _render_ascii_map --------------------------------------


class RebuildGameTest(absltest.TestCase):
    def test_step_0_strips_header(self):
        # Step 0's engine frame carries `[game_id, "W H", ...]`; the harness
        # must drop those before feeding `_update`, or the first two data
        # lines (research points) get silently swallowed.
        header = ["0", "8 8"]
        body = _turn_updates(width=8, p0_units=[("u_1", 2, 2)], p0_rp=7)
        obs = _observation(step=0, width=8, updates=header + body)
        game = _rebuild_game(obs)
        self.assertEqual(game.map.width, 8)
        self.assertEqual(game.map.height, 8)
        self.assertEqual(len(game.players[0].units), 1)
        # Research points must survive -- this is the regression from the
        # header-stripping bug where turn==0 dropped the first two lines.
        self.assertEqual(game.players[0].research_points, 7)

    def test_step_1_no_header_stripped(self):
        # Step 1 is the first turn-shaped frame -- no header, so the
        # research-point lines at the start of `updates` must be preserved.
        updates = _turn_updates(width=6, p0_units=[("u_1", 1, 1)], p0_rp=42)
        obs = _observation(step=1, width=6, updates=updates)
        game = _rebuild_game(obs)
        self.assertEqual(game.players[0].research_points, 42)

    def test_turn_number_derived_from_step(self):
        # step 0 = init (before any update), step 1 = first turn, so
        # game.turn should equal step - 1 for step >= 1.
        for step in [1, 5, 42, 359]:
            obs = _observation(step=step, updates=_turn_updates(p0_units=[("u_1", 0, 0)]))
            game = _rebuild_game(obs)
            self.assertEqual(game.turn, step - 1, f"step={step}")

    def test_resources_and_cities(self):
        updates = _turn_updates(
            width=6,
            p0_units=[("u_1", 1, 1)],
            p1_units=[("u_2", 4, 4)],
            p0_cities=[("c_1", [(2, 2), (2, 3)])],
            resources=[("wood", 3, 3, 500), ("coal", 5, 0, 100)],
        )
        obs = _observation(step=5, width=6, updates=updates)
        game = _rebuild_game(obs)
        self.assertEqual(len(game.players[0].units), 1)
        self.assertEqual(len(game.players[1].units), 1)
        self.assertEqual(len(game.players[0].cities), 1)
        self.assertEqual(game.players[0].city_tile_count, 2)
        self.assertTrue(game.map.get_cell(3, 3).has_resource())
        self.assertEqual(game.map.get_cell(3, 3).resource.type, "wood")


class RenderAsciiMapTest(absltest.TestCase):
    def test_uppercase_for_current_player(self):
        updates = _turn_updates(
            width=6,
            p0_units=[("u_1", 1, 1)],
            p1_units=[("u_2", 4, 4)],
            p0_cities=[("c_1", [(2, 2)])],
            resources=[
                ("wood", 0, 0, 500),
                ("coal", 0, 5, 100),
                ("uranium", 5, 5, 50),
            ],
        )
        obs = _observation(step=2, player=0, width=6, updates=updates)
        game = _rebuild_game(obs)

        rendered_p0 = _render_ascii_map(game, player_id=0)
        self.assertIn("U", rendered_p0)  # own worker uppercase
        self.assertIn("u", rendered_p0)  # opponent worker lowercase
        self.assertIn("T", rendered_p0)  # own city tile uppercase
        self.assertIn("w", rendered_p0)  # wood glyph
        self.assertIn("k", rendered_p0)  # coal glyph
        self.assertIn("x", rendered_p0)  # crystal glyph (rebranded uranium)

        rendered_p1 = _render_ascii_map(game, player_id=1)
        # From p1's perspective the letters flip.
        self.assertIn("U", rendered_p1)
        self.assertIn("u", rendered_p1)
        self.assertIn("t", rendered_p1)  # p0's city is now opponent's


# --- _phase -----------------------------------------------------------------


class PhaseTest(absltest.TestCase):
    def test_day_night_boundaries(self):
        self.assertEqual(_phase(0), "day")
        self.assertEqual(_phase(29), "day")
        self.assertEqual(_phase(30), "night")
        self.assertEqual(_phase(39), "night")
        self.assertEqual(_phase(40), "day")  # next cycle starts
        self.assertEqual(_phase(70), "night")


# --- generate_prompt --------------------------------------------------------


class GeneratePromptTest(absltest.TestCase):
    def test_renders_key_sections(self):
        updates = _turn_updates(
            width=8,
            p0_units=[("u_1", 2, 2)],
            p0_cities=[("c_1", [(3, 3)])],
            resources=[("wood", 5, 5, 500)],
            p0_rp=42,
        )
        obs = _observation(step=16, player=0, width=8, updates=updates)
        prompt = generate_prompt(obs, move_history=[])

        # Board dimensions surface from the observation, not hardcoded.
        self.assertIn("8x8", prompt)
        # Turn + phase.
        self.assertIn("Turn 15", prompt)
        self.assertIn("day", prompt)
        # Player id.
        self.assertIn("player 0", prompt)
        # Structured summary carries unit + city + research info.
        self.assertIn("u_1", prompt)
        self.assertIn("c_1", prompt)
        self.assertIn("42", prompt)
        # Command grammar is documented.
        self.assertIn("bcity", prompt)
        # JSON response format is documented.
        self.assertIn('"actions"', prompt)
        # Rebrand: the prose talks about "crystal", not "uranium".
        self.assertIn("crystal", prompt)
        self.assertNotIn("uranium", prompt)

    def test_recent_moves_included_when_history_nonempty(self):
        obs = _observation()
        prompt = generate_prompt(obs, move_history=["[m u_1 n, bcity u_2]"])
        self.assertIn("previous turn", prompt.lower())
        self.assertIn("[m u_1 n, bcity u_2]", prompt)

    def test_no_history_block_when_empty(self):
        obs = _observation()
        prompt = generate_prompt(obs, move_history=[])
        self.assertNotIn("previous turn", prompt.lower())

    def test_rethink_illegal_appended_when_previous_action_set(self):
        obs = _observation()
        prompt = generate_prompt(
            obs,
            move_history=[],
            previous_response="whatever",
            previous_action="[junk1, junk2]",
        )
        self.assertIn("malformed", prompt)
        self.assertIn("[junk1, junk2]", prompt)

    def test_rethink_unparsable_appended_when_no_previous_action(self):
        obs = _observation()
        prompt = generate_prompt(
            obs,
            move_history=[],
            previous_response="I ate a sandwich.",
            previous_action=None,
        )
        self.assertIn("No JSON `actions` object could be parsed", prompt)
        self.assertIn("I ate a sandwich.", prompt)

    def test_empty_observation_returns_placeholder(self):
        prompt = generate_prompt({}, move_history=[])
        self.assertIn("has not started", prompt)

    def test_terminal_conditions_include_elimination_and_draw(self):
        obs = _observation()
        prompt = generate_prompt(obs, move_history=[])
        self.assertIn("elimination", prompt)
        self.assertIn("draw", prompt)
        self.assertIn("360", prompt)

    def test_adjacency_formula_uses_10_per_pair(self):
        # Engine double-counts adjacentCityTiles so the coefficient in the
        # net-upkeep formula is 10 per pair, not 5.
        obs = _observation()
        prompt = generate_prompt(obs, move_history=[])
        self.assertIn("23 * num_tiles - 10 * num_adjacent_pairs", prompt)

    def test_one_command_per_unit_per_turn_disclosed(self):
        obs = _observation()
        prompt = generate_prompt(obs, move_history=[])
        self.assertIn("AT MOST ONE command", prompt)

    def test_bcity_disallows_existing_city_tile(self):
        obs = _observation()
        prompt = generate_prompt(obs, move_history=[])
        self.assertIn("not on an existing city tile", prompt)

    def test_movement_stacking_rules_disclosed(self):
        obs = _observation()
        prompt = generate_prompt(obs, move_history=[])
        # Unlimited stacking on friendly city tiles.
        self.assertIn("UNLIMITED stacking", prompt)
        # Outside cities: at most one unit per cell.
        self.assertIn("at most ONE unit", prompt)
        # Colliding movers cancel (bounce back), and cascade.
        self.assertIn("cancelled", prompt)
        self.assertIn("cascade", prompt)
        # Cannot enter opponent city tile.
        self.assertIn("opponent city tile", prompt)

    def test_spawn_cost_and_unit_cap_disclosed(self):
        obs = _observation()
        prompt = generate_prompt(obs, move_history=[])
        # Spawning is free; unit count is capped by number of city tiles.
        self.assertIn("FREE", prompt)
        self.assertIn("total number of friendly city", prompt.lower().replace("tiles", "tile"))

    def test_bcity_cost_and_consumption_order_disclosed(self):
        obs = _observation()
        prompt = generate_prompt(obs, move_history=[])
        # bcity costs 100 resource units; consumed wood -> coal -> crystal.
        self.assertIn("100 resource units", prompt)
        self.assertIn("wood -> coal -> crystal", prompt)

    def test_wood_regeneration_disclosed(self):
        obs = _observation()
        prompt = generate_prompt(obs, move_history=[])
        # Only wood regenerates; coal and crystal do not.
        self.assertIn("Only WOOD", prompt)
        self.assertIn("do not replenish", prompt)

    def test_research_thresholds_disclosed(self):
        obs = _observation()
        prompt = generate_prompt(obs, move_history=[])
        # Numeric thresholds from game_constants: coal=50, uranium=200.
        self.assertIn("50", prompt)
        self.assertIn("200", prompt)
        # Explanation that `r` adds a point.
        self.assertIn("research point", prompt)

    def test_night_upkeep_rules_disclosed(self):
        obs = _observation()
        prompt = generate_prompt(obs, move_history=[])
        # City tile upkeep + adjacency bonus formula are surfaced.
        self.assertIn("23", prompt)  # per-tile base
        # Worker/cart night-consumption numbers and the "in a city is safe" carve-out.
        self.assertIn("4", prompt)  # worker night burn
        self.assertIn("10", prompt)  # cart night burn
        self.assertIn("city tile at night", prompt.lower().replace("standing on a friendly ", ""))

    def test_player_summary_shows_total_upkeep(self):
        updates = _turn_updates(
            width=6,
            p0_units=[("u_1", 0, 0)],
            p0_cities=[("c_1", [(2, 2), (2, 3)])],  # single city, 2 tiles
        )
        obs = _observation(step=5, player=0, width=6, updates=updates)
        prompt = generate_prompt(obs, move_history=[])
        # Test fixture emits `c 0 c_1 100 5` → per-city light_upkeep=5 in
        # the reconstructed Game. Sum across cities must appear.
        self.assertIn("total night upkeep 5/turn", prompt)

    def test_player_summary_shows_research_next_unlock(self):
        updates = _turn_updates(width=6, p0_units=[("u_1", 0, 0)], p0_rp=17)
        obs = _observation(step=5, player=0, width=6, updates=updates)
        prompt = generate_prompt(obs, move_history=[])
        # 17 points → coal in 33 more.
        self.assertIn("coal in 33 more points", prompt)

    def test_bcity_empty_tile_requirement_disclosed(self):
        # Engine's Unit.can_build requires the worker's cell to have no
        # resource. Prompt must say so or models issue silently-ignored
        # bcity commands while standing on wood.
        obs = _observation()
        prompt = generate_prompt(obs, move_history=[])
        # Both the "empty" language and the specific resource names appear
        # in the bcity line -- verify the bcity clause is present.
        self.assertIn("empty cell", prompt)

    def test_step_0_reports_turn_0_day(self):
        # At step 0, game.turn == -1 pre-update; the prompt must clamp so
        # it doesn't read "Turn -1 (night)".
        header = ["0", "6 6"]
        body = _turn_updates(width=6, p0_units=[("u_1", 1, 1)])
        obs = _observation(step=0, player=0, width=6, updates=header + body)
        prompt = generate_prompt(obs, move_history=[])
        self.assertIn("Turn 0 of 360 (day)", prompt)
        self.assertNotIn("Turn -1", prompt)


if __name__ == "__main__":
    absltest.main()
