import pytest

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    FARM_HAND_COST_MULT,
    LAND_ORDER,
    LAND_PRICES,
    MARKET_PARAMS,
    MAX_SHOP_INSTANCES,
    PRODUCTS,
    SHOPS,
    TOWN_CENTER_PRODUCTS,
    _apply_unit_action,
    _commit_unit,
    _daily_refresh_animals,
    _daily_refresh_plants,
    _decay_plants,
    _do_buy_land,
    _do_hire,
    _drop_inventories_to_shed,
    _end_of_day,
    _new_animal,
    _new_farm,
    _new_market,
    _new_plant,
    _new_private,
    _quadrant_of,
    _shape,
    _shed_access_tiles,
    _town_consume,
    market_price,
)


# --- Smoke / lifecycle ------------------------------------------------------

def test_episode_completes():
    env = make("kaggriculture", configuration={"episodeSteps": 50})
    env.run(["pass", "pass"])
    j = env.toJSON()
    assert j["name"] == "kaggriculture"
    assert j["statuses"] == ["DONE", "DONE"]


def test_pass_keeps_starting_money():
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": 50, "startingMoney": 200},
    )
    env.run(["pass", "pass"])
    j = env.toJSON()
    assert j["rewards"] == [200.0, 200.0]


def test_seed_is_scrubbed_from_configuration():
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": 5, "seed": 12345},
    )
    env.run(["pass", "pass"])
    j = env.toJSON()
    assert j["configuration"].get("seed") is None
    assert env.info.get("seed") == 12345


def test_renderer():
    env = make("kaggriculture", configuration={"episodeSteps": 5})
    env.run(["pass", "pass"])
    out = env.render(mode="ansi")
    assert isinstance(out, str)
    assert len(out) > 0


def test_random_agent_runs_full_episode():
    env = make("kaggriculture", configuration={"episodeSteps": 100, "seed": 1})
    env.run(["random", "random"])
    j = env.toJSON()
    assert j["statuses"] == ["DONE", "DONE"]


# --- Land / quadrants -------------------------------------------------------

def test_initial_board_only_nw_unlocked():
    farm = _new_farm(10, 100)
    assert farm["unlocked_quadrants"] == ["NW"]
    # NW quadrant tiles are None; everything else is "LOCKED".
    for y in range(10):
        for x in range(10):
            if _quadrant_of(x, y, 10) == "NW":
                assert farm["tiles"][y][x] is None
            else:
                assert farm["tiles"][y][x] == "LOCKED"


def test_default_spawn_is_inner_corner_of_nw():
    farm = _new_farm(10, 100)
    # NW inner-corner of the four shed-access tiles is (4, 4).
    assert farm["farmer"] == [4, 4]


def test_buy_land_unlocks_quadrants_in_fixed_order():
    farm = _new_farm(10, 10_000)
    for expected, price in zip(LAND_ORDER, LAND_PRICES):
        before = farm["money"]
        _do_buy_land(farm, 10)
        assert expected in farm["unlocked_quadrants"]
        assert farm["money"] == before - price
    # Past the last quadrant, BUY_LAND is a no-op.
    before = farm["money"]
    _do_buy_land(farm, 10)
    assert farm["money"] == before
    # All tiles should now be unlocked.
    for row in farm["tiles"]:
        for tile in row:
            assert tile != "LOCKED"


def test_buy_land_rejected_when_too_expensive():
    farm = _new_farm(10, 500)
    _do_buy_land(farm, 10)
    assert "NE" not in farm["unlocked_quadrants"]
    assert farm["money"] == 500


def test_movement_allowed_onto_locked_tiles():
    farm = _new_farm(10, 100)
    private = _new_private()
    # Farmer starts at (4,4). EAST -> (5,4) which is in NE quadrant (locked).
    # Movement onto locked tiles is permitted so units are never stranded.
    _apply_unit_action(farm, private, 0, ["EAST"], 10, 0, 24)
    assert farm["farmer"] == [5, 4]
    assert farm["tiles"][4][5] == "LOCKED"
    # And they can step back off the locked tile.
    _apply_unit_action(farm, private, 0, ["WEST"], 10, 0, 24)
    assert farm["farmer"] == [4, 4]


def test_hand_spawned_on_locked_tile_can_move_off():
    """A hand spawned on a locked shed-access tile must not be stranded."""
    farm = _new_farm(10, 100)
    private = _new_private()
    # Force a hand onto a locked SE shed-access tile (5,5).
    farm["hands"].append([5, 5])
    private["inventories"].append({})
    assert farm["tiles"][5][5] == "LOCKED"
    _apply_unit_action(farm, private, 1, ["NORTH"], 10, 0, 24)
    assert farm["hands"][0] == [5, 4]


def test_movement_blocked_off_map():
    """Allowing movement onto locked tiles must not allow moving off the board."""
    farm = _new_farm(10, 100)
    private = _new_private()
    # Main farmer at NW corner (0, 0): NORTH and WEST both leave the board.
    farm["farmer"] = [0, 0]
    _apply_unit_action(farm, private, 0, ["NORTH"], 10, 0, 24)
    assert farm["farmer"] == [0, 0]
    _apply_unit_action(farm, private, 0, ["WEST"], 10, 0, 24)
    assert farm["farmer"] == [0, 0]
    # A hand at the SE corner (9, 9) cannot move SOUTH or EAST off the board.
    farm["hands"].append([9, 9])
    private["inventories"].append({})
    _apply_unit_action(farm, private, 1, ["SOUTH"], 10, 0, 24)
    assert farm["hands"][0] == [9, 9]
    _apply_unit_action(farm, private, 1, ["EAST"], 10, 0, 24)
    assert farm["hands"][0] == [9, 9]


def test_shed_ops_work_from_a_locked_shed_access_tile():
    """Three of the four shed-access tiles start locked; the shed still works."""
    farm = _new_farm(10, 100)
    private = _new_private()
    private["shed"]["WHEAT"] = 5
    # (5, 4) is a shed-access tile in the NE quadrant, so it starts locked.
    farm["farmer"] = [5, 4]
    assert farm["tiles"][4][5] == "LOCKED"

    _apply_unit_action(farm, private, 0, ["PICKUP", "WHEAT", 2], 10, 0, 24)
    assert private["inventories"][0]["WHEAT"] == 2
    assert private["shed"]["WHEAT"] == 3

    _apply_unit_action(farm, private, 0, ["PLACE", "WHEAT", 1], 10, 0, 24)
    assert private["inventories"][0]["WHEAT"] == 1
    assert private["shed"]["WHEAT"] == 4

    _apply_unit_action(farm, private, 0, ["DROP"], 10, 0, 24)
    assert private["inventories"][0].get("WHEAT", 0) == 0
    assert private["shed"]["WHEAT"] == 5


def test_tile_ops_still_noop_on_locked_shed_access_tile():
    """Resolving shed ops early must not let tile ops run on locked ground."""
    farm = _new_farm(10, 100)
    private = _new_private()
    private["seeds"]["CARROT"] = 3
    farm["farmer"] = [5, 4]
    assert farm["tiles"][4][5] == "LOCKED"

    for action in (["PLANT", "CARROT"], ["BUILD_COOP"], ["BUILD_PASTURE"], ["DIG"]):
        _apply_unit_action(farm, private, 0, action, 10, 0, 24)
        assert farm["tiles"][4][5] == "LOCKED", action
    assert private["seeds"]["CARROT"] == 3


def test_place_animal_still_noop_on_locked_tile():
    """PLACE moved above the guard, but animals need an owned structure."""
    farm = _new_farm(10, 100)
    private = _new_private()
    # (5, 4) is locked and shed-adjacent: the animal branch must not match, and
    # the shed fallback must not silently swallow the animal either.
    farm["farmer"] = [5, 4]
    private["inventories"][0]["CHICKEN"] = 1
    _apply_unit_action(farm, private, 0, ["PLACE", "CHICKEN"], 10, 0, 24)
    assert farm["tiles"][4][5] == "LOCKED"
    # It went into the shed as a plain item, not onto the locked tile.
    assert private["inventories"][0].get("CHICKEN", 0) == 0
    assert private["shed"]["CHICKEN"] == 1


# --- Shed / pickup / inventory ---------------------------------------------

def test_pickup_requires_shed_adjacency():
    farm = _new_farm(10, 100)
    private = _new_private()
    private["shed"]["WHEAT"] = 5
    # Move away from shed-adjacent tile (4, 4) -> (3, 4).
    _apply_unit_action(farm, private, 0, ["WEST"], 10, 0, 24)
    assert farm["farmer"] == [3, 4]
    _apply_unit_action(farm, private, 0, ["PICKUP", "WHEAT", 1], 10, 0, 24)
    assert private["shed"]["WHEAT"] == 5
    assert private["inventories"][0].get("WHEAT", 0) == 0


def test_pickup_moves_item_to_inventory():
    farm = _new_farm(10, 100)
    private = _new_private()
    private["shed"]["WHEAT"] = 5
    # Farmer is at (4, 4) which is shed-adjacent.
    _apply_unit_action(farm, private, 0, ["PICKUP", "WHEAT", 3], 10, 0, 24)
    assert private["shed"]["WHEAT"] == 2
    assert private["inventories"][0]["WHEAT"] == 3


def test_pickup_does_not_pull_from_seed_pool():
    """Seeds live only in the shared seed pool and are consumed directly by
    PLANT — they never pass through farmer inventory via PICKUP."""
    farm = _new_farm(10, 100)
    private = _new_private()
    private["seeds"]["CARROT"] = 3
    _apply_unit_action(farm, private, 0, ["PICKUP", "CARROT", 2], 10, 0, 24)
    assert private["seeds"]["CARROT"] == 3
    assert private["inventories"][0].get("CARROT", 0) == 0


def test_drop_action_dumps_inventory_when_shed_adjacent():
    farm = _new_farm(10, 100)
    private = _new_private()
    # Farmer spawns at (4, 4) which is shed-adjacent.
    private["inventories"][0] = {"WHEAT": 3, "CARROT": 2}
    _apply_unit_action(farm, private, 0, ["DROP"], 10, 0, 24)
    assert private["shed"]["WHEAT"] == 3
    assert private["shed"]["CARROT"] == 2
    assert private["inventories"][0] == {}


def test_drop_action_noop_when_not_shed_adjacent():
    farm = _new_farm(10, 100)
    private = _new_private()
    private["inventories"][0] = {"WHEAT": 3}
    # Step west off the shed-adjacent spawn.
    _apply_unit_action(farm, private, 0, ["WEST"], 10, 0, 24)
    _apply_unit_action(farm, private, 0, ["DROP"], 10, 0, 24)
    assert private["inventories"][0] == {"WHEAT": 3}
    assert private["shed"].get("WHEAT", 0) == 0


def test_drop_action_discards_overflow_past_shed_capacity():
    farm = _new_farm(10, 100)
    private = _new_private()
    private["shed"]["WHEAT"] = 98
    private["inventories"][0] = {"WHEAT": 5, "CARROT": 4}
    _apply_unit_action(farm, private, 0, ["DROP"], 10, 0, 24, shed_capacity=100)
    # Shed fills to capacity, overflow lost.
    assert sum(private["shed"].values()) == 100
    assert private["inventories"][0] == {}


def test_drop_inventories_respects_shed_capacity():
    private = _new_private()
    # Pre-fill shed near capacity (capacity 100).
    private["shed"]["WHEAT"] = 99
    private["inventories"][0] = {"WHEAT": 5}
    _drop_inventories_to_shed(private, 100)
    assert private["shed"]["WHEAT"] == 100
    # Overflow discarded.
    assert private["inventories"][0] == {}


def test_drop_inventories_routes_crops_to_shed_as_products():
    """Items in farmer inventory are always treated as harvested products and
    go to the shed (capped). The seed pool is untouched."""
    private = _new_private()
    private["inventories"][0] = {"WHEAT": 2, "CARROT": 1, "MELON": 4}
    _drop_inventories_to_shed(private, 100)
    assert private["shed"]["WHEAT"] == 2
    assert private["shed"]["CARROT"] == 1
    assert private["shed"]["MELON"] == 4
    assert private["seeds"]["WHEAT"] == 0


# --- Plants / fertilizer ----------------------------------------------------

def test_plant_consumes_seed_and_occupies_tile():
    farm = _new_farm(10, 100)
    private = _new_private()
    private["seeds"]["WHEAT"] = 2
    _apply_unit_action(farm, private, 0, ["PLANT", "WHEAT"], 10, 0, 24)
    assert private["seeds"]["WHEAT"] == 1
    fx, fy = farm["farmer"]
    assert farm["tiles"][fy][fx]["crop"] == "WHEAT"


def test_plant_does_not_consume_from_inventory():
    """PLANT only draws from the shared seed pool; product wheat sitting in a
    farmer's inventory cannot be planted."""
    farm = _new_farm(10, 100)
    private = _new_private()
    private["inventories"][0]["WHEAT"] = 1
    _apply_unit_action(farm, private, 0, ["PLANT", "WHEAT"], 10, 0, 24)
    fx, fy = farm["farmer"]
    assert farm["tiles"][fy][fx] is None
    assert private["inventories"][0]["WHEAT"] == 1


def test_plant_dies_if_planting_day_unwatered():
    farm = _new_farm(10, 100)
    private = _new_private()
    private["seeds"]["WHEAT"] = 1
    _apply_unit_action(farm, private, 0, ["PLANT", "WHEAT"], 10, 0, 24)
    fx, fy = farm["farmer"]
    _daily_refresh_plants(farm, 0, 24)
    assert farm["tiles"][fy][fx] == {"kind": "WEED"}


def test_water_in_bonus_window_increments_yield():
    farm = _new_farm(10, 100)
    private = _new_private()
    private["seeds"]["WHEAT"] = 1
    _apply_unit_action(farm, private, 0, ["PLANT", "WHEAT"], 10, 0, 24)
    fx, fy = farm["farmer"]
    for d in (2, 3, 4):
        farm["tiles"][fy][fx]["watered_today"] = False
        _apply_unit_action(farm, private, 0, ["WATER"], 10, d, 24)
    # Base 1 + 3 bonus = 4 (also the cap).
    assert farm["tiles"][fy][fx]["yield_units"] == 4


def test_fertilize_doubles_watering_bonus_for_three_days():
    farm = _new_farm(10, 100)
    private = _new_private()
    private["seeds"]["MELON"] = 1
    private["inventories"][0]["FERTILIZER"] = 1
    _apply_unit_action(farm, private, 0, ["PLANT", "MELON"], 10, 0, 24)
    fx, fy = farm["farmer"]
    # MELON: max_yield_day=12, window_start = (12+1)//2 = 6.
    # Fertilize at day 6, then water on days 6, 7, 8 (each +2 = +6).
    _apply_unit_action(farm, private, 0, ["FERTILIZE"], 10, 6, 24)
    assert "FERTILIZER" not in private["inventories"][0]
    for d in (6, 7, 8):
        farm["tiles"][fy][fx]["watered_today"] = False
        _apply_unit_action(farm, private, 0, ["WATER"], 10, d, 24)
    # Base 1 + 6 from 3 fertilized waterings = 7, capped at MELON max_yield (6).
    assert farm["tiles"][fy][fx]["yield_units"] == 6


def test_fertilize_requires_fertilizer_in_inventory():
    farm = _new_farm(10, 100)
    private = _new_private()
    private["seeds"]["WHEAT"] = 1
    _apply_unit_action(farm, private, 0, ["PLANT", "WHEAT"], 10, 0, 24)
    fx, fy = farm["farmer"]
    _apply_unit_action(farm, private, 0, ["FERTILIZE"], 10, 0, 24)
    assert farm["tiles"][fy][fx]["fertilized_until_day"] == -1


def test_one_time_plant_decays_after_max_lifespan():
    farm = _new_farm(10, 100)
    private = _new_private()
    private["seeds"]["WHEAT"] = 1
    _apply_unit_action(farm, private, 0, ["PLANT", "WHEAT"], 10, 0, 24)
    fx, fy = farm["farmer"]
    farm["tiles"][fy][fx]["yield_units"] = 4
    # Plant on day 0 -> max_lifespan_step = (0+4+1)*24 = 120.
    _decay_plants(farm, 119)
    assert farm["tiles"][fy][fx]["yield_units"] == 4
    _decay_plants(farm, 120)
    assert farm["tiles"][fy][fx]["yield_units"] == 3


def test_unwatered_plant_becomes_weed_not_none():
    farm = _new_farm(10, 100)
    private = _new_private()
    private["seeds"]["WHEAT"] = 1
    _apply_unit_action(farm, private, 0, ["PLANT", "WHEAT"], 10, 0, 24)
    fx, fy = farm["farmer"]
    _daily_refresh_plants(farm, 0, 24)
    assert farm["tiles"][fy][fx] == {"kind": "WEED"}
    # DIG removes the weed.
    _apply_unit_action(farm, private, 0, ["DIG"], 10, 1, 24)
    assert farm["tiles"][fy][fx] is None


# --- Harvest goes to inventory ---------------------------------------------

def test_harvest_routes_to_inventory_not_money():
    farm = _new_farm(10, 100)
    private = _new_private()
    private["seeds"]["WHEAT"] = 1
    _apply_unit_action(farm, private, 0, ["PLANT", "WHEAT"], 10, 0, 24)
    fx, fy = farm["farmer"]
    farm["tiles"][fy][fx]["yield_units"] = 3
    _apply_unit_action(farm, private, 0, ["HARVEST"], 10, 4, 24)
    assert farm["money"] == 100  # unchanged
    assert private["inventories"][0]["WHEAT"] == 3
    assert farm["tiles"][fy][fx] is None  # one-time crop removed


def test_ongoing_harvest_keeps_plant():
    farm = _new_farm(10, 100)
    private = _new_private()
    private["seeds"]["TOMATO"] = 1
    _apply_unit_action(farm, private, 0, ["PLANT", "TOMATO"], 10, 0, 24)
    fx, fy = farm["farmer"]
    farm["tiles"][fy][fx]["yield_units"] = 2
    _apply_unit_action(farm, private, 0, ["HARVEST"], 10, 8, 24)
    assert private["inventories"][0]["TOMATO"] == 2
    assert farm["tiles"][fy][fx] is not None


# --- Animals ----------------------------------------------------------------

def test_build_coop_and_place_goose():
    farm = _new_farm(10, 1000)
    private = _new_private()
    private["shed"]["GOOSE"] = 1
    _apply_unit_action(farm, private, 0, ["PICKUP", "GOOSE", 1], 10, 0, 24)
    fx, fy = farm["farmer"]
    _apply_unit_action(farm, private, 0, ["BUILD_COOP"], 10, 0, 24)
    assert farm["tiles"][fy][fx] == {"kind": "COOP"}
    _apply_unit_action(farm, private, 0, ["PLACE", "GOOSE"], 10, 0, 24)
    tile = farm["tiles"][fy][fx]
    assert tile["animal"] == "GOOSE"
    assert tile["kind"] == "COOP"


def test_place_requires_correct_structure():
    farm = _new_farm(10, 1000)
    private = _new_private()
    private["inventories"][0]["COW"] = 1
    # Move off the default shed-adjacent spawn so PLACE doesn't fall through to
    # the shed-drop branch.
    farm["farmer"] = [0, 0]
    fx, fy = farm["farmer"]
    farm["tiles"][fy][fx] = {"kind": "COOP"}  # wrong structure for cow
    _apply_unit_action(farm, private, 0, ["PLACE", "COW"], 10, 0, 24)
    # Cow stays in inventory, coop unchanged.
    assert farm["tiles"][fy][fx] == {"kind": "COOP"}
    assert private["inventories"][0].get("COW") == 1


def test_place_shed_drop_respects_shed_capacity():
    """PLACE of a non-placeable item while shed-adjacent falls through to a
    shed-drop, which must obey shedCapacity (partial fill, overflow retained
    in inventory)."""
    farm = _new_farm(10, 100)
    private = _new_private()
    private["shed"]["WHEAT"] = 98  # 2 slots left
    # Farmer spawns at (4, 4), which is shed-adjacent.
    private["inventories"][0] = {"WHEAT": 5}
    _apply_unit_action(farm, private, 0, ["PLACE", "WHEAT", 5], 10, 0, 24, shed_capacity=100)
    assert sum(private["shed"].values()) == 100  # filled to capacity, not over
    assert private["inventories"][0]["WHEAT"] == 3  # unbanked units retained


def test_feed_consumes_wheat_from_farmer_inventory():
    farm = _new_farm(10, 1000)
    private = _new_private()
    fx, fy = farm["farmer"]
    farm["tiles"][fy][fx] = _new_animal("GOOSE", 0)
    private["inventories"][0]["WHEAT"] = 2
    _apply_unit_action(farm, private, 0, ["FEED"], 10, 0, 24)
    assert farm["tiles"][fy][fx]["fed_today"] is True
    assert private["inventories"][0]["WHEAT"] == 1
    # Idempotent within day.
    _apply_unit_action(farm, private, 0, ["FEED"], 10, 0, 24)
    assert private["inventories"][0]["WHEAT"] == 1


def test_feed_requires_wheat_in_inventory_not_shed():
    farm = _new_farm(10, 1000)
    private = _new_private()
    fx, fy = farm["farmer"]
    farm["tiles"][fy][fx] = _new_animal("GOOSE", 0)
    private["shed"]["WHEAT"] = 5  # shed only — should not auto-pull
    _apply_unit_action(farm, private, 0, ["FEED"], 10, 0, 24)
    assert farm["tiles"][fy][fx]["fed_today"] is False
    assert private["shed"]["WHEAT"] == 5


def test_animal_produces_on_schedule_and_partial_harvest_works():
    farm = _new_farm(10, 1000)
    private = _new_private()
    fx, fy = farm["farmer"]
    farm["tiles"][fy][fx] = _new_animal("GOOSE", 0)
    # Goose: first_yield_day=4, interval=1, max_held=4.
    # Feed every day, then refresh end-of-day.
    for d in range(0, 6):
        farm["tiles"][fy][fx]["fed_today"] = True
        _daily_refresh_animals(farm, d)
    tile = farm["tiles"][fy][fx]
    # First production at end of day 3 (next_day=4); produces on days 3,4,5 → 3 units.
    assert tile["yield_units"] == 3
    # Partial harvest before cap is fine — harvest now empties to 0.
    _apply_unit_action(farm, private, 0, ["HARVEST"], 10, 6, 24)
    assert private["inventories"][0]["EGG"] == 3
    assert farm["tiles"][fy][fx]["yield_units"] == 0


def test_animal_escapes_after_two_unfed_days():
    farm = _new_farm(10, 1000)
    fx, fy = farm["farmer"]
    farm["tiles"][fy][fx] = _new_animal("GOOSE", 0)
    _daily_refresh_animals(farm, 0)
    _daily_refresh_animals(farm, 1)
    # Animal escapes; coop remains.
    assert farm["tiles"][fy][fx] == {"kind": "COOP"}


def test_collect_fertilizer_action_yields_one_per_day():
    farm = _new_farm(10, 1000)
    private = _new_private()
    fx, fy = farm["farmer"]
    farm["tiles"][fy][fx] = _new_animal("GOOSE", 0)
    farm["tiles"][fy][fx]["fertilizer_available"] = True
    _apply_unit_action(farm, private, 0, ["COLLECT_FERTILIZER"], 10, 0, 24)
    assert private["inventories"][0]["FERTILIZER"] == 1
    # Second COLLECT_FERTILIZER same day is a no-op.
    _apply_unit_action(farm, private, 0, ["COLLECT_FERTILIZER"], 10, 0, 24)
    assert private["inventories"][0]["FERTILIZER"] == 1
    # After a daily refresh fertilizer regenerates (animal must be alive — feed it).
    farm["tiles"][fy][fx]["fed_today"] = True
    _daily_refresh_animals(farm, 0)
    assert farm["tiles"][fy][fx]["fertilizer_available"] is True


def test_care_bonus_adds_to_next_production():
    farm = _new_farm(10, 1000)
    fx, fy = farm["farmer"]
    farm["tiles"][fy][fx] = _new_animal("GOOSE", 0)
    # Feed + care for several days before first yield — bonus accumulates.
    for d in range(0, 4):
        farm["tiles"][fy][fx]["fed_today"] = True
        farm["tiles"][fy][fx]["cared_today"] = True
        _daily_refresh_animals(farm, d)
    # Day 4 refresh triggers first production at end of day 4 (next_day=5).
    farm["tiles"][fy][fx]["fed_today"] = True
    farm["tiles"][fy][fx]["cared_today"] = True
    _daily_refresh_animals(farm, 4)
    # Care bonus capped at max_held = 4.
    assert farm["tiles"][fy][fx]["yield_units"] == 4


def test_dig_does_not_remove_placed_animal():
    farm = _new_farm(10, 1000)
    private = _new_private()
    fx, fy = farm["farmer"]
    farm["tiles"][fy][fx] = _new_animal("GOOSE", 0)
    _apply_unit_action(farm, private, 0, ["DIG"], 10, 0, 24)
    assert farm["tiles"][fy][fx]["animal"] == "GOOSE"


# --- Market dynamic pricing -------------------------------------------------

def test_market_price_at_I0_equals_base():
    for item, p in MARKET_PARAMS.items():
        assert market_price(item, p["I0"]) == p["base"], item


def test_market_price_falls_with_excess_inventory():
    p_base = market_price("WHEAT", MARKET_PARAMS["WHEAT"]["I0"])
    p_excess = market_price("WHEAT", MARKET_PARAMS["WHEAT"]["I0"] * 4)
    assert p_excess < p_base


def test_market_price_rises_with_low_inventory():
    p_base = market_price("WHEAT", MARKET_PARAMS["WHEAT"]["I0"])
    p_zero = market_price("WHEAT", 0)
    assert p_zero > p_base


def test_hinge_shape_is_calibrated_so_f_of_T_is_one():
    # amp = target * base / f(T), so f(T) == 1 makes `target` mean the same
    # thing for hinge as it does for every other shape.
    assert _shape("hinge", 450, 450) == pytest.approx(1.0)


def test_hinge_degenerates_to_linear_without_T():
    for x in (0, 1, 50, 1000):
        assert _shape("hinge", x, None) == x
        assert _shape("hinge", x, 0) == x


def test_hinge_is_flat_below_the_knee_then_spikes_above_it():
    T = MARKET_PARAMS["CARROT"]["T"]
    base = MARKET_PARAMS["CARROT"]["base"]
    I0 = MARKET_PARAMS["CARROT"]["I0"]
    at_knee = market_price("CARROT", I0 - T)
    # Below the knee the curve is linear: half the depletion, half the rise.
    half = market_price("CARROT", I0 - T // 2)
    assert half - base == pytest.approx((at_knee - base) / 2, abs=1)
    # Past the knee it accelerates well beyond the linear continuation.
    beyond = market_price("CARROT", I0 - 2 * T)
    assert beyond > base + 4 * (at_knee - base)


def test_carrot_price_at_I0_minus_T_is_base_times_one_plus_target():
    p = MARKET_PARAMS["CARROT"]
    expected = p["base"] * (1 + p["below_target"])
    assert market_price("CARROT", p["I0"] - p["T"]) == expected


@pytest.mark.parametrize("item", ["TOMATO", "EGG"])
def test_hinge_matches_the_old_linear_curve_below_the_knee(item):
    # Tomato and egg switched linear -> hinge with below_target unchanged at
    # 0.40. That is deliberately a no-op for ordinary demand: linear's amp
    # normalises to x/T, which is exactly hinge's below-knee branch, so every
    # price from I0 down to I0-T is untouched. Only past the knee do the curves
    # diverge.
    p = MARKET_PARAMS[item]
    base, I0, T, target = p["base"], p["I0"], p["T"], p["below_target"]
    assert p["below_func"] == "hinge" and target == 0.40
    for depletion in range(0, T + 1):
        linear = base + target * base * depletion / T
        assert market_price(item, I0 - depletion) == max(1, round(linear))


@pytest.mark.parametrize("item", ["TOMATO", "EGG"])
def test_price_spikes_once_demand_passes_T(item):
    p = MARKET_PARAMS[item]
    I0, T = p["I0"], p["T"]
    at_knee = market_price(item, I0 - T)
    # The observed high-demand tail (~2.5x T) has to be worth chasing.
    assert market_price(item, I0 - 5 * T // 2) > 3 * at_knee


def test_market_price_floored_at_one_dollar():
    # At sufficiently high inventory, price floors at 1.
    # Use CARROT: above-side is sqrt (steep glut), so it actually hits the
    # floor at a reasonable inventory.
    very_high = MARKET_PARAMS["CARROT"]["I0"] * 1000
    assert market_price("CARROT", very_high) == 1


def test_sell_credits_at_current_price_and_grows_inventory():
    farm = _new_farm(10, 100)
    private = _new_private()
    private["shed"]["WHEAT"] = 1
    market = _new_market()
    base = market["inventory"]["WHEAT"]
    price = market_price("WHEAT", base)
    ok = _commit_unit("SELL", "WHEAT", price, farm, private, market)
    assert ok
    assert farm["money"] == 100 + price
    assert private["shed"]["WHEAT"] == 0
    assert market["inventory"]["WHEAT"] == base + 1


def test_sell_at_one_dollar_does_not_grow_inventory():
    # CARROT's above-side is sqrt (steep glut), so a massive inventory floors
    # the price at $1 quickly. WHEAT uses log above and never floors here.
    farm = _new_farm(10, 0)
    private = _new_private()
    private["shed"]["CARROT"] = 1
    market = _new_market()
    market["inventory"]["CARROT"] = MARKET_PARAMS["CARROT"]["I0"] * 1000
    price = market_price("CARROT", market["inventory"]["CARROT"])
    assert price == 1
    inv_before = market["inventory"]["CARROT"]
    _commit_unit("SELL", "CARROT", price, farm, private, market)
    assert market["inventory"]["CARROT"] == inv_before  # unchanged


def test_buy_product_charges_and_depletes_market():
    farm = _new_farm(10, 1000)
    private = _new_private()
    market = _new_market()
    inv0 = market["inventory"]["WHEAT"]
    price = market_price("WHEAT", inv0)
    ok = _commit_unit("BUY_PRODUCT", "WHEAT", price, farm, private, market)
    assert ok
    assert farm["money"] == 1000 - price
    assert private["shed"]["WHEAT"] == 1
    assert market["inventory"]["WHEAT"] == inv0 - 1


def test_buy_seed_uses_fixed_cost():
    farm = _new_farm(10, 100)
    private = _new_private()
    market = _new_market()
    ok = _commit_unit("BUY_SEED", "WHEAT", CROPS["WHEAT"]["seed"], farm, private, market)
    assert ok
    assert farm["money"] == 100 - CROPS["WHEAT"]["seed"]
    assert private["seeds"]["WHEAT"] == 1


def test_buy_animal_uses_fixed_cost():
    farm = _new_farm(10, 1000)
    private = _new_private()
    market = _new_market()
    ok = _commit_unit("BUY_ANIMAL", "GOOSE", ANIMALS["GOOSE"]["cost"], farm, private, market)
    assert ok
    assert farm["money"] == 1000 - ANIMALS["GOOSE"]["cost"]
    assert private["shed"]["GOOSE"] == 1


def test_buy_product_blocked_when_shed_full():
    farm = _new_farm(10, 1000)
    private = _new_private()
    private["shed"]["WHEAT"] = 100  # shed already at capacity
    market = _new_market()
    inv0 = market["inventory"]["WHEAT"]
    price = market_price("WHEAT", inv0)
    ok = _commit_unit("BUY_PRODUCT", "FERTILIZER", price, farm, private, market, shed_capacity=100)
    assert not ok
    assert farm["money"] == 1000  # no charge
    assert private["shed"].get("FERTILIZER", 0) == 0  # nothing added
    assert market["inventory"]["WHEAT"] == inv0  # market untouched


def test_buy_animal_blocked_when_shed_full():
    farm = _new_farm(10, 1000)
    private = _new_private()
    private["shed"]["WHEAT"] = 100  # shed already at capacity
    market = _new_market()
    ok = _commit_unit("BUY_ANIMAL", "GOOSE", ANIMALS["GOOSE"]["cost"], farm, private, market, shed_capacity=100)
    assert not ok
    assert farm["money"] == 1000  # no charge
    assert private["shed"].get("GOOSE", 0) == 0  # nothing added


def test_buy_then_sell_wheat_round_trip_nets_zero():
    """Buying N wheat and immediately selling them back must not change cash
    when nothing else touches the market in between."""
    env = make("kaggriculture", configuration={"episodeSteps": 5, "startingMoney": 10_000})
    env.reset(num_agents=2)
    money_before = env.state[0].observation.farms[0]["money"]
    env.step([
        {"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 10], ["SELL", "WHEAT", 10]]},
        {"farmer": ["PASS"], "hands": [], "market": []},
    ])
    assert env.state[0].observation.farms[0]["money"] == money_before
    assert env.state[0].observation.private["shed"].get("WHEAT", 0) == 0


def test_buy_product_partial_fills_up_to_shed_capacity():
    """With 5 slots left and an order for 10, buy exactly 5 (shed -> 100) and
    stop; the remaining 5 units of the order are dropped."""
    env = make("kaggriculture", configuration={"episodeSteps": 5, "startingMoney": 1_000_000})
    env.reset(num_agents=2)
    # Leave exactly 5 slots of headroom in the shed.
    env.state[0].observation.private["shed"]["FERTILIZER"] = 95
    money_before = env.state[0].observation.farms[0]["money"]
    env.step([
        {"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 10]]},
        {"farmer": ["PASS"], "hands": [], "market": []},
    ])
    shed = env.state[0].observation.private["shed"]
    assert sum(shed.values()) == 100  # filled to capacity, not over
    assert shed.get("WHEAT", 0) == 5  # only 5 of the 10 requested bought
    # Charged for exactly the 5 units that landed in the shed.
    assert env.state[0].observation.farms[0]["money"] < money_before


def test_buy_then_sell_fertilizer_round_trip_nets_zero():
    env = make("kaggriculture", configuration={"episodeSteps": 5, "startingMoney": 10_000})
    env.reset(num_agents=2)
    money_before = env.state[0].observation.farms[0]["money"]
    inv_before = env.state[0].observation.market["inventory"]["FERTILIZER"]
    env.step([
        {"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "FERTILIZER", 5], ["SELL", "FERTILIZER", 5]]},
        {"farmer": ["PASS"], "hands": [], "market": []},
    ])
    assert env.state[0].observation.farms[0]["money"] == money_before
    assert env.state[0].observation.market["inventory"]["FERTILIZER"] == inv_before


def test_buy_product_only_allows_wheat_and_fertilizer():
    """BUY_PRODUCT for anything other than WHEAT/FERTILIZER must be a no-op."""
    env = make("kaggriculture", configuration={"episodeSteps": 5, "startingMoney": 10_000})
    env.reset(num_agents=2)
    money_before = env.state[0].observation.farms[0]["money"]
    env.step([
        {"farmer": ["PASS"], "hands": [], "market": [
            ["BUY_PRODUCT", "CARROT", 1],
            ["BUY_PRODUCT", "TOMATO", 1],
            ["BUY_PRODUCT", "STRAWBERRY", 1],
            ["BUY_PRODUCT", "MELON", 1],
            ["BUY_PRODUCT", "EGG", 1],
            ["BUY_PRODUCT", "MILK", 1],
            ["BUY_PRODUCT", "WOOL", 1],
        ]},
        {"farmer": ["PASS"], "hands": [], "market": []},
    ])
    # No money spent, no inventory added.
    assert env.state[0].observation.farms[0]["money"] == money_before
    shed = env.state[0].observation.private["shed"]
    for item in ("CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"):
        assert shed.get(item, 0) == 0


def test_sell_fertilizer_is_allowed():
    """Players should be able to sell FERTILIZER back to the market."""
    env = make("kaggriculture", configuration={"episodeSteps": 5, "startingMoney": 0})
    env.reset(num_agents=2)
    env.state[0].observation.private["shed"]["FERTILIZER"] = 3
    inv_before = env.state[0].observation.market["inventory"]["FERTILIZER"]
    env.step([
        {"farmer": ["PASS"], "hands": [], "market": [["SELL", "FERTILIZER", 3]]},
        {"farmer": ["PASS"], "hands": [], "market": []},
    ])
    assert env.state[0].observation.private["shed"].get("FERTILIZER", 0) == 0
    assert env.state[0].observation.farms[0]["money"] > 0
    assert env.state[0].observation.market["inventory"]["FERTILIZER"] == inv_before + 3


def test_concurrent_sells_get_same_quoted_price_per_unit():
    """End-to-end: both players SELL 2 wheat from full sheds — they should each
    receive the same price for unit 1, then both for unit 2 (post-update)."""
    def seller(obs):
        if obs.get("step", 0) == 0:
            return {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 2]]}
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env = make("kaggriculture", configuration={"episodeSteps": 5, "startingMoney": 0})
    env.reset(num_agents=2)
    env.state[0].observation.private["shed"]["WHEAT"] = 2
    env.state[1].observation.private["shed"]["WHEAT"] = 2
    env.step([
        {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 2]]},
        {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 2]]},
    ])
    # Both players sold in lockstep, so they earned the same amount.
    assert env.state[0].observation.farms[0]["money"] == env.state[0].observation.farms[1]["money"]
    assert env.state[0].observation.private["shed"]["WHEAT"] == 0
    assert env.state[1].observation.private["shed"]["WHEAT"] == 0


# --- Hire farm hands --------------------------------------------------------

def test_hire_spawns_hand_at_shed_access():
    farm = _new_farm(10, 1000)
    private = _new_private()
    _do_hire(farm, private, 10)
    assert len(farm["hands"]) == 1
    # First hand should spawn at the first free shed-access tile after the
    # main farmer (also there). (4,4) is taken by main farmer, so (5,4)/(4,5)/(5,5)
    # are options; NWSE preference picks (5,4) (NE) but that is locked initially.
    # The spawn helper picks the lowest-occupancy free tile, which here is (5,4).
    # We just check it landed on one of the access tiles.
    assert tuple(farm["hands"][0]) in set(_shed_access_tiles(10))


def test_hire_cost_increases_with_each_hire_today():
    farm = _new_farm(10, 100_000)
    private = _new_private()
    initial = farm["money"]
    for i in range(5):
        _do_hire(farm, private, 10)
    # Fib: 1, 1, 2, 3, 5 → 12 units × default mult 10 = 120.
    spent = FARM_HAND_COST_MULT * (1 + 1 + 2 + 3 + 5)
    assert farm["money"] == initial - spent
    assert farm["hires_today"] == 5
    assert len(farm["hands"]) == 5


def test_hire_rejected_when_too_expensive():
    farm = _new_farm(10, FARM_HAND_COST_MULT - 1)
    private = _new_private()
    _do_hire(farm, private, 10)
    assert farm["hands"] == []
    assert farm["money"] == FARM_HAND_COST_MULT - 1


def test_hand_actions_dispatched_via_hands_field():
    """End-to-end: hire a hand, then issue a movement to the hand only."""
    def hirer(obs):
        step = obs.get("step", 0)
        farms = obs.get("farms", [])
        farm = farms[obs.get("player", 0)]
        if step == 0:
            return {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]}
        if step == 1 and farm["hands"]:
            # Move the hand WEST (within NW quadrant).
            return {"farmer": ["PASS"], "hands": [["WEST"]], "market": []}
        return {"farmer": ["PASS"], "hands": [["PASS"]] if farm["hands"] else [], "market": []}

    env = make("kaggriculture", configuration={"episodeSteps": 5, "startingMoney": 1000})
    env.run([hirer, "pass"])
    j = env.toJSON()
    assert j["statuses"] == ["DONE", "DONE"]
    # Hand started at a shed-access tile and moved WEST once.
    farm = j["steps"][2][0]["observation"]["farms"][0]
    assert len(farm["hands"]) == 1


def test_hands_dismissed_at_end_of_day():
    """Hire a hand on hour 0; by end of day they should be gone."""
    def hirer(obs):
        if obs.get("step", 0) == 0:
            return {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]}
        return {"farmer": ["PASS"], "hands": [["PASS"]] if obs.get("farms", [])[obs.get("player", 0)]["hands"] else [], "market": []}

    env = make("kaggriculture", configuration={"episodeSteps": 26, "startingMoney": 1000, "turnsPerDay": 24})
    env.run([hirer, "pass"])
    j = env.toJSON()
    # After end-of-day refresh, hands list is reset.
    final_farm = j["steps"][-1][0]["observation"]["farms"][0]
    assert final_farm["hands"] == []
    assert final_farm["hires_today"] == 0


# --- Town buildings ---------------------------------------------------------

def test_town_unlocks_a_shop_after_three_days():
    env = make("kaggriculture", configuration={"episodeSteps": 24 * 3 + 2, "seed": 7, "turnsPerDay": 24})
    env.run(["pass", "pass"])
    j = env.toJSON()
    final_town = j["steps"][-1][0]["observation"]["town"]
    assert len(final_town["unlocked_shops"]) >= 1
    assert all(s in SHOPS for s in final_town["unlocked_shops"])


def test_town_shops_can_unlock_duplicates():
    """Shops are drawn with replacement, so over a full season at least one
    seed should produce a repeated shop. Sample a handful of seeds and assert
    duplicates occur (the with-replacement draw makes this overwhelmingly likely)."""
    saw_duplicate = False
    for seed in range(8):
        env = make("kaggriculture", configuration={"episodeSteps": 24 * 30, "seed": seed, "turnsPerDay": 24})
        env.run(["pass", "pass"])
        shops = env.toJSON()["steps"][-1][0]["observation"]["town"]["unlocked_shops"]
        assert all(s in SHOPS for s in shops)
        if len(set(shops)) < len(shops):
            saw_duplicate = True
            break
    assert saw_duplicate, "expected at least one duplicated shop across sampled seeds"


def test_shop_unlocks_stop_at_the_instance_cap():
    """_end_of_day must not push past MAX_SHOP_INSTANCES."""
    env = make("kaggriculture", configuration={"seed": 3, "turnsPerDay": 24, "townShopUnlockInterval": 1})
    state = env.reset()
    town = state[0].observation.town
    # 40 unlock opportunities against a cap of 8.
    for day in range(40):
        _end_of_day(state, env, day)
    assert len(town["unlocked_shops"]) == MAX_SHOP_INSTANCES


def test_duplicate_shops_consume_independently():
    """Two copies of a shop must drain twice as much as one copy."""
    def drain(shops):
        env = make("kaggriculture", configuration={"seed": 5, "townShopSellInterval": 1})
        state = env.reset()
        market = state[0].observation.market
        state[0].observation.town["unlocked_shops"] = list(shops)
        before = dict(market["inventory"])
        _town_consume(env, state, 1)  # step 1: shops tick, town center does not
        return {k: before[k] - v for k, v in market["inventory"].items()}

    one = drain(["BAKERY"])
    two = drain(["BAKERY", "BAKERY"])
    assert one["WHEAT"] == 1 and one["EGG"] == 1
    assert two["WHEAT"] == 2 and two["EGG"] == 2


def test_town_center_rate_is_flat_across_the_season():
    """No ramp: the center pulls exactly 1 of each non-fertilizer product on
    every center tick, on day 0 and on day 29 alike."""
    for step in (0, 24 * 29):
        env = make("kaggriculture", configuration={"seed": 5, "turnsPerDay": 24})
        state = env.reset()
        market = state[0].observation.market
        state[0].observation.town["unlocked_shops"] = []
        before = dict(market["inventory"])
        _town_consume(env, state, step)
        for item in TOWN_CENTER_PRODUCTS:
            assert before[item] - market["inventory"][item] == 1, f"{item} at step {step}"


def test_town_center_buys_once_per_day_by_default():
    """Default townCenterSellInterval equals turnsPerDay, so exactly one center
    tick lands in each day."""
    env = make("kaggriculture", configuration={"turnsPerDay": 24})
    cfg = env.configuration
    assert cfg.townCenterSellInterval == 24
    assert cfg.townCenterSellInterval == cfg.turnsPerDay

    ticks = 0
    for step in range(24):
        state = env.reset()
        market = state[0].observation.market
        state[0].observation.town["unlocked_shops"] = []
        before = market["inventory"]["WHEAT"]
        _town_consume(env, state, step)
        if market["inventory"]["WHEAT"] < before:
            ticks += 1
    assert ticks == 1


def test_town_consumes_market_inventory():
    env = make("kaggriculture", configuration={"episodeSteps": 50, "seed": 11})
    env.run(["pass", "pass"])
    j = env.toJSON()
    market = j["steps"][-1][0]["observation"]["market"]
    # After many turns of town center consumption (every 2 turns), wheat
    # inventory must have dropped from its initial I0.
    assert market["inventory"]["WHEAT"] < MARKET_PARAMS["WHEAT"]["I0"]


# --- Observation visibility -------------------------------------------------

def test_opponent_private_state_is_per_agent():
    """Each agent's observation should carry only their own private dict."""
    def buyer(obs):
        if obs.get("step", 0) == 0:
            return {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]}
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env = make("kaggriculture", configuration={"episodeSteps": 5, "startingMoney": 50})
    env.run([buyer, "pass"])
    j = env.toJSON()
    # In step 1 (after the buy), player 0's private has 1 wheat seed.
    p0_priv = j["steps"][1][0]["observation"]["private"]
    p1_priv = j["steps"][1][1]["observation"]["private"]
    assert p0_priv["seeds"]["WHEAT"] == 1
    assert p1_priv["seeds"]["WHEAT"] == 0


# --- Sanity over the static tables ------------------------------------------

def test_crop_table_has_expected_crops():
    assert set(CROPS) == {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"}


def test_animal_table_has_expected_animals():
    assert set(ANIMALS) == {"GOOSE", "COW", "SHEEP"}


def test_products_table_includes_animal_products_and_fertilizer():
    for p in ("EGG", "MILK", "WOOL", "FERTILIZER"):
        assert p in PRODUCTS


# --- Balance changes --------------------------------------------------------

def test_no_shop_consumes_melon():
    for shop, products in SHOPS.items():
        assert "MELON" not in products, f"{shop} should not consume MELON"


def test_town_center_excludes_fertilizer():
    assert "FERTILIZER" not in TOWN_CENTER_PRODUCTS
    # All other PRODUCTS are still consumed by the town center.
    for p in PRODUCTS:
        if p == "FERTILIZER":
            continue
        assert p in TOWN_CENTER_PRODUCTS


def test_town_center_does_not_drain_fertilizer_inventory():
    """Run a short episode and confirm fertilizer inventory never drops below
    its initial I0 from town/shop consumption (no one consumes it)."""
    env = make("kaggriculture", configuration={"episodeSteps": 100, "seed": 7})
    env.run(["pass", "pass"])
    j = env.toJSON()
    initial = MARKET_PARAMS["FERTILIZER"]["I0"]
    for step in j["steps"]:
        inv = step[0]["observation"]["market"]["inventory"]["FERTILIZER"]
        assert inv >= initial, f"FERTILIZER inventory dropped to {inv} (initial {initial})"


def test_market_orders_capped_at_default_ten():
    """An 11th market order in a single turn is silently dropped."""
    def buyer(obs):
        if obs.get("step", 0) == 0:
            # 11 BUY_SEED orders for cheap wheat seeds (cost 10 each).
            return {
                "farmer": ["PASS"],
                "hands": [],
                "market": [["BUY_SEED", "WHEAT", 1]] * 11,
            }
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env = make("kaggriculture", configuration={"episodeSteps": 5, "startingMoney": 1000})
    env.run([buyer, "pass"])
    j = env.toJSON()
    p0_priv = j["steps"][1][0]["observation"]["private"]
    assert p0_priv["seeds"]["WHEAT"] == 10  # 11th order dropped


def test_market_order_limit_is_configurable():
    def buyer(obs):
        if obs.get("step", 0) == 0:
            return {
                "farmer": ["PASS"],
                "hands": [],
                "market": [["BUY_SEED", "WHEAT", 1]] * 5,
            }
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env = make(
        "kaggriculture",
        configuration={"episodeSteps": 5, "startingMoney": 1000, "maxMarketOrdersPerTurn": 3},
    )
    env.run([buyer, "pass"])
    j = env.toJSON()
    p0_priv = j["steps"][1][0]["observation"]["private"]
    assert p0_priv["seeds"]["WHEAT"] == 3


