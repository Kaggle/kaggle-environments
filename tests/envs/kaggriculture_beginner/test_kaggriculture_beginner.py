from kaggle_environments import make
from kaggle_environments.envs.kaggriculture_beginner.kaggriculture_beginner import (
    CROPS,
    _apply_farmer_action,
    _daily_refresh,
    _decay_plants,
    _new_farm,
    _try_harvest,
)


def test_episode_completes():
    env = make("kaggriculture_beginner", configuration={"episodeSteps": 50})
    env.run(["pass", "pass"])
    j = env.toJSON()
    assert j["name"] == "kaggriculture_beginner"
    assert j["statuses"] == ["DONE", "DONE"]


def test_pass_keeps_starting_money():
    env = make(
        "kaggriculture_beginner",
        configuration={"episodeSteps": 50, "startingMoney": 200},
    )
    env.run(["pass", "pass"])
    j = env.toJSON()
    assert j["rewards"] == [200.0, 200.0]


def test_buy_seed_reduces_money_and_adds_seed():
    def buyer(obs):
        if obs.get("step", 0) == 0:
            return {"farmer": ["PASS"], "market": [["BUY_SEED", "WHEAT", 3]]}
        return {"farmer": ["PASS"], "market": []}

    env = make(
        "kaggriculture_beginner",
        configuration={"episodeSteps": 5, "startingMoney": 100},
    )
    env.run([buyer, "pass"])
    j = env.toJSON()
    assert j["rewards"] == [70.0, 100.0]
    final_obs = j["steps"][-1][0]["observation"]
    assert final_obs["farms"][0]["seeds"]["WHEAT"] == 3


def test_buy_seed_rejected_if_too_expensive():
    def overspender(obs):
        if obs.get("step", 0) == 0:
            return {"farmer": ["PASS"], "market": [["BUY_SEED", "MELON", 99]]}
        return {"farmer": ["PASS"], "market": []}

    env = make(
        "kaggriculture_beginner",
        configuration={"episodeSteps": 3, "startingMoney": 50},
    )
    env.run([overspender, "pass"])
    j = env.toJSON()
    assert j["rewards"][0] == 50.0


def test_plant_consumes_seed_and_occupies_tile():
    farm = _new_farm(5, 100)
    farm["seeds"]["WHEAT"] = 2
    _apply_farmer_action(farm, ["PLANT", "WHEAT"], 5, 0)
    assert farm["seeds"]["WHEAT"] == 1
    assert farm["tiles"][4][4] is not None
    assert farm["tiles"][4][4]["crop"] == "WHEAT"


def test_plant_requires_seed():
    farm = _new_farm(5, 100)
    _apply_farmer_action(farm, ["PLANT", "WHEAT"], 5, 0)
    assert farm["tiles"][4][4] is None


def test_plant_dies_after_two_unwatered_days():
    farm = _new_farm(5, 100)
    farm["seeds"]["WHEAT"] = 1
    _apply_farmer_action(farm, ["PLANT", "WHEAT"], 5, 0)
    _daily_refresh(farm)
    assert farm["tiles"][4][4] is not None
    _daily_refresh(farm)
    assert farm["tiles"][4][4] is None


def test_watering_keeps_plant_alive():
    farm = _new_farm(5, 100)
    farm["seeds"]["WHEAT"] = 1
    _apply_farmer_action(farm, ["PLANT", "WHEAT"], 5, 0)
    for _ in range(5):
        _apply_farmer_action(farm, ["WATER"], 5, 0)
        _daily_refresh(farm)
    assert farm["tiles"][4][4] is not None


def test_water_is_idempotent_within_day():
    farm = _new_farm(5, 100)
    farm["seeds"]["WHEAT"] = 1
    _apply_farmer_action(farm, ["PLANT", "WHEAT"], 5, 0)
    initial = farm["tiles"][4][4]["yield_units"]
    # Water 5 times on day 2 (inside wheat's bonus window). Only the first
    # call counts -- yield_units bumps by exactly 1.
    for _ in range(5):
        _apply_farmer_action(farm, ["WATER"], 5, 2)
    assert farm["tiles"][4][4]["yield_units"] == initial + 1


def test_watering_outside_bonus_window_gives_no_yield_bonus():
    farm = _new_farm(5, 100)
    farm["seeds"]["WHEAT"] = 1
    _apply_farmer_action(farm, ["PLANT", "WHEAT"], 5, 0)
    # Wheat bonus window is [ceil(4/2), 4] = [2, 4]. Day 0 and day 1 are out.
    _apply_farmer_action(farm, ["WATER"], 5, 0)
    assert farm["tiles"][4][4]["yield_units"] == 1
    farm["tiles"][4][4]["watered_today"] = False
    _apply_farmer_action(farm, ["WATER"], 5, 1)
    assert farm["tiles"][4][4]["yield_units"] == 1


def test_watering_in_bonus_window_increments_yield():
    farm = _new_farm(5, 100)
    farm["seeds"]["WHEAT"] = 1
    _apply_farmer_action(farm, ["PLANT", "WHEAT"], 5, 0)
    for d in (2, 3, 4):
        farm["tiles"][4][4]["watered_today"] = False
        _apply_farmer_action(farm, ["WATER"], 5, d)
    # Base 1 + 3 watered days in window = 4 (also the crop's max_yield cap).
    assert farm["tiles"][4][4]["yield_units"] == 4


def test_one_time_plant_decays_after_max_lifespan():
    farm = _new_farm(5, 100)
    farm["seeds"]["WHEAT"] = 1
    # Plant on day 0 -- max_lifespan_step = (0 + 4 + 1) * 24 = 120.
    _apply_farmer_action(farm, ["PLANT", "WHEAT"], 5, 0, turns_per_day=24)
    farm["tiles"][4][4]["yield_units"] = 4
    # Decay applies at offsets 0, 2, 4, ... after step 120.
    _decay_plants(farm, 119)
    assert farm["tiles"][4][4]["yield_units"] == 4
    _decay_plants(farm, 120)
    assert farm["tiles"][4][4]["yield_units"] == 3
    _decay_plants(farm, 121)  # odd offset, no decay
    assert farm["tiles"][4][4]["yield_units"] == 3
    _decay_plants(farm, 122)
    assert farm["tiles"][4][4]["yield_units"] == 2
    _decay_plants(farm, 124)
    assert farm["tiles"][4][4]["yield_units"] == 1
    _decay_plants(farm, 126)
    assert farm["tiles"][4][4] is None


def test_ongoing_plant_does_not_decay_before_max_production():
    farm = _new_farm(5, 100)
    farm["seeds"]["TOMATO"] = 1
    _apply_farmer_action(farm, ["PLANT", "TOMATO"], 5, 0, turns_per_day=24)
    # No production has happened, so max_lifespan_step is still -1 and decay
    # is a no-op for this plant.
    for step in range(0, 1000, 2):
        _decay_plants(farm, step)
    assert farm["tiles"][4][4] is not None


def test_ongoing_plant_produces_each_interval_until_cap():
    """Tomato: first_yield_day=8, interval=1, max_yield=4. Plant on day 0,
    water it daily, refresh end-of-day for 12 days."""
    farm = _new_farm(5, 100)
    farm["seeds"]["TOMATO"] = 1
    _apply_farmer_action(farm, ["PLANT", "TOMATO"], 5, 0, turns_per_day=24)
    for d in range(0, 12):
        _apply_farmer_action(farm, ["WATER"], 5, d)
        _daily_refresh(farm, d, 24)
        if farm["tiles"][4][4] is None:
            break
    tile = farm["tiles"][4][4]
    assert tile is not None
    # First production at end of day 7 (next_day=8), then daily thereafter
    # until cap of 4 is hit at end of day 10 (next_day=11).
    assert tile["total_produced"] == 4
    assert tile["yield_units"] == 4
    assert tile["max_lifespan_step"] == (11 + 1) * 24


def test_ongoing_plant_decays_after_cap_reached():
    """Run a strawberry to its production cap, then watch decay drain the
    remaining yield."""
    farm = _new_farm(5, 100)
    farm["seeds"]["STRAWBERRY"] = 1
    _apply_farmer_action(farm, ["PLANT", "STRAWBERRY"], 5, 0, turns_per_day=24)
    # Strawberry: first_yield_day=10, interval=2, max_yield=4.
    # Productions at end of days 9, 11, 13, 15 (next_day = 10, 12, 14, 16).
    for d in range(0, 16):
        _apply_farmer_action(farm, ["WATER"], 5, d)
        _daily_refresh(farm, d, 24)
    tile = farm["tiles"][4][4]
    assert tile is not None
    assert tile["total_produced"] == 4
    assert tile["yield_units"] == 4
    # max_lifespan_step = (16 + 1) * 24 = 408 -- start of day 17.
    assert tile["max_lifespan_step"] == 408
    _decay_plants(farm, 408)
    assert farm["tiles"][4][4]["yield_units"] == 3
    _decay_plants(farm, 410)
    _decay_plants(farm, 412)
    _decay_plants(farm, 414)
    assert farm["tiles"][4][4] is None


def test_ongoing_harvest_yields_one_unit_and_keeps_plant():
    farm = _new_farm(5, 100)
    farm["seeds"]["TOMATO"] = 1
    _apply_farmer_action(farm, ["PLANT", "TOMATO"], 5, 0, turns_per_day=24)
    farm["tiles"][4][4]["yield_units"] = 3
    farm["tiles"][4][4]["total_produced"] = 3
    result = _try_harvest(farm, 4, 4, 8)
    assert result == ("TOMATO", 1)
    assert farm["tiles"][4][4] is not None
    assert farm["tiles"][4][4]["yield_units"] == 2


def test_movement_stays_in_bounds():
    farm = _new_farm(5, 100)
    assert farm["farmer"] == [4, 4]
    _apply_farmer_action(farm, ["EAST"], 5, 0)
    assert farm["farmer"] == [4, 4]
    _apply_farmer_action(farm, ["WEST"], 5, 0)
    assert farm["farmer"] == [3, 4]
    _apply_farmer_action(farm, ["NORTH"], 5, 0)
    assert farm["farmer"] == [3, 3]


def test_harvest_yields_money_via_full_episode():
    """End-to-end: buy a wheat seed, plant it, water once per day, harvest at
    day 4. Player 0 should end with starting - seed_cost + harvest_revenue."""

    def planter(obs):
        step = obs.get("step", 0)
        # Day cycle: 24 steps. Schedule keyed by step.
        if step == 0:
            return {"farmer": ["PASS"], "market": [["BUY_SEED", "WHEAT", 1]]}
        if step == 1:
            return {"farmer": ["PLANT", "WHEAT"], "market": []}
        if step % 24 == 2:
            return {"farmer": ["WATER"], "market": []}
        if step == 100:
            return {"farmer": ["HARVEST"], "market": []}
        return {"farmer": ["PASS"], "market": []}

    env = make(
        "kaggriculture_beginner",
        configuration={"episodeSteps": 120, "startingMoney": 100},
    )
    env.run([planter, "pass"])
    j = env.toJSON()
    # Wheat seed=10, price=25. With 5 days of watering bonus is capped to 3, so
    # yield = 4 units = $100. Player 0: 100 - 10 + 100 = 190.
    assert j["rewards"][0] == 190.0
    assert j["rewards"][1] == 100.0


def test_seed_is_scrubbed_from_configuration():
    env = make(
        "kaggriculture_beginner",
        configuration={"episodeSteps": 5, "seed": 42},
    )
    env.run(["pass", "pass"])
    j = env.toJSON()
    assert j["configuration"].get("seed") is None
    assert env.info.get("seed") == 42


def test_renderer():
    env = make("kaggriculture_beginner", configuration={"episodeSteps": 5})
    env.run(["pass", "pass"])
    out = env.render(mode="ansi")
    assert isinstance(out, str)
    assert len(out) > 0


def test_random_agent_runs_full_episode():
    env = make("kaggriculture_beginner", configuration={"episodeSteps": 200})
    env.run(["random", "random"])
    j = env.toJSON()
    assert j["statuses"] == ["DONE", "DONE"]


def test_crop_table_has_all_expected_crops():
    assert set(CROPS) == {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"}
