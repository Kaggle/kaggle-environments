import json
import math
import random

import pytest

from kaggle_environments import make
from kaggle_environments.envs.kargo.city import (
    City,
    bpr_multiplier,
    congestion_level,
    time_of_day_shape,
)
from kaggle_environments.envs.kargo.constants import (
    ABANDON_FEE_PER_UNIT,
    ARTERIAL_KMH,
    BLOCKS_PER_DAY,
    DEADLINE_MINUTES,
    DISTRICTS,
    EDGE_KM,
    EPISODE_STEPS,
    FILL_CEILING,
    INDEX_RANGE,
    ROAD_CLASSES,
    SERVICE_INTERVAL_KM,
    SHIFT_MINUTES,
    STARTING_FLEET,
    STEPS_PER_DAY,
    TARGET_NET_PER_TRUCK_DAY,
    VEHICLES,
)
from kaggle_environments.envs.kargo.dispatch import _wear
from kaggle_environments.envs.kargo.fleet import (
    book_value,
    effective_stats,
    make_driver,
    make_truck,
    service_multiplier,
    speed_multiplier,
)
from kaggle_environments.envs.kargo.freight import (
    LotBoard,
    draw_manifest,
    expected_cost,
    solve_reserve,
    solve_truck_day,
)
from kaggle_environments.envs.kargo.kargo import phase_of
from kaggle_environments.envs.kargo.market import accrue, net_worth
from kaggle_environments.envs.kargo.shipper import Shipper

SHORT = {"episodeSteps": 25, "days": 3, "seed": 7}


@pytest.fixture
def city():
    return City(random.Random(7))


@pytest.fixture
def board(city):
    return LotBoard(city, random.Random(7))


# --- Smoke / lifecycle ------------------------------------------------------


def test_episode_completes():
    env = make("kargo", configuration=SHORT)
    env.run(["greedy", "greedy"])
    j = env.toJSON()
    assert j["name"] == "kargo"
    assert j["statuses"] == ["DONE", "DONE"]


def test_four_players():
    env = make("kargo", configuration=SHORT)
    env.run(["greedy", "greedy", "random", "idle"])
    assert env.toJSON()["statuses"] == ["DONE"] * 4


def test_all_builtin_agents_run():
    for agent in ("idle", "random", "greedy"):
        env = make("kargo", configuration=SHORT)
        env.run([agent, agent])
        assert env.toJSON()["statuses"] == ["DONE", "DONE"], agent


def test_full_length_episode():
    env = make("kargo", configuration={"seed": 3})
    env.run(["greedy", "greedy"])
    j = env.toJSON()
    assert j["statuses"] == ["DONE", "DONE"]
    assert len(j["steps"]) == EPISODE_STEPS


def test_rewards_are_net_worth():
    env = make("kargo", configuration=SHORT)
    env.run(["greedy", "greedy"])
    final = env.steps[-1]
    for s in final:
        pub = final[0].observation["public"][s.observation["player"]]
        assert s.reward == pytest.approx(pub["net_worth"], abs=1.0)


def test_idle_agent_loses_money_slowly():
    """Idle players lose net worth to overhead and holding cost."""
    env = make("kargo", configuration=SHORT)
    env.run(["idle", "idle"])
    book = sum(VEHICLES[v]["buy"] for v in STARTING_FLEET)
    assert all(s.reward < 12000 + book for s in env.steps[-1])


def test_crashing_agent_errors_out():
    def boom(obs, cfg=None):
        raise RuntimeError("bad agent")

    env = make("kargo", configuration=SHORT)
    env.run([boom, "greedy"])
    j = env.toJSON()
    assert j["statuses"][0] in ("ERROR", "INVALID")
    assert j["statuses"][1] == "DONE"
    assert j["rewards"][0] is None


def test_malformed_action_is_ignored():
    """An object-typed action coerces to {} rather than killing the agent."""
    env = make("kargo", configuration=SHORT)
    env.run([lambda obs, cfg=None: "not-an-object", "greedy"])
    assert env.toJSON()["statuses"] == ["DONE", "DONE"]


def test_garbage_inside_a_well_formed_action_is_survivable():
    def junk(obs, cfg=None):
        return {
            "bids": [["nope", -1]],
            "fleet": [["NOT_AN_OP"]],
            "trucks": {"ghost": {"route": ["nowhere"]}},
            "max_lots": "lots",
        }

    env = make("kargo", configuration=SHORT)
    env.run([junk, "greedy"])
    assert env.toJSON()["statuses"] == ["DONE", "DONE"]


def test_renderer():
    env = make("kargo", configuration=SHORT)
    env.run(["greedy", "greedy"])
    out = env.render(mode="ansi")
    assert isinstance(out, str) and "Player 0" in out


# --- Event stream (the visualizer's contract) -------------------------------


def _events(env, pid=None):
    return [
        e
        for s in env.steps
        for i, side in enumerate(s)
        for e in (side.observation.get("private") or {}).get("events", [])
        if pid is None or i == pid
    ]


def test_events_are_published():
    env = make("kargo", configuration=SHORT)
    env.run(["greedy", "greedy"])
    evs = _events(env)
    assert evs, "no events reached the replay"
    for e in evs:
        assert {"kind", "player", "truck", "node", "day", "minute"} <= set(e)


def test_events_are_not_republished():
    """Each observation carries only its own step's events."""
    env = make("kargo", configuration=SHORT)
    env.run(["greedy", "greedy"])
    keys = [(e["kind"], e.get("address"), e["minute"], e["player"], e["truck"]) for e in _events(env)]
    assert len(keys) == len(set(keys))


def test_events_do_not_leak_across_players():
    """A player's events contain only their own trucks."""
    env = make("kargo", configuration=SHORT)
    env.run(["greedy", "greedy"])
    for pid in (0, 1):
        assert all(e["player"] == pid for e in _events(env, pid))


def test_event_tallies_match_the_day_report():
    """Event tallies match the day report, per player per day."""
    env = make("kargo", configuration=SHORT)
    env.run(["greedy", "greedy"])

    tally = {}
    for e in _events(env):
        row = tally.setdefault((e["player"], e["day"]), dict.fromkeys(("delivered", "late", "failed", "refused"), 0))
        n = e.get("packages", 0)
        if e["kind"] == "DELIVER":
            row["delivered"] += n
            row["late"] += n if e.get("late") else 0
        elif e["kind"] == "REFUSED":
            row["refused"] += n
        else:
            row["failed"] += n

    reports = {}
    for s in env.steps:
        for pid, pub in enumerate(s[0].observation["public"]):
            for r in pub["results"]:
                reports[(pid, r["day"])] = r

    assert reports, "no day reports published"
    for key, report in reports.items():
        counted = tally.get(key, dict.fromkeys(("delivered", "late", "failed", "refused"), 0))
        assert counted == {f: report[f] for f in counted}, f"day {key} disagrees"


# --- Seeding and determinism ------------------------------------------------


def test_seed_is_scrubbed_from_configuration():
    env = make("kargo", configuration={"episodeSteps": 10, "days": 2, "seed": 12345})
    env.run(["greedy", "greedy"])
    assert env.toJSON()["configuration"].get("seed") is None
    assert env.info.get("seed") == 12345


def test_same_seed_same_episode():
    a, b = (make("kargo", configuration=SHORT) for _ in range(2))
    a.run(["greedy", "greedy"])
    b.run(["greedy", "greedy"])
    assert a.toJSON()["rewards"] == b.toJSON()["rewards"]


def test_different_seed_different_city():
    a = City(random.Random(1))
    b = City(random.Random(2))
    assert a.warehouse_of != b.warehouse_of or a.district_of != b.district_of


# --- Phase clock ------------------------------------------------------------


def test_phase_cycle():
    assert phase_of(0)[0] == "RESET"
    got = [phase_of(s)[0] for s in range(1, 1 + STEPS_PER_DAY)]
    assert got == ["CAPEX", "LABOR", "CONTRACTS"] + ["DRIVING"] * BLOCKS_PER_DAY


def test_day_advances_every_eight_steps():
    assert phase_of(1)[1] == 0
    assert phase_of(STEPS_PER_DAY)[1] == 0
    assert phase_of(STEPS_PER_DAY + 1)[1] == 1


def test_driving_blocks_number_zero_to_four():
    blocks = [phase_of(s)[2] for s in range(4, 4 + BLOCKS_PER_DAY)]
    assert blocks == list(range(BLOCKS_PER_DAY))


def test_episode_steps_covers_sixty_days():
    assert EPISODE_STEPS == 60 * STEPS_PER_DAY + 1


# --- City -------------------------------------------------------------------


def test_arterial_free_flow_is_forty_five_kmh():
    """Deadheads are priced at ARTERIAL_KMH; the road table must agree."""
    assert ARTERIAL_KMH == pytest.approx(45.0)
    assert EDGE_KM / (ROAD_CLASSES["ARTERIAL"]["t_free"] / 60.0) == pytest.approx(45.0)


def test_road_classes_ordered_by_speed():
    t = [ROAD_CLASSES[c]["t_free"] for c in ("ARTERIAL", "COLLECTOR", "LOCAL")]
    assert t == sorted(t)
    caps = [ROAD_CLASSES[c]["capacity"] for c in ("ARTERIAL", "COLLECTOR", "LOCAL")]
    assert caps == sorted(caps, reverse=True)


def test_grid_shape(city):
    labelled = sum(len(v) for v in city.district_nodes.values())
    assert labelled == city.size**2
    assert set(city.district_nodes) == set(DISTRICTS)
    assert len(city.warehouse_of) == 4


def test_congestion_buckets_are_monotone():
    levels = [congestion_level(vc) for vc in (0.0, 0.8, 1.1, 1.3, 1.6, 2.5)]
    assert levels == ["FREE", "LIGHT", "MODERATE", "HEAVY", "SEVERE", "GRIDLOCK"]


def test_bpr_is_increasing_and_starts_at_one():
    assert bpr_multiplier(0.0) == pytest.approx(1.0)
    vals = [bpr_multiplier(v) for v in (0.0, 0.5, 1.0, 1.5)]
    assert vals == sorted(vals)


def test_time_of_day_has_two_peaks():
    morning = time_of_day_shape(30)
    midday = time_of_day_shape(240)
    evening = time_of_day_shape(570)
    assert morning > midday and evening > midday


def test_node_km_is_manhattan(city):
    assert city.node_km(0, 1) == pytest.approx(EDGE_KM)
    assert city.node_km(0, city.size) == pytest.approx(EDGE_KM)
    assert city.node_km(0, city.size + 1) == pytest.approx(2 * EDGE_KM)


def test_forecast_names_a_weather(city):
    from kaggle_environments.envs.kargo.constants import WEATHER

    assert city.forecast_for(1, random.Random(1)) in WEATHER


def test_forecast_predicts_the_day_it_names():
    """Right about 75% of the time, against the weather that day actually gets."""
    hits = n = 0
    for seed in range(40):
        c = City(random.Random(seed))
        rng = random.Random(seed + 100)
        for day in range(1, 30):
            f = c.forecast_for(day, rng)
            c.reset_day(day, rng)
            hits += f == c.weather
            n += 1
    assert 0.70 < hits / n < 0.95


def test_forecast_is_fixed_per_day(city):
    rng = random.Random(4)
    assert len({city.forecast_for(5, rng) for _ in range(20)}) == 1


# --- Freight ----------------------------------------------------------------


def test_truck_day_fits_the_shift():
    """A lot is one shift of work; more deadhead means fewer stops."""
    near = solve_truck_day("MIDTOWN", 5.0)
    far = solve_truck_day("MIDTOWN", 70.0)
    assert near["stops"] > far["stops"] > 0


def test_truck_day_density_ordering():
    """Downtown is dense in packages per stop, industrial is sparse."""
    dt = solve_truck_day("DOWNTOWN", 10.0)
    ind = solve_truck_day("INDUSTRIAL", 10.0)
    assert dt["pkg_per_stop"] > ind["pkg_per_stop"]


def test_reserve_clears_cost_plus_target():
    td = solve_truck_day("MIDTOWN", 10.0)
    cost = expected_cost("MIDTOWN", td, 10.0)
    reserve = solve_reserve("MIDTOWN", td, 10.0)
    assert reserve >= cost
    assert reserve - cost <= TARGET_NET_PER_TRUCK_DAY + 1e-6


def test_pair_is_the_unit_of_pricing(board):
    """Same district, different dock, different lot size."""
    by_district = {}
    for (wid, district), td in board.truck_days.items():
        by_district.setdefault(district, []).append(td["stops"])
    assert any(len(set(v)) > 1 for v in by_district.values())


def test_anchor_sits_in_its_own_district(board):
    for (wid, district), anchor in board.anchors.items():
        nodes = board.city.district_nodes[district]
        assert not nodes or anchor in nodes


def _shipper(board, capacity=12):
    return Shipper(random.Random(3), list(board.truck_days), capacity)


def test_posted_lots_concentrate_on_few_pairs(board):
    """Tonight's freight lands on a handful of territories, not all 24."""
    lots, accounts = _shipper(board).post(board, 0, 0.0, 12)
    pairs = {(x["warehouse"], x["district"]) for x in lots + accounts}
    assert 0 < len(pairs) <= 6


def test_manifest_covers_every_package(board, city):
    lots, _ = _shipper(board).post(board, 0, 0.0, 12)
    for lot in lots:
        man = draw_manifest(lot, city, random.Random(11))
        assert sum(a["packages"] for a in man["addresses"]) == lot["packages"]
        assert man["segments"]


def test_manifest_segments_share_one_anchor(board, city):
    """Intra-lot movement is interior-street driving, not an arterial hop."""
    lots, _ = _shipper(board).post(board, 0, 0.0, 12)
    for lot in lots:
        man = draw_manifest(lot, city, random.Random(11))
        assert len({s["node"] for s in man["segments"]}) == 1


# --- Fleet ------------------------------------------------------------------


def test_driver_degrades_over_the_shift():
    d = make_driver("d0", random.Random(4))
    fresh = effective_stats(d, 0, 0)
    tired = effective_stats(d, SHIFT_MINUTES, 120)
    assert service_multiplier(tired) >= service_multiplier(fresh)
    assert speed_multiplier(tired) >= speed_multiplier(fresh)


def test_book_value_depreciates():
    new = make_truck("T1", "VAN", random.Random(5))
    old = make_truck("T2", "VAN", random.Random(5))
    old["age_days"] = 900
    assert book_value(new) > book_value(old) >= 0


def test_careless_driver_wears_the_truck_faster():
    careful, careless = make_truck("T1", "VAN", random.Random(5)), make_truck("T2", "VAN", random.Random(5))
    _wear(careful, 10.0, {"CARE": 80})
    _wear(careless, 10.0, {"CARE": 20})
    assert careful["odometer"] == careless["odometer"] == 10.0
    assert careless["km_since_service"] > careful["km_since_service"]


def _worn_truck_env():
    """Day 0's CAPEX, with player 0's T1 already at the service interval."""
    env = make("kargo", configuration=SHORT)
    env.reset(2)
    env.kargo["players"][0]["trucks"]["T1"]["km_since_service"] = SERVICE_INTERVAL_KM
    return env


def _status(env, tid, pid=0):
    return next(t["status"] for t in env.state[0].observation["public"][pid]["fleet"] if t["id"] == tid)


def test_service_due_grounds_the_truck_at_close():
    env = _worn_truck_env()
    for _ in range(STEPS_PER_DAY - 1):
        env.step([{}, {}])
        assert _status(env, "T1") != "DISABLED"
    env.step([{}, {}])  # 18:00
    assert _status(env, "T1") == "DISABLED"
    events = env.state[0].observation["private"]["events"]
    assert any(e["kind"] == "SERVICE_DUE" and e["truck"] == "T1" for e in events)


def test_servicing_that_evening_keeps_the_truck_working():
    env = _worn_truck_env()
    for _ in range(STEPS_PER_DAY):
        env.step([{}, {}])
    cash = env.kargo["players"][0]["cash"]
    env.step([{"fleet": [["SERVICE", "T1"]]}, {}])
    truck = env.kargo["players"][0]["trucks"]["T1"]
    assert truck["status"] == "IDLE" and truck["km_since_service"] == 0.0
    assert env.kargo["players"][0]["cash"] == pytest.approx(cash - VEHICLES["VAN"]["service_cost"])


def test_unserviced_truck_sits_out_the_next_day():
    env = _worn_truck_env()
    for _ in range(STEPS_PER_DAY):
        env.step([{}, {}])
    # Win freight for day 1 so there is work T1 could have taken.
    for _ in range(2):
        env.step([{}, {}])
    listings = env.state[0].observation["market"]["listings"]
    env.step([{"bids": [[lot["id"], lot["reserve"]] for lot in listings]}, {}])
    assert not env.kargo["players"][0]["trucks"]["T1"]["lots"]
    for _ in range(BLOCKS_PER_DAY):
        env.step([{}, {}])
    assert _status(env, "T1") == "DISABLED"


def test_vehicle_ladder_is_monotone():
    order = ["VAN", "STEP"]
    for key in ("capacity", "buy", "rent_day", "fuel_per_min"):
        vals = [VEHICLES[v][key] for v in order]
        assert vals == sorted(vals), key


# --- Money ------------------------------------------------------------------


def _bare_player():
    return {"cash": 1000.0, "debt": 0.0, "trucks": {}, "drivers": {}}


def test_overhead_is_charged_daily():
    p = _bare_player()
    accrue(p, 0)
    assert p["cash"] < 1000.0


def test_shortfall_becomes_debt():
    p = _bare_player()
    p["cash"] = 0.0
    accrue(p, 0)
    assert p["cash"] == 0.0 and p["debt"] > 0


def test_cash_repays_debt_before_it_compounds():
    p = _bare_player()
    p["cash"] = 5000.0
    p["debt"] = 1000.0
    accrue(p, 0)
    assert p["debt"] < 1000.0


def test_net_worth_subtracts_debt():
    p = _bare_player()
    before = net_worth(p)
    p["debt"] = 500.0
    assert net_worth(p) == pytest.approx(before - 500.0)


# --- Dispatch invariants ----------------------------------------------------


def _run(agents=("greedy", "greedy"), **cfg):
    env = make("kargo", configuration={**SHORT, **cfg})
    env.run(list(agents))
    return env


def test_one_truck_serves_one_territory():
    """No van loads at two docks -- the pair is a hard constraint."""
    env = _run()
    for player in env.kargo["players"]:
        for truck in player["trucks"].values():
            lots = [player["lots"][lid] for lid in truck.get("lots", []) if lid in player["lots"]]
            assert len({(x["warehouse"], x["district"]) for x in lots}) <= 1


def test_trucks_stay_under_the_fill_ceiling():
    env = _run()
    for player in env.kargo["players"]:
        for truck in player["trucks"].values():
            assert truck.get("fill", 0.0) <= FILL_CEILING + 1e-6


def test_trucks_stay_within_capacity():
    env = _run()
    for player in env.kargo["players"]:
        for truck in player["trucks"].values():
            units = sum(
                player["manifest"]["addresses"][a]["packages"]
                * DISTRICTS[player["manifest"]["addresses"][a]["district"]]["pkg_units"]
                for a in truck["carrying"]
                if a in player["manifest"]["addresses"] and "district" in player["manifest"]["addresses"][a]
            )
            assert units <= VEHICLES[truck["type"]]["capacity"] + 1e-6


def test_no_truck_drives_past_the_day():
    env = _run()
    for player in env.kargo["players"]:
        for truck in player["trucks"].values():
            assert truck["clock"] <= 601.0


def test_greedy_delivers_most_of_what_it_wins():
    env = _run(days=5, episodeSteps=41)
    report = env.kargo["players"][0]["results"]
    delivered = sum(r["delivered"] for r in report)
    failed = sum(r["failed"] for r in report)
    assert delivered > 0
    assert failed / (delivered + failed) < 0.35


def test_greedy_beats_idle():
    """Greedy beats idle over a full episode on every seed."""
    margins = []
    for seed in range(1, 5):
        env = _run(agents=("greedy", "idle"), episodeSteps=481, seed=seed)
        greedy, idle = (s.reward for s in env.steps[-1])
        margins.append(greedy - idle)
    assert all(m > 5000 for m in margins), margins


def test_deliveries_land_before_the_deadline_mostly():
    env = _run(days=5, episodeSteps=41)
    delivers = [e for e in env.kargo["events"] if e["kind"] == "DELIVER"]
    assert delivers
    late = [e for e in delivers if e["minute"] > DEADLINE_MINUTES]
    assert len(late) / len(delivers) < 0.5


# --- Observation surface ----------------------------------------------------


def test_private_state_is_not_shared():
    env = make("kargo", configuration=SHORT)
    env.run(["greedy", "greedy"])
    for step in env.steps:
        for i, s in enumerate(step):
            if i and "private" in s.observation:
                assert s.observation["private"].get("player", i) == i


def test_every_agent_receives_the_shared_fields():
    """Shared fields are stored once, on state[0], and merged in at act time."""
    seen = []

    def probe(obs, cfg=None):
        seen.append(all(k in obs for k in ("city", "market", "traffic", "public", "history", "phase")))
        return {}

    env = make("kargo", configuration=SHORT)
    env.run(["idle", probe])
    assert seen and all(seen)
    assert "market" not in env.steps[5][1].observation  # not duplicated in the replay


def test_rival_wages_are_never_public():
    env = make("kargo", configuration=SHORT)
    env.run(["greedy", "greedy"])
    for step in env.steps:
        for pub in step[0].observation.get("public", []):
            for d in pub["drivers"]:
                assert "wage" not in d


# --- The shipper ------------------------------------------------------------


def test_demand_grows_over_the_episode(board):
    sh = _shipper(board)
    early = sum(sh.demand(d) for d in range(0, 7))
    late = sum(sh.demand(d) for d in range(53, 60))
    assert late > early


def test_demand_ignores_the_current_fleet(board):
    """Demand scales with the starting fleet, not the current one."""
    a, b = _shipper(board), _shipper(board)
    assert a.post(board, 0, 0.0, 12)[0][0]["packages"] == b.post(board, 0, 0.0, 40)[0][0]["packages"]


def test_price_rises_when_the_field_is_short_and_falls_when_it_overbuilds(board):
    short, glut = _shipper(board), _shipper(board)
    for day in range(10):
        short.post(board, day, 0.0, 2)
        glut.post(board, day, 0.0, 60)
    assert short.market > 0 > glut.market


def test_index_stays_in_range(board):
    sh = _shipper(board)
    for day in range(60):
        sh.post(board, day, 0.0, 1)
    lo, hi = INDEX_RANGE
    assert all(lo <= sh.index(p) <= hi for p in sh.pairs)


def test_unsold_lot_comes_back_marked_up(board):
    sh = _shipper(board)
    lots, accounts = sh.post(board, 0, 0.0, 12)
    lot = lots[0]
    sh.observe_auction([lot], [], [])
    again, _ = sh.post(board, 1, 0.0, 12)
    retried = [x for x in again if x["retry"] == 1]
    assert retried and retried[0]["packages"] == lot["packages"]
    assert retried[0]["_patience"] == lot["_patience"]


def test_undelivered_packages_come_back(board):
    sh = _shipper(board)
    lot = sh.post(board, 0, 0.0, 12)[0][0]
    sh.observe_failed(lot, 3)
    again, _ = sh.post(board, 1, 0.0, 12)
    assert any(x["retry"] == 1 and x["packages"] == 3 for x in again)


def test_patience_runs_out(board):
    sh = _shipper(board)
    lot = dict(sh.post(board, 0, 0.0, 12)[0][0], _patience=1, retry=1)
    sh.observe_failed(lot, 5)
    again, _ = sh.post(board, 1, 0.0, 12)
    assert sh.lost == 5
    assert not any(x["retry"] == 2 for x in again)


def test_hidden_market_state_is_not_published():
    env = make("kargo", configuration=SHORT)
    env.reset(2)
    for _ in range(4):
        env.step([{}, {}])
    obs = env.state[0].observation
    lots = obs["market"]["listings"] + obs["market"]["accounts"] + obs["private"]["lots"]
    assert lots and not any(k.startswith("_") for lot in lots for k in lot)
    assert "cover_cost" not in obs["market"] and "brokered" not in obs["history"]


# --- Abandon ----------------------------------------------------------------


def _loaded_env():
    """Day 0, 08:00, with player 0 holding freight."""
    env = make("kargo", configuration=SHORT)
    env.reset(2)
    env.step([{}, {}])
    env.step([{}, {}])
    listings = env.state[0].observation["market"]["listings"]
    env.step([{"bids": [[lot["id"], lot["reserve"]] for lot in listings]}, {}])
    return env


def test_abandon_charges_a_fee_per_parcel_unit():
    env = _loaded_env()
    player = env.kargo["players"][0]
    lot = next(iter(player["lots"].values()))
    env.step([{"abandon": [lot["id"]]}, {}])
    events = [e for e in env.state[0].observation["private"]["events"] if e["kind"] == "ABANDONED"]
    fee = sum(e["cost"] for e in events)
    assert fee == pytest.approx(ABANDON_FEE_PER_UNIT * lot["parcel_units"], rel=0.02)


def test_abandon_only_in_the_first_block():
    env = _loaded_env()
    env.step([{}, {}])  # block 0 passes
    lot = next(iter(env.kargo["players"][0]["lots"].values()))
    env.step([{"abandon": [lot["id"]]}, {}])
    assert not any(e["kind"] == "ABANDONED" for e in env.state[0].observation["private"]["events"])


def test_resale_is_gone():
    env = _loaded_env()
    player = env.kargo["players"][0]
    lot = next(iter(player["lots"].values()))
    cash = player["cash"]
    env.step([{"resale": [["BROKER", lot["id"]]]}, {}])
    assert lot["id"] in player["lots"] and player["cash"] == cash


# --- Valuation and staging ---------------------------------------------------


def test_rentals_add_no_net_worth():
    env = make("kargo", configuration=SHORT)
    env.reset(2)
    env.step([{"fleet": [["RENT", "VAN"]]}, {}])
    pub = env.state[0].observation["public"]
    assert pub[0]["net_worth"] == pytest.approx(pub[1]["net_worth"])


def test_used_truck_is_booked_at_its_price():
    env = make("kargo", configuration=SHORT)
    env.reset(2)
    used = env.state[0].observation["market"]["used"][0]
    env.step([{"fleet": [["BUY_USED", used["id"]]]}, {}])
    pub = env.state[0].observation["public"]
    assert pub[0]["net_worth"] == pytest.approx(pub[1]["net_worth"], abs=1.0)


def test_stage_moves_the_truck():
    env = make("kargo", configuration=SHORT)
    env.reset(2)
    w = env.kargo
    env.step([{"stage": {"T1": "wh_3"}}, {}])
    assert w["players"][0]["trucks"]["T1"]["node"] == w["city"].warehouse_of["wh_3"]


def test_truck_working_through_the_close_is_paid():
    from kaggle_environments.envs.kargo.kargo import greedy_agent

    def greedy(obs):
        return greedy_agent(dict(obs))

    env = _loaded_env()
    w = env.kargo
    for a in w["players"][0]["manifest"]["addresses"].values():
        a["service"] *= 6
    busy = [t for t in w["players"][0]["trucks"].values() if t["carrying"]]
    assert busy
    for _ in range(BLOCKS_PER_DAY):
        env.step([greedy(s.observation) for s in env.state])
    assert all(t["worked"] > SHIFT_MINUTES for t in busy)


# --- Hardening -------------------------------------------------------------

GARBAGE = [None, 7, -1, float("nan"), float("inf"), -1e308, "x", "", "x" * 500, [], {}, [1, 2], {"a": 1}, True]


def _garbage_action(rng, phase):
    g = lambda: rng.choice(GARBAGE)  # noqa: E731
    if phase == "CAPEX":
        return {
            "fleet": [[rng.choice(["BUY", "SELL", "RENT", "BUY_USED", "SERVICE", "BREAK"]), g()] for _ in range(3)]
            + [g()],
            "fuel": [["BULK_REFUEL", g()], g()],
            "stage": {"T1": g(), "T9": "wh_0"},
        }
    if phase == "LABOR":
        return {
            "labor": [
                ["HIRE", g(), g()],
                ["WAGE", g(), g()],
                ["POACH", g(), g(), g()],
                ["FIRE", g()],
                ["ASSIGN", g(), g()],
                g(),
            ]
        }
    if phase == "CONTRACTS":
        return {"bids": [[g(), g()], g()], "standing_bids": [[g(), g(), g()]], "max_lots": g()}
    return {
        "trucks": {
            "T1": {
                "route": [g(), {"via": g()}, {"seg": g()}, "x"] * 3,
                "wait_cap": g(),
                "hold": g(),
                "on_missed_window": g(),
                "then": g(),
            },
            "T2": g(),
        },
        "abandon": [g(), g()],
    }


def test_garbage_actions_never_crash_the_episode():
    rng = random.Random(0)

    def fuzz(obs, cfg=None):
        return _garbage_action(rng, obs["phase"])

    env = make("kargo", configuration={"episodeSteps": 41, "seed": 3})
    env.run([fuzz, "greedy"])
    final = env.steps[-1]
    assert [s.status for s in final] == ["DONE", "DONE"]
    assert all(math.isfinite(s.reward) for s in final)
    # Observations stay JSON-clean; raw actions are recorded verbatim.
    for step in env.toJSON()["steps"]:
        for agent in step:
            json.dumps(agent["observation"], allow_nan=False)


def test_infinite_wage_is_clamped():
    env = make("kargo", configuration=SHORT)
    env.reset(2)
    env.step([{}, {}])
    env.step([{"labor": [["WAGE", "D0_1", float("inf")], ["WAGE", "D0_2", 1e308]]}, {}])
    assert all(math.isfinite(d["wage"]) for d in env.kargo["players"][0]["drivers"].values())


def test_negative_ask_is_dropped():
    from kaggle_environments.envs.kargo.actions import sanitize

    assert sanitize("CONTRACTS", {"bids": [["lot_1", -5], ["lot_2", 10]]})["bids"] == [["lot_2", 10.0]]


def test_route_and_lists_are_capped():
    from kaggle_environments.envs.kargo.actions import MAX_ROUTE, sanitize

    act = sanitize("DRIVING", {"trucks": {"T1": {"route": ["s"] * 100000, "hold": True}}})
    assert len(act["trucks"]["T1"]["route"]) == MAX_ROUTE


def test_unreachable_via_is_dropped_not_retried():
    env = _loaded_env()
    truck = next(t for t in env.kargo["players"][0]["trucks"].values() if t["carrying"])
    w = env.kargo
    # Cut the target off: close every edge touching node 0.
    for idx, (u, v, *_rest) in enumerate(w["city"].edges):
        if 0 in (u, v):
            w["city"].closures[idx] = (0, 10**6, "CONSTRUCTION")
    env.step([{"trucks": {truck["id"]: {"route": [{"via": 0}]}}}, {}])
    assert {"via": 0} not in truck["route"]


# --- Fairness ----------------------------------------------------------------


def test_contested_hire_goes_to_the_higher_offer_not_seat_zero():
    env = make("kargo", configuration=SHORT)
    env.reset(2)
    env.step([{}, {}])
    cand = env.kargo["candidates"][0]
    high = cand["reservation"] + 50
    env.step([{"labor": [["HIRE", cand["id"], high - 20]]}, {"labor": [["HIRE", cand["id"], high]]}])
    names = [d["name"] for d in env.kargo["players"][1]["drivers"].values()]
    assert cand["name"] in names


def test_tied_hires_split_between_seats():
    winners = set()
    for seed in range(12):
        env = make("kargo", configuration={"episodeSteps": 25, "seed": seed})
        env.reset(2)
        env.step([{}, {}])
        cand = env.kargo["candidates"][0]
        offer = cand["reservation"] + 10
        env.step([{"labor": [["HIRE", cand["id"], offer]]}, {"labor": [["HIRE", cand["id"], offer]]}])
        winners |= {
            pid
            for pid, p in enumerate(env.kargo["players"])
            if cand["name"] in [d["name"] for d in p["drivers"].values()]
        }
    assert winners == {0, 1}


def test_unassigned_drivers_draw_a_retainer():
    env = make("kargo", configuration=SHORT)
    env.reset(2)
    env.step([{}, {}])
    hires = [["HIRE", c["id"], c["reservation"] + 1] for c in env.kargo["candidates"]]
    env.step([{"labor": hires}, {}])
    for _ in range(STEPS_PER_DAY - 1):
        env.step([{}, {}])
    p0, p1 = env.kargo["players"]
    assert len(p0["drivers"]) > len(p1["drivers"])
    assert p0["day_report"]["cost"] > p1["day_report"]["cost"]


def test_auction_ignores_trucks_that_cannot_run():
    env = make("kargo", configuration=SHORT)
    env.reset(2)
    env.step([{}, {}])
    env.step([{"labor": [["FIRE", d] for d in list(env.kargo["players"][0]["drivers"])]}, {}])
    listings = env.state[0].observation["market"]["listings"]
    env.step([{"bids": [[lot["id"], lot["reserve"]] for lot in listings]}, {}])
    assert not env.kargo["players"][0]["lots"]


def test_fleet_is_capped():
    from kaggle_environments.envs.kargo.constants import MAX_FLEET

    env = make("kargo", configuration={**SHORT, "startingCash": 10**7})
    env.reset(2)
    env.step([{"fleet": [["BUY", "VAN"]] * 40}, {}])
    assert len(env.kargo["players"][0]["trucks"]) == MAX_FLEET


# --- Dock grace --------------------------------------------------------------


def test_some_docks_take_a_late_truck(board, city):
    graces = []
    for lot in _shipper(board).post(board, 0, 0.0, 12)[0] * 20:
        graces += [
            a["_grace"]
            for a in draw_manifest(lot, city, random.Random(len(graces)))["addresses"]
            if a["window_kind"] == "DOCK"
        ]
    assert graces and 0.3 < sum(1 for g in graces if g > 0) / len(graces) < 0.7


# --- City clock, replay, assignment, close -----------------------------------


def test_incidents_run_on_the_city_clock_without_trucks():
    accidents = closures = days = 0
    moved = False
    for seed in range(10):
        c = City(random.Random(seed))
        rng = random.Random(seed + 1)
        for day in range(30):
            c.reset_day(day, rng)
            accidents += sum(e["kind"] == "ACCIDENT" for e in c.incident_log)
            closures += sum(e["kind"] == "CLOSURE" for e in c.incident_log)
            moved |= any(c.noise_path[-1])
            days += 1
    assert accidents > 0 and moved
    assert 0.15 < closures / days < 0.35


def test_city_does_not_depend_on_who_asked_first():
    def city():
        c = City(random.Random(1))
        c.reset_day(3, random.Random(2))
        return c

    quiet, busy = city(), city()
    for minute in range(0, 600, 7):
        busy.route(0, 399, minute)
        busy.edge_time(minute % len(busy.edges), minute)
    for minute in (0, 240, 480):
        assert quiet.congestion_report(minute) == busy.congestion_report(minute)
        assert quiet.incident_report(minute) == busy.incident_report(minute)


def test_route_does_not_see_incidents_before_they_happen():
    c = City(random.Random(1))
    c.reset_day(0, random.Random(2))
    c.closures[5] = (300, 400, "EMERGENCY")
    assert c.edge_time(5, 350, known=200) is not None
    assert c.edge_time(5, 350) is None


def test_failed_agent_keeps_its_status():
    def broken(obs):
        raise RuntimeError("boom")

    env = make("kargo", configuration=SHORT)
    env.run([broken, "idle"])
    assert env.state[0].status == "ERROR"
    assert env.state[1].status == "DONE"


def test_replay_keeps_the_sanitised_action():
    from kaggle_environments.envs.kargo.actions import MAX_BIDS

    env = make("kargo", configuration=SHORT)
    env.reset(2)
    env.step([{}, {}])
    env.step([{}, {}])
    env.step([{"bids": [["lot_0", 1.0]] * 5000, "junk": "x" * 10000}, {}])
    stored = env.steps[-1][0].action
    assert len(stored["bids"]) <= MAX_BIDS and "junk" not in stored


def test_assignment_covers_any_lot_set_the_decks_fit():
    from kaggle_environments.envs.kargo import kargo as K

    env = make("kargo", configuration=SHORT)
    env.reset(2)
    w = env.kargo
    p = w["players"][0]
    # Deck order VAN, STEP, VAN.
    old = p["trucks"].pop("T2")
    p["drivers"][old["driver"]]["truck"] = "T4"
    p["trucks"]["T4"] = dict(old, id="T4")
    lots = []
    for (td, units), pair in zip([(0.70, 100.0), (0.60, 150.0), (0.30, 250.0)], list(w["board"].truck_days)):
        lot = w["board"]._listing(pair[0], pair[1], 50, td, 100.0, w["rng"])
        lot.update({"parcel_units": units, "truck_days": td})
        lot["manifest"] = draw_manifest(lot, w["city"], w["rng"])
        total = sum(a["packages"] for a in lot["manifest"]["addresses"])
        for a in lot["manifest"]["addresses"]:
            a["units"] = units / max(1, total)
        lots.append(lot)
    p["pending_lots"] = lots
    K._start_day(w, 0)
    assert not p.get("pending_fails")


def test_rental_stays_ordered_to_its_owner_until_it_arrives():
    env = make("kargo", configuration={**SHORT, "episodeSteps": 41})
    env.reset(2)
    env.step([{"fleet": [["RENT", "VAN"]]}, {}])
    w = env.kargo
    rental = next(t for t in w["players"][0]["trucks"].values() if t["ownership"] == "RENTED")
    while w["day"] < rental["arrives"] - 1 or env.state[0].observation["phase"] != "CAPEX":
        env.step([{}, {}])
        mine = next(t for t in env.state[0].observation["private"]["trucks"] if t["id"] == rental["id"])
        if env.state[0].observation["day"] < rental["arrives"]:
            assert mine["status"] == "ORDERED"


def test_last_frame_is_the_close_not_a_new_night():
    env = make("kargo", configuration=SHORT)
    env.run(["idle", "idle"])
    obs = env.state[0].observation
    assert obs["phase"] == "DRIVING" and obs["block"] == BLOCKS_PER_DAY
    assert obs["day"] == (SHORT["episodeSteps"] - 1) // STEPS_PER_DAY - 1
    assert not obs["market"]["listings"] and not obs["market"]["candidates"] and not obs["market"]["used"]


def test_driver_on_a_grounded_truck_draws_a_retainer():
    from kaggle_environments.envs.kargo.constants import RETAINER_SHARE

    env = make("kargo", configuration=SHORT)
    env.reset(2)
    env.step([{}, {}])
    env.step([{}, {}])
    w = env.kargo
    truck = w["players"][0]["trucks"]["T1"]
    truck["status"] = "DISABLED"
    wage = w["players"][0]["drivers"][truck["driver"]]["wage"]
    for _ in range(1 + BLOCKS_PER_DAY):
        env.step([{}, {}])
    retainers = [amt for pid, amt, why in w["charges"] if pid == 0 and why == "RETAINER"]
    assert retainers == [pytest.approx(wage * RETAINER_SHARE)]


def test_used_truck_wear_is_not_its_odometer():
    rng = random.Random(3)
    kms = {round(make_truck("U", "VAN", rng, odometer=123456)["km_since_service"], 3) for _ in range(20)}
    assert len(kms) > 1 and all(0 <= k < SERVICE_INTERVAL_KM for k in kms)
    assert make_truck("N", "VAN", rng)["km_since_service"] == 0.0


def test_city_does_not_depend_on_player_actions():
    """Same seed, different agents: same weather, construction and incidents."""

    def city_days(agents):
        env = make("kargo", configuration=SHORT)
        seen = []
        env.reset(len(agents))
        while not env.done:
            obs = env.state[0].observation
            if obs["phase"] == "DRIVING":
                traffic = obs["traffic"]
                seen.append(
                    (obs["day"], obs["block"], traffic["weather"], traffic["forecast"], str(traffic["incidents"]))
                )
            env.step([a(env.state[i].observation, env.configuration) for i, a in enumerate(agents)])
        return seen

    from kaggle_environments.envs.kargo.kargo import agents

    idle = city_days([agents["idle"], agents["idle"]])
    busy = city_days([agents["greedy"], agents["random"]])
    assert idle and idle == busy


def test_ordered_rental_is_not_on_the_map():
    env = make("kargo", configuration=SHORT)
    env.reset(2)
    env.step([{"fleet": [["RENT", "VAN"]]}, {}])
    rental = next(t for t in env.kargo["players"][0]["trucks"].values() if t["status"] == "ORDERED")
    while env.state[0].observation["day"] < rental["arrives"] and not env.done:
        assert rental["status"] != "ORDERED" or rental["node"] is None
        env.step([{}, {}])
