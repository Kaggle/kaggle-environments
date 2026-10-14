import random

import pytest

from kaggle_environments import make
from kaggle_environments.envs.kargo.city import City
from kaggle_environments.envs.kargo.constants import (
    BULK_DISTRICTS,
    BULK_PATIENCE,
    BULK_STOPS,
    BULK_UNITS,
    VEHICLES,
)
from kaggle_environments.envs.kargo.freight import LotBoard, draw_bulk_manifest
from kaggle_environments.envs.kargo.shipper import Shipper

CFG = {"episodeSteps": 81, "seed": 7}


@pytest.fixture
def city():
    return City(random.Random(7))


@pytest.fixture
def board(city):
    return LotBoard(city, random.Random(7))


def _shipper(board, boxes=4):
    return Shipper(random.Random(3), list(board.truck_days), 12, boxes)


def _bulk(lots):
    return [lot for lot in lots if lot.get("bulk")]


def _env(players=2):
    env = make("kargo", configuration=CFG)
    env.reset(players)
    return env


def _to(env, phase):
    while env.state[0].observation["phase"] != phase:
        env.step([{}] * len(env.state))


def test_fresh_bulk_lots_are_palletised_and_only_fit_a_box(board):
    lots = _bulk(_shipper(board).post(board, 0, 0.0, 12, 4)[0])
    assert lots
    for lot in lots:
        assert BULK_UNITS[0] <= lot["parcel_units"] <= BULK_UNITS[1]
        assert BULK_STOPS[0] <= lot["stops"] <= BULK_STOPS[1]
        assert lot["district"] in BULK_DISTRICTS and lot["kind"] == "SPOT" and lot["retry"] == 0
        assert VEHICLES["VAN"]["capacity"] < lot["parcel_units"] <= VEHICLES["BOX"]["capacity"]


def test_bulk_manifest_is_a_few_dock_stops(board, city):
    lot = _bulk(_shipper(board).post(board, 0, 0.0, 12, 4)[0])[0]
    manifest = draw_bulk_manifest(lot, city, random.Random(1))
    assert len(manifest["segments"]) == lot["stops"]
    assert sum(a["packages"] for a in manifest["addresses"]) == lot["packages"]
    units = sum(a["packages"] * a["units"] for a in manifest["addresses"])
    assert units == pytest.approx(lot["parcel_units"])


def test_a_van_only_player_cannot_win_bulk_and_a_box_player_can():
    env = _env()
    for t in env.kargo["players"][0]["trucks"].values():
        if t["type"] == "BOX":
            t["driver"] = None
    _to(env, "CONTRACTS")
    lot = _bulk(env.state[0].observation["market"]["listings"])[0]
    env.step([{"bids": [[lot["id"], lot["reserve"] * 0.5]]}, {"bids": [[lot["id"], lot["reserve"]]]}])
    assert [(a["lot"], a["player"]) for a in env.kargo["auction_log"]] == [(lot["id"], 1)]


def test_a_box_loads_bulk_and_a_van_is_refused():
    env = _env()
    _to(env, "CONTRACTS")
    lot = _bulk(env.state[0].observation["market"]["listings"])[0]
    env.step([{"bids": [[lot["id"], lot["reserve"]]]}, {}])
    p = env.kargo["players"][0]
    box = next(t for t in p["trucks"].values() if t["type"] == "BOX")
    van = next(t for t in p["trucks"].values() if t["type"] == "VAN")
    dock = env.kargo["city"].warehouse_of[lot["warehouse"]]
    box["node"] = van["node"] = dock
    env.step([{"trucks": {van["id"]: {"load": [lot["id"]]}, box["id"]: {"load": [lot["id"]]}}}, {}])
    events = env.state[0].observation["private"]["events"]
    assert any(e["kind"] == "LOAD_REFUSED" and e["truck"] == van["id"] and e["reason"] == "NOT_BOX" for e in events)
    assert any(e["kind"] == "LOADED" and e["truck"] == box["id"] and e["lot"] == lot["id"] for e in events)


def test_unserved_bulk_raises_the_bulk_index_not_the_parcel_index(board):
    s = _shipper(board)
    for day in range(6):
        listings, _accounts = s.post(board, day, 0.0, 12, 4)
        parcels = [lot for lot in listings if not lot.get("bulk")]
        s.observe_auction(listings, [], [{"lot": lot["id"], "ask": lot["reserve"]} for lot in parcels])
    bulk_dev = sum(s.bulk.dev.values()) / len(s.bulk.dev)
    parcel_dev = sum(s.dev.values()) / len(s.dev)
    assert bulk_dev > 0.2
    assert abs(parcel_dev) < 0.05
    assert all(s.bulk.index(p) <= 2.0 for p in s.bulk.pairs)


def test_unserved_bulk_comes_back_for_longer_than_parcels(board):
    s = _shipper(board)
    retries = 0
    for day in range(BULK_PATIENCE[0] + 1):
        listings, _accounts = s.post(board, day, 0.0, 12, 4)
        bulk = _bulk(listings)
        retries = max(retries, max((lot["retry"] for lot in bulk), default=0))
        assert all(BULK_PATIENCE[0] <= lot["_patience"] <= BULK_PATIENCE[1] for lot in bulk)
        s.observe_auction(listings, [], [])
    assert retries >= BULK_PATIENCE[0]


def test_bulk_demand_does_not_depend_on_other_players_actions():
    from kaggle_environments.envs.kargo.kargo import agents

    greedy, idle = agents["greedy"], agents["idle"]

    def demand_path(second):
        env = make("kargo", configuration=CFG)
        env.run([greedy, second])
        bulk = env.kargo["shipper"].bulk
        return [round(bulk.demand(day), 9) for day in range(10)]

    a, b = demand_path(greedy), demand_path(idle)
    assert a == b and len(set(a)) > 1


def _shrink(lot, packages=40):
    """A partly delivered bulk lot coming back under a VAN's deck."""
    lot["packages"] = packages
    lot["parcel_units"] = packages * 4.0
    return lot


def test_a_van_only_player_cannot_win_a_small_bulk_retry():
    env = _env()
    for t in env.kargo["players"][0]["trucks"].values():
        if t["type"] == "BOX":
            t["driver"] = None
    _to(env, "CONTRACTS")
    lot = _shrink(_bulk(env.kargo["listings"])[0])
    assert lot["parcel_units"] <= VEHICLES["VAN"]["capacity"]
    env.step([{"bids": [[lot["id"], lot["reserve"] * 0.5]]}, {"bids": [[lot["id"], lot["reserve"]]]}])
    assert [(a["lot"], a["player"]) for a in env.kargo["auction_log"]] == [(lot["id"], 1)]


def test_a_van_cannot_load_a_small_bulk_retry():
    env = _env()
    _to(env, "CONTRACTS")
    lot = _shrink(_bulk(env.kargo["listings"])[0])
    env.step([{"bids": [[lot["id"], lot["reserve"]]]}, {}])
    p = env.kargo["players"][0]
    van = next(t for t in p["trucks"].values() if t["type"] == "VAN")
    van["node"] = env.kargo["city"].warehouse_of[lot["warehouse"]]
    env.step([{"trucks": {van["id"]: {"load": [lot["id"]]}}}, {}])
    events = env.state[0].observation["private"]["events"]
    assert any(e["kind"] == "LOAD_REFUSED" and e["truck"] == van["id"] and e["reason"] == "NOT_BOX" for e in events)


def test_a_territory_with_bulk_needs_a_box_deck():
    from kaggle_environments.envs.kargo.market import _decks_fit

    vans = [(200, False), (200, False)]
    assert not _decks_fit([(150, True)], vans)
    assert _decks_fit([(150, True), (190, False)], [(340, True), (200, False)])
    assert not _decks_fit([(150, True), (300, False)], [(340, True), (200, False)])
