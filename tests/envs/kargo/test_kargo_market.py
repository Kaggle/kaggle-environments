import random

import pytest

from kaggle_environments import make
from kaggle_environments.envs.kargo.city import City
from kaggle_environments.envs.kargo.constants import (
    MAX_DRIVERS,
    POACH_NOTICE_DAYS,
    SEVERANCE_DAYS,
    STANDING_BREAK_FEE_LOT_DAYS,
    STANDING_SHARE,
)
from kaggle_environments.envs.kargo.fleet import make_driver
from kaggle_environments.envs.kargo.freight import LotBoard
from kaggle_environments.envs.kargo.market import _headcount, resolve_labor
from kaggle_environments.envs.kargo.shipper import Shipper

CFG = {"episodeSteps": 81, "seed": 7}


def _env(players=2):
    env = make("kargo", configuration=CFG)
    env.reset(players)
    return env


def _to(env, phase, actions=None):
    """Step with empty actions until the published phase is `phase`."""
    n = len(env.state)
    while env.state[0].observation["phase"] != phase:
        env.step(actions or [{}] * n)


def _act(env, pid, action):
    acts = [{}] * len(env.state)
    acts[pid] = action
    env.step(acts)


def _willing(env, pid, did):
    """A driver a 2.5x offer will land."""
    env.kargo["players"][pid]["drivers"][did]["tenure"] = 0


def _labor_log(env):
    return env.state[0].observation["history"]["labor"]


# --- Poaching ---------------------------------------------------------------


def test_poached_driver_serves_notice_then_transfers():
    env = _env()
    _to(env, "LABOR")
    _willing(env, 0, "D0_1")
    _act(env, 1, {"labor": [["POACH", 0, "D0_1", 600]]})
    p0, p1 = env.kargo["players"]
    assert p0["drivers"]["D0_1"]["departs"] == {"to": 1, "offer": 600.0, "day": POACH_NOTICE_DAYS}
    assert p0["trucks"]["T1"]["driver"] == "D0_1"
    roster = env.state[0].observation["public"][0]["drivers"]
    assert next(d for d in roster if d["id"] == "D0_1")["departs"] == {"to": 1, "day": POACH_NOTICE_DAYS}

    for day in range(1, POACH_NOTICE_DAYS + 1):
        _to(env, "LABOR")
        _act(env, 0, {})
        if day < POACH_NOTICE_DAYS:
            assert "D0_1" in p0["drivers"]
    assert "D0_1" not in p0["drivers"] and p0["trucks"]["T1"]["driver"] is None
    moved = next(d for d in p1["drivers"].values() if d["wage"] == 600.0)
    assert moved["tenure"] <= 1 and moved["truck"] is None and moved["departs"] is None
    assert any(e["op"] == "POACH_TRANSFER" and e["driver"] == "D0_1" for e in _labor_log(env))


def test_employer_keeps_driver_by_matching_the_offer():
    env = _env()
    _to(env, "LABOR")
    _willing(env, 0, "D0_1")
    _act(env, 1, {"labor": [["POACH", 0, "D0_1", 600]]})
    _to(env, "LABOR")
    _act(env, 0, {"labor": [["WAGE", "D0_1", 600]]})
    p0 = env.kargo["players"][0]
    assert p0["drivers"]["D0_1"]["departs"] is None
    assert any(e["op"] == "POACH_MATCHED" and e["driver"] == "D0_1" for e in _labor_log(env))
    _to(env, "LABOR")
    _act(env, 0, {})
    assert "D0_1" in p0["drivers"]


def test_one_poach_per_rival_per_night():
    env = _env()
    _to(env, "LABOR")
    for did in ("D0_1", "D0_2"):
        _willing(env, 0, did)
    _act(env, 1, {"labor": [["POACH", 0, "D0_1", 600], ["POACH", 0, "D0_2", 600]]})
    drivers = env.kargo["players"][0]["drivers"]
    assert drivers["D0_1"]["departs"] and not drivers["D0_2"]["departs"]


def test_driver_on_notice_cannot_be_poached_again():
    env = _env(4)
    _to(env, "LABOR")
    _willing(env, 0, "D0_1")
    _act(env, 1, {"labor": [["POACH", 0, "D0_1", 600]]})
    _to(env, "LABOR")
    _act(env, 2, {"labor": [["POACH", 0, "D0_1", 2000]]})
    assert env.kargo["players"][0]["drivers"]["D0_1"]["departs"]["to"] == 1


def test_fire_charges_severance():
    env = _env()
    _to(env, "LABOR")
    p0 = env.kargo["players"][0]
    cash, wage = p0["cash"], p0["drivers"]["D0_1"]["wage"]
    _act(env, 0, {"labor": [["FIRE", "D0_1"]]})
    assert "D0_1" not in p0["drivers"]
    assert p0["cash"] == pytest.approx(cash - SEVERANCE_DAYS * wage)


def test_firing_a_driver_on_notice_releases_them_without_severance():
    env = _env()
    _to(env, "LABOR")
    _willing(env, 0, "D0_1")
    _act(env, 1, {"labor": [["POACH", 0, "D0_1", 600]]})
    _to(env, "LABOR")
    p0, p1 = env.kargo["players"]
    cash, before = p0["cash"], len(p1["drivers"])
    _act(env, 0, {"labor": [["FIRE", "D0_1"]]})
    assert p0["cash"] == pytest.approx(cash)
    assert len(p1["drivers"]) == before + 1


def test_incoming_drivers_count_against_the_roster_cap():
    env = _env()
    p0, p1 = env.kargo["players"]
    rng = random.Random(1)
    while len(p1["drivers"]) < MAX_DRIVERS - 1:
        d = make_driver(f"D1_x{len(p1['drivers'])}", rng)
        p1["drivers"][d["id"]] = d
    p0["drivers"]["D0_1"]["departs"] = {"to": 1, "offer": 600.0, "day": 5}
    assert _headcount(env.kargo["players"], 1) == MAX_DRIVERS
    cand = make_driver("cand_0_0", rng, wage=200)
    world = {"day": 0, "candidates": [cand]}
    resolve_labor(env.kargo["players"], [{}, {"labor": [["HIRE", "cand_0_0", 2000]]}], world, rng)
    assert world["candidates"] == [cand]


# --- Standing accounts ------------------------------------------------------


@pytest.fixture
def board():
    return LotBoard(City(random.Random(7)), random.Random(7))


def test_account_listings_post_as_standing(board):
    sh = Shipper(random.Random(3), list(board.truck_days), 12)
    accounts = [a for day in range(10) for a in sh.post(board, day, 0.0, 12)[1]]
    assert accounts
    for a in accounts:
        assert a["kind"] == "STANDING" and a["term_options"]
        assert a["payout_per_package"] == pytest.approx(a["reserve"] / a["packages"], abs=0.01)


def test_standing_share_is_a_third_of_fresh_listings(board):
    """Retries do not dilute the share."""
    sh = Shipper(random.Random(3), list(board.truck_days), 12)
    accounts = fresh = 0
    for day in range(30):
        listings, accts = sh.post(board, day, 0.0, 12)
        sh.observe_auction(listings + accts, [], [])
        accounts += len(accts)
        fresh += len(accts) + sum(1 for x in listings if not x["retry"])
    assert abs(accounts / fresh - STANDING_SHARE) < 0.06


def _win_account(env, term=5):
    _to(env, "CONTRACTS")
    acct = env.state[0].observation["market"]["accounts"][0]
    _act(env, 0, {"standing_bids": [[acct["id"], acct["reserve"], term]]})
    return acct


def test_won_account_is_held_and_rolls_daily():
    env = _env()
    acct = _win_account(env, term=5)
    p0 = env.kargo["players"][0]
    held = env.state[0].observation["public"][0]["standing"]
    assert [a["id"] for a in held] == [acct["id"]] and held[0]["remaining"] == 4
    first = f"{acct['id']}_d0"
    assert first in p0["lots"] or any(lot["id"] == first for lot in p0.get("pending_fails", []))
    rolled = 1
    while p0["standing"]:
        _to(env, "CONTRACTS")
        _act(env, 0, {})
        rolled += 1
    assert rolled == 5


def test_break_ends_an_account_for_a_fee():
    env = _env()
    acct = _win_account(env)
    p0 = env.kargo["players"][0]
    rate = p0["standing"][0]["rate"]
    _to(env, "CAPEX")
    cash = p0["cash"]
    _act(env, 0, {"fleet": [["BREAK", acct["id"]]]})
    assert not p0["standing"]
    assert p0["cash"] == pytest.approx(cash - STANDING_BREAK_FEE_LOT_DAYS * rate)


def test_plain_bid_on_an_account_is_ignored():
    env = _env()
    _to(env, "CONTRACTS")
    acct = env.state[0].observation["market"]["accounts"][0]
    _act(env, 0, {"bids": [[acct["id"], acct["reserve"]]]})
    assert not env.kargo["players"][0]["standing"]
    assert not any(a["lot"] == acct["id"] for a in env.kargo["auction_log"])


# --- Auction ----------------------------------------------------------------


def test_stacked_bids_cannot_beat_max_lots():
    env = _env()
    _to(env, "CONTRACTS")
    lot = env.state[0].observation["market"]["listings"][0]
    stacked = [[lot["id"], lot["reserve"] * (0.5 + i / 40)] for i in range(20)]
    env.step([{"bids": stacked, "max_lots": 0.0}, {"bids": [[lot["id"], lot["reserve"]]]}])
    assert [(a["lot"], a["player"]) for a in env.kargo["auction_log"]] == [(lot["id"], 1)]


def test_player_with_no_runnable_trucks_wins_nothing():
    env = _env()
    _to(env, "CONTRACTS")
    for truck in env.kargo["players"][0]["trucks"].values():
        truck["driver"] = None
    lot = env.state[0].observation["market"]["listings"][0]
    stacked = [[lot["id"], lot["reserve"] * (0.5 + i / 40)] for i in range(20)]
    _act(env, 0, {"bids": stacked})
    assert not env.kargo["auction_log"]


def test_bid_book_keeps_one_lowest_bid_per_player():
    env = _env()
    _to(env, "CONTRACTS")
    lot = env.state[0].observation["market"]["listings"][0]
    r = lot["reserve"]
    env.step([{"bids": [[lot["id"], r], [lot["id"], r * 0.9], [lot["id"], r * 0.95]]}, {"bids": [[lot["id"], r]]}])
    row = next(b for b in env.kargo["bid_book"] if b["lot"] == lot["id"])
    assert sorted(p for _a, p in row["bids"]) == [0, 1]
    assert min(a for a, p in row["bids"] if p == 0) == pytest.approx(r * 0.9, abs=0.01)


def test_one_player_can_hold_several_accounts():
    env = _env()
    p0 = env.kargo["players"][0]
    for _ in range(4):
        _to(env, "CONTRACTS")
        accounts = env.state[0].observation["market"]["accounts"]
        _act(env, 0, {"standing_bids": [[a["id"], a["reserve"], 20] for a in accounts], "max_lots": 3.0})
    assert len(p0["standing"]) >= 2
    _to(env, "CONTRACTS")
    _act(env, 0, {})
    uncovered = {lot["id"] for lot in p0.get("pending_fails", [])}
    assert all(f"{a['id']}_d4" in p0["lots"] or f"{a['id']}_d4" in uncovered for a in p0["standing"])


def test_held_accounts_count_against_tomorrows_auction():
    env = _env()
    p0 = env.kargo["players"][0]
    for _ in range(6):
        _to(env, "CONTRACTS")
        market = env.state[0].observation["market"]
        _act(
            env,
            0,
            {
                "bids": [[lot["id"], lot["reserve"]] for lot in market["listings"]],
                "standing_bids": [[a["id"], a["reserve"], 20] for a in market["accounts"]],
                "max_lots": 3.0,
            },
        )
        assert not p0.get("pending_fails")
    assert p0["standing"]
