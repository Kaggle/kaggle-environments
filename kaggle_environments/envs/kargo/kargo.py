"""Kargo: a last-mile logistics simulation.

The engine state is too large and too full of live objects to round-trip
through the observation, so it lives on ``env`` and only its public and
per-player views are written into the state each step.
"""

import json
import random
from os import path

from kaggle_environments.utils import resolve_episode_seed

from .actions import sanitize
from .city import City
from .constants import (
    ABANDON_BLOCK,
    ARTERIAL_KMH,
    BLOCKS_PER_DAY,
    DAY_END_MINUTES,
    DISTRICTS,
    FILL_CEILING,
    LOAD_MINUTES,
    OVERTIME_WAGE_MULT,
    RETAINER_SHARE,
    ROUTE_SHOWN,
    SERVICE_INTERVAL_KM,
    SHIFT_MINUTES,
    SIGHTINGS_KEPT,
    STARTING_DRIVER_STAT,
    STARTING_FLEET,
    STARTING_VEHICLE,
    STEPS_PER_DAY,
    TRAIL_SHOWN,
    VEHICLES,
)
from .dispatch import abandon, advance_truck, end_of_day, service_check
from .fleet import make_driver, make_truck
from .freight import LotBoard, draw_manifest, noisy_service_estimate
from .market import (
    accrue,
    net_worth,
    post_candidates,
    post_used,
    rental_pool,
    resolve_auction,
    resolve_capex,
    resolve_labor,
)
from .shipper import Shipper

dirpath = path.dirname(__file__)
with open(path.abspath(path.join(dirpath, "kargo.json"))) as json_file:
    specification = json.load(json_file)

PHASES = ["CAPEX", "LABOR", "CONTRACTS"] + ["DRIVING"] * BLOCKS_PER_DAY


def phase_of(step):
    """Each day opens with its three overnight steps, then drives."""
    if step <= 0:
        return "RESET", 0, 0
    idx = (step - 1) % STEPS_PER_DAY
    return PHASES[idx], (step - 1) // STEPS_PER_DAY, max(0, idx - len(PHASES) + BLOCKS_PER_DAY)


# --- setup ------------------------------------------------------------------


def _initialize(state, env):
    seed = resolve_episode_seed(env)
    rng = random.Random(seed)
    cfg = env.configuration
    city = City(rng, truck_pcu=str(cfg.get("truckPcu", "low")))
    # Weather, traffic and incidents draw from their own stream, so players' actions cannot shift them.
    city_rng = random.Random(rng.getrandbits(64))

    players = []
    fleet_size = 0
    for pid in range(len(state)):
        player = {
            "cash": float(cfg.get("startingCash", 12000)),
            "debt": 0.0,
            "trucks": {},
            "drivers": {},
            "standing": [],
            "pending_lots": [],
            "lots": {},
            "manifest": {"segments": {}, "addresses": {}},
            "day_report": _blank_report(),
            "results": [],
            "sightings": [],
            "unserved": [],
            "truck_seq": 0,
            "driver_seq": 0,
        }
        # A count longer than the fleet template repeats it; a shorter one takes its front.
        count = int(cfg.get("startingTrucks", len(STARTING_FLEET)))
        fleet_size += count
        for index in range(count):
            player["truck_seq"] += 1
            tid = f"T{player['truck_seq']}"
            vehicle = STARTING_FLEET[index % len(STARTING_FLEET)] if STARTING_FLEET else STARTING_VEHICLE
            truck = make_truck(tid, vehicle, rng)
            truck["player"] = pid
            truck["arrives"] = 0
            truck["staged"] = city.warehouse_ids[index % len(city.warehouse_ids)]
            truck["node"] = city.warehouse_of[truck["staged"]]
            truck["home"] = truck["node"]
            truck["clock"] = 0.0
            truck["overtime"] = 0.0
            truck["carrying"] = set()
            truck["route"] = []
            truck["plan"] = {}
            truck["anchor"] = None
            truck["pos"] = (0.0, 0.0)
            truck["last_t"] = 0.0
            player["trucks"][tid] = truck

            player["driver_seq"] += 1
            did = f"D{pid}_{player['driver_seq']}"
            quality = rng.uniform(*STARTING_DRIVER_STAT)
            driver = make_driver(did, rng, quality=quality)
            driver["truck"] = tid
            truck["driver"] = did
            player["drivers"][did] = driver
        players.append(player)

    board = LotBoard(city, rng)
    world = {
        "seed": seed,
        "rng": rng,
        "city_rng": city_rng,
        "city": city,
        "board": board,
        # Its own stream, so the market's draws do not shift the city's.
        "shipper": Shipper(random.Random(rng.getrandbits(64)), list(board.truck_days), fleet_size),
        "players": players,
        "listings": [],
        "accounts": [],
        "used": [],
        "candidates": [],
        "rental_pool": {},
        "public_standing": [],
        "bid_book": [],
        "auction_log": [],
        "capex_log": [],
        "labor_log": [],
        "events": [],
        "charges": [],
        "credits": [],
        "day": 0,
    }
    env.kargo = world
    _open_night(world, 0)
    _publish(state, env, world, "CAPEX", 0, 0, 0)
    return state


def _blank_report():
    return {"delivered": 0, "late": 0, "failed": 0, "refused": 0, "revenue": 0.0, "cost": 0.0}


def _world(state, env):
    world = getattr(env, "kargo", None)
    if world is None:
        _initialize(state, env)
        world = env.kargo
    return world


# --- overnight --------------------------------------------------------------


def _open_night(world, day):
    """Post the night's capital, labor and freight markets."""
    rng = world["rng"]
    players = world["players"]
    world["used"] = post_used(rng, day)
    world["rental_pool"] = rental_pool(rng, players)
    world["candidates"] = post_candidates(rng, day)

    # Held standing accounts are demand the shipper has already placed.
    committed = sum(a["truck_days"] for p in players for a in p["standing"])
    # What the shipper can see of the field: trucks that can roll, capped by
    # the drivers to crew them, off the public roster.
    supply = sum(
        min(
            len(p["drivers"]),
            len([t for t in p["trucks"].values() if t.get("arrives", 0) <= day and t["status"] != "DISABLED"]),
        )
        for p in players
    )
    listings, accounts = world["shipper"].post(world["board"], day, committed, supply)
    for lot in listings + accounts:
        lot["manifest"] = draw_manifest(lot, world["city"], rng)
    world["listings"] = listings
    world["accounts"] = accounts
    world["bid_book"] = []


def _clear_boards(world):
    for key in ("listings", "accounts", "used", "candidates", "bid_book"):
        world[key] = []
    world["rental_pool"] = {}


def _roll_standing(world, day):
    """A standing account posts its lot every day of its term, at the held rate."""
    for player in world["players"]:
        for acct in list(player["standing"]):
            if acct["remaining"] <= 0:
                player["standing"].remove(acct)
                world["public_standing"] = [a for a in world["public_standing"] if a["id"] != acct["id"]]
                continue
            board = world["board"]
            key = (acct["warehouse"], acct["district"])
            td = board.truck_days[key]
            fraction = acct["truck_days"]
            packages = max(1, int(round(td["packages"] * fraction)))
            lot = board._listing(  # noqa: SLF001 -- same module family, one code path
                acct["warehouse"],
                acct["district"],
                packages,
                fraction,
                acct["rate"],
                world["rng"],
            )
            lot["id"] = f"{acct['id']}_d{day}"
            lot["kind"] = "STANDING"
            lot["price"] = acct["rate"]
            lot["payout_per_package"] = round(acct["rate"] / packages, 4)
            lot["manifest"] = draw_manifest(lot, world["city"], world["rng"])
            player["pending_lots"].append(lot)
            acct["remaining"] -= 1
            if acct["remaining"] <= 0:
                player["standing"].remove(acct)
                world["public_standing"] = [a for a in world["public_standing"] if a["id"] != acct["id"]]


# --- driving day ------------------------------------------------------------


def _start_day(world, day):
    """08:00. Manifests land, trucks load, repositions are charged."""
    city = world["city"]
    city.reset_day(day, world["city_rng"])
    world["events"] = []
    # The day's list is new, so the step's mark has to come back to its start.
    world["events_mark"] = 0
    world["charges"] = []
    world["credits"] = []

    for pid, player in enumerate(world["players"]):
        player["day_report"] = _blank_report()
        player["manifest"] = {"segments": {}, "addresses": {}}
        player["lots"] = {}
        for truck in player["trucks"].values():
            truck["clock"] = 0.0
            truck["worked"] = 0.0
            truck["overtime"] = 0.0
            truck["carrying"] = set()
            truck["route"] = []
            truck["last_t"] = 0.0
            truck["anchor"] = None
            truck["pos"] = (0.0, 0.0)
            truck["pair"] = None
            truck["fill"] = 0.0
            truck["lots"] = []
            truck["revisited"] = {}
            if truck.get("arrives", 0) == day and truck["status"] == "ORDERED":
                truck["status"] = "IDLE"
                truck["staged"] = truck["staged"] or city.warehouse_ids[0]
                truck["node"] = city.warehouse_of[truck["staged"]]
            if truck["node"] is None and truck["status"] != "ORDERED":
                truck["staged"] = truck["staged"] or city.warehouse_ids[0]
                truck["node"] = city.warehouse_of[truck["staged"]]
            truck["home"] = truck["node"]
            truck["runnable"] = (
                truck.get("arrives", 0) <= day
                and truck["status"] not in ("DISABLED", "ORDERED")
                and bool(truck["driver"])
            )
        available = [t for t in player["trucks"].values() if t["runnable"]]

        # Pairs by deck space needed, largest first; within a pair, biggest lot first.
        pairs = {}
        for lot in player["pending_lots"]:
            pairs.setdefault((lot["warehouse"], lot["district"]), []).append(lot)
        for lots in sorted(pairs.values(), key=lambda ls: -sum(lot["parcel_units"] for lot in ls)):
            need = sum(lot["parcel_units"] for lot in lots)
            for lot in sorted(lots, key=lambda x: -x["truck_days"]):
                _assign_lot(player, lot, available, world, pid, need)
                need -= lot["parcel_units"]
        player["pending_lots"] = []


def _assign_lot(player, lot, available, world, pid, need=0.0):
    """Pair a lot with a truck. Lots on one pair share a truck up to the fill
    ceiling; a lot no truck can take is UNCOVERED. `need` is the deck space the
    pair still needs, this lot included.
    """
    city = world["city"]
    origin = city.warehouse_of[lot["warehouse"]]
    units = lot["parcel_units"]
    size = lot["truck_days"]
    pair = (lot["warehouse"], lot["district"])
    # A truck works one pair a day.
    fits = [
        t
        for t in available
        if t.get("pair") in (None, pair)
        and VEHICLES[t["type"]]["capacity"] - _loaded_units(t, player) >= units
        and t.get("fill", 0.0) + size <= FILL_CEILING
        and t["clock"] + LOAD_MINUTES < SHIFT_MINUTES
    ]
    if not fits:
        # `truck` is empty rather than absent so consumers can key on it; the
        # node is the origin dock.
        world["events"].append(
            {
                "kind": "UNCOVERED",
                "player": pid,
                "truck": "",
                "node": origin,
                "day": world["day"],
                "minute": 0,
                # The lot id keeps same-minute events at one dock distinct.
                "address": lot["id"],
                "lot": lot["id"],
                "packages": lot["packages"],
            }
        )
        player["pending_fails"] = player.get("pending_fails", [])
        player["pending_fails"].append(lot)
        return
    # Top up a truck already on this pair; otherwise the smallest deck that
    # holds the whole pair, then one already at the dock.
    fits.sort(
        key=lambda t: (
            t.get("pair") is None,
            -t.get("fill", 0.0),
            VEHICLES[t["type"]]["capacity"] < need,
            VEHICLES[t["type"]]["capacity"],
            t["node"] != origin,
            t["clock"],
        )
    )
    truck = fits[0]
    if truck["node"] != origin:
        km = city.node_km(truck["node"], origin)
        truck["clock"] += km / ARTERIAL_KMH * 60.0
        world["charges"].append((pid, km * 0.62 + km / ARTERIAL_KMH * 31.0, "REPOSITION"))
        truck["node"] = origin
    truck["home"] = origin
    truck["pair"] = pair
    truck["fill"] = truck.get("fill", 0.0) + size
    truck["clock"] += LOAD_MINUTES
    truck["status"] = "ACTIVE"

    manifest = lot.pop("manifest")
    for seg in manifest["segments"]:
        player["manifest"]["segments"][seg["id"]] = seg
    for addr in manifest["addresses"]:
        player["manifest"]["addresses"][addr["id"]] = addr
        truck["carrying"].add(addr["id"])
    player["lots"][lot["id"]] = lot
    truck.setdefault("lots", []).append(lot["id"])


def _loaded_units(truck, player):
    """Deck space in use, in parcel-units."""
    total = 0.0
    for aid in sorted(truck["carrying"]):
        addr = player["manifest"]["addresses"].get(aid)
        if addr:
            per = addr.get("units")
            if per is None:
                per = DISTRICTS[player["manifest"]["segments"][addr["segment"]]["district"]]["pkg_units"]
            total += addr["packages"] * per
    return total


def _apply_plans(player, act, world, pid):
    """Plans persist: omitting a truck means carry on."""
    trucks = act.get("trucks") if isinstance(act.get("trucks"), dict) else {}
    for tid, plan in trucks.items():
        truck = player["trucks"].get(tid)
        if truck is None or not isinstance(plan, dict):
            continue
        truck["plan"] = {k: v for k, v in plan.items() if k != "route"}
        if isinstance(plan.get("route"), list):
            truck["route"] = list(plan["route"])


def _abandon(player, act, world, pid, block):
    """Write-offs, first block only: the freight is still at the dock."""
    if block == ABANDON_BLOCK:
        abandon(player, pid, act.get("abandon"), world, world["events"])


def _run_block(world, actions, block):
    """Advance every truck to the end of the block, then take sightings."""
    until = (block + 1) * (DAY_END_MINUTES / BLOCKS_PER_DAY)
    rng = world["rng"]
    for pid, player in enumerate(world["players"]):
        act = actions[pid]
        _apply_plans(player, act, world, pid)
        _abandon(player, act, world, pid, block)
    # Player order is shuffled every block.
    order = list(range(len(world["players"])))
    rng.shuffle(order)
    for pid in order:
        player = world["players"][pid]
        for truck in player["trucks"].values():
            if truck["status"] in ("ACTIVE", "IDLE") and truck.get("arrives", 0) <= world["day"]:
                world["events"].extend(advance_truck(truck, truck["plan"], world, player, until, rng))
    _sightings(world)


def _sightings(world):
    """Active trucks of different players on the same node see each other."""
    where = {}
    for pid, player in enumerate(world["players"]):
        for truck in player["trucks"].values():
            if truck["status"] == "ACTIVE":
                where.setdefault(truck["node"], []).append((pid, truck))
    for node, occupants in where.items():
        if len(occupants) < 2:
            continue
        for pid, _mine in occupants:
            for other_pid, other in occupants:
                if other_pid == pid:
                    continue
                world["players"][pid]["sightings"].append(
                    {
                        "day": world["day"],
                        "minute": round(other["clock"], 1),
                        "player": other_pid,
                        "node": node,
                        "type": other["type"],
                        "load": _load_band(other, world["players"][other_pid]),
                        "driver": other["driver"],
                    }
                )
    for player in world["players"]:
        player["sightings"] = player["sightings"][-SIGHTINGS_KEPT:]


def _load_band(truck, player):
    units = _loaded_units(truck, player)
    ratio = units / max(1.0, VEHICLES[truck["type"]]["capacity"])
    return "EMPTY" if ratio < 0.05 else "LIGHT" if ratio < 0.4 else "HALF" if ratio < 0.75 else "FULL"


def _close_day(world, day):
    """18:00. Forfeit what is left, pay wages, settle the ledger."""
    shipper = world["shipper"]
    for pid, player in enumerate(world["players"]):
        for truck in player["trucks"].values():
            end_of_day(truck, world, player, world["events"])
            if truck["status"] not in ("DISABLED", "ORDERED"):
                truck["status"] = "IDLE"
            if truck["status"] == "IDLE":
                truck["staged"] = _nearest_warehouse(world["city"], truck["node"])
                truck["node"] = world["city"].warehouse_of[truck["staged"]]
            # After parking, so a grounded truck waits at a warehouse.
            service_check(truck, world, world["events"])
            driver = player["drivers"].get(truck["driver"])
            on_clock = truck.get("worked", truck["clock"])
            if driver and on_clock > 0:
                worked = min(on_clock, SHIFT_MINUTES)
                over = max(0.0, on_clock - SHIFT_MINUTES)
                pay = driver["wage"] * (worked / SHIFT_MINUTES)
                pay += driver["wage"] / SHIFT_MINUTES * over * OVERTIME_WAGE_MULT
                world["charges"].append((pid, pay, "WAGES"))
                driver["overtime_minutes"] += over
        # A driver on no truck, or on one that could not run today, draws a retainer.
        seated = {t["driver"] for t in player["trucks"].values() if t["driver"] and t.get("runnable")}
        for did, driver in player["drivers"].items():
            if did not in seated:
                world["charges"].append((pid, driver["wage"] * RETAINER_SHARE, "RETAINER"))
        for lot in player.pop("pending_fails", []):
            world["charges"].append((pid, lot["packages"] * lot["fail_penalty"], "UNCOVERED"))
            player["day_report"]["failed"] += lot["packages"]
            shipper.observe_failed(lot, lot["packages"])
        # Whatever did not reach a door goes back to the shipper.
        for lot in player["lots"].values():
            shipper.observe_failed(lot, lot["packages"] - lot.get("delivered", 0))
        for lot, packages in player.get("unserved", []):
            shipper.observe_failed(lot, packages)
        player["unserved"] = []

    for pid, amount, _reason in world["credits"]:
        world["players"][pid]["cash"] += amount
        world["players"][pid]["day_report"]["revenue"] += amount
    for pid, amount, _reason in world["charges"]:
        world["players"][pid]["cash"] -= amount
        world["players"][pid]["day_report"]["cost"] += amount

    for player in world["players"]:
        accrue(player, day)
        report = dict(player["day_report"])
        report["day"] = day
        report["net_worth"] = round(net_worth(player), 2)
        player["results"].append(report)
        player["results"] = player["results"][-10:]


def _nearest_warehouse(city, node):
    return min(city.warehouse_ids, key=lambda wid: city.node_km(node, city.warehouse_of[wid]))


# --- observation views ------------------------------------------------------


def _public_view(world, day):
    out = []
    for player in world["players"]:
        out.append(
            {
                "cash": round(player["cash"], 2),
                "debt": round(player["debt"], 2),
                "net_worth": round(net_worth(player), 2),
                "fleet": [
                    {
                        "id": t["id"],
                        "type": t["type"],
                        "ownership": t["ownership"],
                        "age_days": t["age_days"],
                        "odometer": round(t["odometer"], 1),
                        "status": "ORDERED" if t.get("arrives", 0) > day else t["status"],
                    }
                    for t in player["trucks"].values()
                ],
                "drivers": [
                    {
                        "id": d["id"],
                        "name": d["name"],
                        "tenure": d["tenure"],
                        "resume": d["resume"],
                        "notice": d["notice"],
                        "departs": {k: d["departs"][k] for k in ("to", "day")} if d.get("departs") else None,
                    }
                    for d in player["drivers"].values()
                ],
                "standing": [
                    {k: a[k] for k in ("id", "district", "warehouse", "rate", "term", "remaining")}
                    for a in player["standing"]
                ],
                "results": player["results"][-3:],
            }
        )
    return out


def _private_view(world, player, pid):
    manifest = player["manifest"]
    rng = world["rng"]
    return {
        # This step's events, this player's only.
        "events": [e for e in world["events"][world.get("events_mark", 0) :] if e.get("player") == pid],
        "trucks": [
            {
                "id": t["id"],
                "type": t["type"],
                "node": t["node"],
                "status": t["status"],
                "clock": round(t["clock"], 1),
                "fuel": round(t["fuel"], 1),
                "km_since_service": round(t["km_since_service"], 1),
                "driver": t["driver"],
                "staged": t["staged"],
                "carrying": sorted(t["carrying"]),
                # The head of the remaining route.
                "route": t["route"][:ROUTE_SHOWN],
                # Positions during the block just run, thinned to a cap.
                "trail": _thin(t.get("trail", []), TRAIL_SHOWN),
            }
            for t in player["trucks"].values()
        ],
        "drivers": [
            {
                "id": d["id"],
                "name": d["name"],
                "wage": d["wage"],
                "resume": d["resume"],
                "truck": d["truck"],
                "notice": d["notice"],
                "departs": dict(d["departs"]) if d.get("departs") else None,
            }
            for d in player["drivers"].values()
        ],
        "segments": [
            {
                "id": s["id"],
                "node": s["node"],
                "pos": s["pos"],
                "district": s["district"],
                "pending": [a for a in s["addresses"] if _pending(player, a)],
            }
            for s in manifest["segments"].values()
        ],
        "addresses": [
            {
                "id": a["id"],
                "segment": a["segment"],
                "node": a["node"],
                "t": a["t"],
                "packages": a["packages"],
                "window": a["window"],
                "window_kind": a["window_kind"],
                "lot": a["lot"],
                "service_estimate": noisy_service_estimate(a, rng),
            }
            for a in manifest["addresses"].values()
            if _pending(player, a["id"])
        ],
        "lots": [_public_lot(lot) for lot in player["lots"].values()],
        "pending_lots": [_public_lot(lot) for lot in player["pending_lots"]],
        "sightings": player["sightings"][-20:],
        "day_report": player["day_report"],
    }


def _public_lot(lot):
    """A lot as a player may see it: no manifest, no engine-only `_` keys."""
    return {k: v for k, v in lot.items() if k != "manifest" and not k.startswith("_")}


def _thin(trail, cap):
    """Drop interior crumbs evenly, always keeping the first and the last."""
    if len(trail) <= cap:
        return list(trail)
    keep = [trail[0]]
    stride = (len(trail) - 1) / (cap - 1)
    keep.extend(trail[round(i * stride)] for i in range(1, cap - 1))
    keep.append(trail[-1])
    return keep


def _pending(player, aid):
    return any(aid in t["carrying"] for t in player["trucks"].values())


def _publish(state, env, world, phase, day, block, minute):
    city = world["city"]
    obs0 = state[0].observation
    obs0.city = city.static_view()
    obs0.traffic = {
        "congestion": city.congestion_report(minute),
        "incidents": city.incident_report(minute),
        # The last driven day's weather overnight, today's during the day;
        # None on the first night.
        "weather": None if phase != "DRIVING" and day == 0 else city.weather,
        # Overnight it forecasts the coming day; during a day, tomorrow.
        "forecast": city.forecast_for(day if phase != "DRIVING" else day + 1, world["city_rng"]),
    }
    obs0.market = {
        "listings": [_public_lot(lot) for lot in world["listings"]],
        "accounts": [_public_lot(a) for a in world["accounts"]],
        "used": world["used"],
        "rentals": world["rental_pool"],
        "candidates": [
            {
                "id": c["id"],
                "name": c["name"],
                "resume": c["resume"],
                "asking": c.get("asking", c["wage"]),
            }
            for c in world["candidates"]
        ],
        "fill_ceiling": FILL_CEILING,
        "service_interval_km": SERVICE_INTERVAL_KM,
    }
    obs0.public = _public_view(world, day)
    obs0.history = {
        "auction": list(world["auction_log"]),
        "capex": list(world["capex_log"]),
        "labor": list(world["labor_log"]),
        "bids": world["bid_book"],
        "standing": [
            {
                **{k: a[k] for k in ("id", "warehouse", "district", "rate", "term", "remaining", "truck_days")},
                "player": pid,
            }
            for pid, p in enumerate(world["players"])
            for a in p["standing"]
        ],
    }
    obs0.phase = phase
    obs0.day = day
    obs0.block = block
    obs0.minute = int(minute)
    for i, s in enumerate(state):
        s.observation.player = i
        s.observation.private = _private_view(world, world["players"][i], i)
        # Shared fields live on state[0] only; the framework merges them into
        # every agent's observation at act time.


# --- interpreter ------------------------------------------------------------


def interpreter(state, env):
    obs0 = state[0].observation
    if not getattr(obs0, "city", None):
        return _initialize(state, env)
    if env.done:
        return state

    world = _world(state, env)
    step = int(obs0.step or 0)
    phase, day, block = phase_of(step + 1)
    world["day"] = day
    # Events raised from here to the publish below belong to this step.
    world["events_mark"] = len(world.get("events", []))
    actions = [s.action if isinstance(s.action, dict) else None for s in state]
    for i, act in enumerate(actions):
        # Untrusted input: rebuilt to the exact shapes the resolvers expect.
        actions[i] = sanitize(phase, act)
        if state[i].status == "ACTIVE":
            # The replay keeps what the engine acted on, not the raw reply.
            state[i].action = actions[i]

    if phase == "CAPEX":
        world["capex_log"] = resolve_capex(world["players"], actions, world, world["rng"], day)
    elif phase == "LABOR":
        world["labor_log"] = resolve_labor(world["players"], actions, world, world["rng"])
        _nightly_morale(world)
    elif phase == "CONTRACTS":
        world["auction_log"] = resolve_auction(world["players"], actions, world, world["rng"])
        world["shipper"].observe_auction(world["listings"] + world["accounts"], world["bid_book"], world["auction_log"])
        _roll_standing(world, day)
        _start_day(world, day)
    else:
        _run_block(world, actions, block)
        if block == BLOCKS_PER_DAY - 1:
            _close_day(world, day)

    final = step >= env.configuration.episodeSteps - 2
    next_phase, next_day, next_block = phase_of(step + 2)
    if final and next_phase == "CAPEX":
        # The last frame is the 18:00 close, not a night that never runs.
        next_phase, next_day, next_block = "DRIVING", day, BLOCKS_PER_DAY
        _clear_boards(world)
    elif next_phase == "CAPEX" and next_day > 0:
        # The night's markets open before the board is published, so CAPEX
        # acts on the board it was shown.
        _open_night(world, next_day)
    minute = next_block * (DAY_END_MINUTES / BLOCKS_PER_DAY) if next_phase == "DRIVING" else 0
    _publish(state, env, world, next_phase, next_day, next_block, minute)

    if final:
        for s in state:
            if s.status in ("ACTIVE", "INACTIVE"):
                s.status = "DONE"
            s.reward = round(net_worth(world["players"][s.observation.player]), 2)
    return state


def _nightly_morale(world):
    from .fleet import nightly_morale

    for player in world["players"]:
        for did, driver in list(player["drivers"].items()):
            if nightly_morale(driver, world["rng"]):
                del player["drivers"][did]
                for truck in player["trucks"].values():
                    if truck.get("driver") == did:
                        truck["driver"] = None
                world["labor_log"].append({"op": "QUIT", "driver": did})


# --- rendering --------------------------------------------------------------


def renderer(state, env):
    obs = state[0].observation
    out = (
        f"Step {obs.step}  Day {obs.day}  {obs.phase}"
        f"{f' block {obs.block} (min {obs.minute})' if obs.phase == 'DRIVING' else ''}  "
        f"weather={obs.traffic.get('weather')}\n"
    )
    listings = obs.market.get("listings", [])
    out += f"Board: {len(listings)} spot lots, {len(obs.market.get('accounts', []))} standing\n"
    for i, pub in enumerate(obs.public):
        out += (
            f"Player {i}: cash ${pub['cash']:,.0f}  net ${pub['net_worth']:,.0f}  "
            f"trucks={len(pub['fleet'])}  drivers={len(pub['drivers'])}  "
            f"standing={len(pub['standing'])}\n"
        )
        last = pub["results"][-1] if pub["results"] else None
        if last:
            out += (
                f"  day {last['day']}: delivered {last['delivered']}  late {last['late']}  "
                f"failed {last['failed']}  refused {last['refused']}\n"
            )
    return out


def html_renderer(env, mode):
    jspath = path.join(dirpath, "visualizer", "default", "dist", "index.html")
    if path.exists(jspath):
        with open(jspath, encoding="utf-8") as f:
            return f.read()
    return ""


# --- agents -----------------------------------------------------------------


def _my(obs):
    return obs.get("private", {}) or {}


def idle_agent(obs, config=None):
    """Does nothing."""
    return {}


def random_agent(obs, config=None):
    rng = random.Random()
    phase = obs.get("phase")
    priv = _my(obs)
    if phase == "CONTRACTS":
        listings = obs.get("market", {}).get("listings", [])
        bids = [
            [lot["id"], round(lot["reserve"] * rng.uniform(0.80, 1.0), 2)] for lot in listings if rng.random() < 0.3
        ]
        return {"bids": bids, "max_lots": len(priv.get("trucks", [])) * FILL_CEILING}
    if phase == "DRIVING":
        trucks = {}
        for truck in priv.get("trucks", []):
            segs = [s["id"] for s in priv.get("segments", []) if s["pending"]]
            rng.shuffle(segs)
            trucks[truck["id"]] = {"route": segs[:12], "then": "RETURN"}
        return {"trucks": trucks}
    return {}


def greedy_agent(obs, config=None):
    """Bid at reserve, then run each truck nearest-neighbour over its segments,
    bucketed by window close.
    """
    phase = obs.get("phase")
    priv = _my(obs)
    trucks = priv.get("trucks", [])

    if phase == "CAPEX":
        warehouses = list(obs.get("city", {}).get("warehouses", {}))
        if not warehouses:
            return {}
        stage = {t["id"]: warehouses[i % len(warehouses)] for i, t in enumerate(trucks)}
        fuel = [["BULK_REFUEL", t["id"]] for t in trucks if t["fuel"] < 300]
        service = [["SERVICE", t["id"]] for t in trucks if t["km_since_service"] >= SERVICE_INTERVAL_KM]
        return {"stage": stage, "fuel": fuel, "fleet": service}

    if phase == "LABOR":
        drivers = priv.get("drivers", [])
        idle = [t for t in trucks if not t["driver"]]
        moves = [["ASSIGN", d["id"], t["id"]] for d, t in zip([d for d in drivers if not d["truck"]], idle)]
        if len(drivers) < len(trucks):
            best = sorted(obs.get("market", {}).get("candidates", []), key=lambda c: -c["resume"])
            if best:
                moves.append(["HIRE", best[0]["id"], round(best[0]["asking"] * 1.10 + 0.5)])
        return {"labor": moves}

    if phase == "CONTRACTS":
        # Only trucks that can run tomorrow.
        trucks = [t for t in trucks if t["driver"] and t["status"] not in ("DISABLED", "ORDERED")]
        listings = obs.get("market", {}).get("listings", [])
        listings = sorted(listings, key=lambda lot: -lot["reserve"] / max(lot["truck_days"], 0.01))
        # Up to 2 pairs per truck, at most FILL_CEILING per pair, within decks
        # and a total truck-day budget.
        filled, bids = {}, []
        decks = sorted((VEHICLES[t["type"]]["capacity"] for t in trucks), reverse=True)
        loaded = [0.0] * len(decks)
        budget = len(trucks) * FILL_CEILING
        booked = 0.0
        for lot in listings:
            pair = (lot["warehouse"], lot["district"])
            if pair not in filled and len(filled) >= 2 * len(trucks):
                continue
            if filled.get(pair, 0.0) + lot["truck_days"] > FILL_CEILING:
                continue
            if booked + lot["truck_days"] > budget:
                continue
            units = lot.get("parcel_units", 0.0)
            # Tightest deck that holds it.
            slot, waste = None, None
            for i, d in enumerate(decks):
                room = d - loaded[i]
                if room >= units and (waste is None or room - units < waste):
                    slot, waste = i, room - units
            if slot is None:
                continue
            loaded[slot] += units
            filled[pair] = filled.get(pair, 0.0) + lot["truck_days"]
            booked += lot["truck_days"]
            bids.append([lot["id"], lot["reserve"]])
        return {"bids": bids, "max_lots": round(len(trucks) * FILL_CEILING, 2)}

    # Plan once at 08:00; plans persist.
    if obs.get("block", 0) != 0:
        return {}
    plans = {}
    segments = {s["id"]: s for s in priv.get("segments", [])}
    windows = {a["id"]: a["window"] for a in priv.get("addresses", []) if a["window"]}
    for truck in trucks:
        carrying = set(truck["carrying"])
        mine = [s for s in segments.values() if any(a in carrying for a in s["pending"])]
        plans[truck["id"]] = {
            "route": _sequence(mine, windows),
            "then": "RETURN",
            "wait_cap": 15,
            "on_missed_window": "SKIP",
        }
    return {"trucks": plans}


def _sequence(segments, windows):
    """Nearest-neighbour inside each two-hour window-close bucket."""

    def closes(seg):
        ends = [windows[a][1] for a in seg["pending"] if a in windows]
        return min(ends) if ends else 10_000

    # Windowless segments sort last.
    buckets = {}
    for seg in segments:
        buckets.setdefault(min(closes(seg), 10_000) // 120, []).append(seg)

    here = (0.0, 0.0)
    route = []
    for _key in sorted(buckets):
        remaining = buckets[_key]
        while remaining:
            nxt = min(remaining, key=lambda s: abs(s["pos"][0] - here[0]) + abs(s["pos"][1] - here[1]))
            remaining.remove(nxt)
            route.append(nxt["id"])
            here = nxt["pos"]
    return route


agents = {"idle": idle_agent, "random": random_agent, "greedy": greedy_agent}
