"""Advancing a truck through a driving block.

The simulation is continuous: fractional edge progress, real-numbered service
times. The block is only how often the agent may act.
"""

from .constants import (
    ABANDON_FEE_PER_UNIT,
    BLOCK_MINUTES,
    DAY_END_MINUTES,
    DISTRICTS,
    DOCK_REFUSAL,
    FAIL_PENALTY,
    FUEL_CALL_COST,
    FUEL_CALL_MINUTES,
    GRID_DETOUR,
    LATE_PENALTY,
    LOAD_MINUTES,
    PARK_MINUTES,
    PROMISED_PREMIUM,
    SERVICE_INTERVAL_KM,
    SHIFT_MINUTES,
    VEHICLES,
    WALK_M_PER_S,
)
from .fleet import care_multiplier, effective_stats, service_multiplier, speed_multiplier


def _breadcrumb(truck, kind, at_node=False):
    """Stamp where the truck is, so a replay can retrace the block.

    `off` is the offset in kilometres from `node` while the truck is inside the
    territory anchored there; null on the intersection itself.
    """
    pos = truck.get("pos")
    inside = not at_node and pos is not None and truck.get("anchor") == truck["node"]
    off = [round(pos[0], 4), round(pos[1], 4)] if inside else None
    truck.setdefault("trail", []).append(
        {"t": round(truck["clock"], 1), "node": truck["node"], "kind": kind, "off": off}
    )


def advance_truck(truck, plan, world, player, until, rng):
    """Run one truck from its current clock to `until`. Mutates state in place.

    Returns a list of event dicts for the replay and the end-of-day report.
    """
    events = []
    # A fresh trail per block.
    truck["trail"] = []
    _breadcrumb(truck, "START")
    if truck["status"] == "DISABLED":
        return events
    driver = player["drivers"].get(truck["driver"]) if truck["driver"] else None
    if driver is None:
        for lot_id in truck.get("load", []):
            events.append(_event("LOAD_REFUSED", truck, world, lot=lot_id, address=lot_id, reason="NO_DRIVER"))
        truck["load"] = []
        return events
    if plan.get("hold"):
        truck["clock"] = min(until, truck["clock"])
        return events

    city = world["city"]
    guard = 0

    while truck["clock"] < until and guard < 400:
        guard += 1
        if truck.get("load"):
            if not _load_next(truck, world, player, driver, until, rng, events):
                break
            continue
        target = _next_target(truck, plan, world, player)
        if target is None:
            _finish(truck, plan, world, player, until, rng, events)
            break

        stats = effective_stats(driver, truck["clock"], truck["overtime"])

        arrival = _drive_to(truck, target["node"], world, until, stats, rng, events)
        if arrival is False:
            # Unreachable target: drop it.
            truck["route"].pop(0)
            continue
        if arrival is None:  # ran out of block, or stranded
            truck["worked"] = max(truck.get("worked", 0.0), truck["clock"])
            break

        truck["worked"] = max(truck.get("worked", 0.0), truck["clock"])
        _approach(truck, target, city, stats)
        _serve(truck, target, plan, world, player, stats, rng, events)
        truck["worked"] = max(truck["worked"], truck["clock"])
        _breadcrumb(truck, "DEPART")

    return events


def lot_units(player, addrs):
    """Parcel-units of a set of doors."""
    manifest = player["manifest"]["addresses"]
    total = 0.0
    for aid in sorted(addrs):
        addr = manifest.get(aid)
        if addr:
            per = addr.get("units")
            if per is None:
                per = DISTRICTS[player["manifest"]["segments"][addr["segment"]]["district"]]["pkg_units"]
            total += addr["packages"] * per
    return total


def on_board_pair(truck, player):
    """The (warehouse, district) of the freight on board, or None when empty."""
    manifest = player["manifest"]["addresses"]
    for aid in truck["carrying"]:
        lot = player["lots"].get(manifest[aid]["lot"]) if aid in manifest else None
        if lot:
            return (lot["warehouse"], lot["district"])
    return None


def _load_check(truck, player, lot, dock):
    """Why this lot cannot go on this truck, or None."""
    if lot is None or not dock:
        return "NOT_AT_DOCK"
    pair = on_board_pair(truck, player)
    if pair is not None and pair != (lot["warehouse"], lot["district"]):
        return "PAIR"
    if lot_units(player, truck["carrying"]) + lot_units(player, dock) > VEHICLES[truck["type"]]["capacity"] + 1e-9:
        return "CAPACITY"
    return None


def _load_next(truck, world, player, driver, until, rng, events):
    """Work the head of the load queue: drive to the lot's warehouse, then load it.

    Returns False when the block runs out before the load is done.
    """
    lot_id = truck["load"][0]
    lot = player["lots"].get(lot_id)

    def refuse(reason):
        truck["load"].pop(0)
        events.append(_event("LOAD_REFUSED", truck, world, lot=lot_id, address=lot_id, reason=reason))
        return True

    reason = _load_check(truck, player, lot, player["dock"].get(lot_id))
    if reason:
        return refuse(reason)
    origin = world["city"].warehouse_of[lot["warehouse"]]
    if truck["node"] != origin:
        stats = effective_stats(driver, truck["clock"], truck["overtime"])
        arrival = _drive_to(truck, origin, world, until, stats, rng, events)
        truck["worked"] = max(truck.get("worked", 0.0), truck["clock"])
        if arrival is False:
            return refuse("UNREACHABLE")
        if arrival is None:
            return False
    dock = player["dock"].get(lot_id)
    reason = _load_check(truck, player, lot, dock)
    if reason:
        return refuse(reason)
    if truck["clock"] + LOAD_MINUTES >= SHIFT_MINUTES:
        return refuse("TOO_LATE")

    truck["clock"] += LOAD_MINUTES
    truck["worked"] = max(truck.get("worked", 0.0), truck["clock"])
    truck["carrying"] |= dock
    del player["dock"][lot_id]
    lot["truck"] = truck["id"]
    truck["lots"].append(lot_id)
    truck["home"] = origin
    truck["anchor"] = None
    truck["status"] = "ACTIVE"
    truck["load"].pop(0)
    packages = sum(player["manifest"]["addresses"][a]["packages"] for a in dock)
    events.append(_event("LOADED", truck, world, lot=lot_id, address=lot_id, packages=packages))
    _breadcrumb(truck, "STOP", at_node=True)
    return True


def _approach(truck, target, city, stats):
    """Interior-street travel from the last stop to this one."""
    if target["kind"] == "VIA":
        return
    here = truck["pos"] if truck.get("anchor") == target["node"] else (0.0, 0.0)
    km = abs(target["pos"][0] - here[0]) + abs(target["pos"][1] - here[1])
    truck["clock"] += city.local_minutes(km) * speed_multiplier(stats) + PARK_MINUTES
    truck["anchor"] = target["node"]
    truck["pos"] = (target["pos"][0], target["pos"][1])
    _wear(truck, km * GRID_DETOUR, stats)
    _breadcrumb(truck, "STOP")


def _next_target(truck, plan, world, player):
    """Peek at the head of the route, dropping anything already done."""
    route = truck["route"]
    while route:
        entry = route[0]
        resolved = _resolve(entry, truck, world, player)
        if resolved is None or not resolved["addresses"]:
            route.pop(0)
            continue
        return resolved
    return None


def _resolve(entry, truck, world, player):
    """A route entry is a segment, an address, or a `via` waypoint."""
    if isinstance(entry, dict):
        if "via" in entry:
            return {"kind": "VIA", "node": int(entry["via"]), "addresses": ["via"]}
        seg_id = entry.get("seg")
        only = entry.get("only")
    else:
        seg_id, only = entry, None

    manifest = player["manifest"]
    if seg_id in manifest["segments"]:
        segment = manifest["segments"][seg_id]
        pending = [a for a in segment["addresses"] if a in truck["carrying"]]
        if only == "windowed":
            pending = [a for a in pending if manifest["addresses"][a]["window"]]
        elif only == "free":
            pending = [a for a in pending if not manifest["addresses"][a]["window"]]
        return {"kind": "SEG", "node": segment["node"], "pos": segment["pos"], "addresses": pending, "id": seg_id}
    if seg_id in manifest["addresses"]:
        addr = manifest["addresses"][seg_id]
        segment = manifest["segments"][addr["segment"]]
        pending = [seg_id] if seg_id in truck["carrying"] else []
        return {
            "kind": "ADDR",
            "node": addr["node"],
            "pos": segment["pos"],
            "addresses": pending,
            "id": addr["segment"],
            "address": seg_id,
        }
    return None


def _drive_to(truck, node, world, until, stats, rng, events):
    """Move along edges, committed once entered.

    Returns the arrival minute, None if the block ran out first, or False if
    the target cannot be reached at all.
    """
    city = world["city"]
    if truck["node"] == node:
        return truck["clock"]

    speed = speed_multiplier(stats)
    guard = 0
    while truck["node"] != node and guard < 80:
        guard += 1
        _cost, path = city.route(truck["node"], node, truck["clock"])
        if not path:
            return False  # unreachable from here at this minute
        edge = path[0]
        minutes = city.edge_time(edge, truck["clock"])
        if minutes is None:
            return None
        minutes *= speed
        if truck["clock"] + minutes > until:
            # An edge once entered is committed, so the truck may overrun the
            # block boundary -- but only by the edge it is already on.
            if truck["clock"] >= until:
                return None
        u, v, _k, _t, _c = city.edges[edge]
        city.occupy(edge, truck["clock"], truck["clock"] + minutes)
        truck["node"] = v if truck["node"] == u else u
        truck["clock"] += minutes
        # `anchor` is kept so a return to the same territory resumes from the
        # last kerb.
        _breadcrumb(truck, "DRIVE", at_node=True)
        _wear(truck, 1.5, stats)
        truck["fuel"] -= VEHICLES[truck["type"]]["fuel_per_min"] * minutes

        if truck["fuel"] <= 0 and not _refuel_call(truck, world, rng, events):
            return None
        if truck["clock"] >= until:
            return None if truck["node"] != node else truck["clock"]
    return truck["clock"] if truck["node"] == node else None


def _event(kind, truck, world, **extra):
    """One event record.

    Truck ids are only unique within a player, so every event carries `player`.
    `day` is stamped here because 18:00 events are published in the next day's
    `CAPEX` step.
    """
    return {
        "kind": kind,
        "player": truck["player"],
        "truck": truck["id"],
        "node": truck["node"],
        "day": world["day"],
        "minute": round(truck["clock"], 1),
        **extra,
    }


def _wear(truck, km, stats):
    """The odometer counts kilometres; service wear counts how they were driven."""
    truck["odometer"] += km
    truck["km_since_service"] += km * care_multiplier(stats)


def _refuel_call(truck, world, rng, events):
    """Running dry strands the truck until the fuel call arrives."""
    truck["clock"] += FUEL_CALL_MINUTES
    truck["fuel"] = VEHICLES[truck["type"]]["tank"] * 0.5
    world["charges"].append((truck["player"], FUEL_CALL_COST, "FUEL_CALL"))
    events.append(_event("RAN_DRY", truck, world))
    return truck["clock"] < DAY_END_MINUTES


def _serve(truck, target, plan, world, player, stats, rng, events):
    """Work the doors at this target. Order within a segment is forced."""
    if target["kind"] == "VIA":
        truck["route"].pop(0)
        return

    manifest = player["manifest"]
    district = manifest["segments"].get(target.get("id"), {}).get("district")
    wait_cap = float(plan.get("wait_cap", 20))
    on_missed = plan.get("on_missed_window", "SKIP")
    svc_mult = service_multiplier(stats)

    # `last_t` is the position along this block face; it resets on arrival.
    ordered = sorted(target["addresses"], key=lambda a: manifest["addresses"][a]["t"])
    if ordered:
        truck["last_t"] = manifest["addresses"][ordered[0]]["t"]
    for aid in ordered:
        addr = manifest["addresses"][aid]
        if aid not in truck["carrying"]:
            continue
        if truck["clock"] >= DAY_END_MINUTES:
            break

        window = addr["window"]
        if window:
            if truck["clock"] < window[0]:
                wait = window[0] - truck["clock"]
                if wait > wait_cap:
                    continue  # come back later; the door stays pending
                truck["clock"] = window[0]
            elif truck["clock"] > window[1] and addr["window_kind"] == "DOCK":
                if on_missed == "SKIP":
                    continue
                # Inside the door's hidden grace the attempt delivers.
                if truck["clock"] > window[1] + addr.get("_grace", 0.0):
                    _fail(truck, addr, world, player, "REFUSED", events, extra=DOCK_REFUSAL)
                    continue

        # Walk the block face to the next door, then serve it.
        if district:
            walk_m = DISTRICTS[district]["block_m"] * abs(addr["t"] - truck.get("last_t", 0.0))
            truck["clock"] += walk_m / WALK_M_PER_S / 60.0
        truck["last_t"] = addr["t"]
        truck["clock"] += addr["service"] * svc_mult
        if truck["clock"] > SHIFT_MINUTES:
            truck["overtime"] = truck["clock"] - SHIFT_MINUTES

        _deliver(truck, addr, world, player, events)

    # Pop the head. A segment with servable doors left goes to the back, at
    # most twice per day.
    if truck["route"]:
        head = truck["route"].pop(0)
        servable = [
            a
            for a in target["addresses"]
            if a in truck["carrying"]
            and (not manifest["addresses"][a]["window"] or truck["clock"] < manifest["addresses"][a]["window"][1])
        ]
        seen = truck.setdefault("revisited", {})
        if servable and seen.get(target["id"], 0) < 2:
            seen[target["id"]] = seen.get(target["id"], 0) + 1
            truck["route"].append(head)


def _deliver(truck, addr, world, player, events):
    pid = truck["player"]
    lot = player["lots"].get(addr["lot"])
    payout = (lot["payout_per_package"] if lot else 0.0) * addr["packages"]
    late = truck["clock"] > lot["deadline"] if lot else truck["clock"] > SHIFT_MINUTES
    premium = 0.0
    if (
        addr["window_kind"] == "PROMISED"
        and addr["window"]
        and addr["window"][0] <= truck["clock"] <= addr["window"][1]
    ):
        premium = PROMISED_PREMIUM * addr["packages"]
    penalty = LATE_PENALTY * addr["packages"] if late else 0.0

    world["credits"].append((pid, payout + premium - penalty, "DELIVERY"))
    if lot is not None:
        lot["delivered"] = lot.get("delivered", 0) + addr["packages"]
    truck["carrying"].discard(addr["id"])
    player["day_report"]["delivered"] += addr["packages"]
    if late:
        player["day_report"]["late"] += addr["packages"]
    events.append(
        _event(
            "DELIVER",
            truck,
            world,
            address=addr["id"],
            segment=addr["segment"],
            packages=addr["packages"],
            late=late,
            premium=premium > 0,
        )
    )


def _fail(truck, addr, world, player, reason, events, extra=0.0, cost=None):
    pid = truck["player"]
    if cost is None:
        cost = FAIL_PENALTY * addr["packages"] + extra
    world["charges"].append((pid, cost, reason))
    truck["carrying"].discard(addr["id"])
    key = "refused" if reason == "REFUSED" else "failed"
    player["day_report"][key] += addr["packages"]
    events.append(
        _event(
            reason,
            truck,
            world,
            address=addr["id"],
            segment=addr["segment"],
            packages=addr["packages"],
            cost=round(cost, 2),
        )
    )


def abandon(player, pid, ids, world, events):
    """Write held freight off for a fee per parcel-unit.

    Takes lot, segment or door ids, at the dock or on a truck. The packages go
    back to the shipper.
    """
    manifest = player["manifest"]
    at_dock = {}  # lot id -> doors written off at the dock
    for target in ids if isinstance(ids, list) else []:
        if not isinstance(target, str):
            continue
        if target in manifest["segments"]:
            addrs = list(manifest["segments"][target]["addresses"])
        elif target in manifest["addresses"]:
            addrs = [target]
        else:
            addrs = [a for a, x in manifest["addresses"].items() if x["lot"] == target]
        for aid in addrs:
            addr = manifest["addresses"][aid]
            dock = player["dock"].get(addr["lot"])
            if dock and aid in dock:
                dock.discard(aid)
                at_dock.setdefault(addr["lot"], set()).add(aid)
                continue
            truck = next((t for t in player["trucks"].values() if aid in t["carrying"]), None)
            if truck is None:
                continue
            fee = ABANDON_FEE_PER_UNIT * addr["packages"] * addr.get("units", 1.0)
            _fail(truck, addr, world, player, "ABANDONED", events, cost=fee)
    for lot_id, addrs in at_dock.items():
        fee = ABANDON_FEE_PER_UNIT * lot_units(player, addrs)
        _fail_at_dock(player, pid, lot_id, addrs, world, "ABANDONED", fee, events)
        if not player["dock"].get(lot_id):
            player["dock"].pop(lot_id, None)


def _fail_at_dock(player, pid, lot_id, addrs, world, reason, cost, events):
    """Freight that never left the warehouse: one charge and one event per lot."""
    packages = sum(player["manifest"]["addresses"][a]["packages"] for a in addrs)
    world["charges"].append((pid, cost, reason))
    player["day_report"]["failed"] += packages
    lot = player["lots"][lot_id]
    events.append(
        {
            "kind": reason,
            "player": pid,
            "truck": "",
            "node": world["city"].warehouse_of[lot["warehouse"]],
            "day": world["day"],
            "minute": round(DAY_END_MINUTES if reason == "UNDELIVERED" else 0.0, 1),
            "address": lot_id,
            "lot": lot_id,
            "packages": packages,
            "cost": round(cost, 2),
        }
    )


def dock_failures(player, pid, world, events):
    """Lots still at the warehouse at 18:00 fail like any undelivered freight."""
    for lot_id, addrs in sorted(player["dock"].items()):
        if addrs:
            packages = sum(player["manifest"]["addresses"][a]["packages"] for a in addrs)
            _fail_at_dock(player, pid, lot_id, addrs, world, "UNDELIVERED", FAIL_PENALTY * packages, events)
    player["dock"] = {}


def _finish(truck, plan, world, player, until, rng, events):
    """Route empty: `RETURN` drives home once the truck is empty; otherwise wait."""
    then = plan.get("then", "RETURN")
    if then == "RETURN" and truck["node"] != truck["home"] and not truck["carrying"]:
        stats = effective_stats(player["drivers"][truck["driver"]], truck["clock"], truck["overtime"])
        _drive_to(truck, truck["home"], world, until, stats, rng, events)
    # The clock advances to the block boundary; `worked` does not.
    if truck["clock"] > until - BLOCK_MINUTES:
        truck["worked"] = max(truck.get("worked", 0.0), truck["clock"])
    truck["clock"] = max(truck["clock"], until)


def end_of_day(truck, world, player, events):
    """Undelivered at 18:00 is forfeited and penalised."""
    for aid in sorted(truck["carrying"]):
        addr = player["manifest"]["addresses"].get(aid)
        if addr:
            _fail(truck, addr, world, player, "UNDELIVERED", events)


def service_check(truck, world, events):
    """Past the interval at 18:00: grounded until a `SERVICE` clears it."""
    if truck["status"] == "DISABLED" or truck["km_since_service"] < SERVICE_INTERVAL_KM:
        return
    truck["status"] = "DISABLED"
    events.append(_event("SERVICE_DUE", truck, world, km_since_service=round(truck["km_since_service"], 1)))
