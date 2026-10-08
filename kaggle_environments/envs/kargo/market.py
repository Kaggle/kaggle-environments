"""The three overnight markets: capital, labor, freight.

Each resolves and publishes before the next opens.
"""

from .constants import (
    CANDIDATES_PER_NIGHT,
    CASH_FLOOR,
    CREDIT_LINE_DAILY_RATE,
    FILL_CEILING,
    FINANCE_DAILY_RATE,
    FINANCE_DOWN,
    FUEL_PRICE_BULK,
    MAX_DRIVERS,
    MAX_FLEET,
    OVERHEAD_COST_DAY,
    POACH_NOTICE_DAYS,
    RENTAL_LEAD_DAYS,
    RENTAL_MIN_DAYS,
    RENTAL_POOL_BASE,
    RESTLESS_ON_REFUSED_OFFER,
    SELL_HAIRCUT,
    SEVERANCE_DAYS,
    STANDING_BREAK_FEE_LOT_DAYS,
    STARTING_WAGE,
    USED_DISCOUNT,
    USED_LISTINGS_PER_NIGHT,
    VEHICLES,
    WAGE_FLOOR,
)
from .fleet import book_value, make_driver, make_truck, poach_accepts

# --- CAPEX ------------------------------------------------------------------


def post_used(rng, day):
    """Tonight's used listings: cheap, high-odometer, history undisclosed."""
    out = []
    for i in range(USED_LISTINGS_PER_NIGHT):
        vtype = list(VEHICLES)[rng.randrange(len(VEHICLES))]
        spec = VEHICLES[vtype]
        discount = rng.uniform(*USED_DISCOUNT)
        age = rng.randint(200, 1400)
        out.append(
            {
                "id": f"used_{day}_{i}",
                "type": vtype,
                "price": round(spec["buy"] * (1.0 - discount)),
                "odometer": round(age * rng.uniform(90, 160)),
                "age_days": age,
            }
        )
    return out


def rental_pool(rng, players):
    """Shrinks as the field rents more trucks."""
    taken = sum(len([t for t in p["trucks"].values() if t["ownership"] == "RENTED"]) for p in players)
    return {vtype: max(0, RENTAL_POOL_BASE - taken // 2 + rng.randint(-1, 1)) for vtype in VEHICLES}


def resolve_capex(players, actions, world, rng, day):
    """Fleet moves, fuel, and staging. Staging happens before the auction.

    Each player's own ops run first, then each used listing goes to a random
    one of its buyers and rental requests are filled in a shuffled order.
    """
    log = []
    pool = world["rental_pool"]
    used_bids, rent_bids = {}, []
    for pid, player in enumerate(players):
        act = actions[pid] if isinstance(actions[pid], dict) else {}
        for entry in _list(act.get("fleet")):
            op, arg = entry[0], entry[1] if len(entry) > 1 else None
            if op == "BUY":
                _acquire(player, arg, world, pid, log, financed=False)
            elif op == "FINANCE":
                _acquire(player, arg, world, pid, log, financed=True)
            elif op == "BUY_USED":
                used_bids.setdefault(arg, [])
                if pid not in used_bids[arg]:
                    used_bids[arg].append(pid)
            elif op == "RENT":
                rent_bids.append((pid, arg))
            elif op == "SELL":
                _sell(player, arg, world, pid, rng, log)
            elif op == "RETURN_RENTAL":
                _return_rental(player, arg, pid, log)
            elif op == "SERVICE":
                _service(player, arg, world, pid, log)
            elif op == "BREAK":
                _break_standing(player, arg, world, pid, log)

    for listing_id, buyers in used_bids.items():
        rng.shuffle(buyers)
        for pid in buyers:
            if _buy_used(players[pid], listing_id, world, pid, log, day):
                break
    rng.shuffle(rent_bids)
    for pid, vtype in rent_bids:
        _rent(players[pid], vtype, pool, world, pid, rng, log, day)

    for pid, player in enumerate(players):
        act = actions[pid] if isinstance(actions[pid], dict) else {}
        for entry in _list(act.get("fuel")):
            if entry[0] == "BULK_REFUEL":
                _refuel(player, entry[1] if len(entry) > 1 else None, world, pid)
        # A grounded truck stays put unless `SERVICE` cleared it above; a
        # rental can be staged the night it arrives.
        stage = act.get("stage") if isinstance(act.get("stage"), dict) else {}
        for tid, wid in stage.items():
            truck = player["trucks"].get(tid)
            if not truck or wid not in world["city"].warehouse_of or truck.get("arrives", 0) > day:
                continue
            if truck["status"] in ("IDLE", "ORDERED"):
                truck["staged"] = wid
                truck["node"] = world["city"].warehouse_of[wid]
    return log


def _acquire(player, vtype, world, pid, log, financed):
    if vtype not in VEHICLES or len(player["trucks"]) >= MAX_FLEET:
        return
    spec = VEHICLES[vtype]
    price = spec["buy"]
    down = price * FINANCE_DOWN if financed else price
    if player["cash"] - down < CASH_FLOOR:
        return
    player["cash"] -= down
    tid = _new_truck_id(player)
    truck = make_truck(tid, vtype, world["rng"], ownership="FINANCED" if financed else "OWNED")
    truck["player"] = pid
    truck["principal"] = price - down if financed else 0.0
    truck["arrives"] = 0
    player["trucks"][tid] = truck
    log.append({"player": pid, "op": "FINANCE" if financed else "BUY", "truck": tid, "type": vtype})


def _buy_used(player, listing_id, world, pid, log, day):
    listing = next((u for u in world["used"] if u["id"] == listing_id), None)
    if listing is None or len(player["trucks"]) >= MAX_FLEET or player["cash"] - listing["price"] < CASH_FLOOR:
        return False
    player["cash"] -= listing["price"]
    tid = _new_truck_id(player)
    truck = make_truck(
        tid,
        listing["type"],
        world["rng"],
        age_days=listing["age_days"],
        odometer=listing["odometer"],
        basis=listing["price"],
    )
    truck["player"] = pid
    truck["arrives"] = 0
    player["trucks"][tid] = truck
    world["used"].remove(listing)
    log.append({"player": pid, "op": "BUY_USED", "truck": tid, "type": listing["type"]})
    return True


def _rent(player, vtype, pool, world, pid, rng, log, day):
    if vtype not in VEHICLES or pool.get(vtype, 0) <= 0 or len(player["trucks"]) >= MAX_FLEET:
        return
    pool[vtype] -= 1
    lead = rng.randint(*RENTAL_LEAD_DAYS)
    tid = _new_truck_id(player)
    truck = make_truck(tid, vtype, rng, ownership="RENTED")
    truck["player"] = pid
    truck["arrives"] = day + lead
    truck["status"] = "ORDERED"
    truck["rental_days"] = 0
    player["trucks"][tid] = truck
    log.append({"player": pid, "op": "RENT", "truck": tid, "type": vtype, "arrives": truck["arrives"]})


def _sell(player, tid, world, pid, rng, log):
    truck = player["trucks"].get(tid)
    if truck is None or truck["ownership"] == "RENTED":
        return
    proceeds = book_value(truck) * (1.0 - rng.uniform(*SELL_HAIRCUT))
    player["cash"] += proceeds - truck["principal"]
    _release_driver(player, truck)
    del player["trucks"][tid]
    log.append({"player": pid, "op": "SELL", "truck": tid, "proceeds": round(proceeds, 2)})


def _return_rental(player, tid, pid, log):
    truck = player["trucks"].get(tid)
    if truck is None or truck["ownership"] != "RENTED" or truck["rental_days"] < RENTAL_MIN_DAYS:
        return
    _release_driver(player, truck)
    del player["trucks"][tid]
    log.append({"player": pid, "op": "RETURN_RENTAL", "truck": tid})


def _service(player, tid, world, pid, log):
    truck = player["trucks"].get(tid)
    if truck is None:
        return
    player["cash"] -= VEHICLES[truck["type"]]["service_cost"]
    truck["km_since_service"] = 0.0
    if truck["status"] == "DISABLED":
        truck["status"] = "IDLE"
    log.append({"player": pid, "op": "SERVICE", "truck": tid})


def _break_standing(player, account_id, world, pid, log):
    acct = next((a for a in player["standing"] if a["id"] == account_id), None)
    if acct is None:
        return
    fee = acct["rate"] * STANDING_BREAK_FEE_LOT_DAYS
    player["cash"] -= fee
    player["standing"].remove(acct)
    world["public_standing"] = [a for a in world["public_standing"] if a["id"] != account_id]
    log.append({"player": pid, "op": "BREAK", "account": account_id, "fee": round(fee, 2)})


def _refuel(player, tid, world, pid):
    truck = player["trucks"].get(tid)
    if truck is None:
        return
    spec = VEHICLES[truck["type"]]
    player["cash"] -= (spec["tank"] - truck["fuel"]) * FUEL_PRICE_BULK
    truck["fuel"] = float(spec["tank"])


def _release_driver(player, truck):
    did = truck.get("driver")
    if did and did in player["drivers"]:
        player["drivers"][did]["truck"] = None
    truck["driver"] = None


def _new_truck_id(player):
    player["truck_seq"] = player.get("truck_seq", 0) + 1
    return f"T{player['truck_seq']}"


def _list(v):
    return [e for e in v if isinstance(e, (list, tuple)) and e] if isinstance(v, list) else []


# --- LABOR ------------------------------------------------------------------


def post_candidates(rng, day):
    """Quality correlates with the ask, imperfectly."""
    out = []
    for i in range(rng.randint(*CANDIDATES_PER_NIGHT)):
        ask = STARTING_WAGE * rng.uniform(0.72, 1.45)
        quality = max(5.0, min(95.0, 20.0 + 55.0 * (ask / STARTING_WAGE) + rng.gauss(0, 13)))
        cand = make_driver(f"cand_{day}_{i}", rng, wage=round(ask), quality=quality)
        cand["asking"] = round(ask)
        out.append(cand)
    return out


def resolve_labor(players, actions, world, rng):
    """Wages, firings and assignments settle first; then notices, hires and
    poaches, each resolved across all players at once.

    Each candidate goes to the highest offer that clears their reservation,
    ties broken at random. An accepted poach serves notice: the driver works
    `POACH_NOTICE_DAYS` more days for their employer, who keeps them by
    raising their wage to the offer.
    """
    day = world["day"]
    log = []
    poaches = []
    offers = {}  # candidate id -> {pid: best wage offered}
    for pid, player in enumerate(players):
        act = actions[pid] if isinstance(actions[pid], dict) else {}
        targeted = set()
        for entry in _list(act.get("labor")):
            op = entry[0]
            if op == "HIRE" and len(entry) >= 3:
                mine = offers.setdefault(entry[1], {})
                mine[pid] = max(mine.get(pid, 0.0), float(entry[2]))
            elif op == "WAGE" and len(entry) >= 3:
                driver = player["drivers"].get(entry[1])
                if driver:
                    driver["wage"] = max(WAGE_FLOOR, float(entry[2]))
            elif op == "FIRE" and len(entry) >= 2:
                _fire(players, pid, entry[1], log)
            elif op == "POACH" and len(entry) >= 4:
                # One attempt per rival per night.
                target = int(entry[1])
                if target != pid and target not in targeted:
                    targeted.add(target)
                    poaches.append((pid, target, entry[2], float(entry[3])))
            elif op == "ASSIGN" and len(entry) >= 3:
                _assign(player, entry[1], entry[2])
    log.extend(_resolve_notices(players, day))
    for cand in list(world["candidates"]):
        bids = sorted(((w, pid) for pid, w in offers.get(cand["id"], {}).items()), key=lambda b: (-b[0], rng.random()))
        for wage, pid in bids:
            if wage >= cand["reservation"] and _headcount(players, pid) < MAX_DRIVERS:
                _hire(players[pid], cand, wage, world, pid, log)
                break
    log.extend(_resolve_poaches(players, poaches, rng, day))
    return log


def _headcount(players, pid):
    """Roster plus drivers serving notice to join it."""
    incoming = sum(1 for p in players for d in p["drivers"].values() if d.get("departs") and d["departs"]["to"] == pid)
    return len(players[pid]["drivers"]) + incoming


def _hire(player, cand, wage, world, pid, log):
    world["candidates"].remove(cand)
    player["driver_seq"] = player.get("driver_seq", 0) + 1
    cand["id"] = f"D{pid}_{player['driver_seq']}"
    cand["wage"] = max(WAGE_FLOOR, float(wage))
    cand.pop("asking", None)
    player["drivers"][cand["id"]] = cand
    log.append({"player": pid, "op": "HIRE", "driver": cand["id"], "name": cand["name"]})


def _fire(players, pid, did, log):
    """Severance is charged unless the driver is serving notice, who leaves for the bidder at once."""
    player = players[pid]
    driver = player["drivers"].pop(did, None)
    if driver is None:
        return
    _unseat(player, did)
    if driver.get("departs"):
        log.append(_transfer(players, pid, driver))
        return
    player["cash"] -= driver["wage"] * SEVERANCE_DAYS
    log.append({"player": pid, "op": "FIRE", "driver": did})


def _unseat(player, did):
    for truck in player["trucks"].values():
        if truck.get("driver") == did:
            truck["driver"] = None


def _assign(player, did, tid):
    driver = player["drivers"].get(did)
    truck = player["trucks"].get(tid)
    if driver is None or truck is None:
        return
    for other in player["trucks"].values():
        if other.get("driver") == did:
            other["driver"] = None
    if truck.get("driver"):
        player["drivers"][truck["driver"]]["truck"] = None
    truck["driver"] = did
    driver["truck"] = tid


def _resolve_notices(players, day):
    """A notice matched by the employer is withdrawn; a served one moves the driver."""
    log = []
    for pid, player in enumerate(players):
        for did, driver in list(player["drivers"].items()):
            departs = driver.get("departs")
            if not departs:
                continue
            if driver["wage"] >= departs["offer"]:
                driver["departs"] = None
                log.append({"op": "POACH_MATCHED", "employer": pid, "driver": did})
            elif day >= departs["day"]:
                del player["drivers"][did]
                _unseat(player, did)
                log.append(_transfer(players, pid, driver))
    return log


def _transfer(players, src, driver):
    bidder, offer = driver["departs"]["to"], driver["departs"]["offer"]
    new = players[bidder]
    new["driver_seq"] = new.get("driver_seq", 0) + 1
    old = driver["id"]
    driver["id"] = f"D{bidder}_{new['driver_seq']}"
    driver["wage"] = offer
    driver["tenure"] = 0
    driver["truck"] = None
    driver["departs"] = None
    new["drivers"][driver["id"]] = driver
    return {"op": "POACH_TRANSFER", "from": src, "to": bidder, "driver": old, "new_id": driver["id"]}


def _resolve_poaches(players, poaches, rng, day):
    """Only the highest offer per driver is considered."""
    log = []
    by_driver = {}
    for bidder, target, did, offer in poaches:
        by_driver.setdefault((target, did), []).append((offer, bidder))
    for (target, did), bids in by_driver.items():
        if not 0 <= target < len(players):
            continue
        driver = players[target]["drivers"].get(did)
        if driver is None or driver.get("departs"):
            continue
        bids = [b for b in bids if _headcount(players, b[1]) < MAX_DRIVERS]
        if not bids:
            continue
        bids.sort(key=lambda b: (-b[0], rng.random()))
        offer, bidder = bids[0]
        if poach_accepts(driver, offer, rng):
            departs = day + POACH_NOTICE_DAYS
            driver["departs"] = {"to": bidder, "offer": offer, "day": departs}
            log.append({"op": "POACH_ACCEPTED", "from": target, "to": bidder, "driver": did, "departs": departs})
        else:
            driver["restlessness"] += RESTLESS_ON_REFUSED_OFFER
            log.append({"op": "POACH_REFUSED", "employer": target, "driver": did})
        for _offer, loser in bids[1:]:
            log.append({"op": "OUTBID", "player": loser})
    return log


# --- CONTRACTS --------------------------------------------------------------


def resolve_auction(players, actions, world, rng):
    """First-price sealed-bid reverse auction under a per-player fill cap.

    Provisionally award each lot to its lowest bid, then trim every player who
    is over `max_lots` -- keeping their best-margin wins -- and release the rest
    to the next-lowest bidder. Iterate to a fixed point.
    """
    listings = {lot["id"]: lot for lot in world["listings"] + world["accounts"]}
    bids = {}  # lot id -> {pid: lowest ask}
    caps = {}
    fleets = {}
    decks = {}
    terms = {}
    for pid, player in enumerate(players):
        act = actions[pid] if isinstance(actions[pid], dict) else {}
        live = _runnable(player, world["day"])
        caps[pid] = _cap(act, len(live))
        fleets[pid] = len(live)
        # Every deck, largest first: each territory needs its own truck.
        decks[pid] = sorted((VEHICLES[t["type"]]["capacity"] for t in live), reverse=True)
        for entry in _list(act.get("bids")):
            if len(entry) < 2 or entry[0] not in listings or listings[entry[0]]["kind"] == "STANDING":
                continue
            _bid(bids, entry[0], pid, float(entry[1]))
        for entry in _list(act.get("standing_bids")):
            if len(entry) < 3 or entry[0] not in listings or listings[entry[0]]["kind"] != "STANDING":
                continue
            term = int(entry[2])
            if term not in listings[entry[0]].get("term_options", []):
                continue
            if _bid(bids, entry[0], pid, float(entry[1])):
                terms[(entry[0], pid)] = term

    ranked = {}
    for lot_id, entries in bids.items():
        reserve = listings[lot_id]["reserve"]
        keep = [(ask, pid) for pid, ask in entries.items() if ask <= reserve]
        keep.sort(key=lambda b: (b[0], rng.random()))
        if keep:
            ranked[lot_id] = keep

    # Tomorrow's lots from accounts already held count against the same limits.
    held = {}
    for pid, player in enumerate(players):
        load = {}
        for acct in player["standing"]:
            if acct["remaining"] > 0:
                pair = (acct["warehouse"], acct["district"])
                td, units = load.get(pair, (0.0, 0.0))
                load[pair] = (td + acct["truck_days"], units + acct["parcel_units"])
        held[pid] = load

    # Every pass that trims advances a head, so this ends within one pass per bid.
    awards, head = {}, dict.fromkeys(ranked, 0)
    while True:
        awards = {}
        for lot_id, keep in ranked.items():
            if head[lot_id] < len(keep):
                ask, pid = keep[head[lot_id]]
                awards[lot_id] = (pid, ask)
        over = False
        for pid in range(len(players)):
            mine = [(lid, ask) for lid, (p, ask) in awards.items() if p == pid]
            mine.sort(key=lambda x: -_margin(listings[x[0]], x[1]))
            territory = dict(held[pid])
            used = sum(td for td, _u in territory.values())
            for lid, _ask in mine:
                listing = listings[lid]
                size = listing["truck_days"]
                # Total work fits the fleet's truck-days, each territory fits
                # one truck, and the territories together fit the fleet's decks.
                pair = (listing["warehouse"], listing["district"])
                opened = territory.get(pair, (0.0, 0.0))
                pairs_used = len(territory) + (0 if pair in territory else 1)
                units = opened[1] + listing["parcel_units"]
                trial = dict(territory)
                trial[pair] = (opened[0] + size, units)
                if (
                    used + size <= caps[pid] + 1e-9
                    and opened[0] + size <= FILL_CEILING + 1e-9
                    and pairs_used <= fleets[pid]
                    and _decks_fit([u for _td, u in trial.values()], decks[pid])
                ):
                    used += size
                    territory[pair] = (opened[0] + size, units)
                else:
                    head[lid] += 1
                    over = True
        if not over:
            break

    log = []
    for lot_id, (pid, ask) in awards.items():
        listing = listings[lot_id]
        player = players[pid]
        if listing["kind"] == "STANDING":
            term = terms[(lot_id, pid)]
            acct = {
                "id": lot_id,
                "player": pid,
                "warehouse": listing["warehouse"],
                "district": listing["district"],
                "rate": ask,
                "term": term,
                "remaining": term,
                "truck_days": listing["truck_days"],
                "parcel_units": listing["parcel_units"],
            }
            player["standing"].append(acct)
            world["public_standing"].append(dict(acct))
        else:
            awarded = dict(listing)
            awarded["price"] = ask
            awarded["payout_per_package"] = round(ask / max(1, listing["packages"]), 4)
            player["pending_lots"].append(awarded)
        log.append(
            {"lot": lot_id, "player": pid, "ask": round(ask, 2), "reserve": listing["reserve"], "kind": listing["kind"]}
        )
    world["bid_book"] = [{"lot": lid, "bids": [[round(a, 2), p] for a, p in e]} for lid, e in ranked.items()]
    return log


def _bid(bids, lot_id, pid, ask):
    """Record a player's ask, keeping their lowest per lot. True if it is now their bid."""
    mine = bids.setdefault(lot_id, {})
    if pid in mine and mine[pid] <= ask:
        return False
    mine[pid] = ask
    return True


def _decks_fit(loads, decks):
    """Can each territory's freight get a truck of its own?

    With both sides sorted descending, the greedy match is exact.
    """
    if len(loads) > len(decks):
        return False
    return all(load <= deck + 1e-9 for load, deck in zip(sorted(loads, reverse=True), decks))


def _runnable(player, day):
    """Trucks that will roll tomorrow: arrived, not grounded, and crewed."""
    return [
        t for t in player["trucks"].values() if t.get("arrives", 0) <= day and t["status"] != "DISABLED" and t["driver"]
    ]


def _cap(act, trucks):
    requested = act.get("max_lots")
    if requested is None:
        requested = trucks * FILL_CEILING
    return max(0.0, min(float(requested), trucks * 3.0))


def _margin(listing, ask):
    """Ask over the engine's expected cost for the lot."""
    return ask - listing["_cost"]


# --- nightly accounting -----------------------------------------------------


def accrue(player, day):
    """Holding costs, financing, wages, overhead, and the credit line."""
    for truck in player["trucks"].values():
        if truck.get("arrives", 0) > day:
            continue
        spec = VEHICLES[truck["type"]]
        truck["age_days"] += 1
        if truck["ownership"] == "RENTED":
            player["cash"] -= spec["rent_day"]
            truck["rental_days"] += 1
        else:
            player["cash"] -= spec["own_day"]
        if truck["principal"] > 0:
            interest = truck["principal"] * FINANCE_DAILY_RATE
            payment = min(truck["principal"] + interest, spec["buy"] / 365.0 + interest)
            player["cash"] -= payment
            truck["principal"] = max(0.0, truck["principal"] + interest - payment)
        truck["book_value"] = book_value(truck)
    player["cash"] -= OVERHEAD_COST_DAY
    # Negative cash becomes debt; positive cash repays debt first.
    if player["cash"] < 0:
        player["debt"] += -player["cash"]
        player["cash"] = 0.0
    elif player["debt"] > 0:
        repaid = min(player["cash"], player["debt"])
        player["cash"] -= repaid
        player["debt"] -= repaid
    player["debt"] *= 1.0 + CREDIT_LINE_DAILY_RATE
    return player


def net_worth(player):
    # Rentals are not assets.
    fleet = sum(book_value(t) for t in player["trucks"].values() if t["ownership"] != "RENTED")
    principal = sum(t["principal"] for t in player["trucks"].values())
    return player["cash"] + fleet - principal - player["debt"]
