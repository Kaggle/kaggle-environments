"""Lots, manifests, segments and the auction.

A lot is a fraction of a truck-day of work out of one warehouse. Its manifest
is drawn when posted and revealed to the winner at 08:00.
"""

import math

from .constants import (
    ARTERIAL_KMH,
    DEADLINE_MINUTES,
    DISTRICTS,
    DOCK_GRACE_MINUTES,
    DOCK_GRACE_SHARE,
    FAIL_PENALTY,
    FIXED_COST_DAY,
    GRID_DETOUR,
    LATE_PENALTY,
    LOAD_MINUTES,
    OVERHEAD_COST_DAY,
    PROMISED_PREMIUM,
    PROMISED_WIDTH,
    ROUTE_SLACK,
    SERVICE_INTERVAL_KM,
    SHIFT_MINUTES,
    TARGET_NET_PER_TRUCK_DAY,
    VEHICLES,
    WAGE_PER_MINUTE,
    WINDOW_DRAG,
    WINDOW_GRANULARITY,
    WINDOWS,
)


def stops_per_segment(district):
    """Doors on one block face that get a bundle today."""
    spec = DISTRICTS[district]
    doors = max(1, round(spec["block_m"] / _frontage(district))) * 2
    p_stop = 1.0 - math.exp(-spec["units"] * spec["pen"])
    return max(1.0, doors * p_stop * spec["share"])


def solve_truck_day(district, deadhead_minutes):
    """Stops that fit in one shift out of a given dock.

    Counts load, deadhead both ways, service, walking and driving between park
    spots. Tour length is Beardwood-Halton-Hammersley. The budget is discounted
    by the windowed share.
    """
    spec = DISTRICTS[district]
    win = WINDOWS[district]
    doors_per_seg = max(1, round(spec["block_m"] / _frontage(district))) * 2
    units_per_door = spec["units"]
    per_door_per_day = units_per_door * spec["pen"]
    p_stop = 1.0 - math.exp(-per_door_per_day)
    pkg_per_stop = per_door_per_day / p_stop if p_stop > 0 else 1.0
    stops_per_seg = doors_per_seg * p_stop * spec["share"]
    seg_per_km2 = 2e6 / (spec["block_m"] ** 2)
    stop_density = max(1e-6, stops_per_seg * seg_per_km2)

    windowed = spec["comm"] * win["dock_rate"] + (1.0 - spec["comm"]) * win["prom_rate"]
    budget = (SHIFT_MINUTES - LOAD_MINUTES - 2 * deadhead_minutes) * (1.0 - WINDOW_DRAG * windowed)
    if budget <= 0:
        return {
            "stops": 0,
            "packages": 0,
            "pkg_per_stop": pkg_per_stop,
            "stops_per_seg": stops_per_seg,
            "area": 0.0,
            "tour_km": 0.0,
        }

    lo, hi, best = 1.0, 900.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if _shift_minutes(mid, district, stop_density) <= budget:
            best, lo = mid, mid
        else:
            hi = mid
    stops = max(1, int(best))
    area = stops / stop_density
    return {
        "stops": stops,
        "packages": max(1, int(round(stops * pkg_per_stop))),
        "pkg_per_stop": pkg_per_stop,
        "stops_per_seg": stops_per_seg,
        "area": area,
        "tour_km": _tour_km(stops, area),
    }


def _frontage(district):
    """Metres of street frontage per door."""
    return {
        "DOWNTOWN": 25.0,
        "RIVERSIDE": 9.0,
        "MIDTOWN": 11.0,
        "SUBURBS_N": 18.0,
        "SUBURBS_S": 18.0,
        "INDUSTRIAL": 60.0,
    }[district]


def _tour_km(stops, area):
    return 0.75 * math.sqrt(max(stops, 1) * max(area, 1e-9)) * 1.27


def _shift_minutes(stops, district, stop_density):
    """Minutes to work `stops` doors.

    The truck parks once per block face and walks it, so the tour is over
    segments. The walk is the spread of `n` uniform draws, `(n-1)/(n+1)` of the
    block. `ROUTE_SLACK` scales the BHH tour length.
    """
    from .constants import LOCAL_SPEED_KMH, PARK_MINUTES, WALK_M_PER_S

    spec = DISTRICTS[district]
    area = stops / stop_density
    per_seg = max(1.0, stops_per_segment(district))
    parks = max(1.0, stops / per_seg)
    drive_km = _tour_km(parks, area) * ROUTE_SLACK
    walk_km = parks * spec["block_m"] / 1000.0 * (per_seg - 1.0) / (per_seg + 1.0)
    return (
        stops * spec["svc"]
        + walk_km * 1000.0 / WALK_M_PER_S / 60.0
        + drive_km * GRID_DETOUR / LOCAL_SPEED_KMH * 60.0
        + parks * PARK_MINUTES
    )


def expected_cost(district, truck_day, deadhead_minutes):
    """`E[cost]` of a full truck-day out of this dock."""
    spec = DISTRICTS[district]
    stops = truck_day["stops"]
    if stops <= 0:
        return FIXED_COST_DAY + OVERHEAD_COST_DAY
    local = _shift_minutes(stops, district, max(1e-6, stops / max(truck_day["area"], 1e-9)))
    minutes = LOAD_MINUTES + 2 * deadhead_minutes + local
    straight = min(minutes, SHIFT_MINUTES)
    overtime = max(0.0, minutes - SHIFT_MINUTES)
    wages = straight * WAGE_PER_MINUTE + overtime * WAGE_PER_MINUTE * 1.5
    km = truck_day["tour_km"] + 2 * deadhead_minutes / 60.0 * ARTERIAL_KMH
    fuel = km * 0.621 / 10.0 * 3.80 * (0.8 + spec["pkg_units"] * 0.05)
    # Priced at one wear-km per km on a VAN.
    maintenance = km * VEHICLES["VAN"]["service_cost"] / SERVICE_INTERVAL_KM
    expected_fails = truck_day["packages"] * 0.025
    penalties = expected_fails * FAIL_PENALTY + overtime / 8.0 * LATE_PENALTY
    return wages + fuel + FIXED_COST_DAY + OVERHEAD_COST_DAY + maintenance + penalties


def expected_premium(district, packages):
    """Expected `PROMISED` premium revenue."""
    spec = DISTRICTS[district]
    win = WINDOWS[district]
    return packages * (1.0 - spec["comm"]) * win["prom_rate"] * PROMISED_PREMIUM


def solve_reserve(district, truck_day, deadhead_minutes):
    """Base price for a $220 net margin."""
    cost = expected_cost(district, truck_day, deadhead_minutes)
    premium = expected_premium(district, truck_day["packages"])
    return max(1.0, cost + TARGET_NET_PER_TRUCK_DAY - premium)


class LotBoard:
    """Per-territory lot sizes and base prices, solved against the drawn map.

    Each `(warehouse, district)` pair is one territory with a fixed anchor
    intersection; every lot on that pair draws its addresses around it.
    """

    def __init__(self, city, rng):
        self.city = city
        self.truck_days = {}
        self.reserves = {}
        # Engine-only: published listings strip underscored keys.
        self.costs = {}
        self.deadheads = {}
        self.anchors = {}
        for wid, wnode in city.warehouse_of.items():
            for district in DISTRICTS:
                anchor = self._anchor(wnode, district, rng)
                dh = self._deadhead(wnode, anchor)
                td = solve_truck_day(district, dh)
                self.anchors[(wid, district)] = anchor
                self.deadheads[(wid, district)] = dh
                self.truck_days[(wid, district)] = td
                self.reserves[(wid, district)] = solve_reserve(district, td, dh)
                self.costs[(wid, district)] = self.reserves[(wid, district)] - TARGET_NET_PER_TRUCK_DAY
        self._next_lot = 0
        self._next_account = 0

    def _anchor(self, wnode, district, rng):
        """Anchor drawn from the six district nodes nearest the dock."""
        nodes = self.city.district_nodes[district]
        if not nodes:
            return wnode
        ranked = sorted(nodes, key=lambda x: self.city.node_km(wnode, x))
        return ranked[rng.randrange(min(len(ranked), 6))]

    def _deadhead(self, wnode, anchor):
        return self.city.node_km(wnode, anchor) / 45.0 * 60.0  # 45 km/h arterial

    def _listing(self, wid, district, packages, fraction, reserve, rng):
        spec = DISTRICTS[district]
        win = WINDOWS[district]
        truck_day = self.truck_days[(wid, district)]
        stops = max(1, int(round(truck_day["stops"] * fraction)))
        commercial = sum(1 for _ in range(packages) if rng.random() < spec["comm"])
        dock = sum(1 for _ in range(commercial) if rng.random() < win["dock_rate"])
        promised = sum(1 for _ in range(packages - commercial) if rng.random() < win["prom_rate"])
        lot_id = f"lot_{self._next_lot}"
        self._next_lot += 1
        return {
            "id": lot_id,
            "warehouse": wid,
            "district": district,
            "packages": packages,
            "stops": stops,
            "truck_days": round(fraction, 3),
            "anchor": self.anchors[(wid, district)],
            "area": round(truck_day["area"] * fraction, 4),
            "parcel_units": round(packages * spec["pkg_units"], 1),
            "deadline": DEADLINE_MINUTES,
            "payout_per_package": round(reserve / packages, 2),
            "late_penalty": LATE_PENALTY,
            "fail_penalty": FAIL_PENALTY,
            "dock_packages": dock,
            "promised_packages": promised,
            "reserve": round(reserve, 2),
            "kind": "SPOT",
            "retry": 0,
            "_cost": round(self.costs[(wid, district)] * fraction, 2),
        }


def draw_manifest(lot, city, rng):
    """The addresses, service times and windows behind a listing.

    Segments are positioned in kilometres from the pair's anchor intersection.
    """
    district = lot["district"]
    spec = DISTRICTS[district]
    win = WINDOWS[district]
    stops = max(1, lot["stops"])
    num_segments = max(1, int(round(stops / stops_per_segment(district))))
    anchor = lot.get("anchor", (city.district_nodes[district] or [0])[0])
    side = math.sqrt(max(lot.get("area", 0.0), 1e-4))

    segments, addresses = [], []
    remaining_pkgs = lot["packages"]
    dock_left, prom_left = lot["dock_packages"], lot["promised_packages"]
    for s in range(num_segments):
        pos = [round(rng.uniform(0, side), 4), round(rng.uniform(0, side), 4)]
        seg_id = f"{lot['id']}_seg_{s}"
        doors = []
        count = max(1, int(round(stops / num_segments)))
        # Every windowed door on a block face shares a start.
        seg_start = rng.randrange(0, max(1, (spec["span"] - win["dock_width"]) // WINDOW_GRANULARITY))
        seg_start *= WINDOW_GRANULARITY
        for d in range(count):
            if len(addresses) >= stops or remaining_pkgs <= 0:
                break
            pkgs = max(1, min(remaining_pkgs, int(round(rng.uniform(0.5, 1.5) * lot["packages"] / stops))))
            remaining_pkgs -= pkgs
            kind, window = None, None
            if dock_left > 0 and rng.random() < 0.5:
                kind = "DOCK"
                dock_left -= pkgs
                window = [seg_start, seg_start + win["dock_width"]]
            elif prom_left > 0 and rng.random() < 0.5:
                kind = "PROMISED"
                prom_left -= pkgs
                window = [seg_start, seg_start + PROMISED_WIDTH]
            addr = {
                "id": f"{lot['id']}_a_{len(addresses)}",
                "segment": seg_id,
                "node": anchor,
                "t": round(rng.random(), 3),
                "packages": pkgs,
                "units": spec["pkg_units"],
                "service": round(spec["svc"] * rng.uniform(0.6, 1.4), 2),
                "window": window,
                "window_kind": kind,
                "lot": lot["id"],
            }
            if kind == "DOCK":
                # Engine-only: how late this dock still takes a truck.
                late_ok = rng.random() < DOCK_GRACE_SHARE
                addr["_grace"] = round(rng.uniform(*DOCK_GRACE_MINUTES), 1) if late_ok else 0.0
            addresses.append(addr)
            doors.append(addr["id"])
        segments.append({"id": seg_id, "node": anchor, "pos": pos, "addresses": doors, "district": district})
    if remaining_pkgs > 0 and addresses:
        addresses[-1]["packages"] += remaining_pkgs
    return {"segments": segments, "addresses": addresses}


def noisy_service_estimate(address, rng):
    """True service time x U(0.7, 1.3)."""
    return round(address["service"] * rng.uniform(0.7, 1.3), 2)
