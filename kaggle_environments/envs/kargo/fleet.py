"""Trucks, drivers, and the labor market.

Stats are hidden and travel with the driver; the public résumé rating is
drawn once from them, with noise.
"""

from .constants import (
    DRIVER_STATS,
    MORALE_NOTICE,
    MORALE_OVERTIME_COST,
    MORALE_QUIT,
    MORALE_START,
    MORALE_WAGE_GAIN,
    RESTLESS_DECAY,
    SERVICE_INTERVAL_KM,
    STARTING_DRIVER_STAT,
    STARTING_WAGE,
    STAT_DECAY_PER_HOUR,
    TENURE_MORALE_GAIN,
    VEHICLES,
    WAGE_FLOOR,
)

FIRST_NAMES = [
    "Alex",
    "Blair",
    "Casey",
    "Devin",
    "Emery",
    "Frankie",
    "Gray",
    "Harper",
    "Indigo",
    "Jules",
    "Kai",
    "Lane",
    "Marlow",
    "Nico",
    "Onyx",
    "Parker",
    "Quinn",
    "Reese",
    "Sasha",
    "Tatum",
    "Umber",
    "Val",
    "Wren",
    "Yael",
    "Zuri",
]
LAST_NAMES = [
    "Alvarez",
    "Brennan",
    "Cho",
    "Diallo",
    "Eriksen",
    "Fontaine",
    "Gupta",
    "Haddad",
    "Ibarra",
    "Jansen",
    "Kowalski",
    "Lindqvist",
    "Moreau",
    "Nakamura",
    "Okonkwo",
    "Petrov",
    "Rahman",
    "Silva",
    "Tanaka",
    "Ueda",
    "Vargas",
    "Weiss",
]


def make_name(rng):
    return f"{FIRST_NAMES[rng.randrange(len(FIRST_NAMES))]} {LAST_NAMES[rng.randrange(len(LAST_NAMES))]}"


def resume_rating(stats, rng):
    """1-5 from the stat mean plus noise."""
    mean = sum(stats[s] for s in DRIVER_STATS) / len(DRIVER_STATS)
    noisy = mean + rng.gauss(0, 11)
    return max(1, min(5, int(round(noisy / 20.0 + 0.5))))


def make_driver(did, rng, wage=None, quality=None):
    """`SPEED`'s deviation from base is offset in `CARE` and `SERVICE`."""
    base = quality if quality is not None else rng.uniform(*STARTING_DRIVER_STAT)
    speed_tilt = rng.uniform(-14, 14)
    stats = {
        "SPEED": _clamp(base + speed_tilt),
        "CARE": _clamp(base - speed_tilt * 0.6 + rng.gauss(0, 6)),
        "SERVICE": _clamp(base - speed_tilt * 0.5 + rng.gauss(0, 6)),
    }
    ask = wage if wage is not None else STARTING_WAGE
    return {
        "id": did,
        "name": make_name(rng),
        "stats": stats,
        "resume": resume_rating(stats, rng),
        "wage": float(ask),
        "reservation": float(ask) * rng.uniform(0.88, 1.10),
        "morale": MORALE_START,
        "restlessness": 0.0,
        "tenure": 0,
        "overtime_minutes": 0.0,
        "notice": False,
        "departs": None,  # {"to", "offer", "day"} while serving notice after a poach
        "truck": None,
    }


def _clamp(v):
    return max(1.0, min(100.0, v))


def effective_stats(driver, elapsed_minutes, overtime_minutes):
    """Stats degrade through the day, twice as fast in overtime."""
    straight = max(0.0, elapsed_minutes - overtime_minutes)
    decay = STAT_DECAY_PER_HOUR * (straight + 2.0 * overtime_minutes) / 60.0
    morale_penalty = 0.0
    if driver["morale"] < 35.0:
        morale_penalty = (35.0 - driver["morale"]) * 0.4
    return {s: _clamp(driver["stats"][s] - decay - morale_penalty) for s in DRIVER_STATS}


def speed_multiplier(stats):
    """A 50-stat driver is the reference. Range is roughly 0.88x to 1.14x."""
    return 1.14 - 0.26 * (stats["SPEED"] / 100.0)


def service_multiplier(stats):
    return 1.30 - 0.60 * (stats["SERVICE"] / 100.0)


def care_multiplier(stats):
    return 1.60 - 1.20 * (stats["CARE"] / 100.0)


def make_truck(tid, vtype, rng, ownership="OWNED", age_days=0, odometer=0.0, basis=None):
    """A truck, complete enough to publish the night it is acquired."""
    spec = VEHICLES[vtype]
    return {
        "id": tid,
        "type": vtype,
        "ownership": ownership,  # OWNED | FINANCED | RENTED
        "age_days": age_days,
        "odometer": odometer,
        "km_since_service": rng.uniform(0.0, SERVICE_INTERVAL_KM) if odometer else 0.0,
        "status": "IDLE",  # IDLE | ACTIVE | DISABLED | ORDERED
        "fuel": float(spec["tank"]),
        "principal": 0.0,
        "rental_days": 0,
        "node": None,
        "staged": None,
        "driver": None,
        # Purchase price and age; book value depreciates from these.
        "basis": float(spec["buy"] if basis is None else basis),
        "age_at_purchase": age_days,
        "book_value": float(spec["buy"] if basis is None else basis),
        "clock": 0.0,
        "worked": 0.0,
        "overtime": 0.0,
        "carrying": set(),
        "route": [],
        "plan": {},
        "anchor": None,
        "pos": (0.0, 0.0),
        "last_t": 0.0,
        "load": [],
        "lots": [],
        "revisited": {},
    }


def book_value(truck):
    spec = VEHICLES[truck["type"]]
    basis = truck.get("basis", spec["buy"])
    owned = truck["age_days"] - truck.get("age_at_purchase", 0)
    return max(spec["buy"] * 0.25, basis - spec["depreciation_day"] * owned)


def nightly_morale(driver, rng):
    """Morale drifts from wage-vs-worth and cumulative overtime."""
    surplus = driver["wage"] - driver["reservation"]
    driver["morale"] += MORALE_WAGE_GAIN * surplus
    driver["morale"] -= MORALE_OVERTIME_COST * driver["overtime_minutes"]
    driver["morale"] += TENURE_MORALE_GAIN
    driver["morale"] -= driver["restlessness"] * 0.25
    driver["morale"] = max(0.0, min(100.0, driver["morale"]))
    driver["restlessness"] *= RESTLESS_DECAY
    driver["overtime_minutes"] *= 0.5  # cumulative, but it fades
    driver["tenure"] += 1
    driver["notice"] = driver["morale"] < MORALE_NOTICE
    return driver["morale"] < MORALE_QUIT


def poach_accepts(driver, offer, rng):
    """The driver weighs the number against actual pay, morale and tenure."""
    if offer < WAGE_FLOOR:
        return False
    lift = (offer - driver["wage"]) / max(driver["wage"], 1.0)
    pull = lift * 2.4 + driver["restlessness"] / 40.0
    stickiness = driver["morale"] / 90.0 + driver["tenure"] / 60.0
    return pull - stickiness > rng.uniform(0.0, 0.6)
