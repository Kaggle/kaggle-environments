"""Constants for kargo."""

# --- Time -------------------------------------------------------------------

BLOCK_MINUTES = 120
BLOCKS_PER_DAY = 5
OVERNIGHT_PHASES = ["CAPEX", "LABOR", "CONTRACTS"]
STEPS_PER_DAY = BLOCKS_PER_DAY + len(OVERNIGHT_PHASES)
SHIFT_MINUTES = 480  # straight time, 08:00-16:00
DEADLINE_MINUTES = 480  # service deadline, 16:00
DAY_END_MINUTES = BLOCK_MINUTES * BLOCKS_PER_DAY  # 18:00
OVERTIME_WAGE_MULT = 1.5
OVERTIME_DECAY_MULT = 2.0
LOAD_MINUTES = 45
ABANDON_BLOCK = 0  # a held lot can be written off only in the first block

# --- City -------------------------------------------------------------------

GRID_SIZE = 20
ARTERIAL_EVERY = 4
COLLECTOR_SHARE = 0.35
EDGE_KM = 1.5

# Free-flow minutes to cross one 1.5 km edge.
ROAD_CLASSES = {
    #                free-flow min   capacity veh/min   km/h
    "ARTERIAL": {"t_free": 2.0, "capacity": 12.0},  # 45
    "COLLECTOR": {"t_free": 3.6, "capacity": 6.0},  # 25
    "LOCAL": {"t_free": 6.0, "capacity": 2.4},  # 15
}
ARTERIAL_KMH = EDGE_KM / (2.0 / 60.0)  # 45.0, the deadhead-pricing speed

LOCAL_SPEED_KMH = 28.0  # interior streets, no modelled congestion
GRID_DETOUR = 1.27  # BHH grid-distance correction
# Nearest-neighbour tour length over the BHH limit.
ROUTE_SLACK = 1.4
WALK_M_PER_S = 1.3
PARK_MINUTES = 0.75

# Fixed per archetype. Only position and patch size are drawn per episode.
#   c_d      structural traffic load
#   block_m  interior street grid spacing
#   doors    doors per segment (from real lot frontage / block length)
#   units    dwelling units behind one door
#   svc      minutes of service at one stop
#   pen      fraction of units receiving a parcel on a given day
#   run      stops served per park-and-walk run
#   share    fraction of a segment's doors a route actually works
#   comm     commercial share of recipients
#   pkg_units  parcel-units per package (bulk)
DISTRICTS = {
    "DOWNTOWN": {
        "c_d": 1.02,
        "block_m": 80,
        "doors": 6,
        "units": 24.0,
        "svc": 4.6,
        "pen": 0.09,
        "run": 5,
        "share": 1.00,
        "comm": 0.55,
        "pkg_units": 1.0,
        "weight": 1.0,
        "span": 420,
    },
    "RIVERSIDE": {
        "c_d": 0.88,
        "block_m": 110,
        "doors": 24,
        "units": 6.0,
        "svc": 2.4,
        "pen": 0.09,
        "run": 8,
        "share": 0.55,
        "comm": 0.15,
        "pkg_units": 1.0,
        "weight": 1.2,
        "span": 420,
    },
    "MIDTOWN": {
        "c_d": 0.78,
        "block_m": 160,
        "doors": 30,
        "units": 2.5,
        "svc": 1.9,
        "pen": 0.09,
        "run": 5,
        "share": 0.70,
        "comm": 0.40,
        "pkg_units": 1.1,
        "weight": 1.6,
        "span": 420,
    },
    "SUBURBS_N": {
        "c_d": 0.55,
        "block_m": 250,
        "doors": 28,
        "units": 1.05,
        "svc": 1.6,
        "pen": 0.09,
        "run": 3,
        "share": 0.85,
        "comm": 0.05,
        "pkg_units": 1.3,
        "weight": 1.5,
        "span": 400,
    },
    "SUBURBS_S": {
        "c_d": 0.55,
        "block_m": 250,
        "doors": 28,
        "units": 1.05,
        "svc": 1.6,
        "pen": 0.09,
        "run": 3,
        "share": 0.85,
        "comm": 0.05,
        "pkg_units": 1.3,
        "weight": 1.5,
        "span": 400,
    },
    "INDUSTRIAL": {
        "c_d": 0.42,
        "block_m": 450,
        "doors": 16,
        "units": 1.0,
        "svc": 7.5,
        "pen": 0.30,
        "run": 1,
        "share": 1.00,
        "comm": 0.90,
        "pkg_units": 1.5,
        "weight": 1.3,
        "span": 380,
    },
}
DISTRICT_NAMES = list(DISTRICTS)

# Delivery windows. dock_rate applies to commercial recipients,
# prom_rate to residential.
WINDOWS = {
    "DOWNTOWN": {"dock_rate": 0.45, "dock_width": 120, "prom_rate": 0.40},
    "RIVERSIDE": {"dock_rate": 0.45, "dock_width": 120, "prom_rate": 0.40},
    "MIDTOWN": {"dock_rate": 0.40, "dock_width": 120, "prom_rate": 0.30},
    "SUBURBS_N": {"dock_rate": 0.40, "dock_width": 120, "prom_rate": 0.20},
    "SUBURBS_S": {"dock_rate": 0.40, "dock_width": 120, "prom_rate": 0.20},
    "INDUSTRIAL": {"dock_rate": 0.20, "dock_width": 240, "prom_rate": 0.20},
}
PROMISED_WIDTH = 120
WINDOW_GRANULARITY = 30  # window starts land on the half hour

# --- Travel time ------------------------------------------------------------

BPR_ALPHA = 0.15
BPR_BETA = 4
# P(m) = 0.55 + g(10, 60, 0.60) + g(570, 100, 0.92), m in minutes since 08:00
TOD_BASE = 0.55
TOD_BUMPS = [(10.0, 60.0, 0.60), (570.0, 100.0, 0.92)]
DOW_MULT = [1.00, 1.03, 1.05, 1.07, 1.12, 0.90, 0.70]  # Mon..Sun
LDAY_SIGMA = 0.16
AR1_RHO = 0.94
AR1_SIGMA = 0.085
TRUCK_PCU_LEVELS = {"low": 1.0, "medium": 3.0, "high": 8.0}

# Cut points are spaced by travel cost, not v/c.
CONGESTION_LEVELS = ["FREE", "LIGHT", "MODERATE", "HEAVY", "SEVERE", "GRIDLOCK"]
CONGESTION_CUTS = [0.70, 1.00, 1.25, 1.50, 1.80]

WEATHER = {
    #            probability  t_free uplift  accident mult
    "CLEAR": {"p": 0.68, "travel": 1.00, "accident": 1.0},
    "RAIN": {"p": 0.24, "travel": 1.12, "accident": 1.6},
    "SNOW": {"p": 0.08, "travel": 1.25, "accident": 2.4},
}
WEATHER_FORECAST_ACCURACY = 0.75
# Half of dock doors refuse a late truck outright; the rest take one up to a
# hidden grace past the window.
DOCK_GRACE_SHARE = 0.5
DOCK_GRACE_MINUTES = (15, 60)

ACCIDENT_HAZARD_PER_EDGE_MIN = 2.0e-6  # scaled by (v/c)^2
ACCIDENT_CAPACITY = (0.30, 0.50)
ACCIDENT_DURATION = (30, 150)
ACCIDENT_REPORT_DELAY = 10
CLOSURE_DAILY_P = 0.25
CLOSURE_DURATION = (90, 300)
CONSTRUCTION_PER_DAY = (0, 3)

# --- Fleet ------------------------------------------------------------------

VEHICLES = {
    "VAN": {
        "capacity": 200,
        "buy": 38000,
        "rent_day": 170,
        "own_day": 60,
        "fuel_per_min": 1.0,
        "depreciation_day": 11,
        "service_cost": 100,
        "tank": 640,
    },
    "STEP": {
        "capacity": 340,
        "buy": 62000,
        "rent_day": 270,
        "own_day": 92,
        "fuel_per_min": 1.6,
        "depreciation_day": 18,
        "service_cost": 140,
        "tank": 1000,
    },
}
VEHICLE_TYPES = list(VEHICLES)

FINANCE_DOWN = 0.20
CASH_FLOOR = -20000.0  # BUY, FINANCE and BUY_USED may not take cash below this
FINANCE_DAILY_RATE = 0.00035  # ~13% APR on outstanding principal
RENTAL_LEAD_DAYS = (2, 3)
RENTAL_MIN_DAYS = 7
RENTAL_POOL_BASE = 6
USED_DISCOUNT = (0.25, 0.45)
USED_LISTINGS_PER_NIGHT = 2
SELL_HAIRCUT = (0.25, 0.45)
CREDIT_LINE_DAILY_RATE = 0.0025

FUEL_PRICE_BULK = 0.42  # $/unit at a warehouse
FUEL_PRICE_STATION_MULT = (1.15, 1.40)
FUEL_CALL_COST = 150
FUEL_CALL_MINUTES = 60

# Maintenance. Wear accrues as km x the driver's CARE multiplier; a truck past
# the interval at 18:00 is grounded until serviced.
SERVICE_INTERVAL_KM = 250

# --- Drivers ----------------------------------------------------------------

DRIVER_STATS = ["SPEED", "CARE", "SERVICE"]
STARTING_DRIVER_STAT = (42, 62)
STARTING_WAGE = 240
WAGE_FLOOR = 120
CANDIDATES_PER_NIGHT = (2, 4)
MORALE_START = 60.0
MORALE_WAGE_GAIN = 0.06  # per $ of daily surplus over reservation
MORALE_OVERTIME_COST = 0.02  # per cumulative overtime minute
MORALE_NOTICE = 22.0
MORALE_QUIT = 8.0
MORALE_DEGRADE_BELOW = 35.0
RESTLESS_ON_REFUSED_OFFER = 14.0
RESTLESS_DECAY = 0.85
TENURE_MORALE_GAIN = 0.4
STAT_DECAY_PER_HOUR = 1.6  # effective-stat decay through the day
# A driver on the roster but not on a truck is paid this share of their wage
# each night. Assigned drivers are paid for the minutes their truck works.
RETAINER_SHARE = 0.5
MAX_DRIVERS = 30  # per player

# --- Contracts --------------------------------------------------------------

WAREHOUSES = 4
# Lot size as a fraction of a truck-day.
LOT_FRACTION = (0.15, 0.70)
# Effective shift lost per unit of windowed share.
WINDOW_DRAG = 0.33
FILL_CEILING = 1.00

TARGET_NET_PER_TRUCK_DAY = 220.0
FIXED_COST_DAY = 60.0
OVERHEAD_COST_DAY = 55.0
WAGE_PER_MINUTE = 31.0 / 60.0

FAIL_PENALTY = 45.0
LATE_PENALTY = 6.0
PROMISED_PREMIUM = 4.0
DOCK_REFUSAL = 45.0
# Per parcel-unit written off in block 0.
ABANDON_FEE_PER_UNIT = 15.0

STANDING_TERMS = [5, 10, 20]
STANDING_DISCOUNT = (0.08, 0.15)
STANDING_BREAK_FEE_LOT_DAYS = 3
STANDING_SHARE = 1.0 / 3.0

# --- Starting position ------------------------------------------------------

STARTING_CASH = 12000.0
STARTING_TRUCKS = 3
STARTING_VEHICLE = "VAN"
STARTING_FLEET = ("VAN", "VAN", "STEP")
MAX_FLEET = 20  # trucks per player, ordered rentals included
SIGHTINGS_KEPT = 50
ROUTE_SHOWN = 50  # route entries republished per truck per step
TRAIL_SHOWN = 24  # position samples per truck per block
DAYS = 60
EPISODE_STEPS = DAYS * STEPS_PER_DAY + 1

# --- The shipper (shipper.py) -----------------------------------------------
#
# Every range below is drawn once per episode and never published.

# Fresh demand, in truck-days per starting truck in the field, follows a
# logistic curve from START to CEILING. MIDPOINT is a fraction of the episode;
# STEEPNESS is per day.
DEMAND_START = (0.20, 0.30)
DEMAND_CEILING = (0.55, 0.85)
DEMAND_MIDPOINT = (0.30, 0.65)
DEMAND_STEEPNESS = (0.10, 0.30)
# Mon..Sun, jittered +-10% per episode and renormalised to a mean of 1.
DEMAND_WEEK = [1.05, 1.05, 1.00, 1.00, 1.10, 0.75, 0.55]
DEMAND_AR1 = (0.70, 0.07)  # rho, sigma of the daily log-noise
# A sticky hidden regime. Each night it stays with probability REGIME_STAY,
# otherwise it moves one step up or down.
REGIMES = {"SLUMP": 0.80, "NORMAL": 1.00, "BOOM": 1.20}
REGIME_STAY = (0.90, 0.96)
# Where demand lands. Each (warehouse, district) pair has a log-weight that
# random-walks around a drawn home level.
PAIR_WEIGHT_SPREAD = 0.8
PAIR_WEIGHT_DRIFT = 0.05  # daily sigma
PAIR_WEIGHT_REVERT = 0.03  # daily pull toward home

# Price. Each pair's reserve is its solved base price times a hidden index,
# the product of a market level and a territory deviation.
#
# The market level chases ELASTICITY x log(tonight's demand / what the field
# can move), at SPEED per night. Demand is tonight's fresh freight plus held
# standing accounts. "What the field can move" is the public roster -- trucks
# with a driver and not grounded -- times a hidden THROUGHPUT.
PRICE_THROUGHPUT = (0.30, 0.42)
PRICE_ELASTICITY = (0.50, 0.90)
PRICE_SPEED = (0.15, 0.35)
# A territory deviates on its own outcomes: up on the share of its packages
# that went unserved, down when carriers queue under its reserve while the
# market has slack. Pulled back toward the market each night.
INDEX_UP = (0.08, 0.16)
INDEX_DOWN = (0.15, 0.35)
INDEX_SPILLOVER = (0.10, 0.30)
INDEX_IDLE_REVERT = 0.05
PAIR_DEVIATION = 0.5  # log-space cap either way
INDEX_NOISE = 0.02
INDEX_RANGE = (0.5, 3.0)
LISTING_NOISE = 0.04  # per-lot jitter on the reserve

# Packages the shipper could not get moved come back the next night, marked up
# per retry, until a per-lot patience runs out. Then they are lost and the pair
# loses demand weight.
RETRY_MARKUP = (0.06, 0.18)
PATIENCE_MAX = (2, 5)
REPUTATION_HIT = (0.04, 0.10)
MAX_LISTINGS = 160
