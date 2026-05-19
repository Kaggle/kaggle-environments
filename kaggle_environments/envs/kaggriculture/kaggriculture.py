import json
import math
import random
from os import path

dirpath = path.dirname(__file__)


CROPS = {
    "WHEAT":      {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 600, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]

# Pricing model:
#     price(inv) = base + sign * amp * f(|inv - I0|)
#     sign = +1 below I0 (scarcity), -1 above I0 (glut)
#     amp  = target * base / f(T)            (derived; selling T units moves
#                                             price by `target` * base)
#     T    = production capacity of one 5x5 field over a 24-day game at
#            optimal watering, no fertilizer (animal T pre-discounted 30% for
#            wheat-feed overhead)
#     f    in {linear, sq, sqrt, log, log10}; log uses ln(1+x) so f(0)=0
# Floored at PRICE_FLOOR.
MARKET_I0 = 10000
PRICE_FLOOR = 1

MARKET_PARAMS = {
    "WHEAT":      {"base":  25, "I0": MARKET_I0, "T": 400, "below_func": "sqrt",   "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base":  35, "I0": MARKET_I0, "T": 450, "below_func": "log",    "below_target": 0.20, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base":  60, "I0": MARKET_I0, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": MARKET_I0, "T": 100, "below_func": "sqrt",   "below_target": 0.70, "above_func": "linear", "above_target": 0.40},
    "MELON":      {"base": 250, "I0": MARKET_I0, "T": 300, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 0.90},
    "EGG":        {"base":  50, "I0": MARKET_I0, "T": 332, "below_func": "linear", "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "I0": MARKET_I0, "T": 122, "below_func": "sqrt",   "below_target": 0.60, "above_func": "linear", "above_target": 0.40},
    "WOOL":       {"base": 200, "I0": MARKET_I0, "T": 105, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 0.80},
    "FERTILIZER": {"base": 100, "I0": MARKET_I0, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}


def _shape(func, x):
    x = max(0.0, x)
    if func == "linear": return x
    if func == "sq":     return x * x
    if func == "sqrt":   return math.sqrt(x)
    if func == "log":    return math.log(1.0 + x)
    if func == "log10":  return math.log10(1.0 + x)
    return x


def _resolve_market_params(overrides):
    """Merge per-resource overrides onto MARKET_PARAMS defaults (sparse)."""
    resolved = {item: dict(p) for item, p in MARKET_PARAMS.items()}
    if not overrides:
        return resolved
    for item, patch in overrides.items():
        if item in resolved and isinstance(patch, dict):
            resolved[item].update(patch)
    return resolved

# (dx, dy); y grows downward.
FARMER_MOVES = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "EAST":  (1, 0),
    "WEST":  (-1, 0),
}

# NW is always unlocked; players unlock the rest in this order.
LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]

# n-th hire of the day -> cost = FARM_HAND_COST_MULT * fib(n), where
# fib starts 1, 1, 2, 3, 5, 8, 13, ... Configurable via `farmHandCostMult`.
FARM_HAND_COST_MULT = 10

SHOPS = {
    "BAKERY":         ["EGG", "WHEAT"],
    "PIZZA_SHOP":     ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT":    ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE":     ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE":       ["CARROT"],
    "SMOOTHIE_SHOP":  ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}

TOWN_CENTER_PRODUCTS = [p for p in PRODUCTS if p != "FERTILIZER"]

# Town center demand schedule: (day_threshold, multiplier), highest threshold first.
TOWN_CENTER_DEMAND_SCHEDULE = [(20, 4), (10, 2), (0, 1)]


def get(d, key, default):
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def _quadrant_of(x, y, board_size):
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def _shed_access_tiles(board_size):
    """Four inner-corner tiles around the shed, in NWSE order."""
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def _is_shed_adjacent(pos, board_size):
    return tuple(pos) in {(x, y) for (x, y) in _shed_access_tiles(board_size)}


def _new_farm(board_size, starting_money):
    return {
        "money": float(starting_money),
        # tiles[y][x] = None (empty unlocked) | "LOCKED" | dict structure
        "tiles": [
            [_initial_tile(x, y, board_size) for x in range(board_size)]
            for y in range(board_size)
        ],
        "farmer": list(_default_spawn(board_size)),
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }


def _initial_tile(x, y, board_size):
    return None if _quadrant_of(x, y, board_size) == "NW" else "LOCKED"


def _default_spawn(board_size):
    """First free shed-access tile, NWSE preference."""
    for tile in _shed_access_tiles(board_size):
        if _quadrant_of(tile[0], tile[1], board_size) == "NW":
            return tile
    return (0, 0)


def _new_private():
    return {
        "shed": {item: 0 for item in PRODUCTS + list(ANIMALS)},
        "seeds": {crop: 0 for crop in CROPS},
        # inventories[0] = main farmer; hands appended/removed each day.
        "inventories": [{}],
    }


def _new_market(params=None):
    params = params or MARKET_PARAMS
    inv = {item: params[item]["I0"] for item in PRODUCTS}
    prices = {item: params[item]["base"] for item in PRODUCTS}
    market = {"inventory": inv, "prices": prices}
    if params is not MARKET_PARAMS:
        market["params"] = params
    return market


def _new_town():
    return {"unlocked_shops": []}


def market_price(item, inventory, params=None):
    """Floor at PRICE_FLOOR."""
    p = (params or MARKET_PARAMS)[item]
    base = p["base"]
    I0 = p["I0"]
    T = p["T"]
    if inventory < I0:
        f = p["below_func"]
        amp = p["below_target"] * base / _shape(f, T)
        price = base + amp * _shape(f, I0 - inventory)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / _shape(f, T)
        price = base - amp * _shape(f, inventory - I0)
    return max(PRICE_FLOOR, int(round(price)))


def _refresh_prices(market):
    params = market.get("params")
    for item in PRODUCTS:
        market["prices"][item] = market_price(item, market["inventory"][item], params)


def _new_plant(crop, day, turns_per_day):
    cd = CROPS[crop]
    return {
        "kind": "PLANT",
        "crop": crop,
        "planted_day": day,
        "watered_today": False,
        "consecutive_unwatered": 1,  # planting day counts as unwatered
        "yield_units": 0 if cd["ongoing"] else 1,
        "max_lifespan_step": (-1 if cd["ongoing"] else (day + cd["max_yield_day"] + 1) * turns_per_day),
        "fertilized_until_day": -1,
    }


def _new_animal(animal, day):
    a = ANIMALS[animal]
    return {
        "kind": a["structure"],
        "animal": animal,
        "placed_day": day,
        "yield_units": 0,
        "consecutive_unfed": 0,
        "fed_today": False,
        "cared_today": False,
        "fertilizer_available": False,
        "pending_care_bonus": 0,
    }


def _initialize(state, env):
    configuration = env.configuration
    num_agents = len(state)
    obs0 = state[0].observation

    if not hasattr(env, "info") or env.info is None:
        env.info = {}

    seed = env.info.get("seed")
    if seed is None:
        seed = get(configuration, "seed", None)
    if seed is None:
        seed = random.randrange(2**31)
    try:
        configuration.seed = None
    except (AttributeError, TypeError):
        configuration["seed"] = None
    env.info["seed"] = seed

    board_size = int(get(configuration, "boardSize", 10))
    starting_money = int(get(configuration, "startingMoney", 150))

    farms = [_new_farm(board_size, starting_money) for _ in range(num_agents)]
    privates = [_new_private() for _ in range(num_agents)]
    market_overrides = get(configuration, "marketParams", None)
    resolved_params = _resolve_market_params(market_overrides) if market_overrides else None
    market = _new_market(resolved_params)
    town = _new_town()

    obs0.farms = farms
    obs0.market = market
    obs0.town = town
    obs0.day = 0
    obs0.hour = 0

    for i in range(num_agents):
        state[i].observation.player = i
        state[i].observation.private = privates[i]
        if i > 0:
            state[i].observation.farms = farms
            state[i].observation.market = market
            state[i].observation.town = town
            state[i].observation.day = 0
            state[i].observation.hour = 0


def _farmer_position(farm, idx):
    """idx 0 = main farmer, 1+ = hand index."""
    if idx == 0:
        return farm["farmer"]
    return farm["hands"][idx - 1] if idx - 1 < len(farm["hands"]) else None


def _set_farmer_position(farm, idx, pos):
    if idx == 0:
        farm["farmer"] = list(pos)
    else:
        farm["hands"][idx - 1] = list(pos)


def _farmer_inventory(private, idx):
    """Inventories list is [main_farmer, *hands]; grow it if idx is past the end."""
    while len(private["inventories"]) <= idx:
        private["inventories"].append({})
    return private["inventories"][idx]


def _inv_add(inv, item, n=1):
    inv[item] = inv.get(item, 0) + n


def _inv_take(inv, item, n=1):
    if inv.get(item, 0) < n:
        return False
    inv[item] -= n
    if inv[item] == 0:
        del inv[item]
    return True


def _apply_unit_action(farm, private, idx, action, board_size, day, turns_per_day, shed_capacity=100):
    """Process one farmer/hand's action. Invalid / illegal actions are silent no-ops."""
    if not isinstance(action, list) or not action:
        return
    op = action[0]
    pos = _farmer_position(farm, idx)
    if pos is None:
        return
    fx, fy = pos[0], pos[1]
    inv = _farmer_inventory(private, idx)

    if op in FARMER_MOVES:
        dx, dy = FARMER_MOVES[op]
        nx, ny = fx + dx, fy + dy
        if not (0 <= nx < board_size and 0 <= ny < board_size):
            return
        if farm["tiles"][ny][nx] == "LOCKED":
            return
        _set_farmer_position(farm, idx, (nx, ny))
        return

    if op == "PASS":
        return

    tile = farm["tiles"][fy][fx]
    if tile == "LOCKED":
        return

    if op == "PICKUP":
        if not _is_shed_adjacent((fx, fy), board_size):
            return
        if len(action) < 2:
            return
        item = action[1]
        n = int(action[2]) if len(action) >= 3 else 1
        if n <= 0:
            return
        # Seeds live in private["seeds"] and are consumed directly by PLANT;
        # they never pass through farmer inventory or the shed.
        available = private["shed"].get(item, 0)
        n = min(n, available)
        if n <= 0:
            return
        private["shed"][item] -= n
        _inv_add(inv, item, n)
        return

    if op == "PLANT":
        if len(action) < 2:
            return
        crop = action[1]
        if crop not in CROPS:
            return
        if tile is not None:
            return
        if private["seeds"].get(crop, 0) <= 0:
            return
        private["seeds"][crop] -= 1
        farm["tiles"][fy][fx] = _new_plant(crop, day, turns_per_day)
        return

    if op == "WATER":
        if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
            return
        if tile["watered_today"]:
            return
        tile["watered_today"] = True
        crop_data = CROPS[tile["crop"]]
        if not crop_data["ongoing"]:
            age_days = day - tile["planted_day"]
            window_start = (crop_data["max_yield_day"] + 1) // 2
            if window_start <= age_days <= crop_data["max_yield_day"]:
                bonus = 2 if tile["fertilized_until_day"] >= day else 1
                tile["yield_units"] = min(crop_data["max_yield"], tile["yield_units"] + bonus)
        return

    if op == "HARVEST":
        if not isinstance(tile, dict):
            return
        if tile.get("yield_units", 0) <= 0:
            return
        if tile.get("kind") == "PLANT":
            crop_data = CROPS[tile["crop"]]
            if day - tile["planted_day"] < crop_data["first_yield_day"]:
                # Ongoing crops only accumulate yield_units after first_yield_day,
                # so reaching here with yield_units > 0 indicates a bug.
                if crop_data["ongoing"]:
                    print(
                        f"WARNING: HARVEST on immature ongoing {tile['crop']} "
                        f"(planted day {tile['planted_day']}, current day {day}, "
                        f"first_yield_day {crop_data['first_yield_day']}, "
                        f"yield_units {tile['yield_units']}); should never happen"
                    )
                return
            units = tile["yield_units"]
            tile["yield_units"] = 0
            _inv_add(inv, tile["crop"], units)
            if not crop_data["ongoing"]:
                farm["tiles"][fy][fx] = None
        elif "animal" in tile:
            units = tile["yield_units"]
            tile["yield_units"] = 0
            _inv_add(inv, ANIMALS[tile["animal"]]["product"], units)
        return

    if op == "FERTILIZE":
        if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
            return
        if not _inv_take(inv, "FERTILIZER", 1):
            return
        # Active for `day`, `day+1`, `day+2` (3 days inclusive).
        tile["fertilized_until_day"] = max(tile.get("fertilized_until_day", -1), day + 2)
        return

    if op == "DIG":
        if tile is None:
            return
        # Removes plants, weeds, empty coop/pasture. Does NOT remove a placed animal.
        if isinstance(tile, dict) and "animal" in tile:
            return
        farm["tiles"][fy][fx] = None
        return

    if op == "BUILD_COOP":
        if tile is not None:
            return
        farm["tiles"][fy][fx] = {"kind": "COOP"}
        return

    if op == "BUILD_PASTURE":
        if tile is not None:
            return
        farm["tiles"][fy][fx] = {"kind": "PASTURE"}
        return

    if op == "PLACE":
        if len(action) < 2:
            return
        item = action[1]
        # Animal placement: standing on a matching unoccupied structure.
        if (
            item in ANIMALS
            and isinstance(tile, dict)
            and tile.get("kind") == ANIMALS[item]["structure"]
            and "animal" not in tile
        ):
            if _inv_take(inv, item, 1):
                farm["tiles"][fy][fx] = _new_animal(item, day)
            return
        # Shed drop: orthogonally adjacent to the shed; obeys shedCapacity.
        if _is_shed_adjacent((fx, fy), board_size):
            n = int(action[2]) if len(action) >= 3 else 1
            if n <= 0:
                return
            n = min(n, inv.get(item, 0))
            if n <= 0:
                return
            current = sum(private["shed"].values())
            room = max(0, shed_capacity - current)
            n = min(n, room)
            if n <= 0:
                return
            inv[item] -= n
            if inv[item] == 0:
                del inv[item]
            private["shed"][item] = private["shed"].get(item, 0) + n
        return

    if op == "FEED":
        if not (isinstance(tile, dict) and "animal" in tile):
            return
        if tile["fed_today"]:
            return
        if not _inv_take(inv, "WHEAT", 1):
            return
        tile["fed_today"] = True
        return

    if op == "COLLECT_FERTILIZER":
        if not (isinstance(tile, dict) and "animal" in tile):
            return
        if not tile["fertilizer_available"]:
            return
        tile["fertilizer_available"] = False
        _inv_add(inv, "FERTILIZER", 1)
        return

    if op == "CARE":
        if not (isinstance(tile, dict) and "animal" in tile):
            return
        if tile["cared_today"]:
            return
        tile["cared_today"] = True
        return


def _spawn_hand(farm, board_size):
    """First free shed-access tile (NWSE order); ties broken by min occupancy."""
    occupants = {tile: 0 for tile in _shed_access_tiles(board_size)}
    all_pos = [tuple(farm["farmer"])] + [tuple(p) for p in farm["hands"]]
    for pos in all_pos:
        if pos in occupants:
            occupants[pos] += 1
    best = sorted(occupants.items(), key=lambda kv: (kv[1], _shed_access_tiles(board_size).index(kv[0])))
    return list(best[0][0])


def _process_market(state, env):
    """Per-unit lockstep: at each step, quote both players' current-unit prices, then commit both."""
    obs0 = state[0].observation
    market = obs0.market
    farms = obs0.farms
    privates = [s.observation.private for s in state]
    board_size = int(get(env.configuration, "boardSize", 10))
    max_orders = max(1, int(get(env.configuration, "maxMarketOrdersPerTurn", 10)))
    hire_mult = int(get(env.configuration, "farmHandCostMult", FARM_HAND_COST_MULT))

    queues = []
    for s in state:
        action = s.action if isinstance(s.action, dict) else {}
        m = action.get("market", []) if isinstance(action, dict) else []
        q = list(m) if isinstance(m, list) else []
        queues.append(q[:max_orders])

    max_len = max((len(q) for q in queues), default=0)
    for i in range(max_len):
        order_states = []
        for player_id, q in enumerate(queues):
            ostate = None
            if i < len(q):
                ostate = _parse_order(q[i])
            order_states.append(ostate)

        # Atomic orders (HIRE, BUY_LAND): handle once, in player order.
        for player_id, ostate in enumerate(order_states):
            if ostate is None:
                continue
            op = ostate["type"]
            if op == "HIRE":
                _do_hire(farms[player_id], privates[player_id], board_size, hire_mult)
                order_states[player_id] = None
            elif op == "BUY_LAND":
                _do_buy_land(farms[player_id], board_size)
                order_states[player_id] = None

        # Per-unit lockstep loop for SELL / BUY_*.
        idx_esc = 0
        while True:
            idx_esc += 1
            if idx_esc >= 100_000:
                print("WARNING: kaggriculture market loop exceeded 100k iterations; aborting")
                break
            quoted = [None, None]
            for player_id, ostate in enumerate(order_states):
                if ostate is None or ostate["remaining"] <= 0:
                    continue
                op = ostate["type"]
                item = ostate["item"]
                if op == "SELL" and item in PRODUCTS and item != "FERTILIZER":
                    quoted[player_id] = ("SELL", item, market_price(item, market["inventory"][item], market.get("params")), ostate)
                elif op == "BUY_PRODUCT" and item in PRODUCTS:
                    quoted[player_id] = ("BUY_PRODUCT", item, market_price(item, market["inventory"][item], market.get("params")), ostate)
                elif op == "BUY_SEED" and item in CROPS:
                    quoted[player_id] = ("BUY_SEED", item, CROPS[item]["seed"], ostate)
                elif op == "BUY_ANIMAL" and item in ANIMALS:
                    quoted[player_id] = ("BUY_ANIMAL", item, ANIMALS[item]["cost"], ostate)
                else:
                    order_states[player_id] = None  # malformed sub-op; abort

            if all(q is None for q in quoted):
                break

            # Both players see the same pre-commit inventory for this unit.
            committed_any = False
            for player_id, q in enumerate(quoted):
                if q is None:
                    continue
                op, item, price, ostate = q
                ok = _commit_unit(op, item, price, farms[player_id], privates[player_id], market)
                if ok:
                    ostate["remaining"] -= 1
                    committed_any = True
                else:
                    order_states[player_id] = None  # can't continue this order

            if not committed_any:
                break

        _refresh_prices(market)


def _parse_order(order):
    if not isinstance(order, list) or not order:
        return None
    op = order[0]
    if op == "HIRE":
        return {"type": "HIRE"}
    if op == "BUY_LAND":
        return {"type": "BUY_LAND"}
    if op in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"):
        if len(order) < 3:
            return None
        try:
            n = int(order[2])
        except (TypeError, ValueError):
            return None
        if n <= 0:
            return None
        return {"type": op, "item": order[1], "remaining": n}
    return None


def _commit_unit(op, item, price, farm, private, market):
    if op == "SELL":
        if private["shed"].get(item, 0) <= 0:
            return False
        private["shed"][item] -= 1
        farm["money"] += price
        # Sales at $1 do not increase market supply.
        if price > 1:
            market["inventory"][item] += 1
        return True
    if op == "BUY_PRODUCT":
        if farm["money"] < price:
            return False
        farm["money"] -= price
        private["shed"][item] = private["shed"].get(item, 0) + 1
        market["inventory"][item] -= 1
        return True
    if op == "BUY_SEED":
        if farm["money"] < price:
            return False
        farm["money"] -= price
        private["seeds"][item] = private["seeds"].get(item, 0) + 1
        return True
    if op == "BUY_ANIMAL":
        if farm["money"] < price:
            return False
        farm["money"] -= price
        private["shed"][item] = private["shed"].get(item, 0) + 1
        return True
    return False


def _fib(n):
    """Indexed so _fib(0)=1, _fib(1)=1, _fib(2)=2, _fib(3)=3, _fib(4)=5..."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _hire_cost(n_already_today, mult=FARM_HAND_COST_MULT):
    return mult * _fib(n_already_today)


def _do_hire(farm, private, board_size, mult=FARM_HAND_COST_MULT):
    cost = _hire_cost(farm["hires_today"], mult)
    if farm["money"] < cost:
        return
    farm["money"] -= cost
    farm["hires_today"] += 1
    farm["hands"].append(_spawn_hand(farm, board_size))
    private["inventories"].append({})


def _do_buy_land(farm, board_size):
    n_unlocked_extra = len(farm["unlocked_quadrants"]) - 1  # NW is always there
    if n_unlocked_extra >= len(LAND_ORDER):
        return
    cost = LAND_PRICES[n_unlocked_extra]
    if farm["money"] < cost:
        return
    farm["money"] -= cost
    quadrant = LAND_ORDER[n_unlocked_extra]
    farm["unlocked_quadrants"].append(quadrant)
    for y in range(board_size):
        for x in range(board_size):
            if _quadrant_of(x, y, board_size) == quadrant and farm["tiles"][y][x] == "LOCKED":
                farm["tiles"][y][x] = None


def _town_consume(env, state, step):
    obs0 = state[0].observation
    market = obs0.market
    town = obs0.town
    cfg = env.configuration
    shop_interval = max(1, int(get(cfg, "townShopSellInterval", 2)))
    center_interval = max(1, int(get(cfg, "townCenterSellInterval", 6)))
    turns_per_day = max(1, int(get(cfg, "turnsPerDay", 24)))
    day = step // turns_per_day

    if step % shop_interval == 0:
        for shop_name in town.get("unlocked_shops", []):
            products = SHOPS[shop_name]
            multiplier = 2 if len(products) == 1 else 1
            for item in products:
                market["inventory"][item] -= multiplier

    if step % center_interval == 0:
        center_mult = next(m for threshold, m in TOWN_CENTER_DEMAND_SCHEDULE if day >= threshold)
        for item in TOWN_CENTER_PRODUCTS:
            market["inventory"][item] -= center_mult

    _refresh_prices(market)


def _decay_plants(farm, step):
    board_size = len(farm["tiles"])
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
                continue
            mls = tile["max_lifespan_step"]
            if mls < 0 or step < mls:
                continue
            if (step - mls) % 2 != 0:
                continue
            tile["yield_units"] -= 1
            if tile["yield_units"] <= 0:
                farm["tiles"][y][x] = {"kind": "WEED"}


def _daily_refresh_plants(farm, current_day, turns_per_day):
    board_size = len(farm["tiles"])
    next_day = current_day + 1
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
                continue
            was_watered = tile["watered_today"]
            if was_watered:
                tile["consecutive_unwatered"] = 0
            else:
                tile["consecutive_unwatered"] += 1
            tile["watered_today"] = False
            if tile["consecutive_unwatered"] >= 2:
                farm["tiles"][y][x] = {"kind": "WEED"}
                continue
            cd = CROPS[tile["crop"]]
            if not cd["ongoing"]:
                continue
            days_since_first = next_day - tile["planted_day"] - cd["first_yield_day"]
            if days_since_first < 0:
                continue
            interval = cd["interval"]
            if days_since_first % interval != 0:
                continue
            production_count = days_since_first // interval + 1
            if production_count > cd["max_yield"]:
                continue
            # Fertilizer bonus only applies on watered days (basic needs first).
            fertilized = was_watered and tile.get("fertilized_until_day", -1) >= current_day
            tile["yield_units"] = min(cd["max_yield"], tile["yield_units"] + (2 if fertilized else 1))
            if production_count == cd["max_yield"]:
                tile["max_lifespan_step"] = (next_day + 1) * turns_per_day


def _daily_refresh_animals(farm, day):
    board_size = len(farm["tiles"])
    next_day = day + 1
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            if not (isinstance(tile, dict) and "animal" in tile):
                continue
            if tile["fed_today"]:
                tile["consecutive_unfed"] = 0
            else:
                tile["consecutive_unfed"] += 1
            if tile["consecutive_unfed"] >= 2:
                # Animal escapes; structure remains.
                farm["tiles"][y][x] = {"kind": ANIMALS[tile["animal"]]["structure"]}
                continue
            a = ANIMALS[tile["animal"]]
            days_since_first = next_day - tile["placed_day"] - a["first_yield_day"]
            if days_since_first >= 0 and days_since_first % a["interval"] == 0:
                base = 1
                # Care bonus only consumed on a fed production day.
                bonus = tile.pop("pending_care_bonus", 0) if tile["fed_today"] else 0
                tile["yield_units"] = min(a["max_held"], tile["yield_units"] + base + bonus)
                tile["pending_care_bonus"] = 0
            if tile["cared_today"] and tile["fed_today"]:
                tile["pending_care_bonus"] = tile.get("pending_care_bonus", 0) + 1
            tile["fertilizer_available"] = True
            tile["fed_today"] = False
            tile["cared_today"] = False


def _spawn_weeds(farm, board_size, weed_chance, rng):
    for y in range(board_size):
        for x in range(board_size):
            if farm["tiles"][y][x] is None and rng.random() < weed_chance:
                farm["tiles"][y][x] = {"kind": "WEED"}


def _drop_inventories_to_shed(private, capacity):
    """Drop every per-farmer inventory into the shed up to `capacity`; overflow is discarded.
    Seeds are tracked separately in private["seeds"] and don't pass through the shed."""
    shed = private["shed"]
    for inv in private["inventories"]:
        for item, n in list(inv.items()):
            if n <= 0:
                del inv[item]
                continue
            current = sum(v for k, v in shed.items())
            room = max(0, capacity - current)
            take = min(n, room)
            if take > 0:
                shed[item] = shed.get(item, 0) + take
            del inv[item]


def _end_of_day(state, env, day):
    obs0 = state[0].observation
    cfg = env.configuration
    board_size = int(get(cfg, "boardSize", 10))
    turns_per_day = max(1, int(get(cfg, "turnsPerDay", 24)))
    weed_chance = float(get(cfg, "weedSpawnChance", 0.005))
    shed_cap = int(get(cfg, "shedCapacity", 100))
    shop_interval = max(1, int(get(cfg, "townShopUnlockInterval", 3)))

    # Stable RNG keyed off env.info["seed"] + day so replays reproduce.
    seed = env.info.get("seed", 0)
    rng = random.Random((seed * 1_000_003) ^ day)

    for player_id, farm in enumerate(obs0.farms):
        private = state[player_id].observation.private
        _daily_refresh_plants(farm, day, turns_per_day)
        _daily_refresh_animals(farm, day)
        _spawn_weeds(farm, board_size, weed_chance, rng)
        _drop_inventories_to_shed(private, shed_cap)
        farm["farmer"] = list(_default_spawn(board_size))
        farm["hands"] = []
        farm["hires_today"] = 0
        private["inventories"] = [{}]

    next_day = day + 1
    town = obs0.town
    if next_day > 0 and next_day % shop_interval == 0:
        remaining = [s for s in SHOPS if s not in town["unlocked_shops"]]
        if remaining:
            choice = rng.choice(sorted(remaining))
            town["unlocked_shops"].append(choice)


def interpreter(state, env):
    num_agents = len(state)
    obs0 = state[0].observation

    if not hasattr(obs0, "farms") or not obs0.farms:
        _initialize(state, env)
        return state

    if env.done:
        return state

    cfg = env.configuration
    turns_per_day = max(1, int(get(cfg, "turnsPerDay", 24)))
    board_size = int(get(cfg, "boardSize", 10))
    shed_capacity = int(get(cfg, "shedCapacity", 100))

    step = get(obs0, "step", 0)
    day = step // turns_per_day

    for i, s in enumerate(state):
        action = s.action if isinstance(s.action, dict) else {}
        farmer_action = action.get("farmer", ["PASS"]) if isinstance(action, dict) else ["PASS"]
        hands_actions = action.get("hands", []) if isinstance(action, dict) else []
        if not isinstance(hands_actions, list):
            hands_actions = []

        # Atomic PLANT validation: if total PLANT requests for a crop this turn
        # exceed available seeds, drop ALL PLANT requests for that crop.
        unit_actions = [farmer_action, *hands_actions]
        plant_demand = {}
        for a in unit_actions:
            if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT":
                plant_demand[a[1]] = plant_demand.get(a[1], 0) + 1
        seeds = s.observation.private.get("seeds", {}) if hasattr(s.observation.private, "get") else {}
        blocked = {crop for crop, n in plant_demand.items() if n > seeds.get(crop, 0)}

        def _allowed(a):
            if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT" and a[1] in blocked:
                return ["PASS"]
            return a

        _apply_unit_action(obs0.farms[i], s.observation.private, 0, _allowed(farmer_action),
                           board_size, day, turns_per_day, shed_capacity)
        for h_idx, hand_action in enumerate(hands_actions):
            _apply_unit_action(obs0.farms[i], s.observation.private, h_idx + 1,
                               _allowed(hand_action), board_size, day, turns_per_day, shed_capacity)

    _process_market(state, env)
    _town_consume(env, state, step)
    for farm in obs0.farms:
        _decay_plants(farm, step)
    if (step + 1) % turns_per_day == 0:
        _end_of_day(state, env, day)

    next_step = step + 1
    obs0.day = next_step // turns_per_day
    obs0.hour = next_step % turns_per_day
    for i in range(1, num_agents):
        state[i].observation.farms = obs0.farms
        state[i].observation.market = obs0.market
        state[i].observation.town = obs0.town
        state[i].observation.day = obs0.day
        state[i].observation.hour = obs0.hour

    # `step` here is the previous step counter; framework records the post-interpreter
    # state at the next index. -2 fires DONE on the final recorded step.
    if step >= cfg.episodeSteps - 2:
        for s in state:
            s.status = "DONE"
            s.reward = float(obs0.farms[s.observation.player]["money"])

    return state


def _render_tile(tile):
    if tile is None:
        return "."
    if tile == "LOCKED":
        return "#"
    if isinstance(tile, dict):
        kind = tile.get("kind")
        if kind == "WEED":
            return "x"
        if kind == "PLANT":
            return tile["crop"][0].lower()
        if "animal" in tile:
            return tile["animal"][0]
        if kind == "COOP":
            return "C"
        if kind == "PASTURE":
            return "P"
    return "?"


def renderer(state, env):
    obs = state[0].observation
    out = f"Step {get(obs, 'step', 0)}  Day {get(obs, 'day', 0)}  Hour {get(obs, 'hour', 0)}\n"
    market = get(obs, "market", {}) or {}
    town = get(obs, "town", {}) or {}
    out += f"Town shops: {town.get('unlocked_shops', [])}\n"
    out += "Prices: " + ", ".join(f"{k}=${v}" for k, v in (market.get("prices", {}) or {}).items()) + "\n"
    for i, s in enumerate(state):
        farm = obs.farms[i] if i < len(obs.farms) else None
        if farm is None:
            continue
        priv = get(s.observation, "private", {}) or {}
        out += (
            f"Player {i}: ${farm['money']:.0f}  farmer={farm['farmer']}  "
            f"hands={len(farm['hands'])}  unlocked={farm['unlocked_quadrants']}  "
            f"shed={priv.get('shed')}  seeds={priv.get('seeds')}\n"
        )
        for row in farm["tiles"]:
            out += "  " + " ".join(_render_tile(t) for t in row) + "\n"
    return out


json_path = path.abspath(path.join(dirpath, "kaggriculture.json"))
with open(json_path) as json_file:
    specification = json.load(json_file)


def html_renderer(env, mode):
    jspath = path.join(dirpath, "visualizer", "default", "dist", "index.html")
    if path.exists(jspath):
        with open(jspath, encoding="utf-8") as f:
            return f.read()
    return ""


def pass_agent(obs):
    return {"farmer": ["PASS"], "hands": [], "market": []}


def random_agent(obs):
    rng = random.Random()
    farms = obs.get("farms", [])
    player = obs.get("player", 0)
    private = obs.get("private", {}) or {}
    farm = farms[player] if farms and player < len(farms) else None
    if farm is None:
        return {"farmer": ["PASS"], "hands": [], "market": []}

    farmer_ops = ["NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST", "PASS"]
    market = []
    seeds = private.get("seeds", {})

    affordable = [c for c in CROPS if CROPS[c]["seed"] <= farm["money"]]
    if affordable and rng.random() < 0.1:
        market.append(["BUY_SEED", rng.choice(affordable), 1])

    available_seeds = [c for c, n in seeds.items() if n > 0]
    if available_seeds and rng.random() < 0.3:
        farmer = ["PLANT", rng.choice(available_seeds)]
    else:
        farmer = [rng.choice(farmer_ops)]

    hands_actions = [[rng.choice(farmer_ops)] for _ in farm.get("hands", [])]
    return {"farmer": farmer, "hands": hands_actions, "market": market}


def starter_agent(obs):
    """Carrot loop: buy seed, plant on the current tile, water, harvest at max_yield_day."""
    farms = obs.get("farms", [])
    player = obs.get("player", 0)
    private = obs.get("private", {}) or {}
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    farm = farms[player]
    fx, fy = farm["farmer"]
    tile = farm["tiles"][fy][fx]
    day = obs.get("day", 0)
    seeds = private.get("seeds", {})
    shed = private.get("shed", {})

    market = []
    if shed.get("CARROT", 0) > 0:
        market.append(["SELL", "CARROT", shed["CARROT"]])
    if seeds.get("CARROT", 0) == 0 and farm["money"] >= CROPS["CARROT"]["seed"]:
        market.append(["BUY_SEED", "CARROT", 1])

    farmer = ["PASS"]
    if tile is None and seeds.get("CARROT", 0) > 0:
        farmer = ["PLANT", "CARROT"]
    elif isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile["crop"] == "CARROT":
        age = day - tile["planted_day"]
        if age >= CROPS["CARROT"]["max_yield_day"]:
            farmer = ["HARVEST"]
        elif not tile["watered_today"]:
            farmer = ["WATER"]
    return {"farmer": farmer, "hands": [], "market": market}


agents = {"pass": pass_agent, "random": random_agent, "starter": starter_agent}
