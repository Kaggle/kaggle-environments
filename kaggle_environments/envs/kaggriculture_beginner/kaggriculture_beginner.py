import json
import random
from os import path

dirpath = path.dirname(__file__)


CROPS = {
    "WHEAT": {
        "seed": 10,
        "price": 25,
        "first_yield_day": 2,
        "max_yield_day": 4,
        "interval": 0,
        "max_yield": 4,
        "ongoing": False,
    },
    "CARROT": {
        "seed": 20,
        "price": 35,
        "first_yield_day": 2,
        "max_yield_day": 3,
        "interval": 0,
        "max_yield": 4,
        "ongoing": False,
    },
    "TOMATO": {
        "seed": 50,
        "price": 60,
        "first_yield_day": 8,
        "max_yield_day": 8,
        "interval": 1,
        "max_yield": 4,
        "ongoing": True,
    },
    "STRAWBERRY": {
        "seed": 100,
        "price": 120,
        "first_yield_day": 10,
        "max_yield_day": 10,
        "interval": 2,
        "max_yield": 4,
        "ongoing": True,
    },
    "MELON": {
        "seed": 80,
        "price": 250,
        "first_yield_day": 10,
        "max_yield_day": 12,
        "interval": 0,
        "max_yield": 6,
        "ongoing": False,
    },
}

# (dx, dy) deltas; y grows downward (row index), so NORTH decreases y.
FARMER_MOVES = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "EAST": (1, 0),
    "WEST": (-1, 0),
}


def get(d, key, default):
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def _new_farm(board_size, starting_money):
    return {
        "money": float(starting_money),
        "seeds": {crop: 0 for crop in CROPS},
        "farmer": [board_size - 1, board_size - 1],
        "tiles": [[None for _ in range(board_size)] for _ in range(board_size)],
    }


def _new_plant(crop, day, turns_per_day):
    crop_data = CROPS[crop]
    ongoing = crop_data["ongoing"]
    return {
        "crop": crop,
        "planted_day": day,
        "watered_today": False,
        # Start at 1 so a freshly planted seed dies if it isn't watered on its
        # planting day -- the planting day counts as the first "unwatered" day.
        "consecutive_unwatered": 1,
        # One-time crops start at 1; ongoing crops
        # start at 0 and accumulate via _daily_refresh.
        "yield_units": 0 if ongoing else 1,
        # First step at which decay applies. For one-time crops, decay begins
        # the day AFTER max_yield_day (the peak day). For ongoing crops, set lazily
        # in _daily_refresh once production hits max_yield.
        "max_lifespan_step": (-1 if ongoing else (day + crop_data["max_yield_day"] + 1) * turns_per_day),
    }


def _initialize(state, env):
    configuration = env.configuration
    num_agents = len(state)
    obs0 = state[0].observation

    board_size = int(get(configuration, "boardSize", 5))
    starting_money = int(get(configuration, "startingMoney", 150))

    farms = [_new_farm(board_size, starting_money) for _ in range(num_agents)]
    obs0.farms = farms
    obs0.day = 0
    obs0.hour = 0
    for i in range(num_agents):
        state[i].observation.player = i
        if i > 0:
            state[i].observation.farms = farms
            state[i].observation.day = 0
            state[i].observation.hour = 0


def _apply_farmer_action(farm, action, board_size, day, turns_per_day=24):
    """Mutates farm in place. Returns (crop, yield_units) for a successful HARVEST,
    None otherwise.

    NOTE: invalid or illegal actions are silently no-ops (not status=INVALID).
    A malformed action will not end the game.
    """
    if not isinstance(action, list) or not action:
        return None
    op = action[0]
    fx, fy = farm["farmer"]

    if op in FARMER_MOVES:
        dx, dy = FARMER_MOVES[op]
        nx, ny = fx + dx, fy + dy
        if 0 <= nx < board_size and 0 <= ny < board_size:
            farm["farmer"] = [nx, ny]
        return None

    if op == "PLANT":
        if len(action) < 2:
            return None
        crop = action[1]
        if crop not in CROPS:
            return None
        if farm["tiles"][fy][fx] is not None:
            return None
        if farm["seeds"].get(crop, 0) <= 0:
            return None
        farm["seeds"][crop] -= 1
        farm["tiles"][fy][fx] = _new_plant(crop, day, turns_per_day)
        return None

    if op == "WATER":
        tile = farm["tiles"][fy][fx]
        if tile is None or tile["watered_today"]:
            return None
        tile["watered_today"] = True
        crop_data = CROPS[tile["crop"]]
        if not crop_data["ongoing"]:
            age_days = day - tile["planted_day"]
            window_start = (crop_data["max_yield_day"] + 1) // 2
            if window_start <= age_days <= crop_data["max_yield_day"]:
                tile["yield_units"] = min(crop_data["max_yield"], tile["yield_units"] + 1)
        return None

    if op == "HARVEST":
        return _try_harvest(farm, fx, fy, day)

    return None


def _try_harvest(farm, fx, fy, day):
    tile = farm["tiles"][fy][fx]
    if tile is None:
        return None
    crop = CROPS[tile["crop"]]
    age_days = day - tile["planted_day"]
    if age_days < crop["first_yield_day"]:
        return None

    if crop["ongoing"]:
        # Ongoing plant: take all ready units. The plant is left standing --
        # it may still produce more, or decay may eventually remove it.
        units = tile["yield_units"]
        if units <= 0:
            return None
        tile["yield_units"] = 0
        return (tile["crop"], units)

    # One-time crop: yield_units accumulated via watering during the bonus
    # window and decayed over time after max lifespan. Tile removed on harvest.
    yield_units = tile["yield_units"]
    crop_name = tile["crop"]
    farm["tiles"][fy][fx] = None
    if yield_units <= 0:
        return None
    return (crop_name, yield_units)


def _process_market(state):
    """Round-robin process market queues across players. With BUY_SEED at fixed
    prices the order doesn't matter, but we still keep it consistent with the 
    behavior for the advanced version of kaggriculture."""
    obs0 = state[0].observation
    queues = []
    for s in state:
        action = s.action if isinstance(s.action, dict) else {}
        market = action.get("market", []) if isinstance(action, dict) else []
        queues.append(list(market) if isinstance(market, list) else [])

    max_len = max((len(q) for q in queues), default=0)
    for i in range(max_len):
        for player_id, q in enumerate(queues):
            if i >= len(q):
                continue
            order = q[i]
            if not isinstance(order, list) or not order:
                continue
            op = order[0]
            farm = obs0.farms[player_id]
            if op == "BUY_SEED" and len(order) >= 3:
                crop = order[1]
                try:
                    n = int(order[2])
                except (TypeError, ValueError):
                    continue
                if crop not in CROPS or n <= 0:
                    continue
                cost = CROPS[crop]["seed"] * n
                if farm["money"] >= cost:
                    farm["money"] -= cost
                    farm["seeds"][crop] = farm["seeds"].get(crop, 0) + n


def _decay_plants(farm, step):
    """One-time crops past their max lifespan lose 1 yield_unit every other
    turn (offsets 0, 2, 4, ... after max_lifespan_step). Plants are removed
    when yield_units hits zero."""
    board_size = len(farm["tiles"])
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            if tile is None:
                continue
            mls = tile["max_lifespan_step"]
            if mls < 0 or step < mls:
                continue
            if (step - mls) % 2 != 0:
                continue
            tile["yield_units"] -= 1
            if tile["yield_units"] <= 0:
                farm["tiles"][y][x] = None


def _daily_refresh(farm, current_day=0, turns_per_day=24):
    """End-of-day plant maintenance. Runs at the end of `current_day`:
    1. Kill plants unwatered for 2+ consecutive days.
    2. For surviving ongoing crops, produce 1 unit if the next day is a
       scheduled production day (and the lifetime cap hasn't been hit).
       When the cap is reached, set max_lifespan_step so decay begins the
       day after.
    3. Reset watered_today for the new day."""
    board_size = len(farm["tiles"])
    next_day = current_day + 1
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            if tile is None:
                continue

            if tile["watered_today"]:
                tile["consecutive_unwatered"] = 0
            else:
                tile["consecutive_unwatered"] += 1
            tile["watered_today"] = False
            if tile["consecutive_unwatered"] >= 2:
                farm["tiles"][y][x] = None
                continue

            crop_data = CROPS[tile["crop"]]
            # One-time crops gain yield only via watering during the bonus
            # window (handled in _apply_farmer_action); they have no scheduled
            # daily production, so skip the rest of this loop body.
            if not crop_data["ongoing"]:
                continue
            days_since_first = next_day - tile["planted_day"] - crop_data["first_yield_day"]
            if days_since_first < 0:
                continue
            interval = crop_data["interval"]
            if days_since_first % interval != 0:
                continue
            production_count = days_since_first // interval + 1
            if production_count > crop_data["max_yield"]:
                continue
            tile["yield_units"] += 1
            if production_count == crop_data["max_yield"]:
                tile["max_lifespan_step"] = (next_day + 1) * turns_per_day


def interpreter(state, env):
    num_agents = len(state)
    obs0 = state[0].observation

    if not hasattr(obs0, "farms") or not obs0.farms:
        _initialize(state, env)
        return state

    if env.done:
        return state

    configuration = env.configuration
    turns_per_day = max(1, int(get(configuration, "turnsPerDay", 24)))
    board_size = int(get(configuration, "boardSize", 5))

    step = get(obs0, "step", 0)
    day = step // turns_per_day

    for i, s in enumerate(state):
        action = s.action if isinstance(s.action, dict) else {}
        farmer_action = action.get("farmer", ["PASS"]) if isinstance(action, dict) else ["PASS"]
        result = _apply_farmer_action(obs0.farms[i], farmer_action, board_size, day, turns_per_day)
        if result is not None:
            crop_name, units = result
            obs0.farms[i]["money"] += float(CROPS[crop_name]["price"] * units)

    _process_market(state)

    for farm in obs0.farms:
        _decay_plants(farm, step)

    if (step + 1) % turns_per_day == 0:
        for farm in obs0.farms:
            _daily_refresh(farm, day, turns_per_day)

    next_step = step + 1
    obs0.day = next_step // turns_per_day
    obs0.hour = next_step % turns_per_day
    for i in range(1, num_agents):
        state[i].observation.farms = obs0.farms
        state[i].observation.day = obs0.day
        state[i].observation.hour = obs0.hour

    if step >= configuration.episodeSteps - 2:
        for s in state:
            s.status = "DONE"
            s.reward = float(obs0.farms[s.observation.player]["money"])

    return state


def renderer(state, env):
    obs = state[0].observation
    out = f"Step {get(obs, 'step', 0)}  Day {get(obs, 'day', 0)}  Hour {get(obs, 'hour', 0)}\n"
    for i, farm in enumerate(get(obs, "farms", []) or []):
        out += f"Player {i}: ${farm['money']:.0f}  farmer={farm['farmer']}  seeds={farm['seeds']}\n"
        for row in farm["tiles"]:
            cells = []
            for tile in row:
                cells.append("." if tile is None else tile["crop"][0])
            out += "  " + " ".join(cells) + "\n"
    return out


json_path = path.abspath(path.join(dirpath, "kaggriculture_beginner.json"))
with open(json_path) as json_file:
    specification = json.load(json_file)


def html_renderer(env, mode):
    jspath = path.join(dirpath, "visualizer", "default", "dist", "index.html")
    if path.exists(jspath):
        with open(jspath, encoding="utf-8") as f:
            return f.read()
    return ""


def pass_agent(obs):
    return {"farmer": ["PASS"], "market": []}


def random_agent(obs):
    rng = random.Random()
    farmer_ops = ["NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST", "PASS"]
    market = []
    farms = obs.get("farms", [])
    player = obs.get("player", 0)
    money = farms[player]["money"] if farms and player < len(farms) else 0
    seeds = farms[player]["seeds"] if farms and player < len(farms) else {}

    affordable = [c for c in CROPS if CROPS[c]["seed"] <= money]
    if affordable and rng.random() < 0.1:
        crop = rng.choice(affordable)
        market.append(["BUY_SEED", crop, 1])

    available_seeds = [c for c, n in seeds.items() if n > 0]
    if available_seeds and rng.random() < 0.3:
        farmer = ["PLANT", rng.choice(available_seeds)]
    else:
        farmer = [rng.choice(farmer_ops)]

    return {"farmer": farmer, "market": market}


def starter_agent(obs):
    """Deterministic carrot loop: buy a carrot seed, plant on the current
    tile, water through the bonus window, harvest at max_yield_day, repeat."""
    farms = obs.get("farms", [])
    player = obs.get("player", 0)
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "market": []}
    farm = farms[player]
    fx, fy = farm["farmer"]
    tile = farm["tiles"][fy][fx]
    day = obs.get("day", 0)
    carrot_seeds = farm["seeds"].get("CARROT", 0)
    money = farm["money"]
    carrot = CROPS["CARROT"]

    if tile is None:
        if carrot_seeds > 0:
            return {"farmer": ["PLANT", "CARROT"], "market": []}
        market = []
        if money >= carrot["seed"]:
            market.append(["BUY_SEED", "CARROT", 1])
        return {"farmer": ["PASS"], "market": market}

    if tile["crop"] != "CARROT":
        return {"farmer": ["PASS"], "market": []}

    age_days = day - tile["planted_day"]
    if not tile["watered_today"] and age_days <= carrot["max_yield_day"]:
        return {"farmer": ["WATER"], "market": []}
    if age_days >= carrot["max_yield_day"]:
        return {"farmer": ["HARVEST"], "market": []}
    return {"farmer": ["PASS"], "market": []}


agents = {"pass": pass_agent, "random": random_agent, "starter": starter_agent}
