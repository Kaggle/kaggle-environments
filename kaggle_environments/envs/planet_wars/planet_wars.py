"""Planet Wars — port of the 2009/2010 Google AI Challenge competition.

The simulation rules are a bit-exact reproduction of Jeff Cameron's reference
engine (Apache-2.0), specifically the battle resolution, ExecuteOrder
validation, DoTimeStep ordering, and Winner conditions from the C++ port by
Albert Zeyer. Map generation approximates the distribution of the original
contest's map_generator_v2.py.
"""

import json
import math
import random
from collections import namedtuple
from os import path

# Named tuples for agent convenience. The list-of-lists shape stored on
# observations matches the orbit_wars convention so JSON replays stay compact.
Planet = namedtuple("Planet", ["id", "x", "y", "owner", "num_ships", "growth_rate"])
Fleet = namedtuple(
    "Fleet",
    ["owner", "num_ships", "source", "dest", "total_trip", "turns_remaining"],
)

# Map generator constants — match map_generator_v2.py from the original
# starterpackage so the procedural distribution mirrors the contest maps.
MIN_PLANETS = 15
MAX_PLANETS = 30
MAX_CENTRAL = 5
MIN_SHIPS = 1
MAX_SHIPS = 100
MIN_GROWTH = 1
MAX_GROWTH = 5
HOME_SHIPS = 100
HOME_GROWTH = 5
MIN_DISTANCE = 2
MIN_STARTING_DISTANCE = 4
MAX_RADIUS = 15
# Reject placements where the floating-point Euclidean distance is within
# epsilon of an integer — small platform-dependent rounding around `ceil`
# would otherwise produce different trip lengths on different machines.
EPSILON = 0.002
# Hard cap on rejection-sampling retries inside generate_map. Real seeds
# converge in a handful of tries; this just bounds pathological inputs so a
# bad seed surfaces as an error instead of an infinite loop.
MAX_PLACEMENT_TRIES = 1000


# ---------------------------------------------------------------------------
# Geometry & parsing
# ---------------------------------------------------------------------------


def distance(p1, p2):
    """Trip length between two planets — ceil of Euclidean distance.

    Matches GameDesc::Distance in game.h. Accepts Planet namedtuples, raw
    lists ([id, x, y, ...]), or (x, y) tuples.
    """
    x1, y1 = _xy(p1)
    x2, y2 = _xy(p2)
    return math.ceil(math.hypot(x1 - x2, y1 - y2))


def _xy(p):
    if isinstance(p, Planet):
        return p.x, p.y
    if isinstance(p, (list, tuple)):
        if len(p) >= 3 and isinstance(p[1], (int, float)) and isinstance(p[2], (int, float)):
            # [id, x, y, ...] form used in observations
            return p[1], p[2]
        return p[0], p[1]
    return p["x"], p["y"]


def parse_map(text):
    """Parse a Planet Wars Point-in-Time map.

    Returns (planets, fleets) where each entry is the list shape used in
    observations. Mirrors Game::ParseGameState in game.cpp.
    """
    planets = []
    fleets = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        if tokens[0] == "P":
            if len(tokens) != 6:
                raise ValueError(f"invalid planet line: {raw_line!r}")
            x = float(tokens[1])
            y = float(tokens[2])
            owner = int(tokens[3])
            num_ships = int(tokens[4])
            growth_rate = int(tokens[5])
            planets.append([len(planets), x, y, owner, num_ships, growth_rate])
        elif tokens[0] == "F":
            if len(tokens) != 7:
                raise ValueError(f"invalid fleet line: {raw_line!r}")
            fleets.append(
                [
                    int(tokens[1]),  # owner
                    int(tokens[2]),  # num_ships
                    int(tokens[3]),  # source
                    int(tokens[4]),  # dest
                    int(tokens[5]),  # total_trip
                    int(tokens[6]),  # turns_remaining
                ]
            )
        else:
            raise ValueError(f"unknown map line: {raw_line!r}")
    return planets, fleets


def format_map(planets, fleets=()):
    """Serialise the current state into the Point-in-Time format."""
    lines = []
    for p in planets:
        lines.append(f"P {p[1]} {p[2]} {p[3]} {p[4]} {p[5]}")
    for f in fleets:
        lines.append(f"F {f[0]} {f[1]} {f[2]} {f[3]} {f[4]} {f[5]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Map generator (port of tools/map_generator_v2.py)
# ---------------------------------------------------------------------------


def _rand_radius(rng, min_r, max_r):
    val = min_r - 1
    while val < min_r:
        val = math.sqrt(rng.random()) * max_r
    return val


def _polar(r, theta_deg):
    theta = math.radians(theta_deg)
    return r * math.cos(theta), r * math.sin(theta)


def _too_close_or_ambiguous(x, y, planets):
    """True if (x, y) is too close to any existing planet or sits at a
    Euclidean distance within EPSILON of an integer (which would make the
    ceil-distance platform-dependent).
    """
    for p in planets:
        actual = math.hypot(p[1] - x, p[2] - y)
        if math.ceil(actual) < MIN_DISTANCE:
            return True
        if abs(actual - round(actual)) < EPSILON:
            return True
    return False


def _pair_invalid(x1, y1, x2, y2, planets):
    """Validation for a candidate pair of symmetric planets."""
    a = math.hypot(x1 - x2, y1 - y2)
    if math.ceil(a) < MIN_DISTANCE or abs(a - round(a)) < EPSILON:
        return True
    if _too_close_or_ambiguous(x1, y1, planets):
        return True
    if _too_close_or_ambiguous(x2, y2, planets):
        return True
    return False


def generate_map(seed):
    """Procedurally generate a symmetric Planet Wars map.

    Approximates the distribution produced by the original contest's
    map_generator_v2.py: 15-30 planets, point-symmetric or reflective
    symmetry, home planets at distance >= 4, all other planets at
    pairwise distance >= 2.

    Returns (planets, fleets) in the list-of-lists form used in
    observations.
    """
    rng = random.Random(seed)

    planets_to_generate = rng.randint(MIN_PLANETS, MAX_PLANETS)
    # symmetry_type: +1 = radial (point-symmetric, 180° rotation through
    # origin), -1 = linear (mirror across an axis). Radial requires an odd
    # total because the centre planet is unpaired.
    if rng.randint(0, 1):
        symmetry_type = 1
        if planets_to_generate % 2 == 0:
            planets_to_generate += 1
            if planets_to_generate > MAX_PLANETS:
                planets_to_generate = MIN_PLANETS + (MIN_PLANETS % 2 == 0)
    else:
        symmetry_type = -1

    # Internal "planet" records keep x/y in original-centred coordinates;
    # we translate them at the end. Shape matches the observation form
    # ([id, x, y, owner, ships, growth]) so validation helpers can reuse it.
    planets = []

    def add(x, y, owner, ships, growth):
        planets.append([len(planets), x, y, owner, ships, growth])

    # Centre planet at origin — always neutral, may have growth 0.
    add(0.0, 0.0, 0, rng.randint(MIN_SHIPS, MAX_SHIPS), rng.randint(0, MAX_GROWTH))
    planets_to_generate -= 1

    # Home planets.
    home1_x = home1_y = home2_x = home2_y = 0.0
    theta1 = theta2 = 0.0
    for _ in range(MAX_PLACEMENT_TRIES):
        r = _rand_radius(rng, MIN_DISTANCE, MAX_RADIUS)
        theta1 = rng.uniform(0, 360)
        if symmetry_type == 1:
            theta2 = theta1 + 180 if theta1 < 180 else theta1 - 180
        else:
            theta2 = rng.uniform(0, 360)
        home1_x, home1_y = _polar(r, theta1)
        home2_x, home2_y = _polar(r, theta2)
        if _pair_invalid(home1_x, home1_y, home2_x, home2_y, planets):
            continue
        if math.ceil(math.hypot(home1_x - home2_x, home1_y - home2_y)) < MIN_STARTING_DISTANCE:
            continue
        break
    else:
        raise RuntimeError(f"generate_map: failed to place home planets in {MAX_PLACEMENT_TRIES} tries (seed={seed})")
    add(home1_x, home1_y, 1, HOME_SHIPS, HOME_GROWTH)
    add(home2_x, home2_y, 2, HOME_SHIPS, HOME_GROWTH)
    planets_to_generate -= 2

    # Central neutrals — placed along the symmetry axis, equidistant from
    # both home planets.
    if symmetry_type == 1:
        no_central = 2 * rng.randint(0, MAX_CENTRAL // 2)
        theta_a = (theta1 + theta2) / 2
        theta_b = theta_a + 180
        for _ in range(no_central // 2):
            ships = rng.randint(MIN_SHIPS, MAX_SHIPS)
            growth = rng.randint(MIN_GROWTH, MAX_GROWTH)
            for _try in range(MAX_PLACEMENT_TRIES):
                r = _rand_radius(rng, MIN_DISTANCE, MAX_RADIUS)
                ax, ay = _polar(r, theta_a)
                bx, by = _polar(r, theta_b)
                if not _pair_invalid(ax, ay, bx, by, planets):
                    break
            else:
                raise RuntimeError(f"generate_map: failed to place central neutral pair in {MAX_PLACEMENT_TRIES} tries (seed={seed})")
            add(ax, ay, 0, ships, growth)
            add(bx, by, 0, ships, growth)
            planets_to_generate -= 2
    else:
        # Linear symmetry: central neutrals sit on the perpendicular
        # bisector. The remaining count must be even (pairs of mirrored
        # neutrals); pick `no_central` accordingly so we end up even.
        min_central = planets_to_generate % 2
        no_central = rng.randrange(min_central, MAX_CENTRAL + 1, 2)
        theta = (theta1 + theta2) / 2
        if rng.randint(0, 1) == 1:
            theta += 180
        for _ in range(no_central):
            ships = rng.randint(MIN_SHIPS, MAX_SHIPS)
            growth = rng.randint(MIN_GROWTH, MAX_GROWTH)
            for _try in range(MAX_PLACEMENT_TRIES):
                r = _rand_radius(rng, 0, MAX_RADIUS)
                x, y = _polar(r, theta)
                if not _too_close_or_ambiguous(x, y, planets):
                    actual = math.hypot(x, y)
                    if abs(actual - round(actual)) >= EPSILON:
                        break
            else:
                raise RuntimeError(f"generate_map: failed to place linear-axis neutral in {MAX_PLACEMENT_TRIES} tries (seed={seed})")
            add(x, y, 0, ships, growth)
            planets_to_generate -= 1

    # Remaining symmetric pairs of neutrals.
    assert planets_to_generate % 2 == 0, "odd remainder after central neutrals"
    home_distance = math.ceil(math.hypot(home1_x - home2_x, home1_y - home2_y))
    for i in range(planets_to_generate // 2):
        if i == 0:
            # Cap the first pair's ship count so neutrals near home planets
            # aren't unconquerable in the early game.
            cap = min(MAX_SHIPS, 5 * home_distance - 1)
            cap = max(cap, MIN_SHIPS)
            ships = rng.randint(MIN_SHIPS, cap)
        else:
            ships = rng.randint(MIN_SHIPS, MAX_SHIPS)
        growth = rng.randint(MIN_GROWTH, MAX_GROWTH)
        for _try in range(MAX_PLACEMENT_TRIES):
            r = _rand_radius(rng, MIN_DISTANCE, MAX_RADIUS)
            delta = rng.uniform(0, 360)
            ax, ay = _polar(r, theta1 + delta)
            bx, by = _polar(r, theta2 + symmetry_type * delta)
            if not _pair_invalid(ax, ay, bx, by, planets):
                break
        else:
            raise RuntimeError(f"generate_map: failed to place symmetric neutral pair in {MAX_PLACEMENT_TRIES} tries (seed={seed})")
        add(ax, ay, 0, ships, growth)
        add(bx, by, 0, ships, growth)

    # Translate so all coordinates are non-negative.
    for p in planets:
        p[1] += MAX_RADIUS
        p[2] += MAX_RADIUS

    return planets, []


# ---------------------------------------------------------------------------
# Simulation — mirrors game.cpp
# ---------------------------------------------------------------------------


def _fight_battle(planet, arriving_fleets):
    """Resolve battle on a planet. Mirrors PlanetState::FightBattle in
    game.cpp lines 74-117.

    `planet` is mutated in place. `arriving_fleets` is the subset of fleets
    whose turns_remaining == 0 and dest == planet.id.
    """
    forces = {}  # owner -> ships
    forces[planet[3]] = forces.get(planet[3], 0) + planet[4]
    for f in arriving_fleets:
        forces[f[0]] = forces.get(f[0], 0) + f[1]

    # Top two by ship count using the original strict-greater walk so the
    # tie semantics line up with the C++ engine: a tie at the top leaves
    # the planet's prior owner intact with zero ships.
    winner_owner, winner_ships = 0, 0
    second_ships = 0
    for owner, ships in forces.items():
        if ships > second_ships:
            if ships > winner_ships:
                second_ships = winner_ships
                winner_owner, winner_ships = owner, ships
            else:
                second_ships = ships

    if winner_ships > second_ships:
        planet[3] = winner_owner
        planet[4] = winner_ships - second_ships
    else:
        planet[4] = 0


def _validate_orders(orders, planets, player_owner):
    """Replay a player's orders against a snapshot of their planet ships
    using the same checks as GameState::ExecuteOrder (game.cpp:180-210).

    Returns True iff `orders` is a list (possibly empty) and every order in
    it is valid. None is treated as invalid — "no action" is `[]`, and
    None only appears upstream when core.py marked the agent
    TIMEOUT/ERROR/INVALID.
    """
    if not isinstance(orders, list):
        return False
    if not orders:
        return True

    # Per-source running totals so we catch "sum of orders > planet ships".
    ships_remaining = {}
    num_planets = len(planets)
    for order in orders:
        if not isinstance(order, (list, tuple)) or len(order) != 3:
            return False
        try:
            src = int(order[0])
            dst = int(order[1])
            ships = int(order[2])
        except (TypeError, ValueError):
            return False
        if ships <= 0:
            return False
        if src < 0 or src >= num_planets:
            return False
        if dst < 0 or dst >= num_planets:
            return False
        if src == dst:
            return False
        source = planets[src]
        if source[3] != player_owner:
            return False
        available = ships_remaining.setdefault(src, source[4])
        if ships > available:
            return False
        ships_remaining[src] = available - ships
    return True


def _apply_orders(orders, planets, fleets, player_owner):
    """Deduct ships from sources and create (or merge into) fleets.

    Mirrors GameState::ExecuteOrder including the same-turn merge by
    matching (owner, source, dest, turns_remaining) — see game.cpp:162-208.

    Precondition: caller must have run `_validate_orders` first; this
    function trusts the orders are well-formed and within budget.
    """
    if not orders:
        return
    for order in orders:
        src, dst, ships = int(order[0]), int(order[1]), int(order[2])
        source = planets[src]
        dest = planets[dst]
        source[4] -= ships
        trip = distance((source[1], source[2]), (dest[1], dest[2]))
        existing = None
        for f in fleets:
            # f[5] == trip means the fleet was launched this same turn
            # (turns_remaining still equals total_trip), so it's the merge
            # target. In-flight fleets from earlier turns have f[5] < trip.
            if f[0] == player_owner and f[2] == src and f[3] == dst and f[5] == trip:
                existing = f
                break
        if existing is not None:
            existing[1] += ships
        else:
            fleets.append([player_owner, ships, src, dst, trip, trip])


def _do_time_step(planets, fleets):
    """Advance one turn: decrement fleets, grow planets, resolve arrivals,
    drop landed fleets. Mirrors GameState::DoTimeStep (game.cpp:123-130).
    """
    for f in fleets:
        if f[5] > 0:
            f[5] -= 1

    arrivals_by_planet = {}
    for f in fleets:
        if f[5] == 0:
            arrivals_by_planet.setdefault(f[3], []).append(f)

    for p in planets:
        if p[3] > 0:
            p[4] += p[5]
        arriving = arrivals_by_planet.get(p[0])
        if p[3] != 0 or arriving:
            _fight_battle(p, arriving or ())

    fleets[:] = [f for f in fleets if f[5] > 0]


def _alive_players(planets, fleets):
    alive = set()
    for p in planets:
        if p[3] > 0:
            alive.add(p[3])
    for f in fleets:
        if f[0] > 0:
            alive.add(f[0])
    return alive


def _total_ships(planets, fleets, owner):
    total = 0
    for p in planets:
        if p[3] == owner:
            total += p[4]
    for f in fleets:
        if f[0] == owner:
            total += f[1]
    return total


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------


def _get(d, key, default=None):
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def _broadcast(state, planets, fleets):
    obs0 = state[0].observation
    obs0.planets = planets
    obs0.fleets = fleets
    for i in range(1, len(state)):
        state[i].observation.planets = planets
        state[i].observation.fleets = fleets


def interpreter(state, env):
    configuration = env.configuration
    obs0 = state[0].observation

    # ---- Init (env.reset). Done before the env.done early-return so it
    # runs even though agents are temporarily INACTIVE at reset.
    if not _get(obs0, "planets", None):
        if not hasattr(env, "info") or env.info is None:
            env.info = {}
        seed = env.info.get("seed")
        if seed is None:
            seed = _get(configuration, "seed", None)
        if seed is None:
            seed = random.randrange(2**31)
        # Publish the resolved seed: env.info for the replay, and back into
        # configuration so replaying with the same configuration is exactly
        # reproducible. NOT scrubbed — the map is fully observable.
        env.info["seed"] = seed
        try:
            configuration.seed = seed
        except (AttributeError, TypeError):
            configuration["seed"] = seed

        map_text = _get(configuration, "map", "random")
        if map_text is None or map_text == "random":
            planets, fleets = generate_map(seed)
        elif isinstance(map_text, str) and map_text.lstrip().startswith("P "):
            planets, fleets = parse_map(map_text)
        else:
            raise ValueError(f"configuration.map must be 'random' or a map text starting with 'P '; got {map_text!r}")

        _broadcast(state, planets, fleets)
        # `player` is per-agent and is initialised from the spec defaults
        # ([1, 2]); we don't need to set it manually here.
        return state

    if env.done:
        return state

    planets = obs0.planets
    fleets = obs0.fleets

    # ---- Validate both players' actions against the original ExecuteOrder
    # checks. Bad orders forfeit the game. core.py may have already marked
    # an agent TIMEOUT/ERROR/INVALID (with action=None) before calling us;
    # treat those as forfeits too rather than letting _validate_orders(None)
    # pass and silently freezing that player for the rest of the episode.
    actions = [_get(state[i], "action", []) for i in range(len(state))]
    prior_bad = [_get(state[i], "status", "ACTIVE") in ("TIMEOUT", "ERROR", "INVALID") for i in range(len(state))]
    valid = [
        not prior_bad[i] and _validate_orders(actions[i], planets, state[i].observation.player)
        for i in range(len(state))
    ]

    if not valid[0] or not valid[1]:
        # Still apply the valid player's orders and advance one tick so the
        # final recorded frame reflects ship growth and fleet arrivals for
        # this turn — otherwise visualizers show fleets stuck mid-flight.
        for i in range(len(state)):
            if valid[i]:
                _apply_orders(actions[i], planets, fleets, state[i].observation.player)
        _do_time_step(planets, fleets)
        _broadcast(state, planets, fleets)
        for i, ok in enumerate(valid):
            if not ok:
                # Preserve TIMEOUT / ERROR from core.py; otherwise INVALID.
                if _get(state[i], "status", "ACTIVE") not in ("TIMEOUT", "ERROR"):
                    state[i].status = "INVALID"
                state[i].reward = None
            else:
                state[i].status = "DONE"
                state[i].reward = 1
        return state

    # ---- Apply orders (player 1 then player 2), advance one tick, then
    # check the winner conditions in the same order as Game::Winner.
    for i in range(len(state)):
        _apply_orders(actions[i], planets, fleets, state[i].observation.player)

    _do_time_step(planets, fleets)
    _broadcast(state, planets, fleets)

    alive = _alive_players(planets, fleets)
    step = _get(obs0, "step", 0)
    # obs.step is the previous state's step; core.py bumps it to len(steps)
    # after we return. So `step + 2 >= episodeSteps` detects the final call.
    max_turns_reached = step + 2 >= configuration.episodeSteps

    terminated = False
    if len(alive) <= 1 or max_turns_reached:
        terminated = True

    if terminated:
        for s in state:
            s.status = "DONE"
        if len(alive) == 1:
            winner = next(iter(alive))
            for s in state:
                s.reward = 1 if s.observation.player == winner else -1
        elif len(alive) == 0:
            for s in state:
                s.reward = 0
        else:
            scores = {s.observation.player: _total_ships(planets, fleets, s.observation.player) for s in state}
            top = max(scores.values())
            tied = sum(1 for v in scores.values() if v == top)
            if tied > 1:
                for s in state:
                    s.reward = 0
            else:
                for s in state:
                    s.reward = 1 if scores[s.observation.player] == top else -1

    return state


# ---------------------------------------------------------------------------
# Renderer + html_renderer
# ---------------------------------------------------------------------------


def renderer(state, env):
    obs = state[0].observation
    step = _get(obs, "step", 0)
    out = [f"Turn {step} / {env.configuration.episodeSteps}"]
    out.append("Planets:")
    for p in _get(obs, "planets", []) or []:
        out.append(f"  {p[0]:3d}  owner={p[3]}  ships={p[4]:4d}  growth={p[5]}  ({p[1]:6.2f}, {p[2]:6.2f})")
    fleets = _get(obs, "fleets", []) or []
    out.append(f"Fleets ({len(fleets)}):")
    for f in fleets:
        out.append(f"  owner={f[0]}  ships={f[1]:4d}  {f[2]}->{f[3]}  turns_left={f[5]}/{f[4]}")
    return "\n".join(out) + "\n"


dir_path = path.dirname(__file__)
with open(path.join(dir_path, "planet_wars.json")) as f:
    specification = json.load(f)


def html_renderer():
    html_path = path.join(dir_path, "visualizer", "default", "dist", "index.html")
    if path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return f.read()
    return ""


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


def do_nothing(obs, config=None):
    return []


def random_agent(obs, config=None):
    """Each owned planet has a 30% chance of sending half its ships to a
    random other planet."""
    player = _get(obs, "player", 1)
    planets = _get(obs, "planets", []) or []
    moves = []
    if not planets:
        return moves
    for p in planets:
        if p[3] != player or p[4] < 2:
            continue
        if random.random() > 0.3:
            continue
        ships = p[4] // 2
        targets = [t for t in planets if t[0] != p[0]]
        if not targets:
            continue
        target = random.choice(targets)
        moves.append([p[0], target[0], ships])
    return moves


def nearest_enemy(obs, config=None):
    """Each owned planet sends half its ships to its nearest non-self
    planet that isn't already owned by us."""
    player = _get(obs, "player", 1)
    planets = _get(obs, "planets", []) or []
    moves = []
    for p in planets:
        if p[3] != player or p[4] < 2:
            continue
        ships = p[4] // 2
        candidates = [t for t in planets if t[3] != player]
        if not candidates:
            continue
        target = min(candidates, key=lambda t: math.hypot(p[1] - t[1], p[2] - t[2]))
        moves.append([p[0], target[0], ships])
    return moves


agents = {
    "do_nothing": do_nothing,
    "random": random_agent,
    "nearest_enemy": nearest_enemy,
}
