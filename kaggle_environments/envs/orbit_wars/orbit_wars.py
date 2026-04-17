import json
import math
from collections import namedtuple
from os import path
import random

# Named tuples for agent convenience.
# Planets and fleets share a common [id, owner, x, y, ...] prefix.
Planet = namedtuple(
    "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
)
Fleet = namedtuple(
    "Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"]
)

# Constants
BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
PLANET_CLEARANCE = 7
MIN_PLANET_GROUPS = 5
MAX_PLANET_GROUPS = 10
MIN_STATIC_GROUPS = 3
COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def point_to_segment_distance(p, v, w):
    """Minimum distance from point p to line segment v-w."""
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return distance(p, v)
    t = max(
        0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2)
    )
    projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return distance(p, projection)


def generate_planets():
    planets = []
    num_q1 = random.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS)
    id_counter = 0

    # Phase 1: Generate 3 guaranteed static planet groups using polar coordinates.
    # Sample within the circular region where orbital_radius + r >= ROTATION_RADIUS_LIMIT.
    static_groups = 0
    for _ in range(5000):
        if static_groups >= MIN_STATIC_GROUPS:
            break
        prod = random.randint(1, 5)
        r = 1 + math.log(prod)
        angle = random.uniform(0, math.pi / 2)  # Q1 angle from center
        min_orbital = ROTATION_RADIUS_LIMIT - r
        # Max orbital radius constrained by board edges
        max_orbital = (BOARD_SIZE - CENTER - r) / max(math.cos(angle), math.sin(angle))
        if min_orbital > max_orbital:
            continue
        orbital_r = random.uniform(min_orbital, max_orbital)
        x = CENTER + orbital_r * math.cos(angle)
        y = CENTER + orbital_r * math.sin(angle)

        # Verify board bounds for all four symmetric copies
        if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
            continue
        if (BOARD_SIZE - x) - r < 0 or (BOARD_SIZE - y) - r < 0:
            continue
        # Ensure symmetric copies don't overlap: Q1 must be far enough from axes
        if (x - CENTER) < r + 5 or (y - CENTER) < r + 5:
            continue

        ships = min(random.randint(5, 99), random.randint(5, 99))
        temp_planets = [
            [id_counter, -1, x, y, r, ships, prod],
            [id_counter + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
            [id_counter + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
            [id_counter + 3, -1, BOARD_SIZE - x, BOARD_SIZE - y, r, ships, prod],
        ]

        # Check overlap with existing planets
        valid = True
        for tp in temp_planets:
            for p in planets:
                if distance((p[2], p[3]), (tp[2], tp[3])) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            planets.extend(temp_planets)
            id_counter += 4
            static_groups += 1

    # Phase 1.5: Generate one guaranteed orbiting group on the y=x diagonal.
    # In 4p games, starting on an orbiting planet is only fair when all 4
    # copies are evenly spaced (π/2 apart). The y=x diagonal (angle=π/4)
    # gives exactly this: copies at π/4, 3π/4, 5π/4, 7π/4.
    for _ in range(1000):
        prod = random.randint(1, 5)
        r = 1 + math.log(prod)
        min_orbital = SUN_RADIUS + r + 10
        max_orbital = ROTATION_RADIUS_LIMIT - r
        if min_orbital >= max_orbital:
            continue
        orbital_r = random.uniform(min_orbital, max_orbital)
        x = CENTER + orbital_r * math.cos(math.pi / 4)
        y = CENTER + orbital_r * math.sin(math.pi / 4)

        ships = min(random.randint(5, 99), random.randint(5, 99))
        temp_planets = [
            [id_counter, -1, x, y, r, ships, prod],
            [id_counter + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
            [id_counter + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
            [id_counter + 3, -1, BOARD_SIZE - x, BOARD_SIZE - y, r, ships, prod],
        ]

        valid = True
        for tp in temp_planets:
            tp_orbital = distance((tp[2], tp[3]), (CENTER, CENTER))
            for p in planets:
                p_orbital = distance((p[2], p[3]), (CENTER, CENTER))
                p_is_static = p_orbital + p[4] >= ROTATION_RADIUS_LIMIT

                if distance((p[2], p[3]), (tp[2], tp[3])) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False
                    break

                # Orbiting vs static cross-check
                if p_is_static:
                    if abs(tp_orbital - p_orbital) < tp[4] + p[4] + PLANET_CLEARANCE:
                        valid = False
                        break
            if not valid:
                break

        if valid:
            planets.extend(temp_planets)
            id_counter += 4
            break

    # Phase 2: Fill remaining planet groups with the normal random loop.
    attempts = 0
    max_attempts = 5000
    has_orbiting = False

    while len(planets) < num_q1 * 4 or (not has_orbiting and attempts < max_attempts):
        attempts += 1
        if attempts >= max_attempts:
            break
        prod = random.randint(1, 5)
        r = 1 + math.log(prod)
        x = random.uniform(CENTER + 15, BOARD_SIZE - r - 5)
        y = random.uniform(CENTER + 15, BOARD_SIZE - r - 5)

        orbital_radius = distance((x, y), (CENTER, CENTER))

        # Reject if too close to sun
        if orbital_radius < SUN_RADIUS + r + 10:
            continue

        # Reject if planet body would extend past board edge during rotation
        if orbital_radius + r >= ROTATION_RADIUS_LIMIT:
            # Planet is static (won't rotate), just check it stays on board
            if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
                continue

        valid = True
        ships = random.randint(5, 30)
        temp_planets = [
            [id_counter, -1, x, y, r, ships, prod],
            [id_counter + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
            [id_counter + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
            [id_counter + 3, -1, BOARD_SIZE - x, BOARD_SIZE - y, r, ships, prod],
        ]

        for tp in temp_planets:
            tp_orbital = distance((tp[2], tp[3]), (CENTER, CENTER))
            tp_is_rotating = tp_orbital + tp[4] < ROTATION_RADIUS_LIMIT

            for p in planets:
                p_orbital = distance((p[2], p[3]), (CENTER, CENTER))
                p_is_rotating = p_orbital + p[4] < ROTATION_RADIUS_LIMIT

                # Standard initial distance check
                if distance((p[2], p[3]), (tp[2], tp[3])) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False
                    break

                # Cross-check: one rotating, one static -> min distance over
                # full rotation is |orbital_radius_1 - orbital_radius_2|
                if tp_is_rotating != p_is_rotating:
                    if abs(tp_orbital - p_orbital) < tp[4] + p[4] + PLANET_CLEARANCE:
                        valid = False
                        break

            if not valid:
                break

        if valid:
            if orbital_radius + r < ROTATION_RADIUS_LIMIT:
                has_orbiting = True
            planets.extend(temp_planets)
            id_counter += 4

    return planets


def generate_comet_paths(
    initial_planets,
    angular_velocity,
    spawn_step,
    comet_planet_ids=None,
    comet_speed=4.0,
):
    """Generate 4 symmetric elliptical orbit paths for extra-solar objects.

    Returns list of 4 paths (one per quadrant symmetry), each path a list
    of [x, y] positions at comet_speed units/turn.  Returns None on failure.
    """
    if comet_planet_ids is None:
        comet_planet_ids = set()
    else:
        comet_planet_ids = set(comet_planet_ids)
    for _ in range(300):
        # Highly eccentric ellipse with sun at one focus
        e = random.uniform(0.75, 0.93)
        a = random.uniform(60, 150)
        perihelion = a * (1 - e)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            continue

        b = a * math.sqrt(1 - e**2)
        c_val = a * e
        # Orientation: perihelion direction from sun (keep in Q4 quadrant)
        phi = random.uniform(math.pi / 6, math.pi / 3)

        # Dense sample around perihelion half of orbit
        dense = []
        num = 5000
        for i in range(num):
            t = 0.3 * math.pi + 1.4 * math.pi * i / (num - 1)
            # Ellipse with focus at origin
            ex = c_val + a * math.cos(t)
            ey = b * math.sin(t)
            # Rotate and translate to board
            x = CENTER + ex * math.cos(phi) - ey * math.sin(phi)
            y = CENTER + ex * math.sin(phi) + ey * math.cos(phi)
            dense.append((x, y))

        # Re-sample at constant comet_speed arc-length intervals
        path = [dense[0]]
        cum = 0.0
        target = comet_speed
        for i in range(1, len(dense)):
            cum += distance(dense[i], dense[i - 1])
            if cum >= target:
                path.append(dense[i])
                target += comet_speed

        # Extract contiguous on-board segment
        board_start = None
        board_end = None
        for i, (x, y) in enumerate(path):
            if 0 <= x <= BOARD_SIZE and 0 <= y <= BOARD_SIZE:
                if board_start is None:
                    board_start = i
                board_end = i

        if board_start is None:
            continue
        visible = path[board_start : board_end + 1]
        if not (5 <= len(visible) <= 40):
            continue

        # Build 4 symmetric paths
        paths = [
            [[x, y] for x, y in visible],
            [[BOARD_SIZE - x, y] for x, y in visible],
            [[x, BOARD_SIZE - y] for x, y in visible],
            [[BOARD_SIZE - x, BOARD_SIZE - y] for x, y in visible],
        ]

        # Separate planets into static and orbiting (exclude other comets)
        static_planets = []
        orbiting_planets = []
        for planet in initial_planets:
            if planet[0] in comet_planet_ids:
                continue
            pr = distance((planet[2], planet[3]), (CENTER, CENTER))
            if pr + planet[4] < ROTATION_RADIUS_LIMIT:
                orbiting_planets.append(planet)
            else:
                static_planets.append(planet)

        valid = True
        buf = COMET_RADIUS + 0.5
        for k, (cx, cy) in enumerate(visible):
            # Check sun
            if distance((cx, cy), (CENTER, CENTER)) < SUN_RADIUS + COMET_RADIUS:
                valid = False
                break

            # Check all 4 symmetric positions against static planets
            sym_pts = [
                (cx, cy),
                (BOARD_SIZE - cx, cy),
                (cx, BOARD_SIZE - cy),
                (BOARD_SIZE - cx, BOARD_SIZE - cy),
            ]
            for planet in static_planets:
                for sp in sym_pts:
                    if distance(sp, (planet[2], planet[3])) < planet[4] + buf:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

            # Check against orbiting planets at their actual positions
            # Use tighter buffer — sweep mechanics handle runtime near-misses
            game_step = spawn_step - 1 + k
            for planet in orbiting_planets:
                dx = planet[2] - CENTER
                dy = planet[3] - CENTER
                orb_r = math.sqrt(dx**2 + dy**2)
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + angular_velocity * game_step
                px = CENTER + orb_r * math.cos(cur_angle)
                py = CENTER + orb_r * math.sin(cur_angle)
                for sp in sym_pts:
                    if distance(sp, (px, py)) < planet[4] + COMET_RADIUS:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if valid:
            return paths
    return None


def interpreter(state, env):
    configuration = env.configuration
    num_agents = len(state)
    obs0 = state[0].observation

    if env.done:
        return state

    # Initialize game state if not already done
    if not hasattr(obs0, "planets") or not obs0.planets:
        angular_velocity = random.uniform(0.025, 0.05)
        obs0.angular_velocity = angular_velocity
        obs0.planets = generate_planets()
        obs0.initial_planets = [p.copy() for p in obs0.planets]
        obs0.fleets = []
        obs0.next_fleet_id = 0
        obs0.comets = []
        obs0.comet_planet_ids = []

        # Assign home planets — pick a random symmetric group of 4
        num_groups = len(obs0.planets) // 4
        if num_groups > 0:
            home_group = random.randint(0, num_groups - 1)
            base = home_group * 4

            if num_agents == 4:
                # In 4p, orbiting planets introduce asymmetry unless on the
                # y=x diagonal (where all 4 copies stay evenly spaced under
                # rotation). If the randomly picked group is orbiting,
                # redirect to the diagonal orbiting group.
                q1 = obs0.planets[base]
                orb_r = distance((q1[2], q1[3]), (CENTER, CENTER))
                if orb_r + q1[4] < ROTATION_RADIUS_LIMIT:
                    # Find the diagonal group (Q1 planet where x ≈ y)
                    for g in range(num_groups):
                        gb = g * 4
                        gp = obs0.planets[gb]
                        g_orb = distance((gp[2], gp[3]), (CENTER, CENTER))
                        if g_orb + gp[4] < ROTATION_RADIUS_LIMIT:
                            if abs((gp[2] - CENTER) - (gp[3] - CENTER)) < 0.01:
                                home_group = g
                                base = gb
                                break

            if num_agents == 2:
                obs0.planets[base][1] = 0  # Q1
                obs0.planets[base][5] = 10
                obs0.planets[base + 3][1] = 1  # Q4
                obs0.planets[base + 3][5] = 10
            elif num_agents == 4:
                for j in range(4):
                    obs0.planets[base + j][1] = j
                    obs0.planets[base + j][5] = 10

        for i in range(num_agents):
            state[i].observation.player = i
            if i > 0:
                state[i].observation.angular_velocity = obs0.angular_velocity
                state[i].observation.planets = obs0.planets
                state[i].observation.initial_planets = obs0.initial_planets
                state[i].observation.fleets = obs0.fleets
                state[i].observation.next_fleet_id = obs0.next_fleet_id
                state[i].observation.comets = obs0.comets
                state[i].observation.comet_planet_ids = obs0.comet_planet_ids

        return state

    # Remove expired comets before fleet launch so agents can't act on them
    expired_comet_pids = []
    for group in obs0.comets:
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            if idx >= len(group["paths"][i]):
                expired_comet_pids.append(pid)
    if expired_comet_pids:
        expired_set = set(expired_comet_pids)
        obs0.planets = [p for p in obs0.planets if p[0] not in expired_set]
        obs0.initial_planets = [
            p for p in obs0.initial_planets if p[0] not in expired_set
        ]
        obs0.comet_planet_ids = [
            pid for pid in obs0.comet_planet_ids if pid not in expired_set
        ]
        for group in obs0.comets:
            group["planet_ids"] = [
                pid for pid in group["planet_ids"] if pid not in expired_set
            ]
        obs0.comets = [g for g in obs0.comets if g["planet_ids"]]

    # Spawn extra-solar comets at designated steps
    step = get(obs0, "step", 0)
    comet_speed = configuration.cometSpeed
    if (step + 1) in COMET_SPAWN_STEPS:
        comet_paths = generate_comet_paths(
            obs0.initial_planets,
            obs0.angular_velocity,
            step + 1,
            obs0.comet_planet_ids,
            comet_speed,
        )
        if comet_paths:
            next_id = max(p[0] for p in obs0.planets) + 1
            comet_ships = min(
                random.randint(1, 99),
                random.randint(1, 99),
                random.randint(1, 99),
                random.randint(1, 99),
            )
            group = {"planet_ids": [], "paths": comet_paths, "path_index": -1}
            for i, p_path in enumerate(comet_paths):
                pid = next_id + i
                group["planet_ids"].append(pid)
                obs0.comet_planet_ids.append(pid)
                # Start off-board; first advancement will place at path[0]
                planet = [
                    pid,
                    -1,
                    -99,
                    -99,
                    COMET_RADIUS,
                    comet_ships,
                    COMET_PRODUCTION,
                ]
                obs0.planets.append(planet)
                obs0.initial_planets.append(planet[:])
            obs0.comets.append(group)

    # 0. Fleet Launch
    def process_moves(player_id, action):
        if not action or not isinstance(action, list):
            return
        for move in action:
            if len(move) != 3:
                continue
            from_id, angle, ships = move
            ships = int(ships)  # Sanitize to integer

            from_planet = next((p for p in obs0.planets if p[0] == from_id), None)

            if from_planet and from_planet[1] == player_id:
                if from_planet[5] >= ships and ships > 0:
                    from_planet[5] -= ships
                    # Start fleet just outside the planet so it doesn't
                    # immediately collide with its origin.
                    start_x = from_planet[2] + math.cos(angle) * (from_planet[4] + 0.1)
                    start_y = from_planet[3] + math.sin(angle) * (from_planet[4] + 0.1)
                    obs0.fleets.append(
                        [
                            obs0.next_fleet_id,
                            player_id,
                            start_x,
                            start_y,
                            angle,
                            from_id,
                            ships,
                        ]
                    )
                    obs0.next_fleet_id += 1

    for i in range(num_agents):
        process_moves(i, state[i].action)

    # 1. Production
    for planet in obs0.planets:
        if planet[1] != -1:
            planet[5] += planet[6]

    # 2. Fleet Movement (with continuous collision detection)
    # Speed scales with fleet size: 1 ship = 1/turn, max = shipSpeed (default 6)
    max_speed = configuration.shipSpeed
    fleets_to_remove = []
    combat_lists = {p[0]: [] for p in obs0.planets}

    for fleet in obs0.fleets:
        angle = fleet[4]
        ships = fleet[6]
        speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
        speed = min(speed, max_speed)
        old_pos = (fleet[2], fleet[3])
        fleet[2] += math.cos(angle) * speed
        fleet[3] += math.sin(angle) * speed
        new_pos = (fleet[2], fleet[3])

        # Check if fleet went out of bounds
        if not (0 <= fleet[2] <= BOARD_SIZE and 0 <= fleet[3] <= BOARD_SIZE):
            fleets_to_remove.append(fleet)
            continue

        # Check if fleet path crossed the sun
        if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS:
            fleets_to_remove.append(fleet)
            continue

        # Check if fleet path intersected any planet (continuous collision)
        for planet in obs0.planets:
            planet_pos = (planet[2], planet[3])
            if point_to_segment_distance(planet_pos, old_pos, new_pos) < planet[4]:
                combat_lists[planet[0]].append(fleet)
                fleets_to_remove.append(fleet)
                break

    # 3. Planet Movement & Sweep
    angular_velocity = obs0.angular_velocity
    step = get(obs0, "step", 1)
    comet_pid_set = set(obs0.comet_planet_ids)
    initial_by_id = {p[0]: p for p in obs0.initial_planets}

    def sweep_fleets(planet, old_pos, new_pos):
        """Check if any fleet is caught by a planet moving from old to new."""
        if old_pos == new_pos:
            return
        for fleet in obs0.fleets:
            if fleet not in fleets_to_remove:
                if (
                    point_to_segment_distance((fleet[2], fleet[3]), old_pos, new_pos)
                    < planet[4]
                ):
                    combat_lists[planet[0]].append(fleet)
                    fleets_to_remove.append(fleet)

    # Regular planet rotation
    for planet in obs0.planets:
        if planet[0] in comet_pid_set:
            continue
        initial_p = initial_by_id.get(planet[0])
        if not initial_p:
            continue
        dx = initial_p[2] - CENTER
        dy = initial_p[3] - CENTER
        r = math.sqrt(dx**2 + dy**2)
        old_pos = (planet[2], planet[3])

        if r + planet[4] < ROTATION_RADIUS_LIMIT:
            initial_angle = math.atan2(dy, dx)
            current_angle = initial_angle + angular_velocity * step
            planet[2] = CENTER + r * math.cos(current_angle)
            planet[3] = CENTER + r * math.sin(current_angle)

        sweep_fleets(planet, old_pos, (planet[2], planet[3]))

    # Comet movement along pre-computed paths
    expired_comet_pids = []
    for group in obs0.comets:
        group["path_index"] += 1
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            planet = next((p for p in obs0.planets if p[0] == pid), None)
            if planet is None:
                continue
            p_path = group["paths"][i]
            if idx >= len(p_path):
                expired_comet_pids.append(pid)
            else:
                old_pos = (planet[2], planet[3])
                planet[2] = p_path[idx][0]
                planet[3] = p_path[idx][1]
                # Skip sweep on first placement (old_pos is off-board placeholder)
                if old_pos[0] >= 0:
                    sweep_fleets(planet, old_pos, (planet[2], planet[3]))

    # Remove expired comets immediately
    if expired_comet_pids:
        expired_set = set(expired_comet_pids)
        obs0.planets = [p for p in obs0.planets if p[0] not in expired_set]
        obs0.initial_planets = [
            p for p in obs0.initial_planets if p[0] not in expired_set
        ]
        obs0.comet_planet_ids = [
            pid for pid in obs0.comet_planet_ids if pid not in expired_set
        ]
        for group in obs0.comets:
            group["planet_ids"] = [
                pid for pid in group["planet_ids"] if pid not in expired_set
            ]
        obs0.comets = [g for g in obs0.comets if g["planet_ids"]]

    obs0.fleets = [f for f in obs0.fleets if f not in fleets_to_remove]

    # 4. Combat Resolution
    for pid, planet_fleets in combat_lists.items():
        planet = next((p for p in obs0.planets if p[0] == pid), None)
        if not planet or not planet_fleets:
            continue

        # Sum ships per player
        player_ships = {}
        for fleet in planet_fleets:
            owner = fleet[1]
            player_ships[owner] = player_ships.get(owner, 0) + fleet[6]

        if not player_ships:
            continue

        sorted_players = sorted(
            player_ships.items(), key=lambda item: item[1], reverse=True
        )
        top_player, top_ships = sorted_players[0]

        if len(sorted_players) > 1:
            second_ships = sorted_players[1][1]
            survivor_ships = top_ships - second_ships

            if sorted_players[0][1] == sorted_players[1][1]:
                survivor_ships = 0

            survivor_owner = top_player if survivor_ships > 0 else -1
        else:
            survivor_owner = top_player
            survivor_ships = top_ships

        if survivor_ships > 0:
            if planet[1] == survivor_owner:
                planet[5] += survivor_ships
            else:
                planet[5] -= survivor_ships
                if planet[5] < 0:
                    planet[1] = survivor_owner
                    planet[5] = abs(planet[5])

    for i in range(1, num_agents):
        state[i].observation.planets = obs0.planets
        state[i].observation.initial_planets = obs0.initial_planets
        state[i].observation.fleets = obs0.fleets
        state[i].observation.next_fleet_id = obs0.next_fleet_id
        state[i].observation.comets = obs0.comets
        state[i].observation.comet_planet_ids = obs0.comet_planet_ids

    terminated = False
    step = get(obs0, "step", 0)
    if step >= configuration.episodeSteps - 2:
        terminated = True

    alive_players = set()
    for p in obs0.planets:
        if p[1] != -1:
            alive_players.add(p[1])
    for f in obs0.fleets:
        alive_players.add(f[1])

    if len(alive_players) <= 1:
        terminated = True

    if terminated:
        for s in state:
            s.status = "DONE"

        scores = [0] * num_agents
        for p in obs0.planets:
            if p[1] != -1:
                scores[p[1]] += p[5]
        for f in obs0.fleets:
            scores[f[1]] += f[6]

        max_score = max(scores)
        for i in range(num_agents):
            if scores[i] == max_score and max_score > 0:
                state[i].reward = 1
            else:
                state[i].reward = -1

    return state


def get(d, key, default):
    """Helper to get from dict or SimpleNamespace."""
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def renderer(state, env):
    obs = state[0].observation
    out = f"Step {get(obs, 'step', 0)}\n"
    out += "Planets:\n"
    for p in get(obs, "planets", []):
        out += f"  ID: {p[0]}, Owner: {p[1]}, Pos: ({p[2]:.1f}, {p[3]:.1f}), R: {p[4]:.1f}, Ships: {p[5]}, Prod: {p[6]}\n"
    out += "Fleets:\n"
    for f in get(obs, "fleets", []):
        out += f"  ID: {f[0]}, Owner: {f[1]}, Pos: ({f[2]:.1f}, {f[3]:.1f}), Angle: {f[4]:.2f}, Ships: {f[6]}\n"
    return out


dir_path = path.dirname(__file__)
json_path = path.abspath(path.join(dir_path, "orbit_wars.json"))
with open(json_path) as json_file:
    specification = json.load(json_file)


def html_renderer(env, mode):
    # In ipython/notebook mode, use the lightweight single-file JS renderer
    if mode == "ipython":
        js_path = path.abspath(path.join(dir_path, "orbit_wars.js"))
        if path.exists(js_path):
            with open(js_path, encoding="utf-8") as js_file:
                return js_file.read()
    # Default: use the full Vite-built visualizer
    jspath = path.join(dir_path, "visualizer", "default", "dist", "index.html")
    if path.exists(jspath):
        with open(jspath, encoding="utf-8") as f:
            return f.read()
    # Fallback to single-file JS renderer
    js_path = path.abspath(path.join(dir_path, "orbit_wars.js"))
    if path.exists(js_path):
        with open(js_path, encoding="utf-8") as js_file:
            return js_file.read()
    return ""


def random_agent(obs):
    moves = []
    player = obs.get("player", 0)
    planets = [Planet(*p) for p in obs.get("planets", [])]
    for p in planets:
        if p.owner == player and p.ships > 0:
            angle = random.uniform(0, 2 * math.pi)
            ships = p.ships // 2
            if ships >= 20:
                moves.append([p.id, angle, ships])
    return moves


def starter_agent(obs):
    moves = []
    player = obs.get("player", 0)
    planets = [Planet(*p) for p in obs.get("planets", [])]

    # Find static planets (orbital_radius + planet_radius >= ROTATION_RADIUS_LIMIT)
    static_targets = []
    for p in planets:
        orbital_r = math.sqrt((p.x - CENTER) ** 2 + (p.y - CENTER) ** 2)
        if orbital_r + p.radius >= ROTATION_RADIUS_LIMIT and p.owner != player:
            static_targets.append(p)

    my_planets = [p for p in planets if p.owner == player]
    for mp in my_planets:
        if mp.ships <= 0:
            continue
        # Find closest static planet not owned by us
        closest = None
        min_dist = float("inf")
        for t in static_targets:
            dist = math.sqrt((mp.x - t.x) ** 2 + (mp.y - t.y) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest = t

        if closest:
            angle = math.atan2(closest.y - mp.y, closest.x - mp.x)
            ships = mp.ships // 2
            if ships >= 20:
                moves.append([mp.id, angle, ships])

    return moves


agents = {"random": random_agent, "starter": starter_agent}
