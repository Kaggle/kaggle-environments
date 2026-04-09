import json
import math
from os import path
import random

# Constants
BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
BLACK_HOLE_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def point_to_segment_distance(p, v, w):
    l2 = (v[0] - w[0])**2 + (v[1] - w[1])**2
    if l2 == 0.0:
        return distance(p, v)
    t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
    projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return distance(p, projection)

def generate_planets():
    planets = []
    num_q1 = random.randint(2, 3)
    id_counter = 0
    attempts = 0
    max_attempts = 1000
    
    while len(planets) < 4 or (len(planets) < num_q1 * 4 and attempts < max_attempts):
        attempts += 1
        r = random.randint(1, 5)
        x = random.uniform(CENTER + 15, BOARD_SIZE - r - 5)
        y = random.uniform(CENTER + 15, BOARD_SIZE - r - 5)
        
        if distance((x, y), (CENTER, CENTER)) < BLACK_HOLE_RADIUS + r + 10:
            continue
            
        valid = True
        temp_planets = [
            [id_counter, x, y, r, -1, random.randint(5, 30)],
            [id_counter + 1, BOARD_SIZE - x, y, r, -1, random.randint(5, 30)],
            [id_counter + 2, x, BOARD_SIZE - y, r, -1, random.randint(5, 30)],
            [id_counter + 3, BOARD_SIZE - x, BOARD_SIZE - y, r, -1, random.randint(5, 30)]
        ]
        
        for tp in temp_planets:
            for p in planets:
                if distance((p[1], p[2]), (tp[1], tp[2])) < p[3] + tp[3] + 10:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            planets.extend(temp_planets)
            id_counter += 4
            
    print(f"DEBUG generate_planets: attempts={attempts}, generated={len(planets)}")
    return planets

def interpreter(state, env):
    configuration = env.configuration
    num_agents = len(state)
    obs0 = state[0].observation

    if env.done:
        return state

    # Initialize game state if not already done
    # Safer to check if planets exist rather than step == 0
    if not hasattr(obs0, "planets") or not obs0.planets:
        print("DEBUG interpreter: Initializing state")
        angular_velocity = random.uniform(0.025, 0.05)
        obs0.angular_velocity = angular_velocity
        obs0.planets = generate_planets()
        obs0.initial_planets = [p.copy() for p in obs0.planets]
        obs0.fleets = []
        obs0.next_fleet_id = 0
        
        # Assign home planets
        if len(obs0.planets) >= 4:
            if num_agents == 2:
                obs0.planets[0][4] = 0 # Q1
                obs0.planets[0][5] = 10
                obs0.planets[3][4] = 1 # Q4
                obs0.planets[3][5] = 10
            elif num_agents == 4:
                obs0.planets[0][4] = 0 # Q1
                obs0.planets[0][5] = 10
                obs0.planets[1][4] = 1 # Q2
                obs0.planets[1][5] = 10
                obs0.planets[2][4] = 2 # Q3
                obs0.planets[2][5] = 10
                obs0.planets[3][4] = 3 # Q4
                obs0.planets[3][5] = 10
                
        for i in range(num_agents):
            state[i].observation.player = i
            if i > 0:
                state[i].observation.angular_velocity = obs0.angular_velocity
                state[i].observation.planets = obs0.planets
                state[i].observation.initial_planets = obs0.initial_planets
                state[i].observation.fleets = obs0.fleets
                state[i].observation.next_fleet_id = obs0.next_fleet_id
                
        return state

    # 0. Fleet Launch
    def process_moves(player_id, action):
        if not action or not isinstance(action, list):
            return
        for move in action:
            if len(move) != 3:
                continue
            from_id, angle, ships = move
            
            from_planet = next((p for p in obs0.planets if p[0] == from_id), None)
            
            if from_planet and from_planet[4] == player_id:
                if from_planet[5] >= ships and ships > 0:
                    from_planet[5] -= ships
                    obs0.fleets.append([
                        obs0.next_fleet_id,
                        player_id,
                        from_id,
                        angle,
                        from_planet[1],
                        from_planet[2],
                        ships
                    ])
                    obs0.next_fleet_id += 1

    for i in range(num_agents):
        process_moves(i, state[i].action)

    # 1. Production
    for planet in obs0.planets:
        if planet[4] != -1:
            planet[5] += planet[3]

    # 2. Fleet Movement
    ship_speed = configuration.shipSpeed
    fleets_to_remove = []
    combat_lists = {p[0]: [] for p in obs0.planets}
    
    for fleet in obs0.fleets:
        angle = fleet[3]
        fleet[4] += math.cos(angle) * ship_speed
        fleet[5] += math.sin(angle) * ship_speed
        
        if not (0 <= fleet[4] <= BOARD_SIZE and 0 <= fleet[5] <= BOARD_SIZE):
            fleets_to_remove.append(fleet)
            continue
            
        if distance((fleet[4], fleet[5]), (CENTER, CENTER)) < BLACK_HOLE_RADIUS:
            fleets_to_remove.append(fleet)
            continue
            
        for planet in obs0.planets:
            dist = distance((fleet[4], fleet[5]), (planet[1], planet[2]))
            if dist < planet[3]:
                combat_lists[planet[0]].append(fleet)
                fleets_to_remove.append(fleet)
                break

    # 3. Planet Movement & Sweep
    angular_velocity = obs0.angular_velocity
    step = get(obs0, "step", 1) # Fallback if step not present
    
    for i, planet in enumerate(obs0.planets):
        initial_p = obs0.initial_planets[i]
        dx = initial_p[1] - CENTER
        dy = initial_p[2] - CENTER
        r = math.sqrt(dx**2 + dy**2)
        old_pos = (planet[1], planet[2])
        
        if r < ROTATION_RADIUS_LIMIT:
            initial_angle = math.atan2(dy, dx)
            current_angle = initial_angle + angular_velocity * step
            planet[1] = CENTER + r * math.cos(current_angle)
            planet[2] = CENTER + r * math.sin(current_angle)
        
        new_pos = (planet[1], planet[2])
        
        if old_pos != new_pos:
            for fleet in obs0.fleets:
                if fleet not in fleets_to_remove:
                    if point_to_segment_distance((fleet[4], fleet[5]), old_pos, new_pos) < planet[3]:
                        combat_lists[planet[0]].append(fleet)
                        fleets_to_remove.append(fleet)

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
            
        sorted_players = sorted(player_ships.items(), key=lambda item: item[1], reverse=True)
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
            if planet[4] == survivor_owner:
                planet[5] += survivor_ships
            else:
                planet[5] -= survivor_ships
                if planet[5] < 0:
                    planet[4] = survivor_owner
                    planet[5] = abs(planet[5])


    for i in range(1, num_agents):
        state[i].observation.planets = obs0.planets
        state[i].observation.fleets = obs0.fleets
        state[i].observation.next_fleet_id = obs0.next_fleet_id

    terminated = False
    # Use fallback if step not present
    step = get(obs0, "step", 0)
    if step >= configuration.episodeSteps - 1:
        terminated = True
        
    alive_players = set()
    for p in obs0.planets:
        if p[4] != -1:
            alive_players.add(p[4])
    for f in obs0.fleets:
        alive_players.add(f[1])
        
    if len(alive_players) <= 1:
        terminated = True
        
    if terminated:
        for s in state:
            s.status = "DONE"
        
        scores = [0] * num_agents
        for p in obs0.planets:
            if p[4] != -1:
                scores[p[4]] += p[5]
        for f in obs0.fleets:
            scores[f[1]] += f[6]
            
        for i in range(num_agents):
            state[i].reward = scores[i]

    return state

def get(d, key, default):
    # Helper to get from dict or SimpleNamespace
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)

def renderer(state, env):
    obs = state[0].observation
    out = f"Step {get(obs, 'step', 0)}\n"
    out += "Planets:\n"
    for p in get(obs, "planets", []):
        out += f"  ID: {p[0]}, Pos: ({p[1]:.1f}, {p[2]:.1f}), R: {p[3]}, Owner: {p[4]}, Ships: {p[5]}\n"
    out += "Fleets:\n"
    for f in get(obs, "fleets", []):
        out += f"  ID: {f[0]}, Owner: {f[1]}, Angle: {f[3]:.2f}, Pos: ({f[4]:.1f}, {f[5]:.1f}), Ships: {f[6]}\n"
    return out

dir_path = path.dirname(__file__)
json_path = path.abspath(path.join(dir_path, "orbit_wars.json"))
with open(json_path) as json_file:
    specification = json.load(json_file)

def html_renderer():
    js_path = path.abspath(path.join(dir_path, "orbit_wars.js"))
    if path.exists(js_path):
        with open(js_path, encoding="utf-8") as js_file:
            return js_file.read()
    return ""

def random_agent(obs):
    moves = []
    my_planets = [p for p in get(obs, "planets", []) if p[4] == get(obs, "player", 0)]
    for mp in my_planets:
        if mp[5] > 0:
            angle = random.uniform(0, 2 * math.pi)
            ships = mp[5] // 2
            if ships > 0:
                moves.append([mp[0], angle, ships])
    return moves

def starter_agent(obs):
    moves = []
    my_planets = [p for p in get(obs, "planets", []) if p[4] == get(obs, "player", 0)]
    all_planets = get(obs, "planets", [])
    
    # Find static planets (distance to center >= ROTATION_RADIUS_LIMIT)
    static_planets = []
    for p in all_planets:
        dx = p[1] - CENTER
        dy = p[2] - CENTER
        r = math.sqrt(dx**2 + dy**2)
        if r >= ROTATION_RADIUS_LIMIT:
            static_planets.append(p)
            
    for mp in my_planets:
        if mp[5] > 0:
            # Find closest static planet
            closest_target = None
            min_dist = float('inf')
            for sp in static_planets:
                if sp[0] != mp[0]: # Don't target self
                    dist = math.sqrt((mp[1] - sp[1])**2 + (mp[2] - sp[2])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_target = sp
                        
            if closest_target:
                # Calculate angle
                dx = closest_target[1] - mp[1]
                dy = closest_target[2] - mp[2]
                angle = math.atan2(dy, dx)
                ships = mp[5] // 2
                if ships > 0:
                    moves.append([mp[0], angle, ships])
                    
    return moves

agents = {"random": random_agent, "starter": starter_agent}
