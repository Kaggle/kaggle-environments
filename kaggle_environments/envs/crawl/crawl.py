import json
from collections import defaultdict
from os import path
from random import Random

from .agents import agents  # noqa: F401

# --- Constants ---

FACTORY, SCOUT, WORKER, MINER = 0, 1, 2, 3
WALL_N, WALL_E, WALL_S, WALL_W = 1, 2, 4, 8

DIR_OFFSETS = {
    "NORTH": (0, 1),
    "SOUTH": (0, -1),
    "EAST": (1, 0),
    "WEST": (-1, 0),
}
OPPOSITE_DIR = {"NORTH": "SOUTH", "SOUTH": "NORTH", "EAST": "WEST", "WEST": "EAST"}
DIR_WALL_BIT = {"NORTH": WALL_N, "EAST": WALL_E, "SOUTH": WALL_S, "WEST": WALL_W}

MOVE_DIRS = {"NORTH", "SOUTH", "EAST", "WEST"}
JUMP_ACTIONS = {"JUMP_NORTH", "JUMP_SOUTH", "JUMP_EAST", "JUMP_WEST"}
WALL_BUILD_ACTIONS = {"BUILD_NORTH", "BUILD_SOUTH", "BUILD_EAST", "BUILD_WEST"}
WALL_REMOVE_ACTIONS = {"REMOVE_NORTH", "REMOVE_SOUTH", "REMOVE_EAST", "REMOVE_WEST"}
TRANSFER_ACTIONS = {
    "TRANSFER_NORTH",
    "TRANSFER_SOUTH",
    "TRANSFER_EAST",
    "TRANSFER_WEST",
}
FACTORY_BUILD_ACTIONS = {"BUILD_SCOUT", "BUILD_WORKER", "BUILD_MINER"}


def is_fixed_wall(col, direction, width):
    """Walls workers cannot build or remove: E/W perimeter and the central mirror axis."""
    if direction == "WEST" and col == 0:
        return True
    if direction == "EAST" and col == width - 1:
        return True
    half = width // 2
    if direction == "EAST" and col == half - 1:
        return True
    if direction == "WEST" and col == half:
        return True
    return False

# Crush table: type that wins against another type
# Factory crushes all non-factory units (and is indestructible against them);
# Miner crushes Worker and Scout; Worker crushes Scout.
CRUSHES = {
    (FACTORY, MINER): True,
    (FACTORY, WORKER): True,
    (FACTORY, SCOUT): True,
    (MINER, WORKER): True,
    (MINER, SCOUT): True,
    (WORKER, SCOUT): True,
}

TYPE_NAMES = {FACTORY: "Factory", SCOUT: "Scout", WORKER: "Worker", MINER: "Miner"}


# --- Maze Generation (Eller's algorithm, left half mirrored) ---


def generate_maze_row(rng, width, eller_state, door_probability):
    """Generate one maze row using Eller's algorithm on the left half, then mirror."""
    half = width // 2
    sets = list(eller_state["sets"])
    next_id = eller_state["next_set_id"]

    # Initialize unassigned columns
    for i in range(half):
        if sets[i] == 0:
            sets[i] = next_id
            next_id += 1

    # Left half walls: start with all walls
    left_walls = [WALL_N | WALL_E | WALL_S | WALL_W] * half

    # Horizontal merging: randomly remove E/W walls between adjacent cells in same row
    for c in range(half - 1):
        if sets[c] != sets[c + 1] and rng.random() < 0.5:
            # Merge: remove E wall from c and W wall from c+1
            left_walls[c] &= ~WALL_E
            left_walls[c + 1] &= ~WALL_W
            # Unify sets
            old_set = sets[c + 1]
            new_set = sets[c]
            for i in range(half):
                if sets[i] == old_set:
                    sets[i] = new_set

    # Vertical passages: for each set, ensure at least one cell has a south passage
    set_cells = defaultdict(list)
    for i in range(half):
        set_cells[sets[i]].append(i)

    next_row_sets = [0] * half
    for set_id, cells in set_cells.items():
        # Randomly assign south passages (prob 0.4), ensure at least one
        passages = [c for c in cells if rng.random() < 0.4]
        if not passages:
            passages = [rng.choice(cells)]
        for c in passages:
            left_walls[c] &= ~WALL_S  # Remove south wall
            next_row_sets[c] = set_id  # Keep set membership

    # Build full row by mirroring
    row_walls = [0] * width

    # Left half
    for c in range(half):
        row_walls[c] = left_walls[c]

    # Boundary: left edge always has west wall
    row_walls[0] |= WALL_W

    # Right half mirrors left half, swapping E and W
    for c in range(half):
        mirror_c = width - 1 - c
        w = left_walls[c]
        mirrored = 0
        if w & WALL_N:
            mirrored |= WALL_N
        if w & WALL_S:
            mirrored |= WALL_S
        if w & WALL_E:
            mirrored |= WALL_W  # E becomes W
        if w & WALL_W:
            mirrored |= WALL_E  # W becomes E
        row_walls[mirror_c] = mirrored

    # Boundary: right edge always has east wall
    row_walls[width - 1] |= WALL_E

    # Center wall between half-1 and half (with occasional doors)
    if rng.random() >= door_probability:
        row_walls[half - 1] |= WALL_E
        row_walls[half] |= WALL_W
    else:
        row_walls[half - 1] &= ~WALL_E
        row_walls[half] &= ~WALL_W

    # Mirror next_row_sets for the right half
    full_next_sets = [0] * width
    for c in range(half):
        full_next_sets[c] = next_row_sets[c]
    for c in range(half):
        mirror_c = width - 1 - c
        # Right half cells that had passages keep mirrored set ids
        if next_row_sets[c] != 0:
            full_next_sets[mirror_c] = next_row_sets[c] + 1000000  # Offset to avoid collision
        else:
            full_next_sets[mirror_c] = 0

    eller_state["sets"] = full_next_sets[:half]  # Only track left half
    eller_state["next_set_id"] = next_id

    return row_walls


def ensure_wall_consistency(walls_dict, row_num, width):
    """Ensure north/south wall consistency between adjacent rows."""
    row_key = str(row_num)
    prev_key = str(row_num - 1)

    if row_key not in walls_dict:
        return

    row = walls_dict[row_key]

    # If previous row exists, sync south walls
    if prev_key in walls_dict:
        prev_row = walls_dict[prev_key]
        for c in range(width):
            # If current row has south wall, previous row must have north wall
            if row[c] & WALL_S:
                prev_row[c] |= WALL_N
            else:
                prev_row[c] &= ~WALL_N
            # If previous row has north wall, current row must have south wall
            if prev_row[c] & WALL_N:
                row[c] |= WALL_S
            else:
                row[c] &= ~WALL_S


def place_crystals(rng, width, crystals, row_num, density, min_e, max_e):
    """Place crystals on a row. Symmetric: left half mirrored to right."""
    half = width // 2
    for c in range(half):
        if rng.random() < density:
            energy = rng.randint(min_e, max_e)
            crystals[f"{c},{row_num}"] = energy
            mirror_c = width - 1 - c
            if mirror_c != c:
                crystals[f"{mirror_c},{row_num}"] = energy


def place_mining_nodes(rng, width, nodes, row_num, density, crystals):
    """Place mining nodes on a row. Symmetric: left half mirrored to right. Avoids crystal cells."""
    half = width // 2
    for c in range(half):
        if rng.random() < density:
            key = f"{c},{row_num}"
            mirror_c = width - 1 - c
            mirror_key = f"{mirror_c},{row_num}"
            if key in crystals or mirror_key in crystals:
                continue
            nodes[key] = 1
            if mirror_c != c:
                nodes[mirror_key] = 1


# --- Helper functions ---


def get_robot_max_energy(robot_type, config):
    if robot_type == FACTORY:
        return float("inf")
    elif robot_type == SCOUT:
        return config.scoutMaxEnergy
    elif robot_type == WORKER:
        return config.workerMaxEnergy
    elif robot_type == MINER:
        return config.minerMaxEnergy
    return 0


def get_move_period(robot_type, config):
    if robot_type == FACTORY:
        return config.factoryMovePeriod
    elif robot_type == SCOUT:
        return 1
    elif robot_type == WORKER:
        return config.workerMovePeriod
    elif robot_type == MINER:
        return config.minerMovePeriod
    return 1


def get_vision(robot_type, config):
    if robot_type == FACTORY:
        return config.visionFactory
    elif robot_type == SCOUT:
        return config.visionScout
    elif robot_type == WORKER:
        return config.visionWorker
    elif robot_type == MINER:
        return config.visionMiner
    return 0


def robot_to_list(r):
    return [r["type"], r["col"], r["row"], r["energy"], r["owner"], r["move_cooldown"], r["jump_cooldown"], r["build_cooldown"]]


def list_to_robot(uid, lst):
    return {
        "uid": uid,
        "type": lst[0],
        "col": lst[1],
        "row": lst[2],
        "energy": lst[3],
        "owner": lst[4],
        "move_cooldown": lst[5],
        "jump_cooldown": lst[6],
        "build_cooldown": lst[7] if len(lst) > 7 else 0,
    }


def create_uid(obs):
    uid = f"{obs.step}-{obs.nextUid}"
    obs.nextUid = obs.nextUid + 1
    return uid


def can_move_through(walls_dict, width, col, row, direction):
    """Check if movement from (col, row) in direction is unblocked by walls."""
    row_key = str(row)
    if row_key not in walls_dict:
        return False
    row_walls = walls_dict[row_key]
    if col < 0 or col >= width:
        return False
    if row_walls[col] & DIR_WALL_BIT[direction]:
        return False
    dc, dr = DIR_OFFSETS[direction]
    nc, nr = col + dc, row + dr
    if nc < 0 or nc >= width:
        return False
    nr_key = str(nr)
    if nr_key not in walls_dict:
        return False
    return True


def get_visible_cells(robots, config):
    """Return set of (col, row) tuples visible to a player's robots."""
    visible = set()
    for r in robots:
        v = get_vision(r["type"], config)
        rc, rr = r["col"], r["row"]
        for dc in range(-v, v + 1):
            for dr in range(-v, v + 1):
                if abs(dc) + abs(dr) <= v:
                    c = rc + dc
                    if 0 <= c < config.width:
                        visible.add((c, rr + dr))
    return visible


def get_scroll_interval(step, config):
    if step >= config.scrollRampSteps:
        return config.scrollEndInterval
    progress = step / max(1, config.scrollRampSteps)
    interval = config.scrollStartInterval - (config.scrollStartInterval - config.scrollEndInterval) * progress
    return max(config.scrollEndInterval, round(interval))


def _resolve_tiebreak(robots):
    """Cascade: total energy → unit count → 0.5/0.5 draw. Returns (reward_0, reward_1)."""
    energy = [0, 0]
    units = [0, 0]
    for r in robots.values():
        energy[r["owner"]] += r["energy"]
        units[r["owner"]] += 1
    if energy[0] != energy[1]:
        return (1, 0) if energy[0] > energy[1] else (0, 1)
    if units[0] != units[1]:
        return (1, 0) if units[0] > units[1] else (0, 1)
    return (0.5, 0.5)


# --- Initialization ---


def initialize_game(state, env):
    """Set up the initial game state."""
    config = env.configuration
    obs = state[0].observation
    width = config.width
    height = config.height

    # Resolve the episode seed and stash it on env.info so it persists into
    # the replay (via toJSON) but stays out of `configuration`, which agents
    # can read. The seed determines maze layout and scroll-time row
    # generation — both hidden info that agents must not be able to predict.
    if not hasattr(env, "info") or env.info is None:
        env.info = {}
    seed = env.info.get("seed")
    if seed is None:
        seed = getattr(config, "randomSeed", None)
        if seed is None and isinstance(config, dict):
            seed = config.get("randomSeed")
    if seed is None:
        import time

        seed = int(time.time() * 1000) % (2**31)
    # Scrub the seed from configuration so agents can't read it.
    try:
        config.randomSeed = None
    except (AttributeError, TypeError):
        config["randomSeed"] = None
    env.info["seed"] = seed
    rng = Random(seed)

    # Initialize hidden state
    obs.nextUid = 0
    obs.scrollCounter = config.scrollStartInterval
    obs.globalWalls = {}
    obs.globalCrystals = {}
    obs.globalRobots = {}
    obs.globalMines = {}
    obs.globalMiningNodes = {}
    obs.southBound = 0
    obs.northBound = height - 1

    # Eller state for maze generation
    eller_state = {"sets": [0] * (width // 2), "next_set_id": 1}

    # Generate initial maze rows
    for row_num in range(height):
        row_walls = generate_maze_row(rng, width, eller_state, config.doorProbability)
        obs.globalWalls[str(row_num)] = row_walls
        if row_num > 0:
            ensure_wall_consistency(obs.globalWalls, row_num, width)
        place_crystals(
            rng,
            width,
            obs.globalCrystals,
            row_num,
            config.crystalDensity,
            config.crystalMinEnergy,
            config.crystalMaxEnergy,
        )
        place_mining_nodes(rng, width, obs.globalMiningNodes, row_num, config.miningNodeDensity, obs.globalCrystals)

    # First row: add south wall to all cells (boundary)
    for c in range(width):
        obs.globalWalls["0"][c] |= WALL_S

    obs.ellerState = eller_state

    # Place factories symmetrically
    p0_col = width // 4
    p1_col = width - 1 - p0_col
    factory_row = 2

    for player_idx, col in enumerate([p0_col, p1_col]):
        uid = create_uid(obs)
        obs.globalRobots[uid] = robot_to_list(
            {
                "type": FACTORY,
                "col": col,
                "row": factory_row,
                "energy": config.factoryEnergy,
                "owner": player_idx,
                "move_cooldown": 0,
                "jump_cooldown": 0,
                "build_cooldown": 0,
            }
        )

    # Initialize discovered cells per player
    obs.discoveredCells = [[], []]
    obs.discoveredMines = [[], []]

    # Compute initial vision
    for player_idx in range(2):
        player_robots = [list_to_robot(uid, data) for uid, data in obs.globalRobots.items() if data[4] == player_idx]
        visible = get_visible_cells(player_robots, config)
        obs.discoveredCells[player_idx] = [list(cell) for cell in visible]

    # Build per-player observations
    _update_player_observations(state, env)

    return state


# --- Observation building ---


def _update_player_observations(state, env):
    """Build per-player observations from global state."""
    config = env.configuration
    obs = state[0].observation
    width = config.width
    south = obs.southBound
    north = obs.northBound
    window_height = north - south + 1

    for player_idx in range(2):
        # Get this player's robots
        player_robots = [list_to_robot(uid, data) for uid, data in obs.globalRobots.items() if data[4] == player_idx]

        # Compute current vision
        visible = get_visible_cells(player_robots, config)

        # Update discovered cells
        discovered = set()
        for cell in obs.discoveredCells[player_idx]:
            discovered.add((cell[0], cell[1]))
        discovered.update(visible)
        # Prune cells below south bound
        discovered = {(c, r) for c, r in discovered if r >= south}
        obs.discoveredCells[player_idx] = [list(cell) for cell in discovered]

        # Build walls array
        walls_array = [-1] * (window_height * width)
        for c, r in discovered:
            if south <= r <= north:
                idx = (r - south) * width + c
                row_key = str(r)
                if row_key in obs.globalWalls and 0 <= c < width:
                    walls_array[idx] = obs.globalWalls[row_key][c]

        # Build crystals (only currently visible)
        crystals = {}
        for c, r in visible:
            key = f"{c},{r}"
            if key in obs.globalCrystals:
                crystals[key] = obs.globalCrystals[key]

        # Build robots (own always visible, enemy only if in vision)
        robots = {}
        for uid, data in obs.globalRobots.items():
            col, row, owner = data[1], data[2], data[4]
            if owner == player_idx or (col, row) in visible:
                robots[uid] = list(data)

        # Build mines (discovered mines are remembered)
        # Update discovered mines with newly visible ones
        disc_mines = set()
        for key in obs.discoveredMines[player_idx]:
            disc_mines.add(key)
        for c, r in visible:
            key = f"{c},{r}"
            if key in obs.globalMines:
                disc_mines.add(key)
        # Remove mines that no longer exist
        disc_mines = {k for k in disc_mines if k in obs.globalMines}
        obs.discoveredMines[player_idx] = list(disc_mines)

        mines = {}
        for key in disc_mines:
            if key in obs.globalMines:
                mines[key] = list(obs.globalMines[key])

        # Build mining nodes (visible only, like crystals)
        mining_nodes = {}
        for c, r in visible:
            key = f"{c},{r}"
            if key in (obs.globalMiningNodes or {}):
                mining_nodes[key] = 1

        # Set per-player observation
        state[player_idx].observation.walls = walls_array
        state[player_idx].observation.crystals = crystals
        state[player_idx].observation.robots = robots
        state[player_idx].observation.mines = mines
        state[player_idx].observation.miningNodes = mining_nodes


# --- Interpreter ---


def interpreter(state, env):
    obs = state[0].observation
    config = env.configuration

    if env.done:
        return initialize_game(state, env)

    # Deserialize robots from global state
    robots = {}
    for uid, data in obs.globalRobots.items():
        robots[uid] = list_to_robot(uid, data)

    walls = obs.globalWalls
    crystals = obs.globalCrystals if obs.globalCrystals else {}
    mines = obs.globalMines if obs.globalMines else {}
    mining_nodes = obs.globalMiningNodes if obs.globalMiningNodes else {}
    width = config.width
    south = obs.southBound
    north = obs.northBound

    # Collect actions from both players
    actions = {}
    for player_idx in range(2):
        player_actions = state[player_idx].action
        if player_actions and isinstance(player_actions, dict):
            actions.update(player_actions)

    # --- Phase 0: Cooldown tick ---
    for uid, r in robots.items():
        if r["move_cooldown"] > 0:
            r["move_cooldown"] -= 1
        if r["jump_cooldown"] > 0:
            r["jump_cooldown"] -= 1
        if r["build_cooldown"] > 0:
            r["build_cooldown"] -= 1

    # --- Phase 1: Action validation ---
    validated_actions = {}
    for uid, action in actions.items():
        if uid not in robots:
            continue
        r = robots[uid]
        if not isinstance(action, str):
            continue

        valid = False
        rtype = r["type"]

        if action == "IDLE":
            valid = True
        elif action in MOVE_DIRS:
            valid = True  # Movement validity checked later with cooldowns/walls
        elif action in JUMP_ACTIONS:
            valid = rtype == FACTORY
        elif action in WALL_BUILD_ACTIONS or action in WALL_REMOVE_ACTIONS:
            valid = rtype == WORKER
        elif action == "TRANSFORM":
            valid = rtype == MINER
        elif action in FACTORY_BUILD_ACTIONS:
            valid = rtype == FACTORY
        elif action in TRANSFER_ACTIONS:
            valid = True

        if valid:
            validated_actions[uid] = action
        else:
            validated_actions[uid] = "IDLE"

    # UIDs not in actions default to IDLE
    for uid in robots:
        if uid not in validated_actions:
            validated_actions[uid] = "IDLE"

    # --- Phase 2: Energy consumption ---
    energy_depleted = set()
    for uid, r in robots.items():
        r["energy"] -= config.energyPerTurn
        if r["energy"] < 0:
            r["energy"] = 0
        if r["energy"] == 0:
            energy_depleted.add(uid)

    # Robots with no energy can't act (forced IDLE)
    for uid in energy_depleted:
        validated_actions[uid] = "IDLE"

    # --- Phase 3: Special actions ---
    destroyed = set()

    # 3a: TRANSFORM (Miner -> Mine, requires mining node)
    for uid in list(robots.keys()):
        if uid in destroyed:
            continue
        if validated_actions.get(uid) != "TRANSFORM":
            continue
        r = robots[uid]
        key = f"{r['col']},{r['row']}"
        if key not in mining_nodes:
            validated_actions[uid] = "IDLE"
            continue
        if r["energy"] < config.transformCost:
            validated_actions[uid] = "IDLE"
            continue
        mine_energy = min(r["energy"] - config.transformCost, config.mineMaxEnergy)
        mines[key] = [mine_energy, config.mineMaxEnergy, r["owner"], config.mineRate]
        # Remove the mining node (consumed)
        del mining_nodes[key]
        destroyed.add(uid)

    # 3b: BUILD_DIR / REMOVE_DIR (Worker toggles wall in direction)
    # Worker survives. Costs `wallBuildCost` (BUILD_*) or `wallRemoveCost`
    # (REMOVE_*) regardless of effect (no-op if the wall already exists for
    # BUILD or doesn't exist for REMOVE, or if the wall is fixed). Out-of-bounds
    # neighbor (off the map) is also a no-op but still charges. Insufficient
    # energy → IDLE, no charge.
    for uid in list(robots.keys()):
        if uid in destroyed:
            continue
        action = validated_actions.get(uid, "IDLE")
        is_build = action in WALL_BUILD_ACTIONS
        is_remove = action in WALL_REMOVE_ACTIONS
        if not (is_build or is_remove):
            continue
        r = robots[uid]
        cost = config.wallBuildCost if is_build else config.wallRemoveCost
        if r["energy"] < cost:
            validated_actions[uid] = "IDLE"
            continue
        r["energy"] -= cost
        direction = action.split("_")[1]
        col, row = r["col"], r["row"]
        row_key = str(row)
        if is_fixed_wall(col, direction, width):
            continue  # charged, no effect
        bit = DIR_WALL_BIT[direction]
        opp_bit = DIR_WALL_BIT[OPPOSITE_DIR[direction]]
        dc, dr = DIR_OFFSETS[direction]
        nc, nr = col + dc, row + dr
        nr_key = str(nr)
        neighbor_in_map = 0 <= nc < width and nr_key in walls
        if is_build:
            if row_key in walls:
                walls[row_key][col] |= bit
            if neighbor_in_map:
                walls[nr_key][nc] |= opp_bit
        else:  # is_remove
            if row_key in walls:
                walls[row_key][col] &= ~bit
            if neighbor_in_map:
                walls[nr_key][nc] &= ~opp_bit

    # 3d: BUILD (Factory spawns robot to the north, combat resolves in Phase 4)
    for uid in list(robots.keys()):
        if uid in destroyed:
            continue
        action = validated_actions.get(uid, "IDLE")
        if action not in FACTORY_BUILD_ACTIONS:
            continue
        r = robots[uid]

        if action == "BUILD_SCOUT":
            cost = config.scoutCost
            new_type = SCOUT
            new_energy = config.scoutCost
        elif action == "BUILD_WORKER":
            cost = config.workerCost
            new_type = WORKER
            new_energy = config.workerCost
        elif action == "BUILD_MINER":
            cost = config.minerCost
            new_type = MINER
            new_energy = config.minerCost
        else:
            continue

        if r["energy"] < cost:
            validated_actions[uid] = "IDLE"
            continue
        if r["build_cooldown"] > 0:
            validated_actions[uid] = "IDLE"
            continue

        # Spawn cell is always north of factory
        sc, sr = r["col"], r["row"] + 1

        # Check wall between factory and spawn cell
        if not can_move_through(walls, width, r["col"], r["row"], "NORTH"):
            validated_actions[uid] = "IDLE"
            continue
        if sr > north:
            validated_actions[uid] = "IDLE"
            continue

        r["energy"] -= cost
        r["build_cooldown"] = config.factoryBuildCooldown
        new_uid = create_uid(obs)
        new_period = get_move_period(new_type, config)
        robots[new_uid] = {
            "uid": new_uid,
            "type": new_type,
            "col": sc,
            "row": sr,
            "energy": new_energy,
            "owner": r["owner"],
            "move_cooldown": new_period - 1,
            "jump_cooldown": 0,
            "build_cooldown": 0,
        }
        validated_actions[new_uid] = "IDLE"

    # 3e: TRANSFER
    for uid in list(robots.keys()):
        if uid in destroyed:
            continue
        action = validated_actions.get(uid, "IDLE")
        if action not in TRANSFER_ACTIONS:
            continue
        r = robots[uid]
        direction = action.split("_")[1]
        if not can_move_through(walls, width, r["col"], r["row"], direction):
            continue
        dc, dr = DIR_OFFSETS[direction]
        tc, tr = r["col"] + dc, r["row"] + dr
        # Find friendly robot at target
        target_uid = None
        for tuid, tr_robot in robots.items():
            if tuid in destroyed:
                continue
            if tr_robot["col"] == tc and tr_robot["row"] == tr and tr_robot["owner"] == r["owner"] and tuid != uid:
                target_uid = tuid
                break
        if target_uid is None:
            continue
        target = robots[target_uid]
        max_e = get_robot_max_energy(target["type"], config)
        transfer_amount = r["energy"]
        space = max_e - target["energy"]
        if space != float("inf"):
            transfer_amount = min(transfer_amount, max(0, int(space)))
        target["energy"] += transfer_amount
        r["energy"] -= transfer_amount

    # Remove destroyed robots
    for uid in destroyed:
        del robots[uid]

    # --- Phase 4: Movement + combat resolution ---
    movements = {}  # uid -> (target_col, target_row)
    stationary_uids = set()
    off_board_destroyed = set()  # units that walked/jumped off the N/S edge

    for uid, r in robots.items():
        action = validated_actions.get(uid, "IDLE")

        if action in MOVE_DIRS:
            if r["move_cooldown"] > 0:
                stationary_uids.add(uid)
                continue
            direction = action
            dc, dr_off = DIR_OFFSETS[direction]
            tc, tr = r["col"] + dc, r["row"] + dr_off
            if not (south <= tr <= north):
                # Off-board N/S move: only the source cell's wall can block.
                # (can_move_through also requires the neighbor row to exist,
                # which is false off the edge.) E/W is impossible here because
                # perimeter walls always block.
                row_key = str(r["row"])
                source_wall = walls.get(row_key, [0] * width)[r["col"]] if 0 <= r["col"] < width else 0
                if source_wall & DIR_WALL_BIT[direction]:
                    stationary_uids.add(uid)
                else:
                    off_board_destroyed.add(uid)
                continue
            if can_move_through(walls, width, r["col"], r["row"], direction):
                movements[uid] = (tc, tr)
            else:
                stationary_uids.add(uid)

        elif action in JUMP_ACTIONS:
            if r["move_cooldown"] > 0 or r["jump_cooldown"] > 0:
                stationary_uids.add(uid)
                continue
            direction = action.split("_")[1]
            dc, dr_off = DIR_OFFSETS[direction]
            tc, tr = r["col"] + dc * 2, r["row"] + dr_off * 2
            # Jump always happens (no wall check). Off-board landing kills the
            # factory; cooldown is consumed either way.
            r["jump_cooldown"] = config.factoryJumpCooldown
            if 0 <= tc < width and south <= tr <= north and str(tr) in walls:
                movements[uid] = (tc, tr)
            else:
                off_board_destroyed.add(uid)
        else:
            stationary_uids.add(uid)

    # Remove off-board units before combat resolution.
    for uid in off_board_destroyed:
        if uid in robots:
            del robots[uid]

    # Build position map for stationary robots
    position_map = defaultdict(list)  # (col, row) -> [uid, ...]
    for uid in stationary_uids:
        if uid in robots:
            r = robots[uid]
            position_map[(r["col"], r["row"])].append(uid)

    # Group movements by target cell; also include cells with multiple
    # stationary robots (e.g. from spawn collisions) so combat resolves.
    target_groups = defaultdict(list)  # (col, row) -> [mover_uid, ...]
    for uid, (tc, tr) in movements.items():
        target_groups[(tc, tr)].append(uid)
    for pos, uids in position_map.items():
        if len(uids) > 1 and pos not in target_groups:
            target_groups[pos] = []  # no movers, but multiple occupants

    move_destroyed = set()
    moved = set()
    combat_cells = set()  # cells where a multi-robot collision happened

    for target, mover_uids in target_groups.items():
        occupant_uids = position_map.get(target, [])
        all_uids = mover_uids + occupant_uids

        if len(all_uids) <= 1:
            if mover_uids:
                moved.add(mover_uids[0])
            continue

        # Multi-robot collision: apply crush rules. Friendly fire is real —
        # ownership doesn't matter. Factories are indestructible vs anything
        # except an enemy factory (mutual destruction).
        combat_cells.add(target)
        survivors = set()
        destroyed_in_combat = set()
        all_types = [(uid, robots[uid]["type"]) for uid in all_uids]

        factory_uids = [uid for uid, rtype in all_types if rtype == FACTORY]
        factory_owners = {robots[uid]["owner"] for uid in factory_uids}
        factories_mutual = len(factory_owners) > 1

        for uid, rtype in all_types:
            if rtype == FACTORY:
                if factories_mutual:
                    destroyed_in_combat.add(uid)
                else:
                    survivors.add(uid)
                continue
            crushed = False
            for other_uid, other_type in all_types:
                if other_uid == uid:
                    continue
                if (other_type, rtype) in CRUSHES:
                    crushed = True
                    break
                elif other_type == rtype:
                    # Same type → both destroyed (mutual), regardless of owner.
                    crushed = True
                    destroyed_in_combat.add(other_uid)
                    break
            if crushed:
                destroyed_in_combat.add(uid)
            else:
                survivors.add(uid)

        move_destroyed.update(destroyed_in_combat)
        for uid in mover_uids:
            if uid in survivors:
                moved.add(uid)

    # Apply movements
    for uid in moved:
        if uid in move_destroyed:
            continue
        if uid in movements:
            tc, tr = movements[uid]
            robots[uid]["col"] = tc
            robots[uid]["row"] = tr
            period = get_move_period(robots[uid]["type"], config)
            robots[uid]["move_cooldown"] = period - 1

    # Remove combat casualties (factories only die via mutual factory destruction)
    for uid in move_destroyed:
        if uid in robots:
            del robots[uid]

    # Consume crystals at combat cells where no robot survived.
    # (If a robot did survive, Phase 5 will hand it the crystal energy.)
    for (cc, cr_) in combat_cells:
        ckey = f"{cc},{cr_}"
        if ckey not in crystals:
            continue
        survivor_here = any(r["col"] == cc and r["row"] == cr_ for r in robots.values())
        if not survivor_here:
            del crystals[ckey]

    # --- Phase 5: Crystal collection ---
    crystals_to_remove = []
    for uid, r in robots.items():
        key = f"{r['col']},{r['row']}"
        if key in crystals:
            max_e = get_robot_max_energy(r["type"], config)
            space = max_e - r["energy"]
            if space == float("inf"):
                r["energy"] += crystals[key]
            else:
                r["energy"] += min(crystals[key], int(space))
            crystals_to_remove.append(key)
    for key in crystals_to_remove:
        if key in crystals:
            del crystals[key]

    # --- Phase 6: Mine energy fill ---
    for uid, r in robots.items():
        key = f"{r['col']},{r['row']}"
        if key in mines and mines[key][2] == r["owner"]:
            mine = mines[key]
            max_e = get_robot_max_energy(r["type"], config)
            space = max_e - r["energy"]
            if space == float("inf"):
                transfer = mine[0]
            else:
                transfer = min(mine[0], int(space))
            r["energy"] += transfer
            mine[0] -= transfer

    # --- Phase 7: Mine energy generation ---
    for key, mine in mines.items():
        mine[0] = min(mine[0] + mine[3], mine[1])

    # --- Phase 8: Scroll advancement ---
    # Pull the hidden seed from env.info (see initialize_game). Falling back
    # to 0 keeps determinism if env.info is somehow missing.
    env_info = getattr(env, "info", None) or {}
    episode_seed = env_info.get("seed", 0) or 0
    rng = Random(episode_seed + obs.step)
    obs.scrollCounter = obs.scrollCounter - 1
    if obs.scrollCounter <= 0:
        obs.southBound += 1
        obs.northBound += 1
        south = obs.southBound
        north = obs.northBound

        # Generate new north row
        eller_state = obs.ellerState
        if isinstance(eller_state, dict):
            pass
        else:
            eller_state = dict(eller_state)
            eller_state["sets"] = list(eller_state["sets"])

        new_row_walls = generate_maze_row(rng, width, eller_state, config.doorProbability)
        walls[str(north)] = new_row_walls
        ensure_wall_consistency(walls, north, width)
        obs.ellerState = eller_state

        place_crystals(
            rng,
            width,
            crystals,
            north,
            config.crystalDensity,
            config.crystalMinEnergy,
            config.crystalMaxEnergy,
        )
        place_mining_nodes(rng, width, mining_nodes, north, config.miningNodeDensity, crystals)

        # Clean up old rows
        old_key = str(south - 1)
        if old_key in walls:
            del walls[old_key]

        obs.scrollCounter = get_scroll_interval(obs.step, config)

    # --- Phase 9: Boundary destruction ---
    south = obs.southBound
    boundary_destroyed = set()
    factory_destroyed = [False, False]

    for uid, r in list(robots.items()):
        if r["row"] < south:
            if r["type"] == FACTORY:
                factory_destroyed[r["owner"]] = True
            boundary_destroyed.add(uid)

    for uid in boundary_destroyed:
        del robots[uid]

    # Remove mines below boundary
    mine_keys_to_remove = [key for key in mines if int(key.split(",")[1]) < south]
    for key in mine_keys_to_remove:
        del mines[key]

    # Remove crystals below boundary
    crystal_keys_to_remove = [key for key in crystals if int(key.split(",")[1]) < south]
    for key in crystal_keys_to_remove:
        del crystals[key]

    # Remove mining nodes below boundary
    node_keys_to_remove = [key for key in mining_nodes if int(key.split(",")[1]) < south]
    for key in node_keys_to_remove:
        del mining_nodes[key]

    # --- Phase 10: Win condition check ---
    # Check if any player has no factory
    for player_idx in range(2):
        has_factory = any(r["type"] == FACTORY and r["owner"] == player_idx for r in robots.values())
        if not has_factory and state[player_idx].status == "ACTIVE":
            factory_destroyed[player_idx] = True

    if factory_destroyed[0] and factory_destroyed[1]:
        # Both eliminated same turn - tiebreak via energy → unit count → draw
        r0, r1 = _resolve_tiebreak(robots)
        state[0].reward = r0
        state[1].reward = r1
        state[0].status = "DONE"
        state[1].status = "DONE"
    elif factory_destroyed[0]:
        state[0].reward = obs.step - config.episodeSteps - 1
        state[1].reward = sum(r["energy"] for r in robots.values() if r["owner"] == 1)
        state[0].status = "DONE"
        state[1].status = "DONE"
    elif factory_destroyed[1]:
        state[1].reward = obs.step - config.episodeSteps - 1
        state[0].reward = sum(r["energy"] for r in robots.values() if r["owner"] == 0)
        state[0].status = "DONE"
        state[1].status = "DONE"
    elif obs.step + 2 >= config.episodeSteps:
        # Time limit: both factories alive on the final interpreter call →
        # tiebreak via energy → unit count → draw.
        r0, r1 = _resolve_tiebreak(robots)
        state[0].reward = r0
        state[1].reward = r1
        state[0].status = "DONE"
        state[1].status = "DONE"

    # --- Phase 12: Update rewards for active players ---
    for player_idx in range(2):
        if state[player_idx].status == "ACTIVE":
            total = sum(r["energy"] for r in robots.values() if r["owner"] == player_idx)
            state[player_idx].reward = total

    # --- Serialize robots back to global state ---
    obs.globalRobots = {uid: robot_to_list(r) for uid, r in robots.items()}
    obs.globalCrystals = crystals
    obs.globalMines = mines
    obs.globalMiningNodes = mining_nodes
    obs.globalWalls = walls

    # --- Phase 11: Update per-player observations ---
    _update_player_observations(state, env)

    return state


# --- Renderer ---


def renderer(state, env):
    config = env.configuration
    width = config.width
    obs = state[0].observation
    south = obs.southBound
    north = obs.northBound

    g_walls = obs.globalWalls or {}
    g_robots = obs.globalRobots or {}
    g_mines = obs.globalMines or {}
    g_crystals = obs.globalCrystals or {}

    TYPE_CHAR = {FACTORY: "F", SCOUT: "S", WORKER: "W", MINER: "M"}

    # Build cell content map
    cell_content = {}

    for uid, data in g_robots.items():
        rtype, col, row, energy, owner = data[0], data[1], data[2], data[3], data[4]
        cell_content[(col, row)] = f"{TYPE_CHAR[rtype]}{owner}"

    for key, data in g_mines.items():
        col, row = int(key.split(",")[0]), int(key.split(",")[1])
        if (col, row) not in cell_content:
            cell_content[(col, row)] = f"m{data[2]}"

    g_mining_nodes = obs.globalMiningNodes or {}
    for key in g_mining_nodes:
        col, row = int(key.split(",")[0]), int(key.split(",")[1])
        if (col, row) not in cell_content:
            cell_content[(col, row)] = " <> "

    for key, energy in g_crystals.items():
        col, row = int(key.split(",")[0]), int(key.split(",")[1])
        if (col, row) not in cell_content:
            cell_content[(col, row)] = f"*{min(energy, 99)}"

    out = ""
    for row in range(north, south - 1, -1):
        row_key = str(row)
        rw = g_walls.get(row_key, [0] * width)

        # Top border (north walls)
        top = f"{row:3d} +"
        for col in range(width):
            w = rw[col] if col < len(rw) else 0
            top += "----+" if (w & WALL_N) else "    +"
        out += top + "\n"

        # Cell row
        cell_line = "    "
        for col in range(width):
            w = rw[col] if col < len(rw) else 0
            wall_char = "|" if (w & WALL_W) else " "
            content = cell_content.get((col, row), "")
            cell_line += wall_char + content.center(4)
        # Right edge
        last_w = rw[width - 1] if width - 1 < len(rw) else 0
        cell_line += "|" if (last_w & WALL_E) else " "
        out += cell_line + "\n"

    # Bottom border
    bottom = "    +"
    south_key = str(south)
    if south_key in g_walls:
        rw = g_walls[south_key]
        for col in range(width):
            w = rw[col] if col < len(rw) else 0
            bottom += "----+" if (w & WALL_S) else "    +"
    else:
        bottom += "----+" * width
    out += bottom + "\n"

    out += f"\nStep: {obs.step}  South: {south}  North: {north}\n"
    for uid, data in g_robots.items():
        rtype, col, row, energy, owner = data[0], data[1], data[2], data[3], data[4]
        tname = TYPE_NAMES[rtype]
        out += f"  {uid}: P{owner} {tname} ({col},{row}) E={energy}\n"

    return out


# --- Module exports ---

dir_path = path.dirname(__file__)
json_path = path.abspath(path.join(dir_path, "crawl.json"))
with open(json_path) as json_file:
    specification = json.load(json_file)


def html_renderer(env, mode):
    # In ipython/notebook mode, use the lightweight single-file JS renderer
    if mode == "ipython":
        js_path = path.abspath(path.join(dir_path, "crawl.js"))
        if path.exists(js_path):
            with open(js_path, encoding="utf-8") as js_file:
                return js_file.read()
    # Default: use the full Vite-built visualizer
    jspath = path.join(dir_path, "visualizer", "default", "dist", "index.html")
    if path.exists(jspath):
        with open(jspath, encoding="utf-8") as f:
            return f.read()
    # Fallback to single-file JS renderer
    js_path = path.abspath(path.join(dir_path, "crawl.js"))
    if path.exists(js_path):
        with open(js_path, encoding="utf-8") as js_file:
            return js_file.read()
    return ""
