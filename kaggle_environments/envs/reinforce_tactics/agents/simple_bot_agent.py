"""
Simple bot agent for Reinforce Tactics Kaggle environment.

This agent implements a basic strategy:
1. Create warriors/archers at available buildings
2. Move units toward the nearest enemy or enemy HQ
3. Attack adjacent enemies
4. Seize enemy structures when standing on them

Usage as a Kaggle submission:
    Copy this file and submit it.
"""
# Unit costs for budget tracking
UNIT_COSTS = {
    "W": 200, "M": 300, "C": 200, "A": 250,
    "K": 350, "R": 350, "S": 400, "B": 400,
}

# Unit movement ranges
UNIT_MOVEMENT = {
    "W": 3, "M": 2, "C": 2, "A": 3,
    "K": 4, "R": 4, "S": 2, "B": 5,
}

# Preferred unit creation order
UNIT_PRIORITY = ["W", "A", "K", "B", "R", "M", "C", "S"]


def agent(observation, configuration):
    """
    Simple strategic bot that creates units, attacks, and captures.

    Args:
        observation: Kaggle observation struct
        configuration: Kaggle configuration struct

    Returns:
        list of action dicts
    """
    actions = []
    player_idx = observation.player
    player = player_idx + 1
    gold = observation.gold[player_idx]

    units = observation.units if hasattr(observation, "units") else []
    structures = observation.structures if hasattr(observation, "structures") else []
    board = observation.board if hasattr(observation, "board") else []
    map_w = observation.mapWidth if hasattr(observation, "mapWidth") else 20
    map_h = observation.mapHeight if hasattr(observation, "mapHeight") else 20

    # Parse enabled units
    enabled_str = getattr(configuration, "enabledUnits", "W,M,C,A,K,R,S,B")
    enabled_units = set(u.strip() for u in enabled_str.split(",") if u.strip())

    my_units = [u for u in units if u["owner"] == player]
    enemy_units = [u for u in units if u["owner"] != player]
    occupied = {(u["x"], u["y"]) for u in units}

    # ---- Phase 1: Create units ----
    my_buildings = [
        s for s in structures
        if s["owner"] == player and s["type"] == "b"
        and (s["x"], s["y"]) not in occupied
    ]

    for bldg in my_buildings:
        best = None
        for ut in UNIT_PRIORITY:
            if ut in enabled_units and ut in UNIT_COSTS and UNIT_COSTS[ut] <= gold:
                best = ut
                break
        if best:
            actions.append({
                "type": "create_unit",
                "unit_type": best,
                "x": bldg["x"],
                "y": bldg["y"],
            })
            gold -= UNIT_COSTS[best]
            occupied.add((bldg["x"], bldg["y"]))

    # ---- Phase 2: Unit actions (attack, seize, move) ----
    # Find enemy HQ
    enemy_hq = None
    for s in structures:
        if s["owner"] != player and s["owner"] != 0 and s["type"] == "h":
            enemy_hq = (s["x"], s["y"])
            break

    for unit in my_units:
        if not unit["canMove"] and not unit["canAttack"]:
            continue
        if unit["paralyzedTurns"] > 0:
            continue

        ux, uy = unit["x"], unit["y"]

        # Attack adjacent enemies
        if unit["canAttack"]:
            for enemy in enemy_units:
                dist = abs(ux - enemy["x"]) + abs(uy - enemy["y"])
                attack_range = _get_attack_range(unit["type"])
                if attack_range[0] <= dist <= attack_range[1]:
                    actions.append({
                        "type": "attack",
                        "from_x": ux,
                        "from_y": uy,
                        "to_x": enemy["x"],
                        "to_y": enemy["y"],
                    })
                    break  # One attack per unit

        # Seize if on enemy structure
        tile_structure = _get_structure_at(structures, ux, uy)
        if tile_structure and tile_structure["owner"] != player and tile_structure["owner"] != 0:
            actions.append({
                "type": "seize",
                "x": ux,
                "y": uy,
            })
            continue

        # Move toward nearest enemy or enemy HQ
        if unit["canMove"]:
            target = None
            if enemy_units:
                # Find nearest enemy
                nearest = _find_nearest_enemy(ux, uy, enemy_units)
                target = (nearest["x"], nearest["y"])
            elif enemy_hq:
                target = enemy_hq

            if target:
                next_pos = _step_toward(
                    ux, uy, target[0], target[1],
                    board, occupied, map_w, map_h
                )
                if next_pos and next_pos != (ux, uy):
                    actions.append({
                        "type": "move",
                        "from_x": ux,
                        "from_y": uy,
                        "to_x": next_pos[0],
                        "to_y": next_pos[1],
                    })
                    # Update occupied set
                    occupied.discard((ux, uy))
                    occupied.add(next_pos)

    actions.append({"type": "end_turn"})
    return actions


def _find_nearest_enemy(ux, uy, enemy_units):
    """Find the nearest enemy unit by Manhattan distance."""
    return min(
        enemy_units,
        key=lambda e: abs(ux - e["x"]) + abs(uy - e["y"])
    )


def _get_attack_range(unit_type):
    """Return (min_range, max_range) for a unit type."""
    if unit_type in ("M", "S"):
        return (1, 2)
    if unit_type == "A":
        return (2, 3)
    return (1, 1)


def _get_structure_at(structures, x, y):
    """Find a structure at the given position."""
    for s in structures:
        if s["x"] == x and s["y"] == y:
            return s
    return None


def _step_toward(from_x, from_y, to_x, to_y, board, occupied, map_w, map_h):
    """
    Return the best adjacent position that moves toward the target.
    Uses simple greedy Manhattan distance minimisation.
    """
    best_pos = None
    best_dist = abs(from_x - to_x) + abs(from_y - to_y)

    non_walkable = {"w", "o", "m"}

    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx, ny = from_x + dx, from_y + dy
        if 0 <= nx < map_w and 0 <= ny < map_h:
            if (nx, ny) not in occupied:
                tile = board[ny][nx] if ny < len(board) and nx < len(board[ny]) else "o"
                if tile not in non_walkable:
                    dist = abs(nx - to_x) + abs(ny - to_y)
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (nx, ny)

    return best_pos
