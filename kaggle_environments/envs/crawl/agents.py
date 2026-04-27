from random import choice


def random_agent(observation, configuration):
    """Simple agent: builds scouts, moves robots north, collects energy."""
    actions = {}
    my_robots = {}

    if not observation.robots:
        return actions

    for uid, data in observation.robots.items():
        rtype, col, row, energy, owner = data[0], data[1], data[2], data[3], data[4]
        if owner == observation.player:
            my_robots[uid] = {
                "type": rtype,
                "col": col,
                "row": row,
                "energy": energy,
                "move_cooldown": data[5] if len(data) > 5 else 0,
                "build_cooldown": data[7] if len(data) > 7 else 0,
            }

    width = configuration.width

    for uid, robot in my_robots.items():
        rtype = robot["type"]
        col = robot["col"]
        row = robot["row"]
        energy = robot["energy"]

        idx = (row - observation.southBound) * width + col
        w = 0
        if 0 <= idx < len(observation.walls) and observation.walls[idx] != -1:
            w = observation.walls[idx]

        if rtype == 0:  # Factory
            wall_north = w & 1
            worker_count = sum(1 for r in my_robots.values() if r["type"] == 2)
            if wall_north:
                # Wall north — jump over it first
                actions[uid] = "JUMP_NORTH"
            elif robot["energy"] >= configuration.workerCost and worker_count < 1 and robot["build_cooldown"] == 0:
                actions[uid] = "BUILD_WORKER"
            else:
                actions[uid] = "NORTH"
        elif rtype == 2:  # Worker
            if (w & 1) and energy >= configuration.explodeCost:
                # Wall north — blow it up
                actions[uid] = "EXPLODE_NORTH"
            elif not (w & 1):
                actions[uid] = "NORTH"
            else:
                passable = []
                if not (w & 2):
                    passable.append("EAST")
                if not (w & 8):
                    passable.append("WEST")
                if not (w & 4):
                    passable.append("SOUTH")
                actions[uid] = choice(passable) if passable else "IDLE"
        else:
            # Scouts/miners: try to move north, fallback to other directions
            passable = []
            if not (w & 1):
                passable.append("NORTH")
            if not (w & 2):
                passable.append("EAST")
            if not (w & 4):
                passable.append("SOUTH")
            if not (w & 8):
                passable.append("WEST")
            if passable:
                if "NORTH" in passable:
                    actions[uid] = "NORTH"
                else:
                    actions[uid] = choice(passable)
            else:
                actions[uid] = "IDLE"

    return actions


agents = {"random": random_agent}
