# Planet Wars

A bit-exact port of the 2009/2010 University of Waterloo / Google AI Challenge
"Planet Wars" competition (Jeff Cameron). Two players battle for control of a
galaxy of planets by sending fleets of ships across a symmetric map.

## Overview

The map contains 15-30 planets, each with some number of ships. Planets are
owned by player 1, player 2, or no one (neutral). Owned planets generate new
ships each turn at their `growth_rate`. Players issue fleet orders to send
ships from planets they own to any other planet. Fleets travel in straight
lines and take `ceil(euclidean_distance)` turns to arrive, then fight whoever
holds the destination.

The game ends when one player has no planets and no fleets, or when the turn
limit (200) is reached — in which case the player with the most total ships
(on planets + in fleets) wins.

## Board

- Planets are placed in continuous 2D space at fixed positions for the whole
  game.
- Every map is symmetric: either point-symmetric about the centre (180°
  rotation) or reflective about an axis. Player 1 and player 2's home planets
  are mirror images of each other, so neither player has a positional
  advantage.
- Each planet has a fixed `growth_rate` in `[1, 5]` (centre planet may be 0).
- Coordinates and trip distances use the original 2010 contest conventions:
  trip length = `ceil(sqrt((x1-x2)^2 + (y1-y2)^2))`.

## Turn order

Each turn (in this order):

1. The engine collects both players' orders.
2. **Departure**: orders are applied — ships leave their source planets and
   form new fleets. If a player issues multiple orders from the same source
   to the same destination on the same turn, they are merged into a single
   fleet.
3. **Advancement**: every in-flight fleet's `turns_remaining` decrements by
   one. Every owned planet's ship count increases by its `growth_rate`
   (neutral planets do not grow).
4. **Arrival**: fleets that just landed (`turns_remaining == 0`) fight at
   their destination. Sum ships per owner — the largest force wins and ends
   up with `(largest - second_largest)` ships. On a tie at the top the
   planet's prior owner is retained with zero ships.

## Winning

- If only one player has any planets or fleets after a turn, they win.
- If the turn limit (default 200) is reached, the player with more total
  ships (across planets + fleets) wins. A tie is a draw.
- Issuing any invalid order forfeits the game immediately. If both players
  forfeit on the same turn, it's a draw.

### Invalid orders

An order `[source, dest, ships]` is invalid if any of these hold:

- `ships <= 0`
- `source == dest`
- `source` or `dest` is not a valid planet index
- `source` is not owned by the player issuing the order
- the sum of all `ships` sent from a single planet on this turn exceeds the
  ships currently on that planet

## Observation

Both agents see the same global state (no point-of-view switching).

| Field | Type | Description |
|---|---|---|
| `planets` | `[[id, x, y, owner, num_ships, growth_rate], ...]` | All planets. Owner is 0 (neutral), 1, or 2. |
| `fleets` | `[[owner, num_ships, source, dest, total_trip, turns_remaining], ...]` | All in-flight fleets. |
| `player` | `int` | This agent's owner id (1 for agent 0, 2 for agent 1). |
| `remainingOverageTime` | `float` | Remaining overage time budget. |

## Action

A list of fleet orders. Each order is `[source_planet_id, dest_planet_id,
num_ships]`. An empty list is a legal no-op turn.

```python
def my_agent(obs, config):
    # Send half of every owned planet's ships to the nearest non-self planet
    import math
    moves = []
    planets = obs.get("planets", [])
    me = obs.get("player", 1)
    for p in planets:
        pid, x, y, owner, ships, growth = p
        if owner != me or ships < 2:
            continue
        target = min(
            (t for t in planets if t[0] != pid),
            key=lambda t: math.hypot(t[1] - x, t[2] - y),
        )
        moves.append([pid, target[0], ships // 2])
    return moves
```

## Agent convenience

The module exports named tuples for easier field access:

```python
from kaggle_environments.envs.planet_wars.planet_wars import Planet, Fleet, distance

def agent(obs, config):
    planets = [Planet(*p) for p in obs.get("planets", [])]
    fleets = [Fleet(*f) for f in obs.get("fleets", [])]
    me = obs.get("player", 1)
    # distance(p_a, p_b) -> ceil-Euclidean trip length in turns
    ...
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `episodeSteps` | 200 | Maximum number of turns. |
| `actTimeout` | 1 | Seconds per turn (original contest used 1 s). |
| `map` | `"random"` | Either `"random"` or a raw Point-in-Time map text starting with `"P "`. |
| `seed` | `null` | Map-generation seed. When `null`, the interpreter resolves it to a random integer and writes it back into `configuration.seed` so the episode is reproducible. The seed is intentionally public — the map is fully observable from turn 0, so nothing is concealed by hiding it. |

## Maps

Maps are generated procedurally from `seed` using a port of the original
contest's `map_generator_v2.py` (Apache-2.0, Jeff Cameron). The procedural
distribution approximates the original 100 contest maps:

- 15-30 planets
- One of two symmetry types per map: point-symmetric or reflective
- Home planets are mirror images of each other and at least 4 turns apart
- All other planets are at least 2 turns apart
- Floating-point coordinates are rejected if `ceil(distance)` would be
  platform-dependent

You can also pass a literal map in the original Point-in-Time text format
via `configuration.map`:

```
P <x:float> <y:float> <owner:int> <ships:int> <growth:int>
F <owner:int> <ships:int> <source:int> <destination:int> <total_turns:int> <remaining_turns:int>
```

## Built-in agents

- `do_nothing` — never issues an order.
- `random` — each owned planet has a 30% chance per turn of sending half its
  ships to a random other planet.
- `nearest_enemy` — each owned planet sends half its ships to its nearest
  non-owned planet.

## Credits

The simulation rules and map generator are direct ports of Jeff Cameron's
original engine (Apache-2.0), via Albert Zeyer's C++ port and the
[xtevenx/planet-wars-starterpackage](https://github.com/xtevenx/planet-wars-starterpackage)
re-release of the original specification.
