# Reinforce Tactics - Kaggle Environment

A turn-based tactical strategy game for the
[Kaggle Environments](https://github.com/Kaggle/kaggle-environments) framework.
Two players command armies of diverse unit types on a grid map, competing to
capture the enemy headquarters or eliminate all enemy units.

**Repository:** [github.com/kuds/reinforce-tactics](https://github.com/kuds/reinforce-tactics)
**Website & Docs:** [reinforcetactics.com](https://reinforcetactics.com)

## How the Game Works

### Overview

Reinforce Tactics is a two-player, turn-based strategy game played on a grid
map. Players alternate turns, issuing a list of commands each turn: recruiting
units at buildings, moving them across terrain, attacking enemies, seizing
structures, and casting support abilities. The game ends when a player captures
the enemy headquarters or eliminates all enemy units. If neither happens within
the turn limit, the game is a draw.

### Economy

Each player starts with **250 gold** (configurable). Income is earned at the
start of each turn from owned structures:

| Structure    | Income per Turn |
|--------------|-----------------|
| Headquarters | 150 gold        |
| Building     | 100 gold        |
| Tower        | 50 gold         |

Gold is spent to recruit units at buildings or the headquarters.

### Terrain

| Tile     | Code | Effect |
|----------|------|--------|
| Grass    | `p`  | Normal movement |
| Forest   | `f`  | +30% evasion for Rogues, blocks ranged line of sight |
| Mountain | `m`  | Walkable; units gain +1 vision when standing on it |
| Water    | `w`  | Impassable |
| Road     | `r`  | Normal movement |
| Building | `b`  | Capturable; provides income and unit recruitment |
| Tower    | `t`  | Capturable; provides income and a defensive position |
| HQ       | `h`  | Capturable; losing your HQ loses the game |

### Unit Types

| Code | Unit      | Cost | HP | ATK   | DEF | Move | Special Ability |
|------|-----------|------|----|-------|-----|------|-----------------|
| W    | Warrior   | 200  | 15 | 10    | 6   | 3    | Reliable melee fighter |
| M    | Mage      | 300  | 10 | 8/12  | 4   | 2    | Paralyze enemy (range 2, 3 turns) |
| C    | Cleric    | 200  | 8  | 2     | 4   | 2    | Heal and cure allies (range 2) |
| A    | Archer    | 250  | 15 | 5     | 1   | 3    | Ranged attack 2-3 tiles (+1 on mountain) |
| K    | Knight    | 350  | 18 | 8     | 5   | 4    | Charge: +50% damage after moving 3+ tiles |
| R    | Rogue     | 350  | 12 | 9     | 3   | 4    | Flank: +50% damage; 15% evasion (30% in forest) |
| S    | Sorcerer  | 400  | 10 | 6/8   | 3   | 2    | Haste, Attack Buff, Defence Buff (+35%, range 2) |
| B    | Barbarian | 400  | 20 | 10    | 2   | 5    | High mobility, high HP melee |

### Turn Structure

Each turn, the active player submits a list of action commands:

1. **Recruit** units at owned buildings/HQ (costs gold)
2. **Move** units across walkable terrain (limited by movement range)
3. **Attack** adjacent enemies (melee) or distant enemies (ranged units)
4. **Seize** enemy or neutral structures with a unit standing on them
5. **Cast abilities** (heal, cure, paralyze, haste, buffs)
6. **End turn** to pass control to the opponent

Units can move and act once per turn (unless hasted).

### Win Conditions

- **Capture the enemy HQ** - Move a unit onto the enemy headquarters and seize it
- **Eliminate all enemy units** - Destroy every unit the opponent has on the board
- **Draw** - If neither condition is met within the turn limit (default 200)

### Fog of War (Optional)

When enabled, each player can only see tiles within their units' vision range.
Enemy units outside visible tiles are hidden from the observation.

## Actions Reference

Each turn, an agent returns a list of action dicts:

| Action         | Fields                                  | Description |
|----------------|-----------------------------------------|-------------|
| `create_unit`  | `unit_type`, `x`, `y`                   | Recruit a unit at a building |
| `move`         | `from_x`, `from_y`, `to_x`, `to_y`     | Move a unit |
| `attack`       | `from_x`, `from_y`, `to_x`, `to_y`     | Attack an enemy unit |
| `seize`        | `x`, `y`                                | Seize a structure |
| `heal`         | `from_x`, `from_y`, `to_x`, `to_y`     | Cleric heals an ally |
| `cure`         | `from_x`, `from_y`, `to_x`, `to_y`     | Cleric cures paralysis |
| `paralyze`     | `from_x`, `from_y`, `to_x`, `to_y`     | Mage paralyzes an enemy |
| `haste`        | `from_x`, `from_y`, `to_x`, `to_y`     | Sorcerer hastes an ally |
| `defence_buff` | `from_x`, `from_y`, `to_x`, `to_y`     | Sorcerer buffs defence |
| `attack_buff`  | `from_x`, `from_y`, `to_x`, `to_y`     | Sorcerer buffs attack |
| `end_turn`     | *(none)*                                | End the current turn |

## Configuration

| Parameter      | Default            | Description |
|----------------|--------------------|-------------|
| `episodeSteps` | 200                | Max turns before draw |
| `mapName`      | `"beginner"`       | Built-in map name (see below), or empty for random generation |
| `mapWidth`     | 20                 | Map width (10-40) &mdash; only used when `mapName` is empty |
| `mapHeight`    | 20                 | Map height (10-40) &mdash; only used when `mapName` is empty |
| `mapSeed`      | -1 (random)        | Seed for map generation &mdash; only used when `mapName` is empty |
| `enabledUnits` | `W,M,C,A,K,R,S,B` | Which unit types are available |
| `fogOfWar`     | false              | Enable fog of war |
| `startingGold` | 250                | Starting gold per player |

### Map Selection

You can play on a **built-in map** or a **randomly generated** one.

#### Built-in maps

Set the `mapName` configuration parameter to one of these map names:

| Name          | Description |
|---------------|-------------|
| `beginner`    | Small 6x6 symmetrical map with 4 central towers. Good for learning. |
| `crossroads`  | 8x8 map with forests, mountains, and 4 central towers. |
| `tower_rush`  | 8x8 map with scattered towers encouraging aggressive play. |

Built-in maps smaller than 20x20 are automatically padded to 20x20 with an
ocean border (matching the upstream
[reinforce-tactics](https://github.com/kuds/reinforce-tactics) map editor
behaviour). More maps from the main repository may be added in the future.

```python
# Play on the "beginner" built-in map
env = make("reinforce_tactics", configuration={"mapName": "beginner"})
```

#### Random generation

When `mapName` is set to an empty string, a random map is generated using
`mapWidth`, `mapHeight`, and `mapSeed`. The random generator places terrain
features (forests ~10%, mountains ~5%, water ~3%), two headquarters with
adjacent buildings, and four neutral towers near the centre.

```python
# Random map with a fixed seed for reproducibility
env = make("reinforce_tactics", configuration={
    "mapName": "",
    "mapWidth": 20,
    "mapHeight": 20,
    "mapSeed": 42,
})

# Fully random map (different each run)
env = make("reinforce_tactics", configuration={"mapName": ""})
```

#### Map format reference

Maps use single-letter tile codes, optionally suffixed with `_player` for
ownership (e.g. `h_1` = Player 1 headquarters, `b_2` = Player 2 building,
`t` = neutral tower). See the
[upstream repository](https://github.com/kuds/reinforce-tactics) for the full
map editor and additional maps.

## Quick Start

```python
from kaggle_environments import make

# Create the environment with a built-in map
env = make("reinforce_tactics", configuration={"mapName": "beginner"})

# -- or with random generation --
# env = make("reinforce_tactics", configuration={"mapSeed": 42})

# Run with built-in agents
result = env.run(["random", "aggressive"])

# Print results
for i, agent in enumerate(result[-1]):
    print(f"Agent {i}: status={agent.status}, reward={agent.reward}")
```

## Writing an Agent

```python
def my_agent(observation, configuration):
    actions = []
    player = observation.player + 1  # 1-indexed
    gold = observation.gold[observation.player]
    units = observation.units
    structures = observation.structures

    # Recruit warriors at owned buildings
    my_buildings = [s for s in structures if s["owner"] == player and s["type"] == "b"]
    occupied = {(u["x"], u["y"]) for u in units}
    for bldg in my_buildings:
        if gold >= 200 and (bldg["x"], bldg["y"]) not in occupied:
            actions.append({
                "type": "create_unit",
                "unit_type": "W",
                "x": bldg["x"],
                "y": bldg["y"],
            })
            gold -= 200

    # Move, attack, seize, etc.
    # ...

    actions.append({"type": "end_turn"})
    return actions
```

## File Structure

```
reinforce_tactics/
    reinforce_tactics.json       # Environment specification
    reinforce_tactics.py         # Interpreter, renderer, built-in agents
    reinforce_tactics_engine/    # Vendored game engine (self-contained)
        constants.py             # Game constants and unit data
        core/                    # Tile, Unit, Grid, Visibility, GameState
        game/                    # Combat and ability mechanics
    agents/
        random_agent.py          # Minimal baseline (ends turn immediately)
        simple_bot_agent.py      # Strategic bot (recruits, attacks, seizes)
```

The game engine is vendored as a self-contained sub-package with relative
imports, following the same pattern as
[Lux AI Season 3](https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/lux_ai_s3).
No external pip install is required.

## Links

- **Source Code:** [github.com/kuds/reinforce-tactics](https://github.com/kuds/reinforce-tactics)
- **Documentation:** [reinforcetactics.com](https://reinforcetactics.com)
- **License:** Apache License 2.0
