# **Kaggriculture (Beginner)**

A farming sim where two players compete to maximize their income from farming by selling to a shared market.

## **Overview**

Each player starts with an empty farm and a small amount of income (seed money, if you will). Each turn, they can perform actions such as moving around the board, purchasing seeds, planting seeds, watering plants, harvesting produce, and selling that produce at the market. The game runs for a fixed amount of time representing one season, and the winner is determined by who has the most money in the bank at the end.

## **Object Types**

| Type | Yield Type | Seed Cost | Base Market Price | Time to First Yield | Time to Max Yield | Subsequent Yields | Max Yield | Max yield / tile / DAY |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Wheat** | One-time | 10 | 25 | 2 days | 4 days | none | 4 | 1 |
| **Carrot** | One-time | 20 | 35 | 2 days | 3 days | none | 4 | 1.333 |
| **Tomato** | Ongoing | 50 | 60 | 8 days | NA | every day | 4 | 4 |
| **Strawberry** | Ongoing | 100 | 120 | 10 days | NA | every other day | 4 | 2 |
| **Melon** | One-time | 80 | 250 | 10 days | 12 days | none | 6 | .5 |

Plants must be watered every day, starting on the day they are planted. The planting day counts as the first watering day — a freshly planted seed that goes unwatered for its planting day plus the following day will die.

## **Actions**

Each turn, the player may take one action. There are 24 turns per day, and 30 days in the season \- 720 total turns.

### **Farmer Action**

Each Farmer can be given an action every turn.

#### Movement

- NORTH, SOUTH, EAST, WEST — Move one cell in that direction

#### Plants

- PLANT — Plant a seed purchased from the market  
  - The seed is consumed from the player's seed inventory  
- WATER — Water a plant (only needs to be done once per day)  
- HARVEST — Gather produce from a plant. If the plant does not have subsequent yields, it will be removed from the map. Each harvest action will yield at least one unit of the crop, with the potential of a double yield if the plant has been watered consistently (see harvest yields below). Harvest auto sells to the market immediately.

#### Other

- PASS — Default if there is nothing to do

### **Market Action**

Each turn you can submit up to `maxMarketOrdersPerTurn` (default 10) market actions; any orders past that limit are silently dropped. This is an ordered list and market orders will be processed in order simultaneously (one from each player) while both players have orders.

- BUY\_SEED — Purchase N units of a single seed type from the market.  
  - BUY\_SEED WHEAT 1  

Harvested produce is auto-sold to the market the moment it is harvested; there is no separate SELL action in the beginner version.

## **Watering**

Plants must be watered every day, including the day they are planted. A new seed begins life with one unwatered day already on the clock (the planting day itself), so it will die at the end of the very next day if it has not been watered. After the first watering, the plant tolerates one missed day before dying — i.e. two consecutive unwatered days at the end-of-day refresh kill the plant.

Note that watering one-time yield plants during their yield window results in a higher yield. This is NOT true for ongoing yield plants. See below.

## **Harvest Yields**

When a HARVEST action succeeds, the harvested units are immediately sold to the market at the crop's fixed price and the proceeds are added to the player's bank — there is no separate inventory or SELL step. Plants will potentially have higher yields based on how well they have been cared for.

* Starting at half the plant’s maximum lifespan rounded up, watering will add one unit per day to the total havestable yield.   
* Once the plant has hit its maximum lifespan, the total yield available on the plant will reduce by 1 every other turn until it hits 0 (plant disappears).

## **Map Features**

Each player has their own farm with a set number of squares. Players can see the state of their opponent’s farm.

### **Farm Space**

- Each player's farm is a square grid of size `boardSize` × `boardSize` (default 5 × 5).
- Each plant occupies one square on the farm.  
- Players can allocate these squares however they choose between crops. There are no specific limits per type.  
- Squares on the farm can be either a plant or empty.

## **Market Mechanics**

The market has an unlimited number of each type of seeds to purchase, and the costs for those products is fixed. The price for produce you sell is also fixed.

## **Turn Processing Order**

1. **Player actions** — each player's farmer action is applied (happening simultaneously). Invalid or illegal actions are silently no-ops.  
2. **Market actions** — process each player's market queue in round-robin order  
3. **Plant decay** — for plants past their max lifespan, decrement yield each eligible turn  
4. **Day refresh** (end of day only) — kill plants unwatered for two consecutive days, produce yield for ongoing crops on their scheduled days, and reset everyone's `watered_today` flag

## **Win Conditions**

The win condition is simple- whoever has the greatest number of coins at the end of the season is the winner. It is also possible that the two players will tie.

## **Reward**

Each player's reward is set once, at the terminal step, to the float value of their final bank balance (`money`). It is not a normalized win/loss/tie signal — to determine the winner, compare the two players' rewards directly. A tie is possible if both players finish with exactly the same balance.

## **Observation Format**

Each agent receives an `obs` dict-like object every turn. Top-level fields:

| Field | Type | Shared | Description |
| :---- | :---- | :---- | :---- |
| `player` | int | no | Your player index (0 or 1) |
| `step` | int | yes | Current step in the episode (0-indexed) |
| `day` | int | yes | Current in-game day (0-indexed) |
| `hour` | int | yes | Current turn within the day (0-indexed, 0..`turnsPerDay`-1) |
| `farms` | list | yes | One entry per player, in player-index order. See below. |

Each entry in `farms` describes a single player's farm:

```py
{
  "money":  float,                  # current bank balance
  "seeds":  {"WHEAT": int, "CARROT": int, "TOMATO": int,
             "STRAWBERRY": int, "MELON": int},
  "farmer": [x, y],                 # farmer position; x is column, y is row
  "tiles":  [[tile_or_none, ...],   # 2D array indexed as tiles[y][x]
             ...]
}
```

Each tile is either `None` (empty) or:

```py
{
  "crop":                  "WHEAT" | "CARROT" | "TOMATO" | "STRAWBERRY" | "MELON",
  "planted_day":           int,    # day the seed was planted
  "watered_today":         bool,   # reset to False at the end of each day
  "consecutive_unwatered": int,    # days in a row without water; 2+ → plant dies. Starts at 1 on the planting day, so a brand-new seed dies if its planting day ends unwatered AND it isn't watered the next day either.
  "yield_units":           int,    # units currently harvestable
  "max_lifespan_step":     int     # step at which decay begins (-1 if not yet set)
}
```

## **Quick Start**

An agent is a callable that takes a single `obs` argument and returns an action dict of the form `{"farmer": [op, ...args], "market": [[op, ...args], ...]}`. The `farmer` field is a single op for that turn; the `market` field is an ordered list of market orders to process.

```py
from kaggle_environments import make


def my_agent(obs):
    # Buy one wheat seed on the very first turn, then PASS forever after.
    if obs.get("step", 0) == 0:
        return {"farmer": ["PASS"], "market": [["BUY_SEED", "WHEAT", 1]]}
    return {"farmer": ["PASS"], "market": []}


env = make("kaggriculture_beginner", configuration={"episodeSteps": 200})
env.run([my_agent, "random"])
env.render(mode="ipython", width=800, height=800)
```

Three built-in agents are available by name: `"pass"`, `"random"`, and `"starter"` (a deterministic carrot-loop baseline).

## **Configuration Defaults**

| Parameter | Default | Description |
| :---- | :---- | :---- |
| `episodeSteps` | 720 | Total turns in an episode (24 per day × 30 days by default) |
| `boardSize` | 5 | Side length (in tiles) of each player's square farm |
| `startingMoney` | 150 | Coins each player starts with |
| `turnsPerDay` | 24 | Number of turns that make up one in-game day |
| `maxMarketOrdersPerTurn` | 10 | Max market orders processed per player per turn; extras are silently dropped |
| `actTimeout` | 1 | Per-turn agent action timeout (seconds) |

Per-crop seed costs and sale prices are not configurable in the beginner version — they are fixed in the `CROPS` table inside the interpreter (see the Object Types table above for the values).