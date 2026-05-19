ß# Kaggriculture

A farming sim where two players compete to maximize their income from farming by selling to a dynamic market.

## Overview

Each player starts with an empty farm and a small amount of income (seed money, if you will). Each turn, they can perform actions such as moving around the board, purchasing seeds or livestock, planting seeds, watering plants, harvesting produce or animal products, and selling that produce at the market. The game runs for a fixed amount of time representing one season, and the winner is determined by who has the most money in the bank at the end.

## Object Types

| Type | Yield Type | Seed Cost | Base Market Price | Time to First Yield | Time to Max Yield | Subsequent Yields | Max Yield | Action Cost | Max yield / tile / DAY |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Wheat** | One-time | 10 | 25 | 2 days | 4 days | none | 6 | 1 | 1.5 |
| **Carrot** | One-time | 20 | 35 | 2 days | 3 days | none | 4 | 1 | 1.333 |
| **Tomato** | Ongoing | 50 | 60 | 8 days | NA | every day | 4 | 1 | 4 |
| **Strawberry** | Ongoing | 100 | 120 | 10 days | NA | every other day | 4 | 1 | 2 |
| **Melon** | One-time | 80 | 250 | 10 days | 12 days | none | 6 | 1 | .5 |
| **Goose/Egg** | Ongoing | 200 | 80 | 3 days | NA | every day | 4 | 1 \+ 1 (build coop) | 2 |
| **Cow/Milk** | Ongoing | 500 | 240 | 6 days | NA | every two days | 6 | 1 \+ 1 (build pasture) | 1 |
| **Sheep/Wool** | Ongoing | 400 | 300 | 5 days | NA | every three days | 6 | 1 \+ 1 (build pasture) | .67 |
| **Fertilizer** | NA | 100 | X |  | X | X |  | 1 |  |

All plants must be watered every day. They will turn into weeds if they are not watered for two successive days. All animals must be fed every day using wheat. They will escape and be unrecoverable if they are not fed for two successive days. Wheat is also available to buy at the market and can be purchased at the current market price.

## Actions

Each turn, the player may take one action. There are 24 turns per day, and 30 days in the season \- 720 total turns.

### Farmer / Farm Hand Action

Each Farmer / Farm Hand can be given an action every turn. Farmer/Farm Hand CAN occupy the same space.

#### Movement

- NORTH, SOUTH, EAST, WEST — Move one cell in that direction

#### Shed

Picks up an item from the shed (must be orthogonally adjacent) into the inventory

- PICKUP `<item>` `[n]` — move up to `n` of `<item>` (default 1) from the shed into the active farmer/hand's inventory. Any item present in the shed is valid (animals, fertilizer, harvested produce, etc.). Seeds live in a separate slot and are never picked up — `PLANT` consumes them directly.

#### Plants

- PLANT — Plant a seed purchased from the market  
  - Seeds are automatically available to all Farmers / Farm Hands   
  - If you try to plant too many in a specific turn, none are planted  
    - ie if you have 1 melon seed, but two units do the PLANT MELON command  
- WATER — Water a plant. This only needs to be done once per day, and subsequent waterings on the same day are a no-op.  
- HARVEST — Gather produce from a plant. If the plant does not have subsequent yields, it will be removed from the map. Each harvest action will yield at least one unit of the crop, with the potential of additional yield depending on watering and fertilizer (the formula differs by crop type — see harvest yields below). Harvested items are added to the inventory. 
- FERTILIZE — Fertilize a plant to increase its potential yield (see harvest yields below).  
  - Doubles the per-day yield bonus for the next 3 days. The bonus only applies on days the plant is also watered (basic needs first).

#### Animals

- PLACE `<item>` `[n]` — Drop items from the active farmer/hand inventory into either a tile or the shed:  
  - **Animal placement**: standing on a matching unoccupied structure (`GOOSE` on a coop, `SHEEP`/`COW` on a pasture) places one animal from inventory onto the tile. The `n` argument is ignored.  
  - **Shed drop**: standing orthogonally adjacent to the shed moves up to `n` (default 1) of `<item>` from inventory into the shed. Capped by `shedCapacity`; excess stays in inventory.
- FEED — Feed an animal using wheat (only needs to be done once per day)  
- HARVEST — Collect the eggs/milk/wool produced by the animal.   
- COLLECT\_FERTILIZER — Collect 1 fertilizer from the animal. Each surviving animal makes 1 fertilizer available at the end of every day; collecting consumes that day's stock and the next becomes available after the next end-of-day refresh.
- CARE — Care for an animal (once per day, no-op if already cared for). See animal care below.

#### Animal Care

CARE banks a yield bonus that is paid out on the animal's next scheduled production:

* At end of day, if the animal was both fed AND cared for that day, `pending_care_bonus` increments by 2. Days where the animal was unfed do not bank a bonus (basic needs first).
* On a scheduled production day, if the animal is fed, the entire banked bonus is added to that production's yield (in addition to the base 1) and the bank resets to 0.
* If the animal is unfed on the production day, no yield is produced that day and the bank is also reset.
* `pending_care_bonus` is capped indirectly by the per-animal `max_held` cap on `yield_units`.

#### Terrain

- BUILD\_COOP \- adds a coop to an unoccupied tile  
- BUILD\_PASTURE \- add pasture to an unoccupied tile  
- DIG — Remove a plant from a square to free up space OR remove a weed from a square (does not yield any produce) OR remove a goose coop / pasture.

#### Other

- PASS — Default if there is nothing to do (optional)

### Market Action

Each turn you can submit up to `maxMarketOrdersPerTurn` (default 10) market actions; any orders past that limit are silently dropped. This is an ordered list and market orders will be processed in order simultaneously (one from each player) while both players have orders.

- BUY\_SEED — Purchase N units of a single item from the market.  
  - BUY\_SEED WHEAT 1  
- BUY\_ANIMAL \-   
  - BUY\_ANIMAL GOOSE 1  
- BUY\_PRODUCT  
  - BUY\_PRODUCT WHEAT 1  
  - BUY\_PRODUCT FERTILIZER 1  
- SELL — Sell N units of a single item to the market.  
  - SELL WHEAT 1  
- HIRE — Hire a farm hand for the day. Cost increases for each extra hand hired on the same day.  
- BUY\_LAND \- unlock a new 5x5 segment of land to plant on. Increasing in cost.   
  - Costs are: $1k, $2k, $4k

## Watering / Animal Feed

Plants (and animals) must be watered/fed a minimum of every other day. Watering only needs to be done once per day, and subsequent watering actions are a no-op. In the case of plants not watered for two consecutive days, at the end of the day they turn into a WEED. In the case of animals they escape (unrecoverable).

Note that watering one-time yield plants during their yield window results in a higher yield. This is NOT true for ongoing yield plants/animals. See below.

## Harvest Yields

Plants will potentially have higher yields based on how well they have been cared for. 

* **One-time crops** (wheat, carrot, melon): Starting at half the plant's `max_yield_day` (Time to Max Yield) rounded up, watering during the bonus window will add one unit per day to the total harvestable yield.  
  * Fertilized plants add 2 per day instead.  
* **Ongoing crops** (tomato, strawberry): Scheduled production happens at fixed intervals. The base yield is 1 per scheduled production. If the plant is fertilized AND watered that day, yield is doubled to 2.  
* Once a plant has hit its maximum lifespan, the total yield available on the plant will reduce by 1 every other turn until it hits 0, at which point the plant becomes a weed.
  * **One-time crops** reach max lifespan one day after `max_yield_day`.
  * **Ongoing crops** start decay one day after their cumulative production count reaches `max_yield` (i.e. they've fired enough scheduled productions to hit the cap, regardless of whether the produce has been harvested).

## Map Features

Each player has their own farm with a set number of squares. Players are unable to see the state of the other’s shed, but can see the state of their opponent’s farm.

### Farm Space

- The land near your farm is a `boardSize` × `boardSize` grid (default 10×10), divided into four 5×5 quadrants. At first, your farm covers one quadrant (25% of the squares). For an increasingly large fee, you can buy the neighboring quadrants and eventually cover 100% of the squares.  
- Each plant or animal occupies one square on the farm.  
- Players can allocate these squares however they choose between crops and livestock. There are no specific limits per type.  
- Weeds have a chance of spawning on any empty cells on the farm, and must be cleared before the land can be used for other purposes.  
- Squares on the farm can be either a plant, a coop/pasture, a weed, or empty.

### Shed (Inventory)

- Functions as an inventory for items that are harvested but not yet sold, or for seeds that have not yet been planted  
- Farmer and hired farm hands will spawn at the shed at the start of each day  
- Farmer and hired farm hands drop their inventory at the end of the day in the shed (if there is room)  
- Limited to 100 items, excluding seeds. Once the shed is full, any further items added (via `PLACE` mid-day or end-of-day inventory drop) are discarded — there is no overflow holding area, so stockpiling on farmer/hand inventories does not bypass the cap.

### Farmer/Farm Hand

#### Hiring

- Hiring is a market order (`HIRE`). It costs more every time you want to hire an additional hand each day. At the end of the day all, hands drop inventory at the farm and disappear (need to be re-hired each day)  
- Cost is `farmHandCostMult * fib(n)` where `n` is the number of hires already made today (fib starts 1, 1, 2, 3, 5, 8, 13, ...).  
  - With the default `farmHandCostMult = 10`: 10, 10, 20, 30, 50, 80, 130, 210, etc… (resets at the start of each day)  
- A hired hand appears orthogonally adjacent to the shed in a free space following NWSE. If there are not open spaces, it looks for the one with the least occupants, breaking ties by NWSE preference

#### Inventory

- When harvesting or picking items up, they are added to inventory.  
- Can drop items in the shed  
- At the end of the day, all items in all inventory will be added to shed inventory (if there is room). Anything that doesn't fit is discarded — overflow is lost.

### Town Buildings

As the season progresses, new shops unlock at regular intervals (every `townShopUnlockInterval` days, default 3). Each unlock is randomly selected from the shops that have not yet been added; once unlocked, a shop stays active for the rest of the game. Total demand grows monotonically as more shops unlock.

Each unlocked shop consumes one of every product it demands every `townShopSellInterval` turns (default 2). So with the default interval, a shop demanding wheat removes 12 wheat from the market per day. Single-product shops consume 2x.

In addition, the town center consumes one of every product (excluding fertilizer) every `townCenterSellInterval` turns (default 6). After day 10 this is increased to 2 of each, and after day 20 it is increased to 4 of each.

| Shop Type | Increases Demand For |
| :---- | :---- |
| Bakery | eggs, wheat  |
| Pizza Shop | milk, tomatoes, wheat |
| Brunch Spot | eggs, wheat, strawberries |
| Yarn Store | wool (2x) |
| Ice Cream Shop | strawberries, milk, wheat |
| Pet Cafe | carrots (2x) |
| Smoothie Shop | strawberries, milk |
| Farmers Market | wheat, carrots, tomatoes, strawberries |

## Market Mechanics

The market has an unlimited supply of seeds and animals at fixed prices. Sell prices, however, move dynamically per resource and persist across days.

Every product (and fertilizer) starts the game with a market inventory of `I0 = 10,000` units, far above any single game's realistic production volume so that inventory is essentially guaranteed to stay positive. The sell price for a product is `base` at `I0`, rises as inventory falls (players buying or town consumption draining supply), and falls as inventory grows (players selling).

### Selling inventory to the market

Players can queue any number of sell or buy orders (for any quantity) in the market action list. Orders are processed concurrently across players, one unit at a time. For example, when both players issue `SELL CARROT 10` first, we take the current carrot price, give both players that price for their first carrot, then add 2 carrots to the market (1 from each player) — which may shift the price — and repeat until both orders complete.

If the sell price has been driven down to `$1` (the price floor), the unit is still purchased but is *not* added to market inventory, so the floor remains responsive to subsequent buys.

### Buying inventory from the market

Two things drain market inventory: town buildings (town center and shops, which consume products for free) and player `BUY_PRODUCT` orders. Buy orders follow the same one-unit-at-a-time concurrent procedure as sell orders. If a player runs out of money mid-order, the order is stopped.

### The Price Function

For each resource the curve is defined by a base price, an anchor throughput `T`, and an independent **shape function** + **target move** for each side of the equilibrium:

```
price(inv) = base + sign · amp · f(|inv − I0|)
  sign = +1  if inv < I0   (scarcity → price up)
  sign = −1  if inv > I0   (glut    → price down)
  amp  = target · base / f(T)        (derived; not stored)
  f    ∈ { linear, sq, sqrt, log, log10 }   (log uses ln(1+x), so f(0)=0)
```

Floored at `$1` and rounded to the nearest dollar.

`T` is the production capacity of a single 5×5 field over a 24-day game at optimal watering with no fertilizer (animal totals are pre-discounted by 30% to account for wheat-feed overhead). `target` says "moving `T` units past `I0` shifts the price by `target × base`." Picking different `f` and `target` on each side lets resources with similar production profiles play very differently strategically — wheat panics on scarcity but absorbs gluts, carrot is the opposite; melon barely reacts to scarcity but crashes hard on overproduction; wool mirrors melon at a smaller scale.

| Resource | Base | I0 | T | Below func | Below target | Above func | Above target | P(I0−T) | P(I0+T) | P(I0+2T) |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| **Wheat** | 25 | 10,000 | 400 | sqrt | 0.80 | log | 0.20 | $45 | $20 | $19 |
| **Carrot** | 35 | 10,000 | 450 | log | 0.20 | sqrt | 0.70 | $42 | $10 | $1 |
| **Tomato** | 60 | 10,000 | 200 | linear | 0.40 | sqrt | 0.60 | $84 | $24 | $9 |
| **Strawberry** | 120 | 10,000 | 100 | sqrt | 0.70 | linear | 0.40 | $204 | $72 | $24 |
| **Melon** | 250 | 10,000 | 300 | log | 0.20 | sq | 0.90 | $300 | $25 | $1 |
| **Egg** | 50 | 10,000 | 332 | linear | 0.40 | log | 0.20 | $70 | $40 | $39 |
| **Milk** | 160 | 10,000 | 122 | sqrt | 0.60 | linear | 0.40 | $256 | $96 | $32 |
| **Wool** | 200 | 10,000 | 105 | log | 0.20 | sq | 0.80 | $240 | $40 | $1 |
| **Fertilizer** | 100 | 10,000 | 200 | linear | 0.40 | linear | 0.40 | $140 | $60 | $20 |

The defaults live in `MARKET_PARAMS` in `kaggriculture.py`. Per-resource overrides (sparse: any subset of `base`, `I0`, `T`, `below_func`, `below_target`, `above_func`, `above_target`) can be supplied at episode creation via `env.configuration["marketParams"]` without touching code, e.g. `{"WOOL": {"above_target": 0.95}}`.

## Turn Processing Order

1. **Action validation** — verify action legality  
2. **Player actions** — record the actions taken by each player (happening simultaneously)  
3. **Market actions** \- process market queue in order by player (described above)  
4. **Town buy actions** \- town center and shops reduce inventory  
5. **Update observations**  
   - **Day refresh** — if applicable, update the condition of plants and animals for a new day, and reset their fed/watered to condition to false  
   - **Market refresh** — modify the price of items on the market based on sells from previous turn  
   - **Income update** — update the player’s bank based on any buys or sells  
   - **Farm update** — clear plants that have been harvested, items from the inventory that have been used or sold, add new plants/animals to the farm, etc

## Win Conditions

The win condition is simple- whoever has the greatest number of coins at the end of the season is the winner. It is also possible that the two players will tie.

## Reward

The player who has the most money in the bank at the end of the game wins. Unsold items in the inventory do not count towards that total.

## Observation Format

The top-level observation passed to each agent:

```py
{
  "player": int,           # 0 or 1
  "day":    int,           # 0-indexed in-game day
  "hour":   int,           # 0-indexed turn within the day
  "farms":  [farm, farm],  # public per-player state, indexed by player id (shared)
  "market": {              # shared
    "inventory": { "WHEAT": int, "CARROT": int, ... },
    "prices":    { "WHEAT": int, "CARROT": int, ... },
  },
  "town": {                # shared
    "unlocked_shops": ["BAKERY", ...],
  },
  "private": {             # this player only; opponent's private state is not visible
    "shed":        { "WHEAT": int, "GOOSE": int, "FERTILIZER": int, ... },
    "seeds":       { "WHEAT": int, "CARROT": int, ... },
    "inventories": [farmer_inv, hand_inv, ...],  # [0] is the main farmer
  },
}
```

Each `farm` dict (public, visible to both players):

```py
{
  "money":              float,
  "tiles":              [[tile, ...], ...],   # tiles[y][x]
  "farmer":             [x, y],
  "hands":              [[x, y], ...],         # hired hands for the current day
  "unlocked_quadrants": ["NW", ...],          # subset of {"NW","NE","SW","SE"}
  "hires_today":        int,                  # used to price the next HIRE
}
```

A `tile` is one of:

- `None` — empty unlocked tile
- `"LOCKED"` — tile in a quadrant the player has not yet bought
- a plant dict:
  ```py
  {
    "kind":                 "PLANT",
    "crop":                 "WHEAT" | "CARROT" | "TOMATO" | "STRAWBERRY" | "MELON",
    "planted_day":          int,
    "watered_today":        bool,   # reset to False each end-of-day
    "consecutive_unwatered": int,   # 2+ → tile turns to a weed
    "yield_units":          int,    # units currently harvestable
    "max_lifespan_step":    int,    # step at which decay begins; -1 for ongoing crops
    "fertilized_until_day": int,    # last day fertilizer bonus applies; -1 if none
  }
  ```
- a weed dict: `{"kind": "WEED"}`
- an animal structure dict (coop/pasture, optionally occupied):
  ```py
  {
    "kind":                 "COOP" | "PASTURE",
    "animal":               "GOOSE" | "COW" | "SHEEP" | None,  # None until PLACEd
    "placed_day":           int,
    "yield_units":          int,
    "fed_today":            bool,
    "consecutive_unfed":    int,    # 2+ → animal escapes
    "cared_today":          bool,
    "fertilizer_available": bool,   # set after CARE; cleared by COLLECT_FERTILIZER
    "pending_care_bonus":   int,    # banked CARE bonus, applied on the next yield tick
  }
  ```

## Quick Start

```py
from kaggle_environments import make


def my_agent(obs):
    # Buy one wheat seed on the very first turn, then PASS forever after.
    if obs.get("step", 0) == 0:
        return {"farmer": ["PASS"], "market": [["BUY_SEED", "WHEAT", 1]]}
    return {"farmer": ["PASS"], "market": []}


env = make("kaggriculture", configuration={"episodeSteps": 200})
env.run([my_agent, "random"])
env.render(mode="ipython", width=800, height=800)
```

## Configuration Defaults

Per-crop seed costs and per-product base prices are not configurable; they are documented in the Object Types and Price Function tables above. The configurable knobs are:

| Parameter | Default | Description |
| :---- | :---- | :---- |
| episodeSteps | 720 | Total turns in the season (24 turns × 30 days) |
| boardSize | 10 | Width and height (in tiles) of each player's square farm. Advanced uses 10 = four 5x5 quadrants |
| startingMoney | 2000 | Coins each player starts with |
| maxMarketOrdersPerTurn | 10 | Maximum number of market orders processed per player per turn; extras are silently dropped |
| turnsPerDay | 24 | Number of turns that make up one in-game day |
| shedCapacity | 100 | Max non-seed items the shed can hold; overflow at end-of-day drop is discarded |
| weedSpawnChance | 0.005 | Per-tile probability of a weed spawning on an empty unlocked tile during end-of-day refresh |
| townShopUnlockInterval | 3 | Days between successive town shop unlocks |
| townShopSellInterval | 2 | Turns between consumption ticks by every unlocked town shop |
| townCenterSellInterval | 6 | Turns between consumption ticks by the town center |
| seed | null | Optional input seed for deterministic episode generation; cleared from config after read so it stays out of agent observations |

