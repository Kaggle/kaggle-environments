# Kaggriculture: Getting Started

This guide walks you through building an agent, testing it locally, and submitting it to the Kaggriculture competition on Kaggle.

For full game rules, the crop / animal / shop tables, the price function, and turn-processing order, see [README.md](README.md).

## Game Overview

Kaggriculture is a two-player farming sim. Each player manages a farm and competes to earn the most coins by buying seeds and livestock, planting, watering, harvesting, raising animals, hiring help, and trading on a dynamic market over a fixed season.

- **Farm** — each player has a `boardSize` × `boardSize` grid (default 10 × 10) divided into four 5 × 5 quadrants. Only the NW quadrant is unlocked at the start; the other three (`NE`, `SW`, `SE`) can be bought via `BUY_LAND` for $1k / $2k / $4k respectively
- **Starting bank** — `startingMoney` defaults to $2000
- **Farmer & farm hands** — one main farmer per player, plus up to N hired hands per day. Hire cost is `farmHandCostMult * fib(n)` where `n` is the number of hires already made today; with the default `farmHandCostMult = 10` that's `10, 10, 20, 30, 50, 80, 130, 210, ...` and resets at the start of each day. Each unit independently acts every turn
- **Crops** — Wheat, Carrot, Tomato (ongoing), Strawberry (ongoing), Melon. Each has its own seed cost, growth time, yield curve, and base sale price (see the Object Types table in [README.md](README.md))
- **Watering bonus** — for one-time crops, watering during the bonus window (starting at `ceil(max_yield_day / 2)`) adds 1 unit per day to harvestable yield. `FERTILIZE` doubles that bonus for 3 days. For ongoing crops, scheduled production yields 1 by default, doubled to 2 if both fertilized and watered that day
- **Animals** — Goose (eggs, requires coop), Cow (milk, requires pasture), Sheep (wool, requires pasture). Must be fed wheat daily; `CARE` banks a yield bonus paid out on the next scheduled production; `COLLECT_FERTILIZER` gathers 1 fertilizer/animal/day
- **Watering / feeding** — plants must be watered and animals fed daily. Two consecutive missed end-of-day refreshes turn plants into weeds and cause animals to escape (unrecoverable). The planting day counts as the first unwatered day
- **Decay** — once a plant passes its max lifespan (one day after `max_yield_day` for one-time crops, one day after the cumulative production cap for ongoing crops), `yield_units` drops by 1 every other turn until 0, at which point the tile becomes a weed
- **Weeds** — every empty unlocked tile has a `weedSpawnChance` (default 0.005) of spawning a weed at end-of-day; clear with `DIG`
- **Shed** — non-seed inventory cap of 100 items. Items beyond the cap at end-of-day drop are discarded. Seeds live in their own slot (no cap, never picked up by `PICKUP` — `PLANT` consumes them directly)
- **Market** — fixed prices for seeds, animals, and `BUY_PRODUCT` orders; sale prices for harvested produce vary dynamically with market inventory (linear when inventory is low, log-decreasing when high). Fertilizer can only be bought, not sold. Each turn, at most `maxMarketOrdersPerTurn` (default 10) orders are processed per player; extras are silently dropped
- **Town** — town center always demands product (1 of each non-fertilizer product every `townCenterSellInterval` turns, default 6, scaling to 2× after day 10 and 4× after day 20). Additional shops unlock every `townShopUnlockInterval` days (default 3, random selection from the remaining pool); each unlocked shop consumes one of every product it demands every `townShopSellInterval` turns (default 2, single-product shops consume 2×) — see the Town Buildings table in [README.md](README.md)
- **Season length** — 24 turns per day × 30 days = 720 turns by default
- **Win condition** — most coins in the bank at the end of the season; ties are possible

## Your Agent

Your agent is a function that receives an observation and returns an action dict.

**Observation fields:**
- `player` — your player index (0 or 1)
- `step` — current turn (0-indexed; supplied by the kaggle-environments framework)
- `day` — current in-game day (0-indexed)
- `hour` — turn within the day (0-indexed, 0..`turnsPerDay`-1)
- `farms` — list with one entry per player (public; both farms visible). Each entry has:
  - `money` — current bank balance
  - `tiles` — 2D array indexed `tiles[y][x]`. Each tile is `None` (empty unlocked), `"LOCKED"` (locked quadrant), a plant dict (`kind="PLANT"`, `crop`, `planted_day`, `watered_today`, `consecutive_unwatered`, `yield_units`, `max_lifespan_step`, `fertilized_until_day`), a weed dict (`kind="WEED"`), or an animal-structure dict (`kind="COOP"` or `"PASTURE"`, optionally with `animal`, `placed_day`, `yield_units`, `fed_today`, `consecutive_unfed`, `cared_today`, `fertilizer_available`, `pending_care_bonus`)
  - `farmer` — `[x, y]` position of the main farmer (x = column, y = row)
  - `hands` — list of `[x, y]` positions for any hired farm hands this day
  - `unlocked_quadrants` — subset of `["NW", "NE", "SW", "SE"]`
  - `hires_today` — number of hands hired so far today (drives the next HIRE cost)
- `private` — your player only; opponent's private state is hidden:
  - `shed` — `{item: count}` for harvested produce, animals, and fertilizer in storage
  - `seeds` — `{crop: count}`; seeds live in their own slot and are consumed directly by `PLANT`
  - `inventories` — `[main_farmer_inv, hand1_inv, ...]`; per-unit inventories carried in the field
- `market` — shared:
  - `inventory` — `{product: int}` current market supply
  - `prices` — `{product: int}` current per-unit sale price (rounded, floor 1)
- `town` — shared: `unlocked_shops` — list of currently-active shop names

**Action format:**

```py
{
  "farmer": [op, ...args],          # one main-farmer op this turn
  "hands":  [[op, ...args], ...],   # one op per hired hand, in hands order
  "market": [[op, ...args], ...],   # ordered list of market orders, capped at maxMarketOrdersPerTurn
}
```

Farmer / hand ops:
- Movement: `"NORTH"`, `"SOUTH"`, `"EAST"`, `"WEST"`, `"PASS"`
- Shed / inventory: `"PICKUP" <item> [n]` (from shed), `"PLACE" <item> [n]` (places an animal on a matching structure when standing on it, or drops items into the shed when adjacent to it)
- Plants: `"PLANT" <crop>`, `"WATER"`, `"HARVEST"`, `"FERTILIZE"`
- Animals: `"BUILD_COOP"`, `"BUILD_PASTURE"`, `"FEED"`, `"COLLECT_FERTILIZER"`, `"CARE"`
- Terrain: `"DIG"` (removes a plant, weed, coop, or pasture from the current tile)

Market ops: `["BUY_SEED", crop, n]`, `["BUY_PRODUCT", item, n]`, `["BUY_ANIMAL", animal, n]`, `["SELL", item, n]`, `["HIRE"]`, `["BUY_LAND"]`. Invalid actions are silent no-ops.

**Example — Wheat Loop:**

For wheat (`first_yield_day = 2`, `max_yield_day = 4`), the bonus watering window is days 2–4. We plant, water during the window, and harvest at day 2 or later.

```python
def agent(obs):
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    fx, fy = me["farmer"]
    tile = me["tiles"][fy][fx]

    market = []
    if private["seeds"].get("WHEAT", 0) == 0 and me["money"] >= 10:
        market.append(["BUY_SEED", "WHEAT", 1])
    # Sell any wheat sitting in the shed.
    wheat_in_shed = private["shed"].get("WHEAT", 0)
    if wheat_in_shed > 0:
        market.append(["SELL", "WHEAT", wheat_in_shed])

    if tile is None and private["seeds"].get("WHEAT", 0) > 0:
        return {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": market}
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop_age = obs["day"] - tile["planted_day"]
        if crop_age >= 2:  # WHEAT first_yield_day = 2; harvest as soon as possible
            return {"farmer": ["HARVEST"], "hands": [], "market": market}
        if not tile["watered_today"]:
            return {"farmer": ["WATER"], "hands": [], "market": market}

    return {"farmer": ["PASS"], "hands": [], "market": market}
```

For a fuller example (and per-crop yield/cost details for Carrot, Tomato, Strawberry, and Melon), see the Object Types and Quick Start sections in [README.md](README.md).

## Test Locally

Install the environment from PyPI (any recent release that includes Kaggriculture):

```bash
pip install -U kaggle-environments
```

Run a game from Python or a notebook — you can pass agent functions directly, or paths to `.py` files:

```python
from kaggle_environments import make

env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
env.run([agent, "random"])  # or env.run(["main.py", "random"]) to load from a file

# View result
final = env.steps[-1]
for i, s in enumerate(final):
    print(f"Player {i}: reward={s.reward}, status={s.status}")

# Render in a notebook
env.render(mode="ipython", width=1200, height=800)

# Or dump a replay JSON for the visualizer / offline analysis
import json
with open("replay.json", "w") as f:
    json.dump(env.toJSON(), f)
```

Three built-in agents are available by name: `"pass"`, `"random"`, and `"starter"` (a deterministic baseline).

## Set Up the Kaggle CLI

Install the CLI:

```bash
pip install kaggle
```

You'll need a Kaggle account — sign up at https://www.kaggle.com if you don't have one. Then download your API credentials at https://www.kaggle.com/settings/api by clicking **"Generate New Token"** under the "API" section.

**Recommended: API token file.** Save the token string to `~/.kaggle/access_token`:

```bash
mkdir -p ~/.kaggle
# Paste the token from the Kaggle settings UI into this file
nano ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Alternative auth methods:
- **OAuth (browser flow):** `kaggle auth login`
- **Environment variable:** `export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx`

Verify the CLI is wired up:

```bash
kaggle competitions list -s "kaggriculture"
```

## Find the Competition

```bash
kaggle competitions list -s "kaggriculture"
kaggle competitions pages kaggriculture
kaggle competitions pages kaggriculture --content
```

## Accept the Competition Rules

Before submitting, you **must** accept the rules on the Kaggle website. Navigate to `https://www.kaggle.com/competitions/kaggriculture` and click **"Join Competition"**.

Verify you've joined:

```bash
kaggle competitions list --group entered
```

## Download Competition Data

```bash
kaggle competitions download kaggriculture -p kaggriculture-data
```

## Submit Your Agent

Your submission must have a `main.py` at the root with an `agent` function.

**Single file agent:**

```bash
kaggle competitions submit kaggriculture -f main.py -m "Wheat loop v1"
```

**Multi-file agent** — bundle into a tar.gz with `main.py` at the root:

```bash
tar -czf submission.tar.gz main.py helper.py model_weights.pkl
kaggle competitions submit kaggriculture -f submission.tar.gz -m "Multi-file agent v1"
```

**Notebook submission:**

```bash
kaggle competitions submit kaggriculture -k YOUR_USERNAME/kaggriculture-agent -f submission.tar.gz -v 1 -m "Notebook agent v1"
```

## Monitor Your Submission

Check submission status:

```bash
kaggle competitions submissions kaggriculture
```

Note the submission ID from the output — you'll need it for episodes.

## List Episodes

Once your submission has played some games:

```bash
kaggle competitions episodes <SUBMISSION_ID>
```

CSV output for scripting:

```bash
kaggle competitions episodes <SUBMISSION_ID> -v
```

## Download Replays and Logs

Download the replay JSON for an episode (for visualization or analysis):

```bash
kaggle competitions replay <EPISODE_ID>
kaggle competitions replay <EPISODE_ID> -p ./replays
```

Download agent logs to debug your agent's behavior:

```bash
# Logs for the first agent (index 0)
kaggle competitions logs <EPISODE_ID> 0

# Logs for the second agent (index 1)
kaggle competitions logs <EPISODE_ID> 1 -p ./logs
```

## Check the Leaderboard

```bash
kaggle competitions leaderboard kaggriculture -s
```

## Typical Workflow

```bash
# Test locally
python -c "
from kaggle_environments import make
env = make('kaggriculture', debug=True)
env.run(['main.py', 'random'])
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
"

# Submit
kaggle competitions submit kaggriculture -f main.py -m "v1"

# Check status
kaggle competitions submissions kaggriculture

# Review episodes
kaggle competitions episodes <SUBMISSION_ID>

# Download replay and logs
kaggle competitions replay <EPISODE_ID>
kaggle competitions logs <EPISODE_ID> 0

# Check leaderboard
kaggle competitions leaderboard kaggriculture -s
```
