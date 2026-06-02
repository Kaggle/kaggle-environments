# Kaggriculture (Beginner): Getting Started

This guide walks you through building an agent, testing it locally, and submitting it to the Kaggriculture (Beginner) competition on Kaggle.

For full game rules, the crop table, watering rules, and turn-processing order, see [README.md](README.md).

## Game Overview

Kaggriculture (Beginner) is a two-player farming sim. Each player manages a small farm and competes to earn the most coins by buying seeds, planting, watering, and harvesting crops over a fixed season. Produce is auto-sold to the market at fixed prices the moment it is harvested.

- **Farm** — each player has a `boardSize` × `boardSize` grid (default 5 × 5). Every tile is unlocked from the start
- **Starting bank** — `startingMoney` defaults to $150
- **Farmer** — one farmer per player; one action per turn (no hired hands, no animals, no land-buying)
- **Crops** — Wheat, Carrot, Tomato (ongoing), Strawberry (ongoing), Melon. Each has its own seed cost, growth time, yield curve, and fixed sale price (see the Object Types table in [README.md](README.md))
- **Watering bonus** — for one-time crops, watering during the bonus window (starting at half `max_yield_day`, rounded up) adds 1 unit per day to harvestable yield. Ongoing crops produce a fixed 1 unit per scheduled production
- **Watering** — plants must be watered every day, including the day they are planted. The planting day counts as the first unwatered day, so a brand-new seed dies if it isn't watered by the end of its planting day AND the next day. After the first watering, the plant tolerates one missed day before dying (two consecutive unwatered end-of-day refreshes kill it)
- **Decay** — once a plant passes its max lifespan, `yield_units` drops by 1 every other turn until 0, at which point the plant disappears
- **Market** — fixed prices for seeds (BUY) and harvested produce (auto-sold on HARVEST). There is no separate SELL action, no inventory, no shed, and no dynamic market. Each turn, at most `maxMarketOrdersPerTurn` (default 10) orders are processed per player; extras are silently dropped
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
  - `seeds` — `{crop: count}` seed inventory; consumed directly by `PLANT`
  - `farmer` — `[x, y]` position of the farmer (x = column, y = row)
  - `tiles` — 2D array indexed `tiles[y][x]`. Each tile is `None` (empty) or a plant dict (`crop`, `planted_day`, `watered_today`, `consecutive_unwatered`, `yield_units`, `max_lifespan_step`)

**Action format:**

```py
{
  "farmer": [op, ...args],          # one farmer op this turn
  "market": [[op, ...args], ...],   # ordered list of market orders, capped at maxMarketOrdersPerTurn
}
```

Farmer ops:
- Movement: `"NORTH"`, `"SOUTH"`, `"EAST"`, `"WEST"`, `"PASS"`
- Plants: `"PLANT"`, `"WATER"`, `"HARVEST"`

Market ops: `["BUY_SEED", crop, n]`. Invalid actions are silent no-ops.

**Example — Wheat Loop:**

For wheat (`first_yield_day = 2`, `max_yield_day = 4`), the bonus watering window starts at day 2. We plant, water during the window, and harvest at day 2 or later. Harvested wheat auto-sells immediately.

```python
def agent(obs):
    player = obs["player"]
    me = obs["farms"][player]
    fx, fy = me["farmer"]
    tile = me["tiles"][fy][fx]

    market = []
    if me["seeds"].get("WHEAT", 0) == 0 and me["money"] >= 10:
        market.append(["BUY_SEED", "WHEAT", 1])

    if tile is None and me["seeds"].get("WHEAT", 0) > 0:
        return {"farmer": ["PLANT", "WHEAT"], "market": market}
    if isinstance(tile, dict):
        crop_age = obs["day"] - tile["planted_day"]
        if crop_age >= 2:  # WHEAT first_yield_day = 2; harvest as soon as possible
            return {"farmer": ["HARVEST"], "market": market}
        if not tile["watered_today"]:
            return {"farmer": ["WATER"], "market": market}

    return {"farmer": ["PASS"], "market": market}
```

For per-crop yield/cost details (Carrot, Tomato, Strawberry, and Melon), see the Object Types and Quick Start sections in [README.md](README.md).

## Test Locally

Install the environment from PyPI (any recent release that includes Kaggriculture Beginner):

```bash
pip install -U kaggle-environments
```

Run a game from Python or a notebook — you can pass agent functions directly, or paths to `.py` files:

```python
from kaggle_environments import make

env = make("kaggriculture_beginner", configuration={"episodeSteps": 720}, debug=True)
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

Three built-in agents are available by name: `"pass"`, `"random"`, and `"starter"` (a deterministic carrot-loop baseline).

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
kaggle competitions list -s "kaggriculture-beginner"
```

## Find the Competition

```bash
kaggle competitions list -s "kaggriculture-beginner"
kaggle competitions pages kaggriculture-beginner
kaggle competitions pages kaggriculture-beginner --content
```

## Accept the Competition Rules

Before submitting, you **must** accept the rules on the Kaggle website. Navigate to `https://www.kaggle.com/competitions/kaggriculture-beginner` and click **"Join Competition"**.

Verify you've joined:

```bash
kaggle competitions list --group entered
```

## Download Competition Data

```bash
kaggle competitions download kaggriculture-beginner -p kaggriculture-beginner-data
```

## Submit Your Agent

Your submission must have a `main.py` at the root with an `agent` function.

**Single file agent:**

```bash
kaggle competitions submit kaggriculture-beginner -f main.py -m "Wheat loop v1"
```

**Multi-file agent** — bundle into a tar.gz with `main.py` at the root:

```bash
tar -czf submission.tar.gz main.py helper.py model_weights.pkl
kaggle competitions submit kaggriculture-beginner -f submission.tar.gz -m "Multi-file agent v1"
```

**Notebook submission:**

```bash
kaggle competitions submit kaggriculture-beginner -k YOUR_USERNAME/kaggriculture-beginner-agent -f submission.tar.gz -v 1 -m "Notebook agent v1"
```

## Monitor Your Submission

Check submission status:

```bash
kaggle competitions submissions kaggriculture-beginner
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
kaggle competitions leaderboard kaggriculture-beginner -s
```

## Typical Workflow

```bash
# Test locally
python -c "
from kaggle_environments import make
env = make('kaggriculture_beginner', debug=True)
env.run(['main.py', 'random'])
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
"

# Submit
kaggle competitions submit kaggriculture-beginner -f main.py -m "v1"

# Check status
kaggle competitions submissions kaggriculture-beginner

# Review episodes
kaggle competitions episodes <SUBMISSION_ID>

# Download replay and logs
kaggle competitions replay <EPISODE_ID>
kaggle competitions logs <EPISODE_ID> 0

# Check leaderboard
kaggle competitions leaderboard kaggriculture-beginner -s
```
