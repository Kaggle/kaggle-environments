# Kargo: Getting Started

This guide walks you through building an agent, testing it locally, and submitting it to the Kargo competition on Kaggle.

For the full rules -- the city, traffic, delivery windows, fleet, drivers, the shipper market, the auction and every observation field -- see [README.md](README.md).

## Game Overview

Kargo is a last-mile delivery business sim for 2 or 4 players over 60 days. Each player runs a fleet of trucks out of four shared warehouses and competes on final **net worth** (cash + owned-fleet book value − loan principal − debt).

- **A day is 8 turns.** Three overnight steps -- `CAPEX` (fleet), `LABOR` (drivers), `CONTRACTS` (auction) -- then five two-hour driving blocks from 08:00 to 18:00. 60 days is 480 turns.
- **Freight comes from one external shipper.** Its demand grows over the episode along a hidden path, and its prices rise when demand outruns the field's trucks and fall when the field over-builds. Unsold or undelivered packages come back the next night at a markup.
- **You win freight in a sealed-bid reverse auction.** Lowest ask wins and is paid its own ask. A lot is a fraction of a truck-day out of one `(warehouse, district)` territory; a truck carries one territory's freight at a time.
- **You see the territory, not the doors.** Addresses, service times and delivery windows land at 08:00, and your lots wait at their warehouses. You load them onto trucks and sequence the routes; anything still at a warehouse at 18:00 fails.
- **Costs never stop.** Trucks cost $60-92 a day owned, drivers are paid for the minutes their truck works (and a retainer if benched), and every undelivered package costs $45.
- **Starting position:** $12,000 cash, a `VAN`, a `VAN` and a `STEP`, one driver each. Net worth $150,000.

## Your Agent

Your agent is a function `agent(obs, config)` that returns an action dict. What it returns depends on `obs["phase"]`:

| Phase | Return |
|---|---|
| `CAPEX` | `{"fleet": [[op, arg], ...], "fuel": [["BULK_REFUEL", truck], ...], "stage": {truck: warehouse}}` |
| `LABOR` | `{"labor": [["HIRE", candidate, wage], ["WAGE", driver, wage], ["ASSIGN", driver, truck], ["POACH", player, driver, wage], ["FIRE", driver]]}` |
| `CONTRACTS` | `{"bids": [[lot, ask], ...], "standing_bids": [[account, rate, term], ...], "max_lots": truck_days}` |
| `DRIVING` | `{"trucks": {truck: {"load": [lot, ...], "route": [...], "wait_cap": minutes, "on_missed_window": "SKIP" or "ATTEMPT", "then": "RETURN", "hold": bool}}, "abandon": [ids]}` |

An empty dict is always legal. Plans persist across driving blocks: omit a truck to let it carry on. Entries of the wrong shape are dropped, never fatal (see README § Actions).

**Observation fields** you will use most:

- `phase`, `day` (0-based; day 0 is a Monday), `block` (0-4), `minute` (since 08:00), `player` (your index)
- `market` -- tonight's `listings` and standing `accounts` (each with `warehouse`, `district`, `anchor`, `packages`, `truck_days`, `parcel_units`, `reserve`, `retry`, window counts), `used` trucks, `rentals`, labor `candidates` (id, name, résumé, asking wage), `service_interval_km`
- `public` -- per player: `cash`, `debt`, `net_worth`, `fleet`, `drivers` (never wages), `standing`, last 3 days' `results`
- `history` -- last night's `auction` awards and `bids`, the latest `capex` and `labor` logs, live `standing` accounts
- `city` (static road graph, `warehouses`) and `traffic` (`congestion` per edge, `incidents`, `weather`, `forecast`)
- `private` -- yours only: `trucks` (node, status, clock, fuel, `km_since_service`, carried door ids, `lots` on board, route), `drivers` (with wages), today's `segments` and `addresses` (windows, noisy `service_estimate`), `lots` (with `status`: `AT_DOCK`, `ON_TRUCK`, `DONE`), `events`, `sightings`, `day_report`

### Starter agent

A complete, minimal agent: crew every truck, bid 10% under reserve on the best-paying territories, load each territory onto one truck at 08:00, and drive its segments nearest-first.

```python
CAPACITY = {"VAN": 200, "STEP": 340}


def agent(obs, config=None):
    phase = obs["phase"]
    me = obs["private"]
    trucks = me["trucks"]

    if phase == "CAPEX":
        # Service anything flagged at 18:00 (or about to be), top up fuel.
        due = obs["market"]["service_interval_km"]
        fleet = [["SERVICE", t["id"]] for t in trucks if t["km_since_service"] >= due]
        fuel = [["BULK_REFUEL", t["id"]] for t in trucks if t["fuel"] < 300]
        return {"fleet": fleet, "fuel": fuel}

    if phase == "LABOR":
        free = [d for d in me["drivers"] if not d["truck"]]
        empty = [t for t in trucks if not t["driver"]]
        labor = [["ASSIGN", d["id"], t["id"]] for d, t in zip(free, empty)]
        short = len(empty) - len(free)
        pool = sorted(obs["market"]["candidates"], key=lambda c: -c["resume"])
        # A candidate's hidden reservation is at most 1.10x the ask.
        labor += [["HIRE", c["id"], round(c["asking"] * 1.1 + 1)] for c in pool[: max(0, short)]]
        return {"labor": labor}

    crewed = [t for t in trucks if t["driver"] and t["status"] not in ("DISABLED", "ORDERED")]

    if phase == "CONTRACTS":
        # Best paying per truck-day first; at most one territory per truck,
        # at most one truck-day in each.
        lots = sorted(obs["market"]["listings"], key=lambda lot: -lot["reserve"] / lot["truck_days"])
        bids, load = [], {}
        for lot in lots:
            pair = (lot["warehouse"], lot["district"])
            if pair not in load and len(load) >= len(crewed):
                continue
            if load.get(pair, 0.0) + lot["truck_days"] > 1.0:
                continue
            load[pair] = load.get(pair, 0.0) + lot["truck_days"]
            bids.append([lot["id"], round(lot["reserve"] * 0.9, 2)])
        return {"bids": bids, "max_lots": len(crewed)}

    # DRIVING: at 08:00 load each territory onto one truck, biggest territory
    # on the biggest deck, then visit its segments nearest-first. Plans persist.
    if obs["block"] != 0:
        return {}
    territories = {}
    for lot in me["lots"]:
        if lot["status"] == "AT_DOCK":
            territories.setdefault((lot["warehouse"], lot["district"]), []).append(lot)
    territories = sorted(territories.values(), key=lambda ls: -sum(lot["parcel_units"] for lot in ls))
    crewed.sort(key=lambda t: -CAPACITY[t["type"]])
    lot_of = {a["id"]: a["lot"] for a in me["addresses"]}
    plans = {}
    for truck, lots in zip(crewed, territories):
        room, load = CAPACITY[truck["type"]], []
        for lot in lots:
            if lot["parcel_units"] <= room:
                room -= lot["parcel_units"]
                load.append(lot["id"])
        mine = [s for s in me["segments"] if any(lot_of.get(a) in load for a in s["pending"])]
        here, route = (0.0, 0.0), []
        while mine:
            nxt = min(mine, key=lambda s: abs(s["pos"][0] - here[0]) + abs(s["pos"][1] - here[1]))
            mine.remove(nxt)
            route.append(nxt["id"])
            here = nxt["pos"]
        if load:
            plans[truck["id"]] = {"load": load, "route": route, "wait_cap": 15}
    return {"trucks": plans}
```

It never expands its fleet, takes no standing accounts and ignores delivery windows.

## Test Locally

Install the environment from PyPI (any release that includes Kargo):

```bash
pip install -U kaggle-environments
```

Run a game from Python or a notebook -- pass agent functions directly, or paths to `.py` files:

```python
from kaggle_environments import make

env = make("kargo", debug=True)
env.run(["main.py", "greedy", "greedy", "random"])  # 2 or 4 players

for i, s in enumerate(env.steps[-1]):
    print(f"Player {i}: net worth={s.reward:,.0f}, status={s.status}")

# Dump a replay for the visualizer or offline analysis
import json
with open("replay.json", "w") as f:
    json.dump(env.toJSON(), f)
```

A full 60-day game takes about 15 seconds plus your agent's time. Built-in agents: `"idle"` (does nothing), `"random"`, and `"greedy"` (bids at reserve; the baseline to beat). Pass `configuration={"seed": 7}` for a repeatable city, and `{"episodeSteps": 41}` for a short 5-day game while iterating.

Each turn allows 3 seconds, plus a 120-second overage bank for the whole episode.

## Set Up the Kaggle CLI

Install the CLI:

```bash
pip install kaggle
```

You'll need a Kaggle account -- sign up at https://www.kaggle.com if you don't have one. Then download your API credentials at https://www.kaggle.com/settings/api by clicking **"Generate New Token"** under the "API" section.

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
kaggle competitions list -s "kargo"
```

## Find the Competition

```bash
kaggle competitions list -s "kargo"
kaggle competitions pages kargo
kaggle competitions pages kargo --content
```

## Accept the Competition Rules

Before submitting, you **must** accept the rules on the Kaggle website. Navigate to `https://www.kaggle.com/competitions/kargo` and click **"Join Competition"**.

Verify you've joined:

```bash
kaggle competitions list --group entered
```

## Submit Your Agent

Your submission must have a `main.py` at the root with an `agent` function.

**Single file agent:**

```bash
kaggle competitions submit kargo -f main.py -m "Starter v1"
```

**Multi-file agent** -- bundle into a tar.gz with `main.py` at the root:

```bash
tar -czf submission.tar.gz main.py helper.py model_weights.pkl
kaggle competitions submit kargo -f submission.tar.gz -m "Multi-file agent v1"
```

**Notebook submission:**

```bash
kaggle competitions submit kargo -k YOUR_USERNAME/kargo-agent -f submission.tar.gz -v 1 -m "Notebook agent v1"
```

## Monitor Your Submission

```bash
kaggle competitions submissions kargo
```

Note the submission ID from the output -- you'll need it for episodes.

## List Episodes

```bash
kaggle competitions episodes <SUBMISSION_ID>
kaggle competitions episodes <SUBMISSION_ID> -v   # CSV
```

## Download Replays and Logs

```bash
kaggle competitions replay <EPISODE_ID> -p ./replays
kaggle competitions logs <EPISODE_ID> 0            # your agent's logs, by seat index
```

## Check the Leaderboard

```bash
kaggle competitions leaderboard kargo -s
```

## Typical Workflow

```bash
# Test locally
python -c "
from kaggle_environments import make
env = make('kargo', debug=True)
env.run(['main.py', 'greedy', 'greedy', 'greedy'])
print([(i, round(s.reward)) for i, s in enumerate(env.steps[-1])])
"

# Submit, then follow it
kaggle competitions submit kargo -f main.py -m "v1"
kaggle competitions submissions kargo
kaggle competitions episodes <SUBMISSION_ID>
kaggle competitions replay <EPISODE_ID>
kaggle competitions leaderboard kargo -s
```
