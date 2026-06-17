# Planet Wars: Getting Started

This guide walks you through building an agent, testing it locally, and submitting it to the Planet Wars competition on Kaggle.

For full game rules, the map format, turn-order details, and combat resolution, see [README.md](README.md).

## Game Overview

Planet Wars is a two-player real-time strategy game ported bit-exactly from the 2009/2010 Google AI Challenge. Players battle for control of a galaxy of 15-30 planets by sending fleets of ships across a symmetric map.

- **Map** — 15-30 planets at fixed positions in continuous 2D space. Every map is either point-symmetric (180° rotation) or reflective about an axis, so neither player has a positional advantage
- **Planets** — owned by player 1, player 2, or no one (neutral). Owned planets generate ships each turn at their `growth_rate` (`[1, 5]`, or 0 for an optional central planet). Neutral planets do not grow
- **Home planets** — each player starts with one home planet (`HOME_SHIPS = 100`, `HOME_GROWTH = 5`). Other non-neutral planets start with 1-100 ships and growth 1-5
- **Fleets** — issued as `[source, dest, num_ships]` orders. Ships leave the source on departure and take `ceil(sqrt((x1-x2)^2 + (y1-y2)^2))` turns to arrive. Trip length is fixed at launch and does not change in flight
- **Order merging** — multiple orders from the same source to the same destination on the same turn are merged into a single fleet
- **Turn order** — each turn runs in this fixed order: collect orders → departure (apply orders, spawn fleets) → advancement (decrement `turns_remaining`, grow owned planets) → arrival (resolve battles for fleets with `turns_remaining == 0`)
- **Combat** — at a destination, sum incoming ships per owner. The largest force wins and ends up with `(largest - second_largest)` ships. On a tie at the top, the planet's prior owner is retained with zero ships
- **Win condition** — if only one player has any planets or fleets after a turn, they win. If the turn limit (default 200) is reached, the player with more total ships across planets + fleets wins; equal totals are a draw
- **Forfeit** — issuing any invalid order forfeits the game immediately. Invalid means `ships <= 0`, `source == dest`, an unknown planet id, a source you don't own, or the sum of ships sent from one planet this turn exceeds its current garrison

## Your Agent

Your agent is a function that receives an observation and returns a list of fleet orders.

**Observation fields:**
- `player` — your owner id in the global state (1 for agent 0, 2 for agent 1)
- `planets` — list of `[id, x, y, owner, num_ships, growth_rate]`. `owner` is `0` (neutral), `1`, or `2`. Shared between both agents — the map is fully observable from turn 0
- `fleets` — list of `[owner, num_ships, source_planet, dest_planet, total_trip, turns_remaining]` for every in-flight fleet
- `remainingOverageTime` — your remaining overage time budget (seconds)
- `step` — current turn (0-indexed; supplied by the kaggle-environments framework)

**Action format:**

A list of fleet orders. Each order is `[source_planet_id, dest_planet_id, num_ships]`. An empty list is a legal no-op turn.

```python
def agent(obs, config):
    # Send half of every owned planet's ships to its nearest non-self planet.
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

The module exports named tuples and a `distance` helper for ergonomics:

```python
from kaggle_environments.envs.planet_wars.planet_wars import Planet, Fleet, distance

def agent(obs, config):
    planets = [Planet(*p) for p in obs.get("planets", [])]
    fleets = [Fleet(*f) for f in obs.get("fleets", [])]
    me = obs.get("player", 1)
    # distance(p_a, p_b) -> ceil-Euclidean trip length in turns
    ...
```

For a fuller treatment of map symmetry, the Point-in-Time map format, and combat resolution, see [README.md](README.md).

## Test Locally

Install the environment from PyPI (any recent release that includes Planet Wars):

```bash
pip install -U kaggle-environments
```

Run a game from Python or a notebook — you can pass agent functions directly, or paths to `.py` files:

```python
from kaggle_environments import make

env = make("planet_wars", configuration={"episodeSteps": 200}, debug=True)
env.run([agent, "nearest_enemy"])  # or env.run(["main.py", "random"]) to load from a file

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

Three built-in agents are available by name: `"do_nothing"`, `"random"`, and `"nearest_enemy"`.

You can also pin the map for reproducible local runs by passing a `seed` (or a literal Point-in-Time map text starting with `"P "`) in the configuration:

```python
env = make("planet_wars", configuration={"seed": 42}, debug=True)
```

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
kaggle competitions list -s "planet_wars"
```

## Find the Competition

```bash
kaggle competitions list -s "planet_wars"
kaggle competitions pages planet_wars
kaggle competitions pages planet_wars --content
```

## Accept the Competition Rules

Before submitting, you **must** accept the rules on the Kaggle website. Navigate to `https://www.kaggle.com/competitions/planet_wars` and click **"Join Competition"**.

Verify you've joined:

```bash
kaggle competitions list --group entered
```

## Download Competition Data

```bash
kaggle competitions download planet_wars -p planet_wars-data
```

## Submit Your Agent

Your submission must have a `main.py` at the root with an `agent` function.

**Single file agent:**

```bash
kaggle competitions submit planet_wars -f main.py -m "Nearest-enemy baseline v1"
```

**Multi-file agent** — bundle into a tar.gz with `main.py` at the root:

```bash
tar -czf submission.tar.gz main.py helper.py model_weights.pkl
kaggle competitions submit planet_wars -f submission.tar.gz -m "Multi-file agent v1"
```

**Notebook submission:**

```bash
kaggle competitions submit planet_wars -k YOUR_USERNAME/planet-wars-agent -f submission.tar.gz -v 1 -m "Notebook agent v1"
```

## Monitor Your Submission

Check submission status:

```bash
kaggle competitions submissions planet_wars
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
kaggle competitions leaderboard planet_wars -s
```

## Typical Workflow

```bash
# Test locally
python -c "
from kaggle_environments import make
env = make('planet_wars', debug=True)
env.run(['main.py', 'nearest_enemy'])
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
"

# Submit
kaggle competitions submit planet_wars -f main.py -m "v1"

# Check status
kaggle competitions submissions planet_wars

# Review episodes
kaggle competitions episodes <SUBMISSION_ID>

# Download replay and logs
kaggle competitions replay <EPISODE_ID>
kaggle competitions logs <EPISODE_ID> 0

# Check leaderboard
kaggle competitions leaderboard planet_wars -s
```
