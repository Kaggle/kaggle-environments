# Pyxis Portfolio Challenge: Getting Started

This guide walks you through building an agent, testing it locally, and submitting it to the `gsk-simulation` competition on Kaggle.

For full game rules, the asset pipeline, market mechanics, and the observation layout, see [README.md](README.md).

## Game Overview

Pyxis is a two-player pharmaceutical R&D game. Each player runs a drug-development portfolio, deciding which assets to fund through clinical trials, which to abandon, and how to spend against a shared market — while a rival does the same in the same indications.

- **Horizon** — 100 acting steps. Before your first action the market is pre-rolled 500 steps under a do-nothing policy and the clock rebased to 0, so you inherit a mature, populated market
- **Starting cash** — £5B. All money is GBP
- **Portfolio** — up to 40 asset slots (equilibrium ~35). Each asset sits in one of 3 therapeutic areas and one of 3 indications within it (9 indications total)
- **Pipeline** — Phase 1 → 2 → 3 → regulatory approval, then on-market revenue until patent expiry. Each phase has a cost, a duration, and a probability of success (PTRS). Roughly 80% of assets fail
- **Hidden PTRS** — the true per-phase PTRS is never observed. Each asset arrives with one free noisy reading, and that estimate is what the observation shows; outcomes roll against the hidden truth. Buy more readings to sharpen it (a single sample has ~21% mean absolute error)
- **Clinical sites** — a hard cap on concurrent trials. You start with 4. A site frees the moment its asset leaves development, including between phases. Excess trial starts become costless no-ops, arbitrated by ascending asset index
- **Shared market** — both players compete in the same indications. Revenue is scaled by `1 / n^α` as drugs pile into an indication; the first entrant is lightly penalised and later entrants ramp to the full `α = 2.0` by the 4th
- **Intelligence** — an opponent advancing a phase leaks to you with probability 20%/50%/70% (Phase 1→2 / 2→3 / 3→Approval). Marketing spend leaks too
- **Auctions** — business-development assets (pre-progressed, skip early development) and clinical sites are sold by first-price sealed bid. Highest bid wins and pays its own bid. **There is no affordability cap: an overbid can bankrupt you**
- **Win condition** — bankruptcy-aware. A bankrupt agent loses; if both go bankrupt the one that survived longer wins; same-step bankruptcy is a draw. Otherwise the higher net cash flow wins

## Your Agent

Your agent is a function that receives an observation and returns an action dict.

**Observation fields:**
- `obs` — the flattened engine observation: a list of 1534 floats with 40 asset slots and the shipped config. See [README.md](README.md) for the layout, and "Decoding the observation" below for turning it into a dict
- `actionMask` — per-head masks for the discrete heads. `investments` is 40×3, `ptrs_research` 43×11, `demand_creation` 9×2, `brand_equity` 40×2, `upgrade` length 2. Indexed `mask[slot][choice]`
- `agentIndex` — your seat (0 or 1)
- `cash` — your cash this step (GBP)
- `enpv` — your expected net present value this step (GBP)
- `bankrupt` — whether you are bankrupt (cash < 0)
- `step` — current step (0-indexed; supplied by the kaggle-environments framework)
- `remainingOverageTime` — your remaining overage time budget (seconds)

Your observation contains only your own portfolio. The opponent's is never in it — what you learn about them arrives through the alert feed.

**Action format:**

A dict with one entry per action head. Every head is optional: missing or `null` heads are filled with their no-op, so `{}` is a legal pass.

```python
action = {
    "investments":     [...],   # 40 ints, 0 = nothing / 1 = invest / 2 = drop
    "bd_bids":         [...],   # 3 floats, cash bid per BD slot (£M, 0 = pass)
    "ptrs_research":   [...],   # 43 ints, 0-10 readings (40 assets, then 3 BD slots)
    "demand_creation": [...],   # 9 ints, binary, per indication
    "brand_equity":    [...],   # 40 ints, binary, per asset
    "upgrade":         0,       # int, 1 = buy a clinical site
    "site_bid":        [...],   # 1 float, site-auction cash bid (£M, 0 = pass)
}
```

**Illegal actions forfeit the match.** Your status becomes `INVALID`, your reward `None`, and your opponent is awarded 1.0. An action is illegal if it is not a dict, names a head that doesn't exist, or picks a choice the mask forbids. The two cash-bid heads (`bd_bids`, `site_bid`) are unmasked, so the action space itself is the validator — a wrong length, a negative bid, or a bid above the £100B cap forfeits just the same.

Raising an exception (`ERROR`) or exceeding `actTimeout` (`TIMEOUT`) forfeits the same way.

This agent invests wherever the mask says it can:

```python
def agent(observation, configuration):
    masks = observation["actionMask"]
    # mask[slot][1] is True when "invest" is legal and affordable for that slot.
    return {"investments": [1 if slot[1] else 0 for slot in masks["investments"]]}
```

Affordability in the masks is first-order: each option is checked on its own, ignoring what the rest of your action costs. Taking several masked-legal actions in one step can still overspend.

### Decoding the observation

`obs` is a flat vector, but the engine that produced it ships inside the environment and is importable from your submission. That gives you a dict view without reimplementing the layout:

```python
_ENV = None


def _engine():
    global _ENV
    if _ENV is None:
        import kaggle_environments.envs.pyxis.pyxis  # puts the engine on sys.path
        from pyxis_portfolio_challenge.config import config
        from pyxis_portfolio_challenge.environment.env_factory import (
            _build_multi_agent_env_kwargs,
        )
        from pyxis_portfolio_challenge.environment.multi_agent_training_gym import (
            MultiAgentInvestmentGameEnv,
        )

        _ENV = MultiAgentInvestmentGameEnv(
            **_build_multi_agent_env_kwargs(
                flatten_obs=True,
                num_agents=2,
                assets_dir=config.evaluation_data_dir,
                bd_assets_dir=config.multi_agent.bd_eval_assets_dir,
            )
        )
    return _ENV


def agent(observation, configuration):
    import numpy as np

    view = _engine().unflatten_to_dict_obs(np.asarray(observation["obs"], dtype=np.float32))
    # view: {"cash", "time", "assets", "bd_market", "indication_markets", "alerts"}
    masks = observation["actionMask"]
    investments = [0] * len(masks["investments"])
    for i, (asset, mask) in enumerate(zip(view["assets"], masks["investments"])):
        # state 0 = Idle. Fund the assets whose noisy PTRS looks best.
        if mask[1] and asset["state"] == 0 and asset["trials"][0]["ptrs"] > 0.5:
            investments[i] = 1
    return {"investments": investments}
```

This instance is a decoder only — it is not the live match, and stepping it does nothing to your game. Build it once and reuse it; construction is the expensive part.

## Test Locally

Install the environment from PyPI (any recent release that includes Pyxis):

```bash
pip install -U kaggle-environments
```

Run a game from Python or a notebook — you can pass agent functions directly, or paths to `.py` files:

```python
from kaggle_environments import make

env = make("pyxis", configuration={"seed": 42}, debug=True)
env.run([agent, "knapsack"])  # or env.run(["main.py", "random"]) to load from a file

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

Three built-in agents are available by name: `"knapsack"` (a budget-optimising heuristic, the strong baseline), `"random"`, and `"do_nothing"`.

An exception in your agent forfeits the match: status `ERROR`, reward `None`, opponent awarded 1.0. Use `debug=True` while developing to see the traceback instead of just the status.

A full match is 101 steps and takes roughly 15 seconds against `do_nothing`. Pass a `seed` to make a match reproducible; the seed is scrubbed from the configuration so neither agent can read it.

`make("pyxis")` also exposes the standard kaggle-environments helpers:

```python
from kaggle_environments import evaluate

# Several matches at once; returns a list of [reward_0, reward_1] rows.
rewards = evaluate("pyxis", [agent, "knapsack"], {"seed": 11}, num_episodes=10)

# Gym-style single-agent loop, for RL training.
trainer = make("pyxis").train([None, "knapsack"])
obs = trainer.reset()
obs, reward, done, info = trainer.step({})
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
kaggle competitions list -s "gsk-simulation"
```

## Find the Competition

```bash
kaggle competitions list -s "gsk-simulation"
kaggle competitions pages gsk-simulation
kaggle competitions pages gsk-simulation --content
```

## Accept the Competition Rules

Before submitting, you **must** accept the rules on the Kaggle website. Navigate to `https://www.kaggle.com/competitions/gsk-simulation` and click **"Join Competition"**.

Verify you've joined:

```bash
kaggle competitions list --group entered
```

## Download Competition Data

```bash
kaggle competitions download gsk-simulation -p gsk-simulation-data
```

## Submit Your Agent

Your submission must have a `main.py` at the root with an `agent` function.

**Single file agent:**

```bash
kaggle competitions submit gsk-simulation -f main.py -m "Mask-following baseline v1"
```

**Multi-file agent** — bundle into a tar.gz with `main.py` at the root:

```bash
tar -czf submission.tar.gz main.py helper.py model_weights.pkl
kaggle competitions submit gsk-simulation -f submission.tar.gz -m "Multi-file agent v1"
```

**Notebook submission:**

```bash
kaggle competitions submit gsk-simulation -k YOUR_USERNAME/pyxis-agent -f submission.tar.gz -v 1 -m "Notebook agent v1"
```

## Monitor Your Submission

Check submission status:

```bash
kaggle competitions submissions gsk-simulation
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
kaggle competitions leaderboard gsk-simulation -s
```

## Typical Workflow

```bash
# Test locally
python -c "
from kaggle_environments import make
env = make('pyxis', debug=True)
env.run(['main.py', 'knapsack'])
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
"

# Submit
kaggle competitions submit gsk-simulation -f main.py -m "v1"

# Check status
kaggle competitions submissions gsk-simulation

# Review episodes
kaggle competitions episodes <SUBMISSION_ID>

# Download replay and logs
kaggle competitions replay <EPISODE_ID>
kaggle competitions logs <EPISODE_ID> 0

# Check leaderboard
kaggle competitions leaderboard gsk-simulation -s
```
