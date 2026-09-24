# GSK Pyxis Portfolio Challenge

Two pharma companies compete to build the most valuable drug-development portfolio. The Kaggle environment is `pyxis`; the competition is `gsk-simulation`. To build and submit an agent, start with [AGENTS.md](AGENTS.md) — this file is the rules reference.

## Overview

Each player manages a portfolio of R&D assets, funding each through clinical trials toward market. Outcomes are stochastic and delayed: most assets fail (~80% attrition), results arrive phases after the money is spent, and cash committed to one asset constrains every other option. Both players sell into the same indication markets, bid for the same business-development assets, and receive noisy intelligence about each other's pipelines. After 100 steps the player with the higher net cash flow wins; a bankrupt player loses.

You can play against an AI opponent at [gsk.ai/pyxis-portfolio-challenge](https://gsk.ai/pyxis-portfolio-challenge).

## Game Parameters

- 2 players
- 100 acting steps. Before the first action the market is advanced 500 steps under a do-nothing policy and the clock is rebased to 0, so play starts on a mature, populated market
- Starting cash: £5B. All money is GBP
- Up to 40 asset slots. New assets arrive each step by a mean-reverting process around 35 (the equilibrium)
- 35% of on-market revenue is credited to cash (`reinvestment_percentage`). Costs are charged in full. The cash-basis eNPV (`ask_cash_enpv`) uses the same rate

## Asset Pipeline

- Assets belong to one of 3 therapeutic areas (TAs): oncology, respiratory & immunology, vaccines & infectious disease. Each targets one of 3 indications within its TA (9 total)
- Phase 1 → Phase 2 → Phase 3 → regulatory approval → on market. Each trial phase has a cost, a duration, and a probability of success (PTRS)
- Approval takes 1-3 steps, succeeds 85-95% of the time, and costs a £50M filing fee
- On-market assets earn revenue until patent expiry
- Investing starts an Idle asset's next phase. Idle assets cost nothing
- PTRS is hidden. Each asset arrives with one free noisy reading, and the observation shows that estimate; trial outcomes roll against the true value. More readings can be bought (see [PTRS Readings](#ptrs-readings))

## Shared Market

- Both players' drugs compete in the same indications. Each drug's revenue share is scaled by `1 / n^α`, where the exponent ramps with entry order: the first entrant takes ~15% of the full penalty, and later entrants ramp to the full `α = 2.0` by the 4th
- First-mover exclusivity and the first-mover revenue bonus are disabled
- Pipeline leaks: when an opponent's asset advances a phase, you receive an alert with probability 20% / 50% / 70% (Phase 1→2 / 2→3 / 3→Approval)
- Marketing spend also leaks (see [Marketing](#marketing))

## Business Development (BD)

- BD assets arrive at random each step (Poisson λ = 1.3, up to 3 slots). They are pre-progressed — already in Phase 1, 2, or 3
- An unsold BD asset stays up to 3 steps
- First-price sealed-bid auction per slot: each player bids cash (£M, capped at £100B; 0 passes). Highest bid wins and pays its own bid; ties break randomly
- There is no affordability check. A winning bid above your cash bankrupts you
- `ask_cash_enpv` in the observation is a value estimate, not a price

## Dropping Assets

- Any non-terminal asset (Idle, InDevelopment, OnMarket) can be dropped. An InDevelopment asset's trial stops
- Drop fee = `0.25 × cost_remaining` of the current phase, rounded to the nearest £1M; 0 with no active trial
- A dropped asset enters the `Dropped` state (distinct from `Failed`)

## Marketing

Two binary per-step spends.

- **Demand creation (DC)** — per indication. Each spend adds `+0.10` to that indication's shared demand multiplier, which scales the revenue of every on-market drug there (both players'). The multiplier decays toward 1.0 (~3-step half-life). Cost = `0.035 × pool-peak max_revenue` (anchored to the largest drug in the pool, not yours). Allowed in indications where you hold no drug
- **Brand equity (BE)** — per asset. Each spend adds `+0.25` to the drug's `brand_score`, which scales its market share by `brand_mult = 1 + 3.5·(1−floor)·max(0, brand_score − floor)`. `floor = min(raw_max_revenue / pool_peak, 1)`, so BE has the most effect on small drugs and almost none on the market leader. The score decays toward the floor (~3-step half-life). Cost = `0.0175 × the drug's own max_revenue`. Affects on-market drugs only; spend on a pre-market asset is still charged, and its score is reset to the floor at market entry
- **Leaks** — each BE spend leaks with probability 0.8, as a `BE_SPEND` alert carrying the TA, indication, and number of leaked spends (never the amount). DC leaks are off with 2 players; the demand multiplier is already public

## Clinical Sites

A hard cap on concurrent trials.

- Each operational site hosts one InDevelopment asset. It frees the moment that asset leaves development, including between phases. You start with 4
- The cap gates only new trial starts; running trials are never blocked. Excess starts are costless no-ops (the asset stays Idle), granted in ascending asset (arrival) order
- `upgrade` buys one site. Cost follows a Fibonacci curve off £500M (1×, 1×, 2×, 3×, 5×, … for successive purchases), rounded to £1M. A bought site takes 2 steps to build
- Site auction: at step 10 and every 20 steps after, one immediately-operational site is sold by first-price sealed bid (£M), as for BD. No affordability check

## PTRS Readings

- A reading draws a noisy logit-normal sample of the asset's true PTRS. Readings are combined in a precision-weighted mean, so error shrinks as `~1/√N`. One sample has ~21% mean absolute error
- Readings can target your 40 asset slots and the 3 BD slots (43 total), 0-10 per slot per step. One reading refreshes every pending phase of the asset; nearer phases are less noisy (σ multipliers 1.0 / 1.5 / 2.0 by phase distance)
- Cost per reading = `0.05 × the trial's cost_remaining`, rounded to £1M, scaled for multiple readings in one step: 1×, 2×, 4×, 7×, 12× total for 1-5
- Readings on a BD asset refine your private estimate only; opponents don't see them
- Per-trial confidence is exposed as `ptrs_equiv_n_norm` ∈ [0, 1]

## Turn Order

Each step:

1. **BD auctions** — each slot resolves; winners pay and receive the asset
2. **Site auction** — if one is running
3. **Per player:**
   1. Sites whose build finished become operational
   2. Ongoing trial costs are charged
   3. Site purchase (`upgrade`)
   4. New trial starts (gated by free sites) and drops are charged
   5. PTRS readings are charged and applied
   6. Revenue is collected
   7. Marketing is charged
   8. Trials resolve — advance, pass, or fail
   9. New assets arrive
4. **Market update** — demand creation boosts apply, multipliers decay, new BD assets spawn

Bankruptcy is checked after each charge in step 3. Trial costs and readings are charged before revenue, so a player can go bankrupt on a step whose revenue would have covered them.

## Termination and Scoring

A player's game ends when its cash drops below 0 (bankrupt) or at the 100-step horizon. A bankrupt player's actions are ignored for the rest of the match; the match ends when both players' games have ended, 101 framework steps at most.

Net cash flow (NCF) is final cash minus cash at step 0: `0.35 × revenue − costs`, summed over the 100 steps, with auction payments included in costs.

Outcomes:
- A bankrupt player loses to a solvent one
- If both go bankrupt, the one that lasted longer wins; same step is a draw
- Otherwise higher NCF wins; equal NCF is a draw

Reward is **1.0 win, 0.5 draw, 0.0 loss**. An illegal action, exception, or timeout forfeits: the offender's reward is `None` and the opponent gets 1.0.

## Observation

| Field | Type | Description |
|-------|------|-------------|
| `obs` | list[float] | Flattened engine observation, 1534 floats. See [Observation Layout](#observation-layout) |
| `actionMask` | dict | Masks for the discrete action heads, indexed `mask[slot][choice]` |
| `agentIndex` | int | Your seat, 0 or 1 |
| `cash` | float | Your cash (GBP) |
| `enpv` | float | Your portfolio's expected net present value (GBP) |
| `bankrupt` | bool | Whether you are bankrupt (cash < 0) |
| `step` | int | Current step, 0-indexed |
| `remainingOverageTime` | float | Remaining overage time budget (seconds) |

The observation holds your portfolio only. The opponent's appears only through alerts.

### Observation Layout

`obs` concatenates, in order:

| Block | Floats | Contents |
|-------|--------|----------|
| Globals | 6 | `cash`, `time`, `operational_sites`, `free_sites`, `sites_in_development`, `site_auction_active` |
| Assets | 40 × 29 | Per asset: 13 scalars, then 4 trials × 4 |
| BD slots | 3 × 18 | Per BD offer |
| Indication markets | 9 × 6 | 3 TAs × 3 indications |
| Alerts | 20 × 13 | Event type as a 6-float one-hot, then 7 fields |

**Asset scalars** (13): `max_revenue`, `time_until_max_revenue`, `time_until_patent_expiry`, `pending_trial_phase` (0 = none, 1-4 = Phase 1/2/3/Approval), `time_on_market`, `cost_this_step`, `revenue_this_step`, `enpv`, `eroi`, `state` (0 = Idle, 1 = InDevelopment, 2 = OnMarket, 3 = Failed, 4 = Expired, 5 = Dropped), `ta_index` (0-2), `indication` (0-2), `brand_score`.

**Trial** (4, one per phase): `cost_remaining`, `time_remaining`, `ptrs` (noisy estimate), `ptrs_equiv_n_norm` (reading confidence).

**BD slot** (18): `available`, `max_revenue`, `time_until_max_revenue`, `time_until_patent_expiry`, `ta_index`, `indication`, `enpv`, `trial_phase`, `ptrs`, `steps_remaining`, `eroi`, `ask_cash_enpv`, then `ph0_ptrs`, `ph0_equiv_n_norm`, `ph1_ptrs`, `ph1_equiv_n_norm`, `ph2_ptrs`, `ph2_equiv_n_norm`.

**Indication market** (6): `exclusivity_remaining`, `my_avg_share`, `first_mover`, `my_drugs`, `competitor_drugs`, `demand_multiplier` (public). `exclusivity_remaining` and `first_mover` are always 0.

**Alert** (7 after the one-hot): `agent_index`, `ta_index`, `indication`, `age`, `phase`, `bd_price` (price paid, BD and site deals), `be_count` (leaked BE spends). Event types: 0 = drug release, 1 = BD deal, 2 = pipeline leak, 3 = site deal, 4 = BE spend leak, 5 = DC spend leak.

Padding: empty asset slots have `state = 4` (Expired), empty BD slots `available = 0`, empty alerts an all-zero one-hot.

[AGENTS.md](AGENTS.md) shows how to decode `obs` into a dict with the bundled engine.

## Action Format

A dict with one entry per head. Every head is optional; missing or `null` heads are filled with their no-op, so `{}` passes.

| Head | Shape | Values |
|------|-------|--------|
| `investments` | 40 ints | 0 = nothing, 1 = invest, 2 = drop |
| `bd_bids` | 3 floats | Cash bid per BD slot (£M, 0 = pass, max 100000) |
| `ptrs_research` | 43 ints | Readings to buy, 0-10 (40 assets, then 3 BD slots) |
| `demand_creation` | 9 ints | 1 = spend, per indication |
| `brand_equity` | 40 ints | 1 = spend, per asset |
| `upgrade` | int | 1 = buy a clinical site |
| `site_bid` | 1 float | Site-auction cash bid (£M, 0 = pass, max 100000). Ignored when no auction is running |

### Action Masks

`actionMask` covers the discrete heads. `bd_bids` and `site_bid` are unmasked.

| Head | Shape | A choice is legal when |
|------|-------|------------------------|
| `investments` | 40 × 3 | 0: always. 1: the asset is Idle and the phase is affordable. 2: the slot holds an asset and the drop fee is affordable. Padding slots allow only 0 |
| `ptrs_research` | 43 × 11 | 0: always. `n > 0`: the slot holds an asset with a pending trial and `n` readings are affordable |
| `demand_creation` | 9 × 2 | Always; not affordability-checked |
| `brand_equity` | 40 × 2 | Always; not affordability-checked |
| `upgrade` | 2 | 0: always. 1: cash ≥ next site cost |

Affordability is first-order: each choice is judged on its own cost, ignoring the rest of the action. A set of individually legal choices can still overspend.

### Illegal Actions

An illegal action forfeits the match (status `INVALID`). An action is illegal if it:
- is not a dict
- names a head that doesn't exist
- picks a choice the mask forbids
- has the wrong length, type, or range for its head — a negative bid or one above 100000 included

## Quick Start

```python
from kaggle_environments import make

def agent(observation, configuration):
    masks = observation["actionMask"]
    return {"investments": [1 if slot[1] else 0 for slot in masks["investments"]]}

env = make("pyxis", configuration={"seed": 42}, debug=True)
env.run([agent, "knapsack"])
print([s.reward for s in env.steps[-1]])
```

## Built-in Agents

| Name | Description |
|------|-------------|
| `"knapsack"` | Budget-optimising heuristic: solves a 0/1 knapsack over investments each step |
| `"random"` | Invests in random idle assets. Unseeded, so matches involving it are not reproducible |
| `"do_nothing"` | Never acts |

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `episodeSteps` | 1000 | Framework step cap. Matches end at 101 steps, when the engine ends both players' games |
| `actTimeout` | 60 | Seconds per agent per step |
| `runTimeout` | 1200 | Seconds for the whole episode, excluding the warmup at reset |
| `seed` | `null` | Episode seed. Scrubbed from the configuration agents see; stored on `env.info["seed"]` |

## Engine Training API

The engine package is bundled with the environment and importable for local training. It is not how a Kaggle submission is scored.

```python
import kaggle_environments.envs.pyxis.pyxis  # puts the engine on sys.path
from pyxis_portfolio_challenge.environment import make_multi_agent_train_env

env = make_multi_agent_train_env()                    # flat observations
env = make_multi_agent_train_env(flatten_obs=False)   # nested dict observations
```

The env is a PettingZoo `ParallelEnv`. Agent ids are `pharma_0` and `pharma_1`.

```python
obs, infos = env.reset(seed=42)
done = False
while not done:
    actions = {aid: my_policy(obs[aid], env.action_masks(aid)) for aid in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    done = all(terms.values()) or all(truncs.values())
```

Engine step rewards are net cash flow per step, not the Kaggle win/loss reward. With `flatten_obs=False`, observations are dicts with `cash`, `time`, `assets`, `bd_market`, `indication_markets`, `alerts`, and `clinical_sites`, using the field names above.

### Single-Agent Trainer

`.train()` returns a `gym.Env`. `None` marks the trainee; opponents are names or callables.

```python
trainer = env.train([None, "knapsack"])
obs, info = trainer.reset(seed=42)
while True:
    action = my_policy(obs, trainer.action_masks())  # masks work with MaskablePPO
    obs, reward, terminated, truncated, info = trainer.step(action)
    if terminated or truncated:
        break
```

### Single Match and Replay

```python
reports, playthrough = env.run([my_agent, "knapsack"], seed=42, flat_obs={0: True})
with open("replay.json", "w") as f:
    f.write(playthrough.model_dump_json(indent=2))
```

Upload `replay.json` to [gsk.ai/pyxis-portfolio-challenge](https://gsk.ai/pyxis-portfolio-challenge) to watch it. `flat_obs={0: True}` gives flat observations to the agent at index 0.

### Evaluate

```python
from pyxis_portfolio_challenge.environment.competition import evaluate

reports, _ = evaluate(agents=[my_agent, "knapsack"], num_episodes=100, num_workers=4, flat_obs={0: True})
```

Each seed is played twice with seats swapped, so `num_episodes=100` runs 200 episodes; results are keyed `agent_0` / `agent_1` by agent, not seat. `symmetric=False` disables the swap and keys results by seat (`pharma_0` / `pharma_1`). With `num_workers > 1`, guard the call with `if __name__ == "__main__":` on macOS and Windows.

Each agent's report is a list of three groups — `PerEvaluationMetrics`, `PerEpisodeMetrics`, `PerStepMetrics` — covering cash, revenue, cost, pipeline state, market share, and head-to-head outcomes. Key metrics:

| Metric | Group | Description |
|--------|-------|-------------|
| `PerEpisodeWinLoss` | PerEpisode | 1.0 / 0.5 / 0.0, as in [Termination and Scoring](#termination-and-scoring). The mean is the win rate |
| `PerEpisodeCumulativeReward` | PerEpisode | NCF per episode |
| `PerEvaluationCumulativeReward` | PerEvaluation | Mean, stdev, min, max of NCF |
| `PerEvaluationBankruptcyRate` | PerEvaluation | Fraction of episodes ending in bankruptcy |

`pyxis_portfolio_challenge/config.yaml` lists every enabled metric.
