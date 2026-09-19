# GSK Pyxis Portfolio Challenge

## Overview

The Pyxis Portfolio Challenge is a multi-agent reinforcement learning environment for sequential capital allocation under uncertainty. Agents manage a portfolio of R&D assets, each progressing through a multi-phase development pipeline with stochastic outcomes, compounding costs, and long time horizons. The objective is to maximise portfolio value through investment timing, resource allocation, and competitive positioning against other agents.

The environment is highly stochastic, reflecting the reality of R&D pipeline investment where the majority of assets fail and individual outcomes carry outsized financial consequences. Agents face ~80% asset attrition rates, long development timelines where investment outcomes are delayed by multiple phases, and coupled investment decisions where capital committed to one asset constrains all future options. In the multi-agent setting, agents share indication markets, compete for business development acquisitions, and receive noisy intelligence about rival pipelines. Agents must balance exploration (investing in uncertain early-stage assets) against exploitation (scaling proven late-stage assets), while adapting to opponents' strategies in a high-variance environment where robust decision-making under uncertainty is essential.

The repository provides a standardised environment with multiple baseline agents spanning different algorithmic paradigms: a budget-optimising heuristic (knapsack) and stochastic baselines (random, do-nothing). The environment exposes a PettingZoo `ParallelEnv` API, a gym-compatible single-agent training wrapper, and evaluation utilities for head-to-head matchups.

## Play the Game

Before building an agent, you can play the game yourself against a provided AI opponent at [gsk.ai/pyxis-portfolio-challenge](https://gsk.ai/pyxis-portfolio-challenge). This is the best way to develop an intuition for the environment dynamics — asset pipelines, trial outcomes, cash management, and competitive market timing.

## Getting Started

### Installation

```bash
# From source
uv sync
```

### Quick Start

Run a match from the CLI and generate a replay file:

```bash
uv run python -m pyxis_portfolio_challenge.multi_agent_cli 'knapsack' random --seed 42 -o replay.json
```

Upload `replay.json` to [gsk.ai/pyxis-portfolio-challenge](https://gsk.ai/pyxis-portfolio-challenge) to watch the replay in the browser.

Or use the Python API to evaluate agents over multiple episodes:

```python
from pyxis_portfolio_challenge.environment import make_multi_agent_train_env
from pyxis_portfolio_challenge.environment.competition import evaluate

env = make_multi_agent_train_env()
reports, _ = evaluate(agents=["knapsack", "random"], num_episodes=100)
# 100 seeds × 2 positions = 200 episodes; results keyed by agent_0/agent_1
```

### Provided Agents

The following agents are available as named opponents in `env.train()`, `env.run()`, and `evaluate()`, as well as in the interactive frontend:

| Name | API string | Description |
|------|-----------|-------------|
| **Knapsack** | `"knapsack"` | Budget-optimising heuristic that solves a 0/1 knapsack each step. Concurrent trials are limited naturally by the clinical-sites cap rather than a hard per-agent capacity. Strong baseline. |
| **Random** | `"random"` | Randomly invests in available idle assets each step. Useful as a lower-bound baseline. |
| **Do Nothing** | `"do_nothing"` | Never invests in anything. Useful for testing and as an absolute floor. |

### CLI

Run matches from the command line with the `multi_agent_cli` module. Specify two agents by name or script path:

```bash
# Named agents
uv run python -m pyxis_portfolio_challenge.multi_agent_cli 'knapsack' random --seed 42

# Custom agent script vs named agent
uv run python -m pyxis_portfolio_challenge.multi_agent_cli ./my_bot.py 'knapsack' -o replay.json

# Export replay with custom display names
uv run python -m pyxis_portfolio_challenge.multi_agent_cli 'knapsack' random -o replay.json -n "Alpha" -n "Beta"
```

Custom agent scripts must define a `create_agent(agent_name, **kwargs)` factory function returning a callable with an optional `set_env(env)` method.

## Environment

### Overview

**Game Parameters:**
- 2 agents compete head-to-head
- 100-step horizon (the agent's acting window)
- 500-step warmup pre-roll: before the agent takes its first action, the market is advanced 500 steps under a do-nothing policy, then the clock is rebased to 0. The agent therefore starts on a mature, already-populated market and plays a full 100 steps on top of it.
- Starting cash: £5B
- Up to 40 assets in portfolio (equilibrium ~35)
- 35% reinvestment percentage (fraction of on-market revenue reinvested; also scales cash-basis eNPV)
- Reward function: net cash flow per step

All monetary values are in GBP (£).

> **Official competition settings:** 40 asset slots (equilibrium ~35), £5B starting cash, 100-step acting horizon after a 500-step warmup pre-roll.

**Asset Pipeline:**
- Assets arrive in one of 3 therapeutic areas (TAs): oncology, respiratory & immunology, vaccines & infectious disease
- Each asset targets a specific indication within its TA (3 indications per TA, 9 total)
- Assets have 3 trial phases (Phase 1, 2, 3) plus a regulatory approval phase
- Each trial phase has a cost, duration, and probability of success (PTRS)
- **PTRS is hidden**: the true per-phase PTRS is not observed. Each asset arrives with a single free noisy reading, and the PTRS shown in the observation is that noisy estimate. Trial outcomes are rolled against the hidden true value. Agents can pay for additional **PTRS research readings** to sharpen their estimate (see below)
- Assets that pass all phases reach market and generate revenue until patent expiry
- Most assets fail during trials (~80% attrition)

**Approval Phase:**
- After Phase 3, assets enter regulatory approval (1-3 steps, 85-95% success rate, £50M filing fee)

**Shared Market & Competition:**
- Agents share indication markets — multiple drugs can compete in the same indication
- Market congestion: revenue drops as more drugs compete in the same indication. Each drug's share is scaled by `1 / n^α`, where the effective exponent ramps with entry order — the incumbent (first entrant) is lightly penalised (~15% of full penalty) and later entrants ramp to the full `α=2.0` by the 4th entrant. First-mover exclusivity and the first-mover revenue bonus are **disabled** in the shipped config
- Pipeline leak alerts: when an opponent advances a trial phase, there's a probability of an intelligence leak (20%/50%/70% for Phase 1→2/2→3/3→Approval)
- Marketing-spend leaks: brand-equity and demand-creation spending can also leak to opponents as alerts (see Marketing Spend below)

**Business Development (BD):**
- BD assets appear randomly each step (Poisson λ=1.3, up to 3 slots per step)
- BD assets are pre-progressed (already in Phase 1, 2, or 3) — buying one skips early development
- An unwon BD asset persists for up to 3 steps before disappearing
- **Continuous cash-bid auction**: agents submit a raw cash bid per slot (£M, capped at £100B; a bid of 0 passes). Highest bid wins and **pays its own bid** (first-price sealed-bid); ties broken randomly. There is no affordability cap — an overbid can bankrupt the winner. The `ask_cash_enpv` field in the observation is value guidance only (the cash-basis eNPV under the 35% reinvestment rate), not the price paid

**Drop Action:**
- Each asset slot can be dropped instead of invested. The per-asset action is ternary: `0` = do nothing, `1` = invest, `2` = drop
- Any non-terminal asset can be dropped (Idle, InDevelopment, or OnMarket); an InDevelopment asset's trial is stopped
- Drop fee = `0.25 × cost_remaining` of the current phase, rounded to the nearest £1M (fee is 0 if the asset has no active trial). The asset transitions to a `Dropped` state (distinct from `Failed`)

**Marketing Spend (Demand Creation & Brand Equity):**

Two independent, per-step binary spend actions let agents grow revenue beyond raw trial success.

- **Demand Creation (DC)** — spent *per indication* (9 indications). Each spend adds a fixed increment (`+0.10`) to that indication's **shared** demand multiplier, which multiplies the revenue of every on-market drug in the indication (yours and rivals'). The multiplier decays toward 1.0 (~3-step half-life). Cost = `0.035 × pool-peak max_revenue` (an indication-wide cost anchored to the largest drug ever seen, not your own), charged each step you spend. DC can be spent in an indication where you hold no drug — it sizes the whole market — so it is a large-market tool.
- **Brand Equity (BE)** — spent *per drug/asset* (40 slots). Each spend raises that drug's `brand_score` (`+0.25`), which boosts its market-share quality via `brand_mult = 1 + 3.5·(1−floor)·max(0, brand_score − floor)`. The `floor` scales with drug size (`min(raw_max_revenue / pool_peak, 1)`), so BE is strong for small underdog drugs and nearly useless for the market leader — a big drug cannot cheaply spend to bury a small rival. Score decays toward the floor (~3-step half-life). Cost = `0.0175 × the drug's own max_revenue`. Only affects on-market drugs.
- **Leaks**: BE spend leaks to opponents with probability 0.8 (a `BE_SPEND` alert carrying the TA, indication, and count of leaked spends — never the amount). DC spend leaks with probability 0.8 but is **gated off below 3 agents** (with 2 agents the per-indication demand multiplier is already public in the observation, so a DC leak would add nothing).

**Clinical Sites:**

A hard concurrency cap on the number of trials an agent can run simultaneously — a runaway-cash-sink control.

- Each **operational site** hosts one InDevelopment asset at a time and frees the instant that asset leaves development, including between phases. Agents start with 4 sites. The cap gates only *new* trial starts; ongoing trials are never blocked
- When an agent requests more trial starts than it has free sites, the excess become costless no-ops (assets stay Idle). Arbitration follows ascending asset (arrival) index in the shipped config (`agent_priority=false`)
- **Upgrade action** (`0`/`1`): buy one new site. Cost follows a Fibonacci curve off a £500M base (1×, 1×, 2×, 3×, 5×, … for successive purchases beyond the starting 4), rounded to £1M. A bought site takes 2 steps to build before it becomes operational
- **PvP site auction**: starting at step 10 and every 20 steps thereafter, one immediately-operational site is auctioned. Agents submit a continuous cash bid (£M); highest bid wins and pays its own bid (first-price, ties random). No affordability cap — an overbid can bankrupt the winner

**PTRS Research Readings:**

Because PTRS is hidden (see Asset Pipeline), agents can buy noisy readings to refine their estimate of an asset's success probability.

- A "reading" draws a stochastic logit-normal sample of the asset's true PTRS. Readings are folded into a precision-weighted running mean, so the estimate converges to the truth as `~1/√N`. A single-sample estimate has a mean absolute error of ~21%
- Readings can be bought on your own portfolio assets (40 slots) and on BD offers (3 slots) — 43 slots total, 0–10 readings each per step. One reading action refreshes all pending phases of an asset at once; nearer phases are sampled less noisily (σ multipliers 1.0/1.5/2.0 by phase distance)
- Cost per reading = `0.05 × the trial's cost_remaining` (rounded to £1M), Fibonacci-scaled for multiple readings in one step (1×, 2×, 4×, 7×, 12× for 1–5). Readings are paid for and applied within the same step, before trial evolution
- Readings on a BD asset (not yet owned) refine your own private estimate without revealing it to opponents. The observation exposes the noisy estimate plus a per-trial confidence scalar (`ptrs_equiv_n_norm` ∈ [0,1]) indicating how much you effectively know

### Observation Space

The observation and action space dimensions scale with `max_num_assets`. The competition uses 40 asset slots (equilibrium ~35). All per-asset counts below refer to the configured `max_num_assets`.

**Flat observation** (default): a numpy array whose length depends on `max_num_assets` and the enabled features (1534 with 40 assets and the shipped config).

**Dict observation** (`flatten_obs=False`): a nested dict with these top-level keys:
- `cash` — current cash (float)
- `time` — current step (int)
- `assets` — tuple of `max_num_assets` asset dicts
- `bd_market` — tuple of 3 BD slot dicts
- `indication_markets` — dict of 3 TAs, each with 3 indication dicts (9 total)
- `alerts` — tuple of 20 alert dicts
- `clinical_sites` — dict of `operational_sites`, `free_sites`, `sites_in_development`, `site_auction_active`

**Per-asset features** (13 fields + trials):
- `max_revenue`, `time_until_max_revenue`, `time_until_patent_expiry`
- `pending_trial_phase` (0=none, 1-4=Phase 1/2/3/Approval)
- `time_on_market`, `cost_this_step`, `revenue_this_step`
- `enpv` (expected NPV), `eroi` (expected ROI)
- `state` (0=Idle, 1=InDevelopment, 2=OnMarket, 3=Failed, 4=Expired; 5=Dropped)
- `ta_index` (0-2), `indication` (0-2)
- `brand_score` — current brand-equity score (marketing)
- `trials` — tuple of 4 trial dicts, each with `cost_remaining`, `time_remaining`, `ptrs` (noisy estimate), `ptrs_equiv_n_norm` (reading-confidence ∈ [0,1])

**Per BD slot** (18 fields): `available`, `max_revenue`, `time_until_max_revenue`, `time_until_patent_expiry`, `ta_index`, `indication`, `enpv`, `eroi`, `trial_phase`, `ptrs`, `steps_remaining` (persistence countdown), `ask_cash_enpv` (value-guidance price anchor), plus per-phase noisy PTRS estimates and reading-confidence: `ph0_ptrs`/`ph1_ptrs`/`ph2_ptrs` and `ph0_equiv_n_norm`/`ph1_equiv_n_norm`/`ph2_equiv_n_norm`

**Per indication market** (6 fields): `my_avg_share`, `my_drugs`, `competitor_drugs`, `demand_multiplier` (public shared DC multiplier), `exclusivity_remaining`, `first_mover`. Note: `exclusivity_remaining` and `first_mover` are always 0 in the shipped config (first-mover mechanics disabled)

**Per alert** (8 fields): `event_type`, `agent_index`, `ta_index`, `indication`, `age`, `phase`, `bd_price` (price paid, for BD/site deals), `be_count` (number of leaked BE spends). `event_type` values: 0=drug release, 1=BD deal, 2=pipeline leak, 3=clinical-site deal, 4=brand-equity spend leak, 5=demand-creation spend leak

Empty/padding slots: empty assets have `state=Expired`, empty BD slots have `available=0`, empty alerts have `event_type=-1`.

### Action Space

Actions are dicts. With the shipped config (drop action, marketing, clinical sites, and PTRS readings all enabled), the action has seven keys:

```python
action = {
    "investments":     np.array([...], dtype=np.int64),   # (40,)  MultiDiscrete, 0=nothing / 1=invest / 2=drop
    "bd_bids":         np.array([...], dtype=np.float32),  # (3,)   continuous cash bid per BD slot (£M, 0=pass)
    "ptrs_research":   np.array([...], dtype=np.int64),    # (43,)  MultiDiscrete, 0-10 readings per slot (40 assets + 3 BD)
    "demand_creation": np.array([...], dtype=np.int64),    # (9,)   MultiDiscrete binary, per indication
    "brand_equity":    np.array([...], dtype=np.int64),    # (40,)  MultiDiscrete binary, per asset
    "upgrade":         0,                                   # Discrete(2), buy one clinical site
    "site_bid":        np.array([...], dtype=np.float32),  # (1,)   continuous cash bid for the site auction (£M, 0=pass)
}
```

- **investments** — ternary per asset: `0` = do nothing, `1` = invest (Idle assets only; starts the next trial phase), `2` = drop (any non-terminal asset; charges the drop fee)
- **bd_bids** — continuous cash bid per BD slot; highest bid wins and pays its own bid (0 = pass)
- **ptrs_research** — number of PTRS readings to buy per slot this step (0–10), over 40 portfolio slots then 3 BD slots
- **demand_creation** — binary per indication; `1` = spend on demand creation
- **brand_equity** — binary per asset; `1` = spend on brand equity
- **upgrade** — `1` = buy one clinical site (Fibonacci-priced, 2-step build)
- **site_bid** — continuous cash bid for the periodic site auction (ignored when no auction is active that step)

**Action Masks**: Call `env.action_masks(agent_id)` before each step. It returns masks for the discrete heads; the continuous cash-bid heads (`bd_bids`, `site_bid`) are unmasked and gated only by the cash you actually have (an overbid can bankrupt you):

```python
masks = env.action_masks("pharma_0")
# masks["investments"]:     list of 40 lists, each length 3 — [asset][action] valid?
# masks["ptrs_research"]:   list of 43 lists, each length 11 — [slot][count] affordable?
# masks["demand_creation"]: list of 9 lists,  each length 2
# masks["brand_equity"]:    list of 40 lists, each length 2
# masks["upgrade"]:         list of length 2 — [no-op, can-afford-a-site]
```

Mask rules (all affordability checks are first-order — they consider each option independently and do not account for the combined cost of taking several actions in the same step):
- **investments**: `1` (invest) valid only for Idle assets the agent can afford; `2` (drop) valid whenever the asset exists and the drop fee is affordable; `0` always valid; padding slots allow only `0`
- **ptrs_research**: count 0 always valid; count `n>0` valid only if the slot holds an asset with a pending trial and the agent can afford the Fibonacci-scaled cost of `n` readings
- **demand_creation** / **brand_equity**: spending is valid when affordable
- **upgrade**: index 0 (no-op) always valid; index 1 (buy) valid only when `cash ≥ next site cost`

Using masks with MaskablePPO or a manual agent:

```python
masks = env.action_masks(agent_id)
# For RL: pass masks to MaskablePPO predict()
# For manual agents: restrict each action head to its valid choices
```

## Competition API

### Creating the Environment

```python
from pyxis_portfolio_challenge.environment import make_multi_agent_train_env

env = make_multi_agent_train_env()
```

This creates a PettingZoo `ParallelEnv` from the YAML configuration. Observations are flat numpy arrays by default for faster processing. Pass `flatten_obs=False` if you prefer structured dict observations:

```python
env = make_multi_agent_train_env(flatten_obs=False)
```

### PettingZoo Environment

Use the env directly for advanced training setups (self-play, population-based training, etc.):

```python
obs, infos = env.reset(seed=42)
done = False
while not done:
    actions = {}
    for agent_id in env.agents:
        actions[agent_id] = my_policy(obs[agent_id])
    obs, rewards, terms, truncs, infos = env.step(actions)
    done = any(terms.values()) or any(truncs.values())
```

### Gym-like Trainer

Call `.train()` on the env to get a single-agent `gym.Env` wrapper. Use `None` to mark your trainee slot and name strings for opponents:

```python
trainer = env.train([None, "knapsack"])

# Standard gym loop
obs, info = trainer.reset(seed=42)
while True:
    masks = trainer.action_masks()  # for MaskablePPO
    action = my_policy(obs, masks)
    obs, reward, terminated, truncated, info = trainer.step(action)
    if terminated or truncated:
        break
```

You can also pass your own callable as an opponent:

```python
from pyxis_portfolio_challenge.agents import MultiAgentKnapsackAgent

custom_opp = MultiAgentKnapsackAgent(agent_name="pharma_1", capacity=8)
trainer = env.train([None, custom_opp])
```

### Run

Call `.run()` to pit two agents against each other for a single episode. It always captures a full playthrough and returns per-agent metrics — useful for quick head-to-head comparisons and generating replay files. Agents can be name strings or callables, just like `.train()` and `evaluate()`:

```python
per_agent_reports, playthrough = env.run(
    [my_agent, "knapsack"],
    seed=42,
    flat_obs={0: True},  # my_agent at index 0 expects flat obs
)

# per_agent_reports: {"pharma_0": [...], "pharma_1": [...]}
# playthrough is a PlaythroughData object — serialize to JSON for the replay viewer
playthrough.model_dump_json(indent=2)
```

You can also run two named agents directly:

```python
reports, playthrough = env.run(["knapsack", "random"], seed=42)

# Save replay to file
with open("replay.json", "w") as f:
    f.write(playthrough.model_dump_json(indent=2))
```

For multi-episode statistical evaluation, use `evaluate()` below instead.

### Evaluate & Metrics

Use the standalone `evaluate()` function. Agents can be strings or callables:

```python
from pyxis_portfolio_challenge.environment.competition import evaluate

per_agent_reports, playthrough = evaluate(
    agents=[my_agent, "knapsack"],
    num_episodes=100,
    num_workers=4,
    flat_obs={0: True},  # my_agent expects flat obs
)

# per_agent_reports: {"agent_0": [...], "agent_1": [...]}
```

> **Note:** With `num_workers > 1`, `evaluate()` spawns subprocesses. In a script, guard the call with `if __name__ == "__main__":` (required on macOS/Windows, which use `spawn`), otherwise you'll get a `BrokenProcessPool` / "importing the main module" error.

By default, `evaluate()` uses **seed-symmetric evaluation**: each seed is played twice with agent positions swapped to control for positional asymmetry. With `num_episodes=100`, this produces 200 total episodes (100 per seat assignment). Results are keyed by original agent identity (`agent_0`, `agent_1`), not seat position. Pass `symmetric=False` to disable this and get position-keyed results (`pharma_0`, `pharma_1`).

**Win conditions** are bankruptcy-aware: bankrupt agents automatically lose; if both go bankrupt, the agent that survived longer wins; same-step bankruptcy is a draw. Non-bankrupt agents are ranked by NCF.

The return value `per_agent_reports` is a dict mapping agent key to a list of three report groups:

```python
[
    {"PerEvaluationMetrics": [...]},  # Aggregated across all episodes
    {"PerEpisodeMetrics": [...]},     # Per-episode breakdowns
    {"PerStepMetrics": [...]},        # Per-step time series
]
```

Each group contains dicts keyed by metric name. The metrics cover financial performance (cumulative reward, cash, revenue, cost), pipeline state (assets idle/in-development/on-market), competitive position (agent rank, relative eNPV, market share), and head-to-head outcomes (win/loss per episode).

Key metrics for competition scoring:

| Metric | Group | Description |
|--------|-------|-------------|
| `PerEvaluationCumulativeReward` | PerEvaluation | Mean, stdev, min, max of cumulative reward across episodes |
| `PerEvaluationBankruptcyRate` | PerEvaluation | Fraction of episodes ending in bankruptcy |
| `PerEpisodeWinLoss` | PerEpisode | 1.0 = win, 0.0 = loss, 0.5 = draw (based on cumulative reward). Mean across episodes gives win rate. |
| `PerEpisodeCumulativeReward` | PerEpisode | Total reward per episode |

Additional metrics cover BD deal activity (`PerEpisodeBDDealsWon`), first-mover advantage (`PerEpisodeFirstMoverRate`), revenue lost to competition (`PerEpisodeRevenueLostToCompetition`), per-drug profitability (`PerEpisodeInvestmentPnL`), pipeline efficiency (`PerEpisodeAssetLifecycle`), and market share dynamics (`PerStepMeanMarketShare`, `PerStepDrugsOnMarket`). See `config.yaml` for the full list of enabled metrics, or define your own by adding entries to the `evaluation_metrics` list.
