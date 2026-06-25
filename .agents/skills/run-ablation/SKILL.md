---
name: run-ablation
description: Run a prompt-ablation study on a kaggle-environments game's LLM harness. Use when the user mentions "ablation", "prompt sensitivity", "test prompt variants", "compare prompts", "ablate the prompt", "prompt rewrite study", or asks whether their prompt wording is doing real work. Bootstraps a prompt_variants.py if missing, proposes new variants interactively, then runs a paired-seat tournament and reports the leaderboards.
---

# Run a Prompt Ablation Study

This skill drives the end-to-end prompt-sensitivity workflow for one game's harness: discover or bootstrap variants, propose additional ones in conversation with the user, run the paired-seat tournament, and report the findings. It is completely standalone — it does not depend on or modify the `create-harness`, `review-harness`, or `create-environment` skills.

The runner library is `kaggle_environments/ablation.py`. The two subcommands you'll invoke are:

```bash
python -m kaggle_environments.ablation check --env <env_name>
python -m kaggle_environments.ablation run --env <env_name> --models <csv> --games <N> --out <dir>
```

The runner contract:

* Every game's harness directory contributes a `prompt_variants.py` exposing `VARIANTS: dict[str, GameHarness]`. Each variant **is** a `GameHarness` (implements `get_legal_moves` / `make_prompt` / `parse_response`), so `create_agent_fn(variant)` works directly.
* The `baseline` variant must be byte-identical to the production `harness.py`. The `check` subcommand enforces this against seeded observations.
* For each variant, the runner plays a round-robin among the M models with both players using that variant. Every `{A, B}` matchup is scheduled in **seat-flipped pairs sharing one chance seed** — the same instance is played twice with seats swapped. `--games` must be even.

## Step 0: Identify the env

Confirm with the user which env they want to ablate. Acceptable forms: `open_spiel_bargaining`, `werewolf`, `tictactoe`, etc. The env name is what gets passed to `kaggle_environments.make(env_name)`.

Resolve the harness directory:
* `open_spiel_<game>` → `kaggle_environments/envs/open_spiel_env/games/<game>/`
* `<env>` → `kaggle_environments/envs/<env>/`

Confirm `harness.py` exists at that path. If not, stop and tell the user: the production harness has to exist before you can ablate it. Suggest they run `create-harness` first.

## Step 1: Inventory existing variants

Read `prompt_variants.py` if it exists at the harness directory. List the names found and one-line summaries:

```
Found prompt_variants.py with: baseline, compact, minimal, no_accept_preview, generic_names
```

If the file doesn't exist, you'll bootstrap it in Step 4. Tell the user you're going to:

```
No prompt_variants.py yet — I'll bootstrap one from harness.py.
The 'baseline' variant will be byte-identical to the production prompt;
I'll also propose a few ablations for your review.
```

## Step 2: Read the production prompt

Open `harness.py` and read the `make_prompt` (or `generate_prompt`) function plus any prompt-template constants it uses. Note the prompt's *structural* features — these are what your variant proposals will target:

* What rules / mechanics paragraphs are in there?
* What helper text (legal-move list, current-state summary, payoff hints, "you would receive...", "you may agree")?
* What domain-specific labels (item names, square notation, color names)?
* Is there a goal/strategy nudge ("end the game with a higher reward than your opponent")?
* What's the JSON schema?

You can't propose a good ablation without understanding the prompt's joints. Spend a turn here.

## Step 3: ALWAYS propose new variants

**This step is mandatory.** Even if `prompt_variants.py` already has variants, propose new ones. Skipping this step defeats the purpose of the skill — the point is to force a "what am I actually testing?" beat before any LLM money is spent.

Generate **3–5 candidate variants** with a one-line rationale each. Cover at least three of these generic axes:

* **Compact** — same information, much terser language. Drops worked examples, repeated reminders, loss-threat sentences. Tests whether the verbose scaffolding is load-bearing.
* **Minimal** — strips everything optional: rules, goal statement, helpers, examples. Just state + JSON schema. Tests how much the model already knows.
* **No goal hint** — removes any "your goal is to..." sentence. Tests whether the goal nudge is doing work.
* **Generic labels** — swaps real names for letters (Book/Hat/Basketball → A/B/C, North/East/South/West → 0/1/2/3). Tests for semantic priors that bias decisions away from stated valuations.
* **No legal-move list** — removes the enumeration of legal moves (if the prompt includes one). Tests whether the model can derive legality from rules.
* **Raw observation** — uses the env's raw observation string instead of any reformatting the prompt does. Tests whether the prompt's restructuring is helping.
* **Drop a specific helper** — pick one piece of computed help in the prompt (e.g. Bargaining's "you would receive [...]" accept preview, Chess's notation explainer, Go's territory hint) and remove it. Tests whether that helper is load-bearing.

Then propose **1–2 game-specific axes** that you identify from reading the prompt in Step 2. These are the most interesting ablations because they're tailored to the actual prompt design choices the author made.

Each proposal should include:
* A short name (snake_case, e.g. `no_payoff_hint`)
* One sentence of rationale (what hypothesis it tests)
* The concrete change vs. baseline (one or two bullet points)

**Use the `AskUserQuestion` tool to surface the proposals**, with one question per group of related variants, options for accept/skip/rewrite/replace. Concrete usage pattern:

```
AskUserQuestion({
  questions: [
    {
      question: "Which of these structural ablations should we include?",
      header: "Structural",
      multiSelect: true,
      options: [
        {label: "compact",       description: "Same info, terser — drops worked example + loss threat"},
        {label: "minimal",       description: "Strip rules/goal/helpers, just state + JSON schema"},
        {label: "no_goal_hint",  description: "Remove the 'win by ending with higher reward' sentence"},
      ]
    },
    {
      question: "Which game-specific ablations should we include?",
      header: "Game-specific",
      multiSelect: true,
      options: [
        {label: "no_accept_preview", description: "Drop 'you would receive [...]' — model must compute complement itself"},
        {label: "generic_names",     description: "Book/Hat/Basketball → A/B/C — strip semantic priors"},
      ]
    },
  ]
})
```

After the user responds, ask one follow-up if anything is ambiguous (a rewritten rationale, a swap of one axis for another). It's fine to loop two or three times. **You may not skip the proposal — even if the user accepts everything immediately, you must have surfaced concrete proposals first.**

## Step 4: Write the variants

Bootstrap `prompt_variants.py` if it doesn't exist; otherwise update it.

Pick the right template based on env type:
* OpenSpiel games (`open_spiel_*`) → start from `templates/prompt_variants_openspiel.py.tmpl`
* Anything else → start from `templates/prompt_variants_generic.py.tmpl`

For each variant:

1. **`baseline`** must be byte-identical to `harness.py`. Port the existing prompt template and helper logic into a `BaselineVariant` class. Do not paraphrase — the next step (`ablation check`) will fail if even one character differs.
2. **Each new variant** is a subclass of `BaselineVariant` (or sibling class) that overrides `make_prompt` (and `parse_response` if the schema changed — e.g. `generic_names` uses `{a, b, c}` JSON keys and needs an alias map).
3. Expose `VARIANTS = {"baseline": BaselineVariant(), ...}` at module level. The runner enforces a `"baseline"` entry.

Keep the variant code in one file. Do **not** import shared prompt fragments from a sibling helper module — `harness.py` is manually deployed and versioned separately, so a shared-module import would be a deploy-isolation footgun.

## Step 5: Verify with `ablation check`

```bash
uv run python -m kaggle_environments.ablation check --env <env_name>
```

Required output:
```
env: <env_name>
variants: ['baseline', 'compact', ...]
  baseline parity obs[0]: ok (NNNN chars)
  baseline parity obs[1]: ok (NNNN chars)
  ...
  variant 'compact': rendered ok across 5 observations
  ...
OK
```

If baseline parity fails, **stop and fix the BaselineVariant port** before proceeding. The most common cause is a paraphrased docstring or a stripped trailing newline. Show the user the diff (use `harness.generate_prompt(obs, [])` vs `VARIANTS["baseline"].make_prompt(obs, [])`) so they can confirm what changed.

If a variant errors on render (template KeyError, missing field), fix it. Do not move on with broken variants.

## Step 6: Confirm the run plan

Before spending API budget, confirm the run with the user. Use `AskUserQuestion` or open prose. Surface:

* The env, the variant list, the model list, `--games` (paired games per matchup — must be even).
* The total cell count: `K · M(M−1)/2 · games`. Add `M · games` per leaderboard if `--self-play`.
* A rough cost estimate. For Bargaining-class games (≤10 prompt-response rounds, ~1k–3k prompt tokens), assume ~20 LLM calls per game.
* The output directory.

Default suggestion if the user hasn't specified: `--games 20`, all variants, `--concurrency 8`. Wait for explicit go before invoking the runner.

## Step 7: Run the tournament

```bash
MODEL_PROXY_KEY=$KEY MODEL_PROXY_URL=$URL \
uv run python -m kaggle_environments.ablation run \
  --env <env_name> \
  --models <csv of model names> \
  --games <N> \
  --concurrency 8 \
  --out results/<env>_<date>/
```

Stream the runner's progress to the user. It writes `games.csv` (one row per game) and `summary.md` (per-variant leaderboards + cross-variant rank shifts + anomalies) into `--out`.

If the run is interrupted, re-run with `--resume` to skip the cells already in `games.csv`.

## Step 8: Report findings

Read `summary.md` and surface to the user:

* **The cross-variant rank-shift table.** This is the most informative artifact. If a model's rank shifts between variants, the prompt is doing real work for that model. If ranks are stable across variants, the prompt is mostly decoration (or the variants weren't aggressive enough).
* **The anomalies section.** Variants with >5% crash rate or hard errors usually indicate the prompt is under-specified for some model. Read those rows in `games.csv` for context.
* **The biggest mean-score deltas.** A variant that moves a model's mean score by ≥0.5 (in a game scaled to ±1) is a strong signal — pay attention to its concrete change vs. baseline.

Make one or two concrete recommendations: which variant(s) look like upgrades to baseline, which look like clear regressions to avoid. Don't just dump the tables — interpret them.

## Common pitfalls

* **Forgetting `baseline` parity.** If the control arm isn't byte-identical to production, every comparison is contaminated. Run `ablation check` after every edit to `prompt_variants.py`, not just at the end.
* **Asking before proposing.** "What variants would you like?" is the wrong question. The user is invoking this skill *because* they want concrete suggestions. Always propose 3–5 concrete variants first, then iterate.
* **Picking too few games per matchup.** With N=2 (one pair per matchup), confidence intervals on win-rate span nearly the full [0%, 100%] range. Default to N=20 (10 pairs) for real signal; drop to N=4 only for smoke tests.
* **Self-play by default.** `--self-play` doubles cost and rarely changes conclusions. Off unless the user asks.
* **Modifying `harness.py`.** The production prompt stays in `harness.py`. All experimental prompts live in `prompt_variants.py`. If a variant turns out to be a win, promote it to `harness.py` in a *separate* PR.
* **Per-game extra metrics.** The runner ships generic columns (score, winner, length, crash, duration). If you want game-specific columns (Bargaining's `agreement_step`, chess's `mate_in_n`), add them to `games.csv` in a follow-up post-processing step — the runner's schema is intentionally minimal.

## Reference: the contract enforced by the runner

```python
# prompt_variants.py
from kaggle_environments.core_harness import GameHarness, ParseResult

class BaselineVariant:  # implements GameHarness
    def get_legal_moves(self, observation): ...
    def make_prompt(self, observation, move_history,
                    previous_response=None, previous_action=None): ...
    def parse_response(self, response, legal_action_strings,
                       *, observation=None): ...

VARIANTS: dict[str, GameHarness] = {
    "baseline": BaselineVariant(),
    "compact":  CompactVariant(),
    # ...
}
```

The runner imports this module, picks variants by name, and passes each to `create_agent_fn(variant, model_override=...)` per game. No further wiring needed.
