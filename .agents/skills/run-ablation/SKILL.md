---
name: run-ablation
description: Run a prompt-ablation study on a kaggle-environments game's LLM harness. Use when the user mentions "ablation", "prompt sensitivity", "test prompt variants", "compare prompts", "ablate the prompt", "prompt rewrite study", or asks whether their prompt wording is doing real work. Bootstraps a prompt_variants.py if missing, proposes new variants interactively, then runs a paired-seat tournament and reports the leaderboards.
---

# Run a Prompt Ablation Study

This skill drives the end-to-end prompt-sensitivity workflow for one game's harness: discover or bootstrap variants, propose additional ones in conversation with the user, run the paired-seat tournament, and report the findings. It is completely standalone — it does not depend on or modify the `create-harness`, `review-harness`, or `create-environment` skills.

Two CLI tools work together:

```bash
# Run the tournament
python -m kaggle_environments.ablation check --env <env_name>
python -m kaggle_environments.ablation run --env <env_name> --models <csv> --games <N> --out <dir>

# Post-hoc statistical analysis (permutation test against the null variant's noise floor)
python -m kaggle_environments.ablation_analysis --csv <dir>/games.csv --baseline baseline --null null
```

The runner contract:

* Every game's harness directory contributes a `prompt_variants.py` exposing `VARIANTS: dict[str, GameHarness]`. Each variant **is** a `GameHarness` (implements `get_legal_moves` / `make_prompt` / `parse_response`), so `create_agent_fn(variant)` works directly.
* The `baseline` variant must be byte-identical to the production `harness.py`. The `check` subcommand enforces this against seeded observations.
* The `null` variant must be a byte-identical second copy of baseline. The runner schedules independent cells for it; the difference between `null`-vs-`baseline` rankings is pure LLM-sampling noise, which calibrates the noise floor for the permutation test.
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

* **Compact** — same *content* as baseline, just tighter prose. Drops worked examples, repeated reminders, loss-threat sentences, flavor padding. **Keeps every computed helper baseline provides** — per-turn counters, accept-previews like "you would receive [...]", rendered complements in history rows, end-of-prompt constraint reminders. Tests whether the verbose prose padding is load-bearing while holding the helper surface constant.
* **Minimal** — strips both prose *and* derived/helper info. Keeps mechanics (private-vs-public state, action semantics, reward formula, terminal conditions) but removes anything the model can compute itself from the basic state: per-turn counters (countable from history), accept-previews (subtractable from the last offer), complement-rendered history (subtractable from pool), constraint reminders. Tests whether the model can derive what baseline spoon-feeds.

Compact and Minimal are **distinct ablation axes**, not "more terse" vs "even more terse." If you're proposing both, make sure they actually test different things — Compact = "do we need the prose?", Minimal = "do we need the helpers?" The mechanics keep is non-negotiable in both (see Step 4).
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
2. **`null`** must be byte-identical to `baseline` but registered under a separate name. The simplest implementation is just `NULL = PromptVariant(name="null", ..., build_body=<same as BASELINE>, ...)` — share every field with `BASELINE` except `name`. This guarantees no accidental drift between the two. The runner will schedule independent cells for `null`, and the resulting `null`-vs-`baseline` Σ|Δrank| measures pure LLM-sampling noise that the analysis step uses as the noise floor. **Always include `null` in `VARIANTS`** — without it, the permutation test has to estimate the noise floor by resampling, which overestimates noise at small N and drowns real effects.
3. **Each new variant** is a subclass of `BaselineVariant` (or sibling class) that overrides `make_prompt` (and `parse_response` if the schema changed — e.g. `generic_names` uses `{a, b, c}` JSON keys and needs an alias map).
4. Expose `VARIANTS = {"baseline": BaselineVariant(), "null": NullVariant(), ...}` at module level. The runner enforces a `"baseline"` entry; the analysis script defaults to looking for `"null"` and degrades gracefully if missing.

Keep the variant code in one file. Do **not** import shared prompt fragments from a sibling helper module — `harness.py` is manually deployed and versioned separately, so a shared-module import would be a deploy-isolation footgun.

### Hard rule: never strip game-mechanics information

Decoration is fair game (worked examples, flavor sentences, repeated reminders, loss-threat language, computed per-turn helpers). **Mechanics is not.** Every variant — including the most aggressive `minimal` — must convey the information the model needs to *understand the game*:

* **What's private vs. public.** If some piece of state is hidden from the opponent (Bargaining: per-player valuations; Werewolf: roles; Avalon: alignment), say so. Models default to assuming symmetric information and play very differently when wrong.
* **Action semantics.** What does each move-type mean? What does each field in the JSON schema represent? ("keep" = items YOU retain; "agree" = accept opponent's last offer.)
* **Reward / win condition.** How is the score computed?
* **Terminal / timeout behavior.** What happens if no resolution is reached?

If you find yourself stripping any of these, stop — you're testing comprehension, not prompt design. The variant's poor performance can't be attributed to the prompt-design choice you wanted to ablate. **Audit each variant by reading it as a model with no prior game knowledge would.** Can you play correctly from this prompt alone? If not, restore the missing mechanics.

### Hard rule: every prompt must ask for reasoning before JSON

Every variant's template — including the most aggressive `minimal` strip — must explicitly instruct the model to **output** its reasoning *before* outputting the final action JSON. The wording must make clear that the reasoning belongs in the response, not just in the model's head. Two safe phrasings:

* baseline-style: `"Respond with your reasoning, then conclude with a JSON block of EITHER form:"`
* terse-style: `"Respond with your reasoning, then end your response with JSON, one of:"`

**Avoid weak verbs** like "Reason briefly through your move, then respond with JSON" — models often interpret this as "think about it internally, then output JSON" and skip writing reasoning into the response. The instruction must start with an output verb (`Respond`, `Write`, `Explain`, `Output`) applied to the reasoning, not just to the JSON.

This is non-negotiable. Models perform noticeably worse on these games when they skip writing out reasoning; the chain-of-thought is load-bearing, not a stylistic preference. When you propose new variants in Step 3, when you write the variant classes in this step, and when you review the final `prompt_variants.py` before running, *check that every template's final-output instruction puts an output verb on the reasoning, not just on the JSON*. If a variant's whole point is "strip everything", strip rules and helpers and examples — but keep the reason-first instruction.

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

* The env, the variant list (always including `null`), the model list, `--games` (paired games per matchup — must be even).
* The total cell count: `K · M(M−1)/2 · games`, where K **includes** the null variant. Add `M · games` per leaderboard if `--self-play`.
* A rough cost estimate. For Bargaining-class games (≤10 prompt-response rounds, ~1k–3k prompt tokens), assume ~20 LLM calls per game.
* The output directory.

Default suggestion if the user hasn't specified: `--games 30`, all variants (including `null`), `--concurrency 8`. **Don't go below N=20** unless smoke-testing — at N=10, the permutation test has poor power and most variants come back inside noise even when their qualitative rank shifts look real. Wait for explicit go before invoking the runner.

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

**First** run the permutation-test analysis. This is the headline output — it tells you which rank shifts are real vs. noise:

```bash
uv run python -m kaggle_environments.ablation_analysis \
  --csv <out>/games.csv \
  --baseline baseline --null null \
  --permutations 2000 \
  --out <out>/analysis.md
```

The script writes a Markdown table per real variant with:
* **observed Σ|Δrank|** — how much the leaderboard reshuffled vs. baseline
* **null-floor Σ|Δrank|** — what the byte-identical null variant produced (pure sampling noise)
* **permutation p-value** — fraction of label-shuffles producing a shift this large or larger

**Lead with the analysis result, not the raw `summary.md` tables.** A variant whose observed Σ|Δrank| is comfortably above the null floor *and* has p < 0.05 is a real prompt-sensitivity finding. A variant whose observed Σ|Δrank| is near or below the null floor isn't moving the leaderboard meaningfully, even if the raw rank table looks suggestive.

Then surface the supporting context from `summary.md`:

* **The cross-variant rank-shift table** — for the variants the analysis flagged as significant, show *what* moved (which model swapped with which).
* **The anomalies section.** Variants with >5% crash rate or hard errors usually indicate the prompt is under-specified for some model — investigate before treating the variant's rankings as comparable to others.
* **Mean-score deltas** for context on the *direction* of effect (variant helps or hurts each model).

Make one or two concrete recommendations: which variant(s) look like statistically-supported upgrades to baseline, which look like clear regressions to avoid. Don't just dump the tables — interpret them through the lens of "did this clear the noise floor?"

### When the noise floor swallows everything

If every real variant has Σ|Δrank| at or below the null floor, the honest answer is **the prompt doesn't meaningfully affect rankings at this sample size with these models on this game.** Don't massage marginal results — say so plainly, and recommend either (a) running at higher N if the user has budget, (b) trying more aggressive variants (the current ones may be too close to baseline), or (c) accepting that prompt scaffolding is mostly decoration for this task.

## Common pitfalls

* **Stripping mechanics, not decoration.** "Minimal" doesn't mean "remove everything." Variants must preserve private-vs-public splits, action semantics, reward computation, and terminal conditions. If a model couldn't play the game from the variant's prompt alone, you're not isolating a prompt-design effect — you're measuring comprehension failure. See the hard-rule subsection in Step 4.
* **Omitting the reason-first instruction from a variant.** Every prompt (baseline + all variants, however terse) must ask the model to reason before outputting JSON. A variant that drops it is testing a confounded prompt — the lost performance from skipping reasoning will dominate whatever else you were trying to ablate. See the hard-rule subsection in Step 4.
* **Forgetting `baseline` parity.** If the control arm isn't byte-identical to production, every comparison is contaminated. Run `ablation check` after every edit to `prompt_variants.py`, not just at the end.
* **Skipping the `null` variant.** Without it, you have no calibrated noise floor and the permutation test has to estimate one by resampling (which overestimates noise at small N). The cost is one extra variant in the matrix — always include it.
* **Reporting raw rank tables as findings.** The rank table in `summary.md` is *unweighted by significance*. A #1↔#2 swap looks dramatic but at N=10 it happens regularly from sampling noise. Always run `ablation_analysis.py` and lead with its p-values; the rank table is supporting evidence, not the headline.
* **Asking before proposing.** "What variants would you like?" is the wrong question. The user is invoking this skill *because* they want concrete suggestions. Always propose 3–5 concrete variants first, then iterate.
* **Picking too few games per matchup.** With N=2 (one pair per matchup), confidence intervals on win-rate span nearly the full [0%, 100%] range. Default to N=30 (15 pairs) for the permutation test to have real power; drop to N=4 only for smoke tests. At N=10 most real prompt effects come back inside the noise envelope.
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
    "null":     NullVariant(),       # byte-identical duplicate of baseline
    "compact":  CompactVariant(),
    # ...
}
```

The runner imports this module, picks variants by name, and passes each to `create_agent_fn(variant, model_override=...)` per game. The analysis step (`ablation_analysis.py`) reads the resulting `games.csv` and uses the `null` variant as the noise floor for permutation tests against every other variant. No further wiring needed.
