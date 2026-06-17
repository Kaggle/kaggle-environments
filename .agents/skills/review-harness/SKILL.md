---
name: review-harness
description: Review an existing LLM harness for correctness and gameplay-impacting bugs. Use when the user asks to "review", "audit", "check", "look over", or "find bugs in" a harness, or asks whether a harness has issues that could affect win rates. Covers static code review (prompt accuracy, parser robustness, common anti-patterns), optional replay-archive scanning to quantify real-world impact, and an optional cross-harness sweep for the same anti-patterns.
---

# Review a Kaggle-Environments LLM Harness

This skill audits an existing harness for bugs that could plausibly affect gameplay. It complements `create-harness` (which builds harnesses).

## Mindset

A good review finds **both** kinds of bugs:

- **Known bugs** — the patterns in the anti-pattern catalogue at the bottom of this document. Catching these is cheap (grep, then verify), high-confidence, and protects against regressions of issues we've already paid for once. *Always run the catalogue checks.* Skipping them because they feel mechanical is how harnesses ship with bugs we already knew how to find.
- **Unknown bugs** — the ones nobody has named yet. These are found by going to the engine, stress-testing the parser with adversarial inputs, reading the prompt as a hostile LLM would, and pulling on threads in the replay data. Each one becomes a new catalogue entry (Step 7) so the next reviewer gets it for free.

Structure the review around three questions, applied with both lenses:

1. **Is the prompt telling the model the truth about the game?** (Verify every concrete claim against the actual game engine; also walk the prompt-pattern row of the catalogue.)
2. **Does the parser robustly recover the model's intent across the messy responses real LLMs produce?** (Stress-test it with adversarial inputs; also walk the parser-pattern row of the catalogue.)
3. **Does the replay data show the harness behaving the way the code says it should?** (Generic intent-vs-action mismatch scan; also targeted detectors for each catalogue pattern.)

Neither lens dominates. The catalogue tells you the cheapest, most reliable bugs to find first; the discovery techniques tell you what to do when the catalogue runs out.

## When to use this skill

The user wants you to look at a harness with a critical eye. Distinct from `create-harness` (which produces new code).

If the user provides a replay archive (`.zip` of episode JSONs), also do the replay-scan section to measure realized impact and to surface bugs that static review can't see.

## Step 0: Establish scope

Before reading any code, confirm with the user:
- Which game/harness?
- Just the harness, or also the proxy/env?
- Replay archive available to scan? (If yes, get its path.)
- Should missing tests/wiring be flagged as bugs, or as separate concerns?
- Does a sibling variant exist (e.g. `<game>_arena` next to `<game>`)? If so, ask whether to include it — arena variants are usually copy-paste descendants of their base, so a bug in one almost always exists in the other.

These shape the depth and priority ranking. If the user has a preferred severity scale (e.g., "ignore stylistic stuff"), get that too.

## Step 1: Build ground truth from the game

You cannot review a prompt or parser without knowing what the game actually does. Don't rely on documentation, prior reviews, or the harness's own claims — go to the engine.

For OpenSpiel games:

```python
import pyspiel
game = pyspiel.load_game("<name>")
state = game.new_initial_state()
print(repr(state.observation_string(0)))
print(state.legal_actions())
state.apply_action(<action>)
# Reproduce edge cases the game has: collisions, captures, chance nodes,
# simultaneous turns, swap/pie rules, terminal states, draws, etc.
```

For custom envs, do the equivalent with the interpreter.

Things to learn *before* opening the harness:
- All distinct game phases (setup, normal turn, post-collision, terminal, …) and what observation each produces.
- Default parameters and what they control. (Is the game configurable? Do non-default configurations produce different observation formats?)
- The full action space, including special actions (PASS, swap, resignation, bidding values, …).
- **Every path to win, loss, and draw.** Read the engine's terminal-state logic exhaustively (the function called from `DoApplyAction`, every place `winner_` or its equivalent is set, every branch of `Returns()`). List them. The prompt must cover every one — including the unglamorous ones (repetition draws, max-length truncation, no-legal-moves-loses). "There are no draws" is a high-confidence red flag; verify it.
- Imperfect information edges: which observation fields are masked for which player.
- Anything the C++/Python source documents as a known quirk (look at `*.h`, `*.cc`, `*_test.cc` in the OpenSpiel install for a struct definition or test that locks in behavior). Numbered rule lists in the header file are gold — they often spell out exactly the edge cases the engine implements but the prompt forgets.
- **Engine vs canonical rules.** If the engine implements a rule differently from the game's standard rulebook (Wikipedia, tournament rules, the source paper), the prompt MUST follow the engine — that's what scores the game.

If a claim in the prompt or harness disagrees with what the engine actually does, that is a bug — full stop. **Prompt accuracy bugs are the highest-impact category** because they cost games on every turn the false invariant fires.

## Step 2: Static code review

Read the source critically. Do both halves of this step — neither alone is sufficient:

- **2a (catalogue walk)** finds the known bugs cheaply. Do this first; it's fast and high-yield.
- **2b–2f (discovery)** find the bugs the catalogue doesn't know about yet. Do this after, with the catalogue findings in mind so you can recognize related patterns.

### 2a. Walk the anti-pattern catalogue

Sweep the **anti-pattern catalogue** at the bottom of this document for every entry. For each one:

1. Run its **Detection** technique (grep, adversarial input, print-and-read, etc.).
2. If a hit, verify it's actually a bug in this codebase (some patterns are conditional — e.g., forward-iter `finditer` only matters in fallback paths).
3. Record severity, evidence, and the suggested fix.

This is mechanical work; do not skip it. Most production bugs are repeats of bugs we've already seen.

### 2b. Verify every prompt claim against ground truth

Print the prompt for a handful of representative states (start of game, mid-game, after a collision, terminal). Read each statement of the form *"you may X / you cannot Y / it always Z"* and check it against `legal_actions()` and `observation_string()` for that state. Examples of the kind of disagreement that has bitten real harnesses:

- *"You may nominate any cell except your own stones"* — but the engine also removes revealed-opponent cells from `legal_actions`.
- *"Move history shows the moves played in this game"* — but the framework appends collision attempts too, so the listed history is not a list of placements.
- *"Rows are numbered 1–9 from top to bottom"* — but the proxy actually emits row 1 at the bottom.

Whenever you find one, ask: *what other class of claim might be wrong?* — and verify those too.

**When the harness emits structurally different prompts for different roles, phases, or turn types** (cluemaster vs guesser; proposal vs utterance; setup vs play; mover vs non-mover at a chance node), reading each branch in isolation is not enough — a rule the model needs may be present in one branch and silently missing from another. Render *one prompt per branch* and build a coverage matrix. Rows are the engine's mechanical rules (enumerated once from `process_action` / `DoApplyAction` / wherever state transitions happen, with file:line refs from Step 1). Columns are the prompt branches.

| Engine rule (file:line) | Branch A prompt | Branch B prompt |
|---|---|---|
| Trap word → instant loss (`word_association.py:217`) | – | – |
| Positive N gives N+1 guesses (`word_association.py:170`) | – | yes |
| Game ends when one team's words depleted (`word_association.py:244`) | – | – |

A `–` in any column whose role's strategy depends on knowing the rule is a finding. `n/a` is fine (the rule doesn't apply to that role). Do not skip rows on the grounds that "this rule is obvious from the goal statement" — if it's a mechanical consequence the engine enforces, the prompt must say so explicitly, because the model only knows what's in the prompt. The matrix is the deliverable; gaps are concrete catalogue hits under "Rule disclosed to one prompt branch but not another."

### 2c. Stress-test the parser with adversarial responses

Construct synthetic LLM responses that *look* plausible and run the actual `parse_response` on them. The point is to expose failure modes the harness author didn't think of:

```python
# Examples of useful adversarial inputs — adapt to the game.
inputs = [
    # Happy path
    'I'll play e5.\n```json\n{"move": "e5"}\n```',
    # Multiple candidates in prose
    "I considered a1 then b3, but I'll play e5.",
    # No JSON, just prose
    "I'll play e5 because it controls the center.",
    # Echoes the board in the response
    "Board:\n    a b c d e f\n 1  . . . . . .\n...\n```json\n{\"move\": \"d3\"}\n```",
    # JSON nested in extra fences
    "```\n```json\n{\"move\": \"e5\"}\n```\n```",
    # Multiple JSON blocks (rethink scenario)
    '{"move":"a1"} ... wait, actually ```json\n{"move":"e5"}\n```',
    # Case variations
    '```json\n{"move":"E5"}\n```',
    # Whitespace / punctuation noise
    '```json\n{"move":"  e5.  "}\n```',
    # Illegal move in JSON
    '```json\n{"move":"z99"}\n```',
    # JSON with extra fields
    '```json\n{"reasoning":"...", "move":"e5", "confidence":0.9}\n```',
    # Empty / refusal
    "I cannot determine a good move.",
]
for r in inputs:
    print(repr(r[:60]), '→', parse_response(r, legal).legal_action)
```

You're looking for:
- Inputs the parser fails on that a human would clearly understand.
- Inputs the parser succeeds on that produce the *wrong* answer (e.g., picks an earlier-rejected move, captures a board-rendering artifact).
- Mismatch between what the JSON says and what gets returned.
- Anything the parser silently swallows (empty `raw_action`, no rethink context).

Don't constrain yourself to the catalogue's examples — invent inputs specific to *this* game's likely model outputs.

### 2d. Read the prompt as a hostile LLM would

Print one full prompt. Ask:
- Are any rule statements ambiguous? An LLM reading the prompt should not have to guess.
- Does the prompt invite mistakes? (E.g., "If the cell is occupied…" without specifying that the cell-is-occupied case is illegal to *initiate*.)
- Does the prompt enumerate legal moves? (Don't.) Does it give strategy advice? (Don't.)
- Does the prompt include data the model can't act on (e.g., raw JSON pasted instead of a readable rendering)?
- Does the rethink suffix actually help the model? (Showing back the previous response and the illegal move; not just "try again".)
- Is the output format described precisely enough that a strict parser will succeed?

### 2e. Trace one full turn end-to-end on paper

Pick a real-looking observation. Walk through:
1. `get_legal_moves(obs)` → what dict comes out?
2. `make_prompt(obs, history, ...)` → render the full text.
3. Imagine the LLM response. Try both an obedient response and a slightly-off one.
4. `parse_response(response, legal_strings)` → what does the framework receive?
5. The framework hands back `legal_action`; how does this become a `submission`?

At each step ask "what if this returned None / empty / a stale value?". Discover edge cases that aren't in your catalogue.

### 2f. Compare against a golden example

`checkers`, `dark_hex`, and `word_association` are reference implementations. If the harness diverges from those patterns, that's not automatically a bug — but ask why. A unique divergence is either a deliberate game-specific choice (document it) or an oversight (fix it).

## Step 3: Replay-archive scan (if available)

Static review tells you what *could* go wrong; the replay scan tells you what *did*, and it routinely surfaces bug categories that static review missed.

### 3a. Understand the replay schema

Replay JSONs from production have this rough shape per episode:

```
{
  "name": "<env_name>",
  "rewards": [r0, r1],
  "statuses": ["DONE", "DONE"],
  "steps": [
    [{agent0_state}, {agent1_state}],   # step 0 (setup)
    [{agent0_state}, {agent1_state}],   # step 1
    ...
  ]
}
```

Each `agent_state` contains:
- `status`: ACTIVE / INACTIVE / DONE / INVALID
- `observation`: the proxy's per-player observation (parse `observation.observationString` as JSON for OpenSpiel proxies)
- `action`: `{submission, actionString, thoughts, status}` — the move this agent made in this step, plus the LLM's final response in `thoughts`
- `info`: `{actionApplied, actionSubmitted, agentSelfReportedStatus, timeTaken}`

**Critical**: the pre-move board view for agent `j`'s move in step `i` is at `steps[i-1][j].observation`. The action in `steps[i][j].action` records what agent `j` played to transition from step `i-1` to `i`. The `thoughts` field is the *final successful* LLM response — retry attempts are not stored unless `include_generate_returns` is enabled in the config.

Always sanity-check the schema on one file first; the structure occasionally evolves:

```python
with open(files[0]) as f: r = json.load(f)
print(list(r['steps'][1][0].keys()))
print(list(r['steps'][1][0].get('action', {}).keys()))
print(set(r.get('statuses', [])))
```

### 3b. Survey aggregate outcomes

```python
status_counter = Counter()
for fp in files:
    with open(fp) as f: r = json.load(f)
    status_counter[tuple(r['statuses'])] += 1
print(status_counter)
```

If you see no `INVALID`/`ERROR` statuses, no game was lost to retry exhaustion — but bugs may still have caused suboptimal moves. Also surface:
- Distribution of `info.timeTaken`: outliers may indicate retry storms.
- Distribution of `info.actionSubmitted == info.actionApplied`: divergence indicates engine-level rejection.
- Distribution of `agentSelfReportedStatus`: anything other than `OK` is interesting.
- Game length distribution: very short games often indicate forfeits or trivial losses.

Any unusual aggregate is a thread to pull on.

### 3c. Compare what the *model* said to what was *submitted*

This is the most generally-useful replay check, and it doesn't require knowing what bug you're looking for. For every turn:

1. Extract the model's intent from `thoughts` (e.g., the JSON `move` field, or whatever your parser would prioritize).
2. Compare it to `actionString`. They should usually match.
3. **When they differ**, investigate. Each mismatch is either (a) a parser issue (intent was overridden) or (b) the recorded thoughts is from a different LLM call than the one that produced the action. Both are worth understanding.

```python
mismatches = []
for fp in files:
    with open(fp) as f: r = json.load(f)
    for i, step in enumerate(r['steps']):
        for j, agent in enumerate(step):
            a = agent.get('action') or {}
            thoughts = a.get('thoughts') or ''
            actionString = a.get('actionString')
            if not (thoughts and actionString): continue
            intent = extract_intent(thoughts)  # your game-specific extractor
            if intent and intent != actionString:
                mismatches.append((fp, i, j, intent, actionString))
```

This single check, applied to dark_hex, surfaced both Issue #1 (prompt overpermits known-opponent cells) and Issue #2 (forward-iter coord scan) and a previously-unknown rendered-board header artifact.

### 3d. Replay-driven differential tests

For any specific bug you suspect from static review, write a detector that walks the replay and counts occurrences:

- **"Did the harness ever do X?"** — e.g., did any submission land on a cell that was already known-occupied on the player's pre-move view?
- **"What would change if I fixed Y?"** — re-run `parse_response` after applying your fix; compare picks; count games where the action would have differed.

These checks turn theories into numbers. A bug that fires once in 40,000 turns is real but probably not urgent; a bug that fires in 5% of turns is.

### 3e. Look for surprises, not just bugs

Skim a dozen random `thoughts` fields. If the model is doing something the harness designer didn't anticipate — citing the move history weirdly, complaining about ambiguous rules, asking for clarification, repeatedly playing the same losing pattern — that's a signal. Often these surprises map back to prompt deficiencies you'd never find via grep.

## Step 4: Cross-harness sweep (optional)

If a bug is structural (in the parser, regex, or framework-glue code), check whether other harnesses share the anti-pattern. Two complementary approaches:

**Pattern-based grep** (catches known anti-patterns):

```bash
# --- All "first-match wins" surfaces (umbrella: last-mention-wins) ---

# Forward-iter finditer / findall over the response.
grep -rnE 'for [a-z_]+ in [a-zA-Z_]+\.find(iter|all)\(' \
    kaggle_environments/envs/*/harness*.py 2>/dev/null \
    | grep -v 'reversed('

# First-match regex extraction from the response.
grep -rnE '\.search\(response\)' \
    kaggle_environments/envs/*/harness*.py 2>/dev/null

# First-substring lookup against the response.
grep -rnE 'response\.find\(' \
    kaggle_environments/envs/*/harness*.py 2>/dev/null

# Iterate-legals "first that appears wins" loops (read each loop body
# to confirm it tests substring/regex containment against the response).
grep -rnE 'for [a-z_]+ in legal_action_strings' \
    kaggle_environments/envs/*/harness*.py 2>/dev/null

# JSON-extractor first-match variants (should use extract_last_json_object).
grep -rnE '_JSON_BLOCK_RE\.search|_BARE_JSON_RE\.search|_JSON_OBJECT_RE\.finditer' \
    kaggle_environments/envs/*/harness*.py 2>/dev/null

# Safe forms (for reference / sanity).
grep -rnE 'reversed\(list\(|reversed\(.*\.findall|response\.rfind|extract_last_json_object' \
    kaggle_environments/envs/*/harness*.py 2>/dev/null

# --- Other parser anti-patterns ---

# Cross-newline coord regex (\s* between letter and digit groups).
grep -rE '_(MOVE|COORD|CELL|MOVE_TOKEN)_RE\s*=\s*re\.compile\(.*\\s\*' \
    kaggle_environments/envs/*/harness*.py

# --- Prompt: hardcoded board dimensions ---
# Module-level board-size constants — suspect on any size-configurable game.
grep -rnE '_(BOARD|GRID|ROWS|COLS|NUM_(ROWS|COLS))[A-Z_]*\s*=\s*[0-9]+' \
    kaggle_environments/envs/*/harness*.py \
    kaggle_environments/envs/*/harness/*.py \
    kaggle_environments/envs/open_spiel_env/games/*/harness*.py 2>/dev/null
# Literal "NxM grid/board" or coordinate ranges baked into prompt templates.
grep -rnE '[0-9]+\s*x\s*[0-9]+\s*(grid|board)|[a-z]-[a-z]|1-[0-9]+' \
    kaggle_environments/envs/*/harness*.py \
    kaggle_environments/envs/*/harness/*.py \
    kaggle_environments/envs/open_spiel_env/games/*/harness*.py 2>/dev/null

# --- Prompt: only per-agent move_history shown ---
# Harnesses whose generate_prompt references move_history but NEVER reads a
# full-game history surface (proxy state_dict, pyspiel state.history(),
# PGN/movetext builder). A hit here means the prompt is likely showing only
# this agent's moves and labelling it as if it were the full game.
for f in kaggle_environments/envs/*/harness*.py \
         kaggle_environments/envs/*/harness/*.py \
         kaggle_environments/envs/open_spiel_env/games/*/harness*.py; do
    [ -f "$f" ] || continue
    grep -q 'move_history' "$f" || continue
    grep -qE 'state\.history\(\)|state\.full_history\(\)|state_dict.*history|"move_history"|movetext|action_history' "$f" \
        || echo "per-agent-only: $f"
done

# --- Prompt: move *count* shown instead of move *list* ---
# Harnesses that interpolate move_number / moves_played / turn_count into the
# template ("Moves played so far: 14") rather than rendering the actual moves
# ("a1b1, b3a3, ..."). A hit needs manual confirmation — counts CAN be useful
# alongside the move list (chess "Move 14:"), but a count with NO move list
# anywhere in the prompt is the clobber bug.
for f in kaggle_environments/envs/*/harness*.py \
         kaggle_environments/envs/*/harness/*.py \
         kaggle_environments/envs/open_spiel_env/games/*/harness*.py; do
    [ -f "$f" ] || continue
    grep -qE '\{(move_number|moves_played|turn_count|num_moves|ply)\}' "$f" || continue
    # Has a count interpolation. Does it also render an actual move list?
    grep -qE '\{(move_history|moves|history|movetext|pgn|action_log|move_list)[_a-z]*\}' "$f" \
        || echo "count-only (no move list in template): $f"
done

# --- Branching prompts: apply the per-branch rule-coverage matrix ---
# (See Step 2b. Hits here mean the harness emits structurally different
# prompts for different roles/phases/turn types, so a rule disclosed to
# one branch may be silently missing from another.)
#
# A. Files with 2+ PROMPT_TEMPLATE constants → multi-branch by template.
for f in kaggle_environments/envs/*/harness*.py \
         kaggle_environments/envs/*/harness/*.py \
         kaggle_environments/envs/open_spiel_env/games/*/harness*.py; do
    [ -f "$f" ] || continue
    n=$(grep -cE '_PROMPT_TEMPLATE\s*=' "$f")
    [ "$n" -ge 2 ] && echo "$n templates: $f"
done

# B. Role/phase predicate helpers (any `_is_<role>(...)` style classifier the
#    harness defines or calls; game-agnostic — catches cluemaster/guesser,
#    proposer, mover, attacker/defender, narrator, etc.).
grep -rnE '\b_is_[a-z_]+\(' \
    kaggle_environments/envs/*/harness*.py \
    kaggle_environments/envs/*/harness/*.py \
    kaggle_environments/envs/open_spiel_env/games/*/harness*.py 2>/dev/null

# C. Branching on a conventional phase/role/turn-type field inside the
#    harness — these field names recur across games. Add to the alternation
#    if a new harness uses a different conventional name.
grep -rnE 'if .*\b(turn_type|phase|role|stage|round_type|sub_phase|action_type)\b' \
    kaggle_environments/envs/*/harness*.py \
    kaggle_environments/envs/*/harness/*.py \
    kaggle_environments/envs/open_spiel_env/games/*/harness*.py 2>/dev/null

# D. Multiple `prompt = X_TEMPLATE.format(...)` sites in one file — another
#    sign of multi-branch composition independent of how dispatch is named.
for f in kaggle_environments/envs/*/harness*.py \
         kaggle_environments/envs/*/harness/*.py \
         kaggle_environments/envs/open_spiel_env/games/*/harness*.py; do
    [ -f "$f" ] || continue
    n=$(grep -cE 'prompt\s*=\s*[A-Z][A-Z0-9_]*_TEMPLATE\.format' "$f")
    [ "$n" -ge 2 ] && echo "$n template-select sites: $f"
done
```

**Behavior-based check** (catches the same logical bug across different syntaxes): for each harness, construct an adversarial response that should trigger the bug, run the harness's `parse_response`, and check the result. This catches variants of the bug that don't textually match a grep pattern.

For each candidate hit, verify:
1. The pattern is in a *fallback* (post-JSON-extraction) path, not a primary parser.
2. The regex / board rendering combination actually fires the bug — some regexes are restrictive enough to be safe even with the anti-pattern.

## Step 5: Report

Structure the writeup as:

1. **Verified correct.** What you checked and found working. Builds trust in the negative findings.
2. **Issues, ranked by gameplay impact.** Each issue:
   - One-line description.
   - Severity (Major / Medium / Minor).
   - Evidence (specific engine behavior, replay file:step references, grep matches, or adversarial-input output).
   - Concrete fix as a code snippet or sentence rewrite.
3. **Minor issues.** Wiring, tests, stylistic concerns — flagged but de-prioritized unless the user said otherwise.
4. **Realized impact** (if a replay scan was done). Numbers, then filenames. Distinguish "the bug fired" from "the bug changed game outcome" — the latter is often unknowable from logs alone, and you should say so.

**Don't bury the lede.** Lead with the most game-impacting bug, not the first one you found.

### Cite specific replay files for every replay-derived finding

When a replay archive is provided, the human reviewer's first instinct on any claim ("models get confused by line X", "the parser drops the move here", "this rule is misread") is to open a replay and see it for themselves. Make that one click, not a hunt. **Every replay-derived finding must name at least one concrete episode file the reviewer can open**, and where possible point to the exact `step` index and player slot. The bar is "could a reviewer who hasn't read your scan script reproduce the finding by opening the file you cited?".

Concretely:
- Cite the episode by its real path or basename (e.g. `replays/episode_01234.json` or `1700123456789.json`), not by the index into your scan list. Internal indices mean nothing to the reviewer.
- Pin the location inside the episode: `steps[12][0]` (step 12, agent 0) for a specific turn, plus the field you read (`action.thoughts`, `action.actionString`, `observation.observationString`).
- For prompt-comprehension findings ("models repeatedly misread the move-count line"), quote the offending model snippet AND name 2–3 episodes where it appears. One example is anecdote; three is a pattern the reviewer can trust without re-running your scan.
- For parser findings, name the episode and step whose `thoughts` exhibit the failure mode (multiple JSON blocks, prose-only response, etc.) — these are the files the reviewer will paste into a unit test.
- For aggregate claims ("fired N times across M episodes"), list a representative handful (3–5) of the episode paths in addition to the totals. Don't dump the full list — a sample is enough to spot-check.
- If your scan produced a per-finding artifact (CSV of mismatches, list of offending episodes), save it alongside the report and reference its path so the reviewer can drill in without re-running anything.

A finding that says "the parser picked the wrong move on 47 turns" is unactionable; the same finding with "47 turns across 31 episodes, e.g. `episode_00481.json` step 14 agent 1 (`thoughts` shows `e5` chosen, `actionString` is `a1`)" is something the reviewer can verify in 30 seconds.

## Step 6: Ask before fixing

When the review surfaces real bugs, ask the user whether to fix any/all before writing patches. The review is the deliverable — fixes are a follow-up.

## Step 7: Add to the catalogue

When the review surfaces a bug that isn't in the anti-pattern catalogue at the bottom of this document, **add it**. The catalogue's value grows by accumulation. A new entry should include: name, symptom, fix, and (ideally) the grep or adversarial-input pattern that catches it.

## Anti-pattern catalogue

Bugs that have been found in real reviews. Treat this as a starting point — find the next one.

### Parser

| Pattern | Symptom | Detection | Fix |
|---|---|---|---|
| **Forward-iter / first-match wins (umbrella)** | Whenever the parser scans the response for any kind of candidate — a regex match, a `findall`, an action tag, a fixed substring, a "first legal action that appears anywhere" loop, a JSON block — and picks the *first* one, it almost always picks a rejected option. Models enumerate alternatives ("considered a1, then b2, going with e5") before stating their final answer. The universal rule is **last-mention-wins**. This bug has shown up on at least six surfaces; treat the catalogue rows below as instances of the same defect, not separate bugs. | Grep for every surface (see `bash` block below this table). For each hit, verify it's a scan of the **response** (not a lookup against a single already-extracted candidate, which is fine). Where a replay archive is available, count fires by re-running the parser with last-wins and counting turns whose chosen action changes. | Use the patterns in the create-harness "Last-mention-wins" section. For JSON specifically, use the shared `extract_last_json_object` helper in `kaggle_environments.core_harness` rather than re-rolling fenced/bare regexes; pass `required_keys=(...)` so unrelated JSON in the reasoning is ignored. |
| ↳ *Forward-iter `finditer` / `findall`* | `for m in r.finditer(response):` or `for x in r.findall(response):` — picks the first match. Fired 13 turns / 10 episodes in the dark_hex prose fallback before the fix. | `grep -rnE 'for [a-z_]+ in [a-zA-Z_]+\.find(iter\|all)\(' kaggle_environments/envs/*/harness*.py` (skip hits already wrapped in `reversed(...)` / `reversed(list(...))`). | `for m in reversed(list(r.finditer(response))):` / `for x in reversed(r.findall(response)):` |
| ↳ *First-JSON-block pick* | `_JSON_BLOCK_RE.search(response)` (or any equivalent first-match regex) selects the first fenced/bare JSON object. Self-corrected later block is ignored. Fired 132 / ~155k LoA turns; structurally present in 18/18 OpenSpiel-game harnesses + word_association. | `grep -rnE '_JSON_BLOCK_RE\.search\|_BARE_JSON_RE\.search\|_JSON_OBJECT_RE\.finditer' kaggle_environments/envs/*/harness*.py`; verify by counting replays where `thoughts.count('```json') >= 2` and first/last `move` values differ. | Replace with `extract_last_json_object(response, required_keys=(...))` from `core_harness`. Do not reintroduce per-harness `_JSON_BLOCK_RE` / `_BARE_JSON_RE` constants. |
| ↳ *First action-tag wins* | `_FINAL_ANSWER_RE.search(response)` (or `response.find("Final Answer:")`) picks the first occurrence of the action tag. Models that revise their answer restate the tag; the trailing one is the intent. Also a faithfulness gap when porting from GameArena, whose `parse_move_from_response` uses `rfind` for the action tag. | `grep -rnE '\.search\(response\)\|response\.find\(' kaggle_environments/envs/*/harness*.py` (look for action-tag patterns specifically). | Take the last match: `matches = list(r.finditer(response)); m = matches[-1] if matches else None`. For plain substrings, use `response.rfind(...)`. |
| ↳ *Iterate legals, first that appears wins* | `for legal in legals: if legal in response: return legal` (or the regex equivalent). Order of `legals` is whatever the engine returns, so which legal "wins" is essentially undefined when several appear. | `grep -rnE 'for [a-z_]+ in legal_action_strings' kaggle_environments/envs/*/harness*.py` then read each loop body — flag any that test substring/regex containment against the response. | Track the legal whose rightmost occurrence (`response.rfind(legal)` or `list(re.finditer(pat, response))[-1].end()`) is latest; tie-break by length so longer/more-specific tokens beat shorter prefixes. See create-harness "Last-mention-wins" for the canonical shape. |
| **`\s*` between letter and digit in coord regex** | Captures `<col_letter>\n<row1>` from board header as a fake coord (`f1`, `j1`, etc.) | `grep -E '\\b\\(\\[a-z.*\\)\\\\s\\*\\(\\[0-9'` | Use `[ \t]*` or remove the gap entirely |
| **Notation tolerance missing for optional-looking engine markers** | OpenSpiel's `action_to_string` often appends markers that models routinely add or drop: backgammon's `*` (hit) and trailing `Pass` (per-die filler), checkers/chess `x` (capture), the hyphen separator `c3-d4`, castling `O-O`. The default matcher's whitespace-strip + case-fold isn't enough — `Bar/24` won't match `Bar/24 Pass`, so the model loses on a notation quibble. Backgammon's audit found 97.3% of episodes forfeited; adding `*` and `Pass` tolerance via a custom `matcher=` alone recovered 34.2% of forfeit turns (275 via `Pass`, 263 via `*`, 110 via both). | Enumerate a handful of `state.action_to_string` outputs at representative states (initial, mid-game, post-capture). For each marker that appears, ask: "would a model naturally omit or add this?" Stress-test the parser with the marker-stripped and marker-added variants of legal actions; any that fail to match are tolerance gaps. | Pass `matcher=` to `parse_json_action`. Build a normalization that strips the optional markers (e.g. `re.sub(r"[\sx\-*]+|\bPass\b", "", raw)`) and matches the normalized form against similarly-normalized legals. Keep the `matcher=` as the *only* place game-specific parsing lives — don't reintroduce a prose-scan fallback. |
| **Ghost-fallback / prose-scan rescue** | Any time the parser submits a move the model didn't explicitly state — by substituting a different legal token when stage-1 extraction was illegal, OR by guessing at intent from a coord/keyword/legal-string mentioned in the prose when stage-1 extracted nothing — that move is a phantom. It's usually a rejected option from the reasoning ("I considered g8 but went with h8" → h8 illegal → parser submits g8) or an incidental mention ("food is to my right, I'll go..." → parser submits "right" even though the model never finished the sentence). The model then sees a move it never chose in next turn's history and can't strategize. Found in 17 harnesses pre-fix; 7,477 illegal-stage1 fires across 1,481 / 2,008 havannah episodes (74%); every model in the dataset affected. | For each harness, find `parse_response`. **Easiest check:** does it call `parse_json_action` (or just delegate to it)? If yes, no second scan exists by construction; move on. If it rolls its own `parse_response`, any second scan after the structured-answer extraction is a ghost fallback — whether it fires when stage 1 was illegal or when stage 1 returned nothing. Both shapes substitute a move the model never explicitly chose. Replay-confirm: for any turn where `actionString` doesn't match any JSON / `Final Answer:` / payload intent in `thoughts`, the parser substituted. | Refactor `parse_response` to delegate to `parse_json_action(response, legal_action_strings, json_key=..., matcher=...)` from `core_harness`. If the harness has game-specific normalization, keep it as the `matcher=` callable — that's the one place game-specific parsing belongs. No secondary scan path, ever. The rethink loop, not a guessing fallback, is how illegal-or-missing structured answers should be handled — the model gets a chance to comply with the format and pick a legal move instead of the harness submitting something on its behalf. |
| **Free-form/enumerable misdispatch** | Free-form turn produces `legal_action=None` and is rejected | Inject an obs with `legal_action_strings=None` | Branch on `legal_action_strings is None` |
| **`raw_action` not set on failure** | Rethink prompt has no context to show the model | Construct an unparseable input; check `ParseResult.raw_action` | Always populate `raw_action` |
| **Over-aggressive normalization** | Parser strips characters that carry move meaning (e.g., chess SAN `x`) | Diff `normalize(legal)` against `legal` for representative moves | Whitelist what to strip, not what to keep |
| **Coordinate regex with no word boundary** | Matches `e5` inside `phase5` | Adversarial input | Add `\b` anchors |

### Prompt

| Pattern | Symptom | Detection | Fix |
|---|---|---|---|
| **Prompt reveals hidden information** | Prompt leaks data the receiving player should not see. Two common shapes: (1) **partial-info adversarial (A vs B)** — e.g. dark hex, where each player has their own per-player board view; the prompt for A must never include B's full board or unrevealed cells. (2) **co-op with teammates (AA vs BB)** — e.g. coin game arena (2v2), where A1 and A2 are teammates but still have private state; the prompt for A1 must not include A2's private observation. Once leaked, the game's information structure is broken and benchmark results become meaningless. | Render the prompt for each player at a state where private info is supposed to be hidden (mid-game in dark hex; partway through a co-op turn) and search the rendered text for the *other* player's private fields. Also audit which fields the harness reads from the obs — anything sourced from a global / shared / cross-player state dict instead of the per-player observation is suspect. | Source all per-player data through the proxy's per-player observation (e.g. `state.observation_string(player)`), never from a global state dict. If the proxy returns both players' boards when called with `player=None`, the harness must always pass an explicit `player`. Add a unit test that asserts player B's private fields do not appear in player A's prompt. |
| **Prompt invariant violation** | Rule statement disagrees with engine; model burns retries on "legal" moves | Print prompt; verify each "you may/cannot" claim against `legal_actions()` | Rewrite the claim |
| **Phantom feature claims (prompt describes behaviour no code implements)** | The prompt promises information or mechanics the harness/proxy doesn't actually surface. Models then waste reasoning trying to use the phantom field — or worse, infer made-up values. Two recurring shapes: (1) **drift** — the prompt was accurate against an older engine/proxy version but a field was renamed, removed, or the env switched parameters underneath it (gin rummy's prompt announced the "Oklahoma variant" but the env loads with `oklahoma=false`; mancala's prompt said remaining pieces stay in their pits at game end, but the engine sweeps them into the side's store); (2) **aspirational copy** — the harness author described a feature they planned but never wired up (oshi-zumo's `generate_prompt` docstring claimed opponent coin counts were "encoded as a hidden suffix" of `move_history` entries — no code path implemented that, the prompt only ever rendered the agent's own bids). | For every concrete field, rule, or value the prompt references, trace it backwards to a code path: (a) which proxy `state_dict()` key produces it? (b) which engine call produces THAT? (c) what env params is the engine actually loaded with — `make(env_name).configuration` or the `pyspiel.load_game(name, params)` call site? Anything you can't trace is a phantom. Cross-check the env factory's actual params against any "variant" or "rules" claims in the prompt. | Either implement the missing code path (and surface the field from the proxy) or delete the claim. When in doubt, delete — a prompt that lies is worse than one that says less. |
| **Missing or denied game-end paths** | Prompt either confidently *denies* a terminal condition the engine implements ("no draws under normal play" — but the engine draws on repetition), or omits one entirely ("a player with no legal moves loses" never stated). Models then can't strategically aim for, avoid, or recognize these outcomes. In LoA, 282/5,164 episodes (5.5%) drew via twofold repetition — a path the prompt told the model didn't exist. | Read the engine's terminal-state code top-to-bottom (`CheckTerminalState`, `DoApplyAction`, anywhere `winner_` or equivalent is set, anywhere `Returns()` can return zero / a draw value) and enumerate every path to win/loss/draw. Cross-check the prompt covers every one. Also check the engine's `.h` header — known quirks are often listed there as numbered rules. | Add the missing rule(s) to the prompt; remove or reword any sentence that confidently denies a condition the engine allows. |
| **Rule disclosed to one prompt branch but not another** | When the harness has multiple prompt branches (different roles, phases, or turn types), a mechanical rule may end up disclosed in only one branch's prompt. The branch that lacks the rule plays as if it doesn't exist. Example: word_association's Guesser prompt explains the bonus-guess mechanic (`number=N` → N+1 attempts) but the Cluemaster prompt doesn't, so the cluemaster systematically under-sizes clues by one. Easy to miss when reading each prompt in isolation — only the per-branch diff surfaces it. | Build the (engine-rule × prompt-branch) coverage matrix described in Step 2b. Any rule present in one branch but absent in another, where the missing branch's strategy depends on knowing it, is a finding. Also grep for harnesses with branching prompts (Step 4) so you know which ones to apply the matrix to. | Copy the missing rule statement into every branch whose strategy depends on it (usually near the existing "your goal is..." preamble). If the rule applies symmetrically, factor it into a shared preamble both branches concatenate. |
| **Engine-vs-rulebook divergence** | The OpenSpiel implementation deviates from the canonical/Wikipedia rules of the game, and the prompt teaches the canonical rule. The model loses confidently because it's playing by the wrong rulebook. Example: standard LoA says "both groups simultaneously connected → opponent wins"; OpenSpiel awards the win to the *moving* player because `current_player_`'s flood-fill is checked first. | Where the engine implements an unusual or contested rule, look it up (Wikipedia, MSO rules, the game's tournament authority). If the engine and rulebook disagree, the prompt MUST match the engine — the engine is what scores the game. | Match the prompt to the engine, not the canonical rules. Consider also flagging the divergence upstream so OpenSpiel can be fixed. |
| **Prompt enumerates legal moves** | Bloated prompt, trivializes legality-finding games | Read prompt | Describe rules; let the model derive legality |
| **Prompt gives strategy advice** | Biases the model toward the author's preferred play | Read prompt | Remove strategy hints; keep only rules and mechanics |
| **Move history framing wrong** | History includes events (collisions, retries) the prompt doesn't disclose | Compare `move_history` content with what the prompt says it contains | Annotate or describe accurately |
| **Prompt shows only this agent's moves, not the opponent's** | The framework's `move_history` argument is per-agent — it lists this agent's past actions and nothing the opponent did. A prompt that interpolates it as "move history" or "moves played so far" is showing the model half the game and labelling it as if it were the full game. The model can't reconstruct the position from incomplete information, and every claim it tries to make about what the opponent has done is grounded in nothing. Worst on games where opponent intent matters most (chess, dots-and-boxes, anything with capture/retaliation dynamics). | Grep for `move_history` use in `generate_prompt`. If the only history surface in the prompt is the per-agent argument (no proxy `state_dict()["move_history"]`, no `state.history()` reconstruction from `serializedGameAndState`, no PGN/movetext builder), it's this bug. Confirm by rendering the prompt mid-game and checking whether opponent moves appear anywhere. | Source full-game history from the proxy's `state_dict()` if it surfaces one (e.g. coin_game, ant_foraging_arena), or reconstruct from the deserialized pyspiel state (`state.history()` / `state.full_history()`, see `chess/harness.py` `_build_pgn_movetext` for a worked example). Render with player labels so the model can tell whose move is whose. If the proxy doesn't expose a full history, add it there rather than papering over the gap in the prompt. |
| **Prompt shows the move *count* instead of the move *list*** | The prompt interpolates a counter — `move_number`, `moves_played`, `turn_count`, `ply` — into a line like `"Moves played so far: 14"` instead of rendering the actual moves. The number tells the model how deep into the game it is and nothing else: it can't see what the opponent has been doing, can't detect repetitions, can't reason about threats that have been declared and not yet executed, and can't compare its current plan against what's already been tried. This is what clobber shipped with — the proxy exposed `move_number` and the harness echoed it; the model was effectively playing every position cold. Often co-occurs with the "per-agent only" bug above, but can stand alone when the harness has no history surface at all. | Grep the prompt template for count placeholders (`{move_number}`, `{moves_played}`, `{turn_count}`, `{num_moves}`, `{ply}`). For each hit, check whether the template ALSO renders an actual move list (a `{move_history}`/`{moves}`/`{movetext}`/`{pgn}` placeholder or equivalent rendered helper output). A count with NO move list in the same template is the bug. Render the prompt mid-game and look for an actual sequence of moves; if you only see `"Moves played so far: 14"` and never `"a1b1, b3a3, ..."`, that's it. | Replace the count interpolation with a rendered move list sourced from the proxy's `state_dict()["move_history"]` (add the field to the proxy if missing — clobber's proxy needed this; see `clobber_proxy.py` `state_dict()` for the parity-based player_id reconstruction), or reconstruct from `serializedGameAndState`. Render `"<move1>, <move2>, ..."` (or PGN-style for chess) and label it accurately: `"Moves played so far (both players, oldest first): a1b1, b3a3, ..."`. Keep `move_number` as a separate field if useful for orientation, but never as a substitute. |
| **Board dimensions hardcoded in the prompt** | The prompt bakes in a specific board size (`"10x10 grid"`, column letters `"a-j"`, `"rows 1-9"`) or coordinate-system text derived from a constant rather than from the live observation. When the env is loaded with a non-default `board_size` / `num_rows` / `num_cols` (havannah, dark_hex, amazons, dots_and_boxes — anything configurable), the prompt lies to the model: it describes a different grid than the one being played on, and the column/row range it permits no longer matches the legal action set. Models then propose moves outside the rendered board and lose to retry exhaustion, or refuse to use coordinates the prompt didn't authorize. | Grep the prompt template for hardcoded dimension strings (digits next to "grid"/"board"/"rows"/"columns"), hardcoded coordinate ranges (`a-j`, `1-9`, etc.), and module-level constants like `_BOARD_SIZE = 10`. For each hit, ask: does the env support multiple sizes? (Check the env's `configuration` / the `pyspiel.load_game(name, params)` site for size parameters.) If yes, render the prompt at the non-default size and verify the dimension text matches. The `amazons/harness.py:113` `_board_dims` helper is the canonical pattern to compare against — anything simpler is suspect on a size-configurable game. | Read board dims from the parsed `observationString` (proxy typically exposes `board`, `num_rows`/`num_cols`, or `board_size`); fall back to `state.get_game().get_parameters()` from the deserialized pyspiel state. Interpolate `{num_rows}`/`{num_cols}` into the template and derive coordinate-range text from those dims (e.g. compute column letters as `string.ascii_lowercase[:num_cols]`, not a literal `"a-j"`). Add a unit test that renders the prompt at two different configured sizes and asserts each renders correctly. |
| **Coordinate convention mismatch** | Prompt says "rows top→bottom" but proxy emits bottom→top | Print proxy's `state_dict()` for known position; cross-check | Align prompt or proxy |
| **Overloaded symbol for distinct state** | The prompt uses one symbol (a glyph, token, label, marker) to represent two or more structurally distinct pieces of game state, with the disambiguator being context the model has to reconstruct (position in a grid, surrounding tokens, the legend, prior knowledge of the rules). Even when the legend documents the overload, models routinely conflate the meanings and reason about state that isn't there — proposing moves they think are legal but aren't, or missing options they think are blocked. This applies to any encoding choice: one character for two cell types, one label for two action kinds, one numeric for two resource pools, etc. | Render the prompt at a non-trivial mid-game state. For each symbol that appears (board glyphs, action tokens, history entries, status labels), list every distinct piece of game state it can represent. Any symbol with two or more meanings is a finding, regardless of whether the legend explains the disambiguation — the burden of disambiguation is the bug. Cross-check by reading replay `thoughts` for traces that confidently misclassify state. | Assign each semantic role its own unique symbol. Update the legend AND verify by eye that no two roles share an encoding. If the state has redundant structured form available (e.g. the proxy's JSON `state_dict`), consider surfacing it alongside the human-readable rendering as a second view the model can cross-check against. |
| **Unit mismatch between two numbers in the same prompt** | The prompt shows two numerics with related meanings in different units, with no label saying which is which. Example: coin_game_arena rendered `episode_length: 20` (per board) and `moves_remaining: 36` (global, across both boards) on the same screen — a model planning end-game urgency had to divide by 2 to reconcile. Tends to appear when an obs field is computed at the engine level (in one unit) and then re-used in a prompt next to a constant defined in another unit. | Render the prompt and pick out paired numerics whose names invite comparison (durations, counts remaining, scores). If their units differ — per-player vs per-team, per-board vs global, per-turn vs per-step — that's the bug. | Normalize at the harness layer (e.g. compute per-board moves remaining from the per-board history length), or rename the noisy field so units are unambiguous. Surface the value as a dedicated sentence with units in the prose, not buried in a JSON dump. |
| **Player-asymmetric prompt text not mirrored** | Sentences that should differ by player are written once with one player's perspective baked in, then served to both. Direction/orientation language is the usual culprit — oshi-zumo's "lower is your goal, higher is the opponent's" was correct only for Player 1; 7.7% of P0 turns echoed the wrong direction in their reasoning, 735 stated it without later correction. Mancala's diagram was labelled "shown from your point of view" but never actually rotated per player. Symmetric for-both-players sentences and "your"/"opponent" labels are fine; *directional* claims ("toward rank N", "lower-numbered", "left half", "first move") almost always need to mirror. | Render the prompt for `player_id=0` and `player_id=1` at the same state; diff them. Anything directional that is byte-identical between the two — and shouldn't be — is the bug. Don't just spot-check Player 0; the asymmetry is usually correct for whichever player the author wrote it for and wrong for the other. | Parameterize the asymmetric text on `player_id` (e.g. `forward_rank = 8 if player_id == 0 else 1`), or factor it through a helper like `_player_info(player_id) -> (label, code, direction_text)`. Add a unit test that asserts the differing words appear in the right prompt and only the right prompt. |
| **Phase-classifier fallback misroutes one phase to another** | Multi-phase games dispatch to per-phase prompt templates. When the dispatcher has a fallback branch (a `default:`, or a "if not phase X, treat as phase Y" rule keyed on legal-action shape), an unhandled or unrecognized phase silently routes to a template whose instructions are completely wrong for the actual situation. Gin rummy's Wall phase was missing from `_PHASE_INSTRUCTION`, and the legals-based fallback (Wall and Layoff both have `{Pass, Knock}`-shaped legals) routed every Wall turn to a prompt that opened with "Your opponent knocked…" — the opponent had not knocked. The model has no way to detect the mismatch. | Enumerate every phase the engine can produce (read the engine's state machine — the `*Phase` enum or its equivalent, every place `phase_` is set). Verify each phase has its own template entry. Look for `else`/`default`/"if not X" branches in the dispatcher; for each, ask which engine-distinguishable phases could land there and whether the fallback text is correct for *all* of them. If two phases share legals-shape, legals are not a sound phase classifier. | Use the engine's explicit phase identifier (string or enum) as the dispatch key, not legals-shape. Make the table exhaustive — assert at construction time that every engine phase has a template. Delete the fallback branch, or make it raise so a new engine phase fails loudly instead of silently misrouting. |
| **No-op rethink suffix** | Rethink does not include previous response or attempted action | Construct a parse-failure case; inspect retry prompt | Include `previous_response[:N]` and `previous_action` |
| **Rethink suffix uses the wrong shape for the failure** | Parse failures come in two flavours and need two different rethinks. (1) **`previous_action is None`** (parser found no extractable answer): no action string to show — lead with `previous_response` (last 500 chars, not first 500 — the model's conclusion is at the end) followed by the output-format spec restated with a clean placeholder and a concrete example. (2) **`previous_action` is set** (model produced parseable JSON but the move was illegal): the action string is the most useful signal — lead with it (`"You suggested {previous_action} but this is not legal."`); don't also include the full previous response (it's noise that dilutes the correction signal). A brief tail in each template mentions the other failure mode (format vs legality). Many harnesses use a single suffix for both cases: either no format reminder at all (breaks unparseable retries) or a format reminder on every retry (wrong lead for illegal-move retries); a common variant truncates the previous response with `[:500]` (first 500 chars) which usually keeps the preamble and drops the model's actual answer. | Render both cases and read them: `make_prompt(obs, [], previous_response="some prose", previous_action=None)` vs `make_prompt(obs, [], previous_response="...", previous_action="z99")`. Verify: (a) illegal-case leads with `previous_action` and does NOT include `previous_response`; (b) unparseable-case includes the last 500 chars of `previous_response` and restates the JSON format; (c) the JSON example uses a clean `<placeholder>` followed by a separate concrete `Example:` line, not a literal-looking `"<placeholder>, e.g. concrete"` string. | Branch `RETHINK_SUFFIX` selection on `previous_action`. Use two named templates (e.g. `RETHINK_UNPARSABLE`, `RETHINK_ILLEGAL`) — the create-harness skill has the canonical shape. Truncate `previous_response` with `[-500:]`, not `[:500]`. |

### `get_legal_moves`

| Pattern | Symptom | Detection | Fix |
|---|---|---|---|
| **No serialized-state fallback** | When `legalActions` isn't in obs, returns `{}` and agent burns turn | Construct an obs missing `legalActions` | Deserialize `serializedGameAndState` and call `state.legal_actions()` |
| **Returns wrong type for free-form** | Returns `{}` instead of `None` on free-form turns; framework treats it as enumerable with no options | Inject free-form-turn obs | Return `None` explicitly |
| **Diagnostic prints in production** | Stderr noise on every turn | `grep print` in the harness | Remove or guard behind a debug flag |

### Structure / wiring

| Pattern | Symptom | Detection | Fix |
|---|---|---|---|
| **Relative imports in harness** | Production loader (single-file) fails to import | `grep 'from \.' harness.py` | Use absolute `kaggle_environments...` imports |
| **No `test_llm_game.py`** | No in-repo end-to-end LLM sanity test | `ls` for `test_llm_game.py` | Create from `create-harness` template |
| **No `harness_test.py`** | No unit tests for prompt/parser/legal-moves | `ls` for the test file | Create from `create-harness` template |

## Reference files

- `kaggle_environments/core_harness.py` — `GameHarness` protocol, `ParseResult`, `create_agent_fn`, retry loop, telemetry. Read this before reviewing any harness.
- `.agents/skills/create-harness/SKILL.md` — the construction-side counterpart; cross-reference rules and conventions.
- Golden examples to anchor expectations:
  - `kaggle_environments/envs/open_spiel_env/games/checkers/harness.py` — modern enumerable shape: delegates to `parse_json_action`, branches `render_rethink_suffix`, demonstrates a phase-conditional prompt section (multi-jump continuation)
  - `kaggle_environments/envs/open_spiel_env/games/dark_hex/harness.py` — same modern shape with a custom `matcher=` callable for notation tolerance; per-player rendering for imperfect-information games
  - `kaggle_environments/envs/word_association/harness/main.py` — mixed free-form + enumerable (non-OpenSpiel)
