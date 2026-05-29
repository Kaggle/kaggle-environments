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

`chess`, `connect_four`, and `word_association` are reference implementations. If the harness diverges from those patterns, that's not automatically a bug — but ask why. A unique divergence is either a deliberate game-specific choice (document it) or an oversight (fix it).

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
| **JSON-only parser, no fallback** | Model adds commentary inside the fence, parser fails | Read parser; check for multi-stage fallback | Multi-stage: fenced block → bare JSON → prose scan |
| **Free-form/enumerable misdispatch** | Free-form turn produces `legal_action=None` and is rejected | Inject an obs with `legal_action_strings=None` | Branch on `legal_action_strings is None` |
| **`raw_action` not set on failure** | Rethink prompt has no context to show the model | Construct an unparseable input; check `ParseResult.raw_action` | Always populate `raw_action` |
| **Over-aggressive normalization** | Parser strips characters that carry move meaning (e.g., chess SAN `x`) | Diff `normalize(legal)` against `legal` for representative moves | Whitelist what to strip, not what to keep |
| **Coordinate regex with no word boundary** | Matches `e5` inside `phase5` | Adversarial input | Add `\b` anchors |

### Prompt

| Pattern | Symptom | Detection | Fix |
|---|---|---|---|
| **Prompt reveals hidden information** | Prompt leaks data the receiving player should not see. Two common shapes: (1) **partial-info adversarial (A vs B)** — e.g. dark hex, where each player has their own per-player board view; the prompt for A must never include B's full board or unrevealed cells. (2) **co-op with teammates (AA vs BB)** — e.g. coin game arena (2v2), where A1 and A2 are teammates but still have private state; the prompt for A1 must not include A2's private observation. Once leaked, the game's information structure is broken and benchmark results become meaningless. | Render the prompt for each player at a state where private info is supposed to be hidden (mid-game in dark hex; partway through a co-op turn) and search the rendered text for the *other* player's private fields. Also audit which fields the harness reads from the obs — anything sourced from a global / shared / cross-player state dict instead of the per-player observation is suspect. | Source all per-player data through the proxy's per-player observation (e.g. `state.observation_string(player)`), never from a global state dict. If the proxy returns both players' boards when called with `player=None`, the harness must always pass an explicit `player`. Add a unit test that asserts player B's private fields do not appear in player A's prompt. |
| **Prompt invariant violation** | Rule statement disagrees with engine; model burns retries on "legal" moves | Print prompt; verify each "you may/cannot" claim against `legal_actions()` | Rewrite the claim |
| **Missing or denied game-end paths** | Prompt either confidently *denies* a terminal condition the engine implements ("no draws under normal play" — but the engine draws on repetition), or omits one entirely ("a player with no legal moves loses" never stated). Models then can't strategically aim for, avoid, or recognize these outcomes. In LoA, 282/5,164 episodes (5.5%) drew via twofold repetition — a path the prompt told the model didn't exist. | Read the engine's terminal-state code top-to-bottom (`CheckTerminalState`, `DoApplyAction`, anywhere `winner_` or equivalent is set, anywhere `Returns()` can return zero / a draw value) and enumerate every path to win/loss/draw. Cross-check the prompt covers every one. Also check the engine's `.h` header — known quirks are often listed there as numbered rules. | Add the missing rule(s) to the prompt; remove or reword any sentence that confidently denies a condition the engine allows. |
| **Engine-vs-rulebook divergence** | The OpenSpiel implementation deviates from the canonical/Wikipedia rules of the game, and the prompt teaches the canonical rule. The model loses confidently because it's playing by the wrong rulebook. Example: standard LoA says "both groups simultaneously connected → opponent wins"; OpenSpiel awards the win to the *moving* player because `current_player_`'s flood-fill is checked first. | Where the engine implements an unusual or contested rule, look it up (Wikipedia, MSO rules, the game's tournament authority). If the engine and rulebook disagree, the prompt MUST match the engine — the engine is what scores the game. | Match the prompt to the engine, not the canonical rules. Consider also flagging the divergence upstream so OpenSpiel can be fixed. |
| **Prompt enumerates legal moves** | Bloated prompt, trivializes legality-finding games | Read prompt | Describe rules; let the model derive legality |
| **Prompt gives strategy advice** | Biases the model toward the author's preferred play | Read prompt | Remove strategy hints; keep only rules and mechanics |
| **Move history framing wrong** | History includes events (collisions, retries) the prompt doesn't disclose | Compare `move_history` content with what the prompt says it contains | Annotate or describe accurately |
| **Coordinate convention mismatch** | Prompt says "rows top→bottom" but proxy emits bottom→top | Print proxy's `state_dict()` for known position; cross-check | Align prompt or proxy |
| **No-op rethink suffix** | Rethink does not include previous response or attempted action | Construct a parse-failure case; inspect retry prompt | Include `previous_response[:N]` and `previous_action` |

### `get_legal_moves`

| Pattern | Symptom | Detection | Fix |
|---|---|---|---|
| **No serialized-state fallback** | When `legalActions` isn't in obs, returns `{}` and agent burns turn | Construct an obs missing `legalActions` | Deserialize `serializedGameAndState` and call `state.legal_actions()` |
| **Returns wrong type for free-form** | Returns `{}` instead of `None` on free-form turns; framework treats it as enumerable with no options | Inject free-form-turn obs | Return `None` explicitly |
| **Diagnostic prints in production** | Stderr noise on every turn | `grep print` in the harness | Remove or guard behind a debug flag |

### Structure / wiring

| Pattern | Symptom | Detection | Fix |
|---|---|---|---|
| **Missing `agent_fn`** | `env.run([harness.py, ...])` doesn't load an LLM agent | `grep create_agent_fn` in the harness | Add adapter class + `agent_fn = create_agent_fn(adapter())` |
| **Relative imports in harness** | Production loader (single-file) fails to import | `grep 'from \.' harness.py` | Use absolute `kaggle_environments...` imports |
| **No `test_llm_game.py`** | No in-repo end-to-end LLM sanity test | `ls` for `test_llm_game.py` | Create from `create-harness` template |
| **No `harness_test.py`** | No unit tests for prompt/parser/legal-moves | `ls` for the test file | Create from `create-harness` template |

## Reference files

- `kaggle_environments/core_harness.py` — `GameHarness` protocol, `ParseResult`, `create_agent_fn`, retry loop, telemetry. Read this before reviewing any harness.
- `.agents/skills/create-harness/SKILL.md` — the construction-side counterpart; cross-reference rules and conventions.
- Golden examples to anchor expectations:
  - `kaggle_environments/envs/open_spiel_env/games/chess/harness.py` — enumerable + byte-identical migration from GameArena
  - `kaggle_environments/envs/open_spiel_env/games/connect_four/harness.py` — uses `reversed(list(finditer))` correctly
  - `kaggle_environments/envs/word_association/harness/main.py` — mixed free-form + enumerable
