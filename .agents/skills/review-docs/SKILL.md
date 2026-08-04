---
name: review-docs
description: Audit a game environment's README.md and AGENTS.md against its engine implementation. Use when the user asks to "review", "audit", "check", or "fact-check" environment docs, when players report doc/engine discrepancies, or before a competition launch. Also use proactively after changing an environment's engine (`kaggle_environments/envs/<game>/<game>.py`) when a sibling README.md or AGENTS.md exists — rebalances and mechanic changes routinely leave those docs stale. The engine is always the source of truth — docs get fixed, not the engine.
---

# Review Environment Docs Against the Engine

Audit `kaggle_environments/envs/<game>/README.md` and `AGENTS.md` for claims that disagree with the engine.

## Core rule

**The engine is the source of truth. Fix the docs, not the engine.** Agents play against the implementation, and changing engine behavior late invalidates existing submissions and replays.

Exception: if a mechanic is clearly a *bug* rather than an undocumented quirk, flag it and ask. Don't silently enshrine a bug as intended.

## Method: simulate, don't reason

Never verify a claim by reading code and reasoning about it. Execute the engine and observe. Cheapest entry point first:

```python
# 1. Internal helpers — isolates one rule
from kaggle_environments.envs.<game> import <game> as G
G._new_<entity>(...)      # constructor: initial field values
G._apply_<action>(...)    # one action
G._<periodic_hook>(...)   # end-of-turn / round / phase transition

# 2. Full env — catches validation the helpers skip
from kaggle_environments import make
env = make("<game>", configuration={...}); env.reset(2); env.step([a0, a1])

# 3. OpenSpiel games
import pyspiel; state = pyspiel.load_game("<name>").new_initial_state()
```

Step through the **entire lifecycle** — creation to terminal state — recording observable state at every step, not just the endpoints. Most doc bugs live in the middle: a value that plateaus early, a counter starting at the wrong number, a transition firing a step sooner than the prose implies.

Gotchas:
- Don't name scratch files after stdlib modules (`numbers.py` breaks the import chain).
- Helpers may *replace* an object rather than mutate it (one entity type becoming another). Re-read from the container after each call.
- Verify action- and transaction-related claims through `make()` too.

## Checklist

Skip sections that don't apply. Each item is a real bug class found in a prior audit.

### Constants, tables, and formulas
- [ ] Every value in a doc table matches the engine's constants. Check every row, not a sample.
- [ ] Derived columns (rates, per-unit costs) use one consistent formula, and the doc says which. Mixed formulas across rows is the tell.
- [ ] "Max" values are reachable. A cap needing an optional booster needs that qualifier; a cap unreachable in a variant lacking the booster is wrong.
- [ ] Column headers mean what the values show. If they disagree, ask the user whether to change the number or the header.
- [ ] Formulas match the code symbol for symbol — exponents, log bases, clamps, order of operations. Recompute any worked example.

### State initialization and lifecycle
- [ ] Initial value of every field in each constructor. A counter starting at 1 instead of 0 silently removes a grace period the prose implies.
- [ ] "Ongoing" / "indefinite" / "repeating" claims: find the cap (a count-vs-max guard, a lifespan field being armed). If one exists, the doc must say so.
- [ ] Exact step an entity or episode ends, and what stays usable on the way out.
- [ ] Ordering inside periodic hooks: which mutations run before which checks. That ordering is often what makes a rule surprising.
- [ ] Cooldowns: when the counter ticks relative to the action, whether a failed attempt consumes it, turns-between vs. turns-until.
- [ ] Rates that ramp or decay: recompute at start, middle, and end. Watch for rounding or a clamp that makes the stated endpoint arrive early.
- [ ] Entities that change type in place (transforming, upgrading, being captured) — what carries over, what resets.

### Actions and preconditions
- [ ] Walk each documented action against its handler. Every early `return`/`continue` is an undocumented precondition.
- [ ] Partial success: an action that works in one state but no-ops in a neighbouring one needs the qualifier.
- [ ] No-op semantics: does a rejected action still consume the resource, turn, or cooldown?
- [ ] Legality rules for movement, placement, targeting — and whether spawn/setup logic obeys the same rules the player does.
- [ ] Malformed input: wrong arity, wrong type, out-of-range. Silently dropping one move from a submitted list is a rule players need.
- [ ] Per-turn action limits and what happens to the excess.

### Turn pipeline and interaction resolution
- [ ] If the doc lists a numbered turn order, walk the engine top to bottom and confirm each stage's position. Late-inserted stages (cleanup, spawning, expiry) get omitted.
- [ ] Where an entity is mutated relative to where it's read. Spawned after the action phase → unusable until next turn; removed before it → never usable.
- [ ] Contested outcomes (combat, collision, simultaneous claims): work ties, three-way contests, and zero-survivor cases by hand. Docs describe only the two-party win case.
- [ ] Continuous vs. endpoint-sampled detection (swept paths vs. position checks). Changes which interactions are possible; state it outright.
- [ ] Precedence when several removal conditions fire on one entity in the same step.

### Procedural generation and randomness
- [ ] Documented ranges match the generator's bounds, including any retry cap that can silently under-deliver.
- [ ] Guarantees ("at least one of X", "always symmetric") are enforced by code, not merely likely. Find the loop that makes them true, or drop "guaranteed".
- [ ] Distribution claims ("skewed low", "weighted toward") imply specific sampling — verify rather than restate.
- [ ] What the seed controls, and whether it's scrubbed from the observation before agents see it.

### Hidden state and partial observability
- [ ] Fields agents are told to use for prediction contain what's claimed, and stay valid as the episode progresses.
- [ ] Anything the doc says is derivable, derive it: recompute from the observation alone and compare to engine state.
- [ ] Information agents should *not* have is genuinely absent — check what the **agent receives**, not the shared state object. Engines routinely keep global state on `state[0].observation` and strip it per player. Run an agent that dumps its own observation keys.
- [ ] If visibility is asymmetric, confirm the players' observations actually differ and each sees only its share.
- [ ] Per-category persistence. In a "visible now / remembered later" table every row is a separate claim, and "remembered" may mean last-known-value rather than live.
- [ ] Sentinels for unknown data (`-1`, `None`, absent key) are documented and mean *unknown*, not a legitimate zero.

### Termination and scoring
- [ ] Every termination condition, not just the step limit. Elimination, stalemate, and early-exit branches get missed.
- [ ] The step the episode actually ends on. Off-by-one against `episodeSteps` is common; run to completion.
- [ ] The reward the interpreter assigns, including ties and all-players-eliminated. Does the doc's "winner" language match a win/loss constant or a raw score?
- [ ] Rewards that change meaning mid-episode (running score during play, win/loss constant at the end) need both forms documented.
- [ ] Multi-stage tiebreakers: each stage exists, in that order, with the stated final fallback.
- [ ] What counts toward the final score and what's excluded — value in intermediate state often doesn't count.

### Observation and action format
- [ ] Every documented field exists, with that type and meaning.
- [ ] Field comments describing *when* something is set: verify the trigger. "Set after X" is often wrong when the engine sets it unconditionally.
- [ ] Boolean vs. counter. A boolean means it doesn't accumulate — say so if a player might expect stockpiling.
- [ ] Anything the docs imply lives in the state structure but doesn't. Agents will search and find nothing; document the coordinates or accessor instead.
- [ ] Documented action strings and argument shapes match what the handler parses.

### Configuration
- [ ] Doc defaults match the `.json` specification.
- [ ] **Every documented knob is actually read.** Diff the doc table against the `.json` keys, then grep each in the engine. A knob ignored in favour of a module constant is worse than undocumented — players tune it and see no effect.
- [ ] Claims holding only at default config are labelled as such.
- [ ] Derived figures quoted in prose recompute from the stated defaults.

### Resource and economy systems (if present)
- [ ] Which items/actions are permitted on each side of a transaction. Check membership tests (`x in SOME_LIST`) separately from explicit whitelists — an unfiltered membership test permits more than the author may have intended.
- [ ] Calibration constants: confirm the stated derivation reproduces them. If not, they're stale from a rebalance or the explanation is wrong.
- [ ] Any horizon or reference quantity differing from the episode length needs a reason in the doc.

### Cross-file consistency
- [ ] README and AGENTS.md agree. When they conflict, both may still be wrong — check the engine.
- [ ] Beginner/advanced or base/arena variants have their own engines. A fix in one usually applies to the sibling, but *verify against that sibling's engine* — constants often differ.
- [ ] Module-level comments and docstrings in the engine are docs too, and drift.

## Style

Docs are declarative rulebooks, not strategy guides.

- State the rule; don't advise. Say what the mechanic does, not what the player should do about it.
- No "you should", "plan on", "budget for", "make sure", "worth noting".
- No bold lead-ins to make a rule feel urgent.
- Leave pre-existing prose alone unless it's factually wrong. Rewriting a doc's voice is out of scope; if a paragraph is hard to read, a paragraph break beats an edit.

## Reporting

Group findings as **confirmed** / **overstated** / **rejected**. For each: file:line, what the engine does, how you verified it. Separate doc bugs from engine bugs — engine bugs need the user's decision, not your fix.

Surface explicitly:
- **Bugs the reporter missed.** A user-filed list is a starting point, not a scope.
- **Judgment calls**, where the engine's behavior looks unintentional. Documenting it is the default, but say so plainly and give the alternative fix.

Before finishing: re-verify every number you wrote with one audit script asserting doc value == engine value, and run the env's test suite.
