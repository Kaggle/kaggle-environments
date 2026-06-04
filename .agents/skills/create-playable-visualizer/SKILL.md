# Create a Playable Visualizer

Build a **standalone interactive (human vs AI) version** of a Kaggle game environment that runs entirely in the browser. The engine and bundled AI agents are re-implemented in TypeScript so there is no Python round-trip — a player picks moves in a React UI, the game state advances in a web worker, and AI opponents play against them.

This is a separate sibling to the read-only replay visualizer. Use this skill **after** the regular `create-visualizer` skill — the playable build reuses the default visualizer's renderer for the board view.

**Related skills:**
- `create-visualizer` — build the read-only replay visualizer first (the playable view depends on its renderer)
- `create-environment` — build the Python backend for a custom game

**Canonical reference implementation:** `kaggle_environments/envs/kaggriculture/visualizer/playable/`

Sibling reference: `kaggle_environments/envs/orbit_wars/visualizer/playable/` — older but slightly simpler engine.

## Architecture at a glance

```
playable/
├── package.json          # React + Vite + vitest (NO @kaggle-environments/core dep)
├── vite.config.ts        # extends web/vite.config.base, adds @vitejs/plugin-react + worker.format='es'
├── tsconfig.json         # extends web/tsconfig.base.json, adds "WebWorker" lib
├── index.html            # <div id="root">, mounts main.tsx
└── src/
    ├── main.tsx          # ReactDOM.createRoot → <App />
    ├── App.tsx           # switches SetupScreen ↔ GameScreen
    ├── style.css         # entire visual system
    ├── engine/           # TS port of the Python interpreter
    │   ├── constants.ts  # often a clone of information in the game json
    │   ├── types.ts      # GameState, PlayerAction, Config, etc
    │   ├── rng.ts        # MT19937 PRNG (Python-compatible) if seed-equivalence matters
    │   ├── state.ts      # initGameState, pickSeed, resolveConfig
    │   ├── interpreter.ts# step() — per-turn pipeline
    │   ├── market.ts     # auxiliary market helpers (optional)
    │   ├── index.ts      # barrel re-export
    │   └── __tests__/    # vitest fixtures comparing TS rollouts to Python replays
    ├── ai/               # TS port of bundled agents
    │   ├── types.ts      # Observation, AgentFn
    │   ├── random.ts
    │   ├── starter.ts
    │   ├── index.ts      # AGENTS registry + DEFAULT_AGENT_ID
    │   └── __tests__/
    ├── worker/
    │   ├── protocol.ts   # Req/Res types + SlotConfig
    │   ├── workerClient.ts # promise-based wrapper
    │   └── gameWorker.ts # owns GameState, evaluates AI slots, runs step()
    └── ui/
        ├── useGameWorker.ts # hook: state, busy, error, stepGame, reset
        ├── SetupScreen.tsx  # pick humans/AIs, seed, episode length
        ├── HUD.tsx          # busy/done/error tags + Reset/New Game
        ├── GameScreen.tsx   # composes HUD + GameView + ActionPanel + modal
        ├── ActionPanel.tsx  # build PlayerAction for the human slot
        ├── GameView.tsx     # React wrapper around default's imperative renderer
        ├── buildView.ts     # GameState → default's ViewModel + rolling price history
        └── GameOverModal.tsx
```

**Key data flow:**
```
User picks setup → SetupScreen → SetupResult { config, numAgents, slots }
                                       │
                                       ▼
                              useGameWorker(setup) spawns Worker
                                       │
                                       ├──── INIT → worker builds initial GameState
                                       │
User picks action → ActionPanel → onSubmit(PlayerAction)
                                       │
                                       ▼
                              stepGame({ humanId: action })
                                       │
                                       ├──── STEP → worker calls AI agents for non-human slots,
                                       │             merges with human action, runs step()
                                       │             returns new GameState
                                       │
                                       ▼
                              GameView re-renders default visualizer
```

The playable visualizer is **not** part of the dev-with-replay flow. It is registered into the pnpm workspace via the existing glob `kaggle_environments/envs/*/visualizer/*` so `pnpm dev`/`pnpm build` pick it up like any other visualizer, but it ships its own UI rather than using `createReplayVisualizer`.

## Stages

Build in 8 stages. Each stage ends in something runnable and reviewable. The order matters — every later stage depends on earlier work.

### Stage 1 — Scaffold

Create `visualizer/playable/` next to the existing `visualizer/default/`. Auto-registered via the existing `pnpm-workspace.yaml` glob.

`package.json`:
```json
{
  "name": "@kaggle-environments/<name>-playable-visualizer",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.0.0",
    "typescript": "^5.0.0",
    "vite": "^5.0.0",
    "vitest": "^2.1.4"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  }
}
```

No `@kaggle-environments/core` dependency — the playable view is self-contained.

`vite.config.ts`:
```typescript
import { defineConfig, mergeConfig } from 'vite';
import react from '@vitejs/plugin-react';
import baseConfig from '../../../../../web/vite.config.base';

export default mergeConfig(
  baseConfig,
  defineConfig({
    plugins: [react()],
    worker: { format: 'es' },
  })
);
```

`tsconfig.json`:
```json
{
  "extends": "../../../../../web/tsconfig.base.json",
  "compilerOptions": {
    "lib": ["ESNext", "DOM", "DOM.Iterable", "WebWorker"],
    "noUnusedLocals": false,
    "noUnusedParameters": false
  },
  "include": ["src"]
}
```

`index.html` mounts to `#root`, `src/main.tsx` calls `ReactDOM.createRoot(...).render(<App />)`.

**Done when:** `pnpm dev` boots, `pnpm build` is green, root renders an empty placeholder.

### Stage 2 — Engine port: data model

Port constants and types from the Python interpreter:
- All gameplay tables (costs, thresholds, board dimensions, episode length, anything the interpreter reads as data).
- TypeScript types for every domain object the interpreter mutates: at minimum `GameState`, `PlayerAction`, `Config`, plus whatever per-player view (`Private`/`Observation`) the Python framework hands to agents.
- `initGameState(numAgents, config, seed)` returning a typed initial `GameState`.

**Pull config defaults from `<env>.json`, don't duplicate them.** The env spec at `kaggle_environments/envs/<name>/<name>.json` is the source of truth for every `configuration.*.default`. Importing it into `constants.ts` and reading defaults dynamically makes "Python and TS slowly drifted" impossible by construction. With `resolveJsonModule: true` (already set in `web/tsconfig.base.json`) the import just works:

```typescript
import spec from '../../../../<name>.json';

function specDefault<T>(key: string): T {
  const entry = (spec.configuration as Record<string, unknown>)[key];
  if (entry !== null && typeof entry === 'object' && 'default' in (entry as object)) {
    return (entry as { default: T }).default;
  }
  return entry as T; // top-level flat values like `episodeSteps`
}

export const DEFAULT_CONFIG = {
  episodeSteps: specDefault<number>('episodeSteps'),
  startingMoney: specDefault<number>('startingMoney'),
  // ... every field that has a `.default` in the spec
} as const;
```

Hard-coded constants pass unit tests trivially (the same wrong number flows through both the engine and its tests) — the drift only surfaces against an external reference. Pulling from the spec is the cheapest way to never write that bug.

If exact seed parity matters, port the PRNG (Python's `random` ≈ MT19937) in `rng.ts`. If randomness only needs to be reproducible per-browser-session, a simple seeded LCG is fine. Note that a bare MT19937 seeded with the integer is **not** bit-identical to CPython's `random.Random(seed)` — CPython runs the seed through `init_by_array`. Without that, every `rng.choice` / `rng.random` call diverges, which matters in Stage 3.

**Done when:** a typed initial `GameState` builds for any seed and matches a Python fixture for known constants.

### Stage 3 — Engine port: turn processing

Port `step()` **phase for phase** from the Python interpreter. The order of side effects matters — write tests against Python-generated replays before you trust your port.

Walk the Python `interpreter` (or equivalent) top to bottom and reproduce each phase as a separate function. Whatever the source does — resolve per-player actions, apply environment dynamics, advance any per-tick counters, check terminal — do in the same order, with the same fall-through cases.

**When is seed-parity testing worth the cost?** The bar for the playable is *behavioral* — a human shouldn't notice obvious rule differences vs. the Python version. That bar is usually met by orbit-wars-style behavioral tests ("launching a fleet deducts ships and creates a Fleet"; "production accrues for owned planets"). A Python-replay parity test is a heavier tool — treat it as a **build-time scaffold for stress-testing the port**, not shipped infrastructure. It earns its keep when:

- The interpreter has many tightly-coupled phases (e.g. 4+ per step) where ordering or fall-through bugs are likely and hard to spot by reading the diff.
- State has enough fields that hand-written behavioral tests would leave gaps the playtester won't notice until late game (silent drift in money, inventory, scoring).
- You want a single fixture that catches the "I missed a Python phase entirely" failure mode end-to-end.

Default to **building the parity test, using it during the port, then deleting it before merge** — same lifecycle as the AI-vs-AI Step button in Stage 5. Keep it past merge only when (a) you expect ongoing iteration on the Python interpreter that needs a regression net, or (b) a behavioral suite genuinely can't cover the deterministic pipeline (rare). Don't keep it just because you wrote it. Orbit wars ships without one; kaggriculture used parity heavily during the build but doesn't need it long-term.

Skip parity entirely (go straight to behavioral tests) when the engine is small, the phases are obvious, or the state is mostly stochastic (parity would filter out most of the interesting fields anyway — see RNG note below).

If you do build a parity fixture, generate references with:
```bash
uv run python -c "
from kaggle_environments import make; import json
env = make('<name>', debug=True, configuration={'episodeSteps': 100}, info={'seed': 42})
env.run(['random', 'random'])
json.dump(env.toJSON(), open('parity_seed42.json', 'w'))
"
```
(Some envs route the seed through `info` rather than `configuration` — check the interpreter's `_initialize` for the convention.)

Then the parity test pattern is **replay Python's recorded actions through TS** and diff per-step state — not run a TS rollout independently. Random-agent rollouts can't be reproduced from a seed alone (the agents themselves use Python's RNG), but replaying the recorded actions makes the test deterministic regardless of agent choices:

```typescript
let state = initGameState(numAgents, cfg, replay._seed);
diffStep('init', state, replay.steps[0][0].observation);
for (let next = 1; next < replay.steps.length; next++) {
  const actions = replay.steps[next].map((s) => s.action);
  state = step(state, actions, cfg);
  diffStep(`step ${next}`, state, replay.steps[next][0].observation);
}
```

**Action timing gotcha.** In `kaggle_environments/core.py`, `steps[i].action` is the action that was applied to **produce** `steps[i].observation` — not the action submitted *from* step `i`. So the loop advances using `steps[next].action`, not `steps[i].action`. Easy off-by-one; the test will quietly pass the first couple of turns (when most random actions are PASS/movement) before drifting.

**RNG-driven divergence.** If `rng.ts` isn't bit-identical to CPython, every stochastic phase (weed spawns, random shop unlocks, anything calling `rng.choice` or `rng.random`) will diverge — and that divergence cascades (a wrong weed blocks a plant, a wrong shop pick drifts the market on every subsequent tick). Two options:

1. *Filter stochastic outputs from the diff* — normalize WEED tiles to `null`, drop `town.unlocked_shops`, drop `market.inventory`/`market.prices` (if shops affect them). You lose strict parity on those fields but still validate the entire deterministic pipeline (movement, all unit actions, decay, daily refresh, EoD inventory drop, hire costs, money flows, private state).

2. *Port CPython's `Random` seeding faithfully* — MT19937 with `init_by_array` on the integer seed's 32-bit chunks. ~30–50 LOC, makes the full diff strict. Spike with `expect(new PyRandom(42).random()).toBeCloseTo(0.6394267984578837)` as your acceptance test.

Option 1 is the default; pick option 2 only if you'll need the parity test to cover late-game state — and remember that need only matters if you're keeping the parity test past merge in the first place (see the worth-the-cost note above). A browser-stable PRNG with filtered diff is usually enough to validate the deterministic pipeline during the build.

Diff the TS output `GameState` against the corresponding `observation` field in each replay step. Drift here will silently break AI agents in Stage 4.

**This is the heaviest stage.** Allocate the most time here; expect to discover Python-side subtleties (order of dict iteration, integer truncation, off-by-one boundaries between turns).

**Done when:** the engine produces behaviorally correct output — either passing a parity fixture for the reference seeds (modulo any filtered RNG fields), or covered by behavioral tests in the orbit-wars style. If parity was a scaffold, decide now whether to keep it; if not, delete before merge.

### Stage 4 — AI agents

Port only the bundled agents you intend to expose in the picker. The `pass` / no-op agent is implicit — missing slots default to whatever your interpreter treats as "do nothing".

Each agent is `(observation: Observation) => PlayerAction`. The `Observation` includes the public state plus that player's private view (matches what the Python framework passes).

Register them in `ai/index.ts`:
```typescript
export const AGENTS: Record<string, { label: string; fn: AgentFn }> = {
  random: { label: 'Random', fn: randomAgent },
  // ... other ported agents
};
export const DEFAULT_AGENT_ID: AgentId = 'random';
```

**Done when:** TS-vs-TS rollouts complete a full episode without errors and produce sensible scores.

### Stage 5 — Worker + state hook

The engine runs in a dedicated web worker so a slow AI agent never freezes the UI.

`worker/protocol.ts`:
```typescript
export type SlotConfig = { kind: 'human' } | { kind: 'ai'; agentId: string };
export type HumanActions = Record<number, PlayerAction>;

export type Req =
  | { type: 'INIT'; reqId: string; config: Config; numAgents: number; slots: SlotConfig[] }
  | { type: 'STEP'; reqId: string; humanActions: HumanActions }
  | { type: 'RESET'; reqId: string }
  | { type: 'GET_STATE'; reqId: string };

export type Res =
  | { type: 'STATE'; reqId: string; state: GameState }
  | { type: 'ERROR'; reqId: string; message: string };
```

`worker/gameWorker.ts` owns the authoritative `GameState`. On `STEP` it:
1. Builds a per-player `Observation`.
2. For each slot: if human, take action from `humanActions[pid]` (default to the no-op action); if AI, call `AGENTS[slot.agentId].fn(obs)` inside `try/catch`.
3. Runs `step(state, actions, config)` and returns the new state.

`worker/workerClient.ts` wraps it in a promise-based API:
```typescript
new Worker(new URL('./gameWorker.ts', import.meta.url), { type: 'module' });
```

`ui/useGameWorker.ts` is a React hook keyed on a `SetupResult` reference. Changing `setup` tears down the worker and re-inits. Returns `{ state, busy, error, stepGame, reset }`.

**Done when:** an AI-vs-AI game runs to completion off the main thread. (AI-vs-AI here is a debugging scaffold — see Stage 7 — not a shipped feature.)

### Stage 6 — Read-only game view

**Do not re-port the renderer.** Reuse the sibling `default/` visualizer's renderer via relative imports — this is the single biggest time-saver in the build.

`ui/buildView.ts` bridges `GameState` → default's `ViewModel`:
```typescript
import { type ViewModel /* + any other types default exports */ } from '../../../default/src/types';
```

If default builds any derived state from a window of replay steps (e.g. rolling histories, deltas vs previous step), the playable view has no replay to look at — you have to accumulate that same derived state live. The pattern is a small tracker held in a `useRef` that exposes `record(state)` / `reset()` and returns the same shape the renderer expects.

`ui/GameView.tsx` mounts an empty `<div>`, calls default's shell-build helpers once (keyed on board dimensions + player names), then calls default's render function on every state change inside `useEffect`. Imports come from `../../../default/src/<renderer>`.

This is a hard dependency on the default visualizer's internal API surface. If default refactors, the playable build breaks at compile time, which is what you want — fix in lockstep. Expect to export a few previously-internal helpers from default (the shell builder, the per-frame render function) as named exports the first time you do this; that's normal.

**Done when:** an AI-vs-AI rollout (still using the temporary Step button from Stage 5) is watchable inside the playable app with identical visuals to the replay visualizer.

### Stage 7 — Action input UI

`ActionPanel.tsx` composes a `PlayerAction` for the human slot. The general pattern, regardless of action shape:
- One control (dropdown, grid click, etc.) per discrete choice the player has to make.
- Op-conditional sub-controls: when the top-level op selection has parameters (target tile, quantity, sub-kind), reveal those inputs only after the op is picked.
- Repeated sections (multiple units to command, a queue of orders, etc.) need an explicit add/remove UI and have to stay reconciled with whatever the live `GameState` says is currently available — use a `useEffect` keyed on the live count to grow/shrink the local draft array.
- On submit: bundle the drafts into a `PlayerAction`, call `onSubmit`, reset local drafts to defaults.

**Second-heaviest stage.** Take care with the action contract — invalid actions silently fall through to the default no-op, which is hard to debug. Mirror the Python interpreter's action shape exactly; if the source uses positional tuples like `['VERB', ARG1, ARG2]`, the TS types and the UI's serializer must produce that exact shape (right casing, right arity, right argument types).

**Tear down the AI-vs-AI scaffolding once a human can play.** The temporary "no human slot → single Step button" path from Stage 5 was a debugging crutch for verifying the engine + worker + renderer end-to-end before any input UI existed. Once the action panel works, an all-AI lineup adds nothing for the user — they can already watch any matchup via the read-only replay visualizer. Remove the AI-vs-AI branch in `ActionPanel`/`GameScreen`, drop the Step button, and require at least one human slot in `SetupScreen` validation. Leftover Step-button code rots quickly and confuses readers about what the playable build is for.

**Done when:** a human can play a full game against AI opponents, and no AI-vs-AI-only code paths remain.

### Stage 8 — Setup + HUD + game over

`SetupScreen.tsx`:
- One dropdown per slot: "Human" or each `AGENTS[id].label`.
- An episode-length picker (offer a few short options as well as the canonical length) — short games matter for testing the end-of-game flow without sitting through a full match.
- Seed text input (blank → random).
- Themed background and title that match the default visualizer's look (reuse its background + an icon sprite via relative import), primary "Start Game" button.

```typescript
const handleStart = () => {
  const seed = seedText.trim() === '' ? Math.floor(Math.random() * 0x7fffffff) : Number(seedText);
  const base = resolveConfig({ seed });
  const config = { ...base, episodeSteps: /* derived from the length picker */ };
  onStart({ config, numAgents: slots.length, slots });
};
```

`HUD.tsx` — minimal top bar: `busy…` / `game over` / `error: …` tags, `Reset` and `New Game` buttons. Do **not** duplicate any state (turn counter, scores, etc.) that the inner `GameView` already shows.

`GameOverModal.tsx` — winner detection handles single-winner, tie, and human-vs-AI cases. Style to match the setup screen (same background, same font, same decorative border). Buttons: `Replay (same setup)` calls `reset()`; `New Game` exits to `SetupScreen`.

`App.tsx` is a switcher:
```typescript
export function App() {
  const [setup, setSetup] = useState<SetupResult | null>(null);
  return setup === null
    ? <SetupScreen onStart={setSetup} />
    : <GameScreen setup={setup} onExit={() => setSetup(null)} />;
}
```

**Done when:** polished end-to-end UX from setup → match → restart.

### Stage 9 — Integration smoke tests

Add a tiny vitest + Testing Library suite that just proves the UI mounts and a game can be started. The point is to catch outright breakage (missing exports, worker URL typos, throw-on-mount regressions), not to test gameplay — engine correctness already has fixtures in Stage 3.

Add the deps and a jsdom environment:
```json
// package.json devDependencies
"@testing-library/react": "^16.0.0",
"@testing-library/user-event": "^14.5.0",
"jsdom": "^25.0.0"
```
```typescript
// vitest.config.ts (or test block in vite.config.ts)
test: { environment: 'jsdom', globals: true }
```

Two tests are enough:

```typescript
// src/ui/__tests__/App.test.tsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { App } from '../../App';

test('renders the setup screen', () => {
  render(<App />);
  expect(screen.getByRole('button', { name: /start game/i })).toBeInTheDocument();
});

test('starts a game when Start is clicked', async () => {
  const user = userEvent.setup();
  render(<App />);
  await user.click(screen.getByRole('button', { name: /start game/i }));
  // Either the HUD's Reset button or the action panel's Submit shows up once
  // the worker has init'd and React has rendered GameScreen.
  expect(await screen.findByRole('button', { name: /reset|submit/i })).toBeInTheDocument();
});
```

Workers need a stub in jsdom — the simplest path is to mock `WorkerClient` so the second test doesn't actually spawn a worker:

```typescript
vi.mock('../../worker/workerClient', () => ({
  WorkerClient: class {
    init = async () => ({ /* minimal valid GameState */ });
    step = async () => ({ /* ... */ });
    reset = async () => ({ /* ... */ });
    terminate = () => {};
  },
}));
```

Keep the fake `GameState` minimal — just enough fields for `GameScreen` and its children to render without throwing. If a real worker round-trip is important, leave that for the engine fixtures; this suite is pure UI smoke.

**Done when:** `pnpm test` runs both UI tests plus the engine/agent fixtures and exits green.

## Polish lessons learned

Worth applying up front rather than after a review pass.

1. **Reuse assets from `default/` via relative imports.** Don't duplicate the asset pipeline:
   ```typescript
   import iconUrl from '../../../default/src/assets/sprites/<icon>.png';
   import bgUrl from '../../../default/src/assets/sprites/<bg>.svg';
   ```
   The Vite bundler resolves them; no workspace dependency needed.

2. **No CSS `container-type`.** It breaks inline playback controls in this codebase. Use `@media` queries instead.

3. **Contrast in the action panel.** The side panel typically sits on a non-white themed background. Default browser text colors disappear — explicitly set `color`, brighten input borders, and use `font-weight: 600` for labels.

4. **Solid borders on form controls, decorative borders on cards.** Dashed `<select>` borders look broken; any sketched/dashed motif is reserved for decorative cards and modals.

5. **AI-vs-AI mode is a debugging scaffold, not a shipped feature.** During Stages 5–6, expose a temporary "no human slot → single Step button" path so you can drive the engine + worker + renderer end-to-end before the action UI exists; it catches action-shape mismatches and AI errors in the first handful of turns. **Delete it as part of Stage 7** once a human can play — users who want to watch AI vs AI already have the read-only replay visualizer, and the leftover branch rots fast.

6. **Keep `package.json` lean.** React + Vite + vitest, no `@kaggle-environments/core`, no MUI. The playable bundle should not pull in the replay framework's dependencies.

7. **`pnpm dev` from playable and from default collide on port 5173.** Both extend `web/vite.config.base.ts` which binds `host: 0.0.0.0, port: 5173`. If you run both concurrently, Vite auto-falls-back the second one to 5174 — and a bare `pnpm dev` from `visualizer/default/` serves a blank page (it expects a replay wired up via `dev-with-replay`). When testing the playable, confirm you're on the port Vite reported in *that* terminal's "Local:" line, not whatever your browser remembered from the last session.

## Checklist

- [ ] Directory at `kaggle_environments/envs/<name>/visualizer/playable/` alongside `default/`
- [ ] `package.json` declares React + Vite + vitest, no `@kaggle-environments/core`
- [ ] `vite.config.ts` sets `worker.format: 'es'` and registers `@vitejs/plugin-react`
- [ ] `tsconfig.json` includes `"WebWorker"` in `lib`
- [ ] `DEFAULT_CONFIG` pulls every field from `<name>.json` via `specDefault(...)` rather than hard-coded numbers
- [ ] Engine port is covered by either behavioral tests (orbit-wars style) or, for complex multi-phase interpreters, Python-replay parity fixtures used as a build-time scaffold (action-replay pattern, not independent rollout) — kept past merge only if ongoing Python iteration is expected
- [ ] Worker uses the INIT/STEP/RESET/GET_STATE protocol with `reqId` correlation
- [ ] `useGameWorker` re-spawns the worker when the `setup` reference changes
- [ ] GameView imports the renderer from `../../../default/src/...` (no duplication)
- [ ] Any rolling derived state default expects from replays is accumulated live across steps
- [ ] ActionPanel reconciles repeated sections to the live `GameState` via `useEffect` and resets after submit
- [ ] SetupScreen exposes an episode-length picker so end-game can be reached in a short test
- [ ] HUD is minimal (no duplicate state the inner GameView already shows)
- [ ] Game-over modal handles single winner, tie, and human-vs-AI cases
- [ ] Stage 5/6 AI-vs-AI debug path (Step button, no-human-slot branch) removed in Stage 7; SetupScreen requires ≥ 1 human
- [ ] Setup + game-over share the same theme as the inner GameView (bg, sprite, font, border)
- [ ] Solid borders on `<select>`/`<input>`; decorative dashed/sketched borders reserved for cards
- [ ] No CSS `container-type` anywhere
- [ ] UI smoke tests: setup screen renders + clicking Start advances to game screen
- [ ] `pnpm test` passes (vitest fixtures + agent rollouts + UI smoke tests)
- [ ] `pnpm build` produces output in `dist/`
- [ ] `pnpm format` passes

## Troubleshooting

**TS rollout diverges from the Python replay after N turns.** Almost always a phase-ordering bug in `step()`. Bisect with snapshot diffs at each phase boundary; the first divergent field tells you which phase is wrong. If divergence starts at step **0** (the initial state itself), it's not a phase-ordering bug — it's a wrong constant in `DEFAULT_CONFIG`. Diff `DEFAULT_CONFIG` against `<name>.json`'s `configuration.*.default` values; see Stage 2 for the import-from-spec pattern that makes this impossible.

**Parity test passes the first few turns then drifts.** Likely the action-timing off-by-one: `steps[i].action` is the action that produced `steps[i].observation`, not the action submitted from it. The early turns are mostly PASS so the diff lines up by coincidence. See Stage 3.

**Worker errors don't surface.** The worker swallows agent exceptions and substitutes the no-op action (intentionally — a buggy AI shouldn't kill the session). Log inside the `try/catch` in `gameWorker.ts` and watch the dev console.

**GameView shows stale data.** `useEffect` in `GameView.tsx` must depend on `state` (the GameState reference). The worker returns a fresh object each STEP, so reference equality is the right signal.

**Default visualizer's renderer breaks when imported.** Default sometimes hides imperative helpers behind a `createReplayVisualizer` façade. Promote the shell builder and per-frame render function to named exports in `default/src/<renderer>.ts` if they aren't already exposed — the playable build is the second consumer and that's where the contract crystallizes.

**Repeated-section count drifts.** If the count of repeated entities (units, queued orders, anything the human can act on per-turn) changes mid-game, the local draft array in ActionPanel must be reconciled, not just appended. Use a `useEffect` keyed on the live count that slices to length and pads with defaults.

**Action submitted but nothing happens.** Almost always an action-shape mismatch. Print the `PlayerAction` in the worker right before `step()` and compare to the Python contract. Positional tuples are case- and arity-sensitive — a misspelled verb or wrong-typed argument silently falls through to the default no-op.
