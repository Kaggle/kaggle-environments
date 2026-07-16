# Create / Update a Visualizer

Build a web visualizer for any Kaggle game environment -- regular or OpenSpiel. Each visualizer is a Vite + TypeScript project within the pnpm workspace.

**Related skills:**
- `create-environment` -- if you need to build the Python backend for a custom game first
- `onboard-open-spiel-game` -- if you need to register or configure an OpenSpiel game first
- `create-playable-visualizer` -- after this, optionally build an interactive human-vs-AI version that re-uses this renderer

## Step 1: Determine your variant

| Variant | Directory | Relative path to `web/` | `gameName` |
|---------|-----------|------------------------|------------|
| Regular env | `kaggle_environments/envs/<name>/visualizer/default/` | `../../../../../` (5 levels) | `"<name>"` |
| OpenSpiel env | `kaggle_environments/envs/open_spiel_env/games/<name>/visualizer/default/` | `../../../../../../../` (7 levels) | `"open_spiel_<name>"` |

Both variants use the same boilerplate, shared workspace dependency, and renderer interface. The only differences are the directory depth (which affects relative paths to base configs) and the replay data shape.

**Standalone (regular env) extra step:** add `'<name>'` to `KNOWN_STANDALONE_GAME_DIRS` in `web/scripts/validate-visualizer-conventions.js`. The validator (run on Kaggle's deployment side and via `pnpm validate-conventions` locally) treats any visualizer directly under `kaggle_environments/envs/<game>/` as suspect by default to catch OpenSpiel games accidentally placed at the standalone path; the allowlist is what tells it "yes, this one really is standalone". OpenSpiel visualizers live under `kaggle_environments/envs/open_spiel_env/games/<game>/visualizer/` and don't need an entry.

## Step 2: Create the project structure

Create the visualizer directory with these files:

```
visualizer/default/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── index.html
├── replays/test-replay.json    (for dev -- see "Generate a test replay" below)
└── src/
    ├── main.ts
    ├── renderer.ts
    └── style.css
```

This directory is automatically part of the pnpm workspace (via root `pnpm-workspace.yaml` pattern `kaggle_environments/envs/*/visualizer/*`).

For OpenSpiel games, also create an empty `games/<name>/__init__.py` if one doesn't exist.

### `package.json`

```json
{
  "name": "@kaggle-environments/<name>-visualizer",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "dev-with-replay": "cross-env VITE_REPLAY_FILE=./replays/test-replay.json vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "devDependencies": {
    "cross-env": "^10.1.0",
    "typescript": "^5.0.0",
    "vite": "^5.0.0"
  },
  "dependencies": {
    "@kaggle-environments/core": "workspace:*"
  }
}
```

Add any game-specific dependencies (e.g., `three` for 3D, `pixi.js` for 2D sprites). The `@kaggle-environments/core` workspace dependency provides React, MUI, emotion, and all shared utilities.

### `vite.config.ts`

```typescript
import { defineConfig, mergeConfig } from "vite";
// Adjust path depth: 5 levels for regular envs, 7 for OpenSpiel
import baseConfig from "../../../../../web/vite.config.base";

export default mergeConfig(
  baseConfig,
  defineConfig({
    publicDir: "replays",
  })
);
```

The base config (at `web/vite.config.base.ts`) provides: tsconfigPaths, TypeScript checker, cssInjectedByJs plugin, dev server on port 5173, relative base path for builds, and CORS.

### `tsconfig.json`

```json
{
  "extends": "../../../../../web/tsconfig.base.json",
  "compilerOptions": {
    "allowJs": true
  },
  "include": ["src"]
}
```

Adjust the `extends` path to match your variant's depth.

### `index.html`

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title><Name> Visualizer</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Mynerve&display=swap" rel="stylesheet" />
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

The `<div id="app">` is required -- `createReplayVisualizer` mounts to it.

### `src/style.css`

See [visualizer-style-guide.md](visualizer-style-guide.md) for the standard CSS and the full visual design system.

### `src/main.ts`

```typescript
import { createReplayVisualizer, ReplayAdapter } from "@kaggle-environments/core";
import { renderer } from "./renderer";
import { gameTransformer } from "./transformers/gameTransformer";
import "./style.css";

const app = document.getElementById("app");
if (!app) {
  throw new Error("Could not find app element");
}

if (import.meta.env?.DEV && import.meta.hot) {
  import.meta.hot.accept();
}

createReplayVisualizer(
  app,
  new ReplayAdapter({
    gameName: "<name>",           // must match the registered env name
    renderer: renderer as any,
    ui: "side-panel",             // "side-panel" (with reasoning logs) or "inline"
    // The side-panel's ReasoningLogs reads step.players[i].thoughts /
    // actionDisplayText. Without a transformer that shape is missing, so
    // the entire sidebar sits empty. See Step 5.
    transformer: (replay) => ({
      ...replay,
      steps: gameTransformer(replay),
    }),
  })
);
```

### ReplayAdapter options

| Option | Type | Description |
|--------|------|-------------|
| `gameName` | string | Environment name (must match spec `name`) |
| `renderer` | RendererFn | DOM/canvas renderer function (see below) |
| `GameRenderer` | React.ComponentType | Alternative: React component instead of DOM renderer |
| `transformer` | function | Optional: preprocess replay data before rendering |
| `ui` | string/component | `"inline"` (controls below game), `"side-panel"` (controls in sidebar with reasoning logs), `"none"` (no playback UI) |
| `layout` | string | `"side-by-side"` or `"stacked"` |
| `initialSpeed` | number | Playback speed multiplier |

### Generate a test replay

For regular environments:

```bash
uv run python -c "
from kaggle_environments import make
import json
env = make('<name>', debug=True)
env.run(['agent1', 'agent2'])
with open('test-replay.json', 'w') as f:
    json.dump(env.toJSON(), f, indent=2)
print(f'Generated replay with {len(env.toJSON()[\"steps\"])} steps')
"
```

For OpenSpiel environments (the `"random"` agent needs `includeLegalActions`):

```bash
uv run python -c "
from kaggle_environments import make
import json
env = make('open_spiel_<name>', debug=True, configuration={'includeLegalActions': True})
env.run(['random', 'random'])
replay = env.toJSON()
with open('test-replay.json', 'w') as f:
    json.dump(replay, f, indent=2)
print(f'Generated replay with {len(replay[\"steps\"])} steps')
print(f'Statuses: {replay[\"statuses\"]}')
"
```

Verify the replay has a reasonable number of steps (not 2-3, which indicates the agent failed).

**If your game can end via forfeit (illegal-move retries exhausted, timeout, or agent crash), also generate a forfeit replay.** Forfeit-ended games are visually different from natural terminals (no `observation.isTerminal`, no on-board winner) and are a common failure mode when running against LLM agents. If you don't dev against a forfeit replay, you won't notice the renderer freezing on "Turn: X" (see Step 5). Craft one by mutating the natural replay: pick a step, rewrite the acting player's action into a forfeit shape, and set terminal rewards.

```bash
uv run python -c "
import json
with open('replays/test-replay.json') as f: r = json.load(f)
# Truncate to a mid-game step and inject a forfeit for the acting player.
r['steps'] = r['steps'][:8]
step = r['steps'][-1]
acting = next(i for i, s in enumerate(step)
              if isinstance(s.get('action'), dict)
              and s['action'].get('submission') not in (None, -1))
step[acting]['action'] = {
    'submission': -1,
    'actionString': '<last illegal attempt>',
    'thoughts': 'I could not find a legal move.',
    'status': 'Failed to parse a legal move after 5 attempts; forfeiting.',
}
r['rewards'] = [1.0, -1.0] if acting == 1 else [-1.0, 1.0]
r['statuses'] = ['DONE', 'DONE']
step[0]['reward'], step[1]['reward'] = r['rewards']
step[0]['status'] = step[1]['status'] = 'DONE'
with open('replays/test-forfeit-replay.json', 'w') as f: json.dump(r, f)
print('wrote test-forfeit-replay.json with', len(r['steps']), 'steps')
"
```

Then run `VITE_REPLAY_FILE=./replays/test-forfeit-replay.json pnpm dev` and verify the final step shows a clear forfeit line (e.g. "Player 2 submitted an illegal move. Player 1 wins by default.") -- not a stale "Turn:" line.

## Step 3: Understand the replay data shape

The renderer function receives a `RendererOptions` object. The shape of `replay.steps` differs between regular and OpenSpiel environments.

### Regular environments

Steps are transformed through the core adapter into `BaseGameStep` objects:

```typescript
interface BaseGameStep {
  step: number;
  players: BaseGamePlayer[];
}

interface BaseGamePlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText?: string;
  thoughts?: string;
}
```

Access via `replay.steps[step].players[i]`. For raw env data, you can write a custom transformer (see "Optional: Add a transformer" below).

### OpenSpiel environments

Steps are raw arrays from the unified interpreter:

```typescript
// Each step is an array of player observations:
replay.steps[stepIndex][playerIndex].observation.observationString  // game state (JSON if proxy exists)
replay.steps[stepIndex][playerIndex].observation.currentPlayer      // whose turn it is
replay.steps[stepIndex][playerIndex].observation.isTerminal         // game over flag
replay.steps[stepIndex][playerIndex].action.submission              // action taken (-1 = not acting)
replay.steps[stepIndex][playerIndex].reward                        // cumulative reward
replay.steps[stepIndex][playerIndex].status                        // "ACTIVE" or "DONE"
```

#### Games with a proxy (default)

If the game has a proxy (see `onboard-open-spiel-game` skill -- this is the default), the `observationString` is **JSON**. The renderer just parses it:

```typescript
function getObservation(step: any, playerIdx: number): any | null {
  const raw = step?.[playerIdx]?.observation?.observationString;
  if (!raw) return null;
  try { return JSON.parse(raw); } catch { return null; }
}

// Usage in renderer:
const obs = getObservation(currentStep, 0);
// obs.board, obs.current_player, obs.is_terminal, obs.winner, obs.scores, obs.last_action, etc.
```

The proxy's `state_dict()` method determines what fields are available. See `onboard-open-spiel-game` for the standard fields: `board`, `current_player`, `is_terminal`, `winner`, `scores`, `last_action`, `phase`.

For **perfect information** games, both players get the same observation. For **imperfect information** games, each player gets a different JSON object containing only their private view -- parse both and render them (e.g., side-by-side boards).

#### Games without a proxy (raw text observations)

Some games may not have a proxy (e.g., games added to `GAMES_LIST` only). In this case, `observationString` is the raw text from OpenSpiel's `ObservationString()` or `InformationStateString()`. You'll need to parse it manually:

```typescript
function getObservationString(step: any, playerIdx: number = 0): string {
  return step?.[playerIdx]?.observation?.observationString ?? '';
}
```

Study the game's C++ source at `open_spiel/games/<game_name>/` (`.h`/`.cc` files) to understand the format of `ObservationString` and `ToString`.

#### Common OpenSpiel step helpers

These helpers work regardless of whether the game has a proxy:

```typescript
function isTerminal(step: any): boolean {
  if (!step || !Array.isArray(step)) return false;
  return step.some((p: any) => p?.status === 'DONE' || p?.observation?.isTerminal);
}

function getCurrentPlayer(step: any): number {
  if (!step || !Array.isArray(step)) return 0;
  for (const player of step) {
    const cp = player?.observation?.currentPlayer;
    if (cp !== undefined && cp >= 0) return cp;
  }
  return 0;
}

function getRewards(step: any): [number, number] {
  if (!step || !Array.isArray(step)) return [0, 0];
  return [step[0]?.reward ?? 0, step[1]?.reward ?? 0];
}
```

## Step 4: Write the renderer (`src/renderer.ts`)

The renderer function is called on every step change. It receives a `RendererOptions` object and should draw into the provided `parent` element.

### RendererOptions fields

| Field | Type | Description |
|-------|------|-------------|
| `parent` | HTMLElement | Container element to render into (persists across calls) |
| `replay` | ReplayData | Full replay: `steps`, `configuration`, `name`, `version` |
| `step` | number | Current step index (0-based) |
| `setStep` | (n: number) => void | Jump to a specific step |
| `setPlaying` | (b: boolean) => void | Start/stop playback |
| `registerPlaybackHandlers` | function | Register custom play/pause/step handlers |
| `agents` | any[] | Agent metadata |

### Visual design requirements

Every visualizer MUST clearly communicate these four things:

1. **Current actor (whose turn it is):** Show player names in the header, highlight the active player's card with `#bdeeff` background and `scale: 1.1`.

2. **Move taken (what just happened):** Compare current and previous step states to detect what changed. Highlight the move visually (glowing ring, gold overlay, dashed outline, etc.).

3. **Move implications (what the move caused):** Show deltas/diffs when state values change (`+N` / `-N` badges). Mark captured/removed pieces distinctly. Highlight score changes.

4. **Current score / game progress:** Show scores, piece counts, progress indicators. At game over, display the final result prominently, and clearly distinguish natural terminal / draw / **forfeit (with reason)**. See Step 5 for how forfeits are detected -- a game that ended because a player exhausted their illegal-move retries, timed out, or crashed will *not* have `observation.isTerminal === true`, and a renderer that only checks that flag will freeze on the last mid-game frame with a stale "Turn: X" line.

### Renderer template

```typescript
import type { RendererOptions } from "@kaggle-environments/core";

export function renderer(options: RendererOptions) {
  const { step, replay, parent } = options;
  const steps = replay.steps as any[];

  // Re-create DOM structure each call (simple, reliable)
  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header"></div>
      <canvas></canvas>
      <div class="status-container sketched-border"></div>
    </div>
  `;
  const header = parent.querySelector('.header') as HTMLDivElement;
  const canvas = parent.querySelector('canvas') as HTMLCanvasElement;
  const statusContainer = parent.querySelector('.status-container') as HTMLDivElement;
  if (!canvas || !replay) return;

  // Size canvas to fill its flex area
  canvas.width = 0;
  canvas.height = 0;
  const { width, height } = canvas.getBoundingClientRect();
  canvas.width = width;
  canvas.height = height;

  const c = canvas.getContext('2d');
  if (!c) return;

  const currentStep = steps[step];

  // --- Parse game state (game-specific) ---
  // For regular envs: currentStep.players[i]
  // For OpenSpiel (with proxy): JSON.parse(currentStep[0].observation.observationString)
  // For OpenSpiel (no proxy): parse raw text from currentStep[0].observation.observationString

  // --- 1. Build header (DOM) ---
  // Player names in sketched-border cards, active player highlighted
  header.innerHTML = `
    <span class="sketched-border" style="padding: 4px 12px; background-color: white; font-weight: 700;">Player 1</span>
    <span style="color: #444343;">vs</span>
    <span class="sketched-border" style="padding: 4px 12px; background-color: white; font-weight: 700;">Player 2</span>
  `;

  // --- 2. Draw game board on canvas ---
  c.clearRect(0, 0, width, height);
  // ... draw board, pieces, move highlights ...

  // --- 3. Update status container (DOM) ---
  statusContainer.textContent = 'Game status here';
}
```

### Rendering tips

- **Reuse DOM elements:** The renderer is called on every step change. The example above recreates innerHTML for simplicity, but for performance-sensitive games, create on first call and update on subsequent calls.
- **Canvas vs DOM:** Canvas works well for game boards. Plain DOM/HTML works for text-heavy games.
- **React alternative:** Pass `GameRenderer` (a React component) to `ReplayAdapter` instead of `renderer` for React-based visualizers. The component receives the same data as props.
- **Responsive sizing:** Use `parent.clientWidth` / `parent.clientHeight` to size your rendering area.

### Follow the style guide

See [visualizer-style-guide.md](visualizer-style-guide.md) for the complete visual design system -- colors, fonts, layout patterns, and CSS.

## Step 5: Add a replay transformer

**Required for any visualizer using `ui: 'side-panel'` (the default) whose raw step data doesn't already conform to the `BaseGameStep` shape.** The side-panel's ReasoningLogs component reads `step.players[i].thoughts` and `step.players[i].actionDisplayText`. If your steps come in a different shape -- for example, OpenSpiel games emit `[{ action, observation, reward, status }, ...]` per step with no `players` array -- then without a transformer the sidebar is silently empty (model thoughts never appear) and terminal-state logic breaks on forfeits (see below).

The transformer file lives **inside the visualizer**, not in `@kaggle-environments/core`. It's passed to `ReplayAdapter` via the `transformer:` option shown in the `main.ts` template. There is no central registry.

### File layout

```
visualizer/default/src/
├── main.ts
├── renderer.ts
└── transformers/
    └── <name>Transformer.ts
```

### Transformer template

```typescript
// src/transformers/<name>Transformer.ts

// Forfeit signals your env emits when a player exhausts illegal-move retries,
// times out, or crashes. Values below are what open_spiel_env uses; adapt to
// whatever your environment sets. See "Handle forfeits" below.
const FORFEIT_STATUSES = new Set(['TIMEOUT', 'ERROR', 'INVALID']);
const FORFEIT_REASONS: Record<string, string> = {
  TIMEOUT: 'ran out of time',
  INVALID: 'submitted an illegal move',
  ERROR: 'failed to produce valid input',
};

interface RawAction {
  submission?: number;
  actionString?: string | null;
  thoughts?: string | null;
  status?: string | null;
  generate_returns?: string[] | null;
}

interface RawPlayer {
  action?: RawAction;
  observation: { observationString?: string; isTerminal?: boolean };
  reward: number;
  status?: string;
}

export interface GamePlayer {
  id: number;
  name: string;
  thumbnail: string;
  isTurn: boolean;
  actionDisplayText: string;
  thoughts: string;
  reward: number;
  generateReturns: string[] | null;
  forfeited: boolean;
  forfeitLastAttempt: string | null;
}

export interface GameBoardState {
  // Fields your proxy emits from state_dict(). Fill in per game.
  is_terminal: boolean;
  winner: string | null;
  // ...
}

export interface GameStep {
  step: number;
  players: GamePlayer[];
  boardState: GameBoardState | null;
  isTerminal: boolean;
  winner: string | null;
  forfeitReason: string | null;
}

// action.thoughts is the harness-curated summary and the preferred source.
// generate_returns[0].main_response_and_thoughts is the raw LLM output;
// use it only when the harness didn't populate thoughts.
function parseThoughts(action?: RawAction): string {
  if (action?.thoughts) return action.thoughts;
  if (action?.generate_returns?.[0]) {
    try {
      const parsed = JSON.parse(action.generate_returns[0]);
      if (parsed.main_response_and_thoughts) return parsed.main_response_and_thoughts;
    } catch {}
  }
  return '';
}

function parseBoardState(step: RawPlayer[]): GameBoardState | null {
  const raw = step?.[0]?.observation?.observationString ?? step?.[1]?.observation?.observationString;
  if (!raw) return null;
  try { return JSON.parse(raw) as GameBoardState; } catch { return null; }
}

// Detect a single-player forfeit and its reason category. Two signals:
//   1. top-level player.status in FORFEIT_STATUSES (strict mode / TIMEOUT / ERROR)
//   2. action.submission === -1 with a non-null action.status
//      (the illegalMoveForfeit path -- open_spiel_env normalizes both
//      top-level statuses to DONE, so signal #1 doesn't fire there)
function detectForfeit(step: RawPlayer[]): { index: number; reasonKey: string } | null {
  if (step.length < 2) return null;

  const byStatus = step.map((p, i) => ({ p, i })).filter(({ p }) => p.status && FORFEIT_STATUSES.has(p.status));
  if (byStatus.length === 1) return { index: byStatus[0].i, reasonKey: byStatus[0].p.status! };
  if (byStatus.length > 1) return null;

  const byAction = step.map((p, i) => ({ p, i })).filter(({ p }) => p.action?.submission === -1 && !!p.action?.status);
  if (byAction.length === 1) return { index: byAction[0].i, reasonKey: 'INVALID' };
  return null;
}

function deriveWinner(step: RawPlayer[], teamNames: string[]): string | null {
  if (step.length < 2) return null;
  const r0 = step[0].reward ?? 0;
  const r1 = step[1].reward ?? 0;
  if (r0 === r1) return 'Draw';
  return r0 > r1 ? `${teamNames[0]} wins!` : `${teamNames[1]} wins!`;
}

export const gameTransformer = (environment: any): GameStep[] => {
  const teamNames: string[] = environment?.info?.TeamNames ?? ['Player 1', 'Player 2'];
  const rawSteps: RawPlayer[][] = environment?.steps ?? [];
  const out: GameStep[] = [];

  rawSteps.forEach((step, index) => {
    const forfeit = detectForfeit(step);

    const players: GamePlayer[] = step.map((p, i): GamePlayer => {
      const submission = p.action?.submission;
      const isForfeiter = forfeit?.index === i;
      // Forfeiters submit -1 but should still be treated as "acting" so
      // the step is retained and their thoughts / last attempt render.
      const isTurn = (submission !== undefined && submission !== -1) || isForfeiter;
      return {
        id: i,
        name: teamNames[i] ?? `Player ${i + 1}`,
        thumbnail: '',
        isTurn,
        actionDisplayText: p.action?.actionString ?? '',
        thoughts: parseThoughts(p.action),
        reward: p.reward ?? 0,
        generateReturns: p.action?.generate_returns ?? null,
        forfeited: isForfeiter,
        forfeitLastAttempt: isForfeiter ? (p.action?.actionString ?? null) : null,
      };
    });

    // Drop setup / no-op steps where neither player acted.
    if (!players.some((pl) => pl.isTurn)) return;

    const observationTerminal = !!step[0]?.observation?.isTerminal;
    // A forfeit ends the episode even though OpenSpiel's own state isn't
    // terminal -- treat it as terminal so the renderer stops showing "Turn: X".
    const isTerminal = observationTerminal || forfeit !== null;

    let forfeitReason: string | null = null;
    if (forfeit) {
      const loser = teamNames[forfeit.index] ?? `Player ${forfeit.index + 1}`;
      const winner = teamNames[1 - forfeit.index] ?? `Player ${2 - forfeit.index}`;
      forfeitReason = `${loser} ${FORFEIT_REASONS[forfeit.reasonKey] ?? 'forfeited'}. ${winner} wins by default.`;
    }

    out.push({
      step: index,
      players,
      boardState: parseBoardState(step),
      isTerminal,
      winner: isTerminal ? deriveWinner(step, teamNames) : null,
      forfeitReason,
    });
  });

  return out;
};
```

### Handle forfeits in the renderer

`open_spiel_env` ends an episode without setting `observation.isTerminal` in three cases:
- `player.status ∈ {'TIMEOUT', 'ERROR'}` -- overtime or crash
- `player.status === 'INVALID'` -- strict mode
- `action.submission === -1` with a non-null `action.status` -- the `illegalMoveForfeit` path, where both top-level statuses are normalized to `DONE`

The transformer template above detects all three and populates `currentStep.forfeitReason`. The renderer should:

1. Prefer `currentStep.isTerminal` over `observation.isTerminal` (they diverge on forfeit).
2. When `observation.winner` is null but `isTerminal`, derive the winner from reward sign.
3. Render `forfeitReason` as a visually distinct annotation (red italic works well) so it's not confused with the natural terminal line.

Example renderer snippet:

```typescript
const isTerminal = !!currentStep?.isTerminal || observation.is_terminal;
const forfeitReason = currentStep?.forfeitReason ?? null;
const forfeiterIdx = currentStep?.players?.findIndex((p) => p.forfeited) ?? -1;

if (isTerminal) {
  let winnerLabel: string;
  if (observation.winner === 'X') winnerLabel = `${playerNames[0]} wins!`;
  else if (observation.winner === 'O') winnerLabel = `${playerNames[1]} wins!`;
  else if (forfeitReason && forfeiterIdx >= 0) {
    const winnerIdx = 1 - forfeiterIdx;
    winnerLabel = `${playerNames[winnerIdx]} wins!`;
  } else winnerLabel = 'Draw';
  statusHTML = winnerLabel;
  if (forfeitReason) {
    statusHTML += `<span class="annotation forfeit-reason">${escapeHtml(forfeitReason)}</span>`;
  }
}
```

Add matching CSS:

```css
.status-container .annotation.forfeit-reason {
  color: #a03030;
  font-style: italic;
}
```

### Reference implementations

- `open_spiel_env/games/checkers/visualizer/default/src/transformers/checkersTransformer.ts` -- canonical in-visualizer transformer, structured board state + players array.
- `open_spiel_env/games/chess/visualizer/default/src/transformers/forfeit.ts` and `chessTransformer.ts` -- full forfeit taxonomy including per-attempt retries (`call_details`).
- `open_spiel_env/games/connect_four/visualizer/default/src/transformers/connectFourTransformer.ts` -- forfeit-by-reward-signal fallback for older replays without `action.status`.

## Step 6: Integrate with the environment

In the environment's Python module, ensure `html_renderer()` reads the built visualizer output:

```python
def html_renderer():
    jspath = path.join(dirpath, "visualizer", "default", "dist", "index.html")
    if path.exists(jspath):
        with open(jspath, encoding="utf-8") as f:
            return f.read()
    return ""
```

For OpenSpiel games, this is handled by the shared framework -- no per-game Python change is needed.

## Step 7: Write E2E tests (required)

Every visualizer **must** ship with a Playwright E2E test file at `visualizer/<variant>/e2e/<name>.test.ts`. Tests are auto-discovered by `playwright.config.ts` (which boots a dev server using `dev-with-replay` against your `replays/test-replay.json`).

**Keep tests minimal and shape-based, not exhaustive.** Do not enumerate every cell, piece, score value, or transition. The point is a smoke test that the visualizer mounts, renders, advances through steps, and reaches a terminal state -- not to lock in pixel-perfect behavior. Excessive assertions create churn every time the renderer changes.

### Four test shapes to copy

Reproduce the shapes below from existing tests; pick the ones that fit your game. Shape 4 (forfeit) is required for any OpenSpiel side-panel visualizer. Reference: `kaggle_environments/envs/connectx/visualizer/default/e2e/connectx.test.ts`, `kaggle_environments/envs/open_spiel_env/games/chess/visualizer/default/e2e/chess.test.ts`, `kaggle_environments/envs/open_spiel_env/games/repeated_poker/visualizer/default/e2e/repeated_poker.test.ts`, `kaggle_environments/envs/open_spiel_env/games/go/visualizer/fallback/e2e/go.test.ts`.

**Shape 1 -- "renders the game":** assert that the top-level container, the board, and one or two distinguishing elements (player names, title, key UI region) are visible. A handful of `toBeVisible` calls -- not an inventory.

```typescript
test('renders the game', async ({ page }) => {
  await expect(page.locator('.renderer-container')).toBeVisible();
  await expect(page.locator('.renderer-container canvas')).toBeVisible();
  // Optional: 1-2 game-specific anchors (title, player card, key region)
});
```

**Shape 2 -- "mid-game state":** drive the slider to the middle step and assert that game-state UI is present (current player indicator, pieces still on the board, etc.). Do not assert a specific position.

```typescript
test('displays correct game state at mid-game', async ({ page }) => {
  const slider = page.locator('input[type="range"]');
  await slider.waitFor({ state: 'visible' });
  const maxValue = await slider.getAttribute('max');
  const midStep = Math.floor(parseInt(maxValue || '0') / 2);
  await slider.fill(String(midStep));
  await page.waitForTimeout(200);
  // 1-2 assertions that mid-game UI is present
});
```

**Shape 3 -- "terminal state":** drive the slider to `max` and assert the game-over UI (winner text, "Match Complete", final score, etc.) appears. Match loosely with a regex; do not bind to a specific winner.

```typescript
test('displays winner status at final step', async ({ page }) => {
  const slider = page.locator('input[type="range"]');
  await slider.waitFor({ state: 'visible' });
  const maxValue = await slider.getAttribute('max');
  await slider.fill(maxValue || '0');
  await page.waitForTimeout(200);
  await expect(page.locator('p').filter({ hasText: /Wins|Winner|Draw/ })).toBeVisible();
});
```

**Shape 4 -- "forfeit terminal state":** point the dev server at your `test-forfeit-replay.json`, drive slider to `max`, and assert a forfeit line appears. This is the test that catches the "renderer freezes on last mid-game frame" bug. Add a second Playwright project in `playwright.config.ts` for this game that boots the dev server with `VITE_REPLAY_FILE=./replays/test-forfeit-replay.json`, or wrap it in a describe block that skips when the file is absent.

```typescript
test('shows forfeit reason at final step', async ({ page }) => {
  const slider = page.locator('input[type="range"]');
  await slider.waitFor({ state: 'visible' });
  const maxValue = await slider.getAttribute('max');
  await slider.fill(maxValue || '0');
  await page.waitForTimeout(200);
  await expect(
    page.locator('.status-container').filter({ hasText: /wins by default|forfeited|illegal move|ran out of time/i })
  ).toBeVisible();
});
```

### Guidelines

- Use loose matchers (`/Winner|Wins|Draw/`, `getBy*` over deep CSS chains) so tests survive cosmetic changes.
- A `beforeEach` that does `await page.goto('/')` is standard.
- Don't add tests for animation timing, hover/click interactions, or visual styling -- the dev replay only has one game.

Run locally with `pnpm test:e2e` (filter via `pnpm test:e2e --project <name>`).

## Step 8: Build and verify

```bash
# Install dependencies (from repo root)
pnpm install

# Run dev server with hot reload (interactive game picker)
pnpm dev

# Run dev server with a specific replay file
pnpm dev-with-replay   # select your game from the picker

# Build for production (interactive picker)
pnpm build

# Build all visualizers
pnpm build-all

# Run E2E tests
pnpm test:e2e

# Format TypeScript/JavaScript
pnpm format
```

## Checklist

- [ ] `package.json` has `@kaggle-environments/core` as `workspace:*` dependency
- [ ] `vite.config.ts` extends `web/vite.config.base` with correct relative path depth
- [ ] `tsconfig.json` extends `web/tsconfig.base.json` with correct relative path depth
- [ ] `index.html` has `<div id="app"></div>`
- [ ] `src/main.ts` uses `createReplayVisualizer` + `ReplayAdapter`
- [ ] `src/style.css` follows the [visualizer-style-guide.md](visualizer-style-guide.md)
- [ ] Renderer handles first call (create elements) and subsequent calls (update)
- [ ] Current actor, move taken, move implications, and score are all visible
- [ ] `html_renderer()` in the Python env reads `dist/index.html` (regular envs only)
- [ ] `test-replay.json` has a full game (not 2-3 steps from agent failure)
- [ ] `test-forfeit-replay.json` present, and the sidebar / status line render the forfeit reason distinctly (OpenSpiel side-panel visualizers)
- [ ] Transformer wired via `ReplayAdapter({ transformer })` -- lives in `src/transformers/`, NOT `web/core/src/transformers.ts` (there is no central registry)
- [ ] `players[].thoughts` populated from `action.thoughts` with fallback to `action.generate_returns[0].main_response_and_thoughts` -- verified by expanding a step in the sidebar
- [ ] Transformer's `detectForfeit()` covers all three forfeit signals (top-level status, action.submission === -1 + action.status)
- [ ] Renderer prefers `currentStep.isTerminal` over `observation.isTerminal`, and derives winner from reward sign when `observation.winner` is null
- [ ] `e2e/<name>.test.ts` covers renders / mid-game / terminal / **forfeit terminal** (shapes 1-4) -- not exhaustive
- [ ] `pnpm test:e2e --project <name>` passes
- [ ] `pnpm build` produces output in `dist/`
- [ ] `pnpm format` passes
- [ ] `pnpm validate-conventions` passes (catches missing allowlist entry)
- [ ] If standalone (non-OpenSpiel): game dir added to `KNOWN_STANDALONE_GAME_DIRS` in `web/scripts/validate-visualizer-conventions.js`

## Reference implementations

- `kaggle_environments/envs/rps/visualizer/default/` -- simple canvas-based renderer
- `kaggle_environments/envs/werewolf/visualizer/default/` -- more complex with custom transformer
- `kaggle_environments/envs/open_spiel_env/games/checkers/visualizer/default/` -- canonical OpenSpiel visualizer with in-visualizer transformer + `parseThoughts` helper
- `kaggle_environments/envs/open_spiel_env/games/chess/visualizer/default/` -- full forfeit taxonomy including per-attempt retry rendering (`src/transformers/forfeit.ts` + `chessTransformer.ts`)
- `kaggle_environments/envs/open_spiel_env/games/connect_four/visualizer/default/` -- forfeit-by-reward-signal fallback for older replays
- `web/core/src/index.ts` -- all exports from `@kaggle-environments/core`

## Troubleshooting

**Replay has only 2-3 steps / "INVALID ACTION DETECTED":** The OpenSpiel `"random"` agent needs `includeLegalActions: True` in the configuration. Generate the replay with:
```python
env = make('open_spiel_<name>', debug=True, configuration={'includeLegalActions': True})
```

**Canvas is blank:** Check the browser console for errors. Common issues: incorrect CSS (canvas has 0 height), parse function returning null because the observation string format doesn't match expectations. Print the raw observation string to debug.

**Observation string is empty (OpenSpiel):** Some games use `information_state_string()` instead of `observation_string()`. The framework handles this automatically -- check the game type in the OpenSpiel source for `provides_observation_string` vs `provides_information_state_string`.

**Game requires list parameters (OpenSpiel):** OpenSpiel uses semicolons inside square brackets for lists: `ship_sizes=[2;3;4]`, `ship_values=[1.0;1.0;1.0]`. These go directly in the game string in `GAMES_LIST`.
