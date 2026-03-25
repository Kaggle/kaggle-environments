# Create / Update a Visualizer

Build a web visualizer for any Kaggle game environment -- regular or OpenSpiel. Each visualizer is a Vite + TypeScript project within the pnpm workspace.

**Related skills:**
- `create-environment` -- if you need to build the Python backend for a custom game first
- `onboard-open-spiel-game` -- if you need to register or configure an OpenSpiel game first

## Step 1: Determine your variant

| Variant | Directory | Relative path to `web/` | `gameName` |
|---------|-----------|------------------------|------------|
| Regular env | `kaggle_environments/envs/<name>/visualizer/default/` | `../../../../../` (5 levels) | `"<name>"` |
| OpenSpiel env | `kaggle_environments/envs/open_spiel_env/games/<name>/visualizer/default/` | `../../../../../../../` (7 levels) | `"open_spiel_<name>"` |

Both variants use the same boilerplate, shared workspace dependency, and renderer interface. The only differences are the directory depth (which affects relative paths to base configs) and the replay data shape.

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

4. **Current score / game progress:** Show scores, piece counts, progress indicators. At game over, display the final result prominently.

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

## Step 5 (optional): Add a transformer

If your game needs data preprocessing (e.g., parsing observation strings into structured step objects), add a transformer in `web/core/src/transformers/`.

1. Create `web/core/src/transformers/<name>/`:
   - `<name>ReplayTypes.ts` -- TypeScript types for raw and transformed steps
   - `<name>Transformer.ts` -- transform function and step label/description helpers

2. Register it in `web/core/src/transformers.ts`:
   ```typescript
   import { myGameTransformer, getMyGameStepLabel, getMyGameStepDescription } from './transformers/<name>/<name>Transformer';
   import { MyGameStep } from './transformers/<name>/<name>ReplayTypes';

   // In processEpisodeData switch:
   case 'open_spiel_<name>':
     transformedSteps = myGameTransformer(environment);
     break;

   // In getGameStepLabel switch:
   case 'open_spiel_<name>':
     return getMyGameStepLabel(gameStep as MyGameStep);

   // In getGameStepDescription switch:
   case 'open_spiel_<name>':
     return getMyGameStepDescription(gameStep as MyGameStep);
   ```

3. Then use the transformed data in your renderer instead of parsing raw observations.

**Reference transformers:** `web/core/src/transformers/chess/`, `web/core/src/transformers/connect_four/`, `web/core/src/transformers/go/`.

A transformer is not required -- games with a proxy already get structured JSON observations, and simpler games can parse observation strings directly in the renderer.

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

## Step 7: Build and verify

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
- [ ] `pnpm build` produces output in `dist/`
- [ ] `pnpm format` passes
- [ ] If transformer: registered in `web/core/src/transformers.ts` switch statements

## Reference implementations

- `kaggle_environments/envs/rps/visualizer/default/` -- simple canvas-based renderer
- `kaggle_environments/envs/werewolf/visualizer/default/` -- more complex with custom transformer
- `kaggle_environments/envs/open_spiel_env/games/connect_four/visualizer/default/` -- OpenSpiel visualizer
- `web/core/src/index.ts` -- all exports from `@kaggle-environments/core`

## Troubleshooting

**Replay has only 2-3 steps / "INVALID ACTION DETECTED":** The OpenSpiel `"random"` agent needs `includeLegalActions: True` in the configuration. Generate the replay with:
```python
env = make('open_spiel_<name>', debug=True, configuration={'includeLegalActions': True})
```

**Canvas is blank:** Check the browser console for errors. Common issues: incorrect CSS (canvas has 0 height), parse function returning null because the observation string format doesn't match expectations. Print the raw observation string to debug.

**Observation string is empty (OpenSpiel):** Some games use `information_state_string()` instead of `observation_string()`. The framework handles this automatically -- check the game type in the OpenSpiel source for `provides_observation_string` vs `provides_information_state_string`.

**Game requires list parameters (OpenSpiel):** OpenSpiel uses semicolons inside square brackets for lists: `ship_sizes=[2;3;4]`, `ship_values=[1.0;1.0;1.0]`. These go directly in the game string in `GAMES_LIST`.
