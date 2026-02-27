---
name: create-visualizer
description: Create or update a web visualizer for a Kaggle game environment (Vite + TypeScript frontend with replay playback)
---

# Create / Update a Visualizer

Use `$ARGUMENTS` as the environment name if provided (e.g., `/create-visualizer hangman`).

## Step 1: Create the project structure

Create `kaggle_environments/envs/<name>/visualizer/default/` with:

```
visualizer/default/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── index.html
└── src/
    ├── main.ts
    └── renderer.ts
```

This directory is automatically part of the pnpm workspace (via root `pnpm-workspace.yaml` pattern `kaggle_environments/envs/*/visualizer/*`).

## Step 2: Write the boilerplate files

### `package.json`

```json
{
  "name": "@kaggle-environments/<name>-visualizer",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
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
import baseConfig from "../../../../../web/vite.config.base";

export default mergeConfig(baseConfig, defineConfig({}));
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

The base config (at `web/tsconfig.base.json`) provides: ESNext target/module, strict mode, JSX react-jsx, and the `@kaggle-environments/core` path alias.

### `index.html`

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title><Name> Visualizer</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

The `<div id="app">` is required -- `createReplayVisualizer` mounts to it.

## Step 3: Write the entry point (`src/main.ts`)

```typescript
import { createReplayVisualizer, ReplayAdapter } from "@kaggle-environments/core";
import { renderer } from "./renderer";

const app = document.getElementById("app");
if (app) {
  if (import.meta.env?.DEV && import.meta.hot) {
    import.meta.hot.accept();
  }
  createReplayVisualizer(
    app,
    new ReplayAdapter({
      gameName: "<name>",
      renderer: renderer,
      ui: "inline",
    })
  );
}
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

## Step 4: Write the renderer (`src/renderer.ts`)

The renderer function is called on every step change. It receives a `RendererOptions` object and should draw into the provided `parent` element.

```typescript
import type { RendererOptions } from "@kaggle-environments/core";

export function renderer(options: RendererOptions) {
  const { replay, parent, step, setStep, setPlaying } = options;
  const currentStep = replay.steps[step];
  const config = replay.configuration;

  // Create DOM elements on first call, update on subsequent calls.
  // The parent element persists across calls -- reuse elements.

  let canvas = parent.querySelector("canvas") as HTMLCanvasElement | null;
  if (!canvas) {
    canvas = document.createElement("canvas");
    parent.appendChild(canvas);
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // Size the canvas to fit the container
  const width = parent.clientWidth || 400;
  const height = parent.clientHeight || 400;
  canvas.width = width;
  canvas.height = height;

  // Clear and draw for the current step
  ctx.clearRect(0, 0, width, height);

  // Access step data:
  // currentStep.players[i].id, .name, .isTurn, .actionDisplayText
  // For raw env data, use replay.steps (array of raw step arrays from env.toJSON())

  // ... your rendering logic ...
}
```

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

### Replay data shape

The `replay` object matches the environment's `toJSON()` output transformed through the core adapter:

```typescript
interface ReplayData {
  name: string;           // environment name
  version: string;
  steps: BaseGameStep[];  // transformed steps (or raw if no transformer)
  configuration: Record<string, any>;
}

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

If you need raw step data (direct access to observations, actions, rewards as defined in the spec), you can write a custom transformer or access the raw steps before transformation.

### Rendering tips

- **Reuse DOM elements:** The renderer is called on every step change. Don't recreate the entire DOM each time -- create on first call, update on subsequent calls.
- **Canvas vs DOM:** Canvas works well for game boards. Plain DOM/HTML works for text-heavy games.
- **React alternative:** Pass `GameRenderer` (a React component) to `ReplayAdapter` instead of `renderer` for React-based visualizers. The component receives the same data as props.
- **Responsive sizing:** Use `parent.clientWidth` / `parent.clientHeight` to size your rendering area.

## Step 5: Integrate with the environment

In `<name>.py`, ensure `html_renderer()` reads the built visualizer output:

```python
def html_renderer():
    jspath = path.join(dirpath, "visualizer", "default", "dist", "index.html")
    if path.exists(jspath):
        with open(jspath, encoding="utf-8") as f:
            return f.read()
    return ""
```

## Step 6: Develop and build

```bash
# Install dependencies (from repo root)
pnpm install

# Run dev server with hot reload (interactive game picker)
pnpm dev

# Build for production (interactive picker)
pnpm build

# Build all visualizers
pnpm build-all

# Run E2E tests
pnpm test:e2e

# Format TypeScript/JavaScript
pnpm format
```

During development, `pnpm dev` runs the `find-games.js` script which prompts you to select a game, then starts a Vite dev server with hot module replacement.

## Checklist

- [ ] `package.json` has `@kaggle-environments/core` as `workspace:*` dependency
- [ ] `vite.config.ts` extends `web/vite.config.base`
- [ ] `tsconfig.json` extends `web/tsconfig.base.json`
- [ ] `index.html` has `<div id="app"></div>`
- [ ] `src/main.ts` uses `createReplayVisualizer` + `ReplayAdapter`
- [ ] Renderer handles first call (create elements) and subsequent calls (update)
- [ ] `html_renderer()` in the Python env reads `dist/index.html`
- [ ] `pnpm build` produces output in `dist/`
- [ ] `pnpm format` passes

## Reference implementations

- `kaggle_environments/envs/rps/visualizer/default/` — simple canvas-based renderer
- `kaggle_environments/envs/werewolf/visualizer/default/` — more complex with custom transformer
- `web/core/src/index.ts` — all exports from `@kaggle-environments/core`
