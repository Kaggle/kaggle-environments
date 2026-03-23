# Create a Visualizer for a New OpenSpiel Game

Build a standalone web visualizer (hybrid DOM + canvas) for an OpenSpiel game that does **not** already have a harness, proxy, or visualizer in kaggle-environments. The game must already exist in OpenSpiel's game registry. No proxy, transformer, or custom game implementation is needed -- the visualizer parses the raw OpenSpiel observation string directly.

**When to use this skill:** You have an OpenSpiel game name (e.g., `"quoridor"`, `"amazons"`) and want to create a complete visualizer so humans can watch replays. The game doesn't need any special observation formatting -- you work with whatever string OpenSpiel's `observation_string()` returns.

**Related skills:**
- `onboard-open-spiel-game` -- if you also need a proxy, custom game, tests, or full onboarding
- `create-visualizer` -- for non-OpenSpiel environments

## Prerequisites

### OpenSpiel source code

The OpenSpiel repository should be cloned locally (typically at `../open_spiel` relative to the kaggle-environments root). This is **required** -- you need to read the game's C++ source to understand:
- The observation string format (`ObservationString` method)
- Game parameters and defaults
- Game type (perfect vs imperfect information, sequential vs simultaneous)
- Action encoding

Find the game source at `open_spiel/games/<game_name>/` (or `open_spiel/games/<game_name>.h` / `open_spiel/games/<game_name>.cc` for single-file games).

### Python virtual environment

Create a venv for running Python commands. Do NOT install packages globally.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

This installs `kaggle-environments` with all dependencies (including `pyspiel`) in an isolated environment. All subsequent Python commands should use `.venv/bin/python`.

### Verify the game exists

```bash
.venv/bin/python -c "import pyspiel; print('$GAME_NAME' in [g.short_name for g in pyspiel.registered_games().values()])"
```

Or simply check that the source directory exists in the OpenSpiel repo: `ls ../open_spiel/open_spiel/games/$GAME_NAME/`

## Step 1: Study the game from OpenSpiel source

Read the game's `.h` file in the OpenSpiel repo to understand:

1. **Observation string format** -- search for `ObservationString`, `ToString`, `OwnBoardString`, etc. The header comments typically show example output with character-by-character explanations.

2. **Game parameters** -- look for default constants and the `GameParameters` section. Note which params affect board size, game length, etc.

3. **Game type** -- check for:
   - `Information::PERFECT_INFORMATION` vs `IMPERFECT_INFORMATION`
   - `Dynamics::SEQUENTIAL` vs `SIMULTANEOUS`
   - Number of players

4. **Action encoding** -- understand `NumDistinctActions()` and how actions are serialized.

**Why this matters:**
- **Imperfect information games** (e.g., Battleship, poker): each player gets a DIFFERENT `observationString`. The renderer must handle per-player observations via `step[0].observation.observationString` and `step[1].observation.observationString` separately. You'll likely want to show multiple boards.
- **Perfect information games** (e.g., Breakthrough, Pentago): all players see the same board. A single `getObservationString(step)` that returns the first non-empty observation is sufficient.

### Choose appropriate parameters

Default parameters may create games that are too large for a good visualizer experience. For example, Battleship defaults to 10x10 with 5 ships and 50 shots. Choose smaller parameters that keep the game visually clear:

```
# Too large:
"battleship"

# Better for visualizer:
"battleship(board_width=5,board_height=5,ship_sizes=[2;3],ship_values=[1;1],num_shots=10,allow_repeated_shots=false)"
```

Note: list parameters in OpenSpiel use semicolons inside square brackets: `ship_sizes=[2;3;4]`.

## Step 2: Register the game in GAMES_LIST

Edit `kaggle_environments/envs/open_spiel_env/open_spiel_env.py` and add the game string to `GAMES_LIST` (around line 850). Keep alphabetical order.

```python
GAMES_LIST = [
    # ...existing games...
    "<game_name>",                           # simple
    "<game_name>(board_size=5,param=val)",   # with parameters
]
```

The environment will be registered as `open_spiel_<game_name>` and accessible via `make("open_spiel_<game_name>")`.

## Step 3: Generate test-replay.json

**Important:** The built-in `"random"` agent requires `includeLegalActions: True` to function. Without it, the agent cannot determine legal moves and will immediately produce an invalid action, ending the game in 2-3 steps.

```bash
.venv/bin/python -c "
from kaggle_environments import make
import json
env = make('open_spiel_$GAME_NAME', debug=True, configuration={'includeLegalActions': True})
env.run(['random', 'random'])
replay = env.toJSON()
with open('test-replay.json', 'w') as f:
    json.dump(replay, f, indent=2)
print(f'Generated replay with {len(replay[\"steps\"])} steps')
print(f'Statuses: {replay[\"statuses\"]}')
print(f'Rewards: {replay[\"rewards\"]}')
"
```

Verify the replay has a reasonable number of steps (not 2-3, which indicates the random agent failed). Inspect a few observation strings to confirm they match the format you studied in Step 1:

```bash
.venv/bin/python -c "
import json
with open('test-replay.json') as f:
    replay = json.load(f)
# Show a mid-game observation
mid = len(replay['steps']) // 2
print(replay['steps'][mid][0]['observation'].get('observationString', '(none)'))
"
```

Move this file to the replay directory after creating the project structure.

## Step 4: Create the visualizer directory

Path: `kaggle_environments/envs/open_spiel_env/games/<game_name>/visualizer/default/`

Also create an empty `kaggle_environments/envs/open_spiel_env/games/<game_name>/__init__.py`.

### File tree

```
games/<game_name>/
  __init__.py                            (empty)
  visualizer/default/
    package.json
    vite.config.ts
    tsconfig.json
    index.html
    replays/test-replay.json             (from Step 3)
    src/
      main.ts
      renderer.ts
      style.css
```

### package.json

```json
{
  "name": "@kaggle-environments/open-spiel-<game_name>-visualizer",
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

### vite.config.ts

```typescript
import { defineConfig, mergeConfig } from 'vite';
import baseConfig from '../../../../../../../web/vite.config.base';

export default mergeConfig(
  baseConfig,
  defineConfig({
    publicDir: 'replays',
  })
);
```

Note: OpenSpiel visualizers are 7 levels deep from the repo root (`kaggle_environments/envs/open_spiel_env/games/<name>/visualizer/default/`), so the relative path to `web/` is `../../../../../../../`.

### tsconfig.json

```json
{
  "extends": "../../../../../../../web/tsconfig.base.json",
  "include": ["src"],
  "compilerOptions": {
    "allowJs": true
  }
}
```

### index.html

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title><Game Name> Visualizer</title>
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

### src/style.css

```css
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap');

html, body, #app {
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
  overflow: hidden;
}

.renderer-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  height: 100%;
  min-height: 0;
  background-image: url('./images/paper.webp');
  background-size: cover;
  background-position: center;
  overflow: hidden;
  font-family: 'Inter', sans-serif;
  box-sizing: border-box;
  padding: 12px;
  color: #050001;
  container-type: inline-size;
}

.renderer-container canvas {
  position: relative;
  flex-grow: 1;
  width: 100%;
  max-width: 512px;
  min-height: 0;
}

.squiggle-border {
  background-image:
    url('./images/squiggle-solid.png'), url('./images/squiggle-solid.png'),
    url('./images/squiggle-v.png'), url('./images/squiggle-v.png');
  background-repeat: repeat-x, repeat-x, repeat-y, repeat-y;
  background-position: top, bottom, left, right;
  background-size: 5rem 2px, 5rem 2px, 2px 5rem, 2px 5rem;
}

.header {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  padding: 8px 0;
  font-size: 1.1rem;
  font-weight: 600;
  flex-shrink: 0;
  gap: 16px;
}

.status-container {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 5px 16px;
  background-color: white;
  font-size: 0.9rem;
  font-weight: 600;
  min-height: 18px;
  min-width: 200px;
  margin-top: 8px;
  flex-shrink: 0;
}
```

### src/main.ts

```typescript
import { createReplayVisualizer, ReplayAdapter } from '@kaggle-environments/core';
import { renderer } from './renderer';
import './style.css';

const app = document.getElementById('app');
if (!app) {
  throw new Error('Could not find app element');
}

if (import.meta.env?.DEV && import.meta.hot) {
  import.meta.hot.accept();
}

createReplayVisualizer(
  app,
  new ReplayAdapter({
    gameName: 'open_spiel_<game_name>',
    renderer: renderer as any,
    ui: 'side-panel',
  })
);
```

## Visual style guide

All OpenSpiel visualizers should match a **paper-and-ink** aesthetic. The goal is a warm, tactile, stationery-like look -- as if the game were drawn on paper with hand-sketched borders.

### Required assets

Copy these shared image assets into your visualizer's `public/images/` directory (source them from the go v2 visualizer at `games/go/visualizer/v2/public/images/`):

- `paper.webp` -- warm parchment background texture
- `squiggle-solid.png` -- hand-drawn solid line tile (horizontal/vertical borders)
- `squiggle-v.png` -- hand-drawn vertical line tile
- `squiggle-dash.png` -- hand-drawn dashed line tile (canvas grid lines)

### Aesthetic principles

1. **Paper-textured, not solid backgrounds.** Use `paper.webp` as the main background via `background-image`, not a solid color. The canvas should have a transparent background so the paper texture shows through from the DOM layer beneath.

2. **Light color scheme.** Use near-black text (`#050001`) on the paper background. Avoid dark backgrounds, white-on-dark text, and neon/diffused glows.

3. **Hand-drawn borders.** Use tiled squiggle PNG patterns for container borders instead of CSS `border` or `box-shadow`. Apply the `.squiggle-border` utility class (defined in `style.css`) to containers. This gives a sketched, woodblock-print quality.

4. **High-resolution text.** Prefer **DOM elements** for all text, labels, and status displays rather than canvas text. Canvas `fillText` cannot use web fonts reliably. Use canvas only for the game board/grid itself. Wrap the canvas in a flex container alongside DOM-based status elements.

5. **Two typefaces.** Use **Inter** (sans-serif) for all UI text -- player names, scores, labels, controls. Use **Mynerve** (cursive) as an optional accent font for annotations, commentary, and decorative text. Load Inter via CSS `@import` in `style.css` and Mynerve via `<link>` in `index.html`.

6. **Hard offset shadows.** For modals and popover panels, use hard black offset shadows (e.g., `box-shadow: -0.75rem 0.75rem`) rather than soft diffused drop-shadows. This matches the woodblock/stamp aesthetic.

7. **Responsive sizing.** Use CSS container queries (`@container (max-width: 680px)`) for responsive layout adjustments. Set `container-type: inline-size` on the main wrapper. The **680px** breakpoint is the mobile threshold. Use `rem`-based font sizes (`0.8rem`, `1rem`, `1.1rem`).

### Color palette

| Element | Color / Treatment | Notes |
|---------|------------------|-------|
| Page background | `paper.webp` texture | Warm parchment via `background-image`, never a solid color |
| Primary text | `#050001` | Near-black, used on all body text |
| Secondary text | `#444343` | Softer dark for table values and metadata |
| Container background | `white` | Player cards, score tables, panels |
| Active player highlight | `#bdeeff` | Light blue background on the active player card |
| Borders | Squiggle PNG tiles | Hand-drawn look via `.squiggle-border` class |
| Buttons / controls bg | `#f1f1f1` | Light gray for interactive elements |
| Button shadow | `box-shadow: -0.125rem 0.125rem 0 #000` | Hard black offset, not diffused |
| Canvas background | Transparent | Paper texture shows through from DOM layer |
| Board grid lines | Squiggle-dash texture | Tiled for hand-drawn line appearance |
| Board labels | `#000000` (Inter font) | Column/row labels around the board |

### Rendering approach

Use a **hybrid DOM + canvas** architecture:

- **Canvas**: game board grid, pieces, move highlights, board decorations. Keep the canvas background transparent so the paper texture shows through.
- **DOM**: player names, score tables, turn indicators, game-over modals, annotations. Use the `.squiggle-border` class for container borders.

Cap the canvas at a maximum width (e.g., `max-width: 512px`) and use `aspect-ratio: 1` for square boards.

```
+------------------------------------------+
|  [DOM] Header: player cards with         |
|  squiggle borders                        |
+------------------------------------------+
|                                          |
|  [Canvas] Game board (transparent bg)    |
|  on paper texture                        |
|                                          |
+------------------------------------------+
|  [DOM] Status / score with squiggle      |
|  borders, annotations in Mynerve font   |
+------------------------------------------+
```

### Squiggle border container pattern

Use white containers with the `.squiggle-border` class for hand-drawn borders:

```typescript
const statusContainer = document.createElement('div');
statusContainer.className = 'squiggle-border';
Object.assign(statusContainer.style, {
  padding: '5px 12px',
  backgroundColor: 'white',
  textAlign: 'center',
  minWidth: '200px',
  marginTop: '10px',
  fontFamily: "'Inter', sans-serif",
});
```

### Active player indication

Use background color change and scale transform on player containers:

```css
.player {
  background-color: white;
  transition: scale 300ms;
}

.player.active {
  background-color: #bdeeff;
  scale: 1.1;
}
```

### Game-over presentation

Use a modal overlay with staggered reveal animations:

```css
.game-over-modal {
  background-image: url('./images/paper.webp');
  background-size: cover;
  background-position: center;
  color: #050001;
}
```

Display results in a table with squiggle borders. Use CSS `@starting-style` and `transition` for staggered element reveals.

## Step 5: Write the renderer

The renderer is the core of the visualizer. It is called on every step change and renders into the parent element using a mix of DOM elements and canvas.

### Renderer architecture

```typescript
import type { RendererOptions } from '@kaggle-environments/core';

// 1. Define a typed interface for the parsed game state
interface GameState {
  // Game-specific fields parsed from the observation string
}

// 2. Parse the observation string into your typed state
function parseObservation(obsString: string): GameState | null {
  // Parse the OpenSpiel observation string format
  // (the format you studied from the C++ source in Step 1)
}

// 3. Helpers to extract data from raw step arrays
//    For PERFECT information games, use a single helper that finds the first obs:
function getObservationString(step: any): string {
  if (!step || !Array.isArray(step)) return '';
  for (const player of step) {
    const obs = player?.observation?.observationString;
    if (obs) return obs;
  }
  return '';
}

//    For IMPERFECT information games, extract per-player observations:
function getPlayerObservationString(step: any, playerIdx: number): string {
  if (!step || !Array.isArray(step)) return '';
  return step[playerIdx]?.observation?.observationString ?? '';
}

// 4. Helper to check if the game is over
function isTerminal(step: any): boolean {
  if (!step || !Array.isArray(step)) return false;
  return step.some((p: any) => p?.status === 'DONE' || p?.observation?.isTerminal);
}

// 5. Helper to get current player
function getCurrentPlayer(step: any): number {
  if (!step || !Array.isArray(step)) return 0;
  for (const player of step) {
    const cp = player?.observation?.currentPlayer;
    if (cp !== undefined && cp >= 0) return cp;
  }
  return 0;
}

// 6. Helper to get rewards
function getRewards(step: any): [number, number] {
  if (!step || !Array.isArray(step)) return [0, 0];
  return [step[0]?.reward ?? 0, step[1]?.reward ?? 0];
}

// 7. Main renderer function
export function renderer(options: RendererOptions) {
  const { step, replay, parent } = options;
  const steps = replay.steps as any[];
  // ... rendering logic
}
```

### Replay data shape

Each step in `replay.steps` is an array of player observations:

```
replay.steps[stepIndex][playerIndex].observation.observationString  -- the OpenSpiel state string
replay.steps[stepIndex][playerIndex].observation.currentPlayer      -- whose turn it is
replay.steps[stepIndex][playerIndex].observation.isTerminal         -- game over flag
replay.steps[stepIndex][playerIndex].action.submission              -- action taken (-1 = not acting)
replay.steps[stepIndex][playerIndex].reward                        -- cumulative reward
replay.steps[stepIndex][playerIndex].status                        -- "ACTIVE" or "DONE"
```

**Key detail for imperfect information games:** `step[0].observation.observationString` and `step[1].observation.observationString` contain DIFFERENT strings -- each player's private view. Parse both and render them (e.g., side-by-side boards for Battleship).

### Visual design requirements

Every visualizer MUST clearly communicate these four things:

#### 1. Current actor (whose turn it is)

- Show a **DOM header** at the top with the current player's name
- Highlight the active player's card with a `#bdeeff` light blue background and `scale: 1.1` (see style guide)
- On game over, show the result in a paper-textured modal with squiggle-border table

#### 2. Move taken (what just happened)

- Compare the current step's state with the previous step's state to detect what changed
- Highlight the move visually: glowing ring on placed piece, gold overlay on moved-from/moved-to squares, dashed outline on played pit, etc.
- Show a brief textual description of the move when appropriate (e.g., "P1 fires B3: HIT!")

#### 3. Move implications (what the move caused)

- Show **deltas/diffs** when state values change: `+N` / `-N` badges near changed elements
- Mark captured/removed pieces distinctly (red X, faded outline, "CAPTURED" label)
- Highlight score changes with colored `+N` indicators near the score display
- For games with complex mechanics (e.g., sowing seeds, rotating quadrants), visually indicate the affected region

#### 4. Current score / game progress

- Show scores in the header or a DOM stats element using player-colored text
- Show piece counts, stones remaining, boxes claimed, hit/miss ratios, etc.
- At game over, display the final result prominently in the status container

### Handling game phases

Some games have distinct phases (e.g., Battleship has ship placement then war). Detect the phase from the observation state and adjust the display accordingly:
- Show phase name in the header or status container
- Adapt what stats are shown (e.g., no shot stats during placement)
- Potentially change the visual emphasis

### Renderer template (hybrid DOM + canvas)

```typescript
export function renderer(options: RendererOptions) {
  const { step, replay, parent } = options;
  const steps = replay.steps as any[];

  // Re-create DOM structure each call (simple, reliable)
  parent.innerHTML = `
    <div class="renderer-container">
      <div class="header"></div>
      <canvas></canvas>
      <div class="status-container squiggle-border"></div>
    </div>
  `;
  const container = parent.querySelector('.renderer-container') as HTMLDivElement;
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

  // Parse current state
  const currentStep = steps[step];
  const obsString = getObservationString(currentStep);
  const state = parseObservation(obsString);

  // Parse previous state for diff visualization
  let prevState: GameState | null = null;
  if (step > 0) {
    prevState = parseObservation(getObservationString(steps[step - 1]));
  }

  if (!state) {
    statusContainer.textContent = 'Waiting for game data...';
    return;
  }

  const terminal = isTerminal(currentStep);
  const cp = getCurrentPlayer(currentStep);

  // --- 1. Build header (DOM) ---
  // Player names in squiggle-border cards, active player highlighted
  const p1Active = !terminal && cp === 0;
  const p2Active = !terminal && cp === 1;
  header.innerHTML = `
    <span class="squiggle-border" style="padding: 4px 12px; background-color: ${p1Active ? '#bdeeff' : 'white'}; font-weight: 700;">Player 1</span>
    <span style="color: #444343;">vs</span>
    <span class="squiggle-border" style="padding: 4px 12px; background-color: ${p2Active ? '#bdeeff' : 'white'}; font-weight: 700;">Player 2</span>
  `;

  // --- 2. Draw game board on canvas ---
  // Canvas is transparent -- paper.webp shows through from the DOM layer
  c.clearRect(0, 0, width, height);
  // ... draw board, pieces, move highlights ...

  // --- 3. Update status container (DOM) ---
  if (terminal) {
    const rewards = getRewards(currentStep);
    let msg = 'Game Over -- Draw';
    if (rewards[0] > rewards[1]) msg = 'Game Over -- Player 1 wins!';
    else if (rewards[1] > rewards[0]) msg = 'Game Over -- Player 2 wins!';
    statusContainer.textContent = msg;
    statusContainer.style.fontWeight = '700';
  } else {
    statusContainer.textContent = `Player ${cp + 1}'s turn`;
  }
}
```

### Layout guidelines

- **Header** (DOM): player names and title at the top, flexbox centered with gap
- **Canvas**: game board centered in the remaining flex space, scaled responsively
- **Status container** (DOM): white container with squiggle border at the bottom with game state text
- Use `Math.min()` to cap board size and ensure it doesn't overflow
- Keep margins around the board for labels (draw labels on canvas or use positioned DOM elements)
- For multi-board games (imperfect info): arrange boards in a 2x2 or side-by-side grid on the canvas

## Step 6: Build and verify

```bash
# From repo root
pnpm install

# Build (catches TypeScript errors)
pnpm build   # select your game from the picker

# Run dev server with replay
pnpm dev-with-replay   # select your game from the picker

# Or run dev server directly for this specific game
cd kaggle_environments/envs/open_spiel_env/games/<game_name>/visualizer/default
pnpm dev-with-replay
```

**Common build errors:**
- Unused variables/parameters: remove them
- Missing type imports: ensure `import type { RendererOptions } from '@kaggle-environments/core'`

### Verification checklist

- [ ] Game string added to `GAMES_LIST` in `open_spiel_env.py`
- [ ] `games/<game_name>/__init__.py` exists (empty is fine)
- [ ] `test-replay.json` has a full game (not just 2-3 steps from invalid actions)
- [ ] `pnpm install` succeeds from repo root
- [ ] `pnpm build` passes with no TypeScript errors
- [ ] `pnpm dev-with-replay` launches and the game appears in the picker
- [ ] Stepping through the replay shows the board updating correctly
- [ ] Current player is clearly indicated with color
- [ ] Last move is highlighted (glow, ring, overlay, etc.)
- [ ] Move diffs are shown (deltas, captures, removed pieces)
- [ ] Score / game progress is visible in the header or status container
- [ ] Game over state displays the result clearly
- [ ] Board scales responsively (try resizing the window)
- [ ] Text uses Inter font (loaded via Google Fonts import in style.css)
- [ ] Status/turn info is in a DOM container with squiggle border (not canvas text)
- [ ] Active player card has `#bdeeff` background highlight
- [ ] `pnpm format` passes (run from repo root)

## Reference implementations

Study these completed visualizers for acceptable code patterns:

- OpenSpiel Chess
- ConnectX

## Troubleshooting

**Replay has only 2-3 steps / "INVALID ACTION DETECTED":** The built-in `"random"` agent needs `includeLegalActions: True` in the configuration to function. Without it, the agent cannot determine which actions are legal. Generate the replay with:
```python
env = make('open_spiel_$GAME_NAME', debug=True, configuration={'includeLegalActions': True})
```

**"Game not found" when running `make()`:** Ensure the game string is in `GAMES_LIST` and spelled correctly. Verify with:
```bash
.venv/bin/python -c "import pyspiel; print(pyspiel.load_game('$GAME_NAME'))"
```

**Observation string is empty:** Some games use `information_state_string()` instead of `observation_string()`. The framework handles this automatically -- check the game type in the OpenSpiel source for `provides_observation_string` vs `provides_information_state_string`.

**Canvas is blank:** Check the browser console for errors. Common issues: incorrect CSS (canvas has 0 height), parse function returning null because the observation string format doesn't match expectations. Print the raw observation string to debug.

**Game requires list parameters:** OpenSpiel uses semicolons inside square brackets for lists: `ship_sizes=[2;3;4]`, `ship_values=[1.0;1.0;1.0]`. These go directly in the game string in `GAMES_LIST`.
