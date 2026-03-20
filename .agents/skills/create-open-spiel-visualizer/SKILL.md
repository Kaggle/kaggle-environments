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
  background: #28303F;
  overflow: hidden;
  font-family: 'Inter', sans-serif;
  box-sizing: border-box;
  padding: 12px;
}

.renderer-container canvas {
  position: relative;
  flex-grow: 1;
  width: 100%;
  min-height: 0;
}

.header {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  padding: 8px 0;
  color: white;
  font-size: 1.1rem;
  font-weight: 600;
  flex-shrink: 0;
  gap: 16px;
}

.status-pill {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 5px 16px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
  color: black;
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

All OpenSpiel visualizers should match the "paper-like and playful" aesthetic used by the chess v2 and go v2 visualizers. The goal is a warm, neutral look that harmonizes with the surrounding UI (e.g., reasoning log panels) rather than standing out with dark, saturated backgrounds.

### Aesthetic principles

1. **Paper-like, not dark-mode.** Use warm neutral backgrounds (`#28303F` or transparent) instead of saturated dark blues (`#1a1a2e`). Status areas should be white with subtle shadows -- like cards or pills floating on the page.

2. **Playful, not sterile.** Use generous border-radius (`8px` for containers, `22-32px` for pills/badges), soft box-shadows, and warm board colors. Avoid harsh borders and sharp edges.

3. **High-resolution text.** Prefer **DOM elements** for all text, labels, and status displays rather than canvas text. Canvas `fillText` renders at device pixel ratio and cannot use web fonts reliably. Use canvas only for the game board/grid itself. Wrap the canvas in a flex container alongside DOM-based status elements.

4. **Inter font.** All text must use the Inter font family. Load it via Google Fonts in `style.css`:
   ```css
   @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap');
   ```
   Apply it to the renderer container: `font-family: 'Inter', sans-serif;`

5. **Responsive sizing.** Use `rem`-based font sizes (`0.8rem`, `1rem`, `1.1rem`) rather than pixel-based canvas text sizes. Check `window.innerWidth < 768` for mobile breakpoints when needed.

### Color palette

| Element | Color | Notes |
|---------|-------|-------|
| Container background | `#28303F` | Neutral dark slate, not saturated blue |
| Player 1 accent | `#4fc3f7` | Turn indicator, active player glow |
| Player 2 accent | `#ff8a65` | Turn indicator, active player glow |
| Active player highlight | `#20BEFF` | Bright blue for current turn border/glow |
| Winner highlight | `#FFEB70` | Gold for winner border/glow |
| Status container bg | `white` | Paper-like status pills |
| Status text | `black` | Dark text on white containers |
| Dim text | `#666` | Secondary labels |
| Info card bg | `rgba(32, 33, 36, 0.70)` | Semi-transparent dark cards for overlays |
| Info card text | `#E8EAED` | Light text on dark cards |
| Stats pill bg | `#3C4043` | Dark pill for stats/badges |
| Positive delta | `#66bb6a` | Score gain |
| Negative delta / capture | `#ef5350` | Score loss, piece removal |
| Last move highlight | `#ffd700` (gold) | Move highlight ring/overlay |
| Board light square | `#f0d9b5` | Warm tan (for chess-like boards) |
| Board dark square | `#b58863` | Warm brown (for chess-like boards) |

### Rendering approach

Use a **hybrid DOM + canvas** architecture:

- **Canvas**: game board grid, pieces, move highlights, board decorations
- **DOM**: header/title, player names, status text, score displays, turn indicators, game-over messages

This gives you crisp, font-rendered text alongside flexible canvas drawing for game-specific visuals.

```
+------------------------------------------+
|  [DOM] Header: Player names, title        |
+------------------------------------------+
|                                          |
|  [Canvas] Game board                     |
|                                          |
+------------------------------------------+
|  [DOM] Status pill: turn / game over     |
+------------------------------------------+
```

### Status container pattern

Use white, rounded containers with soft shadows for status information:

```typescript
const statusContainer = document.createElement('div');
Object.assign(statusContainer.style, {
  padding: '5px 12px',
  backgroundColor: 'white',
  borderRadius: '8px',
  boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06)',
  textAlign: 'center',
  minWidth: '200px',
  marginTop: '10px',
  fontFamily: "'Inter', sans-serif",
});
```

### Active player / winner indication

Use border color and glow on player info containers:

```css
/* Active player */
border-color: #20BEFF;
box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4), 0 0 20px rgba(32, 190, 255, 0.5);

/* Winner */
border-color: #FFEB70;
box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4), 0 0 20px rgba(255, 235, 112, 0.6);
```

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

- Show a **DOM header** at the top with the current player's name and a colored accent
- Use consistent player colors: **Player 1 = `#4fc3f7` (blue)**, **Player 2 = `#ff8a65` (orange)**
- Highlight the active player with `#20BEFF` border/glow (see style guide)
- On game over, show the result with the winner's `#FFEB70` gold highlight

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
- At game over, display the final result prominently in the status pill

### Handling game phases

Some games have distinct phases (e.g., Battleship has ship placement then war). Detect the phase from the observation state and adjust the display accordingly:
- Show phase name in the header or status pill
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
      <div class="status-pill"></div>
    </div>
  `;
  const container = parent.querySelector('.renderer-container') as HTMLDivElement;
  const header = parent.querySelector('.header') as HTMLDivElement;
  const canvas = parent.querySelector('canvas') as HTMLCanvasElement;
  const statusPill = parent.querySelector('.status-pill') as HTMLDivElement;
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
    statusPill.textContent = 'Waiting for game data...';
    return;
  }

  const terminal = isTerminal(currentStep);
  const cp = getCurrentPlayer(currentStep);

  // --- 1. Build header (DOM) ---
  // Player names with accent colors, game title
  header.innerHTML = `
    <span style="color: #4fc3f7; font-weight: 700;">Player 1</span>
    <span style="color: #9ca3af;">vs</span>
    <span style="color: #ff8a65; font-weight: 700;">Player 2</span>
  `;

  // --- 2. Draw game board on canvas ---
  // Clear canvas with neutral background
  c.fillStyle = '#28303F';
  c.fillRect(0, 0, width, height);
  // ... draw board, pieces, move highlights ...

  // --- 3. Update status pill (DOM) ---
  if (terminal) {
    const rewards = getRewards(currentStep);
    let msg = 'Game Over -- Draw';
    if (rewards[0] > rewards[1]) msg = 'Game Over -- Player 1 wins!';
    else if (rewards[1] > rewards[0]) msg = 'Game Over -- Player 2 wins!';
    statusPill.textContent = msg;
    statusPill.style.fontWeight = '700';
  } else {
    statusPill.textContent = `Player ${cp + 1}'s turn`;
  }
}
```

### Layout guidelines

- **Header** (DOM): player names and title at the top, flexbox centered with gap
- **Canvas**: game board centered in the remaining flex space, scaled responsively
- **Status pill** (DOM): white rounded container at the bottom with game state text
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
- Unused variables/parameters: prefix with `_` (e.g., `_playerIdx`) or remove them
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
- [ ] Score / game progress is visible in the header or status pill
- [ ] Game over state displays the winner with correct colors
- [ ] Board scales responsively (try resizing the window)
- [ ] Background is `#28303F` (not `#1a1a2e`)
- [ ] Text uses Inter font (loaded via Google Fonts import in style.css)
- [ ] Status/turn info is in a white DOM pill (not canvas text)
- [ ] `pnpm format` passes (run from repo root)

## Reference implementations

Study these completed visualizers for patterns:

| Game | Type | Observation format | Key rendering patterns |
|------|------|-------------------|----------------------|
| `games/oware/` | Perfect info | `"player \| score0 score1 \| s0 s1 ... s11"` | Pit/seed board, seed count deltas, capture detection, score panel |
| `games/nim/` | Perfect info | `"(player): pile0 pile1 ..."` | Stone columns, recently-removed stones (red X), pile change badges |
| `games/breakthrough/` | Perfect info | Multi-line grid with `b`/`w`/`.` | Chess-like board, move-from/to highlighting, capture markers |
| `games/dots_and_boxes/` | Perfect info | Unicode box-drawing characters | Line ownership tracking across all steps, score deltas, line glow |
| `games/pentago/` | Perfect info | Multi-line grid with `O`/`@`/`.` | Go-like board, rotation detection, quadrant highlighting |
| `games/battleship/` | Imperfect info | Two grids per player (`+---+` borders, letter ships, `#`/`@`/`*` markers) | Dual-board layout (ships + shots per player), per-player observation parsing, phase detection |

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
