# Onboard a New OpenSpiel Game

The OpenSpiel integration (`kaggle_environments/envs/open_spiel_env/`) provides a unified framework that wraps games from Google's [OpenSpiel library](https://github.com/google-deepmind/open_spiel) into Kaggle environments. A shared interpreter handles all games -- you do NOT write a per-game interpreter. Instead, you configure the game and optionally add a proxy, visualizer, or custom game implementation.

## Determine the approach

Before starting, determine which pattern fits your game:

| Situation | Approach | Files to create |
|-----------|----------|-----------------|
| Game already exists in OpenSpiel (`pyspiel.registered_games()`) and default observation strings are fine | **Add to GAMES_LIST only** | None (just edit `open_spiel_env.py`) |
| Game exists in OpenSpiel but you want better observation format (e.g., JSON instead of OpenSpiel strings) | **Add proxy** | `games/<name>/<name>_proxy.py` |
| Game does NOT exist in OpenSpiel and needs a custom Python implementation | **Add custom game** | `games/<name>/<name>_game.py` |

All three approaches can optionally include a **visualizer** and/or **support files** (openings, presets).

## Step 1: Add to GAMES_LIST

Edit `kaggle_environments/envs/open_spiel_env/open_spiel_env.py` and add your game string to the `GAMES_LIST` array (around line 848):

```python
GAMES_LIST = [
    "backgammon",
    "chess",
    # ...existing games...
    "your_game",                              # simple game
    "your_game(board_size=9,variant=foo)",     # game with parameters
]
```

The game string format is: `"<short_name>"` or `"<short_name>(param1=val1,param2=val2)"`.

The environment will be registered as `open_spiel_<short_name>` and accessible via `make("open_spiel_<short_name>")`.

The framework automatically:
- Loads the game via `pyspiel.load_game(game_string)`
- Sets `episodeSteps` to `game.max_history_length() + 100`
- Determines observation type (observation string vs information state string)
- Generates a specification with the game's player count, parameters, etc.
- Provides a built-in `"random"` agent
- Falls back to a default text-based HTML renderer if no custom one exists

## Step 2 (optional): Create a proxy

If you need custom observation/action representations (e.g., returning JSON board state instead of OpenSpiel's default string format), create a proxy.

Create `kaggle_environments/envs/open_spiel_env/games/<name>/<name>_proxy.py`:

```python
"""Custom state/action representations for <Name>."""

import json
from typing import Any

import pyspiel

from ... import proxy


class <Name>State(proxy.State):
    """Wraps OpenSpiel state with custom representations."""

    def state_dict(self) -> dict[str, Any]:
        """Parse the underlying state into a game-specific dict."""
        # self.__wrapped__ gives access to the underlying pyspiel.State
        # self.to_string() returns the OpenSpiel default string representation
        return {
            "board": ...,  # parse from self.__wrapped__
            "current_player": ...,
            "is_terminal": self.is_terminal(),
            "winner": ...,
        }

    def to_json(self) -> str:
        return json.dumps(self.state_dict())

    def observation_string(self, player: int) -> str:
        """Override to return JSON instead of default OpenSpiel string."""
        return self.to_json()

    def __str__(self):
        return self.to_json()


class <Name>Game(proxy.Game):
    """Wraps the OpenSpiel game."""

    def __init__(self, params: Any | None = None):
        params = params or {}
        wrapped = pyspiel.load_game("<name>", params)
        super().__init__(
            wrapped,
            short_name="<name>_proxy",
            long_name="<Name> (proxy)",
        )

    def new_initial_state(self, *args) -> <Name>State:
        return <Name>State(self.__wrapped__.new_initial_state(*args), game=self)


# Register the proxy with OpenSpiel (REQUIRED -- must be at module level)
pyspiel.register_game(<Name>Game().get_type(), <Name>Game)
```

Also create an empty `games/<name>/__init__.py`.

**How discovery works:** The framework auto-imports all `*_proxy.py` files from the `games/` directory via glob at module load time. When `_build_env()` encounters a game whose `short_name` has a matching proxy file at `games/<short_name>/<short_name>_proxy.py`, it loads the proxy version instead.

**Key proxy base classes** (from `proxy.py`):
- `proxy.State` wraps `pyspiel.State` -- all methods delegate to `self.__wrapped__` by default. Override `observation_string()`, `__str__()`, etc. to customize. `__getattr__` falls through to the wrapped state for any method you don't override.
- `proxy.Game` wraps `pyspiel.Game` -- override `new_initial_state()` to return your custom State class.

**Reference:** See `games/connect_four/connect_four_proxy.py` (JSON board/actions) and `games/go/go_proxy.py`.

## Step 2 (alternative): Create a custom game

If the game doesn't exist in OpenSpiel at all, implement it from scratch as a pyspiel-compatible Python game.

Create `kaggle_environments/envs/open_spiel_env/games/<name>/<name>_game.py`:

```python
"""Custom OpenSpiel game implementation for <Name>."""

import numpy as np
import pyspiel

_NUM_PLAYERS = 2

_GAME_TYPE = pyspiel.GameType(
    short_name="<name>",
    long_name="<Name>",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,        # or SIMULTANEOUS
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC, # or EXPLICIT_STOCHASTIC
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,            # or GENERAL_SUM
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,    # or REWARDS
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={
        # Default parameter values (overridable via game string)
        "board_size": 8,
    },
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=64,        # total number of possible actions
    max_chance_outcomes=0,           # 0 for deterministic games
    num_players=_NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,                 # for zero-sum games
    max_game_length=200,
)


class <Name>Game(pyspiel.Game):
    def __init__(self, params=None):
        # Read parameters from game string
        self.board_size = params.get("board_size", 8) if params else 8
        # Rebuild game_info if it depends on parameters
        game_info = pyspiel.GameInfo(
            num_distinct_actions=self.board_size * self.board_size,
            max_chance_outcomes=0,
            num_players=_NUM_PLAYERS,
            min_utility=-1.0,
            max_utility=1.0,
            utility_sum=0.0,
            max_game_length=self.board_size * self.board_size,
        )
        super().__init__(_GAME_TYPE, game_info, params or dict())

    def new_initial_state(self):
        return <Name>State(self)

    def make_py_observer(self, params=None):
        return <Name>Observer(params, self.board_size)


class <Name>State(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self._game = game
        self._is_terminal = False
        self._current_player = 0
        self._returns = [0.0] * _NUM_PLAYERS
        # ... initialize game-specific state

    def current_player(self):
        if self._is_terminal:
            return pyspiel.PlayerId.TERMINAL
        return self._current_player

    def _legal_actions(self, player=None):
        """Return list of legal action integers."""
        if self._is_terminal:
            return []
        return [...]  # game-specific legal actions

    def _apply_action(self, action):
        """Apply action and update game state."""
        # ... game logic
        # Set self._is_terminal = True when game ends
        # Set self._returns when game ends

    def is_terminal(self):
        return self._is_terminal

    def returns(self):
        return self._returns

    def __str__(self):
        """String representation (used as observation if no observer)."""
        return "..."  # game board as string


class <Name>Observer:
    """Observation as tensor for ML agents."""
    def __init__(self, params, board_size):
        self.board_size = board_size
        shape = (board_size, board_size)
        self.tensor = np.zeros(np.prod(shape), np.float32)
        self.dict = {"observation": np.reshape(self.tensor, shape)}

    def set_from(self, state, player):
        """Fill tensor from state for given player."""
        # ... fill self.tensor / self.dict based on state

    def string_from(self, state, player):
        """String observation for given player."""
        return str(state)


# Register with OpenSpiel (REQUIRED -- must be at module level)
pyspiel.register_game(_GAME_TYPE, <Name>Game)
```

Also create an empty `games/<name>/__init__.py`.

**How discovery works:** The framework auto-imports all `*_game.py` files from `games/` via glob at module load time. The `pyspiel.register_game()` call makes the game available to `pyspiel.load_game("<name>")`, which `_build_env()` calls when processing `GAMES_LIST`.

**Reference:** See `games/snake/snake_game.py` for a complete example (sequential turns with simultaneous move processing, 1-4 players, parameterized board size).

## Step 3 (optional): Add support files

### Opening book

Create `games/<name>/openings.jsonl` with one JSON object per line:

```json
{"name": "King's Pawn", "initialActions": [2426, 1258], "fen": "...", "eco": "C20"}
```

Users enable this via `make("open_spiel_<name>", {"useOpenings": True, "seed": 42})`. The framework selects `openings[seed % len(openings)]` and applies the `initialActions` before play begins.

### Image config (chess-specific)

Create `games/<name>/image_config.jsonl` for visualization themes. Selected by seed.

### Preset hands (poker-specific)

Create `games/<name>/preset_hands.jsonl` for deterministic card dealing. Selected by seed.

## Step 4 (optional): Create a visualizer

Visualizers live at `games/<name>/visualizer/default/` within the pnpm workspace.

### Project structure

```
games/<name>/visualizer/default/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── index.html
└── src/
    ├── main.ts
    ├── renderer.ts
    └── style.css (optional)
```

### Boilerplate files

**`package.json`:**
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

**`vite.config.ts`:**
```typescript
import { defineConfig, mergeConfig } from "vite";
// Note: path depth is deeper than regular envs due to games/ subdirectory
import baseConfig from "../../../../../../../web/vite.config.base";

export default mergeConfig(baseConfig, defineConfig({}));
```

**`tsconfig.json`:**
```json
{
  "extends": "../../../../../../../web/tsconfig.base.json",
  "compilerOptions": {
    "allowJs": true
  },
  "include": ["src"]
}
```

Note the path depth: OpenSpiel visualizers are 2 levels deeper than regular env visualizers (`open_spiel_env/games/<name>/visualizer/default/` vs `<name>/visualizer/default/`), so the relative paths to `web/` use `../../../../../../../` instead of `../../../../../`.

**`index.html`:**
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

### Entry point (`src/main.ts`)

```typescript
import { createReplayVisualizer, ReplayAdapter } from "@kaggle-environments/core";
import { renderer } from "./renderer";

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
    gameName: "open_spiel_<name>",  // must match the registered env name
    renderer: renderer as any,
    ui: "side-panel",               // "side-panel" (with reasoning logs) or "inline"
  })
);
```

### Renderer (`src/renderer.ts`)

The renderer receives replay data. For OpenSpiel games, the raw step data comes from the unified interpreter and has this shape per step:

```typescript
// Each step in replay.steps is an array of player observations:
// replay.steps[stepIndex][playerIndex].observation.observationString
// replay.steps[stepIndex][playerIndex].action.submission
// replay.steps[stepIndex][playerIndex].reward
// replay.steps[stepIndex][playerIndex].status
```

If you added a proxy that returns JSON observation strings, parse them in the renderer:

```typescript
import type { RendererOptions } from "@kaggle-environments/core";

export function renderer(options: RendererOptions) {
  const { replay, parent, step } = options;
  const currentStep = replay.steps[step];

  // Parse JSON observation from proxy
  const obs = JSON.parse(currentStep[0].observation.observationString);
  const board = obs.board;

  // Create/update DOM in parent...
}
```

### Optional: Add a transformer

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

A transformer is not required -- simpler games can parse observation strings directly in the renderer.

## Step 5: Add tests

Add test cases to `tests/envs/open_spiel_env/test_open_spiel_env.py`. The tests use `absltest` (not pytest directly). Follow the existing patterns:

```python
def test_<name>_agent_playthrough(self):
    """Test that random agents can play a full game."""
    env = make("open_spiel_<name>", debug=True)
    env.run(["random", "random"])
    json = env.toJSON()
    self.assertEqual(json["name"], "open_spiel_<name>")
    self.assertTrue(all(status == "DONE" for status in json["statuses"]))

def test_<name>_manual_playthrough(self):
    """Test manual step-by-step play."""
    env = make("open_spiel_<name>", debug=True)
    env.reset()
    env.step([{"submission": -1}, {"submission": -1}])  # Setup step (always required)
    # Sequential game: only the current player submits, others send -1
    env.step([{"submission": 0}, {"submission": -1}])   # Player 0 acts
    env.step([{"submission": -1}, {"submission": 0}])   # Player 1 acts
    # ...continue until done...
    self.assertTrue(env.done)

def test_<name>_invalid_action(self):
    """Test that invalid actions are handled correctly."""
    env = make("open_spiel_<name>", debug=True)
    env.reset()
    env.step([{"submission": -1}, {"submission": -1}])  # Setup step
    env.step([{"submission": 999}, {"submission": -1}])  # Invalid action
    self.assertTrue(env.done)
    json = env.toJSON()
    self.assertEqual(json["rewards"][0], open_spiel_env.DEFAULT_INVALID_ACTION_REWARD)  # -1
```

**Key testing patterns:**
- Always include a setup step: `env.step([{"submission": -1}, ...])` as the first step after `env.reset()`.
- For **sequential** games: only the current player submits an action; others send `{"submission": -1}`.
- For **simultaneous** games: all players submit actions on every step.
- `DEFAULT_INVALID_ACTION_REWARD` is `-1` (player gets -1, opponent gets +1).
- `AGENT_ERROR_ACTION` is `-2` (signals agent internal error, both players get `None` rewards and `"ERROR"` status).

## Step 6: Verify

```bash
# Run the OpenSpiel tests
uv sync && uv run pytest tests/envs/open_spiel_env/test_open_spiel_env.py -v -k "<name>"

# Quick smoke test
uv run python -c "
from kaggle_environments import make
env = make('open_spiel_<name>', debug=True)
env.run(['random', 'random'])
print(env.toJSON()['statuses'], env.toJSON()['rewards'])
"

# If visualizer was added
pnpm install && pnpm dev  # select your game from the picker

# Lint
uv run ruff check --fix . && uv run ruff format .
```

## Checklist

- [ ] Game string added to `GAMES_LIST` in `open_spiel_env.py`
- [ ] If proxy: `games/<name>/<name>_proxy.py` created with `pyspiel.register_game()` call at module level
- [ ] If custom game: `games/<name>/<name>_game.py` created with `pyspiel.register_game()` call at module level
- [ ] `games/<name>/__init__.py` exists (can be empty)
- [ ] `make("open_spiel_<name>")` loads without error
- [ ] Random agent playthrough completes with `"DONE"` statuses
- [ ] Invalid action handling works correctly
- [ ] Tests added to `test_open_spiel_env.py`
- [ ] If visualizer: correct relative paths to `web/` configs (7 levels deep)
- [ ] If transformer: registered in `web/core/src/transformers.ts` switch statements
- [ ] Linting passes

## Reference files

- `kaggle_environments/envs/open_spiel_env/open_spiel_env.py` -- main framework, interpreter, GAMES_LIST, registration
- `kaggle_environments/envs/open_spiel_env/proxy.py` -- base proxy classes (State, Game)
- `kaggle_environments/envs/open_spiel_env/games/connect_four/connect_four_proxy.py` -- proxy example
- `kaggle_environments/envs/open_spiel_env/games/go/go_proxy.py` -- proxy example
- `kaggle_environments/envs/open_spiel_env/games/snake/snake_game.py` -- custom game example
- `kaggle_environments/envs/open_spiel_env/games/connect_four/visualizer/default/` -- visualizer example
- `web/core/src/transformers.ts` -- transformer registry
- `web/core/src/transformers/connect_four/` -- simple transformer example
- `tests/envs/open_spiel_env/test_open_spiel_env.py` -- all existing tests
- [OpenSpiel documentation](https://github.com/google-deepmind/open_spiel) -- game types, API reference
