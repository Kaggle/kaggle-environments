# Create / Update an Environment

## Step 1: Create the directory and files

Create `kaggle_environments/envs/<name>/` with:

```
<name>/
├── __init__.py           # empty file
├── <name>.json           # specification
├── <name>.py             # interpreter, renderer, html_renderer, specification, agents
└── agents.py             # agent implementations (optional, can inline in <name>.py)
```

Registration is automatic -- `kaggle_environments/__init__.py` discovers all directories under `envs/` at import time by importing `envs.<name>.<name>` and reading its module-level attributes.

## Step 2: Write the specification (`<name>.json`)

Top-level keys:

| Key | Required | Description |
|-----|----------|-------------|
| `name` | yes | Environment identifier (e.g., `"hangman"`) |
| `title` | yes | Human-readable name (e.g., `"Hangman"`) |
| `description` | yes | Game explanation |
| `version` | yes | Semver string (e.g., `"1.0.0"`) |
| `agents` | yes | Array of valid agent counts, e.g., `[2]` or `[1, 2, 4]` |
| `configuration` | yes | Object of game config properties (JSON Schema fields) |
| `observation` | yes | Per-agent observation schema |
| `action` | yes | Schema for agent actions |
| `reward` | yes | Schema for reward values |

### Configuration

Each property is a JSON Schema field with `type`, `default`, `description`, and optionally `minimum`, `maximum`, `enum`.

Standard framework fields inherited from `schemas.json` (do NOT redefine these):
- `episodeSteps` (default: 1000) -- max steps per episode
- `actTimeout` (default: 6) -- seconds per agent action
- `runTimeout` (default: 1200) -- max episode runtime in seconds

Add game-specific fields as additional properties.

### Observation

Each property is a JSON Schema field. Special modifiers:
- `shared: true` -- same value for all agents (stored on agent 0, copied to others). Use for board state, shared game state.
- `hidden: true` -- tracked in replay but not sent to agents at runtime.
- `default` -- single default for all agents.
- `defaults: [val0, val1, ...]` -- per-agent defaults by position (e.g., `"mark": {"defaults": [1, 2]}` gives agent 0 mark=1, agent 1 mark=2).

The framework auto-injects `step` (shared integer) and `remainingOverageTime` (per-agent float) into observations.

### Action

Can be a simple type (`integer`, `string`) or complex (`object`, `array`). Examples:
- Simple: `{"type": "integer", "default": 0, "minimum": 0, "maximum": 6}`
- Enum: `{"type": "string", "enum": ["NORTH", "SOUTH", "EAST", "WEST"], "default": "NORTH"}`
- Complex: `{"type": "object", "additionalProperties": {"enum": ["SPAWN", "NORTH"]}}`

### Reward

Schema with `default`. Examples:
- Win/lose: `{"enum": [-1, 0, 1], "default": 0}`
- Score: `{"type": "integer", "default": 0}`

### Reference specs

- `kaggle_environments/envs/rps/rps.json` -- simplest (2 agents, integer actions, simple rewards)
- `kaggle_environments/envs/connectx/connectx.json` -- board game with `shared` and `defaults`

## Step 3: Write the interpreter (`<name>.py`)

The main module must define these module-level attributes:

```python
import json
from os import path

dirpath = path.dirname(__file__)

# 1. Load specification from JSON file
specification = json.load(open(path.join(dirpath, "<name>.json")))


# 2. Interpreter: core game logic
def interpreter(state, env):
    """Called each step. Agent actions are already set on state by the framework."""
    # state is a list of Struct objects, one per agent
    # Access via dot notation: state[i].action, state[i].reward, etc.

    if env.done:
        return state

    # ... game logic here ...

    return state


# 3. Renderer: text representation
def renderer(state, env):
    """Return a text/ANSI string showing the current game state."""
    return "game state string"


# 4. HTML renderer: return compiled visualizer
def html_renderer():
    jspath = path.join(dirpath, "visualizer", "default", "dist", "index.html")
    if path.exists(jspath):
        with open(jspath, encoding="utf-8") as f:
            return f.read()
    return ""


# 5. Agents dict (optional)
from .agents import agents  # or define inline
```

### State structure

`state` is a list of agent dicts wrapped as `Struct` (dot-access). Each `state[i]` has:

| Field | Type | Description |
|-------|------|-------------|
| `state[i].action` | varies | Action submitted by agent i. Set by framework BEFORE interpreter is called. |
| `state[i].reward` | float/None | Cumulative reward. Interpreter modifies directly (e.g., `state[i].reward += 1`). |
| `state[i].status` | string | One of: `"INACTIVE"`, `"ACTIVE"`, `"DONE"`, `"ERROR"`, `"INVALID"`, `"TIMEOUT"`. |
| `state[i].observation` | Struct | Observation fields from spec (plus auto-injected `step`). |
| `state[i].info` | dict | Optional metadata. |

### `env` object

| Field | Description |
|-------|-------------|
| `env.configuration` | Game config as Struct (fields from spec's `configuration` section) |
| `env.done` | Boolean, true if game is already over |
| `env.steps` | List of all previous steps (for history) |

### Game lifecycle

1. **Reset:** Framework calls `interpreter(initial_state, env)` once. All agents start as `"INACTIVE"`.
2. **Each step:** Framework collects agent actions -> validates against schema (invalid -> `"INVALID"` status) -> calls `interpreter(state, env)`.
3. **Game over:** Interpreter sets all agent statuses to non-`"ACTIVE"` (typically `"DONE"`). The framework also auto-marks remaining `"ACTIVE"` agents as `"DONE"` when `episodeSteps` is reached.
4. **Error handling:** Agents with `"ERROR"`, `"INVALID"`, or `"TIMEOUT"` status automatically get `reward = None`.

### Common interpreter patterns

**Validate and penalize invalid actions:**
```python
if state[i].action < 0 or state[i].action >= max_val:
    state[i].status = "INVALID"
    state[i].reward = 0
    # Mark other agent as winner
    state[1 - i].status = "DONE"
    return state
```

**Accumulate rewards:**
```python
score = compute_score(state[0].action, state[1].action)
state[0].reward += score
state[1].reward -= score
```

**End the game:**
```python
if game_is_over:
    for i in range(len(state)):
        state[i].status = "DONE"
```

**Update observations:**
```python
state[0].observation.board = board  # shared fields (if marked shared in spec)
state[0].observation.lastOpponentAction = state[1].action
```

## Step 4: Write agents

Agent functions receive `(observation, configuration)` as Struct objects and return an action:

```python
def random_agent(observation, configuration):
    import random
    return random.randint(0, configuration.max_action)

def fixed_agent(observation, configuration):
    return 0

agents = {"random": random_agent, "fixed": fixed_agent}
```

## Step 5: Write tests

Create `tests/envs/<name>/test_<name>.py`:

```python
from kaggle_environments import make


def test_game_completes():
    env = make("<name>", configuration={"episodeSteps": 10})
    env.run([agent1, agent2])
    json = env.toJSON()
    assert json["statuses"] == ["DONE", "DONE"]


def test_rewards():
    env = make("<name>", configuration={"episodeSteps": 5})
    env.run([winning_agent, losing_agent])
    json = env.toJSON()
    assert json["rewards"][0] > json["rewards"][1]


def test_invalid_action():
    env = make("<name>")
    env.run([bad_agent, good_agent])
    json = env.toJSON()
    assert json["statuses"] == ["INVALID", "DONE"]
    assert json["rewards"] == [None, <winner_reward>]


def test_renderer():
    env = make("<name>", configuration={"episodeSteps": 3})
    env.run([agent1, agent2])
    output = env.render(mode="ansi")
    assert isinstance(output, str)
    assert len(output) > 0
```

Assert on `json["statuses"]`, `json["rewards"]`, `json["steps"]`, and `env.render(mode="ansi")`.

Run tests with:
```bash
uv sync && uv run pytest tests/envs/<name>/test_<name>.py -v
```

## Checklist

- [ ] `<name>.json` spec is valid JSON with all required top-level keys
- [ ] `<name>.py` exports `specification`, `interpreter`, `renderer`, `html_renderer`
- [ ] Interpreter handles: normal play, invalid actions, game-over conditions
- [ ] Rewards are set correctly for all outcomes (win/lose/draw/invalid)
- [ ] `__init__.py` exists (can be empty)
- [ ] Tests cover: normal completion, rewards, invalid actions, renderer output
- [ ] `uv run ruff check --fix . && uv run ruff format .` passes
