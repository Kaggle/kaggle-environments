# Create / Update an Environment

## Step 0: Ask whether this is a Game Arena environment

**Before writing any code, ask the user** — the answer changes how failures and
illegal moves must be scored (Step 3b), and there is no way to infer it from the
game rules alone.

```
AskUserQuestion({
  questions: [{
    question: "Is this a Game Arena environment (LLM-vs-LLM, results feed the Elo leaderboard)?",
    header: "Env type",
    multiSelect: false,
    options: [
      {label: "Game Arena",
       description: "Agents are language models behind a harness; episodes are scored into a cross-model Elo rating. Requires the failure semantics in Step 3b."},
      {label: "Regular",
       description: "Classic Kaggle simulation competition or a standalone env. Per-agent INVALID/ERROR statuses are fine; use the framework defaults."},
    ]
  }]
})
```

Signals that it is Game Arena, worth mentioning when you ask: the env ships a
`harness.py`, agents receive natural-language prompts, or the user talks about
comparing models rather than comparing submissions.

If **Regular**, skip Step 3b and use the plain patterns in Step 3.
If **Game Arena**, Step 3b is mandatory.

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

**Validate and penalize invalid actions** (regular envs only -- Game Arena envs
must use Step 3b instead):
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

## Step 3b: Failure and illegal-move handling (Game Arena environments)

Skip this section for regular environments.

Game Arena episodes are scored into a **cross-model Elo leaderboard**, so every
episode either produces a trustworthy result or must produce none at all. That
forces a distinction the framework does not make for you:

| What happened | Framework status | Correct outcome |
|---|---|---|
| Agent process raised, or the model provider errored | `ERROR` | **Void the episode.** Not a game result. |
| Agent exceeded `actTimeout` | `TIMEOUT` | **Void the episode.** Same as above. |
| Agent returned a well-formed action that breaks the rules | `INVALID`, or your own rule check | **Scored forfeit.** Offender loses, opponent wins. |

The reason for the split: a crash or a timeout is a *broken participant*, not a
model playing badly. Scoring it as a loss injects infrastructure flakiness into
the ratings -- a model on a slow provider would rank below one on a fast
provider for reasons that have nothing to do with gameplay. An illegal move is
the opposite: the model *did* play, it just played badly, and failing to follow
the action format is a genuine capability signal that belongs in the rating.

`open_spiel_env` with `strictMode: false` (the default) is the reference
implementation -- see `open_spiel_env.py`, the `agent_error` / `invalid_action`
branches. `word_association` and `word_art` implement the same semantics.
**New Game Arena environments must match it.** Do not add a config flag for
this; the behavior is not per-env negotiable.

### The pattern

Define a module-level forfeit reward matching open_spiel's convention:

```python
# Statuses core.py assigns when an agent crashes or times out, as opposed to
# returning a well-formed-but-illegal action.
_FRAMEWORK_FAILURE_STATUSES = ("ERROR", "TIMEOUT")

# Reward applied to the forfeiting side. The opponent receives the negation.
DEFAULT_INVALID_ACTION_REWARD = -1
```

**Void on crash/timeout.** Force every seat to `ERROR`, except seats already
`TIMEOUT` (which voids the episode identically and is more informative in the
replay). `core.py` then nulls all rewards automatically:

```python
def _abort_on_agent_failure(state):
    """Void the episode if the framework marked any seat ERROR or TIMEOUT.
    Returns True if the episode was ended."""
    if not any(s.status in _FRAMEWORK_FAILURE_STATUSES for s in state):
        return False
    for s in state:
        if s.status != "TIMEOUT":
            s.status = "ERROR"
    return True
```

**Forfeit on an illegal move.** Every seat ends `DONE` -- including the
offender -- so the episode scores normally:

```python
def forfeit(state, offending_seat):
    for i in range(len(state)):
        state[i].status = "DONE"
        state[i].reward = (
            DEFAULT_INVALID_ACTION_REWARD
            if i == offending_seat
            else -DEFAULT_INVALID_ACTION_REWARD
        )
```

Never leave a seat in `INVALID` at episode end. `core.py` nulls the reward of
any `ERROR`/`INVALID`/`TIMEOUT` agent, so an `INVALID` terminal status is
indistinguishable from a crash downstream -- which is exactly the collapse this
section exists to prevent.

### Wiring it into the interpreter

Check for framework failures **before** your action-processing code runs, so a
crash cannot be laundered into a scored forfeit:

```python
def interpreter(state, env):
    if env.done:
        return state

    # A crashed or timed-out seat is a broken participant, not a player
    # making an illegal move. Void before process_action can rescore it.
    if _abort_on_agent_failure(state):
        return state

    forfeited = process_action(state, env.configuration)
    ...
```

### Env-shape adjustments

The two rules above are fixed; how they map onto your env is not. Decide these
explicitly and write the reasoning into a docstring:

* **Team games.** Scope the forfeit to the offender's *team*, not the lone
  seat -- crediting the offender's own partner would reward a team for its own
  foul. `word_association` (2v2) gives both offending seats
  `DEFAULT_INVALID_ACTION_REWARD` and both opponents its negation.
* **Multi-game episodes.** A forfeit is terminal for the whole episode. Return
  a flag from your action processor and gate the next-game rollover on it;
  otherwise the rollover resets the forfeiting seats to `ACTIVE` and overwrites
  the forfeit rewards. (`word_association` had exactly this bug.)
* **Running-score rewards.** If `reward` is a cumulative point total rather
  than a win/loss value, overwriting it with a flat ±1 discards every completed
  round. `word_art` deliberately diverges here: on `INVALID` it ends all seats
  `DONE` and **keeps the accumulated points**. Crash/timeout still voids.

### Visualizers

A visualizer that computes terminal state as `status === 'DONE'` will silently
render nothing on a voided episode. Handle `ERROR`/`TIMEOUT` explicitly and show
why the episode was voided:

```ts
const isVoided = step.some((p) => p?.status === 'ERROR' || p?.status === 'TIMEOUT');
const isGameOver = step.every((p) => p?.status === 'DONE') || isVoided;
```

### Required tests

```python
@pytest.mark.parametrize("crash_seat", range(NUM_AGENTS))
def test_agent_crash_voids_the_episode(crash_seat):
    def crash(observation, configuration):
        raise RuntimeError("provider exploded")
    agents = [legal_agent] * NUM_AGENTS
    agents[crash_seat] = crash
    env = make("<name>")
    env.run(agents)
    assert [s.status for s in env.state] == ["ERROR"] * NUM_AGENTS
    assert [s.reward for s in env.state] == [None] * NUM_AGENTS
```

Cover, at minimum:
- crash on every seat -> all `ERROR`, all rewards `None`
- `DeadlineExceeded` on every seat (from `kaggle_environments.errors`) -> the
  offending seat keeps `TIMEOUT`, the rest are `ERROR`, all rewards `None`
- an illegal-but-well-formed action -> all `DONE` with ±1 rewards, never `None`
- if the env supports multi-game episodes: a forfeit does not start a new game

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
    # Regular envs only. Game Arena envs assert the Step 3b shape instead:
    #   statuses == ["DONE", "DONE"], rewards == [-1, 1]
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

## Step 6: Add a visualizer (optional)

Follow the `create-visualizer` skill to build a web-based replay visualizer for this environment.

## Checklist

- [ ] Asked the user whether this is a Game Arena or regular environment (Step 0)
- [ ] `<name>.json` spec is valid JSON with all required top-level keys
- [ ] `<name>.py` exports `specification`, `interpreter`, `renderer`, `html_renderer`
- [ ] Interpreter handles: normal play, invalid actions, game-over conditions
- [ ] Rewards are set correctly for all outcomes (win/lose/draw/invalid)
- [ ] `__init__.py` exists (can be empty)
- [ ] Tests cover: normal completion, rewards, invalid actions, renderer output
- [ ] `uv run ruff check --fix . && uv run ruff format .` passes

Game Arena environments additionally:

- [ ] Crash/timeout voids the episode (all seats `ERROR`/`TIMEOUT`, all rewards `None`)
- [ ] Illegal move is a scored forfeit (all seats `DONE`, ±`DEFAULT_INVALID_ACTION_REWARD`)
- [ ] No seat can end an episode in `INVALID`
- [ ] Failure check runs before action processing in the interpreter
- [ ] Team scoping, multi-game terminality, and running-score handling decided and documented
- [ ] Visualizer renders a voided episode instead of silently showing nothing
- [ ] Per-seat crash and timeout tests exist, plus an illegal-move forfeit test
- [ ] No config flag was added to make this behavior optional
