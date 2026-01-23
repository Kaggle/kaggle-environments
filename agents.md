# Agent Architecture in kaggle-environments

This document explains how agents work in kaggle-environments, including the multi-container deployment workflow used in production.

## Overview

An "agent" in kaggle-environments is any entity that can produce actions for a game environment. Agents can be:

- **Local functions**: Python callables defined in code
- **Built-in agents**: Named agents registered with an environment (e.g., `"reaction"` for RPS)
- **File paths**: Python scripts that define an agent function
- **HTTP URLs**: Remote agents running on separate servers
- **Protobuf agents**: Remote agents using the protobuf protocol

## Agent Initialization Flow

Agent initialization is now explicit and separated from execution:

1. **Agent specifications** are passed to `env.run()` or `env.initialize_agents()`
2. **`initialize_agents()`** converts specs into `Agent` objects via `build_agent()`
3. **`__agent_runner()`** receives pre-initialized `Agent` objects (not raw specs)
4. **Agent execution** happens via the runner's `act()` method

This separation makes the code more readable and eliminates complex method signatures.

## Agent Types and Resolution

When an agent specification is passed to `env.initialize_agents()`, the `build_agent()` function in `agent.py` resolves it in this order:

1. **Protobuf agent spec**: Dict with `type='proto'` or `proto_config` key
2. **Built-in agent**: String matching a registered agent name
3. **Callable**: Already a Python function/callable
4. **Static action**: Non-string, non-callable value (returned as-is)
5. **HTTP URL**: String starting with `http://` or `https://` → creates `UrlAgent`
6. **File path**: String path to a Python file → loads and executes the file

## Key Classes

### `Agent` (agent.py)

The main wrapper class that:
- Takes a raw agent specification and an environment
- Uses `build_agent()` to resolve the specification into a callable
- Provides `act(observation)` method that calls the underlying agent

```python
class Agent:
    def __init__(self, raw: str | Callable | Any, environment: Any)
    def act(self, observation: Any) -> Tuple[Any, Dict[str, Any]]
```

### `UrlAgent` (agent.py)

Handles remote agents via HTTP JSON requests:
- Sends POST requests with `action: "act"` to the remote server
- The remote server must have a pre-loaded agent (see Multi-Container Workflow below)
- Does NOT include an agent path in requests - relies on cached agent on server

```python
class UrlAgent:
    def __init__(self, raw: str, environment_name: str)
    def __call__(self, observation: Any, configuration: Any) -> Any
```

## Multi-Container Deployment Workflow

In production, Kaggle runs games across multiple containers.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR CONTAINER                       │
│  Started with:                                                       │
│    kaggle-environments run --environment <env> \                     │
│      --agents http://agent-1:8081 http://agent-2:8081 \             │
│      --configuration '...' --out /path/out.json --log /path/log.json│
│                                                                      │
│  Flow:                                                               │
│    1. action_run() calls env.run(agents)                            │
│    2. env.run() calls env.initialize_agents(agents)                 │
│    3. initialize_agents() creates Agent objects (UrlAgent for URLs) │
│    4. UrlAgent sends 'act' requests to agent containers             │
└─────────────────────────────────────────────────────────────────────┘
         │                                    │
         │ HTTP POST (act)                    │ HTTP POST (act)
         ▼                                    ▼
┌─────────────────────────┐      ┌─────────────────────────┐
│   AGENT-1 CONTAINER     │      │   AGENT-2 CONTAINER     │
│                         │      │                         │
│  Started with:          │      │  Started with:          │
│    kaggle-environments  │      │    kaggle-environments  │
│      http-server        │      │      http-server        │
│      --host 0.0.0.0     │      │      --host 0.0.0.0     │
│      --port 8081        │      │      --port 8081        │
│                         │      │                         │
│  Agent pre-loaded via   │      │  Agent pre-loaded via   │
│  initial 'act' request  │      │  initial 'act' request  │
│  with agents=[path]     │      │  with agents=[path]     │
└─────────────────────────┘      └─────────────────────────┘
```

### Step-by-Step Flow in Production

1. **Agent containers start** with `http-server` command (no agent loaded yet):
   ```bash
   kaggle-environments http-server --host 0.0.0.0 --port 8081
   ```

2. **Orchestrator starts** with agent HTTP URLs provided by the backend:
   ```bash
   kaggle-environments run --environment my_game \
     --agents http://agent-1:8081 http://agent-2:8081 ...
   ```

3. **Agents are initialized** via an explicit `initialize_agents` request:
   ```json
   {
     "action": "initialize_agents",
     "environment": "my_game",
     "agents": ["path/to/agent.py"],
     "configuration": {...}
   }
   ```
   This triggers `action_initialize_agents()` in `main.py` which:
   - Creates an `Agent` object from the path
   - Caches it in the global `cached_agent` variable
   - Returns `{"status": "initialized"}`


4. **Game runs**: `UrlAgent` sends `act` requests to each agent container.
   These requests do NOT include an agent path - they rely on the cached agent:
   ```json
   {
     "action": "act",
     "environment": "my_game",
     "state": {"observation": {...}},
     "configuration": {...}
   }
   ```

5. **Agent containers respond** using their cached agent to compute actions.

### Explicit Initialization Benefits

The new explicit initialization approach provides:
- **Clear separation** between initialization and execution
- **Simpler signatures**: `__agent_runner()` accepts `list[Agent | None]` instead of `list[str | Callable | Agent | None]`
- **Better readability**: No hidden initialization in the first `act()` call
- **Easier debugging**: Initialization errors are caught early and separately

### Why Explicit Initialization is Required

`UrlAgent` does not include an `agents` field in its requests. The `action_act()` function in `main.py` requires a pre-initialized agent in the cache. The `initialize_agents` action must be called first to load the agent.

This design allows:
- Agent code to be loaded once and reused across many game steps
- Clear separation between initialization and execution phases
- The orchestrator to only know agent URLs, not their implementation details
- Simpler code with explicit initialization flow

## Local Testing vs Production

### Local Testing (Single Process)

`uv run pytest /tests` is sufficient for most local testing.

### Multi-Container Testing

See `tests/integration/test_multicontainer.py` for the test setup that emulates production:
1. Docker containers run `http-server` for each agent
2. Test initializes agents via `initialize_agents` action
3. Test sends `evaluate` or `run` request to orchestrator with agent URLs

Always use `./run_tests.sh --multicontainer` to run tests in multi-container mode.

## Key Files

| File | Purpose |
|------|---------|
| `agent.py` | Agent resolution, `Agent` class, `UrlAgent` for HTTP agents |
| `main.py` | CLI entry point, `action_initialize_agents()` and `action_act()` for agent lifecycle |
| `core.py` | `Environment.initialize_agents()` and `__agent_runner()` for agent orchestration |

## Agent Caching in http-server Mode

The `main.py` module maintains a global `cached_agent` variable:

```python
cached_agent: Agent | None = None

def action_initialize_agents(args: Any) -> dict[str, Any]:
    """Explicitly initialize and cache an agent."""
    global cached_agent
    raw = args.agents[0]
    env = make(args.environment, args.configuration, args.info, debug=args.debug)
    cached_agent = Agent(raw, env)
    return {"status": "initialized", "agent": raw}

def action_act(args: Any) -> dict[str, Any]:
    """Compute action using the cached agent."""
    global cached_agent
    if cached_agent is None:
        return {"error": "No agent initialized. Call initialize_agents first."}
    
    observation = utils.get(args.state, dict, {}, ["observation"])
    action, log = cached_agent.act(observation)
    return {"action": action}
```

Call `action_dispose()` to clear the cached agent between episodes if needed.
