# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kaggle Environments is a Python framework for evaluating episodes in competitive multi-agent environments. It provides a plugin-based system where game environments (e.g., ConnectX, RPS, Chess, Lux AI) are registered and run with configurable agents. The emphasis is on episode evaluation rather than agent training.

## Common Commands

### Testing
```bash
./run_tests.sh                          # Run all tests locally with uv
./run_tests.sh -k "rps"                 # Run tests matching a pattern
uv sync && uv run pytest tests/envs/rps/test_rps.py  # Run a single test file
./run_tests.sh --docker                 # Run tests in Docker container
./run_tests.sh --multicontainer         # Run multi-container integration tests
```

### Linting & Formatting
```bash
uv run ruff check --fix .              # Lint with auto-fix
uv run ruff format .                   # Format code
```

Pre-commit hooks run `ruff-check --fix` and `ruff-format` automatically.

### Package Management
```bash
uv sync                                # Install/sync dependencies
```

## Code Style

- Python 3.11+, line length 120, double quotes, space indentation
- Import sorting enabled via ruff (`extend-select = ["I"]`)
- Build backend: flit

## Architecture

### Core Framework (`kaggle_environments/`)

- **`__init__.py`** — Auto-discovers and registers all environments from `envs/` at import time. Exports `make`, `evaluate`, `register`, `Agent`.
- **`core.py`** — `Environment` class: the main runtime that manages specification validation, state machine (ACTIVE → DONE/ERROR/INVALID/TIMEOUT), interpreter execution, and agent coordination. Contains `make()`, `evaluate()`, and `register()`.
- **`agent.py`** — `Agent` class: wraps agent functions with timeout handling and error capture. Agents can be Python functions, file paths, URLs, inline strings, or fixed actions.
- **`utils.py`** — Schema validation (via jsonschema), `Struct` (dot-access dicts), file utilities.
- **`main.py`** — CLI entrypoint and Flask HTTP server. CLI actions: `list`, `evaluate`, `run`, `step`, `load`, `act`, `dispose`, `http-server`.
- **`errors.py`** — Exception hierarchy based on canonical error codes.

### Environment Plugin Structure (`kaggle_environments/envs/<name>/`)

Each environment is a self-contained directory that must expose:

1. **`specification`** — Loaded from `<name>.json`; JSON Schema defining agents count, configuration, observation, action, and reward schemas.
2. **`interpreter(state, env)`** — Takes current state and environment, applies game logic, returns new state.
3. **`renderer(state, env)`** — Returns text/ANSI representation.
4. **`html_renderer()`** — Returns JavaScript for HTML visualization.
5. **`agents`** (optional) — Dict of named default agents (e.g., `{"random": fn, "reaction": fn}`).

The `open_spiel_env` is a special case: it wraps the OpenSpiel library and registers multiple game environments from an `ENV_REGISTRY` dict.

### Tests (`tests/envs/`)

Test files mirror the environment structure. Tests typically use `make()` to create an environment, `run()` with agents, and assert on the resulting JSON state/statuses.

### Web Visualizer (`web/`)

React/TypeScript/Vite frontend for HTML game replays. Separate from the Python package; has its own `package.json` with Playwright for E2E testing.
