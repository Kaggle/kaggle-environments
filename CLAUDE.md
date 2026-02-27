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
pnpm format                            # Format TS/JS with prettier
```

Pre-commit hooks run `ruff-check --fix` and `ruff-format` automatically.

### Package Management
```bash
uv sync                                # Install/sync Python dependencies
pnpm install                           # Install frontend dependencies
```

### Frontend (from repo root)
```bash
pnpm dev                                # Run a visualizer dev server (interactive game picker)
pnpm build                              # Build a single visualizer (interactive picker)
pnpm build-all                          # Build all visualizers
pnpm test:e2e                           # Run Playwright end-to-end tests
```

## Code Style

- **Python:** 3.11+, line length 120, double quotes, space indentation. Import sorting via ruff (`extend-select = ["I"]`). Build backend: flit.
- **TypeScript/JS:** Formatted with prettier, linted with eslint.
- **Git hooks:** lefthook runs linters/formatters before committing.

## Architecture

### Core Framework (`kaggle_environments/`)

- **`__init__.py`** — Auto-discovers and registers all environments from `envs/` at import time. Exports `make`, `evaluate`, `register`, `Agent`.
- **`core.py`** — `Environment` class: the main runtime that manages specification validation, state machine (ACTIVE -> DONE/ERROR/INVALID/TIMEOUT), interpreter execution, and agent coordination.
- **`agent.py`** — `Agent` class: wraps agent functions with timeout handling and error capture. Agents can be Python functions, file paths, URLs, inline strings, or fixed actions.
- **`utils.py`** — Schema validation (via jsonschema), `Struct` (dot-access dicts), file utilities.
- **`main.py`** — CLI entrypoint and Flask HTTP server.
- **`errors.py`** — Exception hierarchy based on canonical error codes.

### Environment Plugins (`kaggle_environments/envs/<name>/`)

Each environment is a self-contained directory. The main module (`<name>.py`) must export: `specification`, `interpreter`, `renderer`, `html_renderer`, and optionally `agents`. Discovery is automatic at import time.

The `open_spiel_env` is a special case: it wraps the OpenSpiel library and registers multiple game environments from an `ENV_REGISTRY` dict.

### Tests (`tests/envs/`)

Test files mirror the environment structure. Tests use `make()` to create an environment, `run()` with agents, and assert on `env.toJSON()` (statuses, rewards, steps).

### Web Visualizers

Each environment can have a Vite + TypeScript visualizer at `envs/<name>/visualizer/default/`. The pnpm workspace (`pnpm-workspace.yaml`) links `web/*` and all `visualizer/*` directories. `web/core/` (`@kaggle-environments/core`) provides shared UI components, replay adapters, and playback controls used by all visualizers.

### Key Reference Files

When building environments or visualizers, these files are useful references:
- `kaggle_environments/envs/rps/` — simplest complete environment (good starting template)
- `kaggle_environments/envs/connectx/` — board game with shared observations and per-agent defaults
- `kaggle_environments/schemas.json` — framework-level defaults for configuration, observation, status
- `web/vite.config.base.ts` — shared Vite build config for all visualizers
- `web/tsconfig.base.json` — shared TypeScript config
- `web/core/src/index.ts` — all exports from `@kaggle-environments/core`
