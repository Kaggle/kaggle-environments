# Gemini Codebase Guide

This document provides guidance for an AI assistant to effectively contribute to the `kaggle-environments` repository. It covers project structure, common workflows for creating and updating environments and visualizers, and key commands.

## Project Overview

Kaggle Environments is a Python framework for evaluating episodes in competitive multi-agent environments. It uses a plugin-based system where game environments (e.g., ConnectX, RPS) are registered and run with configurable agents. The web-based visualizers are built using TypeScript and Vite.

## 1. Key Directories

*   **`kaggle_environments/`**: The core Python package.
    *   **`envs/`**: Contains the implementations of the various environments. Each environment is a self-contained module with its own logic, configuration, and a `visualizer` subdirectory.
    *   **`core.py`**: The heart of the library. It contains the `make` function for creating environments and the `Environment` class that manages the game state and agent interactions.
    *   **`__init__.py`**: Auto-discovers and registers all environments from the `envs/` directory.
    *   **`main.py`**: The entry point for the command-line interface and the HTTP server.
*   **`web/`**: The frontend monorepo for the visualizers.
    *   **`core/`**: A core package providing common UI components, hooks, and utilities for the visualizers.
    *   **`scripts/`**: Contains scripts for managing the visualizer projects, such as building and running them.
*   **`tests/`**: Contains the Python tests.
    *   **`envs/`**: Tests for each individual environment.
    *   **`integration/`**: Integration tests that run agents against environments.

## 2. How to Add/Update an Environment

1.  **Create the Directory:** Add a new directory under `kaggle_environments/envs/<my_environment>/`.
2.  **Create the Specification (`<my_environment>.json`):**
    *   This JSON file defines the schema for your environment's `observation`, `action`, `configuration`, and `reward`.
    *   It also specifies the number of agents (`agents`).
    *   Use existing `.json` files as a reference.
3.  **Create the Interpreter (`<my_environment>.py`):**
    *   This Python file contains the core logic of your game.
    *   **`interpreter(state, env)`**: This function is the game engine. It takes the current `state` and the `env` object, processes the agents' actions, and returns the new `state`.
    *   **`renderer(state, env)`**: This function returns a string representation of the environment state, used for ANSI/text-based rendering.
    *   **`html_renderer()`**: This function should return the HTML/JavaScript code for the web-based visualizer. It typically reads the compiled `index.html` from the visualizer's `dist` directory.

## 3. How to Add/Update a Visualizer

1.  **Create the Visualizer Project:**
    *   Inside your environment's directory (`kaggle_environments/envs/<my_environment>/`), create a `visualizer/default` subdirectory.
    *   This directory will contain a Vite-based project. You can copy the structure from an existing visualizer (e.g., `connectx`).
    *   Key files include `package.json`, `vite.config.ts`, `tsconfig.json`, and the `src/` directory.
2.  **Develop the Visualizer:**
    *   The main entry point is `src/main.ts`.
    *   The core rendering logic is typically in a file like `src/renderer.ts`.
    *   The `window.addEventListener("message", ...)` is used to receive game state updates from the parent `player.html`.
    *   Use `pnpm dev` to start the Vite development server for your visualizer. The `find-games.js` script will prompt you to select which game to run.
3.  **Build the Visualizer:**
    *   Run `pnpm build` to compile your visualizer. This will generate a `dist/` directory with the final `index.html` and JavaScript assets.
4.  **Integrate with the Environment:**
    *   Ensure your environment's `html_renderer()` function in `<my_environment>.py` correctly reads and returns the content of the `dist/index.html` file.

## 4. Common Commands

### Python (run from the root directory)

*   **Install/Sync Dependencies:**
    ```bash
    uv sync
    ```
*   **Run All Tests:**
    ```bash
    ./run_tests.sh
    ```
*   **Run Specific Tests:**
    ```bash
    ./run_tests.sh -k "<test_name_pattern>"
    ```
*   **Linting and Formatting:**
    ```bash
    uv run ruff check --fix .
    uv run ruff format .
    ```

### Frontend (run from the root directory)

*   **Install/Sync Dependencies:**
    ```bash
    pnpm install
    ```
*   **Run a Visualizer in Dev Mode:**
    ```bash
    pnpm dev
    ```
*   **Build a Visualizer:**
    ```bash
    pnpm build
    ```
*   **Run End-to-End Tests:**
    ```bash
    pnpm test:e2e
    ```
*   **Formatting:**
    ```bash
    pnpm format
    ```

## 5. Code Style

*   **Python:**
    *   Version: `>=3.11`
    *   Formatter: `ruff format`
    *   Linter: `ruff check`
    *   Line Length: 120
    *   Quote Style: Double quotes
*   **TypeScript/JavaScript:**
    *   Formatter: `prettier`
    *   Linter: `eslint`
*   **Git Hooks:** `lefthook` is used to run linters and formatters before committing.
