# Kaggle Environments Visualizer Development

This document provides technical guidance for developers working on simulation visualizers. It covers the current Vite-based development workflow and explains the legacy Cloud Function-based system for historical context.

## How to Develop a Visualizer (Vite Workflow)

All new visualizers are developed as standalone packages within a `pnpm` monorepo. This approach uses Vite for a fast, modern development experience with hot-reloading and generates static assets for production.

## Getting Started:
```pnpm install``` (at root)
```pnpm lefthook install```  (one time only - sets up pre-commit hooks for testing/linting/formatting)
```pnpm dev```

### 1. Directory Structure

Each visualizer lives in its own package directory. To create a new visualizer for a game named `my-game`, you would create the following structure:

```
/kaggle_environments
└── envs/
    └── my-game/
        ├── my-game.py
        └── visualizer/
            └── default/
                ├── src/
                │   └── main.ts         // Your visualizer's entry point
                ├── index.html          // The HTML entry point for Vite
                ├── package.json        // Defines dependencies (e.g., three.js)
                ├── tsconfig.json       // TypeScript configuration
                └── vite.config.ts      // Vite configuration (can often extend a base config)
```

### 2. Development Workflow

1.  **Install Dependencies**: From the root of the `kaggle-environments` repository, run `pnpm install`. This will install all dependencies for all visualizer packages.

2.  **Run the Dev Server**: To start the development server for a specific visualizer, use the `--filter` flag with the `pnpm dev` script. For example, to run the `connectx` visualizer:

    ```bash
    # The package name is defined in kaggle_environments/envs/connectx/visualizer/default/package.json
    pnpm --filter @kaggle-environments/connectx-visualizer dev
    ```

    This will start a local server with hot-reloading, allowing you to see your changes instantly.

    You can also run the dev server from the folder you are working in, or just run `pnpm dev` from the root of the repository and a selector will find the currently existing visualizers

### 3. Data Handling

Visualizers are rendered inside an iframe on the Kaggle platform. The replay data is not loaded directly by the visualizer. Instead, it is passed from the parent window using `window.postMessage`.

Your `main.ts` file should include a listener to receive this data:

```typescript
// src/main.ts

window.addEventListener("message", (event) => {
  const replayData = event.data;

  render(replayData);
});

function render(data) {
  console.log("Received data:", data);
}
```

A test replay file (e.g., `replays/test-replay.json`) is often included in the visualizer package to allow for local development and testing of the rendering logic.

### 4. Building for Production

To create a production-ready build, run the build script from the root directory:

```bash
pnpm build
```

This command iterates through all visualizer packages and runs `vite build` for each one, generating a `dist` directory containing the optimized static HTML, JavaScript, and CSS files. These are the files that will be deployed to Google Cloud Storage (GCS).

---

## How the Legacy System Worked (Cloud Function)

The previous system relied on a Google Cloud Function to render visualizers on the server side. This process is now deprecated but is documented here for context.

### Rendering Process

1.  **Request**: A user's browser requested a replay from the Kaggle backend.
2.  **Cloud Function Invocation**: If a pre-rendered static HTML file was not available in GCS, the backend invoked a Cloud Function.
3.  **File Fetching**: The Cloud Function would fetch the necessary raw assets from the repository, such as a generic `player.html` template and the game-specific JavaScript file (e.g., `connectx.js`).
4.  **On-the-Fly Bundling**: The function would dynamically inject the game's JavaScript and the replay JSON data directly into the HTML template. The replay data was often placed inside a `<script>` tag as a global variable.
5.  **Response**: The fully-formed HTML file was returned to the user's browser and rendered.

This server-side rendering approach was slow and made for a difficult developer experience, which is why the project was migrated to the current static-site generation model.
