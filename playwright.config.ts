import { defineConfig, devices } from '@playwright/test';
import { globSync } from 'glob';

/**
 * Playwright configuration for kaggle-environments visualizer integration tests.
 *
 * To add tests for a visualizer:
 *   1. Create a test file in your visualizer directory (e.g., connectx.test.ts)
 *   2. Ensure the visualizer has a `dev-with-replay` script in package.json
 *   3. Add a test replay file in the replays/ directory.
 *      - Warning: this replay file is PUBLIC. If you are actively developing a visualizer for a simulation competition,
 *      you run the risk of leaks. Proceed with caution and use dummy data.
 *   4. Run `pnpm test:e2e` - your tests will be automatically discovered
 */

const testPatterns = [
  'kaggle_environments/envs/*/visualizer/**/*.test.ts',
  'kaggle_environments/envs/open_spiel_env/games/*/visualizer/**/*.test.ts',
];

const testFiles = testPatterns.flatMap((pattern) => globSync(pattern));

interface VisualizerInfo {
  name: string;
  testMatch: string;
  port: number;
  packageFilter: string;
}

function getVisualizerInfo(testFile: string): VisualizerInfo | null {
  // Match standard envs: kaggle_environments/envs/{name}/visualizer/...
  const standardMatch = testFile.match(/kaggle_environments\/envs\/([^/]+)\/visualizer\//);
  if (standardMatch) {
    const name = standardMatch[1];
    return {
      name,
      testMatch: `kaggle_environments/envs/${name}/visualizer/**/*.test.ts`,
      port: getPortForVisualizer(name),
      packageFilter: `@kaggle-environments/${name}-visualizer`,
    };
  }

  // Match OpenSpiel games: kaggle_environments/envs/open_spiel_env/games/{name}/visualizer/...
  const openSpielMatch = testFile.match(/kaggle_environments\/envs\/open_spiel_env\/games\/([^/]+)\/visualizer\//);
  if (openSpielMatch) {
    const name = openSpielMatch[1];
    return {
      name: `openspiel-${name}`,
      testMatch: `kaggle_environments/envs/open_spiel_env/games/${name}/visualizer/**/*.test.ts`,
      port: getPortForVisualizer(`openspiel-${name}`),
      packageFilter: `@kaggle-environments/${name}-visualizer`,
    };
  }

  return null;
}

// Generate a deterministic port for a visualizer based on its name
// Uses a simple hash to ensure consistent ports across runs
function getPortForVisualizer(name: string): number {
  const BASE_PORT = 5173;
  const PORT_RANGE = 100; // Ports 5173-5272

  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    const char = name.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash; // Convert to 32-bit integer
  }

  return BASE_PORT + (Math.abs(hash) % PORT_RANGE);
}

// Deduplicate visualizers (multiple test files in same visualizer)
const visualizerMap = new Map<string, VisualizerInfo>();
for (const testFile of testFiles) {
  const info = getVisualizerInfo(testFile);
  if (info && !visualizerMap.has(info.name)) {
    visualizerMap.set(info.name, info);
  }
}

const visualizers = Array.from(visualizerMap.values());

const projects = visualizers.map((viz) => ({
  name: viz.name,
  testMatch: viz.testMatch,
  use: {
    trace: 'on-first-retry' as const,
    baseURL: `http://localhost:${viz.port}`,
    ...devices['Desktop Chrome'],
  },
}));

const webServers = visualizers.map((viz) => ({
  command: `pnpm test-server ${viz.name}`,
  url: `http://localhost:${viz.port}`,
  reuseExistingServer: !process.env.CI,
  timeout: 120000,
  // Use the deterministic port
  env: { VITE_PORT: String(viz.port) },
}));

if (process.env.DEBUG) {
  console.log('Discovered visualizers:', visualizers);
}

export default defineConfig({
  testMatch: testPatterns,
  fullyParallel: true,
  retries: 1,
  reporter: [['line'], ['json'], ['html']],
  projects,
  webServer: webServers.length > 0 ? webServers : undefined,
});
