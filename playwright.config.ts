import { defineConfig, devices } from '@playwright/test';
import { globSync } from 'glob';
import { execSync } from 'child_process';

// Get all visualizer packages at config time
let visualizerPackages: { name: string; path: string }[] = [];
try {
  const pnpmOutput = execSync('pnpm m ls --json --depth -1', { encoding: 'utf8' });
  const packages = JSON.parse(pnpmOutput);
  visualizerPackages = packages.filter(
    (pkg: any) => pkg.name?.startsWith('@kaggle-environments/') && pkg.name?.endsWith('-visualizer')
  );
} catch {
  console.warn('Could not list pnpm workspaces');
}

// Find the best matching package name for a given directory name
function findPackageMatch(dirName: string): string {
  // Try exact match with directory name
  let match = visualizerPackages.find((pkg) => pkg.name.includes(dirName));
  if (match) return dirName;

  // Try with hyphens instead of underscores
  const kebabName = dirName.replace(/_/g, '-');
  match = visualizerPackages.find((pkg) => pkg.name.includes(kebabName));
  if (match) return kebabName;

  // Fallback to directory name
  return dirName;
}

/**
 * Playwright configuration for kaggle-environments visualizer integration tests.
 *
 * To add tests for a visualizer:
 *   1. Create a test file in your visualizer's e2e directory (e.g., visualizer/default/e2e/connectx.test.ts)
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
    const dirName = standardMatch[1];
    // Package names are inconsistent - some use hyphens (kore-fleets), some underscores (llm_20_questions)
    // Find the correct form that matches the actual package name
    const matchingName = findPackageMatch(dirName);
    return {
      name: dirName,
      testMatch: `kaggle_environments/envs/${dirName}/visualizer/**/*.test.ts`,
      port: getPortForVisualizer(dirName),
      packageFilter: matchingName,
    };
  }

  // Match OpenSpiel games: kaggle_environments/envs/open_spiel_env/games/{name}/visualizer/{version}/...
  const openSpielMatch = testFile.match(
    /kaggle_environments\/envs\/open_spiel_env\/games\/([^/]+)\/visualizer\/([^/]+)\//
  );
  if (openSpielMatch) {
    const gameName = openSpielMatch[1];
    const version = openSpielMatch[2];
    // Convert underscores to hyphens for kebab-case package names
    const kebabGameName = gameName.replace(/_/g, '-');
    // Version suffix: "default" -> "", "v2" -> "-v2"
    const versionSuffix = version === 'default' ? '' : `-${version}`;
    const projectName = `open-spiel-${kebabGameName}${versionSuffix}`;
    return {
      name: projectName,
      testMatch: `kaggle_environments/envs/open_spiel_env/games/${gameName}/visualizer/${version}/**/*.test.ts`,
      port: getPortForVisualizer(projectName),
      packageFilter: `@kaggle-environments/open-spiel-${kebabGameName}${versionSuffix}-visualizer`,
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
    hash = hash & hash;
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
  command: `pnpm test-server ${viz.packageFilter}`,
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
  reporter: [['list'], ['json', { outputFile: 'test-results/results.json' }], ['html', { open: 'never' }]],
  projects,
  webServer: webServers.length > 0 ? webServers : undefined,
});
